# 工作交接：retool_math 原生工具协议（native）路线进展与下一步计划

**日期**：2026-10-01
**范围**：从"11 版围栏协议无效定案 → 原生 `<tool_call>` 主线（docs/09）"之后，本轮完成的代码修复、native_p2/p3 两次实跑结论、以及 native_p4 的执行计划。

---

## 0. 一句话现状

**协议机器（prompt/预算/解析）已切换到 native 档，但真实数据证明 `base 4B 模型不会在 `</tool_call>` 后自然停`——native 档默认"不装 stop"的假设在长轨迹上失效，invalid 率高达 ~60%，这是当前所有"没效果"现象的头号原因。下一步必须把 `--native_stop_at_call` 从"兜底开关"升格为"必装部件"，配合档 D 预算，重跑闸门验证。**

---

## 1. 已合入并推送的代码修复（main @ 7b1d17d）

| commit | 内容 | 文件 |
|---|---|---|
| `0ffec5e` fix(rollout) | **F1**：`code_wasted`（末轮废码）并入 advantage 排除口径——与 loss 侧 `sample_weight`（trunc OR wasted）同一人群。旧版只排 trunc，废码样本以 −1 污染组基线、自身又零梯度，与注释声称的"adv=0"矛盾 | [rollout.py](rlab/rollout.py)（`retool_score_flat`）、[test_retool_cpu.py](rlab/tests/test_retool_cpu.py) `test_code_wasted_adv_exclusion`（8 项） |
| `7b1d17d` feat(probe) | **F2**：`probe_difficulty` 新增 `--tool_protocol {fence,native}` / `--native_tool_style` CLI 入口。此前 get_config 恒得 fence、原生分支从 CLI 不可达 → 任何难度表都是围栏档探的，native run 静默混表。协议档先落 overrides 再进 get_config（顺序敏感，自动套 native 预算档与原生提示） | [probe_difficulty.py](rlab/probe_difficulty.py)、[test_native_protocol.py](rlab/tests/test_native_protocol.py)（+4 项接线检查） |

本地验证：`test_native_protocol` 188 项全过；`test_retool_cpu` 本机可跑 46 个测试函数全过（8 个跳过因本机缺 transformers/safetensors，pod 上跑全量确认）。

> ⚠ F1 是语义修正（改变含废码组的 advantage 数值），**修复前后 run 在含废码组上不可比**；对比应锚定 BASE。

---

## 2. native_p2 实测：提示-协议矛盾 → invalid 40%

- 命令问题：`--system_prompt_file rlab/prompts/retool_math_concise.txt` 是**围栏协议提示**（教 ```python 围栏 + `[TOOL RESULT]`），与 `--tool_protocol native` 直接矛盾。该 run 中途健康检查报 `native_invalid` 40%。
- 结论：先撤提示，用 preset 原生提示（`tool_protocol=native` 时 get_config 自动选 `system_prompt_retool_math_native`）。

## 3. native_p3 实测：正确提示 + 档 A 预算，invalid 仍然 ~60% —— 文档盲区实锤

**命令**（存档）：`ATTN_IMPL=flash_attention_2 bash rlab/run_gsm8k.sh retool_math /root/Qwen3.5-4B --chat_template_kwargs '{"enable_thinking": false}' --tool_protocol native --native_tool_style function --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN --vllm_gen_logps --vllm_logprobs_n 1 --gen_gpu_mem 0.6 --micro_rows 1 --optim_8bit --gen_update_steps 8 --overlong_shaping --lr 1e-6 --seed 42 --steps 600 --save_steps 100 --no-eval_during_training --len_penalty_w 0.1 --len_penalty_quantile 50 --len_penalty_gate 0.25 --code_attempt_w 0.05 --code_w 0.05 --out_dir rlab_out/native_p3`

- 预算档 A：5×1024/8192（`NATIVE_PROTOCOL_DEFAULTS`），无难度表（全量 17k 池）。
- 会话 A（00:27:50 ~ 01:11:34，44 分钟）：608 条 ≈ 76 组，gen_ver 0..24（= 训练端 6 次 optimizer 更新）→ **管线活着，训练真实推进**。

**record 分族统计（决定性数据）**：

| 组类 | n | acc率 | trunc率 | invalid率 | 废码率 |
|---|---|---|---|---|---|
| ok（已上传训练） | 30 | **53.8%** | 22.5% | — | — |
| uniform（丢弃） | 46 | 2.7% | **70.7%** | **59.5%** | 26.6% |

**三条硬结论**：

1. 丢弃率 46/76 = **60%**，且丢弃组全是"全错组"（acc≈0）——**没有任何梯度可学**；
2. 丢弃组的 invalid 59.5% × trunc 70.7% 高度重叠：**base 4B 在 `<tool_call>` 后继续写（不吐 im_end），或调用块被 1024 轮预算切半（未闭合 `<tool_call>` → 解析 invalid）** → 无 boxed → −1 → 全错组。docs/09"原生自然停在 im_end、不需 stop"的假设**只对短冒烟成立**；当初 16/16（invalid=0）的 go/no-go 冒烟**严重低估真实 invalid 分布**；
3. ok 组 acc 53.8% 是**好信号**：一旦组内有方差，数据质量正常；问题不在"学不学得会"，在"60% 的组无方差可学"。

> 补充：`--no-eval_during_training` 期间用 `rlab.analysis --record` 看曲线；该表"阶段=dropped"列只是窗口首条记录的 phase，不代表整窗口丢弃——判读时以分族统计（ok/uniform 两族各自 acc/trunc/invalid/废码）为准。

---

## 4. 下一步计划（native_p4，按顺序执行，验收标准写死）

### 第 1 步（不改代码，先判别）：确认 invalid 是"形态错"还是"调用后继续写/截断"

pod 上跑 12 题 × 3 条的判别脚本（复用真实 `multi_turn_rollout_group_native` + 同档引擎参数，`--native_tool_style function`）：

```bash
python3 - <<'EOF'
import os, random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rlab.config import get_config
from rlab.data import load_dapo_math_train
from rlab.protocol import initial_messages, parse_assistant, NATIVE_BAD_WORDS
from rlab.rollout import (multi_turn_rollout_group_native,
                          attention_backend_kwargs, batch_invariant_guard)

cfg = get_config("retool_math", tool_protocol="native", native_tool_style="function",
                 use_wandb=False, model_path="/root/Qwen3.5-4B")
if cfg.get("vllm_batch_invariant"): os.environ["VLLM_BATCH_INVARIANT"] = "1"
batch_invariant_guard(cfg.get("vllm_batch_invariant"), cfg.get("vllm_attention_backend"))
kw = dict(cfg.get("vllm_gen_kwargs") or {})
if cfg.get("vllm_attention_backend"): kw.update(attention_backend_kwargs(cfg["vllm_attention_backend"]))
llm = LLM(model=cfg["model_path"], gpu_memory_utilization=0.5, **kw)
tok = AutoTokenizer.from_pretrained(cfg["model_path"])

random.seed(42)
qas = random.sample(load_dapo_math_train(), 12)
msgs = [initial_messages(cfg["system_prompt"], q["Q"]) for q in qas]
msgs *= 3
sps = [SamplingParams(n=1, temperature=cfg["temperature"],
                      max_tokens=cfg["round_gen_tokens"], top_p=cfg["top_p"],
                      top_k=cfg["top_k"], logprobs=0,
                      bad_words=list(NATIVE_BAD_WORDS), seed=1000 + k)
       for k in range(len(msgs))]
segs, _full, stats = multi_turn_rollout_group_native(llm, sps, tok, msgs, cfg,
                                                     collect_logps=False)
inv = ans = tool = 0; shown = 0
for i, segsi in enumerate(segs):
    for s in segsi:
        if s["kind"] != "assistant": continue
        p = parse_assistant(s["text"].strip(), style="function")
        if p.kind == "invalid": inv += 1
        elif p.kind == "answer": ans += 1
        else: tool += 1
        if p.kind == "invalid" and shown < 8:
            print(f"### 样本{i} invalid: {s['text'][:350]!r}\n"); shown += 1
print(f"kind 分布: tool={tool} answer={ans} invalid={inv} "
      f"→ invalid率={inv/max(1,inv+ans+tool):.0%}")
EOF
```

判读：invalid 片段是 `<tool_call>` 开头但后面没有闭合 → 形态错，改 `--native_tool_style auto/json`；若是调用后还有内容/被切断 → 进第 2 步。**预期结论：尾随文本（第 2 步）**。

### 第 2 步：修配置（两个协议机制项一起上，非消融变量）

```bash
--native_stop_at_call                            # 在 </tool_call> 边界硬停（include_stop 保留闭合）
--max_rounds 5 --round_gen_tokens 1536 --max_context_tokens 12288   # docs/09 档 D（官方回退位，need 9768 ≤ 12288）
```

原理：stop 使调用轮精确结束在 `</tool_call>`，base 无机会"调用后继续写"、调用块也不会被 1024 切半；1536 给"推理+调用同轮"和答案轮留余量（调用本身仅 50–150 token）。

### 第 3 步：20 步闸门（native_p4_smoke）——验收判据写死

```bash
ATTN_IMPL=flash_attention_2 bash rlab/run_gsm8k.sh retool_math /root/Qwen3.5-4B \
    --chat_template_kwargs '{"enable_thinking": false}' \
    --tool_protocol native --native_tool_style function \
    --native_stop_at_call \
    --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
    --vllm_gen_logps --vllm_logprobs_n 1 \
    --gen_gpu_mem 0.6 --micro_rows 1 --optim_8bit --gen_update_steps 8 \
    --overlong_shaping --lr 1e-6 --seed 42 --steps 20 --save_steps 5 \
    --no-eval_during_training \
    --len_penalty_w 0.1 --len_penalty_quantile 50 --len_penalty_gate 0.25 \
    --code_attempt_w 0.05 --code_w 0.05 \
    --out_dir rlab_out/native_p4_smoke
```

判据（对照 native_p3 基线：dropped 60%、丢弃组 invalid 59.5%、丢弃组 trunc 70.7%）：

| 指标 | 达标线 |
|---|---|
| 分组丢弃率 | **< 30%** |
| 丢弃组 invalid 率 | **< 10%** |
| 丢弃组 trunc 率 | 显著低于 70.7%（目标 < 40%） |
| ok 组 acc 率 | ≥ 40%（不应回退） |
| 签名自证 | `-tpnative`、`-nsc1`、`-r5x1536`、`-c12288` |

### 第 4 步：探 native 难度表（F2 入口投入使用）

```bash
python3 -m rlab.probe_difficulty --model_path /root/Qwen3.5-4B --k 8 \
    --tool_protocol native --native_tool_style function \
    --chat_template_kwargs '{"enable_thinking": false}' \
    --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
    --out rlab_out/difficulty_probe_native.jsonl
```

切掉 p≈0/p≈1 的题（全量池是训练侧丢弃率 60% 的一半成因）；表指纹（含 tool_protocol/预算/sp）自动与训练签名对齐，不再有围栏表混用告警。

### 第 5 步：600 步正式跑（native_p4）+ 采样评测

- 训练命令 = 第 3 步命令改 `--steps 600 --save_steps 100 --out_dir rlab_out/native_p4`（**20 步档与 600 步档签名不同，必须换 out_dir**，guard_ckpt_collision 会拦）。
- 评测（手动，采样档）：
  ```bash
  python3 -m rlab.eval --algo retool_math --n 300 --seed 42 --val_n 12 \
      --base_path /root/Qwen3.5-4B \
      --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN --gpu_mem 0.78 \
      --models "step100=./rlab_out/native_p4/step_100,step200=...,step400=...,step600=..." \
      --out eval_native_p4.json
  ```
  口径硬要求：`--val_n 12`（greedy 会把多轮代码"测成灭绝"）、`--base_path` 必须是 4B、`--gpu_mem 0.78` 固定（档差实测 ±7pp）、启动日志须见 `temperature 来源: run_info`。增益判定用 analysis 的配对检验（同题 McNemar/均值 z），噪声地板 ±4–6pp。

### 第 6 步：后续单变量（一轮一个，docs/09 §5.2 顺序）

1. `beta 0.04 → 0.0`（参考实现与 rfpp 实测均指向无 KL）；2. 撤销 shaping（code_w/code_attempt_w/len_penalty_w → 0，outcome-only）；3. 剂量（600 → 1200 步）；4. LoRA r32 / lr 4e-5 评估（需新增工程）；5. **AIME25 OOD 集**（对标参考 +23.89pp 的前提，至今未建）。

---

## 5. 交接备注（避免重复踩坑）

- **`--native_stop_at_call` 默认值应视为过期**：docs/09 把它当"invalid 高才开"的兜底，native_p3 实锤 base 4B 在 1024/round、无 stop 下 invalid ~60%——它是**主协议部件**，native 路线默认应开。
- **go/no-go 冒烟（16/16）不可外推**：短冒烟看不到真实长轨迹的 invalid 分布；判别请用第 1 步脚本或 20 步闸门的 record 分族统计。
- **F1 修复与含废码组的旧 run 不可比**；native_p4 是新的"修复后"起点，对照一律锚定 BASE。
- **record 幸存者偏差**：overlong 整组不落盘，训练期曲线系统性偏乐观；"record 涨 eval 不涨"优先怀疑这里。
- **内嵌评测与 gen_gpu_mem 0.6 互斥**：若恢复 `--eval_during_training`，GPU0 需 ~19G 给 eval，先把 gen_gpu_mem 降到 0.30–0.45。
- **未提交文件**：`docs/13-agentic-rl-survey.md`、`docs/p9-diagnosis.md` 为工作区残留，非本次交接内容，未纳入提交。
- **待办**：pod 上 `git pull` 后跑全量 pytest（本机缺 transformers/safetensors，8 个相关测试未验）；native_p4_smoke 的 20 步 record 用 `rlab.analysis --record` + 分族统计脚本复核。