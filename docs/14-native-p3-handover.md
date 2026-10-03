# 工作交接：retool_math 原生工具协议（native）路线进展与下一步计划

**日期**：2026-09-30
**范围**：从“11 版围栏协议无效定案 → 原生 `<tool_call>` 主线（docs/09）”之后，本轮完成的代码修复、native_p2/p3/p4 实跑结论、token-budget 方案，以及成本敏感工具信用实验的当前进展。

---

## 0. 一句话现状

**成本敏感工具信用第一轮（`rlab_new/native_p4_credit`，从基础模型从头训练，约436组）held-out 评测确认有效：step100/200/300/400 **全部显著**优于 BASE（Δacc `+7.0/+9.7/+9.7/+7.4pp`，McNemar `p=0.017/0.001/0.002/0.013`，N=298），其中 step200 与 step300 并列最佳（69.8% vs BASE 60.1%），也是唯一通过 Bonferroni（0.0125）的两个；step400 回落到 67.4%。训练 record 全程 acc 平坦（61.7%→64.3%）却对应 +7~+10pp 的 held-out 增益——**record 因 overlong 组不落盘而系统性失真，不能用来判断是否学到**。机制侧遗留问题是失败模式迁移：`C_wasted` 19.2%→14.2% 证明废调用信用有效，但 trunc 升到 12.6%、`B_cut_mid_prose` 占无boxed 43%，与 step400 相对 step200/300 的回落吻合。已实现并验证 `trunc_tail_penalty=0.10`（截断轨迹末段局部负信用，任务结果仍不进组均值），签名新增 `-ttp0.1`；下一轮换 `out_dir` 单变量验证。**未验证项：评测缺平均输出 token 与工具轮数，"最少 token/最少轮数"这一半目标尚未证明。**

---

## 1. 已合入并推送的代码修复（main @ `422a8d7`）

| commit | 内容 | 文件 |
|---|---|---|
| `0ffec5e` fix(rollout) | **F1**：`code_wasted`（末轮废码）并入 advantage 排除口径——与 loss 侧 `sample_weight`（trunc OR wasted）同一人群。旧版只排 trunc，废码样本以 −1 污染组基线、自身又零梯度，与注释声称的“adv=0”矛盾 | [rollout.py](rlab/rollout.py)（`retool_score_flat`）、[test_retool_cpu.py](rlab/tests/test_retool_cpu.py) `test_code_wasted_adv_exclusion`（8 项） |
| `7b1d17d` feat(probe) | **F2**：`probe_difficulty` 新增 `--tool_protocol {fence,native}` / `--native_tool_style` CLI 入口。此前 get_config 恒得 fence、原生分支从 CLI 不可达 → 任何难度表都是围栏档探的，native run 静默混表。协议档先落 overrides 再进 get_config（顺序敏感，自动套 native 预算档与原生提示） | [probe_difficulty.py](rlab/probe_difficulty.py)、[test_native_protocol.py](rlab/tests/test_native_protocol.py)（+4 项接线检查） |
| `d5b2f88` feat(rollout) | 原生协议从轮数上限切为整轨迹 token budget；加入剩余预算约束、answer reserve、预算提示、组内效率奖励、一次性代码 shaping、零梯度排除，并修正 terminal active-set bug | [config.py](rlab/config.py)、[rollout.py](rlab/rollout.py)、[reward.py](rlab/reward.py)、[train.py](rlab/train.py) |
| `af33532` fix(config) | 确认并固化 `round_gen_tokens >= max_traj_tokens`：`P>=M` 时续写不可达，消除由任意 chunk 边界导致的调用解析差异；仅在未开 stop 且 `P<M` 时提示 | [config.py](rlab/config.py)、[test_retool_cpu.py](rlab/tests/test_retool_cpu.py) |
| `422a8d7` fix(eval) | eval 的 `mt_cfg` 补传 `round_gen_tokens`；此前训练虽用 `P=8192`，评测会静默回落到 400-token 分块，实际测成另一套终止协议 | [eval_vllm_one.py](eval_vllm_one.py)、[test_retool_cpu.py](rlab/tests/test_retool_cpu.py) |

本地 token-budget 测试累计 **845 项检查全过**。其中 `HonestGen` 回放证明：确定性 token 流下，`P<M` 与 `P>=M` 拼出的 token ids 完全一致；`P>=M` 只需一次 generate，且耗尽 `M` 即终止。该测试证明本地序列拼接语义，不等价于真实 vLLM 跨请求 RNG 一致性；真机因此仍优先使用 `P=M=8192`。

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
    --max_rounds 5 --round_gen_tokens 1536 --max_context_tokens 12288 \
    --out_dir rlab_out/native_p4_smoke
```

> **预算三件套必填**（2026-09-29 补齐）：原生协议下这三个键有 `NATIVE_PROTOCOL_DEFAULTS`
> 兜底（5×1024/8192），**不显式传就等于跑默认档**，签名会是 `-r5x1024 … -c8192`，
> 与本表要求的 `-r5x1536 -c12288` 不符——单变量当场被换掉而日志不报警。启动自证：
> 签名须含 `-r5x1536` 且 `-c12288`，协议自报行须印 `5 轮 × 1536 / ctx 12288 ｜ 前 4 轮可执行代码`。

判据（对照 native_p3 基线：dropped 60%、丢弃组 invalid 59.5%、丢弃组 trunc 70.7%）：

| 指标 | 达标线 |
|---|---|
| 分组丢弃率 | **< 30%** |
| 丢弃组 invalid 率 | **< 10%** |
| 丢弃组 trunc 率 | 显著低于 70.7%（目标 < 40%） |
| ok 组 acc 率 | ≥ 40%（不应回退） |
| 签名自证 | `-tpnative`、`-nsc1`、`-r5x1536`、`-c12288` |

> **判据修订（2026-09-29，据 §4.5 实测）**：`dropped` 族的 `invalid` 与 `trunc` **高度
> 重叠**（都是 `A_cut_mid_call`：调用块写到一半被单轮上限切断），两者不能当独立指标
> 读。20 步闸门的**主读数改为 `--no-boxed-breakdown` 的两行分布**，预期形态：
> `A`（要加单轮额度）与 `C`（旧轮数档的终局废码）是主导桶，`E_clean_no_box` 应
> 仍在个位数百分比。此处是 native_p3 轮数档的历史判据；token-budget 档不再据此推出
> 固定末轮禁调用，当前处置以 §4.7 为准。若闸门后 `E` 显著上升，说明提示层的收尾指令被削弱了
> （这是 6×2048/16384 实测里唯一没被证伪的提示层职责）。

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

## 4.5 新增诊断子命令：`--no-boxed-breakdown`（2026-09-29）

```bash
python -m rlab.analysis --no-boxed-breakdown rlab_out/native_p3/record.jsonl
```

按 **ok/dropped 分族 × 失败机理**拆开"终局没给 boxed"的样本，并给出零梯度占比。桶判定按优先级（写死在 `analysis.NO_BOXED_BUCKETS`，同名表头）：

| 桶 | 含义 | 该动哪个旋钮 |
|---|---|---|
| `A_cut_mid_call` | 轮长切断，且已写出调用开标记（`trunc ∩ invalid`） | **加单轮额度**（`round_gen_tokens`） |
| `C_wasted` | 终局写了完整调用，但剩余预算不足以容纳回包和答案 reserve | **先按 `code_used` 拆分**：首次调用太晚 vs 已成功调用后的重复晚调用；使用通用预算/效率激励治理，不做固定末轮禁调用 |
| `F_ctx_full` | observation 装不进 `max_context_tokens` 而终局 | 加大 `max_context_tokens` |
| `B_cut_mid_prose` | 轮长切断，且**没写出**任何调用（含子计数 `B2` = 全程 `code_used==0`） | **治啰嗦**（提示层减长度） |
| `D_invalid_other` | 调用形态非法，但不是被切断 | 协议/采样形态 |
| `E_clean_no_box` | 干净收尾但没给 boxed | 提示层收尾指令 |

**为什么必须固化**：这些标志**会重叠**，朴素顺序（trunc→wasted→invalid）会把 `trunc ∩ invalid` 整额归进"散文里被截"，而它与"调用里被截"**处置相反**（一个要加额度、一个要治啰嗦）。首版手写脚本正是这么错的。判据与优先级已用测试钉死（`test_smoke_cpu.py` G3 组 22 项）。

### 实测（native_p3，6 轮 ×2048/16384，576 样本 / 332 无 boxed）

```
| 族       | 样本 | 无boxed | A_cut_mid_call | C_wasted | B_cut_mid_prose | D | E_clean_no_box |
| ok       | 336  | 99 (29%)| 12 (12%)       | 71 (72%) |  9 ( 9%)        | 3 |  4 ( 4%)       |
| dropped  | 240  |233 (97%)| 97 (42%)       | 95 (41%) | 35 (15%)        | 2 |  4 ( 2%)       |
零梯度（trunc ∪ 末轮废码 → sw=0）：ok 92/336 (27%)、dropped 227/240 (95%)
```

三条结论（**推翻此前两项估计**）：

1. **`E_clean_no_box` 全局只有 1.3%（4+4/576）** ⇒ "模型不会收尾/提示层治不收尾"这一整类假设**作废**（此前按"无 boxed ≈ 截断 + ≤6.9pp"估的残余 ~17pp 高了约 7 倍）。`条件精度` 高不是巧合：只要能走到答题轮，模型几乎一定给框。
2. **两族失败方向相反** ⇒ 不可能被同一个旋钮治好：ok 族 = `C_wasted` 主导（差一点，输在末轮又调工具）；dropped 族 = `A`(42%) + `C`(41%) 两头顶死（硬题既烧轮次又浪费末轮）。
3. **`max_rounds` 5→6 实测无收益**（6×2048/16384 vs 5×2048/14336）：零方差丢弃占已落盘组 45%→**50%**、dropped 族 invalid 38.3%→43.1%、trunc 51.7%→55.6%，末轮废码率 ~31%→~30% **不动**。机制：`code_wasted` 度量的是"**每轮调用倾向**"（`max_code_calls = max_rounds−1`，末轮恒为答题轮而模型不知道自己在末轮），加轮数只是平移边界，不改变该倾向。**"继续加轮数"这条路已封**。

**零梯度算力 ≈ 27%（ok 族）/ 95%（dropped 族）**：F1（`0ffec5e`）把 `trunc_final OR code_wasted` 同时从 `adv` 与 `sample_weight` 排除（统计口径正确）。在旧轮数档里，“终局该收尾”结构性拿不到梯度，这解释了为何单纯加轮数无效。后续 token-budget 档已把执行条件改为“剩余预算能否容纳工具回包 + answer reserve”；因此本段的“末轮边界”只用于解释历史 native_p3，**不再推出固定末轮禁调用**。当前 native_p4 应按 §4.7 先区分首次晚调用与重复晚调用，再决定通用效率机制。

---

## 4.6 token 预算档：轮数上限 → 整条轨迹 token 预算（2026-09-29）

```bash
--max_traj_tokens 8192 --round_gen_tokens 8192 --answer_reserve 1024 --budget_hint \
--max_context_tokens 9216 --max_prompt_length 1024 \
--len_eff_w 0.1 --code_shaping_once --overlong_ref 4096
```

### 为什么改（4.5 的三条结论合起来指向同一个结论）

4.5 表明"加轮数"这条路已封，但**没解释为什么**。真因在循环结构里：`next_active`
只由 `exec_jobs` 填充，于是 `answer`（含散文被切）与 `invalid`（含调用被切）**都是终局**，
剩余轮数被**整段作废**。轮数从来不是"可用预算"，只是循环次数——所以 5→6 轮平移边界而
不改变任何倾向。

### 三条语义切换（都只在 `max_traj_tokens>0` 且原生协议下生效）

| # | 旧（轮数档） | 新（token 档） |
|---|---|---|
| ① | 能否执行调用 = `not is_final_round and code_used < max_rounds-1` | = 剩余预算装得下 `工具回包 + answer_reserve` |
| ② | `finish_reason=="length"` → `trunc_final=1` 且**剩余轮数作废** | 累积进同一轮**续写**，直到自然收尾或预算耗尽 |
| ③ | 模型看不到还剩多少额度 | 每次工具回包追加 `[budget] N tokens left` |

`trunc_final` 的定义随之变为"**预算耗尽而未能收尾**"（不再看 `finish_reason`）。

### ⚠ 单轮上限该设多少：**设为 ≥ 轨迹预算**（2026-09-29 修正，推翻初版推荐）

每轮实际额度是 `max_tokens = max(1, min(round_gen_tokens, max_traj_tokens − used))`。
**`round_gen_tokens ≥ max_traj_tokens` 时 min 恒取 `mtj − used`**：撞 `length`
⇔ 预算恰好用尽 → 下一轮循环顶部即判 `trunc_final=1` 出局 → **续写分支不可达**。
此时"单轮上限"与"轨迹预算"是同一个数，`round_gen_tokens` 不再是独立旋钮。

初版推荐写的是 `--round_gen_tokens 2048` + `--max_traj_tokens 8192`（P<M），
那会让续写**每轮都触发**（一轮拆成 4 块）而**没有任何语义收益**：续写是逐 token
完全等价的（FakeGen 对拍：P=8/16/20/24 与 P=8192 产出的 ids **逐位相同**——
上下文、采样参数、RNG 都没变，只是把一次 generate 拆成几次）。唯一后果是
generate 调用次数变多（每次重算 prefix、可观测性变差）。

**一个实测出来的、不稳定的边界**（**不要依赖它**）：`parse_assistant` 对"调用块
之后还有文本"判 invalid。模型写出"调用+尾随散文"时，M=600、调用块 90 token、
流 145 token 下实测——

```
P=10 → 执行了调用    P=20 → invalid    P=30 → 执行了调用
P=40 → invalid       P=80 → invalid    P=200/600 → invalid
```

即 P<M 是否"救回"这次调用**取决于 chunk 边界与调用块末尾是否巧合对齐**。
（初版注释曾断言"续写恰好救回 tool"——被这条实测证伪，已改。）
读法：**P<M 使 `invalid` 率依赖一个纯实现参数，P≥M 是确定性的。**
若确要 P<M，必须开 `--native_stop_at_call`（在 `</tool_call>` 处停，尾随文本根本
不产生，两取值行为一致）。`config._check_round_budget_hint` 在"未开 stop 串且
P<M"时打印提示（**对已知基线不叫狼来了**：开了 stop 串或 P≥M 都静默）。

### reward 侧的两处对齐

1. **`code_attempt_w`/`code_w` 改一次性**（`--code_shaping_once`）。旧口径线性累加，
   实测答对时 0 次调用 `+1.00` < 1 次 `+1.10` < 4 次 `+1.40` —— **梯度明确指向"多烧
   token"**，与"最少 token 答最优"的目标反向。改一次性后恒为 `+1.05`（保留"敢写代码"
   的对冲，去掉"多写多拿"）。
2. **`--len_eff_w`：通过轨迹内部的组内相对效率奖励**。旧 reward 里**没有任何一项**
   奖励"用更少 token 答对"：`len_penalty` 只罚**未通过**轨迹、`overlong_shaping` 是
   死开关（见下）。新项按组内通过轨迹的长度中位数起坡，短于中位数者拿正分。
   必须**组内相对**（绝对长度惩罚 = run2"表面收尾"事故根源），且必须**有 cap**
   （`ref/len` 在 len→0 时发散，一条 10 token 蒙对会把 bonus 顶到 weight×99）。

### 死开关修复

`overlong_ref_tokens()` 的自动值 = `max_rounds × round_gen_tokens`，在"预算给满"的档下
**必然够不着**：native_p3 实测 ref=10240（触发线 9984）而 `avg_clen≈2811`，差 7000 token
⇒ 长度压力实际为零。`--overlong_ref` 允许直接指定**目标长度**；token 档下自动值改为
`max_traj_tokens`（那一档声称的轨迹预算）。

### 同步改动（不同步就是"测另一个模型"）

- **eval**：`mt_cfg` 必须带 `max_traj_tokens/answer_reserve/budget_hint`，剔题预算按
  `max_traj_tokens` 取，启动行打印预算档。否则用轮数档的终止结构去评 token 档的 ckpt。
- **probe_difficulty**：新增同名入口 + `probe_meta` 加 `max_traj_tokens/answer_reserve/
  max_prompt_length`；`data.load_difficulty_table` 的比对清单同步（旧表无这些键 →
  不告警，兼容）。
- **`--max_prompt_length` 的 CLI（2026-09-29 补）**：上面 §4.6 的命令把它当 flag 写，
  而它当时**只是 config 键**（BASE=400 / retool_math=1024）→ 照抄即
  `unrecognized arguments` 拒跑。现已补成 budget 第四件套：train/probe 双入口、
  偏离进签名（`-mp<n>`，默认值不加字符）、进 `probe_meta` 与训练端比对清单。
  **传 preset 同值是无副作用 no-op**；它同时是预算不变量加数与 overlong 参考系
  （`ctx − plen`）→ 调大要重核 `validate_retool_budget`，调小到库内多数题之下会让
  采样循环空转（0/负数已在 config 层 fail-fast，非崩溃形态的静默跑废）。
- **health**：`max_clen` 在 token 档下取 `max_traj_tokens`（继续用轮数乘积 = 新死开关）。
- **analysis**：`record_clen_cap` 优先读 `max_traj_tokens`。

### 顺带修掉的一个预存在 bug（HEAD 即在，非本次引入）

`multi_turn_rollout_group_native` 里 `active = next_active` 写在 `if exec_jobs:` **块内**：
某一轮**全员都没调用工具**时 `exec_jobs` 为空 → `active` 不更新 → 本应终局的样本被
**重新生成**到 `max_rounds`（`max_rounds=3` 实测段数 3、打分文本是答案的重复拼接）。
触发面是"整组某轮无人调用工具"（base 直接作答时很常见）。围栏路径**没有**这个 bug
（那里的 `active = next_active` 在条件块之外）。token 档下它是致命的（`_iter_cap` 是
安全阀不是 `max_rounds`），故一并修复；回归锁见 `test_token_budget_mode` (h)。

> ⚠ **触发面要说准**：只有当**整个批**（训练时 = 4 题×8 条 = 32 条）在某一轮**全员
> 都没调用工具**时才触发。native_p3 实测调用率高（87.5%），32 条同时不调用的概率
> 极低 ⇒ **对 native_p3 既有读数的影响很小**，不是系统性伪影。但一旦触发就是静默
> 污染（该样本的 `clen` 被抬高、打分文本重复），且 token 档下会反复重采到安全阀，
> 所以必须修。已落盘的 native_p3 读数无需重跑，与 token 档对比时知道有这一层即可。

---

## 4.7 native_p4 当前训练进展（2026-09-29，约有效 micro-step 109）

### 运行口径与数据来源

本节来自 pod 上的：

```bash
python -m rlab.analysis \
  --record rlab_new/native_p4/record.jsonl \
  --no-boxed-breakdown rlab_new/native_p4/record.jsonl
```

快照共 `1056` 条轨迹，约 `132` 组，其中 `ok=872`（约109组）、`dropped=184`（约23组）。此前生成端已打印：累计尝试60组、有效48组、零方差12组、**overlong=0**、题目过滤4/16267；因此当时真实总丢弃率为20%，且不存在 record 未落盘的隐藏 overlong。当前快照更晚，需用最新 `[rollout] 采样统计` 再确认 overlong 仍为0。

> **步数口径**：`ok≈109组` 对应约109个训练 micro-step；若 `gradient_accumulation_steps=4`，只相当于约27次 optimizer update。`gen_ver=0..96` 与此不矛盾，生成与训练之间还有队列和权重同步节奏。

### 已上传训练样本（ok 族）

| 指标 | native_p4 早期快照 | 当前快照 | 判读 |
|---|---:|---:|---|
| acc | 69.7% | **72.6%** | +2.9pp，方向健康 |
| fmt | 77.1% | **81.0%** | +3.9pp，收尾能力改善 |
| 条件精度（acc/fmt） | 90.4% | **89.7%** | 基本稳定；当前收益主要来自完成/格式，不是已证实的数学能力跃升 |
| code rate | 95.8% | **97.4%** | 仍极高，工具可能过用 |
| code_ok | 90.7% | **93.7%** | 工具调用质量改善 |
| trunc | 6.2% | **4.7%** | -1.5pp |
| invalid | 1.4% | **0.9%** | 协议已稳定 |
| ctx 满 | 0.0% | **0.0%** | 上下文不是瓶颈 |
| 零梯度 | 17.8% | **139/872 = 15.9%** | 有改善，但仍高于目标10% |

`dropped` 族 `184` 条中 acc 1.6%、trunc 22.3%，`174/184=94.6%` 为零梯度，符合零方差组主要是无可学习的全错轨迹，而不是训练样本质量下降。

### 最近窗口趋势：效率明显改善，但存在采样波动

| 窗口（约组） | acc | fmt | trunc | 末轮废码 | avg_clen | ≥7372 |
|---|---:|---:|---:|---:|---:|---:|
| 0~80 | 约54.3% | 约59.6% | 约10.8% | 约25.2% | 约4303 | 约21.5% |
| 80~132 | 约70.0% | 约78.9% | 约3.1% | 约16.9% | **约3258** | **约10.4%** |
| 最近32组 | 约75.0% | 约82.4% | 约3.5% | 约12.5% | **约2993** | **约9.4%** |

以上窗口混合了 `ok` 与 `dropped`，只用于观察生成总体行为；训练质量判断仍以 `ok` 分族为准。全局加权 `avg_clen≈3893`，相比早期快照约4216下降；最近窗口已达到原定 step-100 目标（`avg_clen<3500`、接近预算比例约10%以内）。`800~960` 窗口 acc 80.6% 后，`960~1120` 回落到65.6%，每窗只有约20道题且同题8轨迹高度相关，先按采样波动处理，不判定退化。

staleness 全局均值1.6、最大3；最近均值2.3但最大值未扩大，当前仍正常。只有均值持续升到4以上或最大值不断增加，才检查生成/训练吞吐失衡。

### 当前瓶颈：成功使用工具后又在终局重复调用

无 boxed 分解：

| 族 | 无 boxed | `C_wasted` | `B_cut_mid_prose` | `D_invalid_other` | `E_clean_no_box` |
|---|---:|---:|---:|---:|---:|
| ok | 166/872（19%） | **98** | 40 | 8 | 20 |
| dropped | 181/184（98%） | **133** | 41 | 1 | 6 |

同时 `A_cut_mid_call=0`、`F_ctx_full=0`，所以可以排除“单轮额度不足”和“上下文装不下”。ok 族中只有约23条完全没执行过工具，却有98条 `C_wasted`；即使把这23条全部视为“第一次调用就太晚”，仍至少有 `75/98=76.5%` 的 `C_wasted` 来自**已经用过工具后再次晚调用**。

因此若 step 200 后仍需改配方，目标应是通用的“成功调用后尽快作答/抑制无收益的重复调用”，而不是：

- 增加 `max_traj_tokens` 或 `max_context_tokens`；
- 设置只适用于本任务的固定末轮禁调用；
- 直接把所有代码奖励归零；
- 恢复 `P<M` 的人工续写分块。

### 当前决策与下一闸门

1. **当前 run 不改参数、不重启，继续到有效 step 200。** step 100 已通过协议、正确性和近期效率闸门。
2. **保留并评测 `step_100`**。训练 record 只能证明训练分布内轨迹改善，不能代替 held-out 泛化评测。
3. 暂不直接承诺跑满600；step 200 再决定。
4. step 200 建议继续条件：真实总丢弃率 `<25%` 且 `overlong≈0`；最近窗口 `avg_clen<=3000~3300`；`>=7372` 比例 `<10%`；ok 零梯度 `<12%`（理想 `<10%`）；ok `C_wasted<=8~10%`；held-out 准确率不低于 step 100，且平均 token/工具轮数继续下降。
5. 复核命令仍为：

```bash
python -m rlab.analysis \
  --record rlab_new/native_p4/record.jsonl \
  --no-boxed-breakdown rlab_new/native_p4/record.jsonl
```

并同时保存最新 `[rollout] 采样统计` 与 `step_N/run_info.json` 中的 `round_gen_tokens/max_traj_tokens/answer_reserve/native_stop_at_call/git_head/signature`，避免只凭文档推断 live run 配置。

## 4.8 下一阶段：成本敏感工具信用分配（2026-09-30）

### 背景与决策

native_p4 的 held-out 评测显示 `step400` 相对 BASE 为 `+7.4pp`，McNemar
`p=0.007`，因此旧 run 的最佳模型是 `step400`。训练进程停止后，当前骨架没有
DeepSpeed `engine.save_checkpoint/load_checkpoint`，`step400` 只保留模型权重，不含
AdamW8bit 状态、梯度累积状态或随机数状态；后续实验必须标为 **step400 warm-start**，
不能称为旧 run 的 step500/600 续训。

训练 record 中 `ok` 族 `C_wasted≈11.25%` 在约109到390个有效 micro-step间几乎不动，
根因是旧实现把 `trunc OR code_wasted` 同时设为 `adv=0` 与 `sample_weight=0`：模型产生
预算不足的工具调用，却无法从该动作获得任何梯度。

### 已实现的通用方案

新增配置（retool_math preset默认开启，其他算法默认关闭）：

```text
--tool_call_cost 0.02
--tool_waste_penalty 0.10
```

语义是成本敏感的 turn-level process reward，不是固定末轮禁调用：

- 正常assistant轮保持组内任务优势；
- 已执行工具调用轮收取一次 `tool_call_cost`；
- 预算不足仍调用的assistant轮覆盖为一次 `tool_waste_penalty` 局部负优势；
- 之前成功的推理/工具轮不连坐；工具回包始终 `mask=0`，不进loss；
- `sample_mean`下按该样本assistant token数归一化，保证成本按“每次动作”计，不随代码段长度稀释；
- 启用成本时，废调用样本重新进入loss；关闭成本时保持历史 `trunc OR code_wasted`整行过滤；
- 逐token advantage直接复用现有协议 `(B,T)` 和 loss 广播逻辑，生成端真实token ids标记段边界，不重新tokenize；
- 二维process reward只要存在非零可学习token就上传，即使整组都做了同一种浪费动作也不会被零方差过滤。

同时修复了 `micro_rows` 路径对 `sample_weight` 未按chunk切片的问题；新增配置会进入
run signature（`-tcc...-twp...`）与 `run_info`，防止新旧配方混用checkpoint目录。
`analysis --no-boxed-breakdown` 会根据 `run_info.tool_waste_penalty` 区分“截断零梯度”
和“废调用已有局部信用”。

### 验证结果

- `python -m rlab.tests.test_retool_cpu`：**895项通过**；
- `python -m rlab.tests.test_smoke_cpu`：**134项通过**；
- Python编译、`git diff --check`、pyflakes未定义名检查通过（仅有既存 unused 警告）；
- 梯度探针确认：工具回包梯度严格为0；浪费调用轮梯度方向为降低其概率；此前有效调用/推理仍按任务优势训练。

### 新实验启动方式与验收

不能在旧 `rlab_new/native_p4` 目录复用。建议从 `step400` 权重启动新目录，例如：

```bash
bash rlab/run_gsm8k.sh retool_math /path/to/native_p4/step_400 \
  --tool_protocol native \
  --max_traj_tokens 8192 --round_gen_tokens 8192 \
  --answer_reserve 1024 --max_context_tokens 9216 \
  --tool_call_cost 0.02 --tool_waste_penalty 0.10 \
  --steps 200 --save_steps 50 \
  --out_dir rlab_new/native_p4_credit
```

实际 pod 命令必须继续沿用 native_p4 的真实模型路径、Qwen3.5 `enable_thinking=false`、
vLLM backend/显存/分裂加载参数；上面只展示信用分配和新输出目录，不代表可直接省略
原run的硬件参数。`step400` 是初始化权重，不是可恢复的optimizer checkpoint。

首轮先观察32~64个有效组，再决定是否扩大到200 micro-step。验收优先级：

```text
主指标：held-out acc 不低于 step400 的统计波动范围
C_wasted：11.25% → 目标 <7%
ok 零梯度：16.3% → 目标 <10%
avg token / 平均工具调用轮数：下降
invalid、ctx_full、真实丢弃率：不得明显恶化
```

若 `C_wasted`下降但held-out acc下降，降低 `tool_waste_penalty`；若准确率稳定且
`C_wasted`不动，再单独调整 `tool_call_cost`。不要同时改变预算、温度、组大小或提示，
否则无法判断信用分配是否有效。

### 当前从头训练实验：约80组快照（2026-09-30）

> **实验身份修正**：正在运行的 `rlab_new/native_p4_credit` 是从基础模型**从头训练**，不是从
> `native_p4/step_400` warm-start；因此不能用 §4.8 原定的“保持 step400 准确率、把
> `C_wasted` 从11.25%降到7%”作为早期同起点验收。旧 `step400` 仍只是历史最佳对照。

当前命令口径仍为 token-budget 原生协议，`max_traj_tokens=8192`，并启用
`tool_call_cost=0.02`、`tool_waste_penalty=0.10`。`record.jsonl` 共640条已上传轨迹，
约80组，单会话，`gen_version=0..72`；staleness均值1.4、最大2，生成/训练队列健康。

| 样本窗口（约组） | acc | fmt | 条件精度 | code | code_ok | trunc | invalid | `C_wasted` | avg_clen | ≥7372 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0~160（0~20） | 59.4% | 67.5% | 88.0% | 94.4% | 85.0% | 9.4% | 0.6% | 18.8% | 3900 | 14% |
| 160~320（20~40） | 63.7% | 70.0% | 91.1% | 95.6% | 85.6% | 10.6% | 0.0% | 19.4% | 3687 | 21% |
| 320~480（40~60） | 62.5% | 73.8% | 84.7% | 99.4% | 90.0% | 3.8% | 0.0% | 20.6% | 3524 | 13% |
| 480~640（60~80） | 61.3% | 65.0% | 94.2% | 93.1% | 88.8% | 16.2% | 1.2% | 18.8% | 4075 | 26% |
| **累计** | **61.7%** | **69.1%** | **89.4%** | — | **87.3%** | **10.0%** | ≈0.5% | **19.2%** | ≈3797 | ≈18.5% |

前40组与后40组对比：acc `61.6%→61.9%`、fmt `68.8%→69.4%`、条件精度均约
89.5%、trunc均10%、平均长度 `3794→3800`、`C_wasted 19.1%→19.7%`，均基本持平；
唯一较明确的正向变化是 `code_ok 85.3%→89.4%`。当前形态应判为**整体停滞、工具执行
质量改善、长度长尾高方差**，不是已证实的单调退化。最后20组出现 `trunc=16.2%`、
`≥7372=26%`、`avg_clen=4075`，但紧邻前20组恰为 `3.8%/13%/3524`，单窗只有约20题且
同题8轨迹相关，不能单窗定性。

无 boxed 共198/640（31%），归因为：`C_wasted=123`（占全体19.2%、占无boxed 62%）、
`B_cut_mid_prose=61`（占全体9.5%、占无boxed 31%）、`D_invalid_other=3`、
`E_clean_no_box=11`；`A_cut_mid_call=0`、`F_ctx_full=0`。61条散文截断中仅20条从未调用
工具，其余多数是“调用后未及时组织最终答案”。新版局部信用已让废调用获得负梯度，但
`C_wasted`四窗为 `18.8%→19.4%→20.6%→18.8%`，截至约80组尚未形成可测下降；真正
零梯度仅剩截断64/640（10%）。

**奖励风险备注**：若 live `run_info.json` 确认为 `overlong_ref=6144`、
`overlong_buffer=256`，则绝对overlong项在5888开始、6144达到并封顶`-1`，6144~8192
不再增加边际惩罚；同时截断轨迹整行过滤，确有“预算尾段缺少直接长度信用”的结构性
缺口。但当前均长未增长、trunc未呈连续上升，尚不能称为必然膨胀。不要在本run中途改
奖励；若step100/200后 `trunc≥10%`、`B_cut_mid_prose`仍主导且长尾持续，下一独立实验
优先采用**末段局部截断负信用**，而非给整条轨迹`-2/-3`绝对惩罚。

当前决策：保持配方跑到step100并做held-out评测；重点看最近至少40组的
`C_wasted/trunc/≥7372/avg_clen`，不要按单个20组窗口停训。若连续两个窗口同时满足
`trunc≥15%`、`≥7372≥25%`、`avg_clen>4000`才提前停止。最终判断还必须补最新生成端
`[rollout] 采样统计`：overlong整组不落record，当前表不能证明真实总丢弃率健康。

---

## 4.9 截断末段信用：补齐失败模式迁移（2026-10-01）

### 触发数据：约436组快照

```bash
python -m rlab.analysis --record rlab_new/native_p4_credit/record.jsonl \
    --no-boxed-breakdown rlab_new/native_p4_credit/record.jsonl
```

| 指标 | 前80组 | 当前累计（3488条 ≈436组） |
|---|---:|---:|
| acc | 61.7% | 64.3% |
| fmt | 69.1% | 72.0% |
| 条件精度 | 89.4% | 89.3% |
| code_ok | 87.3% | 88.4% |
| trunc | 10.0% | **12.6%** |
| `C_wasted` | 19.2% | **14.2%** |
| `B_cut_mid_prose` | 9.5% | **11.9%** |
| 真正零梯度 | 10.0% | **12.6%** |
| invalid | ≈0.5% | 0.86% |
| ctx_full | 0% | 0% |

**判读**：`C_wasted` 19.2%→14.2% 证明废调用局部信用**真的有效**（降幅超出窗口波动）；
但失败模式发生迁移 —— 省下的废调用变成"继续写散文直到预算耗尽"
（`B_cut_mid_prose` 占无 boxed 的 43%、trunc 升至 12.6%），而任务准确率没有形成持续
上升。最后约76组（360~436组）trunc≈16.5%、avg_clen≈4170，说明长度问题不是单窗口尖峰。

### 机制：与废调用惩罚同构的末段局部负信用

诊断到改动的映射（三项均为已确认事实，不是推断）：

| 观察 | 机制 |
|---|---|
| `C_wasted` 有局部负信用 → 确实下降 | `tool_waste_penalty=0.10`（保留） |
| trunc 轨迹 `sw=0` 整行零梯度 | `trunc_final` 是**唯一**没有反向信号的长度失控形态 |
| 截断 = 预算耗尽，末段必是 assistant 段 | 可定位"导致失败的那个动作" |

新增 `trunc_tail_penalty`（`retool_math` preset 默认 `0.10`）：

| 项 | 语义 |
|---|---|
| 作用对象 | `trunc_final=1` 轨迹的**最后一个 assistant 段** |
| 该段优势 | 覆盖为 `-trunc_tail_penalty`（按 assistant 总长归一化，每次动作固定成本） |
| 之前的推理/工具轮 | 保持 `0`，不连坐 |
| 是否进组均值 | **否**（继续用 `sample_mask` 排除，任务结果不可信） |
| 是否整条 `sw=0` | **否**，改为只训练末段 |
| 工具 observation | 仍 `mask=0` |

关键性质：**任务结果未知所以不建基线，但"把预算烧光"这个动作拿到直接信用。**
与"整条给 -1/-2/-3 绝对惩罚"的本质区别是不惩罚未知的正确性，因此不会复现
run2"表面收尾"式的捷径。

实现复用现有 `(B,T)` 协议，无新机制面：`tool_credit_advantages(base_adv, segs,
total_len, call_cost, waste_penalty, trunc_flags, trunc_tail_penalty)`；`sw` 构造改为
"启用时不清零截断"；签名新增 `-ttp0.1`；`run_info` 新增 `trunc_tail_penalty`；
`analysis._credit_enabled` 按 run_info 分档判定零梯度人群。

### 本轮刻意不改 overlong

`overlong_ref=6144`/`buffer=256` 使绝对长度惩罚在 6144 封顶，6144~8192 边际成本为 0；
`--overlong_ref 8192 --overlong_buffer 2048` 可消除该平台区，但**不与本项同批上线**：
两者受力面重叠，同开会无法归因；且平台区目前只有二阶证据，而零梯度 12.6% 是一阶、
连续可见的问题。若下一轮 trunc 仍不降，再单独调它。

### 验证

- `test_retool_cpu.test_tool_credit_assignment` 新增 20 项：按 assistant 总长归一化、
  未截断样本不受影响、关闭时逐位退化、参数 fail-fast、梯度方向、observation 严格 0 梯度、
  签名与 CLI 接线；
- `test_smoke_cpu` 全套通过（三处固定签名后缀同步为
  `-stop1-of1-tcc0.02-twp0.1-ttp0.1`）；
- `analysis` 功能验证：同一条 record 在 `trunc_tail_penalty=0.0` 下 `zero_grad=4`、
  `=0.1` 下 `zero_grad=0`（分档判定生效）。

### 下一轮实验

签名新增 `-ttp0.1`，**必须换 out_dir**：

```bash
bash rlab/run_gsm8k.sh retool_math /root/Qwen3.5-4B \
  --tool_protocol native --native_tool_style function \
  --max_traj_tokens 8192 --round_gen_tokens 8192 \
  --answer_reserve 1024 --max_context_tokens 9216 \
  --tool_call_cost 0.02 --tool_waste_penalty 0.10 --trunc_tail_penalty 0.10 \
  --out_dir rlab_new/native_p4_trunc
```

（其余硬件/vLLM/显存参数照抄当前 run；上表只列奖励与预算项。）

验收标准（对照当前 436 组）：

| 指标 | 当前 | 目标 |
|---|---:|---:|
| 真正零梯度 | 12.6% | `<8%` |
| `B_cut_mid_prose` | 11.9% | `<8%` |
| trunc | 12.6% | `<10%` |
| `C_wasted` | 14.2% | 保持或继续降 |
| avg_clen | ≈3797 | 不上升 |
| acc / fmt | 64.3% / 72.0% | 不得下降 |
| invalid / ctx_full | 0.86% / 0 | 不恶化 |

同时必须补生成端 `[rollout] 采样统计`：overlong 整组不落 record，上表不能证明真实丢弃率。

> **执行结果见 §4.11**（2026-10-02）：record 侧验收（trunc/<8% B 桶）**未达标**，held-out 侧本 campaign 内**只有 step400 显著**。

---

## 4.10 第一轮信用实验 held-out 评测：收益确认（2026-10-01）

### 判定口径

- `N=298`（配对 `n=297`）；单臂 95%CI 最坏 `≈±5.7pp`，两臂差 `≈±8.0pp`；
- 判定规则：CI 不跨 0；有 per-item 时改用 McNemar `p<0.05`；
- 对 4 个 checkpoint 同时检验 → Bonferroni 阈值 `0.05/4=0.0125`。
- **档位 = `--val_n 1`（贪心，默认）**：下表出现 `McNemar b/c` 即为此档的签名。采样档（`--val_n 8`）与它**不是同一个测量**，不可并列比较（详见 §4.11 结论 6）。

### 结果

| 模型 | acc%(±95%CI) | fmt% | code% | Δacc vs BASE | McNemar | p | α=0.05 | Bonferroni 0.0125 |
|---|---:|---:|---:|---:|---|---:|---|---|
| BASE | 60.1±5.6 | 69.5 | 92.3 | — | — | — | — | — |
| step100 | 67.1±5.3 | 79.9 | 94.3 | +7.0pp | b=46/c=25 | 0.017 | 显著 | 不通过（临界） |
| **step200** | **69.8±5.2** | 78.5 | 96.0 | **+9.7pp** | b=51/c=22 | **0.001** | 显著 | **通过** |
| **step300** | **69.8±5.2** | 79.5 | 94.6 | **+9.7pp** | b=57/c=28 | **0.002** | 显著 | **通过** |
| step400 | 67.4±5.3 | 79.2 | 91.9 | +7.4pp | b=47/c=25 | 0.013 | 显著 | 不通过（临界） |

### 结论

1. **四个 checkpoint 全部显著优于 BASE**，收益在 step100 就已出现并保持到 step400：本轮信用机制不是"训不动"，而是**确实学到了**。
2. **最佳是 step200 与 step300（并列 69.8%）**，也是唯一通过 Bonferroni 的两个。step100（67.1%）与 step400（67.4%）几乎相同 → **step100 已进入平台期**，之后没有实质提升。
3. **`step200≈step300 > step400 > step100` 的排序不可当结论**：四个 checkpoint 两两差都落在 ±8.0pp 带内；且"从 4 个相关 checkpoint 里挑最大"本身会抬高假阳性风险（此处四个都显著，故不影响"有效"结论，只影响"哪个最好"）。
4. **step400 相对 step200/300 回落 2.4pp**，与 §4.9 训练 record 的后期劣化（trunc 升到 16.5%、avg_clen 升到 4170）方向吻合——后期长度失控是真的，但**没有吃掉整体收益**。
5. 与旧 native_p4 相比：旧 run 四个 checkpoint 里 step100（p=0.416）、step300（p=0.657）**不显著**、曲线抖动大；本轮四个全显著、平台更稳、收益出现更早。形态上更稳定，符合"废调用获得梯度后训练更可靠"的预期。

### ⚠ 与旧 native_p4 评测不可直接比较

本次 BASE 读数 `60.1%`，上一轮评测 BASE 为 `62.8%`——**同一个 BASE 两次差 2.7pp**，说明协议/采样档已经变了，绝对值不可跨 campaign 比较，只能看形态。要严格证明"信用版优于旧版"，必须把旧 `native_p4/step_400` 放进**同一次评测**（同 BASE、同 seed、同 `gpu_mem`）重跑。当前**不能宣称新版模型优于旧版 step400**。

### ⚠ 本轮最重要的一条纪律结论：训练 record 不能判断"是否学到"

训练 record 全程 acc 在 57%~79% 之间来回跳、累计只有 61.7%→64.3%，据此判断是"整体停滞、应停训"；而同一批权重 held-out 是 **+7~+10pp、四个 checkpoint 全部显著**。

根因就是分析脚本自己打印的那行警告：**overlong 整组不落 record**，叠加难度过滤，训练 record 只是"活下来的组"的条件分布。因此：

> 训练 record 只能读**行为趋势与健康签名**（trunc / `C_wasted` / invalid / staleness / avg_clen），**不能读"学到了没有"**；后者一律以 held-out 为准。

本次是该规则的一次实证，与 §5 既有条目同源。

### 未验证项：效率目标只完成了一半

评测表只有 acc/fmt/code，**缺平均输出 token 与平均工具轮数**。而本轮任务的真实目标是"**最少的输出 token + 最少的轮数 + 最优的回答**"：

- "回答更优"：已证明（+7~+10pp，显著）；
- "更省"：**未证明**。训练 record 显示 avg_clen 长期在 3.6k~4.5k 徘徊、后期还升到 4170，因此预期**效率并未改善，甚至可能倒退**。

必须补这两个读数（eval 侧通常有 clen / 轮数字段），否则不能宣布目标达成；若 acc 涨而 token 也涨，本轮只是"更准"而非"更优"。

### 对下一轮决策的影响

1. **主模型取 step200 或 step300**（并列 69.8%，均过 Bonferroni）；step400 不作为首选。
2. **`trunc_tail_penalty` 仍值得做**，但性质变了：不是"救训不动的 run"，而是"消除后期劣化、稳住平台并压 token"。§4.9 的验收标准保留，**追加一条 held-out 硬门槛**：acc 不得低于同评测下的 step200/300（即不得用长度收益换准确率）。
3. **下一轮单变量的对照锚点应为 step200/300 的 held-out 读数**，而不是训练 record 数值。
4. 若要回答"新机制 vs 旧机制"，先补做旧 `step400` 的同 campaign 评测。

---

## 4.11 native_p4_trunc（`trunc_tail_penalty=0.1`）held-out 评测：只有 step400 显著（2026-10-02）

### 判定口径

- `N=298`（配对 `n=297`）；单臂 95%CI 最坏 `≈±5.7pp`，两臂差 `≈±8.0pp`；判定 = CI 不跨 0，有 per-item 时改用 McNemar `p<0.05`；4 个 checkpoint 同时检验 → Bonferroni `0.0125`。
- **档位 = `--val_n 1`（贪心，默认）**。表里出现 `McNemar b/c` 本身就是贪心档的签名：`paired_counts` 按 `acc == 1.0` 二值化，只对 val_n=1 有效（[analysis.py](../rlab/analysis.py) 的 `items_are_binary` / `paired_test_auto`）。**采样档（`--val_n 8`：Average@8、temp 1.0/top_p 0.7）与贪心档不是同一个测量**——acc / `both` / 平均token / 工具轮数**全部不可跨档比较**，跨档只能各档内部自成一套表。
- 数据源：`eval_vllm_one.py`（`--algo retool_math`，per-item 默认落盘）。
- **欠账补齐**：§4.10「未验证项：效率目标只完成了一半」要求的**平均输出 token / 平均工具轮数**，本轮起进评测表（`avg_ans_tokens` / `avg_clen` / `avg_rounds`）。

### 结果

| 模型 | acc%(±95%CI) | fmt% | code% | 平均token | 工具轮数 | Δacc vs BASE | McNemar b/c | p | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| BASE | 61.4±5.5 | 70.5 | 92.3 | 3254 | 3.22 | — | — | — | — |
| step100 | 66.4±5.4 | 78.9 | 94.3 | 3046 | 2.39 | +5.0pp | 46/31 | 0.110 | 噪声内 |
| step200 | 63.1±5.5 | 70.5 | 89.6 | 3934 | 1.40 | +1.7pp | 38/33 | 0.635 | 噪声内 |
| step300 | 65.4±5.4 | 73.8 | 90.6 | 3538 | 1.89 | +4.0pp | 43/31 | 0.201 | 噪声内 |
| **step400** | **69.8±5.2** | **79.5** | **94.3** | **2970** | 2.44 | **+8.4pp** | 44/19 | **0.002** | **显著（唯一过 Bonferroni）** |

### 结论

1. **本 campaign 内唯一硬结论 = step400：+8.4pp（McNemar p=0.002，净翻转 +25/297）**，也是 4 个检验里唯一通过 Bonferroni(0.0125) 的；BH 校正同样在它之后停止（次小 p=0.110 > 0.05×2/4=0.025）。step100(+5.0pp)/step300(+4.0pp)/step200(+1.7pp) **全在噪声内**。
2. **不能宣称 ttp0.1 优于 credit 版，也不能宣称它让训练变差**：本次 BASE 读数 **61.4%**，credit campaign 为 **60.1%**（§4.10）——同一个 BASE 两次差 1.3pp ⇒ 档位已变，绝对值不可跨 campaign 比较。形态差异（credit 4/4 全显著、最佳在 step200/300；本 run 1/4 显著、收益出现在 step400）目前只是**待解释的观察**，不是结论。
3. **checkpoint 之间 1~4pp 的排序一律作废**：四个两两差都落在 ±8.0pp 带内，且单 run 单 seed、checkpoint 间高度相关。"step100 优于 step200/300"不能当结论。
4. **效率侧首次有读数，且三个读数完全同向**（本轮比 acc 更有信息量的部分）：

| 读法 | 证据 |
|---|---|
| 4 个 checkpoint 上 `acc↑ / 平均token↓ / 工具轮数↑` **排序完全同向** | step200 < step300 < step100 < step400（acc 升序）恰好是 token 降序、轮数升序 |
| **step200 = "少调用 + 长散文"形态** | 轮数最少(1.40) 却 token 最多(+21%)、code 率最低(89.6)、fmt 与 BASE 相同(70.5) —— 与训练 record 的 `B_cut_mid_prose`（散文写满预算）同源；它的"省轮数"**不是**效率改善 |
| **轮数本身不是目标** | BASE 轮数最多(3.22) 却最差 —— 那些是**无效轮**（晚调用/废调用，`C_wasted`）；目标是"有效调用 + 及时收尾" |
| 唯一"更准且更省" | **step400**：acc +8.4pp（显著）、token −284(−9%)、code 率并列最高(94.3)、fmt 最高(79.5) |

5. **口径待确认**：「平均token」是 `avg_ans_tokens`（模型自产）还是 `avg_clen`（含工具回包）？eval json 两个都落，**"最少输出 token"的目标口径是前者**，不能混读。（`eval_vllm_one.py` 会把两者分别打印为「平均输出token」与「平均完成长度(含工具回包)」，两者之差 = 回包实测 token：实测 2.05 轮只占 completion 的 **4.8%**（217/4521），再次否掉"工具调用吃掉预算"。）
6. **档位纪律（2026-10-02 补，来自一次真实的误判险情）**：本节全部读数都是 **贪心（`val_n=1`）**档。后续若用采样档（`val_n=8`）通评，**同一 checkpoint 的读数会整体位移**（实测 s200：贪心/采样两档 acc 相差 10pp 量级、token 也不同），届时应：
   - 四个臂（BASE + credit200/300 + trunc400）**在同一次、同 `val_n`** 调用里评完，禁止与本节数字并列；
   - 判定改用**配对均值 z 检验**（`analysis.paired_test_auto` 按 `items_are_binary` 自动分派；`val_n>1` 直接判连续档），**不要再用 McNemar**——按 `acc==1.0` 二值化检验的是"N 条全对率"，p8 事故实测过它会把 −1.5pp 的真实差报成 −4.6pp/p=0.003 的假显著。

### 对下一步的影响

1. **主模型取 step400**（本轮唯一显著、且唯一把平均 token 压到 BASE 以下的 checkpoint）；不要按 checkpoint 间 1~4pp 的排序选点。
2. **先核两次 eval 的 `eval_protocol` 四件套**（`gpu_mem` 优先：实测 0.20 vs 0.78 可产生 **7pp** 差异），再决定"能否与 credit 的 69.8% 比较"。
3. **回答"ttp0.1 有没有用"必须做同 campaign 对照**：把 `native_p4_credit` 的 step200/step300 与本次 step400 放进**同一次评测**（同 BASE / seed / `gpu_mem` / 协议 / `val_n`）。这是 §4.9 验收表里"追加一条 held-out 硬门槛"的执行方式。
4. **验证结论 4 的假说**（step200 = 少调用多散文）：用新增的 `--dump_traj` 抽 step200 与 step400 各 ~50 条，比**末段 assistant 长度分布**与工具段占 `clen` 的比例。命令与阅读陷阱见 §5「轨迹全量落盘」条。

---

## 4.12 评测轨迹归因（`eval_v_s200.traj.jsonl`，`val_n=8` 档）：失败是"写不完"，不是"不会做"（2026-10-02）

数据源：`s200` 评测的轨迹 dump，**2384 条 = 298 题 × 8**（`val_n=8` 采样档；与 §4.11 的贪心档**不可比**，见 §4.11 结论 6）。这是第一次拿文本而不是统计判读失败机理。

### 先记两条 dump 自身的缺陷（已修 `90f0bca`，读旧文件时必须知道）

1. **eval 侧漏传 `code_stats`** → 该文件里 `code_used/code_ok/trunc_final/code_wasted/invalid_final/ctx_full` **全是 0**，桶判定会把 100% 的无 boxed 读成 `E_clean_no_box`（假象）。因此下文结论**全部只用** `segs` 结构 / 长度 / 文本 / per-trajectory `acc/fmt`。
2. **工具段只有 `text`**，而它是**模板渲染**（4898 条全部以同一个 `im_end/im_start/tool_response` 包装开头，含 special token 与回答骨架）——真观察（`[exception] …`、`[budget] …`）在里面。已补 `body` 字段存纯回包。

### 失败机理三分（无 boxed 822 条 = 34.5%；用 `parse_assistant` 判末段形态 + 开/闭调用标记计数）

| 机理 | n | 占全体 | 占无 boxed | clen p50 | 末段 p50 | 贴预算墙(≥8000) |
|---|---:|---:|---:|---:|---:|---:|
| **散文写到墙**（末段 answer 形态、全程无调用） | 354 | 14.9% | 43% | 8192 | 4264 | **99%** |
| **完整调用被预算判据拒**（末段是完整 call） | 241 | 10.1% | 29% | 7405 | 1838 | 11%（**100% 落在 [6902,8192)**） |
| **调用未闭合写到墙**（开标记 > 闭标记） | 219 | 9.2% | 27% | 8192 | 8192 | **100%**（其中 119 条零调用） |
| 其他 invalid（调用后还有内容） | 22 | 0.9% | 3% | — | 756 | — |

三条硬证据：

1. **失败发生在"写"上，不在"会不会"上**：`n_segs=1`（一整段写满 8192、一次工具都没调）**173 条，acc 恒 0.000**；样例开头 `We need to find the largest integer m ≤ 2016 …`、结尾以调用块收束 —— **推理全写在散文里，工具调用排在最后一刻**。
2. **"散文写到墙"那批是在反复自证**：样例末段（4264 token）是拿到 `Code output -1427.5` 之后继续用 LaTeX 反复验算（`… Correct. So 2018 × 0.707 ≈ 1427 …`）直到撞墙。**不是不会，是不肯收尾。**
3. **预算判据签名精确复现**：末段是完整调用的 246 条里 **100% 落在 `[8192−1290, 8192)`**（1290 = 回包估计 266 + `answer_reserve` 1024），与代码 `_can = (budget − used − obs) >= reserve` 逐字吻合。另有 9 条撞 `_iter_cap=20` 轮上限。

### 对「`answer_reserve` 不够」的定量否证

成功收尾的 1562 条，末段（= 实际作答）长度 **p50=266、p75=667、p90=1712；<512 占 68%，超过 1024 的只有 15.2%**。

> **1024 的预留对 85% 的成功作答都够**，而失败样本的末段中位数是 **4264~8192** 的散文。抬 reserve 到 2048 只多覆盖约 7% 的作答形态，代价是把更多调用判废（C↑）→ **抬 reserve 不是对症的杠杆**。

### 题级结构与 `budget_hint`

- 297 个唯一题面：**全对 75（25.2%）/ 全错 47（15.8%）/ 混合 175（58.9%）**；全错题的 376 条里 **203 条（54%）末段是调用形态**（invalid/tool 结尾）→ **一半以上的"全错题"是没交卷，不是不会做**。提升空间主要在"把同题里失败的那几条救回来"。
- **`budget_hint` 已经开着**（4898 条回包都带 `[budget] N tokens left …`），但仍有 **19.3%** 的轨迹死在调用上（10.1% 被拒 + 9.2% 未闭合）→ **"模型看得见额度"是必要不充分的**；先前"开 budget_hint 是最便宜单变量"的建议在本 run 已消费掉，需要的是"拿到工具结果后必须收尾"的硬信号。
- 口径澄清：`code_ok_rate 83.5%` = "轨迹里**至少成功执行过一次**"，不是单次调用成功率；回包很小（含包装 p50 62 token，`tool` 段仅占 completion **4.8%**）→ "工具调用吃掉预算"彻底出局。

### 数据缺陷：held-out 里同一题面 → 两个答案键

`qk=9ffd553d2a25`（"A bug starts at a vertex of an equilateral triangle…"）出现 **16 条 = 两道题**：

| 答案键 | 条数 | acc | 对 |
|---|---:|---:|---:|
| `-1` | 8 | **0.000** | 0 |
| `683` | 8 | 0.875 | 7 |

- 后果①：8 条结构性不可能得分的轨迹**进分母**（全库 `A=-1` 共 32 条 / 4 题）。
- 后果②：配对键 `qk = sha1(Q)` **碰撞** → 配对检验按 qk 建 dict 只留最后一条，298 题被静默并成 **297**。**查 dev 切分是否按 (Q,A) 去重——按 (Q,A) 去重会原样保留"同题多答案"**。eval 侧已加重复 qk 告警。

### 工具语义缺口与修复：`trunc_in_call`

**缺口**：token 预算档下"预算耗尽"只置 `trunc_final`、**不置** `invalid_final`（invalid 只在轮结束分支判）→ 而 A 桶判据是 `trunc ∩ invalid` ⇒ **`A_cut_mid_call` 恒 0**，于是"调用写到一半被墙切"全被 B 桶吞掉，而 B 的定义是"没写出任何调用"——两类**处置相反**的样本在数据里同形（本节的 27%/219 条就是被吞掉的那部分）。

**修复**（`trunc_in_call`，2026-10-02）：预算墙退出与安全阀兜底两个分支用 `trunc_in_call_flag()` 置位——判据 = **开标记多于闭标记 且 `parse_assistant` 判 invalid**（标记从 `protocol` 正则派生，不手写标签字面量）；record 新键 `trunc_in_call`；analysis 的 `--no-boxed-breakdown` 新增一列 **`B_call`（调用未闭合被切）**，与 `B2` **是两个轴、可同时为真**（B2 = 全程零调用，行为轴；B_call = 切点轴）。两个 flag 都做了**旧 record 降级**：缺键 → 列显 "—（无 trunc_in_call）"，历史桶口径逐位不变（测试锁死）。

### 下一步（按序）

1. **别抬 `answer_reserve`**（上面的定量否证）；目标应是"拿到工具结果后收尾"与"别把调用拖到最后"。
2. 下一轮单变量首选 **`len_eff_w`**（更短的正确答案拿正分，已实现）或对"结果之后的散文"给局部负优势。
3. **清数据**：dev 里的同题多答案（`-1` 哨兵键）清掉后重跑评测（影响 32 条分母 + 1 个碰撞配对）。
4. 训练/评测都开 `--traj_dump` / `--dump_traj`：新行带真计数 + `body` + `trunc_in_call`，机理判定不必再靠 parse 反推。

---

## 5. 交接备注（避免重复踩坑）

- **`--native_stop_at_call` 的历史定位**：native_p3 在轮数档、1024/round、无 stop 下 invalid ~60%，说明它对旧档是关键协议部件；token-budget 档使用 `P=M=8192` 时不存在人工续写边界，但是否启用 stop 仍必须以 live `run_info.json` 为准，不能从文档猜测。
- **go/no-go 冒烟（16/16）不可外推**：短冒烟看不到真实长轨迹的 invalid 分布；判别请看完整 record 分族统计和 `--no-boxed-breakdown`。
- **F1 修复与含废码组的旧 run 不可比**；native_p4 是新的“修复后”起点，对照一律锚定 BASE。
- **record 幸存者偏差**：overlong 整组不落盘，训练期曲线系统性偏乐观；必须同时读取生成端 `[rollout] 采样统计`。native_p4 已知60次尝试快照中 overlong=0，但更晚快照仍需复核。
- **内嵌评测与 gen_gpu_mem 0.6 互斥**：若恢复 `--eval_during_training`，GPU0 需约19G给 eval，先把 gen_gpu_mem 降到0.30–0.45。
- **训练 record 不能判断"是否学到"（本轮实证）**：信用版 record 累计 acc 仅 61.7%→64.3% 且窗口在 57%~79% 间跳，但 held-out 是四个 checkpoint 全显著、+7~+10pp（§4.10）。record 只读行为趋势与健康签名，学到没有一律看 held-out。
- **生成端静默死亡（2026-10-01 事故·待定案）**：`native_p4_trunc`（`trunc_tail_penalty=0.1` 首跑，其余与 `native_p4_credit` 相同）在 **step 1 完整成功后**生成端消失，训练端只印"生成端进程已退出（见其 traceback）"。已核实的现场证据：`nvidia-smi` 两卡 `0MiB`、无残留进程；`dmesg` 命中全是 **9月25日的 `cicc`**（旧 FlashInfer 事故）；`memory.failcnt=0`；cgroup 为 **v1**（v2 路径不存在）。**即本次既非宿主 OOM、也非 cgroup OOM。**
  已补**三项**诊断能力（`train.py`）：① `_ensure_gen_alive` 报 `exitcode` **并分档**——`-9` 单独一档说明"内核直接终止，任何 handler 都拦不住，没有 traceback 是必然"，给出三条定位命令；其他负信号（`-11/-6/-7/-8/-4`）指向 ② 的栈；`>0` 说明异常退出、traceback 在上方；③ ② 是 `_spawn_gen` 里的 **`faulthandler.enable()`**——**SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL 这类"没有 Python traceback"的死亡其实可以打出 Python 栈 + C 栈**（vLLM 的 CUDA/C++ 层崩溃正属此类），前提是崩溃前已装好 handler；③ `_spawn_gen` 把子进程 stdout/stderr 改**行缓冲**，避免被信号杀死时缓冲区里最后几行（做到哪一题/哪一轮/采样统计）随进程蒸发。
  **重跑必须先 `git pull`**（否则拿不到 exitcode/faulthandler），并用 `bash run.sh 2>&1 | tee /tmp/run.log` 保证栈落盘。
  **关于归因**：`trunc_tail_penalty` 是该 run 相对 credit run 的**唯一配置差异**，但静态审查找不到机制——它只改 advantage/sample_weight 的**值**（经 `requests.post` 上传的数据），不参与生成端任何显存/内存分配；`group_ok` 变宽只会**少重采**、方向与崩溃相反。因此需用 `--trunc_tail_penalty 0` 做**单变量判定实验**：仍崩 → 排除本项；跑过 step 3 → 确认相关、再深挖 `sw`/`group_ok` 链。
- **当前下一动作（2026-10-01 当时；已被 §4.11 取代，保留作历史）**：第一轮信用实验（`native_p4_credit`，~436组）held-out 已确认有效（step200/300 并列 69.8%，Δ+9.7pp，p=0.001/0.002）；主模型取 step200 或 step300。下一步两件事：①补评测端**平均输出 token / 工具轮数**——效率目标尚未证明；②用 `rlab_new/native_p4_trunc` 单变量验证 `trunc_tail_penalty`（§4.9 验收 + acc 不低于 step200/300 的 held-out 硬门槛），不要在同批同时改 overlong。若要对比新旧机制，先把旧 `native_p4/step_400` 放进同一评测 campaign。**②的首跑在 step 1 后遇到生成端静默死亡（见上一条），先按 exitcode/dmesg 定位死因再重跑。**
- **轨迹与段长落盘（2026-10-02 新增诊断能力，回答"模型到底写了什么"）**：此前 record.jsonl 与 eval per-item **都只有统计**，于是"末段是答题被掐"与"写散文跑飞"在数据里同形（处置相反：前者抬 `answer_reserve`，后者抬 reserve 只会把 B 换成 C）。
  · **段长进 record**：`rollout` 落 per-sample `segl`（assistant 段长表）/`tsegl`（工具段长表），ok 与 uniform 两个落盘点都带；`python -m rlab.analysis --no-boxed-breakdown <record>` 多出「B 桶末段长度画像」表——按桶给末段 p25/p50/p75 与分箱（`<512/512~1024/1024~2048/≥2048`），并以「有 boxed 成功收尾」的末段长度作**可比标尺**；同表还给出回包实测 token 与 `clen≥0.9cap` 自证列。**读法**：B 行落在 `≥2048`（或 B2 零调用占多数）→ 治啰嗦；B 行集中在 reserve 附近且与「有boxed」p50 同量级 → 才该抬 `answer_reserve`。旧 record 无 `segl` → 表上明说"无法画像"，不造假表（需要重新 rollout）。
  · **全量文本落盘**：训练加 `--traj_dump` → `<out_dir>/traj.jsonl`；评测加 `--dump_traj`（`eval_vllm_one.py` / `eval_vllm.py` / `rlab.eval` 三条入口都认）→ `<out>.traj.jsonl`。每行**一条轨迹**，**段级** `segs:[{kind,len,text}]` + `status(ok/uniform/overlong/eval)` + `gen_version` + 与 record 同源的计数与 ±1 分数；**含被丢弃的 uniform/overlong attempt**（它们整组不进 record，是记录口径幸存者偏差的主体）。纯观测面，**不进 `run_signature`**（打开它不会让旧 out_dir 变外来签名），可与任何单变量实验同开。体积：训练全量约 40~80MB/run；评测 `val_n=8`、500 题约 56MB/ckpt。
  · **两个阅读陷阱**：①`traj.jsonl` 是**追加写**，同 `out_dir` 多次 run 会混在一起（按 `gen_version` 回退切会话，或换 out_dir）；②文件里含 `<tool_call>` 一类标签，用 read/控制台看会被渲染成无括号普通词——判字节真伪要逐字符 `ord` 直出，且 **dump 是诊断产物，不要喂回模型**。
- **评测档位（`--val_n`）必须与结论绑定（2026-10-02 险情）**：`--val_n` 默认 **1（贪心）**；`--val_n>1` 是 Average@N 采样档（temp 1.0/top_p 0.7），**同一 checkpoint 的 acc / `both` / 平均token / 工具轮数在两道之间整体位移**（实测 s200 两档 acc 差 10pp 量级）。踩点在于：历史上所有表都忘了记这一列，于是"同名的 step200" 会被当成同一个数比较。配套两条：①**一次评测内档位必须统一**（别一套表里混贪心与采样），四个臂一起评；②`McNemar b/c` 只属于贪心档——采样档由 `analysis.paired_test_auto` 自动改走配对均值 z 检验，若汇总表仍印 b/c，就是档位没被识别（查 per-item 的 `val_n` 字段）。
- **轨迹 dump 的读法与两个坑（2026-10-02 实测）**：①工具段有两个文本字段——`text` 是**模板渲染**（含 `im_end/im_start/tool_response` 等 special token 与回答骨架，4898 条全部以同一个包装开头），**读观察必须用 `body`**（消毒后的纯回包，含 `[etype]` 前缀与 `[budget]` 行）；长度字段 `len` 是 `text` 的 token 数，别当回包大小。②`traj.jsonl` 是**追加写**，同 out_dir 多次 run 会混（按 `gen_version` 回退切会话，或换 out_dir）。③dump 里含调用标记，用 read/控制台看会被渲染成无括号普通词——判字节真伪逐字符 `ord` 直出，且 **dump 是诊断产物，不要喂回模型**。④**落盘行的计数字段必须来自 `code_stats`**：eval 侧首版漏传 → 整份 dump 的 `code_used/trunc_final/code_wasted/invalid_final/code_ok/ctx_full` 全是 0，桶判定 100% 变假 `E`；现已加"含工具段却 `code_used=0`"自检告警，同类接线缺口当场可见。
- **`trunc_in_call`（2026-10-02 新增分析字段·token 档专用）**：token 预算档下"预算耗尽"不置 `invalid_final` → **A 桶恒 0**，"调用写到一半被墙切"被 B 桶吞掉（真机轨迹实测占无 boxed 的 27%）。新键 `trunc_in_call` 标记切点是否落在未闭合调用块里，`--no-boxed-breakdown` 多一列 `B_call`（与 `B2` 是两个轴、可同时为真）；旧 record 缺键 → 列显 "—"、历史口径不变。判据 = 开标记多于闭标记 且 parse 判 invalid（标记从 `rlab.protocol` 正则派生）。
- **held-out 数据缺陷：同题面 → 多答案键（2026-10-02 实测）**：`dev` 里同一道题出现两次，答案键分别是 `-1` 与 `683`（前者 8 条 acc=0，后者 7/8）。后果：①不可能得分的样本进分母；②配对键 `qk=sha1(Q)` 碰撞 → 配对按 qk 建 dict 只留一条，**298 题静默并成 297**。**按 (Q,A) 去重会原样保留"同题多答案"**；eval 侧已加重复 qk 告警，数据侧待清理。
- **轨迹归因的第一结论（§4.12）**：失败是"写不完"不是"不会做"——无 boxed 822 条里 43% 是散文写到墙、27% 是调用未闭合被墙切；成功作答的末段 p75 只有 667 token ⇒ **`answer_reserve=1024` 对 85% 的成功轨迹都够，抬 reserve 不是对症杠杆**；而 `budget_hint` 早已开着，仍有 19.3% 死在调用上 ⇒ 提示必要不充分。
- **生成端 SIGHUP 死亡（2026-10-02 实锤·非崩溃信号）**：生成端 `exitcode=-1` = **SIGHUP(1)**——既不是崩溃（SEGV/ABRT 等 faulthandler 会 dump），也不是 OOM-kill（那是 -9）。**`enable()` 不登记 HUP/TERM/INT——但同日加的 `register(chain=True)` 会为它们打栈**（10-02 20:46 那次的日志里就有全线程栈，落点是 `vllm generate→_prepare_inputs→compute_slot_mapping`，即当时在**正常 decode**）。旧诊断文案把所有负信号都指向「日志上方应有 faulthandler 栈」是误导的一半：对 HUP/TERM/INT 要问的是「有没有那段 register 打的栈」，不是「faulthandler 会不会打」（已修：分崩溃/非崩溃两档，HUP 档给出会话/监控来源排查清单）。现场形态：version=72 权重推送后生成端卡约 5 分钟零产出，训练端只见 ~100 行 "waiting for batch..."，随后被外部 HUP 带走；ref_server 存活到最后清理。查法三连：①启动方式与会话（kubectl exec -it/ssh/tmux 断开、watchdog kill）；②生成端最后一行 [rollout] 日志停在哪（apply_model 权重同步 / vLLM generate / ref_server 上传——三者都无超时，卡住即无限等）；③`curl -s localhost:59875/health` 验 ref_server 是否同冻。**生成端已补 HUP/TERM/INT 的 `faulthandler.register(chain=True)`**（commit 后 pod 需 git pull）：再被外部信号杀死会先 dump **全线程 Python 栈**（卡在哪个调用当场可见）再按原信号终止，父端 exitcode 语义不变（HUP 仍 -1）。**（2026-10-03 追加·探针定案 + 显式定档）** pod 实测：`nohup` 下 HUP=`SIG_IGN`，且 `import torch`、`import deepspeed`、`deepspeed.init_distributed()` 都不改处置，**spawn 子进程照常继承 SIG_IGN** ⇒「生成端天生丢 IGN」不成立（exec 只保留 IGN、只把父进程手里的 **handler** 复位成 SIG_DFL，而这三个嫌疑一个 handler 都没装）。于是两次事故真死出 `-1` 只剩一种解释：**生成端出生时处置是 `SIG_DFL`**（那次 run 不是 nohup 起的，或父进程的 IGN 被谁换成了 handler → exec 复位成 DFL）；另外 train/ref_server 与生成端同进程组却都活到最后 ⇒ 信号是**点名发给生成端**的，不是组投递（组投递会三个一起死）。据此已改 `_spawn_gen`：**先打印出生处置 + pid/pgid/sid（取证，放在改处置之前）→ 显式 `SIGHUP=SIG_IGN` → 再 register** ⇒ HUP 从此只留全线程栈、不再杀生成端；`TERM`/`INT` 终止语义不变（要停 run 用 TERM/-9，别用 HUP）。副作用须知：生成端对 HUP 免疫后，若 train 端被异常带走而没清理子进程，生成端会成孤儿占着 GPU0。**（2026-10-03 收口）** bash 层也排除了：`nohup bash outer.sh → bash inner.sh → {后台 python / 前台 python}` 四层读 /proc 掩码**全 `ignored=1`**；但把 spawn 前整条 Python 路径（imports / `init_distributed` / `get_config` / `mp.Queue` / `Process.start`）在 **IGN 预设**下复现也全是 `ignored=1`（连 spawn 子进程都是）⇒ 真机那次"生成端出生即 DFL"的动作**合成复现不出**，落点只剩 `main()` 的 argparse + `get_config`(真实 overrides) 一带或竞态；继续合成逼近性价比低，改为真机三点定位：`[sig] main 入口` / `[sig] run_training 入口` / `[sig] spawn 生成端之前`，哪一段 `ignored` 从 1 变 0 一目了然。修复不依赖它：两层显式 `SIG_IGN`（train spawn 前 + 生成端入口）已让该故障不可能复发。**探针方法论教训**：`ignored` 类取证探针**必须预设 IGN 或跑在 nohup 下**——从 DFL 起跑时"Ign→DFL 被复位"完全不可见，本轮因此白跑一轮。
- **当前下一动作（2026-10-02）**：`native_p4_trunc` held-out 出炉，**本 campaign 内只有 step400 显著**（+8.4pp，p=0.002，唯一过 Bonferroni），主模型取 step400。四件事按序做：①核两次 eval 的 `eval_protocol`（`gpu_mem` 优先，实测可差 7pp）；②把 `native_p4_credit` 的 step200/step300 与本次 step400 放进**同一次评测**——这是回答"ttp0.1 有没有用"的唯一方式（跨 campaign 不可比：BASE 61.4 vs 60.1）；③用 `--dump_traj` 抽 step200 vs step400 验"少调用多散文"假说；④按 §4.12 的轨迹归因选下一轮单变量（首选 `len_eff_w`，**不要抬 `answer_reserve`**）并清掉 dev 的同题多答案。细节与全部读数见 §4.11 / §4.12。
- **本地工作区状态**：截断末段信用代码已提交推送；本轮改动涉及 [config.py](rlab/config.py)、[rollout.py](rlab/rollout.py)、[train.py](rlab/train.py)、[analysis.py](rlab/analysis.py)、[test_retool_cpu.py](rlab/tests/test_retool_cpu.py)、[test_smoke_cpu.py](rlab/tests/test_smoke_cpu.py) 与本文档。
- **本地工作区状态（2026-10-02 追加）**：两轮诊断能力已提交推送（`f65c62c` B 桶末段长度画像、`62886f6` 轨迹全量落盘），涉及 [config.py](rlab/config.py)、[rollout.py](rlab/rollout.py)、[train.py](rlab/train.py)、[analysis.py](rlab/analysis.py)、[eval.py](rlab/eval.py)、[eval_vllm.py](eval_vllm.py)、[eval_vllm_one.py](eval_vllm_one.py)、[test_retool_cpu.py](rlab/tests/test_retool_cpu.py)、[test_smoke_cpu.py](rlab/tests/test_smoke_cpu.py) 与本文档。**新字段/新档位对旧 record 与旧 eval json 一律向后兼容**（缺键按"无该字段"降级，不造假值）。
