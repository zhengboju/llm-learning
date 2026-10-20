# p11d 启动配置定版与验收判据（2026-10-20）

**上游**：[p11c 训练诊断](p11-diagnosis.md)（信号密度修好，条件精度首次自涨）
**下一步方案**：[p11e token 效率](p11e-token-efficiency-plan.md)（未实现，勿与本轮混跑）
**工具**：`python -m rlab.audit_config --from_run_info <run_info.json>`

---

## 1. 本轮唯一目标：坐实「守卫 + 课程」这条因果链

p11c 拿到了全项目第一次「条件精度自己涨」（前 9 窗 58.9% → 后 10 窗 68.3%），
但**被幸存者偏差污染**（丢弃率从 9% 升到近窗 17%，混合组 acc 机械上抬），
且**没跑 held-out**。所以本轮不是加新机制，而是：

1. 把已修好的接线（`sw` 恒 1、`budget_target` 三信道）真正跑起来；
2. **用 held-out 判定 p11c 的 +8.5pp 是真是假**。

**本轮不引入任何新奖励项**。这是单变量纪律：p11c 的因果链第一次干净，
中途叠 p11e 的效率训练会把归属彻底搅乱（[p11e][§6]）。

---

## 2. 相对你当前 run.sh 的四处改动

| # | 改动 | 类型 | 依据 |
|---|---|---|---|
| 1 | **删** `--tool_waste_penalty 0.20` | 纯配置 | guard+tcw 下被 `validate_guard_conflicts` 自动归零，留着只会在下次读日志时误导。当前废码负信用**完全由 `trunc_counts_as_wrong` 的 acc=−1 承担**（实测 −1.5，比覆盖后的 −0.30 强 5 倍） |
| 2 | **加** `--budget_target 8192` | 纯配置 | p9 锚机制（承诺紧、余量松）此前**只靠提示文本**；`[budget]` 回包与 `_can` 判据仍漏出真实总量 12288。p11c 的 avg_clen 冲到 6056~8056、B 桶 272/272 全部 `clen≥0.9cap` 就是这条漏洞的签名 |
| 3 | **加** `--eval_during_training --eval_n 300` | 纯配置 | 唯一缺的表。p10 的 900s 超时已修为 3600，现在开是安全的 |
| 4 | **换** `--out_dir rlab_new/native_p11d_anchor` | 纯配置 | p11c 目录已占用；签名多了 `-bt8192`，若同签名重跑会**静默覆盖 `step_N`**（p11 已踩过两次，数据链断） |

### 2.1 评测口径：held-out 必须先关难度过滤（2026-10-20 补）

内嵌评测（`--eval_during_training`）用的是 ckpt 的 `run_info`，而 `run_info` 里有
`difficulty_path` → **自动按 `difficulty_band` 过滤**。这会直接毁掉本轮的唯一任务：

| 问题 | 后果 |
|---|---|
| `test` split = `dev.jsonl`，而 `--difficulty_path` 是 `probe --pool train` 探的**只含训练池**的表 | dev 1000 题**全落 missing → 池 0 题**。旧版崩在 `1.96*0.5**0.5/0**0.5` 的 `ZeroDivisionError`，traceback 完全指不到"池子空了" |
| 即使表覆盖 dev，过滤后 acc 是「**条件于难度带 0.125~0.625**」的量 | 与 p9（全量 17k，fmt 93.9%）以及任何 band 不同的 run **不可比**——本项目已因跨池比较出过两次事故（[p11-diagnosis §6](p11-diagnosis.md)） |

**所以本轮主评测口径必须是完整 held-out**：加 `--no-difficulty_filter`。
它已接线到三个入口（`eval_vllm_one.py` / `eval_vllm.py` / `rlab.eval`），并落盘
`eval_protocol.difficulty_filter = {mode, path, band, kept, total}` 自证。

> `--no-difficulty_filter` 只关**评测侧**过滤，训练池仍由 `--difficulty_path` 控制，
> 两者互不影响：课程决定"学什么"，评测口径决定"跟谁比"。


**其余全部保持不变**——包括 `--overlong_ref 8192 --overlong_buffer 2048`
（你的显式值是对的：触发线 6144，实测覆盖 56% 样本，是唯一有效的长度项）。

---

## 3. 定版命令

```bash
REF_GPU=1 ATTN_IMPL=flash_attention_2 bash rlab/run_gsm8k.sh retool_math \
  rlab_out/sft_budget_v3/epoch1 \
  --chat_template_kwargs '{"enable_thinking": false}' \
  --tool_protocol native --native_tool_style function --native_stop_at_call \
  --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
  --vllm_gen_logps --vllm_logprobs_n 1 \
  --gen_gpu_mem 0.7 --micro_rows 1 --optim_8bit \
  --gen_update_steps 8 --gradient_accumulation_steps 8 \
  --overlong_shaping --overlong_fail_only --overlong_max_penalty 0.5 \
  --overlong_ref 8192 --overlong_buffer 2048 \
  --len_penalty_w 0 \
  --code_attempt_w 0.05 --code_shaping_once \
  --len_eff_w 0.15 --len_eff_penalty_w 0 \
  --max_traj_tokens 12288 --budget_target 8192 \
  --round_gen_tokens 2048 --answer_reserve 1024 --budget_hint \
  --tool_call_cost 0.02 \
  --difficulty_path rlab_out/difficulty_v3e1_train.jsonl \
  --difficulty_band 0.125 0.625 \
  --task_adv_guard --aux_clip_alpha 0.5 --trunc_counts_as_wrong \
  --max_context_tokens 13312 --traj_dump \
  --system_prompt_file rlab_out/sp_anchor8192.txt \
  --lr 1e-6 --seed 42 --steps 400 --save_steps 100 \
  --eval_during_training --eval_n 300 \
  --out_dir rlab_new/native_p11d_anchor
```

**删掉的三项**（相对你的原命令）：
- `--no-eval_during_training`（改为开启）
- `--code_w 0.05`（与 `code_attempt_w` 重复；且在 code 率 98% 时两者都是零梯度）
- `--tool_waste_penalty 0.20`（被自动归零）

**内嵌评测的口径**：`--eval_during_training` 现在仍按 `run_info.difficulty_path`
过滤，而那张表是 `--pool train` 探的（不含 dev）→ **test split 会 0 题**。
先确认这一点：

```bash
# 训练前就该看到这个（表覆盖 dev 吗）
python -c "
import json
t=[json.loads(l) for l in open('rlab_out/difficulty_v3e1_train.jsonl',encoding='utf-8') if l.strip()]
m=[r.get('probe_meta') or {} for r in t[:5]]
print('行数',len(t),'pool=',{x.get('pool') for x in m})
"
```

- 若 `pool={'train'}` → 内嵌 test 评测会 0 题（已 fail-fast，但那是白跑一轮）。
  正确做法：**内嵌只看训练池内趋势**（保留过滤），**泛化读数用手动 held-out + `--no-difficulty_filter`**：

```bash
python -m rlab.eval --algo retool_math \
  --base_path=./rlab_out/sft_budget_v3/epoch1 \
  --models step100=./rlab_new/native_p11d_anchor/step_100 \
           step200=./rlab_new/native_p11d_anchor/step_200 \
           step300=./rlab_new/native_p11d_anchor/step_300 \
           step400=./rlab_new/native_p11d_anchor/step_400 \
  --no-difficulty_filter \
  --gpus 0,1 --gpu_mem 0.2
```

- 若 `pool={'train_dev'}` → 内嵌 test 可用，但 acc 仍是「条件于 band」的量，
  跨 run 比较前必须核对两边 `eval_protocol.difficulty_filter.band` 相同。

⚠ `--gpu_mem` 必须给 **0.2**（对齐 `run_info.eval_gpu_mem`），否则同权重同题
差 +7.0pp（[docs/07](../docs/07-vllm-logprobs-non-determinism-4b.md) 已实测）。


---

## 4. 生效值自证（audit 实跑输出）

```
① 生效值（★ = 被自动改写）
   overlong_ref = 8192    overlong_buffer = 2048   max_penalty = 0.5
   overlong_fail_only = on
 ★ tool_waste_penalty = 0   已被 guard+tcw 自动归零
 ★ trunc_tail_penalty = 0   已被 guard+tcw 自动归零
   len_eff_w = 0.15   len_eff_penalty_w = 0   len_penalty_w = 0
   task_adv_guard = on   trunc_counts_as_wrong = on   aux_clip_alpha = 0.5
   budget_target = 8192   max_traj_tokens = 12288   answer_reserve = 1024

③ 预算几何
   保险丝 12288 / 通告 8192 → 差 4096 = 模型看不见的溢出余量
   overlong 触发线 6144，受罚区间宽 6144 token
   参照 clen 样本命中率 56%

④ 逐项组内施力（正确性间距 = 2.0）
   overlong          0.250   12.5%   ← 唯一 >10%，有效
   len_eff_w         0.039    2.0%   ← 保留（通过侧唯一的效率信号）
   code_attempt_w    0.000    0.0%   ← 零梯度，但它是**保险**（见 §6）
   (④b) tool_call_cost 0.040         token 级，不进组均值
```

**读法**：这张表说明本轮的奖励结构是「1 个有效辅助项 + 1 个微弱效率项 +
1 个零成本保险 + 1 个 token 级成本项」。这不是简化到极致，但**每一项都能回答
"它改不改变组内排序、改多少"**——而这正是上一轮 31 个键里做不到的。

---

## 5. 验收判据（写死，防事后解释）

### 5.1 硬判据（必须同批、同 seed、N≥300）

| 指标 | 判据 | 不接受 |
|---|---|---|
| **held-out acc** | 相对 `sft_budget_v3/epoch1` 起点 **+3pp 以上**（McNemar p<0.05） | "训练 acc 涨了" |
| **条件精度**（acc/fmt） | 相对 p11c 训练口径**不能回退** | 单看 acc 或 fmt |
| **fmt** | ≥ 75%（**与 p9 的 93.9% 不可比**——池子不同，见 p11-diagnosis §6） | 跨池比较 |
| **符号翻转** | 0（`sigstats` 的「正确性为正被压成非正」必须 0） | 有翻转还继续跑 |
| **M_task** | ≥ 85% | — |

### 5.2 过程判据（每 25 个 update 看一次）

| 指标 | 期望方向 | 判据 |
|---|---|---|
| `avg_clen` | **下降**（这是 `budget_target` 的直接靶） | 若仍 ≥7000，说明通告没压住，需查回包数字 |
| `trunc率` | **不涨**（涨 >3pp 说明承诺太紧） | 涨超就抬 `budget_target` 到 10240 |
| 零梯度占比 | **显著下降**（`sw` 已修，应为 0） | 若非 0，说明 `sw` 修复没生效——立即停 |
| B 桶（散文撞墙） | 下降 | 这是 `budget_target` 唯一的机制解释 |

### 5.3 三条"立即停"信号

1. **零梯度占比 ≠ 0** → `sw` 修复没生效（本轮最关键的接线）
2. **符号翻转 > 0** → doc15 §7 停线，说明 `aux_clip` 失效
3. **`[budget]` 回包里出现 >8192 的数字** → `budget_target` 没生效，锚仍漏

第 3 条可直接查：
```bash
grep -o '\[budget\] [0-9]*' rlab_new/native_p11d_anchor/traj.jsonl \
  | awk '{print $2}' | sort -n | tail -3     # 必须全都 ≤ 8192
```

---

## 6. 保留但"此刻零梯度"的两项——为什么仍留着

| 项 | 当前状态 | 为什么留 |
|---|---|---|
| `code_attempt_w 0.05` + `code_shaping_once` | code 率 98% 时全组同值 → 零梯度 | 它是**只在坏情况下付费的保险**：p6 曾把 code 率压到 3%。一旦组内有人 0 调用，它立刻变成真实信号（幅度 0.05×次数差） |
| `len_eff_w 0.15` | 2% 幅度（噪音级） | **通过侧唯一**的效率信号。目标含 token 效率，删了就只剩失败侧惩罚。2% 不足以压 token，但方向正确且不干扰 |

**这两项的价值不在当前的幅度，在"配置已经接好、坏情况来时自动生效"。**
删掉它们省不了算力（零梯度项不进反向），但会让将来回溯时多一次"当初为什么删"。

---

## 7. 明确不做的事（防下一轮重复提议）

| 不做 | 原因 |
|---|---|
| 降 GAS 提更多 update | **本轮先不动**。p11c 的因果链只能容纳"接线修复"这一个变量；GAS 改的是更新次数与单步方差，属于第二变量。若本轮 held-out 兑现，p11e 阶段再作为吞吐杠杆单独 A/B |
| 上 p11e 的全对组效率训练 | 见 [p11e][§6]：会改变上传组人群构成，与本轮归属冲突 |
| 改 `adv_mode` 为 `group_std` | p9 定论：8 条组可用，但需单变量 A/B；本轮不动 |
| 调 `aux_clip_alpha` | 它是当前"翻转=0"的结构来源，动了就没守住的东西 |
| 改 `difficulty_band` | 池子固定是 p11c 结论成立的前提；换 band 等于换实验 |
| 抬 `max_traj_tokens` | `budget_target` 已经把"模型看见的预算"与"保险丝"解耦，抬保险丝只增加显存风险 |

---

## 8. 成本预期（用于判断何时该停）

| 项 | p11c 实测 | p11d 预期 | 依据 |
|---|---|---|---|
| token/update | 466k | **350~420k** | `avg_clen` 6056→约 4500~5500（B 桶提前收尾） |
| updates | 42 / 16.7h | 50 / 400 微步 | GAS 不变 |
| 累计 | 16.7h | **约 13~18h** | — |

**若 avg_clen 不降反升**（≥ p11c 的 6056），说明 `budget_target` 的提示没有
被模型接收——不要继续跑，先查 §5.3 的第 3 条（回包数字），那是唯一能区分
"模型不听话"与"接线没生效"的判据。

---

## 9. 下一步（本轮跑完之后）

按顺序，每步都等前一步的 held-out：

1. **p11d held-out 判定** → 若 acc +3pp 且条件精度不回退，则「守卫+课程」定版。
2. **p11e**（全对组效率训练）→ 只在第 1 步兑现后开。见 [p11e](p11e-token-efficiency-plan.md)。
3. **GAS 调优 A/B** → 在 p11e 之后，作为纯吞吐杠杆。

**任何一步不兑现，就停在那一步查，不要叠加下一步**——这是 p1~p11 全部失败
的共同形态（[docs/08 §8.1](08-sft-cold-start.md)：四个变量同时动，故障归因不可能收敛）。
