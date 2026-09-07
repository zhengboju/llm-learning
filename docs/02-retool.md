# 阶段2 报告：ReTool 式代码交织 RL（TIR）

> 状态：**开发完成，待真机验收**（2×H20 pod）。本文先落协议设计、实现清单、
> 关键学习点的数值实锤与运行手册；训练结果出来后补 §7 实测曲线。
> 前置：阶段1 六算法对比闭环（`docs/01-loss-variants.md`）。

---

## 1. 任务定义：为什么是 TIR

把 `Auto_Program/hjy_grpo_program.py` 的玩具实现正规化：模型在 think 段内可写
python 围栏代码块 → 环境执行 → 结果回填上下文 → 继续推理 → 最终在 answer 标签内
给出答案。任务沿用 GSM8K（与阶段1 的 acc/fmt 口径完全可比），loss 复用 GRPO
（group_std + sample_mean + β=0.04 KL）——**阶段2 的全部新东西都在 rollout 与
mask 契约里，不在 loss 里**，这正是它作为教学阶段的价值：多轮轨迹如何过协议。

与玩具实现的本质差异：

| 项 | Auto_Program 玩具 | rlab 阶段2 |
|---|---|---|
| 执行方式 | `exec()` 进程内 + signal.alarm | subprocess 隔离 + SIGKILL 超时 + 内存上限 |
| alarm 限制 | 只能主线程（线程环境必炸，本项目实锤教训） | 无此限制 |
| 轨迹结构 | 文本拼接，工具 token 也进 loss（隐性错误） | 显式多段 + mask 契约：工具 token 不进 loss |
| logps | vLLM prompt_logprobs 按段拼 | 全序列一次前向（因果性保证等价，见 §4.2） |
| 奖励 | 手写权重切换（update_model_num 计数） | 可配置 cold/hot 权重 + 切换步数 |

---

## 2. 轨迹协议（protocol.py）

一条多段轨迹在 token 序列上是一个 prompt + 交替的 assistant/tool 段：

```
[prompt] [assistant_1] [tool_1] [assistant_2] [tool_2] ... [assistant_N]
```

- **assistant 段** = 模型生成的 token → 进 loss（mask=1）
- **tool 段** = 沙箱输出，包裹在 `[TOOL RESULT]` / `[/TOOL RESULT]` 纯文本标记内
  → **只作上下文，mask=0 不进 loss**
- 标记用方括号纯文本而非尖括号，与格式标签（think 起止标签）不在同一命名空间，
  且遵守零标签字面量铁律：一切标签/标记字节从 rlab 导入，文件里绝不手写。

代码块检测：`extract_python_blocks()`（`_PY_FENCE_RE`，DOTALL）取**最后一个完整
围栏块**执行（与玩具实现一致）；本轮无代码块 → 该样本轨迹结束（随后应给答案）。

### mask 契约（本阶段最重要的学习点）

工具返回 token 若进 loss，会注入两类假信号：
1. **KL 污染**：策略对沙箱输出的 logp 是"困惑度"而非"选择"，与 ref 的差可能巨大
   → k3 KL 爆炸，把整条轨迹的 loss 推高；
2. **假梯度**：工具 token 的 ratio 反映一个策略根本无法控制的分布（沙箱写什么
   模型管不着），对它求梯度纯属噪声。

§4.1 有数值实锤。协议上：生成端按段边界精确构造 (B,T) 0/1 mask，meta 带
`"has_mask":1`，extras 布局 `[gen_logps, mask, acc, fmt]`；训练端直接采用，
**绝不允许用 `inputs != pad` 重算**（pad 重算会把工具段当模型生成——正是错误示范）。
无 `has_mask` 时按旧布局解析（阶段0/1 完全向后兼容，已有回归测试锁定）。

---

## 3. 实现清单（commit 见 git log）

| 文件 | 内容 |
|---|---|
| `rlab/sandbox.py` | **新**。`run_code()`：subprocess `-I -E` 隔离执行 + 超时 SIGKILL + Linux RLIMIT_AS 内存上限 + stdout/stderr 截断。返回 `{ok, returncode, timed_out, duration, display, stderr}` |
| `rlab/protocol.py` | `TOOL_START/TOOL_END`、`extract_python_blocks`、`segment_mask_from_spans` 纯函数；`encode/decode_batch` 增 mask 槽位（`has_mask` 元数据，向后兼容） |
| `rlab/reward.py` | `reward_code`（成功执行次数 × 0.1 小权重）、`reward_phase`（cold/hot 切换纯逻辑）、`total_reward_retool`（acc+fmt+code 组合，cold=(1,2,2) / hot=(2,1,1)） |
| `rlab/config.py` | `retool` preset（loss 同 grpo）+ 阶段2 超参（max_rounds=3 / round_gen_tokens=280 / sandbox 超时与内存 / code_w=0.1 / reward_switch_step=256）+ retool 系统提示（复用 BASE 提示 + 代码工具说明，零标签字面量） |
| `rlab/losses.py` | `retool` 映射进 grpo loss 分支；ALGOS 注册 |
| `rlab/rollout.py` | `multi_turn_rollout_group`（组内并行多轮生成→检测→沙箱→回填→续写）+ 分段 tokenize 构造 mask + 全序列 gen_logps + record 增 code_used/code_ok/phase |
| `rlab/train.py` | 有 `has_mask` 时用协议下发的 mask，否则按 pad 重算（兼容） |
| `eval_vllm_one.py` | `--retool`：多轮贪心生成（复用 `multi_turn_rollout_group`）+ 代码调用率/成功率/平均轮次指标 |
| `eval_vllm.py` / `rlab/eval.py` | `--retool` 透传 |
| `rlab/tests/test_retool_cpu.py` | **34 项 CPU 验收**（见 §4） |

设计要点（省掉了规划文档担心的"logps 按段拼接"）：训练端与生成端都做**全序列
一次前向**——因果注意力保证第 t 个 token 的 logp 只依赖前缀，与逐段前向严格等价
（§4.2 用 tiny GPT2 数值验证）。工具段只是被模型"读到"的上下文，不产生 loss 项。

---

## 4. CPU 验收测试（34 项，本机可跑）

### 4.1 【学习点】工具 token mask 错/对对照 A/B

构造 2 样本轨迹（s0: assistant×3 + 工具×2；s1: assistant×4），工具 token 上
policy logp=-8（策略对沙箱输出极困惑）、ref logp=-2：

| 对照 | loss | s0 工具位梯度 |
|---|---|---|
| mask 正确（工具=0） | **0.000**（纯 assistant，KL=0） | **严格 0** |
| mask 错误（工具=1） | **≈3.4**（被工具位 KL 污染） | **非 0**（假信号） |

- 单个工具 token 的 k3 KL = exp(6)−6−1 ≈ 396，β=0.04 后每 token 贡献 ≈15.9 的
  loss——两条工具 token 就把样本均值从 0 推到 3.4；
- β=0 时 loss 数值上几乎不变，但**假梯度仍在**（工具位梯度非 0）——说明 KL 污染
  只是表象，把"模型无法控制的 token"放进 loss 本身就是错的；
- 结论：mask 错误 = 对不可控分布求策略梯度 + KL 虚高 → 训练信号被系统性污染。

### 4.2 logps 对齐证明（tiny GPT2）

assistant 段的逐 token logp：全序列一次前向 == 只跑"prompt+该段"前缀前向
（allclose 1e-5）。这是"训练端不需要逐段拼接"的因果性证明。

### 4.3 其余覆盖

沙箱（成功/异常/死循环超时/空代码/输出炸弹截断/真实子进程隔离）、阶段2 奖励数学
（cold 3.4 / hot 3.1 / 答错 −1）与切换逻辑、协议 mask roundtrip + 旧布局兼容、
retool preset 锁定、轨迹级 mask（工具位梯度 0 + assistant 位有梯度）。

---

## 5. 奖励设计与冷启动切换

```
r = w_acc·acc + w_fmt·fmt + w_code·(code_ok × 0.1)
cold（step<256）: (1, 2, 2)   # 先学会"写代码 + 套格式"，正确性权重小
hot （step≥256）: (2, 1, 1)   # 正确性主导（与阶段1 的 2·acc+fmt 对齐）
```

- `code_ok` = 执行成功（exit 0 且有输出）的代码块次数，上限 3 轮 → 代码项最大 0.3，
  保持"小权重引导"不淹没主信号；
- 切换阈值沿用 Auto_Program 的语义：16 次权重推送 × gen_update_steps(16) = 256 步；
  生成端用"推送次数×gen_update_steps"近似 optimizer step（记录在 record.phase）；
- **消融钩子**：`reward_switch_step` 设超大值（如 10**9）即等价于"全程 cold"，
  可验证权重切换的必要性（规划文档要求的验证点，跑法见 §6）。

---

## 6. 运行手册（pod 2×H20）

```bash
# 训练（协议/卡位/环境变量与阶段1 完全一致；retool 走 passthrough ref_server）
bash rlab/run_gsm8k.sh retool /root/Qwen2.5-3B --seed 42

# 评测（--retool：评测也执行代码，并输出代码调用率；BASE 也用同协议评）
python -m rlab.eval --retool --models retool200=./rlab_out/retool/step_200 retool300=./rlab_out/retool/step_300

# 训练中代码调用率曲线（record.jsonl 新增 code_used/code_ok/phase 字段）
python -m rlab.analysis --record rlab_out/retool/record.jsonl
```

建议跑法（对应验收清单）：
1. **主实验**：retool 300 步 seed=42，看 acc 与代码调用率曲线；
2. **mask 对照小实验**（本阶段最重要的学习点，真机版）：同一 checkpoint，
   一份正常训练，一份把 `train.py` 的 mask 强制改为 pad 重算（即"工具段进 loss"
   的错误版，一行改动）——对比两者 acc 曲线与 KL 量级，把 §4.1 的 CPU 实锤
   升级成真机实锤；
3. **冷启动消融**：`--reward_switch_step 1000000000`（全程 cold）vs 默认 256，
   验证权重切换的必要性。

显存预算：多段轨迹总长上限 `max_context_tokens=2200`（prompt ~400 + 3 轮 ×
(280 生成 + ≤150 工具输出)），超限整组丢弃重采；比阶段1 的 512 长约 3×，H20
单卡 ZeRO-0 3B 可承受，若 OOM 先降 `round_gen_tokens` 或 `max_rounds`。

---

## 7. 实测结果（待真机填）

| 模型 | acc | fmt | 代码调用率 | 成功率 | 平均轮次 |
|---|---|---|---|---|---|
| BASE（--retool 同协议） | 待填 | 待填 | 待填 | 待填 | 待填 |
| retool_step200 | 待填 | 待填 | 待填 | 待填 | 待填 |
| retool_step300 | 待填 | 待填 | 待填 | 待填 | 待填 |

预期与判读：
- **验收硬指标 1**：acc 显著超 BASE（>3pp 噪声地板）；
- **验收硬指标 2**：代码调用率随训练上升（record 窗口滑动平均），且 hot 阶段
  "会用代码"的样本 acc 更高；
- mask 错误版预期：loss 中 KL 项虚高、acc 曲线明显劣化——若不劣化反而说明
  3B/小 batch 下工具 token 的 KL 污染可被组内标准化吸收（这本身也是有价值的结论）。

## 8. 已知局限与后续

- 沙箱禁网是 best-effort（`-I -E` 隔离 + 无凭证），硬禁网需容器/seccomp——教学
  规模记录在案，不引入该复杂度（sandbox.py 文档字符串同步说明）；
- 每轮取"最后一个完整围栏块"执行，多块并发执行（并行工具调用）留待后续；
- 生成端多轮循环按组内 batch 调 vLLM（prefix caching 自动复用前几轮 KV），
  采样 seed 每轮相同（上下文不同→输出不同，可复现性成立）；
- 阶段3（Search-R1）可直接复用本阶段的多段轨迹协议与 mask 契约，只换
  `run_code` → `search_backend` 与 reward 的 EM 口径。
