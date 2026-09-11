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

### "生成序列 == 训练序列"的隐性缺口：批内左 pad（2026-09-11）

多题并采（`gen_questions_per_attempt=4`）时 `tokenizer(..., padding=True,
padding_side="left")` 会把短题补到**批内最长 prompt**，而 vLLM 生成侧是逐条无 pad
的序列。带 pad 前向有**两处**与生成侧不等价：

1. **注意力键**：pad token 被后面所有真实位置读到——打分时"看到的上下文"多了一串
   模型生成时根本没见过的 token；
2. **位置编码**：主流 HF 实现（Qwen/GPT 系）的 `position_ids` 取的是下标
   （`cache_position = arange`），**不按 attention_mask 做 cumsum 修正**——短题的第一批
   真实 token 拿到的是"下标位置"而非"真实位置"，embedding 处就已经与生成时分叉。

只修 ① 不够：传 `attention_mask` 把 pad 键的注意力权重压到 0（tiny 模型逐位取证
确已生效）之后，真实位与无 pad 前向仍差 1.0 量级——②没修，缺口就还在。
真机上这正是减法① 对拍"中位差 1e-6、1% 位置差到 1e1"的来源：绝大多数位置不敏感，
少数强依赖长程状态/绝对位置的位置被放大（Qwen3.5 的线性注意力层带递归状态，前缀
扰动不随距离衰减）。

**修法**：`strip_left_pad()` —— 逐题剥掉左 pad 再建批，plen 用**本题**真实长度。
两处偏差同时消失，且**不依赖模型是否支持 2D 掩码**（Qwen3.5 混合线性注意力层是否
吃 mask 无法离线验证，故不走掩码这条路）。附带收益：超长判定不再把 pad 宽度算进
每个样本的 token 预算；批更短、更快。

**判据**（tests `[AB]` 组锁死）：批内短题行的逐 token logps == 该行单独无 pad 前向
（实测最大差 9.5e-07）；反证：同一行带 pad 前缀时同位置差 0.27（tiny GPT2）。

**实机验证（2026-09-11，4B 长跑 + 减法①）**：修复前后各 5 组对拍里，**无 pad 的那一组
逐位完全相同**（mean 5.953e-03 / p99 0.1094 / max 0.375 / 有效位 23919 全等）——天然
对照组；带 pad 的三组 mean 6.6~7.4e-2 → 5.8~7.5e-3、max 3.4~5.7 → 0.34~0.50，**塌到与
该对照组统计不可区分**。残差（vLLM 与 HF torch 的 kernel/bf16 实现差）在训练端的直接
代价，由 `[train][口径]` 在权重同步后的第一步量出（此时 ratio 理论恒 1）：

```
[train][口径] step 1: clip_frac=0.0007 approx_kl=5.11e-04 mean_ratio=1.0000
```

判读：**mean_ratio=1.0000 说明残差是零均值噪声而非偏置**（有偏会直接腐化 ratio 与
advantage 的关系）；clip_frac 0.07% = 被误裁的 token 占比（裁剪带 (0.8,1.28)）；
approx_kl 5.11e-04 = 残差均方 → RMS 残差 0.023（对比 mean|diff| 6e-3 可知是重尾，与
对拍的 p50/p99 形态一致）。对照 torch 副本档应为 clip_frac=0 / approx_kl=0（严格同源）
——即减法① 的代价被量化在"0.07% 误裁 + 5e-4 的 KL 底噪"，比真实 clip_frac 小两个数量级。
注：此 approx_kl 是监控量（mean((log ratio)²)，= (policy−gen) 的均方），**不是 loss 里
的 ref-KL**——后者用 ref_logps，不受本残差影响。

---

## 3. 实现清单（commit 见 git log）

| 文件 | 内容 |
|---|---|
| `rlab/sandbox.py` | **新**。`run_code()`：subprocess `-I -E` 隔离执行 + 超时 SIGKILL + Linux RLIMIT_AS 内存上限 + stdout/stderr 截断。返回 `{ok, returncode, timed_out, duration, display, stderr}` |
| `rlab/protocol.py` | `TOOL_START/TOOL_END`、`extract_python_blocks`、`segment_mask_from_spans` 纯函数；`encode/decode_batch` 增 mask 槽位（`has_mask` 元数据，向后兼容） |
| `rlab/reward.py` | `reward_code`（成功执行次数 × 0.1 小权重）、`reward_phase`（cold/hot 切换纯逻辑）、`total_reward_retool`（acc+fmt+code 组合，cold=(1,2,2) / hot=(2,1,1)）、`strip_code_blocks`/`reward_format_retool`（**打分域=剥离代码块后的回答文本**，2026-09-08 第三轮修复） |
| `rlab/config.py` | `retool` preset（loss 同 grpo）+ 阶段2 超参（max_rounds=3 / round_gen_tokens=400（280→400 防围栏截断灭绝）/ sandbox 超时与内存 / code_w=0.1 / reward_switch_step=256）+ retool 系统提示（复用 BASE 提示 + **MUST** 代码指令（MAY→MUST 防采样率过低灭绝），零标签字面量） |
| `rlab/losses.py` | `retool` 映射进 grpo loss 分支；ALGOS 注册 |
| `rlab/rollout.py` | `multi_turn_rollout_group`（组内并行多轮生成→检测→沙箱→回填→续写）+ 分段 tokenize 构造 mask + 全序列 gen_logps + record 增 code_used/code_ok/phase |
| `rlab/train.py` | 有 `has_mask` 时用协议下发的 mask，否则按 pad 重算（兼容） |
| `eval_vllm_one.py` | `--retool`：多轮贪心生成（复用 `multi_turn_rollout_group`）+ 代码调用率/成功率/平均轮次指标 |
| `eval_vllm.py` / `rlab/eval.py` | `--retool` 透传 |
| `rlab/tests/test_retool_cpu.py` | **71 项 CPU 验收**（见 §4） |

设计要点（省掉了规划文档担心的"logps 按段拼接"）：训练端与生成端都做**全序列
一次前向**——因果注意力保证第 t 个 token 的 logp 只依赖前缀，与逐段前向严格等价
（§4.2 用 tiny GPT2 数值验证）。工具段只是被模型"读到"的上下文，不产生 loss 项。

---

## 4. CPU 验收测试（71 项，本机可跑）

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
- **打分域铁律（第三轮实锤）**：acc/fmt 一律在 `strip_code_blocks(assistant拼接文本)`
  上判——代码是脚手架：①MUST 提示下代码先行会撞 ^ 锚定格式正则（结构性失败→
  fmt 恒 -1→信号死亡+代码灭绝）；②代码里的数字/打印不能当"模型答案"（与"工具
  stdout 不是答案"同一原则）。剥离后格式语义回到"回答文本本身结构"；
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
(400 生成 + ≤500 工具输出)，典型 1-2 轮远低于上限，极端 3 轮全满会超限丢弃重采），
比阶段1 的 512 长约 3×，H20 单卡 ZeRO-0 3B 可承受，若 OOM 先降
`round_gen_tokens` 或 `max_rounds`。

> 评测必须与训练同协议（第三轮教训）：`eval_vllm_one.py` 的 `--round_tokens` 默认
> 取训练 `round_gen_tokens`(400)、`--max_len` 默认取 400+2200=2600（1280 装不下
> 完整多轮轨迹，代码率一涨就会撞 vLLM max_model_len 报错）；两者均已自动对齐，
> 无需手动传参。

---

## 7. 实测结果（GSM8K test，N=300，seed=42，--retool 同协议）

三轮回合逐次修复后的真机记录（每轮都对应一个被实锤的机制缺陷，见 §7.1）：

| 轮次 | 协议 | BASE acc/fmt/both | retool300 acc/fmt/both | code_rate(300) | avg_rounds |
|---|---|---|---|---|---|
| 1 | 全文打分 bug（fdbb915 前） | 64.0/0.0/0.0 | 77.3/0.0/0.0 | 0.0% | 0.0 |
| 2 | assistant 打分修复，round=280, MAY | 65.0/48.7/37.3 | 72.0/97.7/72.0 | 0.0% | 0.0 |
| 3 | round=400, MUST（4f00008） | 61.3/4.7/3.7 | 72.0/0.0/0.0 | 0.67% | 0.007 |

判读（每轮都是"结构性 bug 的精确签名"，非模型行为）：
- 轮1 fmt 精确 0/300 → 打分域 bug（全文含 prompt，^ 锚定必败），fdbb915 修复；
- 轮2 fmt 97.7% → assistant 打分域修复验证通过；code_rate 仍 0 → round 280 截断
  灭绝 + MAY 提示采样率 ~0.3% 进不了分布，4f00008 修复；
- 轮3 MUST 提示"先写代码"反而让 BASE fmt 48.7→4.7、retool300 fmt 精确 0.0%：
  **MUST 诱导的代码先行文本撞上 ^ 锚定的格式正则 → 所有代码样本 fmt 结构性失败**
  → 训练 fmt 信号再次死亡、代码再次被惩罚灭绝（第三类灭绝机制，见 §7.1）。
  eval 还暴露两个协议错位：round_tokens 280（训练是 400）把代码围栏+尾随标签截断、
  max_len 1280 装不下完整多轮轨迹。

验收硬指标（本轮修复后待重跑）：
- **硬指标1**：acc 显著超 BASE（>3pp 噪声地板；BASE 取单轮 512 协议 ~68 为参照，
  --retool 协议的 BASE 被 MUST 强迫写代码，只能同协议内比较）；
- **硬指标2**：code_rate 随训练上升（MUST 已把代码写进采样分布——轮3 的 fmt 归零
  恰好证明 base 在 MUST 下确实大量开围栏，只是 280 轮长+锚定冲突让它们既没被算成
  代码也没拿到格式分），且 hot 阶段"会用代码"的样本 acc 更高；
- 训练期盯 `[健康检查]`：no_code 告警应消失、fmt 不再恒常数、权重指纹 fp64 连续
  两次推送相同才算真冻结。

### 7.1 三轮三次"代码灭绝/信号死亡"的根因链（本阶段最重要教训）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 训练期 fmt 恒 -1 / eval 精确 0.0% | 打分域=全文：^ 锚定在 prompt 开头必败 | assistant 段拼接打分（fdbb915） |
| 2 | code_rate 恒 0.0% | round 280 截断围栏→写代码结构性惩罚 + MAY 采样率 ~0.3% 组内无代码→code 项常数无梯度 | round→400 + 提示 MUST（4f00008） |
| 3 | MUST 后 BASE fmt 48.7→4.7、retool300 fmt 精确 0.0% | MUST"先写代码"→回答文本以 ```python 开局，撞上 ^ 锚定格式正则→所有代码样本 fmt=-1（结构冲突灭绝） | 打分域=剥离代码块后的回答文本（本 commit） |

**元教训**：一个协议里"引入新行为"（写代码）时，必须先查该行为是否违反既有
**验证规则的锚定假设**——MUST 提示与格式正则的 ^ 锚定是同一份设计里互相打架的两条
指令，代码先行样本从生成那一刻起就注定格式不合格、被结构性惩罚。结构化惩罚会把
行为在探索之前就灭绝（第二轮是截断、第三轮是锚定冲突），比"激励不够"更难发现——
两者的共同签名都是**精确的 0%/恒常数**。

---

## 8. 已知局限与后续

- 沙箱禁网是 best-effort（`-I -E` 隔离 + 无凭证），硬禁网需容器/seccomp——教学
  规模记录在案，不引入该复杂度（sandbox.py 文档字符串同步说明）；
- 每轮取"最后一个完整围栏块"执行，多块并发执行（并行工具调用）留待后续；
- 生成端多轮循环按组内 batch 调 vLLM（prefix caching 自动复用前几轮 KV），
  采样 seed 每轮相同（上下文不同→输出不同，可复现性成立）；
- 阶段3（Search-R1）可直接复用本阶段的多段轨迹协议与 mask 契约，只换
  `run_code` → `search_backend` 与 reward 的 EM 口径。
