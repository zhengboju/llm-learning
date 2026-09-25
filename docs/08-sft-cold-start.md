# SFT 冷启动方案（retool_math TIR 前置阶段）

> 状态：**备选方案（fallback）**，**已从主线降级**（2026-09-25 晚更正）。
>
> ## ⚠ 主线已改为「原生 `<tool_call>` 协议改造」→ 见 `09-native-tool-protocol.md`
>
> 本文档初版把"缺 SFT"当成主要结构性缺口，**这个判断过度概括了**。
> `agentic-rl-lab/05-retool` 用**同一个 `Qwen/Qwen3.5-4B` 基座 + 同一份
> DAPO-Math-17k**，**跳过 cold-start SFT 直接 RL**，拿到 AIME25 Average@12
> **+23.89pp**。它能跳过 SFT 的原因写在它的边界说明里：
>
> > **跳过 cold-start SFT**：对 Qwen3.5-4B 成立（base 工具调用率就够高），
> > 换基座不一定成立。
>
> 关键在于**工具协议**：参考用模型**原生 `<tool_call>`**（base 调用率 87.5%、
> 合法率 85.4%），本项目 p1–p11 用**自造围栏 + `[TOOL RESULT]` 文本**——
> base 无此格式先验，靠 prompt 硬教（docs/03 探针在 6144/concise 档实测
> 成功执行率 `code_ok`≈**0.484 条/轨迹**，即约 48% 的轨迹跑过代码）。
>
> **SFT 只对"自造协议"必需。** ReTool 论文自己造了 `<code></code>` 协议，
> 所以论文需要 SFT；参考实现用原生协议，所以不需要。**本项目选了需要 SFT 的
> 协议，却跳过了 SFT**——两边便宜都没占到。修协议比做 SFT 便宜得多且是
> 上游解，故本方案降为 fallback。
>
> ### 本文档仍然有效的部分
> - §0.4 / §5.2 **剂量缺口**（75 次更新、有效 batch 32）——与协议无关，仍然成立
> - §2 **轨迹合成**（若启用 fallback，合成与过滤逻辑照用）
> - §3.4 **工具段 mask 的数值后果**（原生协议下同样适用，且更重要）
> - §4 **验收门纪律**（四项门槛的形态可复用）
>
> ### 已作废的部分
> - "缺 SFT 是主要缺口"的**定性**（改为 fallback）
> - §3.2 ReTool lr `1e-6` 的**归属**（该值很可能属 RL 段，见 §3.2 存疑说明）
>
> ---
>
> 起因：p1–p11 共 11 版 RL 未能复现参考项目的增益。p11 内嵌评测 6 个存档 ×
> 2 split 全部形同虚设（json 层级错位 bug，commit `97a4915`），修复后对账发现
> 训练**从未稳定超过 BASE**（step100 与 BASE 打平、step200/300 劣化）。
>
> **本文档原先的判断：缺 SFT 冷启动是主要结构性缺口。**
> 该判断**已下调**（见上方横幅）：SFT 是**备选**，主因是工具协议。
> 本文档保留的论证价值在于 §0.4 的剂量缺口与 §2 的合成方案——
> 它们与协议选择无关，两项方案都要用。
> 另外**初稿的论证方式是错的**，见 §0.3/§0.5「初稿事实错误与更正」——初稿把
> "工具行为不在分布里"当作前提，而项目自己的数据（docs/03 撤回段、docs/p8）
> **证伪了这个前提**（代码调用率一直有 45~52%，从未灭绝）。
>
> **注意区分两个不同的"不在分布里"**：
> - ❌ 初稿的说法："代码行为不在分布里" —— **错**，模型会写代码
> - ✅ 正确的说法："**原生 `<tool_call>` 格式**不在分布里，模型只能靠 prompt
>   硬学自造围栏" —— **对**，这才是缺先验的地方，也是 §09 要修的

---

## 0. SFT 的价值与它的适用条件（重写版论证）

### 0.1 最硬的一条证据：ReTool 自身的消融

ReTool（ICLR 2026，[arXiv:2504.11536](https://arxiv.org/abs/2504.11536)）在
AIME2024 上的四组消融，把"冷启动"的价值单独隔离出来了：

| 配置 | AIME2024 | 说明 |
|---|---:|---|
| 原始 base | 26.7 | 未训练 |
| w/o CI（纯文本 RL） | 40.0 | 有冷启动、无代码工具，训 1080 步 |
| **w/o RL（仅冷启动 SFT + CI）** | **40.9** | **只做 SFT，不做 RL** |
| ReTool（冷启动 SFT + 工具 RL） | **67.0** | 400 步 |

读法很关键：**冷启动 SFT 单项就把 base 从 26.7 抬到 40.9（+14.2pp），
并且一步 RL 都没做就追平了纯文本 RL 的 1080 步结果。**

> **⚠ 但这组消融证明的是 SFT「有价值」，不是「必需」。**
> 论文的 base（Qwen2.5-32B-Instruct）面对的是它**自造的 `<code></code>` 协议**，
> 所以必须先教格式。换成一个模板里有原生工具先验的模型（Qwen3.5-4B 的
> `<tool_call>` 调用率 87.5%），SFT 就不是前置条件了——`agentic-rl-lab`
> 在同一基座上跳过 SFT 直接 RL，同样拿到 +23.89pp（见文档顶部横幅）。
>
> 因此本节的正确结论是：**SFT 的价值 = "把模型不认识的工具格式装进分布"，
> 它的必要性取决于协议是否原生。** 协议是原生时，这一项可以省。
> RL 阶段仍负责把它调优到 67.0——这一点两个实现一致。

ReTool 的其他关键配置：PPO，**KL 系数 0.0**，冷启动训 **2 epoch**；
论文另给"AdamW lr 1e-6 / max seq 16384 / mini-batch 512"一组值，
**但那组更像 RL 阶段配置**（见 §3.2 的 lr 存疑说明），本方案不直接照抄。

> 另可参考社区开放数据
> [`open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K)
> ——"SFT ColdStart for RL"已成为 agentic RL 的标准前置环节。

### 0.2 代码有真实价值，但训练的净效果是负的

**这一条初稿写对了，保留**（docs/05 §7.5.6，N=500 dev，per-item 同题配对）：

| 证据 | 数值 |
|---|---|
| BASE 用码题 acc | **74.1%** |
| BASE 纯推理题 acc | **47.4%** |
| 差值 | **+26.7pp**（工具路径有真实增益） |

但训练的**分层迁移**结果（同题双 acc 对账）是：

| 存档点 | 用码层 Δ（228 题） | 纯推理层 Δ（272 题） | 整体 |
|---|---:|---:|---:|
| step100 | −3.9pp | +11.6pp | +4.6pp |
| step200 | −10.9pp | +11.3pp | +1.2pp |
| step300 | **−12.7pp** | **+4.6pp** | **−3.4pp** |

即：**增益 100% 来自纯推理层，用码层相对 BASE 无增益**。训练把用码题
228→8 题，放弃代码的题 acc 从 74.1% 掉到 62.3%。净效果 −3.4pp =
丢代码（−12.7pp×228）远超涨推理（+4.6pp×272）。

docs/05 把这个现象判为 **H1「理性压灭」**：在当前的协议与提示下，写代码的
期望收益不高于纯推理，所以 RL 放弃它是**最优解**。这个判断是**方向性**的，
它不依赖"代码率是否为 0"——事实上恰恰相反，代码率一直很高（§0.3），
是"用了但不划算"。

### 0.3 初稿的事实错误更正（必读，否则会照错误的理由做 SFT）

**初稿§0.1 写了「4B base 自发写代码率 ≈0（13/13 纯 prose）」，并据此断言
"工具行为不在分布里"。这个数字被 docs/03 明确撤回，引用它是错的。**

docs/03 第 83–90 行的原始撤回文字：

> **【2026-09-17 更正·撤回上述"TIR 不必要"的旁证】**：那是在 **3072/13 条样本**
> 下观察到的，而 3072 恰恰是"prose 把第一轮烧穿、写不完围栏"的预算区间——
> **当时的 code_ok≈0 是截断的产物，不是模型的偏好**。6144 预算（concise 提示）
> 的 64 题 × k=8 探针实测 **code_ok = 0.484 条/轨迹**（`<=1` 次执行上限下 ≈48%
> 的轨迹成功跑过代码），提示层把它压到 0.441。**结论：TIR 路径在 4B 上是活的。**

同一处还记录：参考项目（agentic-rl-lab/05-retool，同 Qwen3.5-4B）的
**base 工具调用率起点就有 87.5%**。加上 docs/p8 §1.1 的定案：

> **代码从未被「压灭」**——各存档点 45.8~52.1%（除数修复后真实值），
> record 训练期实测 code 率 60~69% 独立佐证。

所以三条更正：

| 初稿说法 | 真实情况 | 出处 |
|---|---|---|
| 4B base 写代码率 ≈0 | **≈48% 的轨迹成功跑过代码**（6144 预算、code_ok 口径）；参考项目 base 工具调用率 **87.5%** | docs/03 撤回段 |
| code_rate 仍精确 0 | 代码**调用率** 45.8~52.1%（采样档，除数修复后真实值），从未灭绝 | docs/p8 §1.1 |
| 工具行为"不在分布里" | **在分布里，且被大量使用** | 同上 |

（两个数是不同口径，勿混：docs/03 的 48% 是**成功执行率**（code_ok/轨迹），
docs/p8 的 45.8~52.1% 是**调用率**（code_rate/题）。二者都在 ~50%，
结论一致：代码路径是活的。）

**这不推翻"SFT 有价值"的结论，但换掉了理由。** 正确的理由是 §0.1 与 §0.2：
SFT 是参考实现中把工具行为**装进模型**的独立阶段（+14.2pp），而本项目直接从
base 起 RL，于是 RL 只能在"已经会用工具、但用得不划算"的地形上做优化，
最终收敛到放弃工具（H1）。**SFT 的作用不是"教模型写代码"（它会写），
而是"用高质量示范把代码路径的期望收益抬到正号"**——这正是 docs/05 判定
H1 后无处下手的那一步。

### 0.4 剂量：本项目真正被验证的硬缺口

初稿这一节算错了 4 倍（见 §0.5）。正确的口径来自 `train.py:1029-1045` 的
自证注释，它专门警告过这个错觉：

> `all_steps` 计的是 **micro-batch 拉取次数**，不是 optimizer 更新数。
> 旧注释/文档把 300 说成 "300 optimizer steps"，实际只有 75 次，差 4 倍，
> 于是"再多跑就会涨"这类剂量判断一直建立在放大 4 倍的错觉上。

按 p11 实跑的 `all_steps=300 / num_pre_Q=8 / GAS=4`（**run_info 实读确认**）：

| 口径 | 数值 |
|---|---|
| 轨迹总数 | 300 × 8 = **2,400** |
| **optimizer 更新** | 300 / 4 = **75 次** |
| 有效 batch | 8 × 4 = **32 条/更新** |
| 见过的题数 | **300 题** = DAPO-Math-17k 的 **1.8%** |

与 ReTool 对照：**mini-batch 512 vs 32 = 16 倍**；ReTool 400 步 × 512
= 204,800 轨迹 vs 本项目 2,400 轨迹 = **85 倍**。

> **⚠ 一个会改变本节读法的口径问题（需在 pod 上核实）**：
> 上表"见过的题数 = 300 题（占 17k 的 1.8%）"假定训练池是完整 17k。
> 但 p11 开了 `difficulty_path` + `difficulty_band=(0.0,1.0)`，而
> `data.py:454-475` 的 `filter_qas_by_difficulty` 是**开区间**过滤，
> 且**表中缺失的题按丢弃计**（`missing`）：
>
> ```python
> row = table.get(str(x["Q"]))
> if row is None:
>     stats["missing"] += 1
>     continue          # ← 表里没有的题直接不进训练池
> ```
>
> 所以**真实训练池 = 难度表里落在开区间的题数**，可能远小于 17k。
> 若该表只覆盖几百到几千题，则"300 题"占比会显著高于 1.8%，
> **"见过的数据太少"这个论点会被削弱**（但 §5.2 的**有效 batch 32 /
> 75 次更新 / 与 ReTool 差 16 倍**这三条不受影响——它们与池子大小无关）。
>
> 核实：
> ```bash
> wc -l rlab_out/difficulty_probe.jsonl
> python -c "import json;rows=[json.loads(l) for l in open('rlab_out/difficulty_probe.jsonl')];import collections;c=collections.Counter('p0' if r['n_correct']==0 else 'p1' if r['n_correct']==r['k'] else 'band' for r in rows);print(len(rows),c)"
> ```
> 统计行里也能直接读到：`[probe] ... 保留 N 题` / `[rollout] ... 题目过滤中 N/M`。

### 0.5 初稿的事实错误（自查记录，累计八处）

本版核对推翻了前两稿的以下内容，全部已在本文档更正：

| # | 前稿 | 更正 | 错误类型 |
|---|---|---|---|
| 1 | 4B base 写代码率 ≈0（13/13） | ≈48%，且该数字已被 docs/03 撤回 | **引用了被撤回的数据** |
| 2 | code_rate 仍精确 0 | 45.8~52.1%，从未灭绝 | 同上 |
| 3 | 轨迹 300×4×8 = **9,600** | 300×8 = **2,400** | **正是 train.py 警告的 4 倍剂量错觉** |
| 4 | 见过 1200 题（7%） | **300 题**（占完整池 1.8%，但池子被难度表筛过，见 §0.4 警告） | 同上，连带错 |
| 5 | p11 现状 `round_gen_tokens=2048 / ctx=14336` | 实为 **6144 / 26400**（`run_info` 实读确认） | 把 preset 当成实跑档 |
| 6 | ReTool 冷启动 "2.1k 条" | 未能核实，已删 | 未验证的具体数字 |
| 7 | p11 是 outcome-only，待加 shaping | **p11 已有 `code_w=0.05` + `code_attempt_w=0.05` + `len_penalty_w=0.1`** | **虚构了一个不存在的缺口**（见 §0.6） |
| 8 | SFT lr 对齐 ReTool **1e-6** | 该值很可能属 **RL 段**（论文一句里混了两阶段超参），改按 **1e-5 起调** | **数字归属错误**（见 §3.2） |

**元教训（已入 §8）**：本文档的初稿本身就是"照直觉写数字"的又一个实例。
凡是**剂量、预算、协议档位、超参归属**四类数字，必须回源码/`run_info`/
论文原句结构核对，不能用记忆或"preset 应该是"顶替。这与本会话早前
`gpu_mem` 事故同型（根因恰在"以为不用记"的字段里）。

### 0.6 run_info 实读后的三条修正（2026-09-25，用户提供 p11 run_info）

拿到 p11 真实 `run_info.json` 后核对，**§5 的预算/剂量推断全部命中**
（`round_gen_tokens=6144`、`max_context_tokens=26400`、
`_tool_reserve=798`、`all_steps=300/save_steps=50` 恰好 6 个存档、
`num_pre_Q=8`、`gradient_accumulation_steps=4`）。但有两处结论要改：

**修正 A：p11 不是 outcome-only，它已经把 shaping 全开上了。**

| 参数 | 初稿以为 | run_info 实值 |
|---|---|---|
| `code_w` | 0.0（outcome-only） | **0.05** |
| `code_attempt_w` | 0.0 | **0.05** |
| `len_penalty_w` | 0.0 | **0.1** |

签名 `...-caw0.05-cw0.05-lp0.1-lq50-lg0.25-...` 逐项印证。

**这加强了而不是削弱了"缺 SFT"的判断。** 项目此前为救代码路径做过的
全部尝试——冷启动权重（`reward_cold_w`）、per-success shaping、尝试级
shaping（`code_attempt_w`，比 ReTool 官方更激进）、组相对长度惩罚
（`len_penalty_w=0.1`，MiMo Eq.4 形态）——**都已落地生效，仍未产生增益**。
说明缺的不是奖励塑形，而是**触发塑形对象的那个先决条件**（§0.1 的 SFT）。
这也解释了为什么"提高 shape 权重"类的后续建议会继续无效。

> **附带发现：`reward_cold_w` / `reward_hot_w` / `reward_switch_step` 在
> math 路径上是死开关。** p11 设了 `reward_switch_step=1000000000`
> （等于"永远 cold"，`cold_w=(1,2,2)` 即 w_code=2），但 `rollout.py:651`
> 的分派是：
>
> ```python
> if is_math:
>     sc = total_reward_retool_math(...)   # ← 没有 phase/cold_w/hot_w 参数
> else:
>     sc = total_reward_retool(..., phase=phase, cold_w=..., hot_w=...)
> ```
>
> `retool_math` 的 `data_task='dapo_math'` 恒走 math 分支，所以这三个键
> **完全不参与 reward 计算**。此前文档里"调大 cold code 权重"这类候选杠杆
> （docs/05 §7.5.5 的 H2 分支）**在 math 路径上不存在**——真正的活键只有
> `code_w` / `code_attempt_w` / `len_penalty_w`，而这三个 p11 都已开。
>
> ⇒ 下一步若还要在奖励层找杠杆，空间已经很小；这进一步把问题推向 §2/§3。

**修正 B：`attn_implementation=flash_attention_2` 已开 → §5.3 的 T² 告警作废。**

`run_info.config.attn_implementation = "flash_attention_2"`（还有
`vllm_gen_logps=true`、`vllm_logprobs_n=1`、`vllm_batch_invariant=true`、
`vllm_attention_backend="FLASH_ATTN"`）。FA2 下 head_dim=256 走 flash 核，
**T² 矩阵根本不物化**——初稿那张"最坏 41.54G"的表描述的路径在 p11 上不存在。
真实显存约束回到 logits 项，见 §5.3 更正。

**修正 C（协议核对：提示✅ 对齐，难度表⚠ 存疑）**

先说好消息：**p11 的 system prompt 就是定版的 concise 文件**。核对方式——
`rlab/prompts/retool_math_concise.txt` 经 `.strip()` 后 sha1 前 6 位 = **`3aac5d`**，
与 run_info 签名里的 `-sp3aac5d` 逐字相符（对照：preset 默认是 `72078f`，
直接读文件**原始字节**是 `f493f8`）。差异来源：`train.py:996-998` 与
`probe_difficulty.py:256-257` 都用 `f.read().strip()`，
**`-sp` 取的是 strip 后的内容**——直接 `sha1(open(p,'rb').read())` 会得到
不同的值，这一点很容易误判成"提示没对齐"。

存疑的是**难度表**：p11 的 `difficulty_path` 是 `rlab_out/difficulty_probe.jsonl`
（签名 `-t0cbda1`），而 docs/05 §6 **定版**的表是
`difficulty_probe_4b_v5_r6144_concise_full.jsonl`（`-tdf8d21`）。
`_difficulty_tag` 是**内容哈希**（`train.py:67-78`，sha1 前 6 位），
所以两个 tag 不同 = **文件内容不同**。

而 docs/03 §4 与 docs/05 都写明：**"难度表与模型×提示×预算绑定，换档后旧表作废"**，
v5 表正是为 6144 + concise 档重探的。p11 用的是另一份表 ——
需在 pod 上确认它是否也是 6144/concise 档探出的：

```bash
ls -l rlab_out/difficulty_probe.jsonl rlab_out/difficulty_probe_4b_v5_r6144_concise_full.jsonl
# 看表头 _meta / probe_meta 里的 rounds/round_tokens/sp/k（aggregate_rows 写入）
head -c 2000 rlab_out/difficulty_probe.jsonl
```

若它是早期档位（3072 或非 concise）探出的，则 p11 的**题池选择**与它的
提示/预算不同档——这是一个**独立的、尚未被任何诊断覆盖**的候选缺口。
注意 `difficulty_band=(0.0,1.0)` 是**开区间**（`filter_qas_by_difficulty`
只保留 `0 < n_correct/k < 1`），所以这份表**真的在筛选题池**，不是摆设。

---

## 1. 四阶段流水线全景

```
① 轨迹合成（few-shot 强制 + 拒绝采样）        ← §2
        ↓  2k~5k 条「用码 & 码跑通 & 答案对」轨迹
② SFT 冷启动（zero-shot prompt 训练）          ← §3
        ↓  ★验收门：code_rate>60% 且 fmt>90%（不过门不准进 ③）
③ RL（β=0，大批量）                            ← §5
        ↓
④ 固定引擎档评测（同 gpu_mem，Average@N）      ← §6
```

**★ 验收门是本方案的核心纪律。** 门不过就回 ①，绝不进 ③——因为 RL 阶段会
在"SFT 装好的行为"上做优化，SFT 装得不牢，RL 只会把它优化掉（§0.2 的 H1）。

---

## 2. 阶段①：轨迹合成

### 2.1 设计

复用 `rlab/probe_difficulty.py` 的基础设施（它已用
`multi_turn_rollout_group` + `total_reward_retool_math`，采样参数/预算/提示
与训练同口径），改三处：

| 改动 | 内容 | 理由 |
|---|---|---|
| prompt | 加 2–3 个 TIR few-shot 示范 | 把"用码且划算"的形态明确示范出来 |
| 落盘 | 除统计外，把**完整 segs（含 `ids`）**写盘 | SFT 要轨迹本体 |
| 过滤 | `code_used>0 且 code_ok>0 且 acc==1 且 trunc_final==0` | 四连全中才算示范 |

**注意与初稿的差别**：初稿的理由是"把不在分布里的行为诱导出来"；正确理由是
**"把已经在分布里、但期望收益为负的代码路径，示范成收益为正的形态"**。
这改变了 few-shot 示范的设计重点——不是"展示怎么写代码"，而是
**"展示什么样的代码调用真的省下了计算、并在截断前收尾"**（对齐 concise
提示"拿到答案就停"的方向）。

### 2.2 必须存 token ids，不能只存文本

`multi_turn_rollout_group` 返回的 `segs`：

```python
[{"kind": "assistant"|"tool", "text": str, "ids": list[int]}, ...]
```

**必须把 `ids` 一起落盘**（docs/02 发现1）：assistant 段以 `` ``` `` 结尾接
工具段 `\n` 开头时，整串 tokenize 会把 "```\n" 合并成单 token 13874，分段则
是 73594+198。若 SFT 从 text 重新 tokenize，序列与 RL 的
`retool_build_batch` 分段拼接序列不同，SFT 教的边界和 RL 用的边界差 ±2 token。

> **【已核实的现状】** `probe_difficulty.py:394-406` 现有一个 `--dump_samples`
> 落盘路径，但它**写的是 `text`，没有写 `ids`**：
>
> ```python
> fout_samp.write(json.dumps(
>     {"Q":…, "A":…, "acc":…, "trunc":…, "clen":…, "code_ok":…,
>      "n_segs": len(segs[idx]),
>      "segs_kind": [s["kind"] for s in segs[idx]],
>      "text": texts[idx]}, …))          # ← 逐段 ids 全部丢弃
> ```
>
> 且它是**配额制**（截断样本 dump 满 `--dump_samples`、正常样本只留 3 条，
> 额度收满即停写），设计目的是"截断率高时定位 token 去向"，**不是产出训练语料**。
>
> 所以 `synth_tir.py` 不能靠加一个 flag 复用——**落盘层要重写**：
> 逐样本写 `segs` 的全部 `{kind, text, ids}`，无配额，四连过滤在写入前完成。
> 这是阶段①真正的工作量所在（约半天），不是"派生个脚本"那么简单。

### 2.3 过滤判据

```python
keep = (st["code_used"] > 0          # 写了代码
        and st["code_ok"] > 0        # 代码跑通了
        and acc > 0                  # 答案对
        and st["trunc_final"] == 0)  # 末段没被轮长切断
```

第四项容易漏：截断轨迹没有完整 boxed 答案。本项目截断率长期高位
（p8 实测 42%、p10 窗口 corr(trunc,acc)=−0.90），这一项会筛掉相当大一部分。
**但注意**：截断率的根因是 `round_gen_tokens` 与提示层，属 §5 的独立变量，
不要与 SFT 混在一轮里改。

### 2.4 零标签字面量铁律

示范文本里不得出现格式标签/工具标记的硬编码字面量——本项目三次被
"标签经聊天管道被改写"坑到。示范串一律从 `rlab.protocol` 的 `TOOL_START` /
`TOOL_END` / `_PY_FENCE_RE` 派生构造，写完 roundtrip 回查
（喂 `reward_format_retool` 必须得 1.0）。

### 2.5 产出规模

按 docs/03 撤回段的实测（6144 预算、concise 提示）：单轨迹 code_ok ≈ 0.44–0.48、
完成题正确率 ≈ 82%、可学带（k=4 低估，n=8 修正后）≈ 60%。保守估算：

```
17k 题 × k=8 = 136k 轨迹
× P(用码 ~60%) × P(码跑通 ~75%) × P(答对 ~45%) × P(未截断 ~55%)
≈ 15k 条  →  取 2k~5k（去重后按题均衡）
```

算力约等于一次 p11 训练的**一半到一倍**。这是本方案里性价比最高的一步。

### 2.6 运行（go/no-go 闸门）

```bash
CUDA_VISIBLE_DEVICES=0 python -m rlab.synth_tir \
    --model_path /root/Qwen3.5-4B --k 8 --max_questions 200 \
    --fewshot 3 --round_gen_tokens 6144 --max_rounds 4 \
    --max_context_tokens 26400 \
    --chat_template_kwargs '{"enable_thinking": false}' \
    --out rlab_out/tir_synth_probe.jsonl
```

**第 ① 步就是 go/no-go 闸门**（半天）。判据不是"代码率是否 >20%"（初稿写错，
它本来就有 ~48%），而是：

| 判据 | 含义 |
|---|---|
| 四连过滤后**每题至少 1 条**可用轨迹的比例 | 决定 SFT 数据够不够 |
| 可用轨迹的 **`code_ok` 与 `acc` 联合率** | 决定"用码且对"的形态是否存在 |
| 若联合率极低（用码的都对不了） | → H1 在此基座/任务上成立，**SFT 也无解，诚实收尾** |

---

## 3. 阶段②：SFT

### 3.1 唯一需要新写的代码

`rlab/sft.py`。其余全部复用：

| 复用项 | 来源 | 说明 |
|---|---|---|
| 工具段 mask | `protocol.segment_mask_from_spans` | 与 RL 逐字同一函数 |
| 序列拼接 | `rollout.retool_build_batch` 同构 | assistant=1 / tool=0 / pad=0 |
| prompt 构造 | `rollout.build_prompt` | 训练时换回 **zero-shot** |
| 模型加载 | `model_loading.load_causal_lm` | 4B 复合 config 已收口 |
| 存盘 | `save_mm_checkpoint=True` 同路径 | vLLM 可直读 |

### 3.2 训练配方（对齐 ReTool）

| 参数 | 值 | 依据 |
|---|---|---|
| lr | **1e-5**（起点，按 4B 常规区间调） | ⚠ 见下方"lr 存疑"说明——**不要照抄 1e-6** |
| epoch | **2** | ReTool 冷启动 2 epoch（这一条来源明确） |
| max seq len | **8192** | 覆盖 ~99% 轨迹；见下方说明（ReTool 的 16384 是过度预算） |
| loss | **只对 assistant 段** | 工具段 mask 掉，与 RL 同原则 |
| dtype | bf16 | 与 RL/eval 同档 |
| zero_stage | 2（4B） | 4B 优化器全态放不下，docs/04 |

> **⚠ lr 存疑（初稿与第二版都写错过，这里如实标注）**
>
> ReTool 原论文的相关句子是：
> *"RL uses PPO … with a KL coefficient of 0.0. The cold-start model is trained
> for 2 epochs. Optimizer is AdamW with a starting learning rate of 1e-6, max
> sequence length of 16384 tokens, and mini-batch size of 512."*
>
> **这句话是有歧义的**：`1e-6 / 16384 / 512` 紧跟在"2 epochs"后面，但它们
> （尤其 mini-batch 512）明显是 **RL 阶段**的 PPO 配置。而 1e-6 对 SFT 来说
> **异常偏低**（4B 全参 SFT 常规区间是 1e-5 ~ 2e-5，1e-6 基本不动）。
>
> 因此：**`1e-6` 很可能属 RL 段，不是 SFT 的 lr。** 本文档不把它当作
> SFT 配方依据。落地时按 1e-5 起调，并用 §4 的验收门（`acc ≥ BASE`）判
> 是否过大/过小。
>
> **元教训**：论文的一句话里混了两个阶段的超参时，不能按语序就近认领。
> 这与 §8.2「preset ≠ 实跑」同源——都是"数字的归属搞错"。

> **max seq len 为什么用 8192 而不是 ReTool 的 16384**：p11 的
> `max_context_tokens=26400`，训练期实测 `avg_clen` 峰值 **5602**
> （+plen 1024 ≈ 6626）。16384 虽不截断，但对本项目轨迹长度是过度预算，
> 白白吃激活显存；8192 覆盖 ~99% 轨迹且留足余量。
> **落地前先统计 synth 语料的 `clen` 分位数再定档**，不要静默抄论文值。
>
> **关于"初稿写 lr 1e-5 / 3 epoch"**：初稿那两个数是凭直觉给的、无依据；
> 第二版改成"对齐 ReTool 1e-6 / 2 epoch"又犯了归属错误（见上）。
> 本版结论：**epoch=2 保留（来源明确），lr 按 1e-5 起调（不照抄 1e-6）**。

### 3.3 关键契约：prompt 必须换回 zero-shot

合成时用 few-shot 诱导，**SFT 时 prompt 必须换回训练/评测用的 zero-shot
系统提示**，只保留轨迹本体。否则学到的是"看到示范才写代码"，RL/eval 阶段
没有示范，行为立刻消失。

等价于：`system_prompt_sha` 在 SFT / RL / eval 三阶段必须一致。`run_info`
会自动记入签名，可事后核对。

### 3.4 工具段 mask 的数值后果（已实锤，勿省）

docs/02 记录的 A/B：工具 token 上 policy logp=−8 / ref=−2 / β=0.04 时——

- mask 正确：loss=0，工具位梯度**严格** 0
- mask 错误：loss≈3.4，工具位梯度非 0（假信号）

SFT 阶段同理且更危险：**对沙箱输出求 loss 等于教模型预测沙箱的输出**，
这是模型不可控的分布，纯噪声。`sft.py` 必须复用同一 mask 函数，并配
**梯度探针测试**（只测 forward 数值不够——cispo 零梯度 bug 的教训）。

---

## 4. ★ 验收门（不过门不准进 RL）

SFT 完成后，用**与 RL 完全同档**的协议评一次：

```bash
CUDA_VISIBLE_DEVICES=0 python eval_vllm_one.py \
    --model rlab_out/sft_tir/final --proto_from rlab_out/sft_tir/final \
    --algo retool_math --split test --n 300 --seed 42 \
    --gpu_mem 0.78 --out rlab_out/eval_sft.json --name sft_tir
```

| 指标 | 门槛 | 依据 | 不过怎么办 |
|---|---|---|---|
| `code_used>0` 的题占比 | **> 60%** | 本项目 base 已 ~48%（§0.3），SFT 应显著抬升；ReTool 训练后该指标达近 98%，SFT 阶段取其中途值作门槛 | 回 ①：加数据 / 查 few-shot 是否把"划算"示范清楚 |
| `fmt`（有 boxed） | **> 90%** | 保证"答完就停" | 回 ①：查是否漏了 `trunc_final==0` 过滤 |
| `code_ok / code_used` | **> 70%** | 代码质量 | 提高合成时的 `code_ok` 门槛 |
| `acc` | **≥ BASE** | 冷启动不应伤基础能力（ReTool：SFT 后 40.9 远高于 base 26.7） | 降 lr / 减 epoch |

**四项全过才进 ③。** 门槛是**工程判断**，不是论文规定值——落地后按首批
实测校准，但**校准必须在跑 ③ 之前完成**，不能"边跑边放宽"。

---

## 5. 阶段③：RL 配置修正

### 5.1 起点与算法项（对齐 ReTool）

**以下 p11 值全部来自 `run_info.json` 实读**（不再是推定，见 §5.3 更正记录）：

| 参数 | p11 实跑 | 改为 | 理由 |
|---|---|---|---|
| 起点 | 4B base | **SFT ckpt** | §0.1：ReTool 消融的核心变量 |
| `beta` | **0.04** | **0.0** | ReTool **KL 系数 0.0**。KL 锚在"用码不划算"的分布上，对抗 SFT 成果 |
| `loss_norm` | **`sample_mean`** | **`token_mean`** | 多轮长轨迹标准配置（DAPO）；sample_mean 让 6144 token 与 400 token 轨迹等权 |
| `round_gen_tokens` | **6144** ✅实读 | 维持 | 已与 ReTool 量级一致，且预算余量仅 +2（§5.3） |
| `max_context_tokens` | **26400** ✅实读 | 维持 | 同上 |
| `code_attempt_w` | **0.05** | **维持**（~~0.0→0.05~~ 是初稿错误） | **p11 已经开着**——见 §0.6 |
| `code_w` | **0.05** | **维持** | **p11 已经开着**（ReTool per-success 口径） |
| `len_penalty_w` | **0.1** | 维持 | **p11 已经开着**（p10 建议项的落地） |

**★ 这张表最重要的一行是"维持"那三行。** 初稿以为 p11 是 outcome-only
（`code_w=0`、`code_attempt_w=0`），于是把"加 shaping"当成了待做的修复。
**事实是 p11 已经把 ReTool 官方的 per-success shaping（`code_w=0.05`）和
更激进的尝试级 shaping（`code_attempt_w=0.05`）都开上了，仍然没有增益。**
这直接加强了本方案的核心判断，见 §0.6。

### 5.2 剂量：本项目最被低估的缺口

`all_steps` 是 micro-step，不是 optimizer 更新数（`train.py:1029`）：

| | 轨迹总数 | optimizer 更新 | 有效 batch | 见过题数 |
|---|---:|---:|---:|---:|
| **p11 实跑** | 300×8 = **2,400** | 300/4 = **75** | 8×4 = **32** | 300（1.8%） |
| ReTool RL | 400×512 ≈ **204,800** | 400 | **512** | — |
| **差距** | **85×** | — | **16×** | — |

> **已核实（2026-09-25，用户提供 p11 `run_info.json`）**：`all_steps=300`、
> `save_steps=50`（→ 6 个存档）、`num_pre_Q=8`、`gradient_accumulation_steps=4`、
> `train_micro_batch_size_per_gpu=8` —— **下表全部为实读值，不再是推定**。

**这是全项目唯一被源码自证、且此前被系统性高估 4 倍的缺口。**
初稿在这里又错了一次（写 9,600），正好复现了 `train.py` 专门警告的那个错觉。

**可立即执行的修正**：抬 `gradient_accumulation_steps`（4 → 16/32）不增加显存
（micro-batch 契约仍是 `num_pre_Q=8` 行，由 `config.py` 强制），只增加每次
更新前的生成量、并把 optimizer 更新数相应压低但**每次更新更稳**。注意二者
是权衡：抬 GAS 会让同 `all_steps` 下的更新次数更少（300/32 ≈ 9 次），
所以 **`all_steps` 必须同比放大**，否则是反向操作。正确形态是
"总轨迹数放大 + GAS 放大"，让 有效batch 和 更新次数 同时上升——这需要更多
墙钟时间，是**算力换质量**，没有免费午餐。

### 5.3 预算自洽与显存（run_info 实读核对）

初稿在此处写了 `max_context_tokens ≥ 20480`，是**错的且有害**。实跑
`config.validate_retool_budget` 与 run_info 实读值完全吻合：

```
rounds=4  per_round=6144  max_prompt_length=1024
reserve = 3 × (500//2 + 16) = 798          # ← run_info 里 _tool_reserve: 798 ✓
need    = 4×6144 + 1024 + 798 = 26398  ≤ 26400 ✅ 余量仅 +2
```

**余量只有 2 token**——这是"极限预算、无反向压力"（docs/p8 §2.2 已记录）。
任何进一步的预算调整都要重算。注意这是**自洽的上限**：合法轨迹能写满
4×6144，把 T 顶到 26398，正好卡在丢弃线下方。

#### 显存：p11 已开 FA2，T² 项不存在

初稿曾按 SDPA math 回退估算出"最坏 41.54G 的 T² 瞬态"。**run_info 实读
`attn_implementation="flash_attention_2"`，该路径在 p11 上不成立** —— FA2
支持 head_dim=256，T² 矩阵不物化，此项为 0。保留换算表仅供"若换回 SDPA"参考：

| 档位 | T | T² 瞬态（B=1，仅 SDPA math 回退时） |
|---|---|---|
| docs/04 B4 锚点（实测，SDPA 时代） | 6,503 | 2.52 G |
| p11 训练期 clen 均值 | ~6,626 | 2.62 G |
| 上限用满 | 26,398 | 41.54 G ← **p11 不在此路径** |

**FA2 档下真正的显存项是 logits**（`B×T×V×dtype`，V=248320）：

| 项的算法 | 峰值 |
|---|---|
| 整批 8 行 × T=6626 × V=248320 × 2B | 24.5 GiB |
| **按 `seq_chunk` 分块（实现口径）** | 见下 |

实现走 `forward_per_token_logps(seq_chunk=512)` 分块（logps 按 token 独立），
故实际项是 `B × seq_chunk × V × 2B × (log_softmax 翻倍)`：

```
8 × 512 × 248320 × 2B = 1.89 GiB  →  log_softmax 翻倍 ≈ 3.79 GiB
```

（与 docs/04 §logits 行"seq_chunk 分块"的治理一致。）

**结论：预算/显存两项都不是 p11 的瓶颈。** 不要把"抬预算/换 backend"当成
下一步——它们已经做完了。下一步是 §2/§3 的 SFT。

**纪律提醒**：本小节是"初稿凭直觉写预算、被 `validate_retool_budget` 当场
抓住"的实例。预算类数字一律实跑该函数，这与 2026-09-12 的
`3×3072 > 8192` 事故（丢弃率 90%、训练端 5 小时零产出）是同型错误。

---

## 6. 阶段④：评测纪律

一句话：**`gpu_mem` / `batch_invariant` / 协议来源三件套必须对齐**，否则
Δacc 无意义。2026-09-25 真机 A/B 定案：单变量只改 `gpu_mem`（0.20 vs 0.78）
在同权重同题同 seed 下差 **+7.0pp**（52 题翻转，McNemar p=0.0175）。

- 看趋势 → 全用内嵌读数（同档跨 step 可比）
- 报绝对值 → 整条曲线用同一 `gpu_mem`（空闲机 0.78）重评
- 两套混用 → 凭空多出 7pp 假信号

**另一条本次事故的教训**：内嵌评测的 json 是 `{name: result}` 嵌套壳，
消费方必须用 `analysis.read_eval_result()` 读取。p11 的训练日志印了 6 个存档
× 2 split 的 `acc=0.0% (n=0)`，`n=0` 就是"读了不存在的层级、拿到默认值"的签名。

---

## 7. 实施顺序（按信息量/成本排序）

| 步骤 | 工作量 | 产出 | 判据 |
|---|---|---|---|
| **① 小样本合成探针** | 半天 | 200 题 × k=8 | 四连过滤后**每题覆盖率**；"用码且对"联合率 |
| ② 全量合成 | 1 天 | 2k–5k 轨迹 | 条数够 + 题面均衡 |
| ③ 写 `rlab/sft.py` + CPU 测试 | 1 天 | SFT 代码 | **工具位梯度探针 = 0** |
| ④ SFT 训练 | 半天 | SFT ckpt | lr 1e-6 / 2 epoch |
| ⑤ **验收门评测** | 2 小时 | 四项指标 | **全过才继续** |
| ⑥ RL（配置按 §5） | 1–2 天 | p12 | vs SFT ckpt 有增益 |

**第 ① 步是全流程的 go/no-go 闸门**，它能在半天内回答"这个任务在这个基座上
到底有没有戏"这个已消耗 11 版的问题。

---

## 8. 元教训

### 8.1 四个变量同时动，故障归因不可能收敛

阶段 1 在 GSM8K + 3B 上拿到过干净增益，证明 **RL 循环本身是正确的**。
从阶段 1 到 p11，同时换了四个变量：

| 维度 | 阶段 1 | p11 |
|---|---|---|
| 模型 | Qwen2.5-3B | Qwen3.5-4B |
| 数据 | GSM8K | DAPO-Math-17k（竞赛级） |
| 轨迹结构 | 单轮 | 多轮（4 轮 × 6144） |
| 工具 | 无 | 代码沙箱 |

每次诊断都能找到一个真 bug（而且都修对了），但修完仍无效——因为**主缺口
（缺 SFT 冷启动 + 剂量差 85 倍）从第 1 版起就一直在**，且不属于任何单次
诊断的视野。单变量纪律不只适用于超参，**同样适用于实验阶段的推进**。

### 8.2 "preset 值" ≠ "实跑值"

初稿把 preset 的 `2048/14336` 当成 p11 现状，而 p11 实跑是 `6144/26400`。
CLI 覆盖 preset 是本项目的常态（`--trunc_shaping`、`--round_gen_tokens`、
`--difficulty_path` 都改过），**任何"现状"判断都必须读 `run_info.json`**，
不能读 `config.py`。这正是 `run_info` 与 `run_signature` 存在的意义。

### 8.3 引用的数字必须查证它是否已被撤回

初稿引用的"13/13 纯 prose"躺在 docs/03 里，紧邻它的下一段就是撤回声明。
**本项目文档的形态是"后续段落推翻前面段落"，引用时必须读到该条目的结尾**，
不能只读匹配到的第一行。

---

## 附：待建文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `rlab/synth_tir.py` | 待写 | 阶段①；**落盘层需重写**（现有 `--dump_samples` 不存 `ids` 且有配额，见 §2.2），其余复用 `probe_difficulty.py` |
| `rlab/sft.py` | 待写 | 阶段②，唯一真正的新代码 |
| `rlab/tests/test_sft_cpu.py` | 待写 | 工具位梯度=0 探针 + segs↔ids 序列契约 |
| `config.py` 的 `sft` preset | 待加 | lr 1e-5（起调）/ epoch 2 / max_seq 8192 / zero_stage 2 |

## 附：外部参考

- ReTool 论文：[arXiv:2504.11536](https://arxiv.org/abs/2504.11536) ·
  [ICLR 2026 OpenReview](https://openreview.net/pdf?id=tRk1nofSmz)
- 冷启动数据（社区）：
  [`OpenThoughts-Agent-SFT-ColdStartForRL-10K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K)
- 参考实现：`agentic-rl-lab/05-retool`（本项目 docs/05 有逐维对照）
