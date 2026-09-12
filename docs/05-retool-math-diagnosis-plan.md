# retool_math（Qwen3.5-4B）run2 诊断与优化方案

> 状态：**诊断已定案，方案待执行**（2026-09-12 整理）
> 依据：run2 训练日志 `rlab/rlab_out/train_log2.txt`、得分记录 `rlab/rlab_out/record2.jsonl`、
> run2 评测 `eval_vllm_all.json`，以及参考项目
> [agentic-rl-lab/05-retool](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool)。
> 配套：迁移与协议层 checklist 见 `03-qwen35-4b-retool-math-checklist.md`；显存见 `04-qwen35-4b-gpu-memory-oom.md`。
> **进度看板见 §7**（`[x]` 已落地 / `[ ]` 待执行）。

---

## 0. 结论摘要

1. **没有净增益。** run2（1000 步）跑完 719 步的评测显示：m200 = BASE +5.0pp（**不显著**），
   m600 = BASE −0.2pp。**训练后期的退化被独立评测确证**，不是记录曲线的取样假象。
2. **但这个"没学到"不是必然。** 参考项目在**同一基座、同一份 DAPO 数据、同样纯 outcome ±1**
   下把工具调用率从 1.0 推到 2.0 并稳定，Average@12 拿到 **+23.89pp（OOD）**。
3. **主嫌疑是 `trunc_shaping=0.5`**：我们与参考唯一的奖励差异，且它是**唯一与轮数正相关的惩罚**。
   参考专门讨论过 shaping 并**选择不加**，我们加了，方向还装反了（§4.1）。
4. **次嫌疑是协议结构**：`max_rounds=2` 只允许 **1 次**代码执行，而参考实测平均 **2.98 轮**——
   参考报告"学到"的那个行为，在我们的协议里物理上不存在（§4.2）。
5. **测量侧缺口已补齐**（§9）：per-item 落盘、N-aware CI + McNemar、审计字段、run 偏离签名。
   已用 run2 真实数字验证：m200 的 +5.0pp 从"真差异"改判为"噪声内"。

---

## 1. run2 训练侧事实（健康，但方向在退）

| 项 | 值 |
|---|---|
| 算法 / 基座 | `retool_math` / `/root/Qwen3.5-4B-text`（抽取的纯文本） |
| 配置 | `all_steps=1000`、`save_steps=200`、`num_pre_Q=8`、`gen_questions_per_attempt=4`、`lr=5e-6` |
| 长度/奖励 | `trunc_shaping=0.5`、`overlong_shaping=True`、`round_gen_tokens=3072`、`max_rounds=2`、`code_w=0.0`、`reward_switch_step=1e9` |
| 预算 | `max_prompt_length=1024`、`max_context_tokens=8192` |
| 数据 | `dapo_math` + `difficulty_probe_4b_v4.jsonl`、band `(0,1)` |
| 时间 | 15:14:11 → 21:26:17（6.20h），**719/1000 步**，~30 s/it，ETA ~1h55m |
| 偏离签名 | `retool_math-ts0.5-ol1-r2x3072-s1000x200-lr5e-06-d0-1` |

### 1.1 工程与数值：无异常

- 无 error / OOM / NaN；44 次权重推送**全部** `loaded 330/426`（`sync.py:18` 明确：stacked 融合
  如 q/k/v→qkv_proj 使 `loaded < sent` 是常态，计数恒定 = 没丢权重）。
- 采样：累计尝试 1516 / 有效上传 1264 组，丢弃 **14%**（零方差 212，**超长 0** → 预算自洽，
  `overlong_shaping` 的 trigger 真可达）；题目拉黑仅 69/20792。
- Loss 中位数 **~0.002**，step 240 后区间 [-0.011, +0.014]；两次孤立尖峰（0.95@step4、**8.29@step225**），
  之后再未复现。
- clip_frac 0.0165(320-490) → **0.0282 峰**(500-610) → 0.0095(620-710)，max 0.0808@step30；
  KL 1.5e-3、approx_kl 2.8e-3、`mean_ratio ∈ [0.9975, 1.0011]`、frac|d|>0.1 ≈ 5%；staleness 4–20 micro-step。
- **没有发散、没有梯度爆炸、没有信号恒死**（`fmt_const`/`acc_const`/`length_runaway` 均未触发）。
- lr 5e-6 下单步几乎不动，但 1000 步攒出了真实行为改变 —— 像是"每步太弱 + 总量太长"的组合（§4.3）。

### 1.2 质量曲线：acc 在 group≈386 见顶后单调回落

按 200 组窗口（每组 8 样本，≈1600 样本/行）汇总：

| groups | gen_ver | acc 率 | fmt 率 | code 率 | code_ok 率 | trunc 率 | avg_clen |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–200 | 0 | 38.5% | 51.6% | 56.9% | 45.1% | 33.8% | 2891 |
| 200–400 | 96 | **54.8%** | 65.1% | 62.5% | 51.0% | 31.9% | 2875 |
| 400–600 | 208 | 50.9% | 70.0% | 65.4% | 52.5% | 30.1% | 2613 |
| 600–800 | 320 | 49.4% | 75.3% | **66.6%** | **51.1%** | 24.8% | 2626 |
| 800–1000 | 432 | 42.5% | 71.3% | 59.6% | 38.8% | 28.2% | 2814 |
| 1000–1200 | 544 | 38.1% | 71.1% | 48.8% | 25.4% | 28.8% | 2874 |
| 1200–1263 | 672 | **38.1%** | **81.7%** | 53.2% | **23.2%** | **18.8%** | 2805 |

health 口径（32 组窗）：开局 **55.5%** → 峰值 **64.1%**（group 386 结束，gen_ver≈192）→ 末段 **42.2%**。
*（注：开局区间波动大，前 200 组的池化均值 38.5% 与首 32 组的 55.5% 都是真实取样差异，不是口径冲突。）*

**两个反向信号同时出现**：`fmt` 率一路涨到 81.7%（格式越写越规范），`code_ok` 率从峰值 52.5% 崩到
**23.2%**（代码越来越不能跑）。即"结构写对了，但答案错了、代码也跑不通了"。

### 1.3 健康检查告警（重放确认）

`health.py` 的 `maybe_check` 用 `self.fired` 去重，**同一告警只打印一次**——所以日志后期安静
**不等于**已恢复。我重放整个 record 得到：

| 触发位置 | 告警 | 内容 |
|---|---|---|
| group 32 | `retool_trunc` | 最近 32 组 >20% 末段被轮长上限切断，建议调大 `round_gen_tokens`/`max_rounds` |
| group 256 | `flat` | 256 组后 acc 净提升 <1pp（0.11→0.12）→ **没有学习签名** |
| group 288 | `decline` | acc 较开局下滑 >5pp（0.11→−0.08）→ **退化签名，建议早停** |

**`decline` 至今仍然 active**（末 32 组 acc ±1 口径 −0.156 vs 开局 +0.109）。

---

## 2. run2 评测侧事实

协议：`dapo_math dev`（held-out 500，**与训练同分布**）、N=500、seed=42、**greedy**、`--algo retool_math`、
`--base_path /root/Qwen3.5-4B`、`n_dropped_long=0`。checkpoint 已经过 `materialize_mm_ckpt` 物化。

| 模型 | acc | fmt | code_rate | code_ok_rate |
|---|---:|---:|---:|---:|
| BASE | 46.6% | 53.0% | 42.2% | 38.0% |
| **m200** | **51.6%** | 60.2% | **62.6%** | **53.0%** |
| m600 | 46.4% | 67.8% | **13.0%** | **9.6%** |

### 2.1 判定（§9 修正口径之后）

| 对比 | 差值 | 95% CI | p | 判定 |
|---|---|---|---|---|
| m200 − BASE (acc) | +5.0pp | ±6.2pp（[-1.2, +11.2]） | ≈0.11 | **噪声内** |
| m600 − BASE (acc) | −0.2pp | ±6.2pp | 0.95 | 噪声内 |
| m200 − m600 (acc) | +5.2pp | ±6.2pp | ≈0.10 | 噪声内 |
| m600 − BASE (fmt) | **+14.8pp** | ±6.0pp | <0.001 | **显著** |
| m600 − BASE (code_rate) | **−29.2pp** | ±5.2pp | <0.001 | **显著** |

**唯一能确证的：格式达标率真的涨了，工具调用率真的灭绝了。** 准确率的 +5.0pp 方向可信、幅度不可信。

### 2.2 两个口径事实（写进结论前必须知道）

- **`both ≡ acc` 是结构性恒等**：`acc=1` 的前提是 `_extract_last_boxed` 取到 boxed，而 `fmt=1`
  正是"取到 boxed"。所以"双达标"列不提供额外信息，`fmt` 在 boxed 口径下只测"有没有答案框"、不测对错。
- **同题配对检验本可提功效，但 run2 的数据永久补不回来**：三模型评的是同一批 500 题
  （同 seed/split），McNemar 功效远高于未配对两比例；而旧版 `eval_vllm_one.py` 只落盘聚合计数。
  §9 已把它改成默认落盘 per-item。

---

## 3. 诊断：三条独立证据指向同一结论

| # | 证据 | 来源 | 说明 |
|---|---|---|---|
| 1 | `decline` 签名从 group 288 起持续到训练结束 | 训练 record（health 重放） | 训练期可观测，独立于评测 |
| 2 | m600 ≈ BASE（−0.2pp） | held-out 评测 | 1000 步的**净收益 ≈ 0** |
| 3 | `code_rate` 62.6% → 13.0%（z≈−10.9） | held-out 评测 | 行为反转，不是噪声 |

外加**外部参照**：参考项目在同等条件下 OOD +23.89pp、工具使用不灭（§5）。所以这不是"4B 学不动"，
而是**我们的配方在某处把它推反了**。

---

## 4. 根因排序

### 4.1 主嫌疑：`trunc_shaping=0.5`（方向装反）

**结构事实（代码级）**：`rlab/reward.py:270` 明确 retool_math 是 outcome-only ——
"工具使用完全靠结果涌现，**不额外奖励 code_ok**"；preset `code_w=0.0`，
`total_reward_retool_math` 的签名里**根本没有 code / phase 项**。所以 reward 对"要不要写代码"
**零信号**，工具使用只受间接力支配。

**我们唯一的奖励偏离**就是这个 shaping，而它是**唯一与轮数正相关的惩罚**：写代码要占掉
`max_rounds=2` 里的一轮、显著抬高撞 `round_gen_tokens` 的概率 → 扣 0.5；纯 prose 是单发路径、
撞上限风险低。**唯一作用在工具使用上的力是负的。**

**参考项目的处置恰好相反**（`05-retool/readme.md`）：官方 released 代码里有
`score = min(0, score + (num_turns - 2) / 2 * 0.1)`（用在答错分支）—— 这是**保工具**的 shaping
（错但多调工具 → 只扣 0.8），而参考判断它与论文 outcome-only 说法冲突，**选择跟随论文、不加任何 shaping**。
两个方向的证据合起来指向同一结论：**我们加错了项。**

> 注意：这是**强假设而非已证事实**。相关性（截断率↓ + 长度↓ + 代码↓）方向未识别，
> 需要 P1 消融判决。参考只证明了"无 shaping 下工具不灭"，没有直接证明"我们的 shaping 灭了工具"。

### 4.2 次嫌疑：协议只允许 1 次代码执行

| | 参考 | run2 |
|---|---|---|
| 轮次 | `max_assistant_turns=6` / `max_code_calls=4` | **`max_rounds=2`（≤1 次执行）** |
| 实测 | 平均 **2.98 轮**、~2 次调用 | 至多 1 次 |
| 单轮预算 | 1024 | 3072 |
| 工具协议 | **原生 `<tool_call>`**（chat template `tools` 声明） | python 代码围栏检测 + 文本回填 |
| 系统提示 | 原生工具说明 | MAY 口径（"when … helps"） |

参考报告"学到"的行为是**调用 → 看结果 → 再调用 → 作答**的迭代式工具使用——**在我们的协议里不存在**。

当初砍轮数的依据是 `03-...-checklist.md` 的"4B 无代码即终局、`max_rounds` 是虚假预算、TIR 对 4B
可能不必要"。**这条前提被参考项目直接证伪**：Qwen3.5-4B base 的 smoke test 工具调用率
**87.5%（14/16）**、valid 85.4%、eval 下 1.69 calls/轨迹；而 run2 的 BASE 只有 42.2% 轨迹用过代码。
最可能的根因是**协议不同**：原生 `<tool_call>` 下模型一决定调用就停，单轮 1024 够用；围栏方案要求
模型一路生成到写出围栏才停，单轮预算被迫抬到 3072 才能容纳"思考+围栏"。
**"4B 不需要 TIR"很可能是围栏协议的伪影，不是模型属性。**

### 4.3 第三嫌疑：训练预算（步数 / lr / 存盘）

- **步数**：参考 200 步收工且 Average@12 全程单调；run2 的 m200 是**第一个也是最好的** checkpoint，
  之后退化。**多跑的 800 步是负收益。**
- **存盘**：`save_steps=200` 把 0–200 变成盲区；参考每 **50** 步存盘。
- **优化器口径不可直接搬**：参考 LoRA r32/lr 4e-5，run2 全参/lr 5e-6。run2 的
  `kl≈1.5e-3、clip<3%、mean_ratio≈1.0000` 说明单步几乎不动、1000 步才攒出一次行为反转。
  **但在 P1/P2 定性前不要动 lr/LoRA**，否则又是复合变量。

### 4.4 已排除 / 未解释

- **排除**：预算不自洽（超长丢弃 0）、权重同步失败（330/426 恒定）、NaN/发散、数据未过滤、
  MM 物化缺失（评测前已做）。
- **未解释**：退化率。参考 `degenerate_group_rate ≈ 0`（20 步跑为 0，200 步"缓降"），
  run2 零方差丢弃 **14–16%（212/1516）**。同 group=8 定义下差这么多，且参考做的是**全量 17k 不过滤**、
  run2 做了难度带过滤（本意是降退化）——见 §10。

---

## 5. 与参考项目 05-retool 的对照

| 维度 | 参考 05-retool | run2 |
|---|---|---|
| 基座 | `Qwen/Qwen3.5-4B` 原生多模态 | `/root/Qwen3.5-4B-text`（抽取纯文本） |
| 训练 | **LoRA r32，lr 4e-5** | **全参，lr 5e-6** |
| 步数 / 存盘 | **200** / 每 **50** | **1000** / 每 **200** |
| 组大小 | 8 | 8 |
| 奖励 | 纯 outcome ±1，**明确不加任何 shaping** | outcome ±1 **+ `trunc_shaping=0.5` + `overlong_shaping`** |
| 轮次 | 6 turns / 4 calls | **2 rounds（≤1 call）** |
| 单轮预算 | **1024** | **3072** |
| 工具协议 | 原生 `<tool_call>` | python 代码围栏 |
| 训练数据 | DAPO-Math-17k 全量 | 同数据 + 难度带过滤 |
| 评测 | **AIME25（OOD）30 题×12 采样**，temp1.0/top_p0.7 | **dapo_math dev（同分布）500 题**，greedy |
| 结果 | Average@12 **23.61% → 47.50%（+23.89pp）**，Format 25.83→76.11，code_calls 1.69→1.98（step90 峰 2.6），turns 2.69→2.98 | acc 46.6 → 51.6（m200，不显著）→ 46.4（m600）；code_rate 42.2→62.6→**13.0** |
| 自我限定 | "30 题×12 采样、单训练单种子，不足以宣称稳定复现"；8k vs 论文 16k 是最大妥协 | N=500 更扎实，但**没有 OOD 集** |

**结论差异的实质**：
1. **工具使用走向相反**（参考单调强化并稳定 ↔ run2 先强化后灭绝）。
2. **评测强弱不可比**：参考是 OOD、run2 是同分布。同分布只涨 5pp 且不显著，**比参考弱了不止一个量级**
   —— 所以"run2 没学到东西"这个判断跟参考一比只会更刺眼。
3. **"模型学会的顺序"没复现**：参考"先调规范 → 再写能跑的代码 → 最后写对解题有用的代码"；
   run2 第一阶段方向一致（code_ok 38→53）后整个链条反转（→9.6%）。

---

## 6. 优化方案

> **核心纪律：一次只改一个变量。** 当前的困境正是 4 个维度同时偏离的结果——再复合改动就学不到东西。

### P0 · 测量（已落地，§9）

`[x]` per-item 落盘 ｜ `[x]` N-aware CI + McNemar ｜ `[x]` `model_path`/`_meta` 审计 ｜
`[ ]` OOD 评测集（AIME25，见 §6.5）

### P1 · 单变量消融 `trunc_shaping`（**如果只做一件事，就做这个**）

```bash
bash rlab/run_gsm8k.sh retool_math /root/Qwen3.5-4B-text \
  --seed 42 --steps 300 --save_steps 50 --trunc_shaping 0.0
# 签名 = retool_math-ts0-ol1-r2x3072-s300x50-lr5e-06-d0-1
```

**除 `trunc_shaping` 外一切与 run2 一致**（含 `max_rounds=2`），确保单变量。
步数 300 是因为参考 200 步收工、存盘 50 是为了不再出现 0–200 盲区。

**事前写死判据（防事后解释）**：

| 观察 | 判定 | 下一步 |
|---|---|---|
| 晚段 code_rate ≥40% 且 acc ≥ BASE | shaping 是主因 | 进 P2 |
| code_rate 仍 <20% | shaping 不是主因 | 跳过调参，直接进 P2（协议是主因） |
| code_rate 回升但 acc 不涨 | 工具不是瓶颈 | 转向数据/难度带（P4） |

成本：300 步 ≈ 2.6h（按 run2 实测 ~31 s/step）+ 评测。

### P2 · 协议与轮次（结构性，**必须与 P1 分开跑**）

`max_rounds=2` 在物理上只允许 1 次代码执行（`config.retool_math` 注释：round1 写代码 → round2 出 final）。
参考报告的行为需要多轮。预算由 `config.validate_retool_budget()` fail-fast 强制：

`rounds × per_round + max_prompt_length(1024) + 工具段预留(266) ≤ max_context_tokens(8192)`

| rounds | 单轮上限 |
|---:|---:|
| 2（run2） | 3451（用 3072） |
| 3 | 2300 |
| 4 | 1725 |
| **6（≈参考）** | **1150** |

**所以"6 轮"必然要求单轮降到 ~1024，这两件事耦合、不能只改一个。**
`03-...-checklist.md` 的"1024 截断 85%、必须 3072"是**围栏协议**下的结论；参考用原生 `<tool_call>`
时 1024/轮完全够用。**P2 的正解是换协议，而不是继续把单轮预算往上加**（后者只是治标，且会继续挤压轮数）。

```bash
# 目标签名 = retool_math-ts0-ol1-r6x1024-s300x50-lr5e-06-d0-1
```

备选（若协议改造太重）：`max_context_tokens 8192 → 16384`（论文设置）。参考自己说"预算翻倍是最值得
先试的改进"；但 2×H20 跑 4B 全参 + `04-...` 已有 OOM 记录，**先算显存再动**。

工程工作量：1–2 天（原生 tool 协议要动 chat template `tools` 声明 + 轨迹构造）。

### P3 · 训练预算

`[ ]` 步数上限 200–300（不再跑 1000）；`[ ]` `--save_steps 50`；`[ ]` 在 P1/P2 定性后再单独动 lr / LoRA。

### P4 · 数据侧

`[ ]` 无 `--difficulty_path` 的对照跑（回答"静态过滤是否真在降退化"，见 §10）；
`[ ]` 核对 pod 上 `rlab/datasets/dapo_math/train.jsonl` 的真实行数。

### 6.5 · OOD 评测集（P0 剩余项，路线待定）

参考的结论建立在 AIME25（OOD）上；run2 只有同分布 dev。**没有 OOD 集就无法说"复现了 ReTool"。**

阻碍：pod 文档写明 **HF 网络不通**，而参考的 AIME25 走 `04-opsd/00-datasets.py --only aime25`，
本仓库没有对应的准备脚本。两条路线：

1. **modelscope 路线**（与 `data.py` 里 gsm8k 的既有做法一致）：新增 `--eval_task aime25` +
   `load_aime25_test()`（严格报错、不静默回落，照 `load_dapo_math_dev` 的模式）+ 下载脚本；
2. **手工放置**：把 AIME25 jsonl 放到 `rlab/datasets/aime25/test.jsonl`（格式 `{"Q":..., "A":...}`），
   只加加载器，最省事最可控。

评测口径对齐参考：30 题 × 12 采样、temp 1.0 / top_p 0.7、报 Average@12 / Pass@12 / Format。

---

## 7. 进度看板

| # | 项 | 状态 | 备注 |
|---|---|---|---|
| P0-1 | eval per-item 明细默认落盘（`--dump_items`） | `[x]` | `eval_vllm_one.py`，默认开；`--no-dump_items` 关闭 |
| P0-2 | 汇总判定改 N-aware CI + 同题配对 McNemar | `[x]` | `rlab/analysis.py`，旧固定 ±2pp 地板已降级 |
| P0-3 | 审计字段 `model_path` / `eval_protocol` / `_meta` | `[x]` | 修"事后无法核对评的是哪个 ckpt / BASE 用哪条路径" |
| P0-4 | OOD 评测集（AIME25） | `[ ]` | 路线待定，见 §6.5 |
| 习惯 | run 偏离签名三处冗余 | `[x]` | `train.py`：wandb name / `run_info.json` / 启动日志 |
| P1 | `trunc_shaping 0.5 → 0.0`，300 步，存盘 50 | `[ ]` | 判据见 §6 P1 |
| P2 | 原生 `<tool_call>` + rounds 6 / per_round 1024 | `[ ]` | 与 P1 分开跑 |
| P3 | 步数 200–300 + 存盘 50 成为默认 | `[ ]` | |
| P4 | 无难度过滤对照 + `train.jsonl` 行数核对 | `[ ]` | 见 §10 |

---

## 8. 反建议（明确不要做）

- ❌ **不要**继续在围栏协议上抬高单轮预算 —— 治标，且继续挤压轮数。
- ❌ **不要**把 final / step1000 当主结果 —— m600 已回落到 BASE（46.4% vs 46.6%）。
- ❌ **不要**引用旧版 `analysis.py` 的"真差异"标签 —— 它是 GSM8K/N=300 口径的常数。
- ❌ **不要**在补上 OOD 集之前宣称复现了 ReTool —— 参考的 +23.89pp 是 OOD。
- ❌ **不要**一次改多个变量 —— 尤其别把 P1（奖励）和 P2（协议）合到一次跑里。
- ❌ **不要**用 run2 的 `eval_vllm_all.json` 做配对检验 —— 旧产物永远只有聚合值，补不回来。

---

## 9. 附录 A：测量口径修正（本次已落地）

### 9.1 为什么必须改

旧版 `summarize_eval` 用**固定 ±2pp 地板**判"真差异"（`NOISE_FLOOR_PP`），那是 GSM8K/N=300 时代的常数。
实测量级：dapo_math **N=500** 下**单臂 95%CI 就有 ±4.4pp、两臂差 ±6.2pp** —— 于是 m200−BASE 的
+5.0pp（未配对 p≈0.11）会被直接打成"真差异"。同时旧版 eval 只存聚合计数，**同题配对检验永远算不出来**。

### 9.2 改了什么

| 文件 | 改动 |
|---|---|
| `eval_vllm_one.py` | 新增 `--dump_items`（`BooleanOptionalAction`，**默认开**）；每题记 `qk`（题面 sha1 指纹）/`acc`/`fmt`/`code_used`/`code_ok`/`ans_len`/`empty`；结果加 `model_path` + `eval_protocol`；顺带修空答案分支的 `a`/`f` 未赋值隐患 |
| `eval_vllm.py` | 聚合 json 加 `_meta`（`base_path`/`tuned`/抽样口径/`failures`/`created`）；`analysis.py` 汇总时跳过 `_` 前缀键 |
| `rlab/analysis.py` | 新增纯函数 `ci95` / `diff_ci95` / `mcnemar_exact` / `paired_counts` / `_verdict`；`summarize_eval` 输出 CI 列 + 检验列；**有 per-item 用 McNemar，没有则明确标注"（无 per-item）"**，不冒充满配检验 |
| `rlab/train.py` | `run_signature(cfg)` + 三处冗余；`run_info.json` 顶层补 `save_steps`/`max_rounds`/`trunc_shaping`/`overlong_shaping`/`difficulty_path`/`difficulty_band`/`lr`（完整 `config` 仍保留） |
| `rlab/tests/test_smoke_cpu.py` | 新增 `[G]` 段 19 项（原 45 → **64 项**） |
| `rlab/tests/test_retool_cpu.py` | `run_info` 验收补 3 项签名接线检查 |

### 9.3 run 偏离签名

```
retool_math-ts0.5-ol1-r2x3072-s1000x200-lr5e-06-d0-1     ← run2 实跑
retool_math-ts0-ol1-r2x3072-s300x50-lr5e-06-d0-1         ← P1 拟跑
retool_math-ts0-ol1-r6x1024-s300x50-lr5e-06-d0-1         ← P2 目标
retool_math-ts0.5-ol1-r2x3072-s300x50-lr5e-06-nodiff     ← P4 对照
         └algo  └trunc └ol └轮次×单轮 └步数×存盘 └lr    └难度带
```

三处冗余：启动日志 `偏离签名 signature=`（grep 即得）、`run_info.json` 顶层、wandb run name。

### 9.4 用 run2 真实数字验证

```
> N=500 · 单臂 95%CI 最坏 ≈±4.4pp · 两臂差 ≈±6.2pp · 判定 = CI 不跨 0（有 per-item 时改用 McNemar p<0.05）

| 模型 | acc%(±95%CI) | fmt% | code% | Δacc vs BASE | 检验 | 判定 |
| BASE | 46.6±4.4 | 53.0 | 42.2 | — | — | — |
| m200 | 51.6±4.4 | 60.2 | 62.6 | +5.0±6.2pp | 两比例（无 per-item） | 噪声内 |
| m600 | 46.4±4.4 | 67.8 | 13.0 | -0.2±6.2pp | 两比例（无 per-item） | 噪声内 |
```

### 9.5 使用注意

- 命令**不变**：`--dump_items` 默认开，wrapper 不转发也能生效。flag 拼写是**下划线**
  （`--no-dump_items`；argparse 不接受 `--no-dump-items`）。它**只能直接给 `eval_vllm_one.py`**，
  `eval.py`/`eval_vllm.py` 暂无透传出口。
- **配对不要求一次调用评完所有模型**：`random.seed(seed)` → `random.sample(test_data, n)`，
  `test_data` 顺序确定 ⇒ 同 seed/split/n 的不同次调用抽到**同一批题**，跨次也能靠 `qk` 配对；
  若某次改了 `--n`，则只按交集配对，表格 `n=matched` 会显示实际配对数。
- **旧的 `eval_vllm_all.json` 重跑即被覆盖**，先 `cp` 备份（它是 run2 唯一的聚合记录）。

---

## 10. 附录 B：数据侧待核对

`filter_qas_by_difficulty`（`rlab/data.py:312-313`）把"不在探针表里"的题**静默丢弃**（`missing` 口径，
注释："过滤即选择，半覆盖的表不该让未探测题混进训练分布"）。run2 日志：

```
[rollout] 难度过滤 band=(0.0,1.0): 1791200 -> 20792 题（全错 23196 / 全对 8695 / 区间外 0 / 表中缺失 1738517）
```

两点需要查证：

1. **97.1% 的池子是因"未探测"被丢，不是因难度被丢**（23196+8695 = 31891 ≠ 20792，也不等于 17912）。
   即训练分布 = 探针表恰好覆盖到的那部分，**探针覆盖率是承重结构**。
2. **池子总量 1,791,200 与 DAPO-Math-17k 的 17.9k 相差约 100×** ——
   需确认 pod 上 `rlab/datasets/dapo_math/train.jsonl` 到底是 1.79M 行还是 17.9k 行（是否重复展开）。

与 §4.4 的退化率差异（参考 ≈0 / run2 14–16%）对照，这两点可能相关。**优先用一次无
`--difficulty_path` 的 300 步对照跑回答"过滤是否真在降退化"。**

---

## 11. 变更文件与验证状态

| 文件 | 状态 |
|---|---|
| `eval_vllm_one.py` | 已改（per-item + 审计字段） |
| `eval_vllm.py` | 已改（`_meta`） |
| `rlab/analysis.py` | 已改（统计口径） |
| `rlab/train.py` | 已改（偏离签名） |
| `rlab/tests/test_smoke_cpu.py` | 已改（+19 项，**64 项全过**） |
| `rlab/tests/test_retool_cpu.py` | 已改（run_info 验收 +3 项，隔离运行全过） |
| `rlab/readme.md` | 已改（测试计数与 eval/analysis 说明） |

验证：
- `python -m rlab.tests.test_smoke_cpu` → **64 项全过**
- `test_grad_clip_and_run_info()` 隔离运行 → 全过
- 全部改动文件 `py_compile` / AST 通过
- ⚠️ `test_retool_cpu` / `test_train_step_cpu` 在**开发机**跑到需要 `transformers` 的段
  `ModuleNotFoundError` 退出；`git stash` 后复现同样失败 ⇒ **环境缺失，与本次改动无关**（pod 上有）。
