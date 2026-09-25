# 方案 A：原生 `<tool_call>` 协议改造（不做 SFT 直接 RL）

> 状态：**代码已落地**（2026-09-25），**待 pod 核实 + 上机验收**。
> 定位：**这是主线修复**，不是备选。`08-sft-cold-start.md` 的 SFT 方案从"主线"
> 降为 **fallback**（换基座或原生协议验证失败时才启用），理由见 §0。
> 依据：`agentic-rl-lab/05-retool` 用**同一个基座 `Qwen/Qwen3.5-4B` + 同一份
> DAPO-Math-17k**，**跳过 cold-start SFT 直接 RL**，拿到 AIME25 Average@12
> **23.61% → 47.50%（+23.89pp）**。这是干净的存在性证明。
> 本项目的 docs/05 §12.1 早已把它列为 **#4 偏离项 · 方案 A**，触发条件
> （"p6+shaping 都救不回"）在 p11 已满足，但从未执行。
>
> **实施摘要**（详见 §10）：
> - `tool_protocol` ∈ `{"fence","native"}` 运行时开关，**缺省 `"fence"`** =
>   p1–p11 逐位可复现（已用 AST 比对脚本证明围栏分支 71 行逐行相同）；
> - `rlab/native_probe.py`：pod 上第一条命令，把 Q1–Q4 一次问完并直接给出
>   `--native_tool_style` 该钉哪个值；
> - `rlab/tests/test_native_protocol.py`：**162 项** CPU 测试（P1–P7 + 打分域 + P2b 空白容忍全覆盖）。

---

## 0. 为什么这是主线：三方对照

| 实现 | 工具协议 | base 先验 | 是否需要 SFT | 结果 |
|---|---|---|---|---|
| ReTool 论文 | 自定义 `<code></code>` 围栏 | 无 | **需要**（论文做了） | 67.0 (AIME24) |
| **agentic-rl-lab** | **原生 `<tool_call>`** | **调用率 87.5%** | **不需要** | **+23.89pp** |
| **rlab p1–p11** | **自造围栏 + `[TOOL RESULT]` 文本** | **无先验**（实测 `code_ok`≈48%） | **需要，但没做** | 无增益 |

**rlab 选了"需要 SFT 的那条协议"，然后跳过了 SFT——两边的便宜都没占到。**

参考项目的边界说明原文：

> **跳过 cold-start SFT**：对 Qwen3.5-4B 成立（base 工具调用率就够高），换基座不一定成立。

### 0.1 四类"代码灭绝"是同一个病的四个症状

| 前四轮诊断出的机制 | 在原生协议下的本质 |
|---|---|
| ① 280 截断围栏 → 代码样本结构性负奖励 | 围栏要求"写满再停"；原生 tool_call 一决定调用就停 |
| ② `MAY` 采样率 0.3% → 进不了分布 | 模板里有 87.5% 的调用先验，自造协议要从 0 教起 |
| ③ `MUST` ⊕ 格式正则 `^` 锚定冲突 | 用 prompt 硬教一个非原生格式的必然副作用 |
| ④ 理性压灭（写代码期望收益为负） | 写围栏烧 6144 token 换一次执行；原生协议几百 token |

①②③ 是真 bug 且已修（修复本身正确）。但**修好它们只是让自造协议"能用"，
不能让它"有先验"**。④ 不是 bug——在错误协议上它是**正确**的 RL 决策。

### 0.2 另外两条独立欠配（不是本方案修，但要同时知情）

| 维度 | 参考 | rlab p11 | 后果 |
|---|---|---|---|
| `lr` × 可移动性 | LoRA r32 / **4e-5** | 全参 / **1e-6** | p3 实测 5e-6 就崩（条件精度 82→56%）→ 锁死在"太小/太大"窗口 |
| optimizer 更新 | **200** | **75** | 剂量不足 |
| 每步轨迹 | 8 题 × 8 = **64** | 1 题 × 8 = 8（GAS4→32/更新） | 梯度噪声大 |
| 总轨迹 | **12,800** | **2,400** | 5.3× |
| 评测 | **AIME25 (OOD) · Average@12** | dapo dev（同分布）· **greedy** | 看不见采样空间里的增益 |

> **口径更正**：`08-sft-cold-start.md` 写过"与 DAPO 官方差 250 倍 / 与 ReTool 差
> 85 倍"——那是拿 ReTool **论文**（32B、batch 512）比的。对**参考实现**（真正的
> 复现目标）是 **5.3 倍轨迹 / 2.7 倍更新**。后者才是该用的口径。

docs/05 §7.5.2 自己记过："**greedy 会把多轮代码测成灭绝**"。用 greedy +
同分布去测一个采样空间策略，本来就测不出增益。

---

## 1. 实施前必须先在 pod 上核实的事（关键，勿跳）

### 1.1 ⚠ Qwen2.5 与 Qwen3.5 的原生格式**不同**——不能互推

本机用 `D:\tmp\qwen_tok_audit\Qwen\Qwen2___5-3B` 的 tokenizer 实测
（transformers 4.41.2），Qwen2.5 的原生渲染是 **JSON 形态**：

```
<|im_start|>assistant
let me compute
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print(4)"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
4
</tool_response><|im_end|>
```

而参考实现的 `TOOL_CALL_PATTERN` 吃的是 **Qwen3.5 形态**：

```python
r"<tool_call>\s*<function=code_interpreter>\s*<parameter=code>\s*(.*?)\s*"
r"</parameter>\s*</function>\s*</tool_call>"
```

**两者不兼容。** 本机只有 Qwen2.5 tokenizer，**无法验证 Qwen3.5 的真实形态**。

### 1.2 核实脚本（pod 上第一条命令）

**已实现为 `rlab/native_probe.py`**（纯 tokenizer + 可选 vLLM 冒烟，几秒钟）：

```bash
cd ~/llm-learning
python -m rlab.native_probe --model_path /root/Qwen3.5-4B --out /tmp/native_probe.txt
```

它按顺序做五件事，末尾直接打印结论行（可粘贴进 run 命令）：

| 段 | 内容 | 对应问题 |
|---|---|---|
| Q1 | 渲染三档（带 tools 关思考 / 带 tools 默认 / 不带 tools 对照），打印是否含工具声明 | Q1 |
| Q4 | `enable_thinking=False` 与 `tools=` 共存性 + 检查结尾是否为未闭合 `<think>` | Q4 |
| Q2/Q3 | 两种 `arguments` 形态各渲染一次，**repr 落盘**并打出调用段与回包段的原始字节 | Q2 / Q3 |
| 往返证明 | 实测渲染出的 assistant 文本喂 `parse_assistant`，必须 `kind="tool"`；再 `derive_tool_style` 判定形态 | Q2 |
| 端到端拼接 | `build_next_prompt` 真跑一遍，与模板 canonical 序列逐 token 对拍 | Q3 |
| vLLM 冒烟 | base 真采样 `n` 条，统计调用率（`--no_vllm_smoke` 可跳） | §6.2 第 5 步判据 |

**本机实测输出**（Qwen2.5-3B tokenizer，transformers 4.41.2，作为脚本自身可用性的证据）：

```
Q1 tools= 渲染          : ✅ 通过（带 tools 1213 字符 vs 不带 158）
Q2 调用形态              : json
Q3 回包形态              : <|im_start|>user\n<tool_response>\n391\n</tool_response>
Q4 enable_thinking 共存  : ✅
往返证明                 : kind=tool style=json code='print(17*23)' ✅
端到端拼接               : 拼接结果长 313 == canonical 313，逐 token 相同 ✅
⇒ 训练命令加：--tool_protocol native --native_tool_style json
```

> ⚠ **这个 `json` 是 Qwen2.5 的结论，不是 Qwen3.5 的。** 必须在 pod 上对
> `/root/Qwen3.5-4B` 重跑一次，用它的判定值——这正是本节存在的原因。

**必须确认的四个问题**：

| # | 问题 | 影响 |
|---|---|---|
| Q1 | `tools=` 是否被模板接受并渲染 | 不接受 → 本方案不成立，转 docs/08 SFT |
| Q2 | tool_call 是 `<function=…>/<parameter=…>` 还是 JSON | 决定解析正则 |
| Q3 | tool 回包渲染成什么（`<tool_response>`? `role:tool`?） | 决定增量拼接 |
| Q4 | `enable_thinking=False` 与 `tools=` 能否共存 | 冲突 → 思考模式烧穿预算（docs/03 事故） |

> **不要照抄参考的正则**。参考跑在它的 transformers 版本上；本项目 pod 的
> transformers 版本不同（本机 4.41.2）。**正则必须从实测渲染结果派生**，
> 并按 §7 的 roundtrip 测试锁死——这正是"标签经管道被改写"踩过三次的坑。
>
> **两种形态都已原生实现**（`NATIVE_STYLES`），运行期由 `derive_tool_style`
> 从实测渲染派生该用哪种；`--native_tool_style {auto,function,json}` 可显式钉死。
> 加新形态 = 在 `NATIVE_STYLES` 里加一项 + `_find_call` 加分支，
> **不要改现有正则去凑**（会同时废掉已通过 roundtrip 的那一档）。


---

## 2. 改造清单（逐文件）

> **本节的表是设计稿；实际落地见 §10**（已实现并逐项测试）。

### 2.1 `rlab/protocol.py` — 协议层（改动最大）

| 动作 | 内容 |
|---|---|
| **新增** | `CODE_TOOL`：工具声明 dict（照参考的 description 写法——强调 `print()` 输出、每次执行独立） |
| **新增** | `parse_assistant(text)` → `(kind, content, code)`，`kind ∈ {"tool","answer","invalid"}`；**正则从 Qwen3.5 实测形态派生** |
| **新增** | `tool_message(text)` → 结构化回包消息（`role`/`tool_call_id`/`name`/`content` 依实测形态定） |
| **新增** | `build_next_prompt(...)` → **token 级增量拼接**（见 §3，最易错处） |
| **保留** | `sanitize_tool_text`（沙箱 stdout 仍是注入面，消毒点不变） |
| **保留** | `segment_mask_from_spans`（**mask 契约一字不改**，见 §4） |
| **改** | `RETOOL_STOP_KWARGS` → 不再需要（原生协议自然停在 EOS/`im_end`） |
| **改** | `extract_python_blocks` → 保留但仅供旧协议回放/对照，新路径不再调用 |

### 2.2 `rlab/rollout.py`

| 函数 | 改动 |
|---|---|
| `build_prompt`（L70） | 加 `tools` 形参，透传 `apply_chat_template(tools=[CODE_TOOL])` |
| `multi_turn_rollout_group`（L391） | 检测逻辑从 `extract_python_blocks` 换成 `parse_assistant`；回包从"拼 `TOOL_START`+文本"换成 `build_next_prompt` 的 **token 增量**；`is_final_round` / `code_wasted` 语义保留 |
| `retool_build_batch`（L540） | **不改**——它只消费 `segs[*]["ids"]`，与协议无关 |
| `collect_retool_group`（L704） | 基本不改（打分域仍是 assistant 段拼接） |

### 2.3 `rlab/config.py`

- `retool_math` preset：预算/轮数按 §5 改；`retool_stop` 置 False
- **系统提示**换成原生工具说明（**删掉所有围栏/`[TOOL RESULT]` 措辞**）——
  建议直接以参考的 `SYSTEM_PROMPT` 为蓝本（它比本项目 1298 字符那版短得多，
  且不含围栏指令）
- 新增 `sft` fallback preset（docs/08 用，本方案不启用）

### 2.4 `eval_vllm_one.py`

`build_prompt` / `multi_turn_rollout_group` 都是复用，**改动自动生效**；
只需把 `sp_mt` 的 stop 参数从 `RETOOL_STOP_KWARGS` 改为不传（原生 EOS 停）。

### 2.5 需要新增的回归位

`eval_vllm_one.py` 的 fail-fast：请求了 tools 但渲染结果里没有工具声明
→ 立刻告警（对应 §1.2 的 Q1）。与现有 `enable_thinking=False` 的
`rstrip().endswith("<think>")` 检查同型。

---

## 3. token 级增量拼接（本方案最易错处）

原生协议下**不能重渲染整段历史**——参考实现记录了这个坑：

> token-in token-out：续写用的是真实采样出来的 token 序列，不做 text↔token
> 重编码。官方实测重编码不可逆会导致约 100 步后性能崩塌、grad_norm NaN。

而且 Qwen 的 chat template **会 strip assistant 内容**，采样文本含 `</think>`
时还会被重构成 reasoning/content 两段——用真实文本定位边界必然失败。

**正确做法（照参考的 `build_next_prompt`）**：

```
① canonical_assistant_end = render(messages + [{"role":"assistant","content":"x"}],
                                   add_generation_prompt=False)
② 校验 canonical_* 前缀一致（模板没有改写历史），否则 raise
③ assistant_closing_tokens = canonical_assistant_end[len(canonical_action):]
④ 用 _suffix_prefix_overlap 算出 sampler 已返回多少结束符，只补差额
⑤ canonical_next = render(messages + [占位 assistant, tool_message],
                          add_generation_prompt=True)
⑥ observation_tokens = canonical_next[len(canonical_assistant_end):]
⑦ next_ids = prev_ids + sampled_ids + closing[overlap:] + observation_tokens
```

**为什么用占位内容**：增量片段只与 tool 消息有关、与 assistant 内容无关，
所以 canonical 计算可以用占位 `"x"`——绕开"采样文本被模板重构"的坑。
校验 `canonical_assistant_end[:len(canonical_action)] == canonical_action`，
模板行为变了就 fail-fast，不静默产出错位序列。

**这比现有围栏路径更严格**：现有路径是"检测到围栏 → 事后提取代码 →
拼 `TOOL_START` 文本"，`retool_stop` 靠 `stop=["```\n"]` 强行截断。原生路径
不需要 stop 串，模型自然停在 `im_end`（EOS 已在 token_ids 里，
docs/05 §12.3 已核对）。

---

## 4. mask 与 batch 契约（**保持不变**，为什么）

三件事**一行都不用改**：

| 契约 | 为什么不变 |
|---|---|
| `segment_mask_from_spans`（assistant=1 / tool=0 / pad=0） | 掩码按**段边界**给，与段内是什么协议无关 |
| `retool_build_batch`（分段 ids 拼接 + mask） | 只消费 `segs[*]["ids"]` |
| `has_mask:1` 的 extras 布局 | 同上 |

**"工具返回 token 不进 loss"这条核心契约在原生协议下同样成立**，且更重要：
原生回包经 chat template 渲染，可能带 `<tool_response>`、`im_start` 等
**非模型生成**的结构 token——它们必须在 mask=0 区间。docs/02 的 A/B 已实锤：
mask 错时 loss≈3.4 且工具位梯度非 0（假信号）。

> **新增注意点**：原生协议的 observation 段比旧协议的 `[TOOL RESULT]…` 更长
> （含模板包装 token），`tool_result_max_chars` 的预算折算（§5 的 reserve 项）
> **要重新核对**——旧公式按 2 字符/token 折，模板 token 不占字符。

---

## 5. 参数配置（推荐档）

### 5.1 主推荐：**只改协议**（严格单变量）

预算候选实跑 `validate_retool_budget` 核对（`max_prompt_length=1024`、
`tool_result_max_chars=500`）：

| 档 | rounds × per_round | ctx | need | margin | 代码执行次数 |
|---|---|---|---|---|---|
| **A（推荐）** | **5 × 1024** | **8192** | 7208 | **+984** | **4**（= 参考 `max_code_calls=4`） |
| B | 6 × 1024 | 8192 | 8498 | **−306 ✗** | 5 |
| C | 6 × 1024 | 10240 | 8498 | +1742 | 5 |
| D（回退位） | 5 × 1536 | 12288 | 9768 | +2520 | 4 |
| E（p11 现状） | 4 × 6144 | 26400 | 26398 | +2 | 3 |

> 注意：`max_rounds=5` → 只有前 4 轮执行代码（末轮保证是 final 轮，docs/02 发现3
> 的修复语义不变），恰好对应参考的 `max_code_calls=4`。

**只动协议相关项，其余逐字保持 p11 值**：

| 参数 | p11 | 新档 | 依据 |
|---|---|---|---|
| 工具协议 | 围栏 + `[TOOL RESULT]` | **原生 `<tool_call>`** | §0 |
| `max_rounds` | 4 | **5** | 参考 `max_assistant_turns=6`/`max_code_calls=4` |
| `round_gen_tokens` | 6144 | **1024** | 参考单轮 1024（原生协议一次调用很短） |
| `max_context_tokens` | 26400 | **8192** | 参考轨迹 ≤8192（实测训练分布 ~2200） |
| `retool_stop` | `True` | **`False`** | 原生自然停在 EOS |
| `system_prompt` | concise 围栏版 | **原生工具版** | 删围栏措辞 |
| **其余全部** | — | **不变**（保持 `lr 1e-6` / `β 0.04` / 各 shaping 原值） | **单变量纪律** |

### 5.2 后续单变量（**一轮一个，不要合并**）

协议跑通（判据见 §6.2）后按序：

| 序 | 变量 | p11 | 改为 | 依据 |
|---|---|---|---|---|
| 1 | `beta` | 0.04 | **0.0** | 参考与论文均 KL=0.0；rfpp 实验实测 β=0.04 反而更差 |
| 2 | `loss_norm` | `sample_mean` | **`token_mean`** | 多轮长轨迹标准配置（DAPO） |
| 3 | `code_w` / `code_attempt_w` | 0.05 / 0.05 | **0.0** | 参考跟随论文 **outcome-only，不加任何 shaping** |
| 4 | `len_penalty_w` | 0.1 | **0.0** | 参考无此项；协议对了之后长度压力应自然下降 |
| 5 | 剂量 | 300 步 / 75 更新 | **600~1200 步** | 参考 200 步但 LoRA+64/步；本档需相应放大 |
| 6 | `lr` / LoRA | 全参 1e-6 | 评估 **LoRA r32 / 4e-5** | 参考档；但需新增 LoRA 支持（工程量） |

> **第 3、4 项方向与 p11 相反**——p11 是"加 shaping 救代码"，参考是"完全不加"。
> 协议换成原生后，先验本身就在，不再需要 shaping 对冲。**但必须等 ①协议跑通
> 再动**，否则又是复合变量（docs/05 §4.3 的教训：改地形后再测 lr 结论必作废）。

### 5.3 评测档（**与训练同轮改，否则看不见效果**）

```bash
python -m rlab.eval --algo retool_math --n 30 --seed 42 \
  --val_n 12 --temperature 1.0 --top_p 0.7 \
  --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
  --gpu_mem 0.78 \
  --models "step50=...,step100=...,step150=...,step200=..." \
  --out eval_vllm_all_pA.json
```

| 参数 | 值 | 依据 |
|---|---|---|
| `--val_n` | **12** | 参考 `Average@12` 口径 |
| `--temperature` / `--top_p` | **1.0 / 0.7** | 参考评测口径 |
| `--n` | **30**（AIME25 题数） | 参考是同量级；dapo dev 用 500 |
| OOD 集 | **AIME25** | docs/05 §6.6 已列路线，**一直未落地**；没有它无法与参考 +23.89pp 对话 |

---

## 6. 实施顺序与判据（跑之前写死）

### 6.1 顺序

| 步 | 内容 | 工作量 | 状态 |
|---|---|---|---|
| 1 | **部署**：pod 上 `git pull` | 1 分钟 | ⬜ |
| 2 | **§1.2 核实脚本**（Q1–Q4）→ 拿到 `--native_tool_style` | 5 分钟 | ⬜ |
| 3 | 代码（protocol/rollout/config/train/eval/probe）+ CPU 测试 | 1 天 | ✅ **已完成** |
| 4 | **base 冒烟**：只测 base 的调用率（不训练） | 5 分钟 | ⬜ |
| 5 | 20 步验证跑（参考的"两杯瑞幸"档） | 1 小时 | ⬜ |
| 6 | 200 步正式跑 | 1 天 | ⬜ |
| 7 | 评测（`--val_n 12` + AIME25，§5.3） | 半天 | ⬜ |

> 步 2 与步 4 是**同一个**脚本的两个档（`rlab/native_probe.py` 不带/带
> `--no_vllm_smoke`）——步 2 只验 tokenizer 层（几秒），步 4 顺带跑 vLLM 冒烟。
> **步 4 是本方案的 go/no-go 闸门**，不过不要往下走。

### 6.2 判据

**第 5 步（base 冒烟）——本方案的 go/no-go 闸门**：

| 观察 | 判定 |
|---|---|
| base 原生协议工具调用率 **≥50%** | ✅ 先验吃到了，继续 |
| 调用率 <20% | ✗ 模板/`enable_thinking` 没接对，回 §1.2 重查 |
| 调用率高但 `parse_assistant` 大量 `invalid` | ✗ 正则没对齐实测形态，回 §2.1 |

> 对照基线：参考 base **87.5%**；rlab p11（围栏协议）**~48%**。
> 这一档由 `rlab/native_probe.py` 的 vLLM 冒烟段直接给出（含 invalid/answer 分解）。

**第 6 步（20 步验证跑）**——闭环是否成立：

| 观察 | 判定 |
|---|---|
| `code_calls` 上升（参考 1.24→1.72） | ✅ 在学"多试几次" |
| `degenerate group`（整组无梯度）≈ 0 | ✅ 组内 reward 有区分度 |
| `correct` 上升（参考 0.289→0.389） | ✅ 闭环有效 |
| `trunc` 显著低于 p11 的 53.8% | ✅ 协议不再结构性逼长 |

**第 7 步（200 步）**：

| 观察 | 判定 |
|---|---|
| Average@12 单调升 | ✅ 复现成功 |
| code 率不跌回 ~0 | ✅ 摆脱四类灭绝 |
| 与 BASE 配对 McNemar **p<0.05** | ✅ 增益显著 |

---

## 7. CPU 测试（新增 `test_native_protocol`，按既有风格）

**已实现：`rlab/tests/test_native_protocol.py`，162 项全部通过**
（`python -m rlab.tests.test_native_protocol`）。
Mock tokenizer + 纯函数 + FakeGen，**不碰 GPU**：

| 组 | 检查 | 项数 |
|---|---|---|
| **P1 解析** | 两形态各 4 项；无调用→`answer`；两个调用→`invalid`；**跨形态两个调用**→`invalid`；调用后跟文本→`invalid`；工具名不对→`invalid`；空代码→`invalid`；JSON 截断→`invalid`；**开标记但不成块**→`invalid`（未闭合调用绝不当答案）；代码含花括号/引号/尖括号→正确取出；`arguments` 为 JSON 字符串；已知限制（代码含闭标记字面量→`invalid`，锁住当前行为） | 23 |
| **P2 序列契约** | 输出 == `prev + sampled + closing[overlap:] + observation`（**与实现的公式独立手算一遍**）；prev 为前缀；observation 非空；结束符已补；**sampler 已含完整/部分结束符 → 只补差额**；`suffix_prefix_overlap` 纯函数口径；多轮递推 | 9 |
| **P3 模板改写防护** | tools 档不一致→raise；模板改写历史→raise；prev 来自另一份 messages→raise；**模板吞掉回包内容→raise**（前缀校验查不出的"静默空转"，新增校验 ④）；空回包不触发（反证） | 5 |
| **P4 mask/梯度** | 原生回包段 token 全 `mask=0`、assistant 全 1；正确 mask `loss == -1`（= `-(adv·ratio)`，工具位零贡献）；工具位梯度**严格 0**；assistant 位梯度非 0（反证掩码非空）；错误 mask loss 被 KL 污染且**假梯度**；围栏/原生两档 loss **逐位相同**；mask 全 0 → loss 恒 0（反证） | 11 |
| **P5 零标签字面量** | 本文件不手写完整调用字面量（正则扫描）；两形态 roundtrip（派生→解析→code 逐字相同）；`NATIVE_STYLES` 无重复 | 6 |
| **P6 预算** | 档 A（5×1024/8192）通过且 **need==7208**；**档 B（6×1024/8192）必须 FAIL**；档 C/D 通过；`max_rounds=5` → 4 次执行；参考口径对齐 | 8 |
| **D 打分域** | 调用块被剥（载荷里的 `\boxed{}` 不进打分域）；**反证：不剥离→判错(-1)、剥离→判对(+1)**（真差异，非同义反复）；未闭合调用块不剥（结构不合格）；围栏档文本剥离行为逐字不变；两档互不干扰 | 9 |
| **C 配置层** | 四算法缺省 `fence`；fence 档预算/提示/stop 逐字不变；native 档三件套联动；**两档其余参数完全相同**（严格单变量）；非法档 fail-fast；`tool_protocol_of`/`is_native_protocol` 纯函数 | 21 |
| **N FakeGen** | 多轮 active 集合逐轮收缩（4→3→2）；段序列；末轮调用不计 `code_used` 但记 `code_wasted`；**第 2 轮 prompt == prompt_ids + 第1轮 assistant ids + 工具段 ids（逐 token）**；**第 3 轮以第 2 轮为前缀（token-in token-out 递推）**；mask 段边界；**A/B 反证：同一文本在围栏档 `code_used` 全 0** | 17 |
| **W 接线** | `build_prompt_batch` 两档 ids 同源；默认 `tools=False`；左 pad 正确；prompts_messages 层次契约（扩样只在 `collect_retool_group`） | 8 |
| **S 静态** | 14 处接线（rollout 分支、签名、CLI、run_info、eval 回读、health/data/probe） | 14 |
| **P7 pyflakes** | J 组清单含新文件；无未定义名 | 2 |

**往返探针（pod 上做一次，脚本自动做）**：
`构造 tool_call 文本 → 交给 parse_assistant → 必须 kind="tool"` ——
这是"正则与模板同源"的运行时证明，比看代码可靠（本项目三次被显示层/管道改写坑过）。
`native_probe.py` 的"往返证明"与"端到端拼接证明"两段就是它，且**端到端那段会真的
跑一遍 `build_next_prompt` 并与模板 canonical 序列逐 token 对拍**（本机实测：
拼接结果 313 == canonical 313，逐 token 相同）。

**围栏档逐位复现的可验证形式**（比注释里的承诺硬，已实测）：

方法：用 AST 从 `git show HEAD:rlab/rollout.py` 与工作区两版里各取出
`multi_turn_rollout_group` 的围栏分支（从 `n = len(prompts_text)` 起，到
`def retool_build_batch(` 止），**去掉注释与空行后逐行比对**。
（新版函数体开头新增的 native 转调分支不属于围栏体，比对时剥掉——那是新增而非修改。）

结果：
```
旧围栏体 71 行 / 新围栏体 71 行（去注释）
✅ 围栏分支逐行相同 —— fence 档逐位复现成立
```

判据的可复现性：`test_native_protocol.test_p7` 之外，`test_retool_cpu` 的
④⑦ 两组断言已把"围栏分支的 code_used 自增位置"等关键行为**收窄到围栏源码段**
（原先全文件搜 `if is_final_round:` 会先命中原生实现），故围栏档的行为契约
持续被测试锁着，不依赖这一次性比对。

---

## 8. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| `tools=` 不被 Qwen3.5 模板接受 | 低（参考已验证同基座可用） | §1.2 Q1；失败则转 docs/08 SFT |
| 正则与实测形态不符 | **中**（Qwen2.5/3.5 形态已证不同） | §1.2 Q2 + §7 P5 roundtrip；不要照抄参考正则 |
| `enable_thinking=False` 与 tools 冲突 | 低 | §1.2 Q4 + 现有 fail-fast 同型检查 |
| 1024/轮 仍然截断 | 中 | 回退档 D（5×1536/12288）；先看 20 步的 trunc |
| 换成原生后仍无增益 | 低但存在 | 那说明瓶颈不在协议 → 启用 docs/08 SFT（fallback 而非主线） |
| 历史对照全部作废 | **确定** | **刻意**：协议变了，p1–p11 不可与新 run 同表。新开 out_dir 与报告章节 |
| 加了开关但漏接线（新键没人读） | 中 | S 组 14 项静态接线检查 + `-tpnative` 进签名 + 启动自证行 |
| 原生档"调用后又继续写" | **中** | `invalid_final` 列 + `native_invalid` 健康告警；兜底 `--native_stop_at_call` |
| **护栏误杀**（校验把模板归一化当不同源） | **高·已实际发生** | §10.6.2：校验① 改两级判据（空白容忍），P2b 组双向锁死 |

**回退路径**：本方案只动 `protocol.py` / `rollout.py` 的**新增分支**，
旧围栏路径保留（`extract_python_blocks` 不删）。用 `cfg["tool_protocol"]`
（`"fence"` / `"native"`）切换，`"fence"` = p11 行为逐位可复现——
这样"协议"本身成为可 A/B 的单变量，而不是一次性重写。

---

## 9. 本方案**不**解决的问题（如实声明）

| 未解决 | 说明 |
|---|---|
| `lr` 窗口窄 | 全参 1e-6 太小、5e-6 崩 → 需 §5.2 第 6 项（LoRA）才真正对齐参考 |
| 剂量 | 本方案不改步数，§5.2 第 5 项单独做 |
| OOD 评测集 | 需另建（docs/05 §6.6 路线），否则只能说"同分布有增益" |
| 与 p1–p11 的可比性 | **永久断裂**，刻意的 |

**一句话**：本方案把"地形"修对（协议 + 先验），**不**顺手改"步长"（lr/剂量）。
地形不对时测步长，结论必然作废——这是 docs/05 §4.3 已经付过一次学费的教训。

---

## 10. 落地记录（2026-09-25，代码已完成，待上机）

### 10.1 交付清单

| 文件 | 改动 | 关键点 |
|---|---|---|
| `rlab/protocol.py` | **+360 行** | `CODE_TOOL` / `ParsedAssistant` / 两种形态正则 / `parse_assistant` / `derive_tool_style` / `tool_message` / `make_call_id` / `initial_messages` / `render_chat_ids` / `build_next_prompt`（**4 处 fail-fast**）/ `suffix_prefix_overlap` / `stop_sequences`；`sanitize_tool_text` 增剥调用标记 |
| `rlab/rollout.py` | **+321 行** | `tool_protocol_of`/`is_native_protocol`；`multi_turn_rollout_group_native`（158 行）；`build_prompt(tools=)`/`build_prompt_ids`/`build_prompt_batch`/`prompt_messages_for`；`make_retool_sps` 按档分叉 stop；record 增 `invalid_final`/`ctx_full` |
| `rlab/config.py` | +90 行 | `TOOL_PROTOCOLS`；`tool_protocol`/`native_tool_style`/`native_stop_at_call` 三键（缺省 `fence`）；`NATIVE_PROTOCOL_DEFAULTS`（5×1024/8192）；`system_prompt_retool_math_native`；`default_system_prompt(algo, tool_protocol)` |
| `rlab/train.py` | +58 行 | `--tool_protocol`/`--native_tool_style`/`--native_stop_at_call`；签名 `-tp<native>`/`-nsc1`；run_info 落盘协议档；启动自证行 |
| `eval_vllm_one.py` | +112 行 | 协议档从 run_info 回读（CLI 可覆盖）；`tools=` 建 prompt；长度按 ids 量；**prompt 与 messages 按同一索引过滤**；stop 按档分叉；原生诊断打印 |
| `rlab/probe_difficulty.py` | +24 行 | 原生档走 messages；表指纹含 `tool_protocol`（**同模型同提示下两档是两个分布**） |
| `rlab/data.py` | +6 行 | 难度表 meta 比对清单含 `tool_protocol` |
| `rlab/health.py` | +31 行 | `invalid_rate` 观测 + `native_invalid` 告警（>10%，附两种成因的处置） |
| `rlab/reward.py` | +14 行 | `strip_code_blocks` 一并剥调用块（**打分域脚手架**：载荷里的 `\boxed{}` 不得劫持答案）；两档标记互不干扰，一次剥两种是 no-op 安全的 |
| `rlab/native_probe.py` | **新文件 309 行** | pod 核实脚本（Q1–Q4 + 往返 + 端到端 + vLLM 冒烟） |
| `rlab/tests/test_native_protocol.py` | **新文件 ~1100 行** | 162 项 CPU 测试（含 P2b 空白容忍组） |
| `rlab/tests/test_retool_cpu.py` | +47 行 | J 组清单加 `native_probe.py`；④⑦ 两处判据收窄到围栏分支；`apply_chat_template` 判据改 **AST**（注释里提函数名不再误判） |

**回归**：`pytest rlab/tests` → **82 passed**（含新 12 个测试函数）；
`test_native_protocol` 单跑 → **162 项全部通过**（P2b 空白容忍组为 2026-09-25 真机事故后新增）。

### 10.2 实施中发现并修掉的坑（都不是设计里预见到的）

1. **跨形态的"两个调用"会漏判**：初版 `_find_call` 按形态各自统计块数，于是
   "一个 JSON 调用 + 一个 function 调用"在 `style="auto"` 下跌进 JSON 分支
   （它只看得到 1 个 JSON 块）→ 判成合法 `tool`；同一段文本在
   `--native_tool_style function` 下却判 `invalid`。**同一份输出在两种档位下
   得到相反结论**，训练与评测会分叉。修复：合法性判据改成**形态无关的两层**——
   结构层（整段恰一个调用块 + 其后无内容）由 `_RE_TOOL_CALL_ANY` 统一判，
   形态层再判块内载荷。
2. **未闭合的调用被当成答案**：JSON 被轮长切断时会走 `answer` 分支 → 拿去打分，
   于是"协议失效"伪装成"模型给了个错答案"。修复：判据改为"**出现开标记**
   即 `invalid`"（含未闭合），与围栏协议"未闭合围栏不提取"同一语义。
3. **前缀校验查不出"回包被吞掉"**：若某版模板不认识 `role:"tool"`，observation
   只剩一个空 user 轮——前缀校验**照样通过**（前缀没变，只是内容没了），模型
   永远看不到执行结果却照常训练。修复：新增**校验 ④**，用同一条消息 + 空
   content 再渲染一次对比长度。这是本项目最贵的"静默空转"类 bug。
4. **JSON 解析不能用非贪婪**：`<tool_call>\s*(\{.*?\})\s*</tool_call>` 会在代码里
   第一个 `}` 处收口（`print({})` 就中招）。改贪婪 + 回溯到最后一个 `}`。
5. **`extract_python_blocks` 的教训复现在预算终局**：原生档 observation 放不下时
   终局，初版把这次计入 `code_wasted`——而 `code_wasted` 的既有语义是
   "写了代码但不会被执行的**调用**"，混进**预算**性终局会让"预算够不够"这个
   单变量失去可读数。改单列 `ctx_full`。
6. **uniform 组的 `code_wasted` 曾被硬编码 0**：原生档"答案轮之前才想起来调用"
   恰是丢弃组的高发形态，恒 0 会让该列在丢弃组上系统性偏低。已改为如实上送。
7. **测试断言本身的两处错**：①P4 期望 `loss==0`，实际单样本 `adv=+1` →
   `-(adv·ratio) = -1`（手算核对后修正，并补"mask 全 0 → loss 恒 0"作反证）；
   ②FakeGen 的轮次回放数目按我脑中的模型写，与实现不符（`collect_retool_group`
   才是扩样层）——暴露出**扩样层次**需要显式契约，已在 W 组锁死
   "prompts_messages 收每题一条，扩样只在 `collect_retool_group`"。
8. **接线静态检查会被注释误伤**：`"apply_chat_template" not in src` 命中了本轮
   协议的说明注释 → 改注释就能让检查变红。改 **AST 判据**（只看真调用节点）。
9. **打分域漏了原生档的脚手架**：围栏档把 ```python``` 剥掉再判 acc/fmt；原生档
   的等价脚手架是调用块本身——`{"code": "print(\\boxed{7})"}` 会让"最后一个
   boxed"被载荷劫持（实测：不剥离 `-1`，剥离后 `+1`）。修复：`strip_code_blocks`
   一次剥两种标记（两档标记互不干扰，都是 no-op 安全的），避免"每个调用点按协议
   选剥离函数"这个新分叉源。**这是设计稿完全没提到的缺口**——设计只写了
   "打分域仍是 assistant 段拼接"（§2.2），没说 assistant 段里现在多了调用块。
10. **`tool_protocol="native"` 在单轮算法上是静默空转**：只有多轮 rollout 读这个键，
   在 grpo/dapo 上传它 = 用户以为开了原生协议、实际什么都没发生。已在
   `get_config` 里 fail-fast（附"要跑原生请用 --algo retool_math"的改法）。

### 10.3 pod 上的执行序列（照着做即可）

```bash
cd ~/llm-learning && git pull

# ① 核实形态（几秒，不占 GPU）——Qwen3.5-4B 实测结论 = function（§10.5）
python -m rlab.native_probe --model_path /root/Qwen3.5-4B --no_vllm_smoke \
    --out /tmp/native_probe.txt
#    ↑ 看末段 Q2 一行确认形态 + Q5 拼接硬契约是否 ✅

# ② base 冒烟（go/no-go 闸门：调用率 ≥50% 才继续）
#    【2026-09-25 修复】旧版此处必崩：本文件漏传 gdn_prefill_backend=triton →
#    Qwen3.5 的 GDN 层落回 FlashInfer 现场 JIT → ninja 打爆内存被 SIGKILL
#    （只有 `Killed`、无 traceback）。现已默认 triton；若环境里有训练遗留的
#    VLLM_BATCH_INVARIANT=1，先 unset（否则引擎启动即 RuntimeError）。
#    【同轮修复】首版用玩具提示测调用率（等于手把手教模型用工具）→ 已改成
#    从 preset 取正式提示与采样参数（探针与训练同口径铁律）。首轮实测 16/16=100%。
unset VLLM_BATCH_INVARIANT
CUDA_VISIBLE_DEVICES=0 python -m rlab.native_probe --model_path /root/Qwen3.5-4B \
    --n_smoke 16 --native_tool_style function --out /tmp/native_probe_smoke.txt
#    ↑ 若 Q1 段的工具声明段没渲染出来 → 方案 A 不成立，转 docs/08 SFT
#    ↑ 判据看 **auto 与钉死档哪个高**（钉错形态 ≠ base 不会用工具）

# ③ 20 步验证跑（参考的"两杯瑞幸"档）
#    【2026-09-25 第二次上机】首次跑到样本 25 第 5 段被护栏误杀（校验① 把模板
#    的 strip 归一化当"不同源"，§10.6.2）——已修（空白容忍 + 报错带解码文本）。
#    重跑时应看到一次 "[protocol] 模板渲染与拼接上下文差 N 个 token …只报这一次"，
#    那是**正常**的归一化告警，不是错误。
bash rlab/run_gsm8k.sh retool_math /root/Qwen3.5-4B \
    --tool_protocol native --native_tool_style function \
    --out_dir rlab_out/native_p1 --all_steps 20 --save_steps 5 \
    --attn_implementation flash_attention_2 --zero_stage 2 --optim_8bit \
    --micro_rows 4 --vllm_gen_logps --vllm_logprobs_n 1 \
    --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
    --seed 42
# 盯三件事（§6.2 第 6 步）：code_calls 升 / degenerate≈0 / trunc << p11 的 53.8%

# ④ 200 步正式跑（同一命令，只改 --out_dir/--all_steps）
#    ⚠ gpu_mem 必须与训练同档（真机 A/B 差 +7.0pp，tools 变不了这个）
```

> **不要**沿用 `rlab_out/retool_math/` —— 协议变了，必须新 `out_dir`
> （§8：可比性永久断裂是刻意的，护栏 `guard_ckpt_collision` 也会拦）。

### 10.4 eval 命令（与训练同轮改，否则增益看不见）

```bash
python -m rlab.eval --algo retool_math --n 30 --seed 42 \
  --val_n 12 --temperature 1.0 --top_p 0.7 \
  --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
  --gpu_mem 0.78 \
  --models "step50=./rlab_out/native_p1/step_50,step100=...,step200=..." \
  --out eval_vllm_native_p1.json
```
`--tool_protocol` 会自动从每个 ckpt 的 `run_info.json` 回读（原生档训练的 ckpt
用原生协议评）——BASE 是裸模型，靠调度器的 `--proto_from` 同档（`eval_vllm.py`
的 `BASE_PROTO`，已支持）。

### 10.5 上机核实结果（2026-09-25 真机首跑，Qwen3.5-4B）

**步 ① `native_probe --no_vllm_smoke` 通过 —— 方案 A 成立。**

| 问题 | 结论 | 证据 |
|---|---|---|
| Q1 `tools=` 渲染 | ✅ 接受并渲染 | 带 tools 1748 字符 / 不带 166（差 1582 = 工具声明段） |
| Q2 调用形态 | ✅ **`function`** | 实测 `<tool_call>\n<function=code_interpreter>\n<parameter=code>\nprint(17*23)\n</parameter>\n</function>\n</tool_call>` |
| Q3 回包形态 | ✅ `role:"tool"` → user 轮 + `<tool_response>` | `<\|im_end\|>\n<\|im_start\|>user\n<tool_response>\n391\n</tool_response><\|im_end\|>` |
| Q4 thinking 共存 | ✅ 开关生效 | 关思考档结尾 `<think>\n\n</think>`（**已闭合**，不是烧预算的未闭合形态） |
| 往返证明 | ✅ 两种切法都过 | `kind=tool style=function code='print(17*23)'` → `derive_tool_style` ⇒ **function** |
| Q5 拼接硬契约 | ✅ 三条全过 | 前缀 / 采样 token 逐位保留 / observation 真拼入 |
| **base 调用率（go/no-go）** | ✅ **16/16 = 100%**（invalid 0） | 见 §10.5.1（含"玩具提示"口径警告与已修说明） |

**这一条是本次上机的最大价值**：Qwen2.5 实测 JSON 形态、Qwen3.5 实测 function 形态，
**证实两者不能互推**（§1.1 的立论）。双形态实现不是过度设计——若照抄参考正则
只写 function 分支，Qwen2.5 档全废；若只写 JSON 分支，本 pod 的 4B 全废。

**⇒ 训练/eval 命令钉死：`--native_tool_style function`**（依赖 auto 猜测没必要）。

### 10.5.1 步 ② go/no-go 闸门：**通过（16/16 = 100%）**

```
调用率 = 16/16 = 100.0%（invalid 0 / answer 0）
形态分布：命中调用 16 / 有调用标记但解析失败 0
```

**100% > 参考实现的 87.5%（base 调用率），且 invalid=0** —— 这是本方案最强的
成立信号：base 的原生工具先验足够强，"跳过 cold-start SFT 直接 RL"对
Qwen3.5-4B 成立（与参考项目对同一基座的结论一致）。

**但必须记一条口径警告（我自己的首版探针犯了）**：首版冒烟用的是本文件里的
**玩具提示** `SYS = "SYS: you solve math with a python tool."`——它把"用工具"
直接写在提示里，等于手把手教模型调用，测出来的 100% 是"被提示后"的调用率，
而参考的 87.5% 是在**它自己的正式提示**下测的 → **苹果比橘子**。

本项目对探针有一条铁律（`probe_difficulty` 2026-09-17）：
**探针与训练同口径**——提示是协议的一半。已修：`smoke_config()` 从 preset 取
正式系统提示与采样参数，且 Q1–Q4 渲染核实、往返证明、拼接证明**四处全部**改用
正式提示（模板行为可能依赖 system 段内容，玩具提示验出的形态不能外推）。
**⇒ 这 100% 的结论仍成立，但要在"用正式提示重跑"之后再引用为最终数字。**

顺带修了 Q1 的判据：旧版拿 `code_interpreter` 当"是否渲染了工具声明"的标志物，
而**训练提示自己就写满了这个词** → "不带 tools"的对照档误报"含工具声明=是"
（本机实测：对照档 1219 字符 vs 带 tools 2274 字符，确实差了一整个声明段）。
改用 `CODE_TOOL` 描述里的独特短语 `"Execute code in an isolated environment"`
（由模板**原样**插入声明段，故跨模板可移植；`<tools>` 是 Qwen2.5 模板特有的
包装标签，不可移植）。

**`arguments=str` 那一档 TypeError 是预期差异，不是故障**：Qwen3.5 模板用
`arguments.items()` 展开实参，故要求 mapping；本项目从不把 `tool_calls` 喂给模板
（续写走 token 拼接），此项仅作各代模板严格度的对照。

### 10.6 这次上机暴露的、**我自己的探针**的两个 bug（已修）

> 这一节单列，因为它们的共同性质是：**诊断工具本身在说谎**，比协议 bug 更危险
> ——它会把"没测到"伪装成"测出来是坏的"。

1. **【阻塞闸门·我的 bug】`native_probe.py` 是仓库里唯一不传引擎参数的 GPU 入口。**
   冒烟段建 `LLM(model=..., gpu_memory_utilization=0.30)`，没有
   `gdn_prefill_backend="triton"` → Qwen3.5 的 GDN 层落回 **FlashInfer 现场 JIT**
   → `ninja` 打爆宿主内存被 **SIGKILL**（日志只留 `Killed`，**无 traceback**，
   看起来像"冒烟没输出"）。仓库其它入口（`rollout.gen_worker` / `probe_difficulty`
   / `diag_logps` / `eval`）全都传了，`probe_difficulty` 甚至专门写了注释讲这个
   SIGKILL——**只有新写的这个文件漏了**。于是 go/no-go 判据那一关**根本没跑出数字**。
   修复：加 `DEFAULT_ENGINE_KWARGS = {"gdn_prefill_backend": "triton"}` +
   `--vllm_gen_kwargs`/`--vllm_attention_backend` 入口 + 起引擎前对"缺 backend"
   和"继承的 `VLLM_BATCH_INVARIANT`"两条告警；
   **并加了静态检查防再犯**（"新探针又漏引擎档"是这个仓库的复发性错误）。

2. **【诊断自己说谎】Q2 段的"真调用"计数用了 JSON 专用的启发式。**
   旧判据 `"{" in 块 and "name" in 块` 只对 Qwen2.5 的 JSON 形态成立；在 Qwen3.5 的
   `<function=…>` 形态下数出 **0 个**，于是打印 `含实参 0 个`——把一次**完全正常**
   的渲染读成"调用段是空的"，会把排查引向"正则不匹配"的错方向（首版还因此打印过
   `'<tool_call></tool_call>'` 的假象，见代码内注释）。修复：改用**生产解析器**
   `parse_assistant` 判真调用（它才是"正规形态"的唯一定义者），并逐个块打印载荷
   + 标注"真调用 / 格式说明"——顺带把"模板自己印的格式说明块"这件事显式说清
   （实测量：Qwen2.5 与 Qwen3.5 都各印 **2 个说明块** + 1 个真调用）。

3. **【判词掩盖未查明偏差】端到端拼接证明的"⚠ 不完全相同 → 属预期"是危险措辞。**
   真机报 `got 448 / want 444，首个分歧位 389`，旧版打印"差异属预期、判据以不 raise
   为准"——**这句话恰好盖住了一次没查明的 4-token 偏差**。本机拿真模板复核后确认：
   偏差来自**探针的 canonical 构造方式**（旧版用 `content=` 把调用标记当纯文本塞给
   模板，模板于是按纯文本渲染；现在改用模板认可的 `tool_calls` 形态），
   **不是** `build_next_prompt` 的 bug。修复：判据拆成两层——
   **硬契约**（① got 以 prev 开头 ② `got[len(prev):]` 以 sampled 开头
   ③ observation 真的拼进去了）必须成立，违者即"方案不成立"；
   与 canonical 的差异**一律逐 token 解码打印**（含"多出来的是哪几个 token、
   文本是什么"的指认），由人判读归属，不再打印"属预期"这种无信息量判词。

   本机复跑验证（Qwen2.5 真模板）：硬契约三条全 ✅，且与 canonical **逐 token 相同**
   （313 == 313）——反证了旧版那 4 token 是构造伪影。

4. **【口径失真·最隐蔽的一条】冒烟用了玩具提示。** 见 §10.5.1：首版
   `SYS = "SYS: you solve math with a python tool."` 把"用工具"写进提示 = 手把手
   教模型调用，与参考 87.5%（它自己的正式提示下测）不可比。已改为
   `smoke_config()` 从 preset 取正式提示与采样参数，**四处**（渲染/往返/拼接/冒烟）
   全部同口径。**这类 bug 不会让任何东西崩、也不会让测试翻红——它只是让数字偏乐观**，
   正是本项目最需要警惕的一类。

5. **【判据被自己的提示词污染】Q1 拿 `code_interpreter` 当"是否渲染工具声明"的
   标志物**，而训练提示自己就写满了这个词 → "不带 tools"对照档误报"含工具声明=是"。
   本机实测对照档 1219 字符 vs 带 tools 2274 字符（确实差一整个声明段）。
   改用 `CODE_TOOL` 描述里的短语（模板原样插入，跨模板可移植）。**凡"对照档必须
   为否"的判据，标志物必须只在被测那一侧出现**——否则对照形同虚设。

6. **顺手**：冒烟原本用 `17*23` 这种口算题 + `n_smoke` 条**同一 prompt**——base 直接
   心算就把答案说了，"调用率 ≥50%"这个判据被系统性低估。改为 5 道竞赛风格题轮转
   （`SMOKE_QUESTIONS`），并让 `--native_tool_style` 可钉进冒烟（钉死档与 auto
   不一致时两者都打印，**以 auto 高者判断"base 会不会调用"**，防止把"钉错形态"
   读成"base 不会用工具"）。

### 10.6.2 【最重要的一条】协议是对的，是**校验自己把正确的运行打死了**

**真机现象**（20 步跑，样本 25 第 5 段）：

```
ValueError: [protocol] 模板渲染的 prompt 与生成端实际喂给 vLLM 的 token 不一致
            （长度 1556 vs 1557，首个分歧位 1484）
```

这次**不是协议 bug，也不是探针 bug——是护栏误杀**。分析过程值得完整记下来。

**第一步：看数字，别看结论。** `1556 vs 1557` = canonical 比 prev **少一个 token**。
不是"某段特别长"（那会是几百 token 的差），而是**恰好一个**——这个量级只可能是
空白/标点类的单 token 差异。

**第二步：本机复现（Qwen2.5 真模板 + 真 tokenizer）。** 结论：

| 实验 | 结果 |
|---|---|
| 模板渲染 assistant `"  hello  "` | 原文**完整保留**（不 strip） |
| 模板渲染 assistant `"\n\nhello"` | 渲染成 `hello` ← **段首空白被吃掉** |
| `closing(C)` 对 9 种内容变体 | **2 种取值**——但差异只出现在"段首空白"变体上（前缀对不上） |
| `observation(C)` 对 9 种内容变体 | **1 种取值，逐 token 完全相同** ✅ |

**第三步：机制。** 两边对"同一段历史"的表示不同，且**各自都是对的**：

- `ctx_ids[i]`（我们拼接的）= vLLM 采样的**原始 token**，**必须原样保留**
  （那是模型自己吐的字节、也是它下一轮上下文里的真实内容；trim 掉会让 gen_logps
  与序列错位）；
- `msgs[i]`（模板重渲染的）= `asst_text.strip()` 后的文本 → 模板再渲染时
  段首空白没了。

于是校验①（逐 token 全等）**必然**在"某一段 assistant 以换行/空格开头"时报错，
且与"第几段"无关——真机第 5 段才炸，只是因为**前面的段恰好没以空白开头**。
换句话说：**这个 abort 迟早会发生，且越到后面越容易**（段数越多，命中概率越高）。

**第四步：为什么这比协议 bug 更危险。** 判据本身把语义搞错了——
"模板归一化"(strip) 被当成了 "生成/训练不同源"。而**真正的风险从来没有发生**：
`observation` 增量与 assistant 内容完全无关（上表第 4 行实测），这正是占位法成立
的根据，也是"token-in token-out 拼接"正确的根据。**协议一直是对的，是护栏把正确的
运行打死了**——若不修，这个 bug 会让整套方案看起来"上机就崩、方案不成立"。

**修复（`protocol.py`）**：校验① 改成**两级判据**——
1. 逐 token 全等 → 通过（原路径，零额外开销）；
2. 不等时，比较**去掉纯空白 token 后**的序列：相同 → 判为归一化差异，**容忍**
   （只告警一次，附差值 token 数）；不同 → **仍然 raise**，且报错里直接解码
   双方在分歧位附近的**文本**（不再只给数字——那是这次多花一小时的原因）。

实现细节：`_drop_ws_tokens` 按**单 token** 解码判"是否纯空白"并缓存
（Qwen byte-level BPE 下 ASCII 空白都是干净单 token；多字节字符被切开时单 token
解码成替换符 → 保守**保留**，故该判据只会漏容忍、不会错容忍）；
`tokenizer.decode` 失败 → 返回 None → **退回严格比较**（宁严不宽）。
**序列构造路径一个字没动**——容忍只作用于校验，序列永远是原始 token 拼接。

**测试（P2b 组，10 项）**——两个方向都锁死：
- 正例：段首 `\n` / `\n\n` / 空格 / 首尾空白 → 容忍，且**返回序列仍以原样 prev 开头**
  （容忍 ≠ trim，专门有一条断言）；
- 反例：可见字符不同 / 段中 A vs B / **tools 档不一致** / **`enable_thinking` 档
  不一致**（可见 think 段）/ decode 失败 → **全部仍然 raise**。
  最后一条尤其重要：docs/03 那次"eval 没传 `enable_thinking=False` → 烧穿预算 →
  fmt/acc 双灭"的事故档，绝不能被这条容忍静默放过。
- 端到端：5 段**每段都以换行开头**（最坏情况）→ 全过且逐 token 不改动采样 token
  （修复前第 2 段就炸）。

**突变测试（双向）**：删掉容忍 → P2b 失败 ✅；把容忍写成"无限容忍"（不看可见
内容）→ P2b + P3 双双失败 ✅。两个方向都有测试兜住。

**元教训（已入全局）**：
> **护栏的判据必须区分"表示层归一化"与"语义层不一致"。** 逐字节全等是最容易写、
> 也最容易把正确运行打死的判据——凡"两边各自渲染同一份历史"的比对，先问
> "这两条路径会不会对同一内容做等价的归一化（strip/大小写/Unicode 规范化）"。
> 另外：**报错信息要带解码后的文本，不只给 token id / 位置**——
> `1556 vs 1557 @1484` 三个数字花了很久才定位，而"一侧是 `\n\n` 一侧是空"
> 是一眼可见的。

**为什么这个 bug 没被 140 项 CPU 测试抓住**：MockTok 的 `strip_assistant` 与真实
模板行为一致（都会 strip），但**测试里从未让"拼接的 prev"与"模板渲染的 msgs"
在不 strip 档下并排比较**——即没有构造"原样采样 token 累积 vs strip 后重渲染"
这个**真机必然出现**的组合。已由 P2b 补上（用 `strip_assistant=False` 的实例造
生成端序列，与 `strip_assistant=True` 的模板对照）。
**这是"两个各自正确的行为组合起来才出问题"的典型**——单看每一侧都测过。

### 10.6.1 真机冒烟样本观察（判读用，16/16 全中）

```
样本0: "I'll solve this step by step.\n\nFirst, let me understand the condition:
        $n^2 + 1$ is divisible by $n + 1$.\n\nThis means $(n^2+1)/(n+1)$ should be an
        integer.\n\nLet me write a Python program to find all positive integers"
样本1: "I'll solve this equation step by step.\n\nGiven: $\\frac{1}{m} + \\frac{1}{n}
        = \\frac{1}{6}$ ...\n\nLet me manipulate this equation:\n$$...$$\n\nMultiply both si"
```

三点判读：

1. **模型是"先叙述、再调用"**（`Let me write a Python program...` 然后才吐调用块）
   ——与参考实现的节奏一致，说明 `native_stop_at_call=False`（默认）是对的：
   若强行在调用边界 stop，会把这段有用的推理切掉。
2. **`invalid=0` 很关键**：说明 base **调用后不再继续瞎写**（docs/09 §0.1 症状③
   在原生协议下的对应形态没有出现）→ `native_stop_at_call` 兜底开关**不需要开**，
   保持单变量纪律（先看基线行为）。
3. **`answer=0`**：16 条全部走了工具路径，没有一条"直接给答案跳过工具"。
   ⇒ base 在原生协议下的工具先验**不只高，而且稳**（对比 p11 围栏协议 ~48%
   且需 SFT 才有格式）。

> **注意**：以上三点是在**玩具提示**下观察到的；用正式提示复跑后应复核一遍
> （正式提示更长、更详细，可能改变"先叙述 vs 直接调用"的比例）。

**回归**：`pytest rlab/tests` → **83 passed**；`test_native_protocol` → 13 函数 / 162 项全过
（新增 5 项 native_probe 静态检查 + P2b 空白容忍组，两者都含**突变测试**验证：
把 Q2 判据改回 JSON 启发式、以及删掉空白容忍，对应检查都确实翻红）。
诊断工具与护栏的修复也**必须被测试锁住**——否则下次又是"探针/护栏说没事"。

**修正后的静态检查教训（本项目第二次）**：新加的 Q2 判据检查最初用纯文本包含
（`'"{" in ' not in np_`），结果被**我自己写的解释注释**误伤而翻红。改 **AST**
（只读真正被赋值的 `_real = [...]` 推导式）。与 §10.2 第 8 条同一类：
**凡"禁止某写法"的静态检查，一律读 AST，不读文本。**

### 10.7 尚未验证的部分（如实声明，2026-09-25 更新）

| 项 | 状态 | 怎么验 |
|---|---|---|
| Qwen3.5-4B 真实调用形态 | ✅ **已验证 = `function`**（§10.5） | — |
| `tools=` 模板渲染 | ✅ **已验证**（1748 vs 166 字符） | — |
| `enable_thinking` 共存 | ✅ **已验证**（开关生效，think 已闭合） | — |
| 拼接硬契约（本机 Qwen2.5） | ✅ **硬契约三条全过**；pod 上待复跑（§10.6 第 3 条修了判据） | 重跑步 ①（现在会打印 Q5 一行） |
| **base 的真实调用率** | ✅ **16/16 = 100%**（但首版用玩具提示，§10.5.1 已修；建议用正式提示复跑一次确认） | `python -m rlab.native_probe --n_smoke 16 --native_tool_style function` |
| **vLLM 端到端（`prompt_token_ids` + 工具模板）** | ⚠ **首次上机即命中护栏误杀**（§10.6.2，已修）——生成端已跑通到第 5 段 | 重跑步 ③：应能跑完 20 步 |
| 拼接校验①（真机运行条件） | ✅ **已被真机覆盖**：修前在"段首空白"处 abort；修后 5 段全过 | 见 §10.6.2 的 P2b 组 |
| EOS 停 vs `</tool_call>` 停的实际分布 | ⚠ 未验证 | 步 ③ 的 `invalid_final`/`ctx_full` 列 + 健康检查 `native_invalid`（冒烟 invalid=0 是好兆头） |
| `gpu_mem` 档位对结论的影响 | ⚠ 已知 +7.0pp 是引擎档效应 | eval 必须 `--gpu_mem 0.78` 与训练同档 |
| AIME25 OOD 集 | ❌ **仍未建**（docs/05 §6.6） | 与参考 +23.89pp 对话的前置条件 |



