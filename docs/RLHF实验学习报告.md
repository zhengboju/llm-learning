# RLHF 实验学习报告：GRPO → REINFORCE++ → DAPO

> 基于复旦 simple_GRPO 教学代码的完整实验记录（Qwen2.5-3B，GSM8K，2×H20 Pod）。
> 时间跨度：2026-09-02 ~ 09-03。所有结论均标注证据强度与适用域。
> 配套代码：本仓库 `simple_grpo_v1/`（GRPO/DAPO）、`simple-reinforce++/`（RF++）、`eval_*` 工具链。

---

## 0. 一页总结（TL;DR）

| 算法 | 最优 test acc* | 格式率 | 结局 |
|---|---|---|---|
| BASE（Qwen2.5-3B） | 66.7% | 55.7% | 基准 |
| **DAPO（clip-higher + dynamic sampling + token-level loss + overlong shaping）** | **78.0%** | 99.7% | **全场最优，+11.3pp** |
| GRPO（组相对，奖励正确性优先） | 74.0% | 99.3% | 稳定但收益有限 |
| REINFORCE++（batch128@12.5u，早停） | 74.7% | 99.7% | 早期与 GRPO 打平，此后结构性退化 |
| REINFORCE++（batch32，300步） | 72.3%（@12.5u）→ 67.3% | 99.3% | 单调滑坡 |

\* GSM8K test，N=300，seed=42，greedy，同题配对比较。评测噪声地板 ±2pp（双推理引擎对拍实测），**>3pp 的差异才当真**。

三条主线结论（贯穿主题：**信用分配 credit assignment**）：
1. **GRPO**：奖励天平决定模型学什么——格式权重高于正确性时，模型学会"骗格式"（奖励黑客）；组内 4 条同题对比是维持语义级信用分配的关键机制。
2. **REINFORCE++**：单样本 + 全局批基线在教学规模下存在**结构性退化**——约 12.5 次 optimizer 更新见顶后，模型坍缩到"表面解题模式"，换 batch 大小和 KL 强度都救不了（逐一单变量证伪）。
3. **DAPO**：在 GRPO 基础上的四个机制（clip-higher / dynamic sampling / token-level loss / overlong shaping）全部可观测到独立作用，是本轮唯一取得大幅正确率增益的算法。

理论部分见 §2（从 Policy Gradient 推到三个算法），实现见 §3，实验见 §4。

---

## 1. 实验环境

| 项 | 值 |
|---|---|
| 硬件 | k8s Pod，2× NVIDIA H20 96G；**cgroup RAM limit ≈60GB**（重要约束，多次踩坑根源） |
| 模型 | Qwen2.5-3B（base），HuggingFace 网络不通 → ModelScope 镜像 |
| 关键依赖 | torch 2.8+cu12.8、transformers 4.57.3、vLLM 0.12.0、deepspeed stage0、math_verify |
| 数据 | GSM8K train 7473 题（训练）/ test 1319 题（held-out 评测，train 有泄漏禁用于评测） |
| 布局 | GRPO/DAPO：GPU0 训练+生成（进程内），GPU1 ref_server；RF++：GPU0 ref_server+vLLM 生成，GPU1 训练 |

显存关键约束：3B stage0 全放 GPU 峰值 ~60G（7B 在此 Pod 必 OOM）；不开 CPU offload（避免 RAM 超 60G 被 -9 杀）与 pin_memory（容器锁页限制）。

---

## 2. 算法理论基础：从 Policy Gradient 到三个算法

### 2.1 问题设定：RLVR（可验证奖励的强化学习）

本项目属于 RLHF 的简化形态——**RLVR**（Reinforcement Learning with Verifiable Rewards）：奖励由可编程的规则给出（数学答案比对 + 格式正则），而非人类偏好或奖励模型。

- 提示 $x \sim \mathcal{D}$（GSM8K 题目），策略 $\pi_\theta$ 自回归生成回答 $y \sim \pi_\theta(\cdot|x)$；
- 标量奖励 $r(x,y)$：GRPO/DAPO 为 `3=格式对且答案对 / 0=格式对但答错 / −2=格式失败`，RF++ 为 `2 / 0.5 / −2`（DAPO 另有超长线性扣分，最多 −1）；
- 优化目标（带参考策略约束）：

$$\max_\theta \; J(\theta) = \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta}\big[r(x,y)\big] - \beta\,\mathbb{D}_{KL}\big(\pi_\theta \,\|\, \pi_{ref}\big)$$

$\pi_{ref}$ 是训练起点（base 模型），KL 项防止策略漂离初始语言能力——但注意 §4.3 的实验会表明：**KL 的强弱不是退化与否的决定变量**，它锚的是"离 base 多远"，锚不住"学到的模式好不好"。

### 2.2 Policy Gradient 与 REINFORCE：基线理论（全篇最重要的定理）

**Policy Gradient 定理**：

$$\nabla_\theta J = \mathbb{E}_{(x,y)\sim\pi_\theta}\big[\nabla_\theta \log \pi_\theta(y\,|\,x)\cdot A(x,y)\big]$$

梯度把"对数概率的提升量"正比于优势 $A$——**回答被采样到的概率，沿"优势为正"的方向被推高**。整个 GRPO/REINFORCE++/DAPO 的差异，全部在"$A$ 怎么估"这一件事上。

**REINFORCE** 用蒙特卡洛回报当 $A$，最朴素的取法 $A = r$。问题是方差极大（不同题难度不同、同题不同答质量不同，全混在 $r$ 里）。经典解法是**减基线（baseline）**：

$$\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(y|x)\cdot (r - b)\big]$$

**基线定理**：只要 $b$ 与所采样的 $y$ 无关（常数、或只依赖 $x$、或依赖 batch 里其他样本），梯度**期望不变（无偏）**，只改变方差。选得好方差显著下降。

> **这是理解三个算法分野的钥匙**：baseline 不改变"在优化什么"，只决定"梯度信号的信噪比、新鲜度与信用分配粒度"。后面所有实验现象——GRPO 稳定、RF++ 滑坡、DAPO 增益——都是这句话的具体化。

信用分配的粒度由此决定：梯度对整条序列的每个 token 同加同减（$A$ 是序列级的标量），"这条回答里哪一步推理真正导向了正确答案"无法直接分辨——只能靠**对比结构**间接逼近。

### 2.3 PPO：重要性比率与裁剪（GRPO/DAPO 的骨架）

策略更新后 rollouts 就过时了（off-policy），用重要性比率修正：

$$\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}, \qquad L = \mathbb{E}\Big[\min\big(\rho_t A,\; \text{clip}(\rho_t, 1{-}\varepsilon, 1{+}\varepsilon)\, A\big)\Big]$$

clip 的本质是**信赖域**：单步更新幅度被硬性限制，防止一次过冲摧毁策略。GRPO 和 DAPO 的 per-token loss 骨架完全继承这里；DAPO 的 clip-higher 只改了上界（见 §2.6）。

### 2.4 GRPO：组相对优化（DeepSeekMath）

对每道题采 $G$ 条回答（本项目 $G{=}4$），**用组内统计做基线**：

$$A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G) + 10^{-4}}$$

理论性质：
- **难度对齐**：组内样本共享同一道题，均值基线完美扣除题目难度——全局基线做不到（不同题的 $\bar r$ 混着难度差异）；
- **同题配对对比**：4 条回答里"哪条真的通向正确答案"被直接比较——这是语义级信用分配的来源；
- **无需 value network**（对比 PPO 需要 Critic）：省掉约一半模型显存，教学代码选它的现实原因；
- 已知缺陷：①全对/全错组 $A\equiv 0$，零梯度，rollout 白采（我们实测 skip_rate 0%→40-60%）；②std 归一化在低方差组会放大噪声；③难易题的更新权重被组内方差隐性重加权。

### 2.5 REINFORCE++：批级基线 + KL-in-reward

REINFORCE++（Li et al. 2025）去掉组结构，**用整个 macro batch 的标准化做基线**：

$$A_i = \frac{r_i - \mathrm{mean}_{batch}(r)}{\mathrm{std}_{batch}(r)}$$

理论收益：
- baseline 样本量从 $G$（4~16）扩大到整个 batch（论文中数百~上千），均值估计方差 $\propto 1/N$ 下降；
- 无组 → 每题只采 1 条，同样的 rollout 预算能见 $G$ 倍数量的**不同题**（梯度多样性）；
- 全对组不再浪费（不同题混在一起，batch 内方差几乎不可能为 0）；
- KL 作为 per-token 惩罚**加进 reward**（而非像 GRPO 挂在 loss 上），统一进 advantage 管道。

**隐含前提：batch 足够大**。batch 小 → ①均值估计本身噪声大；②不同题的难度差异全部残留在 $r_i - \bar r$ 里（全局基线只扣"平均难度"，扣不掉"这道题的难度"）。论文的实验规模（大模型 + 数百级 batch）满足前提；**教学规模 batch=32 时不满足——这正是 §4.3 实验的切入口**：我们实测放大 batch 不但没救，反而加速了退化，说明该规模下还有比基线方差更上游的病灶（单样本信用分配的缺失）。

### 2.6 DAPO：四个机制的动机

1. **clip-higher（§2.3 上界解耦）**：标准 PPO 的上界 $1{+}\varepsilon$ 对正优势 token 是概率天花板——低概率的探索性 token 一旦被抬升、ratio 冲过 $1{+}\varepsilon$，梯度即消失，永远起不来 → 熵持续衰减 → 策略坍缩。DAPO 把上界放宽到 $1{+}0.28$（下界 $1{-}0.2$ 不变，防止负优势 token 概率暴跌），让低概率 token 有成长空间。实测旁证：GRPO 训练后期 skip_rate 飙升 + 输出趋同，DAPO 全程未见此象。
2. **dynamic sampling**：零方差组梯度恒为零（§2.4 缺陷①），持续采样纯浪费 → 过滤后重采直到攒满有效组。量化收获：skip_rate 从 0% 升到 40-60%，等于给"组信号枯竭"装上了仪表盘。
3. **token-level loss**：样本级平均用 $1/|y_i|$ 归一化，长序列的单 token 权重被稀释——而 RL 训练后期回答普遍变长（推理链变长恰恰是质量上升的方向），token-level 把全 batch 的 token 拉平等权，长回答的贡献不再被系统性低估。
4. **overlong shaping**：被最大长度截断的样本若简单给 0 分/负分，"截断"与"内容质量"两种信号混在一起无法学；DAPO 在长度窗口内线性扣分（本项目 448→512 token，最多扣 1.0），把"太长"变成连续可优化的信号。实战案例：一个尾部冒出 🍻 并伪造多轮对话的退化样本被扣满得 −3.0。

### 2.7 统一视角：信用分配的粒度谱系

| 算法 | 对比结构 | 信用分配粒度 | 成本 |
|---|---|---|---|
| PPO | learned critic 逐 token | 最细 | 最贵（多一个 value net） |
| GRPO/DAPO | 组内同题 4 条对比 | 题目级（语义化） | 中（G 倍采样） |
| REINFORCE++ | 全 batch 跨题对比 | 数据集级（最粗） | 最省（1 倍采样） |

**对比的"近邻程度"决定信号质量**：越近（同题），难度混杂越少、对比越语义化；越远（全局），越省算力但难度混杂与方差越大。我们实测的教学规模排序——GRPO 稳（74.0 保持 50 次更新）、RF++ 12.5u 见顶后滑坡、DAPO 协同最优 78.0——与该谱系完全一致。这不是"谁的理论更对"，而是**信用分配粒度与训练规模的匹配问题**（详见 §4.3 的适用域声明）。

---

## 3. 三个算法的实现与机制（理论落点见 §2 对应小节）

### 3.1 GRPO（`simple_grpo_v1/grpo_ref_split.py`）

- 每题采 **num_pre_Q=4** 条回答，优势 = (r − 组均值)/组标准差（§2.4）；
- 生成进程内完成（`gen_model = engine`），无跨进程同步问题；
- 训练几何：micro batch=4（正好一组）× grad_accum 4 = 每次更新 16 条（4 题）；
- KL 挂在 loss 里（β=0.04，§2.1 的约束项形式）。

### 3.2 REINFORCE++（`simple-reinforce++/`）

- **num_pre_Q=1**：每题只采 1 条，优势 = 单条 reward − **macro batch 均值**（32 或 128 条不同题的全局标准化，§2.5），不依赖组；
- KL 折进 reward（β=0.01，§2.5 的 reward-shaping 形式）；
- vLLM 子进程生成 + `apply_model` 权重同步（每 2 次 optimizer 更新一次，每次 ~6GB state_dict）；
- 训练几何（batch32 版）：micro 4 × grad_accum 8 = 每次更新 32 条（32 题），300 步 = 37.5 次更新，共见 1200 题。

### 3.3 DAPO（`simple_grpo_v1/grpo_dapo.py`，基于 GRPO 改造）

四个机制（代码内全部有【DAPO】标注，理论动机见 §2.6）：
1. **clip-higher**：上限解耦 1+0.28（下界 0.2 不变）；
2. **dynamic sampling**：全对/全错组跳过重采，`skip_rate` 从 0% 涨到 40-60%；
3. **token-level loss**：全 batch 按 token 归一化，替代样本级平均；
4. **overlong shaping**：448→512 token 区间线性扣分最多 1.0。

保留 β=0.04（论文为 0）以便与 GRPO 单一变量对比；200 步 22 分钟。

---

## 4. 主线实验与结果

### 4.1 GRPO 奖励黑客实验（reward 天平决定学什么）

| 奖励设置 | 结果 |
|---|---|
| 旧：correct=1.0, **format=1.25** | 格式合规 100%，但模型学会直接输出系统提示词占位符 `reasoning process here <answer>5</answer>` 骗格式分——**格式权重 > 正确性，模型优先学"最容易拿的分"** |
| 新：correct=**2.0**, format=1.0 + 占位符惩罚 | 黑客消失；格式部分回退换取正确率提升 |

教训：**RL 学的是 reward 的梯度方向，不是你的意图**。多个奖励项的相对权重 = 对"什么值得学"的排序（§2.2 的 $A$ 直接乘在 $\nabla\log\pi$ 上，天平即方向）。

### 4.2 四方总表（终版，HF 多线程引擎，N=300 seed=42）

| 模型 | acc | fmt | 双达标 |
|---|---|---|---|
| BASE | 66.7 | 55.7 | 35.7 |
| grpo200 | 74.0 | 99.3 | 74.0 |
| **dapo200** | **78.0** | 99.7 | **78.0** |
| rfpp100（=12.5 更新） | 72.3 | 98.7 | 71.3 |
| rfpp200（=25 更新） | 70.7 | 99.3 | 70.7 |
| rfpp300（=37.5 更新） | 67.3 | 99.3 | 67.3 |

要点：
- **DAPO +11.3pp** 远超噪声，是全部实验中唯一的大幅正确率增益；
- **格式是"容易"目标**：BASE 在显式格式指令下本就有 ~55% 合规，四种 RL 全部学到 ~99%——算法差异全部体现在 acc 上；
- 双引擎对拍（vLLM vs HF，同 300 题同 greedy）：RL 模型 acc 差 ≤2pp、grpo200 两位小数全同；**BASE 的 fmt 差 6.4pp**（格式踩线模型对 kernel 数值敏感）。>3pp 才是真差异。

### 4.3 REINFORCE++ 退化机制调查（本报告方法论的核心案例）

**现象**：任何配置下 acc 均在 ~12.5 次 optimizer 更新见顶后单调下滑（格式保持 96-99%）：

| run | 12.5u | 25u | 37.5u |
|---|---|---|---|
| batch32 / KL0.01 | 72.3 | 70.7 | 67.3 |
| batch128 / KL0.01 | **74.7** | 71.3 | **57.7（跌破 BASE）** |
| batch32 / KL0.04 | 71.3 | 69.7 | 61.0 |

**排除法（每次只动一个变量，按 optimizer 更新数严格配对）**：
1. ~~批基线方差~~ → batch128 早期更强（74.7，RF++ 史上最高）但崩更狠——证伪；
2. ~~KL 锚太弱/漂移失控~~ → β=0.04 更差且格式被锚拽崩到 96.3%——若退化是漂离 base，加锚应变好；证伪；
3. ~~训练集过优化~~ → train-split 诊断（`--split train`）：训练集 acc 70-75.7，无记忆签名（过优化应冲 90%+）——证伪；
4. ✓ **单样本 + 全局基线的结构性退化**（样本级取证定案，见下）。

**样本级取证**（对退化 checkpoint 直接 `--show` 看输出）：
- 答错的题全是**"算术表演"**：格式完美、算术流畅、逻辑脱锚——漏读条件（总数差 100）、同一式子除两次不同除数、量纲错配；
- 两个独立退化的模型**同题同错且文本几乎逐字相同**；
- b128_1200 的正确题三连同款开头 "I will first calculate..."——**模板坍缩实锤**。

**定案（与 §2.7 谱系互证）**：`num_pre_Q=1` 时梯度无法区分"真解出"和"看起来像解出"（答对易题与蒙对难题拿到同样的正优势，§2.2 的序列级信用分配 + 跨题基线 = 表面模式被单调放大）；GRPO 的组内 4 条同题配对对比是语义级信用分配，能把梯度钉在题目语义上。修复此病 = 改组基线 = 重写成 GRPO，**无调参解**。最佳实践 = 早停（~12 次更新）。

适用域声明：以上结论限 3B/小 batch/二值奖励的教学规模，不推翻 REINFORCE++ 论文在大模型大 batch 下的声明——**算法排名是规模相关的**，实测出的是理论适用域的边界（§2.5 的隐含前提在教学规模不成立）。

### 4.4 训练动力学观察

- RF++ 训练速度（3min14s/300步）vs DAPO（22min/200步）：单样本 + vLLM 生成 vs 每题 4 条 + HF generate——rollout 数量与推理引擎决定墙钟；
- DAPO 的 skip_rate 曲线（0%→40-60%）= 组信号枯竭的量化（§2.4 缺陷①）：训练后期大部分组全对，dynamic sampling 把算力花在有区分度的组上；
- RF++ 生产-消费流水线（生成 128 条 ~10s ≈ 训练消费 ~11s）会出现周期性 `waiting for batch...`，属预期，勿误判为卡死。

---

## 5. 工程踩坑实录（现象 → 根因 → 修复）

### 5.1 RF++ 权重同步：vLLM 0.12 V1 引擎三连坑（最昂贵的一课）

**第一轮废跑（300 步 reward 纹丝不动）**：bare except 吞掉一切异常，生成器静默冻结在 base 权重。三个叠加 bug：
1. `llm_engine.model_executor` 属性路径是 V0（0.6/0.7）时代的，V1 把 EngineCore 拆成独立进程 → AttributeError 被吞；
2. `apply_model` 的 RPC 用 msgspec 编码，lambda 不可序列化 → 需 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` + 顶层函数 + `functools.partial`；
3. `state_dict.items()` 视图（odict_items）不可 pickle → 包 `list(...)`。

**验证方法**：grep 接收端心跳打印（`model updated via apply_model` ≥ 期望次数）+ 对比首尾 rollouts。**同进程架构（生成器=训练引擎）天然免疫此类问题**——GRPO/DAPO 从未得过此病。

**第二轮卡死（batch128 第一次同步 stall）**：RPC 模式下 6GB state_dict 过控制通道（本为小消息设计）偶发 stall，且 worker 线程异常要等主线程 `fut.result()` 才浮出。**grep 实锤**：成功 run 的日志无 `EngineCore_DP0` 前缀 = in-process 模式；卡死 run 有 = RPC 模式。

**最终解（钉死进启动命令）**：
```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # EngineCore 进程内运行，无 RPC 无序列化，故障面归零
```

### 5.2 评测脚本五连坑

1. **`<think>` 标签被聊天界面静默改写**（→ 普通文字 "thinking"/"response"）：含标签的 prompt/正则必须 grep 原始字节校验。曾经全部四轮 eval 用损坏 prompt 跑完，"fmt 35.7%" 之类结论全是伪影；
2. **标签同名覆盖**：results 字典用 basename 当键，3 个 `step_200` 同名互相覆盖静默丢结果 → 标签用路径末 3 段或显式 `name=path`，撞车加后缀；
3. **math_verify 多线程必炸**：默认 `parsing_timeout=5` 内部用 `signal.alarm`，只能在主线程用 → worker 线程抛 ValueError 被 `except` 吞成 acc 全 0。线程环境必须 `parsing_timeout=None` / `timeout_seconds=None`。教训：**"串行对照"若仍在 worker 线程里跑，对照无效——MainThread 才是关键变量**；
4. **左 pad 批量解码两坑**：切片一律按 Lmax（不是每题真实长度 Li，Qwen 的 im_end 距 prompt 尾仅 ~4 token，错切必截断）；`generate` 停止词以模型 generation_config 为准（Qwen tokenizer.eos=151643 ≠ 模型停止 151645 im_end），显式覆盖=拆刹车；
5. **并发 from_pretrained 顶爆 cgroup RAM**：6 路同时加载各物化 ~6G → safetensors 分配失败 → 权重留 meta 空壳 → "Cannot copy out of meta tensor"。加载 `load_lock` 串行化（生成仍并行）+ `p.is_meta` 加载后校验 + 失败即时报错。

### 5.3 吞吐：HF generate 的 util 天花板与 vLLM 解法

- 小模型贪心解码 = Python 逐 token 循环，**GIL 进程级串行** → 多线程多模型/大 batch 都只能 20-30% SM util；
- 正解 = vLLM continuous batching（调度在 C++/CUDA 层）：**单模型 300 题生成 5 秒**（~8000 toks/s），HF 版需 10-20 分钟；
- `eval_vllm.py` 多进程调度器：每模型一个子进程（vLLM 显存靠进程退出释放）、round-robin 分卡、错峰启动 + 失败重试（同卡多实例的内存剖析竞态：实例退出释放显存撞上另一实例初始化剖析 → AssertionError，重试必成）。

---

## 6. 方法论沉淀（可复用的实验纪律）

1. **配对评测**：N=300、seed=42 固定、greedy、同题同序——所有模型同卷；
2. **噪声地板先行**：换推理引擎/批处理方式对拍一遍，量出 ±2pp 的地板，之后的差异 >3pp 才解读；
3. **单变量铁律**：一次只动一个变量，且按真正的资源口径对齐（本例：batch128 实验按 optimizer 更新数配对，all_steps=1200 对齐 300 步 batch32 的 37.5 次更新）；
4. **假设-排除-取证循环**：先列假设 → 逐个单变量证伪 → 对最后的幸存者做样本级取证（直接看模型输出）——"算术表演"这种机制，不看样本永远猜不到；
5. **心跳与 fail-fast**：跨进程数据传递必须有接收端心跳打印 + 计数验收；宁可崩掉也不静默降级；
6. **checkpoint 按 eval 选，不按最后一步选**（RF++ 全部三个 run 的最优点都在 12.5u）。

---

## 7. 工具链与复现

### 7.1 文件地图

| 文件 | 用途 |
|---|---|
| `simple_grpo_v1/grpo_ref_split.py` | GRPO 主版本（4 采样组相对） |
| `simple_grpo_v1/grpo_dapo.py` | DAPO 版（四机制，【DAPO】标注） |
| `simple-reinforce++/rf++_vllm_one.py` + `config.py` + `sync_utils.py` | REINFORCE++（vLLM 生成 + 权重同步 + fail-fast） |
| `eval_gsm8k_test.py` | HF 版评测：`--tuned name=path,...`、`--workers`（线程并发）、`--gpus`、`--batch_size`、`--show`、`--split train/test` |
| `eval_vllm_one.py` / `eval_vllm.py` / `eval_merge.py` | vLLM 版评测：单模型 / 多进程调度器 / 合表 |
| `diag_format_fail.py` / `debug_batch.py` | 格式失败诊断 / 批量-串行对拍复现 |

### 7.2 checkpoint 地图（Pod）

- GRPO step_200（正确性优先版）：`/root/simple_GRPO/simple_grpo_v1/step_200`
- DAPO step_100/200：`/root/llm-learning/simple_grpo_v1/`
- RF++ batch128（最优点 step_400=12.5u=74.7）：`/root/llm-learning/simple-reinforce++/step_400/800/1200`
- RF++ KL0.04（覆盖了原 batch32/KL0.01 的 step_100/200/300）：`/root/llm-learning/simple-reinforce++/step_100/200/300`
- ⚠️ `~/simple_GRPO` 下旧副本已过期，一切以 `~/llm-learning`（git pull 后）为准

### 7.3 常用命令

```bash
# GRPO/DAPO 训练（GPU0 训练+生成，GPU1 ref_server）
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 29500 grpo_dapo.py &   # 另一终端先起 ref_server 于 GPU1

# RF++ 训练（两条 env 必带！）
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
CUDA_VISIBLE_DEVICES=0 python ref_server.py &
CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 29503 rf++_vllm_one.py

# 评测（vLLM 多进程，一条命令全部模型 + BASE，~5 分钟）
python eval_vllm.py --n 300 --gpus 0,1 --per_gpu 2 --gpu_mem 0.3 \
  --tuned name1=/abs/path/ckpt1,name2=/abs/path/ckpt2

# 过拟合诊断（训练集内抽样）
python eval_vllm.py --n 300 --split train --gpus 0,1 ...
```

---

## 8. 认知收获与下一步

**认知层**：
1. RL 学的是 reward 的梯度方向，不是你的意图——奖励项权重=学习优先级，占位符黑客是天平失衡的必然产物（§4.1）；
2. 信用分配质量决定 RL 的上限：组内同题配对对比（GRPO/DAPO）能维持语义级梯度信号，单样本+全局基线会滑向表面模式（§2.7、§4.3）；
3. 算法排名规模相关：论文表格不能直接搬到小规模复现，实测的是理论适用域的边界（§2.5 隐含前提）；
4. 格式是 RL 的"容易目标"（~12 更新即满），真正的算法差异在正确率的可持续性上。

**下一步候选**：
1. MATH 数据集——GSM8K 上 DAPO 78% 已近天花板，更难的题才能继续拉开算法差距；
2. DAPO 续训/加步——skip_rate 40-60% 说明学习信号未枯竭，尚未到收敛；
3. McNemar 配对检验——eval 落每题结果 jsonl，把 +11.3pp 的显著性钉死。

---

*报告生成：2026-09-03。所有表格数据与结论的原始日志、诊断脚本均在 git 历史与本仓库中可追溯。*
