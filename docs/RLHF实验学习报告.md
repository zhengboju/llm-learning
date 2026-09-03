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

## 2. 三个算法的实现与机制

### 2.1 GRPO（`simple_grpo_v1/grpo_ref_split.py`）

- 每题采 **num_pre_Q=4** 条回答，优势 = (r − 组均值)/组标准差 → **组内同题配对对比**；
- 生成进程内完成（`gen_model = engine`），无跨进程同步问题；
- 训练几何：micro batch=4（正好一组）× grad_accum 4 = 每次更新 16 条（4 题）；
- KL 挂在 loss 里（β=0.04）。

### 2.2 REINFORCE++（`simple-reinforce++/`）

- **num_pre_Q=1**：每题只采 1 条，优势 = 单条 reward − **macro batch 均值**（32 或 128 条不同题的全局标准化），不依赖组；
- KL 折进 reward（β=0.01）；
- vLLM 子进程生成 + `apply_model` 权重同步（每 2 次 optimizer 更新一次，每次 ~6GB state_dict）；
- 训练几何（batch32 版）：micro 4 × grad_accum 8 = 每次更新 32 条（32 题），300 步 = 37.5 次更新，共见 1200 题。

### 2.3 DAPO（`simple_grpo_v1/grpo_dapo.py`，基于 GRPO 改造）

四个机制（代码内全部有【DAPO】标注）：
1. **clip-higher**：上限解耦 1+0.28（下界 0.2 不变），保探索；
2. **dynamic sampling**：全对/全错组（组内无区分度）跳过重采，`skip_rate` 从 0% 涨到 40-60%——**首次量化"组信号枯竭"现象**（GRPO 原版只是静默丢弃）；
3. **token-level loss**：全 batch 按 token 归一化，替代样本级平均；
4. **overlong shaping**：448→512 token 区间线性扣分最多 1.0。实战案例：一个退化样本（答案尾部冒出 🍻 和伪造多轮对话）被扣满得 −3.0。

保留 β=0.04（论文为 0）以便与 GRPO 单一变量对比；200 步 22 分钟。

---

## 3. 主线实验与结果

### 3.1 GRPO 奖励黑客实验（reward 天平决定学什么）

| 奖励设置 | 结果 |
|---|---|
| 旧：correct=1.0, **format=1.25** | 格式合规 100%，但模型学会直接输出系统提示词占位符 `reasoning process here <answer>5</answer>` 骗格式分——**格式权重 > 正确性，模型优先学"最容易拿的分"** |
| 新：correct=**2.0**, format=1.0 + 占位符惩罚 | 黑客消失；格式部分回退换取正确率提升 |

教训：**RL 学的是 reward 的梯度方向，不是你的意图**。多个奖励项的相对权重 = 对"什么值得学"的排序。

### 3.2 四方总表（终版，HF 多线程引擎，N=300 seed=42）

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

### 3.3 REINFORCE++ 退化机制调查（本报告方法论的核心案例）

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

**定案**：`num_pre_Q=1` 时梯度无法区分"真解出"和"看起来像解出"（答对易题与蒙对难题拿到同样的正优势），表面模式被单调放大；GRPO 的组内 4 条同题配对对比是语义级信用分配，能把梯度钉在题目语义上。修复此病 = 改组基线 = 重写成 GRPO，**无调参解**。最佳实践 = 早停（~12 次更新）。

适用域声明：以上结论限 3B/小 batch/二值奖励的教学规模，不推翻 REINFORCE++ 论文在大模型大 batch 下的声明——**算法排名是规模相关的**，实测出的是理论适用域的边界。

### 3.4 训练动力学观察

- RF++ 训练速度（3min14s/300步）vs DAPO（22min/200步）：单样本 + vLLM 生成 vs 每题 4 条 + HF generate——rollout 数量与推理引擎决定墙钟；
- DAPO 的 skip_rate 曲线（0%→40-60%）= 组信号枯竭的量化：训练后期大部分组全对，dynamic sampling 把算力花在有区分度的组上；
- RF++ 生产-消费流水线（生成 128 条 ~10s ≈ 训练消费 ~11s）会出现周期性 `waiting for batch...`，属预期，勿误判为卡死。

---

## 4. 工程踩坑实录（现象 → 根因 → 修复）

### 4.1 RF++ 权重同步：vLLM 0.12 V1 引擎三连坑（最昂贵的一课）

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

### 4.2 评测脚本五连坑

1. **`<think>` 标签被聊天界面静默改写**（→ 普通文字 "thinking"/"response"）：含标签的 prompt/正则必须 grep 原始字节校验。曾经全部四轮 eval 用损坏 prompt 跑完，"fmt 35.7%" 之类结论全是伪影；
2. **标签同名覆盖**：results 字典用 basename 当键，3 个 `step_200` 同名互相覆盖静默丢结果 → 标签用路径末 3 段或显式 `name=path`，撞车加后缀；
3. **math_verify 多线程必炸**：默认 `parsing_timeout=5` 内部用 `signal.alarm`，只能在主线程用 → worker 线程抛 ValueError 被 `except` 吞成 acc 全 0。线程环境必须 `parsing_timeout=None` / `timeout_seconds=None`。教训：**"串行对照"若仍在 worker 线程里跑，对照无效——MainThread 才是关键变量**；
4. **左 pad 批量解码两坑**：切片一律按 Lmax（不是每题真实长度 Li，Qwen 的 im_end 距 prompt 尾仅 ~4 token，错切必截断）；`generate` 停止词以模型 generation_config 为准（Qwen tokenizer.eos=151643 ≠ 模型停止 151645 im_end），显式覆盖=拆刹车；
5. **并发 from_pretrained 顶爆 cgroup RAM**：6 路同时加载各物化 ~6G → safetensors 分配失败 → 权重留 meta 空壳 → "Cannot copy out of meta tensor"。加载 `load_lock` 串行化（生成仍并行）+ `p.is_meta` 加载后校验 + 失败即时报错。

### 4.3 吞吐：HF generate 的 util 天花板与 vLLM 解法

- 小模型贪心解码 = Python 逐 token 循环，**GIL 进程级串行** → 多线程多模型/大 batch 都只能 20-30% SM util；
- 正解 = vLLM continuous batching（调度在 C++/CUDA 层）：**单模型 300 题生成 5 秒**（~8000 toks/s），HF 版需 10-20 分钟；
- `eval_vllm.py` 多进程调度器：每模型一个子进程（vLLM 显存靠进程退出释放）、round-robin 分卡、错峰启动 + 失败重试（同卡多实例的内存剖析竞态：实例退出释放显存撞上另一实例初始化剖析 → AssertionError，重试必成）。

---

## 5. 方法论沉淀（可复用的实验纪律）

1. **配对评测**：N=300、seed=42 固定、greedy、同题同序——所有模型同卷；
2. **噪声地板先行**：换推理引擎/批处理方式对拍一遍，量出 ±2pp 的地板，之后的差异 >3pp 才解读；
3. **单变量铁律**：一次只动一个变量，且按真正的资源口径对齐（本例：batch128 实验按 optimizer 更新数配对，all_steps=1200 对齐 300 步 batch32 的 37.5 次更新）；
4. **假设-排除-取证循环**：先列假设 → 逐个单变量证伪 → 对最后的幸存者做样本级取证（直接看模型输出）——"算术表演"这种机制，不看样本永远猜不到；
5. **心跳与 fail-fast**：跨进程数据传递必须有接收端心跳打印 + 计数验收；宁可崩掉也不静默降级；
6. **checkpoint 按 eval 选，不按最后一步选**（RF++ 全部三个 run 的最优点都在 12.5u）。

---

## 6. 工具链与复现

### 6.1 文件地图

| 文件 | 用途 |
|---|---|
| `simple_grpo_v1/grpo_ref_split.py` | GRPO 主版本（4 采样组相对） |
| `simple_grpo_v1/grpo_dapo.py` | DAPO 版（四机制，【DAPO】标注） |
| `simple-reinforce++/rf++_vllm_one.py` + `config.py` + `sync_utils.py` | REINFORCE++（vLLM 生成 + 权重同步 + fail-fast） |
| `eval_gsm8k_test.py` | HF 版评测：`--tuned name=path,...`、`--workers`（线程并发）、`--gpus`、`--batch_size`、`--show`、`--split train/test` |
| `eval_vllm_one.py` / `eval_vllm.py` / `eval_merge.py` | vLLM 版评测：单模型 / 多进程调度器 / 合表 |
| `diag_format_fail.py` / `debug_batch.py` | 格式失败诊断 / 批量-串行对拍复现 |

### 6.2 checkpoint 地图（Pod）

- GRPO step_200（正确性优先版）：`/root/simple_GRPO/simple_grpo_v1/step_200`
- DAPO step_100/200：`/root/llm-learning/simple_grpo_v1/`
- RF++ batch128（最优点 step_400=12.5u=74.7）：`/root/llm-learning/simple-reinforce++/step_400/800/1200`
- RF++ KL0.04（覆盖了原 batch32/KL0.01 的 step_100/200/300）：`/root/llm-learning/simple-reinforce++/step_100/200/300`
- ⚠️ `~/simple_GRPO` 下旧副本已过期，一切以 `~/llm-learning`（git pull 后）为准

### 6.3 常用命令

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

## 7. 认知收获与下一步

**认知层**：
1. RL 学的是 reward 的梯度方向，不是你的意图——奖励项权重=学习优先级，占位符黑客是天平失衡的必然产物；
2. 信用分配质量决定 RL 的上限：组内同题配对对比（GRPO/DAPO）能维持语义级梯度信号，单样本+全局基线会滑向表面模式；
3. 算法排名规模相关：论文表格不能直接搬到小规模复现，实测的是理论适用域的边界；
4. 格式是 RL 的"容易目标"（~12 更新即满），真正的算法差异在正确率的可持续性上。

**下一步候选**：
1. MATH 数据集——GSM8K 上 DAPO 78% 已近天花板，更难的题才能继续拉开算法差距；
2. DAPO 续训/加步——skip_rate 40-60% 说明学习信号未枯竭，尚未到收敛；
3. McNemar 配对检验——eval 落每题结果 jsonl，把 +11.3pp 的显著性钉死。

---

*报告生成：2026-09-03。所有表格数据与结论的原始日志、诊断脚本均在 git 历史与本仓库中可追溯。*
