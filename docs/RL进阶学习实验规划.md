# RL 进阶学习实验规划（基于本地 2×H20，不使用 PyTRIO）

> 制定日期：2026-09。前提：已完成 GRPO / DAPO / Reinforce++ 的学习与实测（见 `AGENTS.local.md` 与 `docs/RLHF实验学习报告.md`）。
> 目标：把 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 的算法内容逐篇在本地复现，**算法学到位、代码自己写**，不依赖 PyTRIO 远程训练框架。

---

## 0. 总原则

1. **不换轮子，升级轮子**：已有 `simple_grpo_v1`（GRPO/DAPO）和 `simple-reinforce++` 的"DeepSpeed 训练端 + vLLM 生成端 + 权重同步"双进程架构是核心资产，所有新实验都在它之上扩展，不从零搭。
2. **严格沿用已验证的实验方法学**：
   - 单变量实验按 **optimizer 更新数严格配对**；
   - 噪声地板 **±2pp**（双引擎对拍实测），>3pp 才算真差异；
   - 评测统一 GSM8K test N=300 seed=42（后续按任务换 NQ/HotpotQA 等，但协议不变：固定样本、固定 seed、vLLM 批量评测）。
3. **每阶段一个产出文档**：`docs/` 下每阶段写一份对比报告，记录超参、曲线、结论，防止"跑完就忘"。
4. **所有已知工程坑先写进骨架**（见附录 A），不再逐个重踩。

### 硬件与模型约定

| 项 | 约定 |
|---|---|
| GPU | 2×H20 96GB：0 号卡 vLLM 生成（gpu_memory_utilization 0.5~0.85），1 号卡训练（或 2 卡 ZeRO-2 训练 + 生成 worker 共驻 0 号卡，沿用现有模式） |
| 模型 | 主力 **Qwen2.5-3B-Instruct**（与已有结论可比）；阶段 2 起可加 **Qwen2.5-7B-Instruct** 做规模对照；阶段 4 蒸馏 teacher 用 7B/14B-Instruct |
| 框架 | DeepSpeed ZeRO-2 + vLLM 0.12，环境变量固定两条（附录 A） |

---

## 1. 阶段 0：统一实验骨架（约 1 周）

**目的**：把三个单文件脚本（`grpo_dapo.py`、`rf++_vllm_one.py`、`grpo_ref_split.py`）收敛成一套可插拔骨架，后续每个算法只改 loss 和 rollout。

**目录建议**：`simple_GRPO/rlab/`

```
rlab/
  config.py          # 全部超参集中
  protocol.py        # rollout 数据结构（messages / tool calls / mask 约定）
  data.py            # 数据集加载（GSM8K / DeepMath / NQ …）
  reward.py          # 各任务 reward 函数（acc / format / tool）
  rollout.py         # vLLM 生成 worker：单轮 / 多轮 / 工具调用统一接口
  sandbox.py         # 阶段 2 加入：代码执行沙箱
  search_backend.py  # 阶段 3 加入：本地检索后端
  train.py           # DeepSpeed 训练端（loss 可插拔）
  losses.py          # GRPO / DAPO / Dr.GRPO / CISPO / GSPO 全部实现
  sync.py            # 权重同步（沿用 rf++ 的 apply_model + fail-fast 方案）
  eval_vllm.py       # 复用现有评测器
  analysis.py        # 曲线/得分解析
```

**关键设计**：loss 可插拔——`GRPO_step()` 拆成 `compute_advantages()` + `compute_loss(policy_logps, ref_logps, gen_logps, advantages, mask, cfg)`，所有算法变体都只落在这一个函数里。

**验收**：
- [ ] 骨架跑通 GSM8K GRPO 基线，acc 复现 ~74（与历史 grpo200 一致）；
- [ ] DAPO 复现 ~78；
- [ ] eval 全自动化（训练完一键评测并追加到汇总表）。

---

## 2. 阶段 1：Loss 修正线 + GSPO（约 1~2 周）

**学什么**：对应 agentic-rl-lab 第 0、1、7 篇。核心是把"GRPO 的两个归一化偏差 → 各家怎么修"这条线吃透，且全部可用极小代码量在 `losses.py` 里实现。

| 实验 | 改动点 | 假设 |
|---|---|---|
| 1a. Dr.GRPO | advantage 组内不除 std；loss 按 token 求和后除**固定常数**（不除批内真实长度） | 消除长度偏置，抑制 RF++ 那种"越训越长/越训越差"的退化 |
| 1b. CISPO | ratio clip 只截断**上升方向**，被截断 token 保留 `min(ratio,1)*A` 梯度（再加可学习 bias 权重可选） | 小 clip 下保留更多梯度信号，小 batch 更稳 |
| 1c. GSPO | importance ratio 从 token 级改为 **sequence 级**：`s_i = exp(mean_log_ratio)`，clip 在序列级 | 训练更平稳；对比 DAPO 的 token-level clip 是否仍有优势 |

**实验设计**：
- 统一配置：batch32、kl 0.01、300 步，与历史 GRPO/DAPO 结果严格配对；
- 每个变体至少跑 1 次，最优者复跑 1 次确认（噪声地板 ±2pp）；
- 记录：acc 曲线、平均输出长度、gradient norm、（GSPO 单独记）序列级 ratio 分布。

**验收**：`docs/01-loss-variants.md`，给出"3B/小 batch 教学规模下，哪个 loss 最稳"的明确结论。

---

## 3. 阶段 2：ReTool 式代码交织 RL（TIR）（约 2~3 周）

**学什么**：对应第 5 篇。这是 `Auto_Program` 的正规化版本——把之前"exec() + 停句拼接"的玩具实现升级成协议化的多轮工具调用训练。

**核心新增**：
1. **多轮 rollout**（`rollout.py`）：生成 → 检测 ```python 代码块 → 沙箱执行 → 结果回填 → 续生成，最多 N 轮；沿用 Auto_Program 的递归续写思路，但轨迹结构改成显式多段（`protocol.py` 定义）。
2. **沙箱**（`sandbox.py`）：subprocess 隔离 + 超时（替代 signal.alarm 主线程限制）+ 内存限制；禁止网络。
3. **关键训练细节（自己推导并验证）**：
   - **工具返回 token 不进 loss**：completion_mask 只保留 assistant 生成的 token，沙箱输出段置 0——这是 TIR 训练最易错的点；
   - 多段轨迹的 logps 拼接：vLLM `prompt_logprobs` 按段取，训练端按段对齐；
   - reward 设计：acc（±1）+ format（±1）+ 代码可用率小权重；冷启动期与后期权重切换（沿用 Auto_Program 的 16 步切换策略并验证其必要性）。

**验收**：
- [ ] acc 相对 base 显著提升，且**代码调用率随训练上升**的曲线可见；
- [ ] 记录"工具 token mask 错/对"的对照小实验（这是本阶段最重要的学习点）；
- [ ] 产出 `docs/02-retool.md`。

---

## 4. 阶段 3：Search-R1 多轮搜索 RL（约 2~3 周）

**学什么**：对应第 3 篇。多轮 + 环境不可预测 + 检索质量影响 reward，是 agentic RL 的第二类工具。

**不依赖外部 API 的本地化方案**：
- 语料：wiki 语料子集（或 2018 Wikipedia dump 切片，5~10 万段足够教学规模）；
- 检索：BM25（`rank_bm25`，零 GPU）起步；有余力换 bge/e5 稠密检索（vLLM 之外再驻 0.3G 显存即可）；
- 协议：`<search>query</search>` → 返回 top-3 段落包在 `<information>` 中回填，最多 4 轮；
- 数据：NQ / HotpotQA train 子集，reward = 最终答案 EM + 格式。

**实验看点**：
- 检索次数随训练的变化（学会"少而准"还是"多而滥"）；
- 多轮轨迹与 TIR 轨迹在 mask / credit assignment 上的异同。

**验收**：`docs/03-search-r1.md`；EM 提升 + 检索行为分析图。

---

## 5. 阶段 4：蒸馏（OPSD / OPD）（约 2 周）

**学什么**：对应第 2、4 篇。换范式：Teacher 打分替代环境 reward。

**方案（2×H20 可行）**：
- Teacher：固定 checkpoint（阶段 1 最优模型，或 Qwen2.5-14B-Instruct 量化/单卡 bf16 推理）；
- OPSD 闭环：Student 自采样 k 条 → Teacher 对 Student 序列算 token logprob → 损失 = reverse KL（或 -teacher_logprob 加权）+ β·(teacher_logps−student_logps) 项 → 更新 Student；
- 显存编排：teacher vLLM 常驻 0 号卡，student 训练在 1 号卡；teacher 打分与 student 训练交替，无需训推同卡。

**验收**：`docs/04-opsd.md`；对比"RL 学的" vs "蒸馏学的" acc 与输出风格；小样本 C-EVAL 验证通用能力不崩。

---

## 6. 阶段 5：ALFWorld 长程 Agent（约 2~3 周，可选终点）

**学什么**：对应第 8 篇。从"答案对不对"到"任务完没完成"，credit assignment 难度跃迁。

- Linux 环境装 `alfworld==0.4.2` + TextWorld（训练机上做，Windows 本机只读代码）；
- 12K 长轨迹 → 显存预算：3B 模型 + ZeRO-2 + 梯度检查点，96GB 单卡可行，序列超长时分段计算 logps；
- reward：任务成功率（稀疏）+ 可选的子目标 shaped reward 对照实验（稀疏 vs shaped 的差异本身就是学习点）。

**验收**：`docs/05-alfworld.md`；成功率先升曲线 + 失败案例分析。

---

## 7. 可选支线（不排期，按兴趣插入）

- **TEMPO**：macro-step 优化 + 生成式 critic，做完阶段 5 后再看会更有体会；
- **Vision GRPO**：需要多模态模型（Qwen2.5-VL-3B），rollout 改图片输入，工程改动集中；
- **Harness-RL**（lab 第十篇预告）：与 agent harness 方向直接相关，发布后优先跟读。

---

## 8. 里程碑总表

| 阶段 | 内容 | 预计 | 硬验收 |
|---|---|---|---|
| 0 | 统一骨架 + 基线复现 | 1 周 | GRPO≈74 / DAPO≈78 复现 |
| 1 | Dr.GRPO / CISPO / GSPO | 1~2 周 | loss 变体对比报告 |
| 2 | ReTool 多轮代码 RL | 2~3 周 | 工具 mask 对照实验 + 代码调用率曲线 |
| 3 | Search-R1 本地检索 RL | 2~3 周 | EM 提升 + 检索行为分析 |
| 4 | OPSD/OPD 蒸馏 | 2 周 | RL vs 蒸馏对比 + 通用能力保持 |
| 5 | ALFWorld 长程（可选） | 2~3 周 | 成功率曲线 + 案例分析 |

总计约 10~14 周（业余节奏可拉长到一季）。

---

## 附录 A：开工前把已知坑写死在骨架里

以下全部来自本项目已踩实的教训（详见 `AGENTS.local.md`），骨架代码中直接内置：

1. **vLLM 权重同步环境**（缺一必卡死）：
   ```bash
   export VLLM_ALLOW_INSECURE_SERIALIZATION=1
   export VLLM_ENABLE_V1_MULTIPROCESSING=0
   ```
   判别：日志出现 `(EngineCore_DP0 pid=...)` = 错的 RPC 模式；必须看到 `model updated via apply_model`。
2. **math_verify 线程环境**：`parsing_timeout`/`timeout_seconds` 必须传 `None`（signal.alarm 仅主线程）。
3. **并发 `from_pretrained`**：load_lock 串行化 + `p.is_meta` 校验，防 cgroup RAM 顶爆出 meta 空壳。
4. **Qwen tokenizer.eos(151643) ≠ 模型停止词(151645 im_end)**；左 pad batch 解码按 Lmax 切片。
5. **评测标签唯一性**：results 字典键用路径末 3 段或显式 name，防同名 checkpoint 静默覆盖。
6. **eval 用 vLLM 多进程调度**（`eval_vllm.py`），不用 HF generate。
7. **RF++/单样本+全局基线的结构性退化教训**：任何新算法若 num_pre_Q<2，无法做组内对比——新实验一律保持组内 ≥4 条采样。
8. **实验配对**：跨批次评测有 ±1pp 数值噪声（bf16 归约顺序），同轮内配对比较才是硬对比。

## 附录 B：每个阶段的公共实验协议模板

```
数据: <任务对应 train set>   模型: Qwen2.5-3B-Instruct
训练: batch32, num_pre_Q=4~8, kl 0.01, lr 1e-6, 300 optimizer steps
评测: <任务 test set> N=300 seed=42, vLLM 批量贪心
对照: 与上一阶段最优 checkpoint 严格按更新数配对
判定: 差异 >3pp 记为真差异; 记录 acc/平均长度/格式率/工具行为
```
