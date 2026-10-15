# 纯 RL 解题能力优化方案：正确性课程、探索与净增量验收

> 日期：2026-10-15；状态：**方案已落文档，新增控制接口尚未实现，尚无新实验结论**。
> 当前运行：`rlab_new/native_p10_best`（SFT v3 epoch1 → RL），不是
> [旧 p10 报告](p10-diagnosis.md)中的 `rlab_out/retool_math_p10`。
> 范围：只优化 RL，不加入 teacher 轨迹，不新增 SFT，不把蒸馏收益算成 RL 收益。

## 1. 目标与当前基线

**目标：在相同推理协议与资源约束下，使模型自行探索到的正确解法更稳定，并迁移到未见题。**

- 起点固定为 `rlab_out/sft_budget_v3/epoch1`；它已有 SFT/RL 历史，不能称作“从 base 开始的纯 RL”。
- 本轮“纯 RL”指起点之后不再引入 teacher 示范或监督蒸馏。验证器只给对错，不提供答案轨迹。
- 优化对象是 held-out 正确率与题级多采样成功率，而不是仅提高 boxed、代码执行率或总 reward。
- 保持锚提示 8192、`max_traj_tokens=12288`、`max_context_tokens=13312`、`answer_reserve=1024`。

### 1.1 已知读数（用户 pod 输出，未在本地复算原始数据）

| 模型 | held-out N | acc% | fmt% | acc/fmt% | 输出 token | 工具轮次 |
|---|---:|---:|---:|---:|---:|---:|
| SFT v2 final | 293 | 79.5 | 89.1 | 89.2 | 未提供 | 未提供 |
| SFT v3 epoch1（RL 起点） | 293 | 76.8 | 92.2 | 83.3 | 2622 | 1.80 |
| SFT v3 final | 293 | 76.8 | 92.2 | 83.3 | 2765 | 2.19 |

v3 final vs v2 final：McNemar `p=0.341`；v3 epoch1 vs v2 final：`p=0.410`。
**只能说未检出 acc 差异，不能说等效、没有退化或覆盖已带来收益。**
本次优先保留 v3 epoch1，是因第二个 epoch 未见总体指标收益、输出与工具轮次更高；
不推广为“所有广覆盖数据只需一个 epoch”。

截至当前 RL 记录：4800 条已上传轨迹，约600个题组/微步、约75次 optimizer update，
单会话 `[10-14 16:05:22 ~ 10-15 10:15:25]`，staleness 均值1.5、最大3。
训练计划1000微步、GAS=8，约125次 optimizer update，不能写成1000次参数更新。

| 连续题组窗口 | acc% | fmt% | acc/fmt% | avg_clen（含工具回包） | 工具轮次 |
|---|---:|---:|---:|---:|---:|
| 0–200 | 80.5 | 92.5 | 87.0 | 3112 | 2.01 |
| 200–400 | 82.3 | 93.6 | 88.0 | 3220 | 1.92 |
| 400–600 | 82.6 | 94.3 | 87.6 | 3282 | 1.87 |

上表按用户给出的等长窗口近似汇总；不同窗口题目不同，同题8条高度相关，不能当配对收益。
**暂见收尾和工具轮次改善，未见持续条件精度增长；held-out 净增量尚未评出。**

无 boxed：314/4800，其中 B 长推理截断192、C 终局废调用106，合计约95%。
正常末段中位数194 token；B 调过工具末段中位数4154；46条零调用 B2 全写到12288。
因此不先抬 reserve/总预算，也不先增加长度惩罚。
record 未包含全部失败 attempt；整组 overlong 的轨迹 dump 有记录，但部分 `acc=None`，不能算全错。

## 2. 先纠正判读边界

1. **p>0.05 不等于打平或安全**：它可能只是样本不足，报告效应方向与不确定性。
2. **acc/fmt 下降不是自动的“比例噪声”**：提交人群变化可使比值下降；acc 的不显著性
   不能否定条件精度变化。始终联合报告 acc/fmt/比值，必要时按题配对 bootstrap，
   以及两臂都提交题目的诊断结果（后者是选择性子集，不是主指标）。
3. **held-out 可以测覆盖带来的迁移**：dev 没被训练见过，正是验证广覆盖是否泛化的方式。
   train eval 只能辅助诊断记忆与过拟合；不能说“覆盖收益在 dev 结构上测不出来”。
4. **4000是 v3 采集预算，不是已确认的 RL 全池大小**：训练调用 `load_qas`；
   以启动日志中去重、剔除 dev 后的实际题数为准。v2 最终数据532题、v3最终数据3560题，
   覆盖约6.7倍；741是混合 RL dump 的题数，不能写成 `741→3563 = 6.7倍`。
5. 固定训练集探测不泄漏到训练梯度；但不能反复用 dev 挑课程、奖励和 checkpoint 后，
   仍把该 dev 当一次性的最终泛化证明。重要结论需要保留未调参的题集或额外评测集。

本节修正 [SFT 手册](08-sft-cold-start.md)与此前讨论中的过强结论。

## 3. RL 解题信号：为什么需要正确性边界题

8条独立同分布采样中出现有对有错的概率为：

`P(mixed) = 1 - p^8 - (1-p)^8`。

| 单次成功概率 p | P(mixed) |
|---|---:|
| 0.50 | 99.2% |
| 0.10 | 57.0% |
| 0.01 | 7.7% |

正确性混合组提供“这条解法正确、另一条错误”的直接信号；全对/全错组没有该对比。
但这不等于全对/全错组一定无梯度：**本项目真实判据是最终 advantage，而不是 acc**。

现有接线：

1. `retool_score_flat` 先算正确性±1，再加代码奖励、长度/效率项；
2. 在 `overlong_filter` 下排除截断/废调用样本后计算组内 advantage；
3. `tool_credit_advantages` 注入局部工具成本、废调用和尾部截断惩罚；
4. `group_ok` 对序列 advantage 检查变化，对 token advantage 检查是否存在非零值。

所以 **uploaded≠正确性混合组**。全错组也可能因辅助项或局部惩罚被上传。
`phase=cold` 不代表 math 分支还在使用旧的格式/代码冷启动加权配方；不能据此调
`reward_cold_w/reward_hot_w` 来改变当前 math outcome 信号。

## 4. 分阶段执行路线

### 阶段 A：先验收当前 p10，参数不动（现有接口可执行）

先把 SFT 起点、step200/400/600 用同一个**无难度筛选**的协议来源评测。
两张卡仍被训练占用时不要强行启动0.78显存占比的评测；待卡空闲后顺序运行。

```bash
# 在 ~/llm-learning；先确认这些 checkpoint 已保存，GPU0 已空闲。
PROTO=rlab_new/native_p10_best

CUDA_VISIBLE_DEVICES=0 python eval_vllm_one.py \
  --model rlab_out/sft_budget_v3/epoch1 \
  --proto_from "$PROTO" --algo retool_math --split test \
  --n 300 --seed 42 --gpu_mem 0.78 \
  --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
  --name sft_v3_start --out rlab_out/eval_p10_sft_start.json

for STEP in 200 400 600; do
  CUDA_VISIBLE_DEVICES=0 python eval_vllm_one.py \
    --model "rlab_new/native_p10_best/step_$STEP" \
    --proto_from "$PROTO" --algo retool_math --split test \
    --n 300 --seed 42 --gpu_mem 0.78 \
    --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
    --name "p10_s$STEP" --out "rlab_out/eval_p10_s$STEP.json"

  python -m rlab.analysis \
    --eval-json "rlab_out/eval_p10_s$STEP.json" \
    --pair-json rlab_out/eval_p10_sft_start.json \
    --pair-a "p10_s$STEP" --pair-b sft_v3_start
done
```

启动/结果核对：sp632ef7、同预算、同 `gpu_mem` 与引擎档、实际 N、同一套 qk；
`PROTO/run_info.json` 必须 `difficulty_path` 未启用，否则主评测样本会被筛。
若请求300实际只评293，如实报告；配对292也要报告并排查缺失/重复 qk，不能默默当293。
已有旧评测只有这些身份项全一致才复用。

若需要排查“偶尔正确→稳定正确”，在选定 checkpoint 与 SFT 起点上追加同题
`--val_n 8` 的 Average@8，显式 `--temperature 1.0 --top_p <训练实际值>`；
这是**8次平均正确率，不是 pass@8**，与 greedy 两张表分开，使用配对均值检验。
先用现有题集定位，最后保留独立题集/额外 seed 验证，避免反复挑点。

**判定**：若只有 fmt 上升而 acc/多采样正确率没有可信增长，下一轮重点改正确性课程，
不把 record 的81.8%与 SFT held-out76.8%相减，不盲目延长训练。

### 阶段 B：观测正确性有效组（先做，不改奖励）

现有 CPU 诊断：

```bash
python -m rlab.passrate --record rlab_new/native_p10_best/record.jsonl
python -m rlab.analysis \
  --record rlab_new/native_p10_best/record.jsonl \
  --no-boxed-breakdown rlab_new/native_p10_best/record.jsonl
```

`passrate` 累计跨权重版本的题级统计，且没有完整 overlong 结果；它是历史观测，
**不是当前 checkpoint 的固定次数通过率表**。现阶段不把 `--merge_out` 自动回填当课程控制。

下一步需增加下面四项观测，均按当前会话与最近200微步窗口统计：

- **M_raw**：已完成且可评分的同题组中，`0<n_correct<n` 的比例；
- **M_task**：应用当前任务 mask 后仍有对有错的比例，同时保留有效样本数分布；
- **M_uploaded**：已上传组中 M_task 的比例；另列全对/全错但有辅助梯度的组；
- attempts/uploaded/各类丢弃、实际生成 token、optimizer updates：报告
  `生成token/update`、`生成token/正确性混合上传组`。

组标识必须含 attempt id、qk、gen_version（同题同版本可多次 attempt，不能只用 qk+版本分组）。
准确记录评分前后任务 mask、纯正确性 advantage、辅助/局部 advantage 的幅度及同向/反向比例；
幅度比只是诊断，不冒充梯度因果分解。

若 M_uploaded 长期很低，说明大部分更新没有正确解法的对比；这是上课程的直接理由，
而不是“总 reward 已经上涨，所以正确性肯定在主导”。

### 阶段 C：正确性课程（先静态最小版，再动态混合版）

**探测对象**：阶段A选中的起点权重，原训练池去重且剔除dev；不给 teacher 解法。
每题固定 `k=8`、新 seed、相同温度/预算/工具协议。不能从成功即停的 v3 采集次数估 p。
首次选4000训练题控制成本；这是候选子池，不是整个 RL 池。

k=8 很粗：0/8 不能证明永远不会（独立二项模型下一侧95%上界约31%）；8/8也不等于稳定掌握。
因此分桶只用于资源调度，不能当成题目难度真值。

**最小可用课程**：训练启动时 `--difficulty_path TABLE --difficulty_band 0.1 0.8`，
按现有开区间语义只留 `1/8..6/8`。未探测题会被丢弃，必须打印候选/保留/缺失题数。
这是“静态边界子池”，**不是下面的动态75/15/10混采**，不能把二者写成已实现同一功能。

**目标动态课程（初始工程配比，非理论最优已证）**：

| 桶 | 探测正确数/8 | 初始采样占比 | 用途 |
|---|---|---:|---|
| 边界 | 1–6 | 75% | 主体正确性对比信号 |
| 容易/维护 | 7–8 | 15% | 防遗忘、保留短正确解法 |
| 困难/探索 | 0 | 10% | 给旧失败题重新入场机会 |

- 桶内按题均衡，不按轨迹行数加权；配比可配置并落 run_info，不由验证集自动寻优。
- 每200微步（约25 updates）检查统计；只有任务通过率分布实际漂移才刷新，不机械全池重探。
- 用最新 checkpoint 对固定训练候选小集重探，保持同题固定 k；至少包含困难桶和维护桶，
  不能仅在当前边界池内更新，否则永久看不到被筛掉题的进步。
- 将探测/刷新开销计入总训练成本，初始控制在总生成 token 的约10%内；不够则延长刷新间隔。
- 难度表使用权重/协议指纹与独立输出路径；旧表保留，分阶段开始时显式加载新表。
- 若从 checkpoint 重新启动，当前入口只承诺加载权重，不能称作 optimizer/scheduler 精确续训；
  对照必须使用相同阶段重启规则，避免将优化器重置收益归给课程。

### 阶段 D：困难题的有限探索，不训练“猜得短”

0/8题仅对预算允许的随机子集再取**固定16次**、不同seed的探索样本，仍用当前策略和验证器。
此批不成功即停，避免通过率估计的选择偏差；计入探索成本。

- 一旦有正确解法，把题纳入边界候选，以当前策略重新生成训练组；不要把额外16次
  离线轨迹直接标成“当前 on-policy 组”或偷偷塞入 SFT。
- 仍全错：暂降权，但保留10%探索入口；后续新 checkpoint 再试，不能永久判为不可学。
- 起始保持温度1.0和训练的 top_p/top_k。只有确有路径重复、且正确率未被破坏时才试
  提高探索性；温度更高不保证解法更丰富。
- 全错的非截断轨迹不应凭“短/代码执行成功”获得正确性正优势；结构性局部惩罚仍可
  教模型避免浪费，但单独统计，不能记为“已经学会解题”。

这里的探索成本是低成功率 RL 的真实难点，不承诺仅靠终局奖励补齐所有缺失知识。

### 阶段 E：正确性优先的奖励/更新，不先暴力删辅助项

先完成B的信号诊断，再决定是否进入奖励调整版。当前 math 的正确性±1已是标量奖励主体；
问题是**组内差异**可能来自其他分量，不能只比较配置权重。

目标实现：

1. `A_task` 仅由可验证正确性计算，沿用并显式记录任务 mask；只在有效混合组学习对错对比。
2. 长度/代码辅助项独立成 `A_aux`，不先混进 reward 再与正确性共用标准化；
   正确但较长的解法仍必须获得正确性强化，禁止辅助项翻转其任务优势符号。
3. 代码执行成功只说明代码能跑，不说明解法正确；本阶段拟将 `code_attempt_w/code_w` 置0，
   但**只有新优势接线验证后才执行**，不直接在当前run上关闭以免引发工具行为退化。
4. 非结构性效率优化作为正确解法之间的次级偏好：全错组不因长短差异获得任务正优势；
   保留 waste/trunc 的局部负约束，初始强度不增加。
5. 保留 KL/裁剪/旧策略修正，监测更新幅度与探索熵；不预设降低 KL、提高温度或增大学习率。
   只有观测到过度约束/严重漂移才调整对应项，且 train/ref 两端参数必须一致。

该更新是需要梯度探针的新实现，不是现有 CLI 权重拼接就能兑现。

## 5. 接口现状与最小开发清单

| 优先级 | 任务 | 已存在部分 | 尚缺/验收要求 |
|---|---|---|---|
| P0 | 固定全量 held-out | eval 的 `--proto_from`、per-item与配对分析 | 难度课程会自动筛dev；主评测明确绕过课程筛选，固定qk集合 |
| P0 | 正确性有效组观测 | record acc数组、q_status、gen_version；traj dump含丢弃 | attempt id、任务mask、纯任务/辅助优势、完整attempt/token计数 |
| P1 | 同档固定k探测 | `probe_difficulty` k/双卡/断点与预算CLI | 增加 `--from_run_info`；继承native_stop_at_call、采样/引擎全档；训练集候选契约与权重指纹 |
| P1 | 静态课程 | `--difficulty_path` + 开区间 `--difficulty_band` | 当前只告警指纹不符，应先核对并严格阻断混表；保留/缺失比例落盘 |
| P2 | 动态75/15/10混采 | QuestionScheduler在线拉黑/TTL | acc分桶、带配比的采样、退役/回流、可复现小集重探 |
| P3 | 有限16次探索 | 多采样rollout基础设施 | 桶内新seed、固定批、成本上限与回流，不混成常规8条组 |
| P3 | 正确性与辅助优势解耦 | 现有局部tool credit | A_task/A_aux分离接线、符号/梯度/归一化探针 |

**特别限制**：当前 `probe_difficulty` 没有 `--from_run_info` 或 `--native_stop_at_call` CLI。
现有代码用preset，再手动覆盖部分参数；当前训练却显式开了调用边界硬停。
此外现有探针把train+dev合成候选再截max_questions，不能把它叫作“4000训练题”。
因此先补P1，不能用旧探针省略参数产出表后宣称已与当前训练同档。

以下是**P1完成后的计划接口**，当前版本不可直接执行；不是此次文档提交已增加的CLI：

```bash
# 计划接口：--from_run_info 尚待开发。MODEL由阶段A选定；每卡输出必须不同。
MODEL=rlab_out/sft_budget_v3/epoch1
PROTO=rlab_new/native_p10_best/run_info.json

for GPU in 0 1; do
  CUDA_VISIBLE_DEVICES=$GPU python -m rlab.probe_difficulty \
    --model_path "$MODEL" --from_run_info "$PROTO" \
    --k 8 --max_questions 4000 --questions_per_wave 16 --seed 42 \
    --shard_index "$GPU" --shard_count 2 --gpu_mem 0.78 \
    --out "rlab_out/p11_difficulty.$GPU.jsonl" &
done
wait
# 合并前需核对两job退出成功、题目无重叠、同权重同协议、每题k=8且train∩dev为空。
cat rlab_out/p11_difficulty.0.jsonl rlab_out/p11_difficulty.1.jsonl \
  > rlab_out/p11_difficulty.jsonl
```

## 6. 下一轮训练的最小改动与计算预算

P0/P1就绪后先跑静态正确性课程：

- 从阶段A选定的起点权重新开 `rlab_new/native_p11_curriculum`，保留p10所有产物。
- 复用本轮已验证的启动命令，只更换位置参数模型、输出目录，并加：
  `--difficulty_path rlab_out/p11_difficulty.jsonl --difficulty_band 0.1 0.8`。
- `--num_pre_Q 8 --gradient_accumulation_steps 8 --lr 1e-6`，初始温度1.0，
  预算/锚/停止协议/生成与训练卡位不变；首段200微步（约25updates）作为产能检查，
  不凭这个短段做泛化结论。累计训练预算参考当前1000微步（约125updates）。
- **不要现在改奖励、样本数和课程三套东西一起找原因**。推荐的最终组合包括三者，
  但开发按P0→P1→P2/P3接线，每个阶段明确它兑现了什么。
- 两臂比较既报告实际updates，也报告全生成token/探测token/GPU小时；不能通过大量
  丢组或预探测获得更多隐藏算力，然后只说“都是1000steps，成本相同”。
- 两个训练seed建议42/43；held-out的抽题seed固定42，两臂都相同。

卡位以真实进程为准：`TRAIN_GPU`决定训练，`REF_GPU`决定ref server，生成卡由
`--gen_device`/配置决定，**不自动跟随REF_GPU**。若重排卡位必须三者一起核对；
不要只改REF_GPU或照搬脚本注释。gen_gpu_mem保持当前已验证值，不把0.78评测值搬到共居训练卡。

## 7. 验收：怎样才算 RL 提升解题能力

| 层次 | 要求 | 不能替代它的指标 |
|---|---|---|
| 产能 | M_task/M_uploaded、全错/全对、成本真实可见，接线无异常 | uploaded高、总reward高 |
| 稳定解题 | 冻结训练诊断题从低成功率向更高Average@8移动；混合题被掌握 | 当前窗口训练acc涨 |
| 泛化 | 固定完整held-out、相同推理预算，RL vs SFT起点acc有可信正增量，额外seed方向一致 | 只选难度带后的test高分、fmt涨 |
| 机制诊断 | acc、fmt、acc/fmt及两臂都提交子集联合报告；新增解对/遗忘题可追溯 | 单看商或不显著p值 |
| 有限token目标 | 相同上限下acc提升；报告平均/尾部输出token与工具轮次 | 只报告轮次下降 |

主指标仍是完整held-out的平均正确率。将初始通过率分桶固定下来后，分层报告
“原低成功率题是否提升”；分层阈值不能由本轮结果重新挑。若诊断用到了dev标签分层，
明确其探索性质，最终验证另用未调参题集。

若M_task明显改善但held-out没有改善，说明产能优化兑现、能力迁移未兑现；
若fmt改善但多采样正确率不改善，说明仍主要在学提交；
若错误/短轨迹出现正任务优势或正确信号被辅助项翻转，立即停并修接线。
对N≈293的小差异不承诺检出，不能用p>0.05证明安全或等效，也不靠追加几条采样冒充增加独立题数。

## 8. 执行顺序与当前停止点

1. **现在**：当前p10 checkpoint与SFT起点做阶段A全量held-out对照；CPU看历史通过率。
2. **下一项开发**：P0正确性有效组/完整成本观测 + P1同档固定k训练池探测。
3. **第一轮课程实验**：现有difficulty过滤的静态边界子池，统一主评测和成本口径。
4. **课程兑现后**：动态混采/困难题回流，随后正确性与辅助优势解耦；只加已验证需要的部分。
5. 当前运行未完成上述held-out验收；未宣布p10解题能力提升、p11新配置有效或teacher补题完成。

上游：[SFT起点与锚机制](08-sft-cold-start.md)、[p9归属诊断](p9-diagnosis.md)。
