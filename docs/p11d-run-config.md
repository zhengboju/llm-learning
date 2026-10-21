# p11f 启动配置定版（p11d 草案的修订版，2026-10-20）

**上游**：[p11c 训练诊断](p11-diagnosis.md)（含 held-out 读数）
**下一步（未实现，勿混跑）**：[p11e token 效率](p11e-token-efficiency-plan.md)
**工具**：`python -m rlab.audit_config --from_run_info <run_info.json>`

> **本文取代 p11d 草案**（原草案的两条判据建立在后来证伪的统计假设上，见 §5.4）。
> 五处修订，其余全部沿用 §3 命令：
>
> | # | 修订 | 一句话依据 |
> |---|---|---|
> | 1 | **训练池改用"在线实测通过率"重建**（`rlab.passrate --merge_out`），**并把 band 放宽到 `0.0 1.0`** | 探针直方图实测：旧 band 是开区间，只保留 `{0.25,0.375,0.5}` = **193 行**（与 record 反解的 m≈210 互证）；而 `{0.625,0.75,0.875}` 那 609 行本身有梯度（0.98/0.90/0.66），被白白排除。放宽 ⇒ 课程 **193 → 850（×4.4）**（§1.1） |
> | 2 | **内嵌评测显式关难度过滤**（代码已改：`train.py::_run_inline_eval` 加 `--no-difficulty_filter`） | run_info 带着只覆盖训练池的表 → `--split test` 必落 0 题 → fail-fast → 该 ckpt 的 test 窗口直接变盲；且过滤后的 acc 是「条件于 band」的量，跨 run 不可比 |
> | 3 | **内嵌 `--eval_n 300`**（趋势/熔断），权威判定改为**离线完整 dev（N=1000）** | 原草案把 1000 放内嵌 = 8 次 ×1000 的评测预算，超时风险高且判据仍不落盘 |
> | 4 | **`[sw]` 零权重样本数进训练日志**（代码已改），作为第一条"立即停"判据的数据源 | 原判据「零梯度占比 ≠ 0」当时**没有数据源**：`analysis` 的 `zero_grad` 是重构列，不是 sw 本身 |
> | 5 | `--out_dir rlab_new/native_p11f_wiring` | 同签名重跑会静默覆盖 `step_N`（p11 踩过两次） |

---

## 0. 先别急着跑：用采样档把已有的 p11c 重测一遍（零训练成本，最高优先）

**这是整个项目此刻性价比最高的一步。** 在投下一轮 16.7h 训练之前，先回答一个
更根本的问题：**p11c 的 +2.0pp 到底是没有效果，还是我们测不出效果？**

### 0.1 为什么必须先做这一步：训练/评测的分布错配

p11c 训练用 `temperature=1.0` 采样（rollout 在采样分布上优化期望），但 held-out
用的是 **greedy（temp=0，取 argmax 众数）**。这是两个不同的分布：

- RL 优化的是**采样分布的期望正确率**——它先移动概率质量，把"答对"的概率
  从 0.4 推到 0.6，这一步**在 Average@N 里立刻可见，在 greedy 里几乎不可见**
  （argmax 要等到概率质量越过 0.5 才翻转）。
- docs/14 §4.11 实测：同一 ckpt，贪心档与采样档 acc 相差 **10pp 量级**；
  docs/05 §7.4：参考项目（agentic-rl-lab/05-retool）的增益**全在采样下显现**。

**⇒ 我们极可能一直用一把错的尺子量 RL。** p9 的 +3.8pp、p11c 的 +2.0pp 都
"不显著"，但那是 greedy 众数的读数；概率质量的真实移动可能一直被漏掉。

### 0.2 同时解决功效问题

| 评测档 | N | 检验 | MDE @80% 功效 |
|---|---|---|---|
| 贪心（现状） | 293 | McNemar | **6.3pp** ← 目标 +3pp 看不见 |
| **采样 val_n=8** | 300 | 配对均值 z | **~2.4pp** ← 能看清 +3pp |

采样档把 per-item 从 0/1 二值变成 8 次平均的连续量，SE 收缩 √8 ≈ 2.8×，
功效直接够到目标效应。**同一个实验，换对的尺子，从"测不出"变"测得出"。**

### 0.3 怎么做（一条命令，~几小时，不训练）

复用 p11c 已有的 ckpt，**BASE 与 step 全程同协议**（同 seed / 同 val_n /
同 gpu_mem / 同难度口径）：

```bash
python -m rlab.eval --algo retool_math \
  --base_path=./rlab_out/sft_budget_v3/epoch1 \
  --models step100=<p11c>/step_100 step200=<p11c>/step_200 \
           step300=<p11c>/step_300 step400=<p11c>/step_400 \
  --val_n 8 --no-difficulty_filter \
  --gpus 0,1 --gpu_mem 0.78
```

- `--val_n 8` 时温度/top_p **自动**取 1.0 / 0.7（参考项目口径），无需手传；
- BASE 无 run_info，会复用 step 的协议——**与训练采样分布对齐**；
- 判据改用**配对均值 z 检验**（`analysis.paired_test_auto` 对 val_n>1 自动分派），
  **不要再用 McNemar**（它把 8 次平均二值化成"全对率"，会假性显著，docs/14 §4.11）；
- 读数只能在本批内自比，**不可与贪心档的 79.2% 并列**（档位不同，相差可达 10pp）。

### 0.4 这一步如何决定后面所有事

| 采样档重测结果 | 结论 | 下一步 |
|---|---|---|
| step400 显著 > BASE（Δ≥2~3pp，z p<0.05） | **RL 一直有效，只是 greedy 测不出**——过去的"无效"是测量假象 | p11f 照旧跑（修三条接线放大效应），**判据改成采样档** |
| 四臂全平（含 val_n=8） | 坐实当前剂量/机制下 RL 真无信号 | p11f **必须叠加剂量**：GAS 8→4（updates 42→84，零额外 token），否则又白跑一轮 |

**为什么这步排最前**：它要么把 p11f 从"赌一把"变成"放大已确认的信号"，
要么告诉你"光修接线不够，得同时加剂量"——两种情况都让你**不会白花 16.7h**。

---

## 1. 本轮目标：把三条"设计好了但没接通"的接线真正跑起来

p11c 的 held-out（完整池 N293 聚合 / N292 配对，BASE = SFT v3 epoch1）：

| 模型 | acc% | fmt% | 条件精度% | assistant tokens | rounds | Δacc | McNemar |
|---|---:|---:|---:|---:|---:|---:|---|
| BASE | 79.2 | 93.5 | 84.7 | 2512 | 1.85 | — | — |
| step100 | 80.2 | 91.8 | 87.4 | 2740 | 1.86 | +1.0 | p=.736 b19/c16 |
| step200 | 76.5 | 93.2 | 82.1 | 2571 | 1.59 | −2.7 | p=.272 b22/c31 |
| step300 | 77.8 | 92.2 | 84.4 | 2598 | 1.47 | −1.4 | p=.590 b25/c30 |
| step400 | 81.2 | 91.1 | 89.1 | 2642 | 1.65 | +2.0 | p=.533 b23/c18 |

**诚实读数**：acc 未检出（+2.0pp，p=.533）、fmt 退 2.4pp、条件精度 +4.4pp（它是两个率的
**商**，fmt 一掉、被答的那部分就换人，商自己会动——不能当机制证据，也不能用 acc 的
不显著反过来判它"纯噪声"）。三者联合 = **这一步什么都没测出来**，而代价是
~19.6M 生成 token、~16.7h。

三个**已定位、可修、且从未真正生效**的接线：

| # | 接线 | p11c 实际状态 | 对两个目标的作用 |
|---|---|---|---|
| 1 | `sample_weight`（守卫 + tcw 下应恒 1） | **563/2696 ≈ 21% 样本整行清零**，且其中 96% 正是无 boxed 的 B/C 桶 | 模型**从未**收到"啰嗦→不交卷"的负梯度 → 同时解释 fmt 下滑与散文尾膨胀。**一条打两个目标** |
| 2 | `budget_target` 三信道同值 | 系统提示承诺 8192，`[budget]` 回包与 `_can` 判据仍漏真实 12288 | `avg_clen` 冲到 6056~8056；这是 token 涨的直接来源 |
| 3 | 训练池刻度 | 早期 **59.5%** 组 uniform → 丢弃 → 零梯度白生成 | 同样 400 微步，把已付钱却扔掉的梯度拿回来 |

本轮**不引入任何新奖励机制**（p11e 的效率项仍然不上）：三条都是"接通已有契约"，
归属清楚。

### 1.1 band 选低了：课程只有 193 题，本该是 850

**结论 1：课程只有 193 题——两条独立证据互证。** 探针表侧：旧 band 是开区间，
只保留 `{0.25, 0.375, 0.5}` = 57+67+69 = **193 行**；record 侧：445 次抽题只碰到
**186 道不同的题**，若池子是 2000 题，
随机抽 445 次应碰到约 420 道；反解 `m(1−e^(−445/m)) = 186` ⇒ **m ≈ 210**。
**193 对 210 只差 8%，池子大小定案**——**400 微步 × 8 条轨迹，
是在 ~200 道题上重复约两遍**。这是"RL 读不出效应"的第一顺位嫌疑：课程太小，
任何提升都会先变成对这批题的记忆。

**在线复测**（19 样本/题，比探针精确近 5 倍）给出

| 在线通过率 | 题数 | 占比 | 能否产生梯度 |
|---|---:|---:|---|
| ∈ (0.125, 0.625)（真混合） | 112 | 60.2% | 能 |
| ≥ 0.625（接近掌握） | 67 | 36.0% | p≈0.7 时组内全对率 5.8% → 多数仍能 |
| ≤ 0.125（死题） | 7 | 3.8% | 不能（全错组 → 零优势） |

**探针表直方图**（2000 行，实测 **k=8** ⇒ 通过率只能取 n/8）：

| 探针通过率 | 行数 | 占比 | k=8 一组里"有对有错"的概率 |
|---|---:|---:|---:|
| 1.0 | 1029 | 51.5% | 0（全对 → 零优势） |
| 0.875 | 351 | 17.6% | 0.66 |
| 0.75 | 155 | 7.8% | 0.90 |
| 0.625 | 103 | 5.2% | 0.98 |
| 0.5 | 69 | 3.5% | 0.99 |
| 0.375 | 67 | 3.4% | 0.98 |
| 0.25 | 57 | 2.9% | 0.90 |
| 0.125 | 48 | 2.4% | 0.66 |
| 0.0 | 121 | 6.1% | 0（全错 → 零优势） |

**结论 2：band 的位置选低了。** 表里 **69% 的题通过率 ≥ 0.875**（对 SFT v3 偏简单），
而旧 band 恰好只收最难的三个刻度，把 `{0.625, 0.75, 0.875}` 共 **609 行全部排除**
——它们根本不是废题（见上表第 4 列）。所以 `--difficulty_band 0.0 1.0` 不是
"放宽标准"，是回到 DAPO 原本的定义（accuracy neither 0 nor 1）：
**课程 193 → 850 题（×4.4）**。

**那 36% 怎么来的**（在线复测 67/186 落到 p≥0.625）：
它**不是探针刻度错，是选择偏差**：在池子均值（≈0.80）附近用 8 样本估计做筛选，
被选中的题真值必然被拉向均值上方——选定集的实测平均通过率 **0.48** vs 探针刻度
均值 **0.35**，正好是 +0.13 的回归量。**这张表本身没坏，坏的是 band 的位置。**

> **本节三次更正（2026-10-20，逐条留档，免得下一轮从错的起点出发）**
> 1. 初稿说"探针 `--k 4` 太粗、34% 误收率"——**实测 k=8**，该解释作废，换成
>    上面的"选择偏差 / 均值回归"。
> 2. 初稿说"探针绝对刻度差一大截、反解真实 p≈0.92"——**不成立**：整段 run 的
>    实测平均通过率是 **0.48**（19 样本/题），不是 0.92。
> 3. 由 2 反推：**p11c 早期 59.5% 的丢弃与"全对组"对不上**（按实测池子构成，
>    M_raw 本应 ≈90%）。另一来源待查（§3.0b 的检查）。注意 `record.jsonl`
>    **没有 `status` 字段**，组状态在 **`q_status`**。

**所以本轮做两件事**（都不是"换实验"）：

1. **用在线实测值重建那 186 行的刻度**（`rlab/passrate.py --merge_out`，纯观测，
   绝不动原表）——唯一能拿到 19 样本精度的途径；**已完成，覆盖 186 行**；
2. **在重建后的表上把 band 放宽到 `0.0 1.0`**（= DAPO "accuracy neither 0 nor 1"
   的本来定义）。三条理由：
   - 被它误排除的 609 行本身有梯度（0.98 / 0.90 / 0.66），排除它们纯属白扔课程；
   - 池子大小才是瓶颈（~210 题），宽 band 是**唯一不花算力就能扩池**的动作；
   - 宽 band 不会失控：题目级动态采样 `filter_question_pool(streak_max…)` 会把
     连续产出零方差组的题**自动拉黑跳过**，废题自己会被筛掉。

---

## 2. 相对 p11d 草案的配置改动

| # | 改动 | 类型 | 依据 |
|---|---|---|---|
| 1 | **删** `--tool_waste_penalty 0.20` | 纯配置 | guard+tcw 下被 `validate_guard_conflicts` 自动归零，留着只会在下次读日志时误导。废码负信用**完全由 `trunc_counts_as_wrong` 的 acc=−1 承担** |
| 2 | **加** `--budget_target 8192` | 纯配置 | 承诺紧、余量松；此前只靠提示文本，回包与判据仍漏出 12288 |
| 3 | **换** `--difficulty_path …_online.jsonl`，**并把 band 放宽到 `0.0 1.0`** | 纯配置 | §1.1：池子只有 ~210 题，宽 band 是唯一不花算力的扩池动作；重建行有真精度，废题由动态采样自动拉黑 |
| 4 | **改** 内嵌评测 | 代码+配置 | `--eval_n 300`；过滤由代码显式关闭（`train.py` 已改） |
| 5 | **换** `--out_dir rlab_new/native_p11f_wiring` | 纯配置 | 防同签名覆盖 `step_N` |

**其余全部保持不变**——包括 `--overlong_ref 8192 --overlong_buffer 2048`
（你的显式值是对的：触发线 6144，实测覆盖 56% 样本，是当前唯一有效的长度项）、
`--lr 1e-6 --steps 400 --seed 42`。

### 2.1 评测口径

| 用途 | 口径 | 为什么 |
|---|---|---|
| **权威判定** | 离线、**完整 dev**、`--no-difficulty_filter`、BASE 与 4 个 ckpt **同一命令同批** | 唯一跨 run 可比；同题配对检验成立 |
| **内嵌曲线** | `--eval_n 300`、两 split 都关过滤 | 只作趋势与熔断（防 p9/p10 那种"训完 11h 才发现深坑"）；口径已与 p11c 不同，**不与 p11c 的数字直接比** |

> `--no-difficulty_filter` 只关**评测侧**过滤；训练池仍由 `--difficulty_path` 控制。
> 课程决定"学什么"，评测口径决定"跟谁比"，两者互不影响。

---

## 3. 定版命令

### 3.0 前置（无 GPU，1 分钟）：重建训练池刻度

```bash
ls -d rlab_new/*p11c*          # 取 p11c 的 out_dir，下称 $P11C
python -m rlab.passrate \
  --record $P11C/record.jsonl \
  --difficulty_path rlab_out/difficulty_v3e1_train.jsonl \
  --difficulty_band 0.125 0.625 \
  --min_k 8 \
  --merge_out rlab_out/difficulty_v3e1_train_online.jsonl
```

- **把输出的三行「表过期诊断」抄进本轮记录**（表判 p≈0 但在线已能做对 / 表判 band 内
  但在线全错 / 表判 p≈1）——它是"池子刻度到底偏了多少"的唯一实测证据。
- `--min_k 8` 而不是默认 16：`passrate.py` 自己在"表诊断"里就用
  `在线 k < 8 → 不下结论` 作阈值，而一个 run 里每题只有 ~8~16 条在线样本，
  用 16 会把绝大多数题排除在回填之外。
- 在线没见过的题保留原表值，所以这一步**只会更准，不会更差**。

### 3.0b 丢弃来源待查（一条命令；决定要不要动 `max_context_tokens`）

`record.jsonl` **没有 `status` 字段**，组状态在 **`q_status`**。按前后段对比，
就能区分"早期丢弃 = 全对组"与"整组超限连坐"：

```bash
python -c "
import json, collections
rows=[]
for l in open('rlab_new/native_p11c_guard/record.jsonl',encoding='utf-8'):
    l=l.strip()
    if not l: continue
    try: r=json.loads(l)
    except Exception: continue
    if isinstance(r.get('acc'),list) and r['acc']: rows.append(r)
n=len(rows); w=max(1,n//4)
def rate(rs):
    k=sum(len(r['acc']) for r in rs)
    return sum(1 for r in rs for a in r['acc'] if a>0)/k
def uni(rs):
    return sum(1 for r in rs if all(a>0 for a in r['acc'])
               or all(a<=0 for a in r['acc']))/len(rs)
print('组数', n, 'q_status',
      collections.Counter(r.get('q_status') for r in rows).most_common())
print('前1/4 通过率 %.3f  uniform %.1f%%' % (rate(rows[:w]), uni(rows[:w])*100))
print('后1/4 通过率 %.3f  uniform %.1f%%' % (rate(rows[-w:]), uni(rows[-w:])*100))
"
```

**判据**：前 1/4 的 `uniform` 率 ≫ 后 1/4 ⇒ p11c 的"信号密度变好"其实是模型在
自己会做的题上被打散了（**不是进步**），这与 held-out 读不出效应一致；若 `uniform`
本身很低而 M_raw 仍低，则丢弃来自**整组超限连坐**，此时才该抬
`--max_context_tokens`（现为 13312 = 12288 保险丝 + 1024，prompt 稍长即触发）。

### 3.1 训练

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
  --difficulty_path rlab_out/difficulty_v3e1_train_online.jsonl \
  --difficulty_band 0.0 1.0 \
  --task_adv_guard --aux_clip_alpha 0.5 --trunc_counts_as_wrong \
  --max_context_tokens 13312 --traj_dump \
  --system_prompt_file rlab_out/sp_anchor8192.txt \
  --lr 1e-6 --seed 42 --steps 400 --save_steps 100 \
  --eval_during_training --eval_n 300 \
  --out_dir rlab_new/native_p11f_wiring
```

**删掉的三项**（相对你的原命令）：`--no-eval_during_training`（改为开启）、
`--code_w 0.05`（与 `code_attempt_w` 重复；code 率 98% 时两者都是零梯度）、
`--tool_waste_penalty 0.20`（被自动归零）。

### 3.2 权威判定（本轮唯一算数的读数）

```bash
python -m rlab.eval --algo retool_math \
  --base_path=./rlab_out/sft_budget_v3/epoch1 \
  --models step100=./rlab_new/native_p11f_wiring/step_100 \
           step200=./rlab_new/native_p11f_wiring/step_200 \
           step300=./rlab_new/native_p11f_wiring/step_300 \
           step400=./rlab_new/native_p11f_wiring/step_400 \
  --no-difficulty_filter \
  --gpus 0,1 --gpu_mem 0.2
```

⚠ `--gpu_mem` 必须给 **0.2**（对齐 `run_info.eval_gpu_mem`），否则同权重同题会差
+7.0pp（[docs/07](../docs/07-vllm-logprobs-non-determinism-4b.md) 已实测）。
跨 run 比较前先核对两边 `eval_protocol.difficulty_filter.mode` 都是 `off`。

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

**读法**：本轮奖励结构 = 1 个有效辅助项 + 1 个微弱效率项 + 1 个零成本保险 +
1 个 token 级成本项。每一项都能回答"它改不改变组内排序、改多少"。

---

## 5. 验收判据（写死，防事后解释）

### 5.1 主判据（离线完整 dev，同批同命令）

| 指标 | 判据 | 不接受 |
|---|---|---|
| **held-out acc** | Δacc ≥ **+3pp** 且同题配对 McNemar **p<0.05** | 训练窗口读数；只报 p 不报 b/c；用别的池子 |
| **b / c（不一致对数）** | **必须与 Δacc、p 一起报** | 只报 Δacc |
| **fmt、条件精度** | 与 acc **三者联合**报告 | 只看 acc 判"格式换推理"；用 acc 不显著反判条件精度为噪声 |
| **符号翻转** | 0（`sigstats` 的「正确性为正被压成非正」） | 有翻转还继续跑 |

> **不预先宣称 MDE 绝对阈值。** McNemar 的功效完全由不一致对数 `b+c` 与净差 `b−c`
> 决定：同一个 Δacc，`b+c` 从 40 变到 200，p 能差一个数量级。要谈功效就**事后用
> 实测的 b、c 反算**，不要用假设的 discordance 预先宣布"这个 N 能保证 +3pp"。
> N=1000 = dev 全量，是本项目能拿到的最大的同一个池子，已经用满。

### 5.2 过程判据（每 25 个 update 看一次）

| 指标 | 期望方向 | 判据 / 数据源 |
|---|---|---|
| **`[sw]` 零权重样本数** | **= 0** | 训练日志（本轮新增；`train.py` 每 10 步打印） |
| **M_raw = uploaded/attempts** | **≥ 90%** | 生成端日志；**必须同时读归因行**（`uniform` / `trajectory_overlong` / `prompt_overlong` / 未归因）。`retool_context_overlong` 是"**任一样本**超限 ⇒ **整组丢弃**"，所以 M_raw 低未必是废题多——可能是一两条啰嗦轨迹把整组 7 条好的连坐扔掉 |
| `avg_clen` | **下降**（`budget_target` 的直接靶） | 若仍 ≥7000，说明通告没压住 |
| `trunc` 率 | 不涨 | 涨 >3pp ⇒ 承诺太紧，抬 `budget_target` 到 10240 |
| B 桶（散文撞墙） | 下降 | `rlab.analysis` 的桶分布 |

### 5.3 三条"立即停"信号

1. **日志出现 `零权重样本 x/y` 且 x > 0** → sw 修复没接到（本轮最关键的接线）。
   注意：`analysis` 的 `zero_grad` 列是 `trunc_final or code_wasted` 的**重构**，
   修复之后这一族是**有**梯度的——能判定 sw 的只有上面那行日志。
2. **符号翻转 > 0** → doc15 §7 停线，说明 `aux_clip` 失效。
3. **`avg_clen` 不降反升（≥ p11c 的 6056）** → `budget_target` 没被模型接收。
   （三信道在代码层的一致性由 `test_budget_target_announcement` 锁死并 fail-fast，
   所以这里只需看模型行为，不需要手抄含标签的 grep。）

### 5.4 本文修正掉的四个不成立说法

原 p11d 草案里有四条**未经支撑**的定量主张，今后引用请以本节为准：

| 原说法 | 为什么不成立 | 现在的说法 |
|---|---|---|
| "N=293 的 MDE=6.3pp、N=1000 的 MDE=3.4pp" | 这是**在假设 discordance ≈15% 的前提**下算的，不是普适阈值；给读者的印象却是"绝对保证" | 见 §5.1：功效取决于实测 `b+c`，事后反算 |
| "p11c 的 42 updates 对应参数位移 ≈4.2e-5，而 SFT 是 268 updates @1e-5（≈2.7e-3）→ RL 只动了 SFT 的 1.6%" | 用 `lr × updates` 当位移是跨目标、跨优化器、跨 batch 的无效换算；且 268 updates 属 **v1** 档，未经 v3 核实 | 删除。剂量问题只按"是多少次更新"陈述 |
| "`--val_n 8` 采样档把配对检验 SE 降到 0.35×" | 只有**条件独立**的那个采样分量会按 1/√N 缩，总方差不会 | 删除。两种解码档不可并列比较（docs/14 §4.11） |
| "GAS 8→2 是零额外 token 的免费加剂量" | token 确实不变，但单步方差、有效批大小、混合组供给都变了，代价是真实的 | 见 §9.1，写成有代价的二选一 |

---

## 6. 保留但"此刻零梯度"的两项——为什么仍留着

| 项 | 当前状态 | 为什么留 |
|---|---|---|
| `code_attempt_w 0.05` + `code_shaping_once` | code 率 98% 时全组同值 → 零梯度 | **只在坏情况下付费的保险**：p6 曾把 code 率压到 3%。一旦组内有人 0 调用，它立刻成为真实信号 |
| `len_eff_w 0.15` | 2% 幅度（噪音级） | **通过侧唯一**的效率信号。目标含 token 效率，删了就只剩失败侧惩罚 |

---

## 7. 明确不做的事（防下一轮重复提议）

| 不做 | 原因 |
|---|---|
| **抬 `--lr` 加剂量** | p3 已实锤 5e-6 会把策略学崩（条件精度 82.4→55.6%、clen −67%）。加剂量的安全途径是增加 updates |
| 降 GAS 提更多 update（本轮） | 它改的是更新次数与单步方差，属第二变量；本轮先把三条接线接通（§9.1 是它的位置） |
| 上 p11e 的"全对组效率训练" | 会改变上传组人群构成，与本轮归属冲突 |
| 改 `adv_mode` 为 `group_std` | p9 定论：8 条组可用，但需单变量 A/B |
| 调 `aux_clip_alpha` | 它是当前"翻转 = 0"的结构来源 |
| 把 band 收窄回 `(0.125, 0.625)` | 实测只保留 193/2000 行，而被它排除的 `{0.625,0.75,0.875}` 共 609 行本身是有梯度的（组内有对有错的概率 0.98/0.90/0.66）——收窄 = 自愿把课程砍掉 4.4 倍 |
| 抬 `max_traj_tokens` | `budget_target` 已把"模型看见的预算"与保险丝解耦，抬保险丝只增加显存风险 |

---

## 8. 成本预期（用于判断何时该停）

| 项 | p11c 实测 | p11f 预期 | 依据 |
|---|---|---|---|
| token/update | 466k | **350~430k** | 丢弃率下降（§3.0）+ `avg_clen` 6056→约 4500~5500 |
| updates | 42 / 16.7h | 50 / 400 微步 | GAS 不变 |
| 累计 | 16.7h | **约 13~18h** | — |

**若 M_raw 仍 <70% 或 `avg_clen` 不降**：不要继续跑完，先查 §5.2 的两行——
那分别能区分"表没接上"与"通告没生效"。

---

## 9. 下一步（本轮跑完之后）

1. **p11f 判定**（§3.2）→ Δacc ≥+3pp 且条件精度不回退 ⇒ 接线定版。
2. **p11e**（全对组效率训练）→ 只在第 1 步兑现后开。
3. 剂量 A/B → 在 p11e 之后。

### 9.1 如果 p11f 又落在"未检出"

p11c 已经预演过一次：机制改了、行为动了、acc 读不出来。届时的正确动作是
**在同一接线、同一张表下只动剂量**，二选一（不要同时动）：

| 杠杆 | 代价 | 收益 | 前提 |
|---|---|---|---|
| `--steps 400 → 1000` | 生成 token ×2.5（~40h） | updates ×2.5 | 无条件可用 |
| `GAS 8 → 4`（steps 不变） | token 不变，但单步方差 ↑、有效批样本减半 | updates ×2 | **M_raw ≥ 90% 才可用**（§1.1：批变小 = 混合组供给变稀）；掉到 70% 以下就先抬 `--steps` |

**先看 p11f 的 M_raw**：它 ≥90% 说明池子刻度对了、有余量降 GAS；否则先抬 steps。

**任何一步不兑现，就停在那一步查，不要叠加下一步**——这是 p1~p11 全部失败的
共同形态（[docs/08 §8.1](08-sft-cold-start.md)：四个变量同时动，故障归因不可能收敛）。

## 10. 工具执行预算单变量对照（不重训）

`tool_execution_budget=0` 沿用原语义：工具执行预算跟随 `budget_target`，本轮有效值
为 8192。显式设为 `12288` **只改变 `_can` 的调用放行判据**；`budget_target=8192`、
`max_traj_tokens=12288`、`answer_reserve=1024` 均不变。回包仍按 8192 通告剩余额度，
包括原有“不会执行”的强警文案；这是有意保持提示不变的单变量对照，**不是把提示
或预算通告同步改成 12288**。默认回包估算 266 时，调用放行线从已用轨迹 6902
移到 10998 token（执行预算 − reserve − 回包估算）。

pod 更新代码后，仅评同一 step400，用新输出名避免覆盖原表：

```bash
git pull
python -m rlab.eval --algo retool_math --skip_base \
  --models step400_exec12288=./rlab_new/native_p11f_wiring/step_400 \
  --tool_execution_budget 12288 \
  --n 300 --seed 42 --no-difficulty_filter \
  --gpus 0 --gpu_mem 0.2 --dump_traj \
  --out eval_p11f_step400_exec12288.json

python -m rlab.analysis \
  --eval-json eval_p11f_step400_exec12288.json \
  --pair-json eval_vllm_all.json \
  --pair-a step400_exec12288 --pair-b step400
```

比较对象是原表的 `step400`，不是 BASE；同题、同 seed、同 greedy、同 `gpu_mem=0.2`。
联合看 acc / fmt / 条件精度 / token，并按落盘轨迹核对新增答对题是否在旧放行线之后
调用工具。仅 C 减少但 acc 不升，不能认定放宽恢复了有效解题。
