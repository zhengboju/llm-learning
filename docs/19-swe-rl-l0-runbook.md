# SWE-RL L0 运行手册（docs/17 的落地实现）

> 状态：**L0 代码已落地并通过 CPU 回归（58 项 L0 + 全仓 198 项）**，
> **待 pod 真机验收**。
> 前置：[docs/17-swe-rl-plan.md](17-swe-rl-plan.md)（方案）、
> [docs/18-search-grpo-runbook.md](18-search-grpo-runbook.md)（同构的上一项目）。
> 本文只写「怎么跑、跑之前核什么、看到什么算对」——不重复方案论证。

---

## 0. 一句话架构

```
prepare_swe_l0（一次，CPU）        train.py --algo swe_l0（GPU）
  种子 + 变异 → 真跑测试          单轮生成 patch ──┐
  → 实测 F2P/P2P → train/dev.jsonl                │
                                                  ▼
                              extract_patch_text → git apply → pytest(F2P 全过 且 P2P 不回归)
                                                  │
                                     reward ±1 ───┴──→ GRPO(group_mean/sample_mean)
```

**L0 是单轮 patch-only**（docs/17 §2.5 定版）：模型读 issue + **已定位的局部代码**
+ 测试，直接输出一个 diff，**没有工具调用**。所以它不进多轮框架、不需要沙箱轮次，
工程量比 Search-GRPO 小得多——但奖励是**真实测试执行**，比 math/检索贵 2~3 个数量级。

---

## 1. 交付清单（本次新增/改动）

| 文件 | 作用 |
|---|---|
| `rlab/swe_l0_problems.py` | **8 个手写种子**（正确实现 + 注入 bug + 单测 + 自然语言 issue），覆盖边界/算术/可变对象/解析/列表/区间/校验/递归 8 类 |
| `rlab/swe_l0_synth.py` | **合成 bug 生成器**：AST 定位 + 文本拼接做单点语义变异（7 类算子），产物必须仍是合法 python |
| `rlab/swe_l0.py` | 核心库：patch 路径消毒/校验（`git apply --check` + 实 apply）、pytest 执行器（JUnit XML）、F2P/P2P 判分、组内 patch 去重、数据加载与 held-out 契约 |
| `rlab/prepare_swe_l0.py` | 池子生成：**执行验收**（原始码全绿 + buggy 至少一条红）+ 实测 F2P/P2P + 按源种子分组切 train/dev + 落盘复读自证 |
| `rlab/tests/test_swe_l0_cpu.py` | 53 项零件回归（P/G/T/E/C 五组，**真 git apply + 真 pytest**） |
| `rlab/tests/test_swe_l0_e2e.py` | 5 项接缝回归（真判分走完 `score_group_swe_l0`，含组内去重与 P2P 回归） |
| `rlab/config.py` | `swe_l0` preset + 4 个判分键 + `_validate_swe_l0_config` 九条 fail-fast |
| `rlab/rollout.py` | `extract_patch_text`（围栏/裸 diff 提取）、`score_group_swe_l0`（判分+组内去重+advantage）、record 附加键 |
| `rlab/train.py` | `--swe_test_timeout/mem_mb/workers/work_root`；签名 `-swt/-swp`；run_info 回读键 |
| `rlab/data.py` | `load_qas("swe_l0")` 分派 + 难度表指纹新增 `swe_test_timeout` |
| `rlab/losses.py` | `swe_l0` 走 GRPO 同形 loss（group_mean + sample_mean） |
| `eval_vllm_one.py` / `rlab/eval.py` / `eval_vllm.py` | `--eval_task swe_l0` 同档评测（同判分执行器、同超时口径）+ 条件修复率落盘 |

**兼容性**：`data_task` 缺省仍是 `gsm8k`、`tool_protocol` 缺省仍是 `fence`——
**p1–p11 / 原生 retool_math / search_qa 的全部行为逐字不变**（198 项回归含历史对照位）。

---

## 2. 上机步骤

### 步骤 1：生成 instance 池（CPU，一次性）

```bash
# 默认：8 个手写种子 + 汇编变异（每个种子最多不限条，按变异族轮转取样）
python -m rlab.prepare_swe_l0
# 产物：rlab/datasets/swe_l0/{train.jsonl, dev.jsonl, manifest.json}

# 池子太小/太大时：
python -m rlab.prepare_swe_l0 --output_dir rlab/datasets/swe_l0_v2 \
    --synth_per_seed 6 --dev_ratio 0.2 --force
```

脚本会：跑「原始码 + 全部测试」要求**全绿**（种子自证）→ 跑「buggy + 全部测试」
要求**≥1 条红**（否则该 bug 未被覆盖，剔除）→ 记录实测 F2P/P2P → 按**源种子**
分组切分（同种子的变体不跨集）→ 落盘后**复读**核对 id/Q 不相交、f2p 非空。

> ⚠ 本机实测（8 种子）：手写 8 条 + 合成 40 条通过、13 条剔除
> （10 条"bug 未被测试覆盖" + 3 条"P2P 为空"）→ train 29 / dev 19。
> **dev 只有 19 个 instance**：L0 的效应读数精度天生比大池子差，报告里必须写明
> （docs/17 §4 的 N≥300 是 L2 级协议）。
> 要放大池子就加种子（`swe_l0_problems.py`）——合成变异只能在同一批实现上变，
> 加种子才是真正扩分布。

### 步骤 2：base 格式率探针（**go/no-go 第一项**）

L0 的第一个要看的数据不是"修复率"，是**模型会不会输出可 apply 的 diff**
（docs/17 §1 把格式错误率列为主要瓶颈）。先用 20 步小跑看 `apply_ok`：

```bash
bash rlab/run_gsm8k.sh swe_l0 /root/Qwen3.5-4B \
    --out_dir rlab_out/swe_l0_probe --steps 20 --save_steps 20 \
    --num_pre_Q 8 --temperature 1.0 \
    --attn_implementation flash_attention_2 --optim_8bit \
    --micro_rows 1 --gen_gpu_mem 0.30 --seed 42
```

> ⚠ **不要传 `--gen_questions_per_attempt`**：它是多轮路径的吞吐杠杆
> （`use_qqueue = is_retool and multi_q>1`），L0 走单轮、**从不读它**——
> 配置层会直接拒跑（`_validate_swe_l0_config` 第 ⑤ 条），防止"调了没反应"。
> L0 每次 attempt 采 `Q_batch_size×num_pre_Q` = 1×8 条轨迹；要提吞吐就抬
> `--num_pre_Q`（同时扩组内 patch 多样性，对本任务是净收益）。

判据（先写死再看数）：

| 观察 | 判定 |
|---|---|
| `apply_ok` ≥ 50% | ✅ 格式先验够，直接长跑 |
| 20~50% | ⚠ 格式是瓶颈：先查生成预算（`--max_gen_tokens`）与提示，再考虑轻量 SFT |
| < 20% | ✗ 「4B 能产出 diff」假设不成立——这本身是值得记录的发现（docs/17 §2 的降级规则同型） |

**「修复率 0% 但 apply 率 90%」和「修复率 0% 但 apply 率 5%」是两种完全不同的病**，
处置相反：前者治推理，后者治输出格式/长度预算。这就是 docs/17 §3 要求两者拆开报的理由。

### 步骤 3：训练

```bash
bash rlab/run_gsm8k.sh swe_l0 /root/Qwen3.5-4B \
    --out_dir rlab_out/swe_l0_p1 --steps 200 --save_steps 50 \
    --num_pre_Q 8 \
    --swe_test_timeout 60 --swe_test_workers 8 \
    --swe_work_root /root/swe_tmp \
    --attn_implementation flash_attention_2 --optim_8bit \
    --micro_rows 1 --gen_gpu_mem 0.30 \
    --vllm_gen_logps --vllm_logprobs_n 1 \
    --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
    --seed 42
```

> ⚠ `--swe_work_root` 必须指**磁盘目录**：pod 上 `/tmp` 常是 tmpfs，
> 8 条并发判分各写一份仓库副本会吃掉 RAM（本项目 60G 上限下这是真实风险）。

启动 30 秒内应看到：

```
[train] 偏离签名 signature=swe_l0-ts0-ol0-r1x1536-s200x50-lr1e-06-...-tp...-swt60-swp<hash>
```

### 步骤 4：评测（held-out dev）

```bash
python -m rlab.eval --algo swe_l0 --n 40 --seed 42 \
  --base_path /root/Qwen3.5-4B \
  --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN \
  --gpu_mem 0.78 \
  --models "step200=./rlab_out/swe_l0_p1/step_200" \
  --out eval_swe_l0_p1.json
```

输出会额外打印三行 L0 专属读数：

```
[swe_l0] apply 成功率 62.5%｜修复率 15.0%（6/40）｜**条件修复率 24.0%**｜平均判分 2.31s
[swe_l0] apply 归因 {'applied': 25, 'empty': 10, 'check': 5}｜F2P 通过 21/58｜P2P 回归 3 条
```

---

## 3. 判据（跑之前写死，别事后编）

### 3.1 主/辅指标（docs/17 §4）

| 指标 | 口径 | 落盘键 |
|---|---|---|
| **修复率**（主） | F2P 全过 **且** P2P 无回归 | `acc`（±1 域）/ `swe_fixed` |
| apply 成功率（辅·格式） | patch 能 `git apply`（与"修对没修对"解耦） | `swe_apply_rate` |
| **条件修复率**（辅·能力） | 修复率 / apply 成功率 | `swe_cond_fix_rate` |
| 平均测试耗时（成本） | 每条轨迹的判分墙钟 | `swe_avg_test_seconds` |
| apply 失败归因 | `empty`/`too_long`/`check`/`apply`/`too_many_files` 分档 | `swe_stat.apply_stage` |

⚠ **条件修复率是两个率的商**（AGENTS.md）：既不能单独当机制证据，也不能用 acc 的
McNemar 不显著去否证它。三项必须**联合报告**。

### 3.2 训练中盯四件事

| 指标 | 健康形态 | 警报形态 |
|---|---|---|
| `apply_ok`（record 列） | 随步数上升或稳定 >0.4 | **恒 0** = 模型完全不会写 diff（先查提示/预算，不是能力问题） |
| `acc`（record 列） | 上升或至少非恒定 | **平坦** = 系统性签名，当场停 |
| `degenerate group`（整组无梯度） | ≈0 | 冲高 = 组内全对/全错（L0 池子小，全对概率天然更高） |
| `n_tested` vs `n_trajs` | 前者 < 后者（去重生效） | **恒相等** = 组内去重没生效（白烧判分） |

⚠ 「**精确 0% / 恒常数 / 平坦**」是本项目认定的系统性 bug 签名，不是"还没学到"。

### 3.3 验收（与 docs/17 M2/§2 的决策规则对应）

| 判据 | 阈值 |
|---|---|
| L0 决策门（docs/17 §2） | SFT/RL 基线在 L0 上 **<20% 修复率** → 4B 不足以支撑 RL 信号密度，项目降级为纯研究性 |
| vs base 配对 McNemar | **p<0.05**（同题、同 seed、同轮） |
| 效应量 | >4~5pp 才可读（单 run 方差可达 ±4pp） |
| 条件指标 | 必报 `修复率 / apply 率` 与其商 |

---

## 4. 本次实现里最容易踩的五件事（都是设计里刻意防的）

1. **F2P 名单永远是实测的**。手写会在"注入的 bug 其实没被覆盖"时静默变成假 F2P
   —— 那时 reward 恒 +1，"精确 100%"就是这把尺子失效的签名。两道闸：
   `prepare` 执行验收 + `validate_swe_rows` 拒收 f2p 为空的行。
2. **测试跑不起来 ≠ 通过**。语法错/收集期失败时 pytest 的 JUnit 里只有一条伪用例
   （id 不含 `::`），与 f2p id 对不上 → 判不通过。若按"没有失败即通过"判，
   语法错误会被算成满分（最贵的一种假分数）。
3. **每次判分用全新仓库副本**。apply 会原地改文件；复用目录会让第二次判分看到
   上一次的 patch——结果看起来正常，只是都等于第一条。
4. **apply 失败单独计数**。它进 `format`（格式指标），不进"能力"叙事。
   条件修复率的分母就是它。
5. **组内相同 patch 只测一次**。测试执行是本项目最贵的一步；`n_tested < n_trajs`
   是去重生效的证据，恒相等就是没生效。

> 另有两条**配置层**的静默空转防线（`_validate_swe_l0_config`）：
> `--gen_questions_per_attempt`（单轮不读）与长度/工具 shaping 项（L0 无受力面，
> 开了只会造就"短 patch 骗分"的捷径）——都直接拒跑而不是警告。

---

## 5. 已知未做（如实声明，别当成已完成）

| 项 | 说明 |
|---|---|
| **真机验收** | 本机无 GPU：**base 的 diff 产出率、训练增益、显存峰值、判分墙钟全部未测**（docs/17 M1/M2） |
| **L1 / L2** | SWE-Gym 子集、oracle 定位窗口、只读 shell 工具、`str_replace_editor`、docker 环境构建**全部未做**——按 docs/17 §2 的决策规则，L1 要等 L0 有显著效应才启 |
| 种子覆盖面 | 只有 8 个手写种子（8 类 bug）。合成变异能放大条数但**不扩分布**；加种子才是真扩（当前池 train 29/dev 19，dev 精度不足） |
| 环境隔离 | 判分在**子进程 + RLIMIT**（Linux）里跑，不是 docker/venv per-instance。L0 的 instance 全是自包含小文件（不装依赖），这够用；L1 起必须换 docker（docs/17 §2.5） |
| reward 塑形 | 只有 outcome-only（F2P 硬判据 ±1）。没有"部分通过"的塑形——tree-sitter 级别的中间信号是 L1 之后的事 |
| 多实例/多文件 patch | schema 支持（`files: {路径: 内容}`、`MAX_PATCH_FILES=8`），但 8 个种子里只有单文件。多文件实例的行为**未验证** |
| 上下文压缩 | L0 的定位是给定的，不需要检索/压缩。L2 的"整库放不下"问题未触碰 |

---

## 6. 复现顺序（照着做）

```bash
cd ~/llm-learning && git pull

# ① 生成池子（CPU；本机约 3 分钟）
python -m rlab.prepare_swe_l0

# ② 冒烟：20 步看 apply_ok（go/no-go，见 §2 步骤 2）
bash rlab/run_gsm8k.sh swe_l0 /root/Qwen3.5-4B --out_dir rlab_out/swe_l0_probe \
    --steps 20 --save_steps 20 --swe_work_root /root/swe_tmp ...

# ③ 判分成本实测（决定 batch）
grep test_duration rlab_out/swe_l0_probe/record.jsonl | head

# ④ 训练（--steps 20 先通闭环，再 200+）
bash rlab/run_gsm8k.sh swe_l0 /root/Qwen3.5-4B --out_dir rlab_out/swe_l0_p1 ...

# ⑤ 评测 + 三项指标（修复率 / apply 率 / 条件修复率）
python -m rlab.eval --algo swe_l0 ... 
python -m rlab.analysis --record rlab_out/swe_l0_p1/record.jsonl
```

---

## 7. 下一步（按 docs/17 的里程碑）

| 里程碑 | 当前状态 |
|---|---|
| **M1 环境探针** | L0 的"环境"= git + pytest 子进程，**已端到端跑通**（CPU 侧）；pod 上的耗时/内存**待测** |
| **M2 L0 训练** | preset/接线/判分**已就绪**；训练增益**待测**。这是 docs/17 §2 决策规则要回答的那一问 |
| **M3 L1 迁移** | 未启。前置：L0 出现 >5pp 的显著效应 + 池子扩到能读出的精度（dev ≥300） |
| **M4 与 Search-GRPO 合流** | 未启。L0 是单轮，与多轮框架的公共面很小——合流点其实在 **reward 插件化**（`score_group_swe_l0` 的 `test_executor` 注入位就是雏形） |
