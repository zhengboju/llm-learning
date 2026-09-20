# retool_math p8 训练「一直没效果」完整诊断报告

**入口**：用户报告 retool_math p8 训练「一直没效果」，需排查。  
**现象签名**：报表显示 `code% = 401.5/406.5/417.0/366.0`（>100% 不可能）、`step100 -4.6pp McNemar p=0.003 显著`。

---

## 1. 测量层：三个连环 bug（已修复 commit 5fcbc98）

### 1.1 除数错误：code% 虚高 8 倍

**bug**：`eval_vllm_one.py` 的 `code_rate/code_ok_rate/avg_rounds` 除数用 `n_valid`（题数），而分子列表长度是 `n_valid × val_n`（采样档轨迹数）。  
**结果**：采样档（`val_n=8`）所有代码率虚高 **8 倍**——401.5% 实为 **50.2%**。

| 模型 | 报表 code% | 修复后真实 code% |
|---|---|---|
| BASE | 366.0 | 45.8 |
| step100 | 401.5 | 50.2 |
| step200 | 406.5 | 50.8 |
| step300 | 417.0 | 52.1 |

**定案**：代码从未被「压灭」——各存档点 45.8~52.1%，record 训练期实测 code 率 60~69% 独立佐证。此前 p5/p6 记的「code% 50→3 压灭」是 greedy 档（`val_n=1`，除数恰好正确）的真实现象，但**不能外推到 p8 采样档**——两档口径不同。

**修复**：除数改 `len(code_used)`（= `n_valid × val_n`），写入 `metrics_version=2` 标记。

---

### 1.2 per-item 索引错位：code-layer 分析口径全错

**bug**：per-item 的 `code_used[i]`（i=题号）从平铺列表 `[题][采样]` 取，只覆盖题 0..24 的全部采样，题 25.. 全缺。  
**结果**：`--code-layer`/`--code-migration` 两个分析命令口径全错且**不可回修**（需重跑 eval）。

**修复**：按题聚合 `val_n` 条轨迹，写入 per-item 时用聚合后的题级统计。

---

### 1.3 配对检验二值化：显著性回答了另一个问题

**bug**：McNemar 无条件按 `acc == 1.0` 二值化，而采样档 per-item acc 是 Average@N 小数（0.0/0.125/.../1.0）→ 检验的是**「N 条全对率」而非 acc**。

**构造反证**：同一份数据上真实 acc 差 +7.5pp 而全对率差为负——**两个口径可以方向相反**，"显著"回答了另一个问题。

| 对比 | 真实 Δacc | 全对率差 | 方向 |
|---|---|---|---|
| A vs B | +7.5pp | −0.04 | **相反** |

**修复**：`paired_test_auto` 按口径分派（二值→McNemar / 连续→配对均值 z 检验）。

---

### 关键自我纠正

一开始据 §1.1 断言「训练有效、是测量假象」，**说过头了**。Δacc（step100 −4.6pp / step300 +0.3pp）来自**正确计算**的聚合 acc，不因修复而改变。

评测层三 bug 摧毁的是「code 压灭」叙事与显著性判定，**不解释增益缺失**。

---

## 2. 训练层：两个独立问题（record 实测定案）

### 2.1 剂量严重不足（70 次 optimizer 更新）

**record 实测**：2240 样本 = 280 组（每组 8 条），GAS=4 → **70 次 optimizer 更新** @ lr=1e-6。  
**参考对比**：agentic-rl-lab/05-retool 是 64 条/step × 数百 step，p8 折合 **35 step 等效**。

**训练期曲线**：acc 55.0%→68.1% 明确在学，**只是还没学完**——「没效果」很大程度是「没跑够」。

| 窗口 | acc率 | 趋势 |
|---|---|---|
| 0~160 | 55.0% | 起步 |
| 320~480 | 66.2% | 上升 |
| 800~960 | 63.1% | 震荡 |
| 2080~2240 | 68.1% | 末尾仍涨 |

**墙钟**：11 小时只跑出 70 次更新，平均 9.4 分钟/更新——丢弃率 81%→27% 的动态采样白跑大量全错组是主因（生成端效率问题已在 859dea8 修复）。

---

### 2.2 ts0 配置 + 长度失控（末段截断 42%、长度膨胀 41%）

**run_info 实锤**：`trunc_shaping=0.0`（CLI 覆盖了 preset 的 0.5），而 §7.3.1 已证伪 **ts0 丢掉 step200 全部增益（−6.4pp, p=0.002）且中段崩盘**。

**健康检查两条告警实打实触发**：
- **末段截断率 42%** >40% 阈值（"final 答案被切、acc 结构性受损"）
- **长度膨胀 41%**（avg_clen 3971→5612）（"长度常与正确性伴随，outcome-only 奖励会把它当代理强化"）

**机制**：`trunc_shaping=0.5` 是唯一能压长度的反向信号，p8 把它关掉 → 无对冲力 → 长度失控。

**CLI 覆盖链**（预算三件套同时被改）：
| 参数 | preset | run_info | 偏离 |
|---|---|---|---|
| `round_gen_tokens` | 2048 | **6144** | 3× |
| `max_context_tokens` | 14336 | **26400** | 1.84× |
| `trunc_shaping` | 0.5 | **0.0** | ts0 |

预算自洽（4×6144+1024+798=26398 ≤ 26400），但极限且无反向压力。

---

### 2.3 次要：末轮代码统计盲区（未修，量级小）

**机制**：`multi_turn_rollout_group` 的 `code_stats[i]["code_used"] += 1` 在 `if is_final_round: continue` **之后** → 末轮写的代码既不执行、也不计入 `code_used`；且 `retool_stop=True` 下它在闭合围栏处 `finish_reason="stop"`（非 length）→ `trunc_final=0`。

该轨迹无 boxed 必 acc=-1，**两个统计量都看不见它**。

**规模对账**（record 会话 A）：
- 无 boxed（1 − fmt）：33.6%
- trunc_final：31.1%
- 差值：**2.5pp**

**定论**：真实但**次要**（差值 <3pp 在噪声内），trunc 才是主因。

---

## 3. 根因排序与修复优先级

### 按影响排序（从测量层到训练层）

| 序号 | 层 | 问题 | 影响 | 状态 |
|---|---|---|---|---|
| 1 | 测量 | code% 除数错 8× | 叙事误导：「压灭」是假象 | ✅ 已修 |
| 2 | 测量 | 配对检验二值化 | 显著性回答另一问题 | ✅ 已修 |
| 3 | 训练 | **剂量不足（70 更新）** | **真实原因**：还没学完 | ⚠️ 待补 |
| 4 | 训练 | **ts0 + 长度失控** | **真实原因**：无反向压 | ⚠️ 待修 |
| 5 | 测量 | per-item 索引错位 | code-layer 需重跑 | ✅ 已修 |
| 6 | 训练 | 末轮代码盲区 2.5pp | 次要 | 留观 |

### 核心结论

**「训练一直没效果」的真因不在测量层——是剂量 + ts0 双重训练侧问题。**

- **剂量**：70 次更新在 lr=1e-6 下远不够，训练期 acc 55→68 明确在学；
- **ts0**：已被 §7.3.1 证伪的配置（−6.4pp），本次 42% 截断 + 41% 膨胀实打实触发其病症。

测量层三 bug 的作用是**摧毁了「code 压灭」这个错误叙事**，但 Δacc 本身（step100 −4.6pp）是正确计算的聚合值，不因修复而改变。

---

## 4. 行动清单

### 立即可做（测量层已修，验证用）

1. ✅ **git pull** 获取 commit 5fcbc98 修复版 eval；
2. **重跑 p8 eval**（若原 json 需回修数值）：
   ```bash
   python -m rlab.eval --retool --val_n 8 --models \
     BASE=./rlab_out/retool_math_p8/base \
     step100=./rlab_out/retool_math_p8/step_100 \
     step200=./rlab_out/retool_math_p8/step_200 \
     step300=./rlab_out/retool_math_p8/step_300
   ```
   （新 json 带 `metrics_version=2`，analysis 自动识别）；
3. **生成修复后报表**：
   ```bash
   python -m rlab.analysis --pair BASE step300 --eval-results rlab_out/retool_math_p8/eval_results.json
   ```
   确认：code% 在 45.8~52.1 区间、配对检验用连续 z 而非 McNemar、`*` 标记说明回修。

### 训练侧验证与修复（单变量纪律）

#### 4.1 验证「剂量不足」假设

**H0**：训练期 acc 曲线 55→68 + 末尾仍上升 = 还没收敛，补剂量即可见效。

**单变量实验**：
```bash
# 保持 ts0（隔离变量），只加剂量
bash rlab/run_gsm8k.sh retool_math /root/Qwen2.5-3B \
  --all_steps 600 \
  --trunc_shaping 0.0 \
  --round_gen_tokens 6144 --max_context_tokens 26400 \
  --seed 42
```
（600 步 = 150 次更新，2× p8；GAS=4 不变）

**判据**：step300/450/600 的 dev acc 曲线若持续上升 → H0 成立（剂量是瓶颈）；若 300 步后平台/下滑 → 剂量非主因。

#### 4.2 验证「ts0 有害」假设

**H0**：§7.3.1 的 ts0 证伪（−6.4pp）在 retool_math 上复现。

**单变量实验**（先修 §4.1 后做）：
```bash
# 同剂量，改 ts0→ts0.5
bash rlab/run_gsm8k.sh retool_math /root/Qwen2.5-3B \
  --all_steps 600 \
  --trunc_shaping 0.5 \
  --round_gen_tokens 6144 --max_context_tokens 26400 \
  --seed 43
```

**判据**：配对比较 ts0@600 vs ts0.5@600，若 Δacc >3pp（超训练方差）→ H0 成立；否则 math 任务对 ts 不敏感。

**注意**：健康检查会在 trunc_rate >40% 时告警——若 ts0.5 下告警消失且 acc 上升，双重验证通过。

#### 4.3 预算归一化（可选，若 §4.2 后 trunc 仍高）

**动机**：6144 单轮 + 4 轮 = 26398 逼近 26400 上限，实为极限预算；若模型需要多轮交互，该预算反而限制轨迹完整性。

**建议**：参考 agentic-rl-lab/05-retool 用 6 轮×1024 + 8192 ctx（更多轮数、单轮更紧凑），或对齐 Qwen3.5-4B 探针 v4 的 3072×4。

---

## 5. 元教训

### 5.1 测量层

1. **">100% 的率"是除数错的免费签名**，要当场算量纲（401.5/8=50.2 一步就能验）；
2. **聚合指标与 per-item 指标改口径（greedy→Average@N）时，所有下游消费者都要跟着改**——本次三个 bug 全部源于同一次 `--val_n` 采样档引入，而除数、索引、检验三个消费点没有一个跟上；
3. **"显著"必须先问"检验的是哪个量"**——Average@N 上的 `acc==1.0` 是个静默换题的陷阱。

### 5.2 训练层

1. **"训练期曲线在学"（55→68）+ "还在涨"（末尾 68.1）= 剂量不够的签名**，不是"没效果"；
2. **已被单变量证伪的配置（ts0）不能无声覆盖回去**——CLI 覆盖 preset 时必须有显式记录或告警；
3. **健康检查的告警（trunc 42% / 长度膨胀 41%）是训练侧问题的实时签名**，不是"跑完再说的注释"。

---

## 附录：p8 run_info 关键参数

```json
{
  "model_path": "/root/Qwen2.5-3B",
  "algo": "retool_math",
  "all_steps": 300,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "num_pre_Q": 8,
  "Q_batch_size": 1,
  "gen_questions_per_attempt": 4,
  "round_gen_tokens": 6144,
  "max_rounds": 4,
  "max_context_tokens": 26400,
  "trunc_shaping": 0.0,
  "temperature": 1.0,
  "top_k": -1,
  "lr": 1e-6,
  "seed": 42
}
```

**optimizer 更新数**：300 micro-batch / GAS 4 = **75 次**（record 实测 70 次，末尾因丢弃少 5 次）。  
**有效 batch**：8 行 × GAS 4 = 32 样本/更新。  
**总轨迹数**：2240 条（record 实测）≈ 35 step 等效（参考 64 条/step 口径）。

---

**报告完成时间**：2026-09-20  
**修复 commit**：5fcbc98（测量层三连修）  
**待验证假设**：剂量不足（§4.1）、ts0 有害（§4.2）
