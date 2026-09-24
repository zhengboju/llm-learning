# retool_math p10 训练诊断 + 内嵌评测盲窗修复（2026-09-24）

**生成时间**：训练至 ~440/600 组（73%）
**数据源**：`rlab_out/retool_math_p10/record.jsonl` + `run_info.json` + 训练日志
**签名**：`retool_math-ts0-ol1-r4x6144-s600x100-lr1e-06-d0-1-t0cbda1-vkgdn_prefill_backend=triton-sp3aac5d-stop1-of1-u8-c26400-sd42`
**会话 A**：3520 样本 ≈ 440 组，gen_ver 0..432，[09-22 16:02 ~ 09-23 23:31]，单会话无混跑

---

## 1. 结论先行

1. **训练没有退化，但也没有在学**（440 组后 acc 与开局持平）。早前"acc 较开局下滑
   >5pp"的 decline 健康告警是截断尖峰（80~100 组 trunc 32.5%）的**测量伪影**——
   record 的 acc率把截断样本（多数错）计入，且 20 组窗口二项噪声 ±5.1pp。
2. **真正的主导病理是截断/长度，不是代码压灭**：窗口级 corr(trunc, acc) = **−0.90**，
   corr(code, acc) = +0.43。code 率在 40~60 组触底 33.9% 后自回升到 47~56%，没有走向
   p5 式灭绝。
3. **本次 run 最大的事故 = 内嵌评测盲窗**：step100/200/300/400 四个存档 × test+train
   共 **8 路内嵌评测全部 TIMEOUT**，eval_*.json 全缺失——30h 训练全程盲跑。
   root cause：`train.py` 硬编码 `timeout=900`，而 retool 多轮采样档
   （n=500×4轮×6144 token，temp1.0）单路评测 >15min 物理上跑不完。
   **2026-09-21 加内嵌评测是为了治 p9"训完 11h 才发现 step100 深坑"的教训，
   被这个超时反手做成了同类盲窗。**

---

## 2. 训练曲线判读（440 组按 80 组聚合，噪声 ±5.1pp）

| 组段 | 0~4 | 4~8 | 8~12 | 12~16 | 16~20 | 20~24(=320~440组) |
|---|---|---|---|---|---|---|
| acc率 | 60.0 | 57.7 | 51.7 | 58.9 | 56.6 | **63.1** |
| code率 | 52.5 | 39.5 | 33.9 | 41.2 | 45.8 | 47.2 |
| trunc率 | 20.2 | 25.0 | 33.0 | 23.3 | 27.0 | 19.7 |

- 首尾 acc 60.0→63.1（≈1.2σ）→ 无显著变化、无下降；对照组 p8 同协议 280 组是
  55→68 的爬升，p10 到 440 组没出现这种形态 → **未在学**。
- 未在学的最可能解释：`overlong_filter` 把 ~25% 截断样本的梯度清零，有效更新剂量
  被砍掉四分之一到三分之一（of1 的已知代价，非 bug）。
- corr(trunc, acc)=−0.90：acc 窗口波动几乎完全由截断率震荡（15↔42%）驱动——
  高截断窗 = acc 假坑 + 真实产能损失（of1 下这些样本零梯度）。

---

## 3. 内嵌评测盲窗事故与修复

### 3.1 事故链

- 2026-09-21 加 `_run_inline_eval`（train.py），本意是 checkpoint 保存后自动跑
  test+train 评测，让 p9 式"训完才发现深坑"变成训练中可判。
- `_sp.run(..., timeout=900)` 硬编码。retool 多轮采样档单路评测实测 >15min。
- 结果：4 存档 × 2 split = 8 路全 TIMEOUT，日志只有一行 `[eval] step N ... TIMEOUT
  (>15min)`，且 `continue` 静默跳过 → `step_N/eval_*.json` 全部缺失。
- 训练日志里 8 次 TIMEOUT 各自相隔 15min+，占用约 2h 墙钟，产出为零。

### 3.2 修复（commit 待填）

三件套，逐项可测：

1. **超时进配置**：`config.py` BASE `eval_timeout_s=900`（旧行为零变化）、
   retool_math preset 抬到 `3600`（=手动跑预算）；CLI `--eval_timeout_s` 覆盖。
2. **超时不静默**：`train.py` 超时不再 `continue` 吞掉，改为写"空结果哨兵"
   `eval_*.json`（`{"acc": None, "n": None, "error": "timeout"}`），并打印
   "此 checkpoint 本轮为盲窗"。让"没评测"与"没结果"可区分。
3. **盲窗可见**：`analysis.py --record` 表新增「评测」列——checkpoint 所在窗口的
   `eval_test.json` 缺失或为哨兵（acc/n 为 None）→ 标 "盲"。p10 这 8 个盲窗从此
   在曲线表上看得见。

### 3.3 验证

- `test_inline_eval_timeout_fix()`（test_retool_cpu [AK] 段，9 项）：锁 config
  默认/preset/CLI 三层接线、哨兵形态、analysis 判盲口径、get_config 端到端。
- 本地合成 fixture 实测 `summarize_record`：缺失→盲 / 哨兵→盲 / 有效→正常，符合预期。

---

## 4. 下一步（p11）建议

- 评测侧修复（本 commit）先合入；p10 若没跑完，用手动 `eval_vllm_one.py` 评
  step_400 做去留判定（配 `--gpu_mem 0.6`、无 15min 限制）。
- p11 训练变量**改为组相对长度惩罚**（`--len_penalty_w 0.1`）——trunc 是本 run 的
  主导病理；`--code_attempt_w 0.05` 降级为后续单变量（code 压灭自回升，非首要约束）。
- `--save_steps 50`：p10 的 100 步存档粒度抓不住 80~100 组的截断尖峰期。
- 若 eval 每路 ~30min，`--eval_n 300`（默认 500）可把每 checkpoint 的评测墙钟砍掉
  40% 而不失 3pp 级判据分辨力。
