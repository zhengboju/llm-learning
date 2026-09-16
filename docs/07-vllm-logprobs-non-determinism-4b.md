# Qwen3.5-4B 上 vLLM 采样 logprobs 不可复现 —— 禁用 `--vllm_gen_logps`

日期：2026-09-15  
模型：Qwen3.5-4B（多模态复合 ckpt，统一目录 `/root/Qwen3.5-4B`）  
vLLM：v0.19.1 V1 引擎，`gdn_prefill_backend=triton`  
后端：2×H20，训练用 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（EngineCore 进程内）

## 一句话结论

在这套环境上，**不开 batch-invariant 时**同一命令跑两次，vLLM 采样出来的 token 只有 2.91% 逐位置相同，对应 logp 最大差 14.5 nat。这一档下 vLLM 报告的 `logprobs` 不能作为训练 gen_logps 的来源，必须回退 torch 副本重算（`vllm_gen_logps=False`），接受约 8–9G 额外显存。

**2026-09-16 更新（§9）：加 `VLLM_BATCH_INVARIANT=1` + 显式 attention backend 后两条都被修好** —— 跨进程 token 一致率 **100.00%**、`|Δlogp|` 全 0；且与 torch 副本的跨引擎残差降到 **max 0.123 nat**（>1 nat 占比 0%），比不开档时同引擎自身的噪声地板（max 0.54）还小。是否把训练档切到 vLLM logps，取决于**验收门**（step 1 `clip_frac`）与**吞吐代价**（§10）——**尚未切换，训练仍走 torch 副本**。

## 证据链

### 1. 训练期对拍地板异常（已知）

`bash rlab/run_gsm8k.sh retool_math /root/Qwen3.5-4B --vllm_gen_logps --verify_gen_logps 5 ...` 的 `[train][口径]` 行：

| | step 1 | step 2 | step 3 |
|---|---|---|---|
| clip_frac | 0.0091 | 0.0094 | — |
| approx_kl | 3.46e-3 | 3.15e-3 | — |
| frac\|d\|>0.1 | 0.041 | 0.047 | — |
| max \|d\| | 2.68 | 5.93 | 12.8（g5 行 0 列） |

与 doc 基线（post-pad-fix，2026-09-11）相比，clip_frac 高约 13×，approx_kl 高约 7×，max 异常大，且极值点**总是落在 列 0**（每 round 首 token = prefill 位置）。

### 2. diag2 跨引擎对拍：probe 与训练形态天壤之别

同一轨迹上：

| 口径 | mean \|Δ\| vs torch | max \|Δ\| |
|---|---|---|
| vLLM probe（`logprobs=20` + raw，`max_tokens=1`） | **0.022** | **0.231** |
| vLLM 轨迹（`logprobs=0`，`max_tokens=3072`，训练实际形态） | **0.055** | **6.798** |

probe 与 torch 几乎一致（top-1 36/36 相同，top-20 交集 0.968），说明**模型分布本身没问题**；训练形态却差出一个数量级，问题出在**训练用的那条报数路径/请求形态**上。

### 3. lpmode：同位置同 token，请求形态一变，logp 就变

`--measure lpmode` 对同一 (prompt, 位置, token) 用四种形态各问一次：

| q, L | A(T1,K0) | B(T1,K) | C(T32,K) | D(T32,K0) | 极差 |
|---|---|---|---|---|---|
| 7, 0 | -2.04 | -1.39 | -0.404 | -0.988 | 1.64 |
| 4, 256 | -0.921 | -1.29 | -0.838 | -1.16 | 1.42 |
| 5, 1024 | -3.42 | -1.86 | -3.37 | -1.78 | 2.92 |

更糟的是，**完全相同的形态 B 问两次（B2 档）**，top-K 字典 0/36 完全相同，交集上 tail token 最大差 **5.98 nat**，mean 2.24。说明读数本身不是确定函数。

同形态两次 top-8 示例（q=0 L=512）：

```text
B_KK_T1  : 25:-0.000 318:-12.938 11:-13.375 17:-13.563 1076:-13.813 ...
B2_KK_T1 : 25:-0.000 318:-12.313 11:-13.000 13:-13.188 17:-13.313 ...
```

只有 argmax（token 25，logp≈0）稳定；第 2 名以后的集合和数值都在变。

### 4. diff_traj：同一命令跑两次，彻底不可复现

```text
python -m rlab.diag_logps --diff_traj rlab_out/diff1/traj1.jsonl rlab_out/diff2/traj2.jsonl

逐位置 n=16221（8 题）；token 逐位置一致率=2.91%
|Δlogp|: mean=0.675 p50=0.136 p99=5.79 max=14.5  >0.1=53.35% >1nat=21.29%
位置0（每轮首 token）: mean=0.909 max=6.28｜其余位置: mean=0.675 max=14.5
最差点 q=7 pos=61 a=-14.523 b=-0.020 Δ=14.503 token相同=False
```

这是**决定性证据**：同一设置、同 seed、同 backend、同模型，两次建轨迹的采样 token 和上报 logp 都完全不是一回事。

### 5. det：最小复现——RNG 是好的，logits 不是

`--measure det`（同一进程、背靠背、同参同 seed、同一 prompt 320 token，重复 3 次）：

```text
#1 ids[:8]=[1206, 1423, 279, 3140, 6572, 1442, 7308, 393] top1=1206 lp(top1)=-0.224  topK字典hash=09b11456
#2 ids[:8]=[1206, 1423, 279, 3140, 6572, 1442, 7308, 393] top1=1206 lp(top1)=-0.0348 topK字典hash=b0f99712
#3 ids[:8]=[1206, 1423, 279, 3140, 6572, 1442, 7308, 393] top1=1206 lp(top1)=-0.0759 topK字典hash=2e2189e8

token 唯一数=1/3；top-K 字典唯一数=3/3；top-1 logp 极差=0.19
```

两条结论都很硬：

- **RNG 没问题**：三次采到完全相同的 token 序列 ⇒ "seed 没生效/没对齐"这个常见归因被排除。
- **logits 不确定**：同一个位置、同一个 token，引擎自己报的 logp 在 -0.035 / -0.076 / -0.224 之间跳（0.19 nat，等价于 p 从 0.97 到 0.80）。这是**同一进程内背靠背**的结果，因此跨进程预热、编译缓存、autotuner 一次性状态都解释不了它；探针里也没有任何 harness 变量。

机制推断：前向的**归约顺序**不固定（split-K / atomic 累加、按批量选择的 kernel），bf16 下表现为 ~0.2 nat 的 logit 抖动；头部抖 0.2 nat 已足以让近并列 token 互换 → 轨迹分叉（这是 2.91% 的来源）。tail token 因为 softmax 分母被 argmax 主导，抖动直接落在 logp 上（lpmode 实测交集 max|Δ| 5.98）。

### 6. batch-invariant：先启动失败，后完全修好

`VLLM_BATCH_INVARIANT` 在 v0.19.1 里**确实存在**（`vllm/envs.py:78` 注册），但直接开会启动即失败：

```text
RuntimeError: VLLM batch_invariant mode requires an attention backend in
['FLASH_ATTN', 'TRITON_ATTN', 'FLASH_ATTN_MLA', 'TRITON_MLA'], but got 'None'
```

原因是 batch-invariant 的检查跑在 attention backend 解析**之前**，必须显式指定。补上 `--attention_backend FLASH_ATTN` 后，`--measure det`（同一进程、背靠背、同参同 seed、重复 3 次）：

```text
token 唯一数=1/3；**top-K 字典唯一数=1/3**；top-1 logp=-0.0274 ×3；极差=0
```

对照第 5 节不开档的 `token 1/3 但字典 3/3、极差 0.19`：**字典也唯一了**。这反向坐实了机制——非确定性来自归约/批量相关的 kernel，与 RNG 无关。

（顺带确认本 pod 的 GDN 走 triton：引擎日志 `[gdn_linear_attn.py:147] Using Triton/FLA GDN prefill kernel`，FlashInfer JIT 全程未启用、无 SIGKILL。）

代价提示：batch-invariant 会关掉 custom all-reduce、改用确定性 kernel（失败运行的 config 里已可见 `disable_custom_all_reduce=True`），吞吐会掉。所以即使可复现性修好，仍是一个 **VRAM（torch 副本 8–9G）vs 吞吐** 的取舍，见 §10。

### 7. 训练端 verify 复现同一结论（2026-09-15 19:xx，第 2 次真机）

同一条训练命令（仍带 `--vllm_gen_logps --verify_gen_logps 5`）的第 2 次运行，5 组对拍：

| 组 | mean | p50 | p99 | max | >0.1 | >1nat | 前后半段 mean\|d\| |
|---|---|---|---|---|---|---|---|
| 1 | 1.41e-2 | 1.94e-5 | 2.17e-1 | 1.05 | 4.11% | 0.00% | 1.71e-2 → 1.06e-2 |
| 2 | 1.30e-2 | 1.15e-5 | 2.04e-1 | **5.10** | 3.53% | 0.03% | 1.23e-2 → 1.46e-2 |
| 3 | 9.48e-3 | 1.57e-5 | 1.48e-1 | 1.00 | 2.53% | 0.00% | 1.12e-2 → 7.79e-3 |
| 4 | 1.73e-2 | 5.38e-4 | 2.08e-1 | 0.90 | 4.76% | 0.00% | 1.99e-2 → 1.45e-2 |
| 5 | 1.96e-2 | 2.80e-4 | 2.52e-1 | **3.23** | 5.75% | 0.01% | 2.20e-2 → 1.60e-2 |

三点与前面的离线证据互相印证：

1. **中位数极紧（p50 1e-5~5e-4）而 >0.1 占 2.5–5.8%**：确定性 token 两侧完全一致，噪声全在非平凡 token 上——与 probe/torch 的 0.022 均值同源。
2. **前后半段无漂移**（4/5 组后半更小）⇒ 排除"递归状态累积漂移"，坐实"局部噪声"。
3. **极值不再集中在列 0**（本次落在 762 / 2195 / 810 / 80 / 0）⇒ 不是位置相关的系统 bug，而是随机抽到近并列 token。极值幅度也在变（上一轮 12.8，本轮 5.10 / 3.23，另有一组仅 0.90）——**不可复现量的自然波动**，与 det 的 0.19 nat 抖动、lpmode 的形态敏感一致。

**代价可以直接读出来**（step 1，`staleness=0`、`mean_ratio=1.0000` ⇒ 两侧策略完全相同，无训练动力学）：

| gen_logps 来源 | clip_frac | approx_kl |
|---|---|---|
| vLLM 报的数 | **0.0087** | **2.23e-3** |
| torch 副本（doc 基线） | **0.0007** | **5.1e-4** |

即 **12× / 4.4×** —— 这就是"用 vLLM 的 logp 当 gen_logps"的全部代价，且它是纯测量噪声，不是训练不稳定。

### 8. 训练端 verify 的由来（代码事实）

`_verifier` 仅在 `_use_vllm_logps and _verify_budget > 0` 时构造（`rlab/rollout.py:1064`）。
所以：
- 看到 `[rollout][verify] vLLM vs torch` ⇒ **该 run 一定开着 `--vllm_gen_logps`**；
- 换成 torch 副本档时，`--vllm_gen_logps` **和** `--verify_gen_logps` 都应去掉——只留 verify 会把副本白加载 ~8G 却没有任何对拍对象。

### 9. batch-invariant 档过关：跨进程可复现 + 跨引擎残差压到口径地板（2026-09-16）

命令（两次独立进程，除输出目录外完全相同；`--build_traj` 用的正是训练形态 `logprobs=0` + `max_tokens=3072`）：

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_BATCH_INVARIANT=1 PYTHONHASHSEED=0
COMMON=(--build_traj --model_path /root/Qwen3.5-4B \
  --chat_template_kwargs '{"enable_thinking": false}' \
  --providers vllm --vllm_backend triton --attention_backend FLASH_ATTN)

time CUDA_VISIBLE_DEVICES=0 python -m rlab.diag_logps "${COMMON[@]}" \
  --traj_out rlab_out/bi1/traj.jsonl --out rlab_out/bi1/vllm.jsonl
time CUDA_VISIBLE_DEVICES=0 python -m rlab.diag_logps "${COMMON[@]}" \
  --traj_out rlab_out/bi2/traj.jsonl --out rlab_out/bi2/vllm.jsonl
python -m rlab.diag_logps --diff_traj rlab_out/bi1/traj.jsonl rlab_out/bi2/traj.jsonl

CUDA_VISIBLE_DEVICES=1 python -m rlab.diag_logps --traj_in rlab_out/bi1/traj.jsonl \
  --model_path /root/Qwen3.5-4B --providers torch --torch_path fallback \
  --out rlab_out/bi1/torch.jsonl
python -m rlab.diag_logps --merge rlab_out/bi1/vllm.jsonl rlab_out/bi1/torch.jsonl
```

**A. 跨进程（两次独立进程）——通过**，与第 4 节的 2.91% 直接对照：

| | 不开档（§4） | batch-invariant（本次） |
|---|---|---|
| token 逐位置一致率 | 2.91% | **100.00%**（n=19952） |
| \|Δlogp\| mean / p50 / p99 / max | 0.675 / 0.136 / 5.79 / **14.5** | **0 / 0 / 0 / 0** |
| >0.1 / >1nat | 53.35% / 21.29% | 0.00% / 0.00% |

两次 `traj_id` 相同（`65681137d466`），逐点差值严格为 0 —— 不是"接近"，是逐位相同。

**B. 跨引擎（vLLM:triton vs torch:torch_ref）——通过**，与第 2 节的训练形态 max 6.798 直接对照：

| | 不开档训练形态（§2） | batch-invariant（本次，同形态） |
|---|---|---|
| target\|Δ\| mean | 0.055 | **0.0138** |
| target\|Δ\| p99 / max | — / **6.798** | 0.123 / **0.123** |
| >1.0 nat | 有 | **0.00%（0 点）** |
| top-1 一致 / overlap@20 | 36/36 / 0.968 | **36/36 / 0.976** |
| confident 分歧 | — | **0** |

`max|Δ|_common=0.687`（top-K 交集上）是两套 bf16 实现的口径地板；**关键量是目标 token 自身的 |Δ| ≤ 0.123**，且它小于不开档时同引擎重采自身的噪声地板（lpmode B vs B2：mean 0.0735 / max 0.54）。即：跨引擎残差已经**小于**本环境原本的自噪声。

最直观的一条：§1 那个 `vLLM=-0.946 / torch=-13.750` 的签名位置，本次同形态同量级位置 `q=6 L=0` 是 `vLLM=-0.942 / torch=-0.956（Δ=0.014）`。

**C. 对验收门的定量预测**：`retool_math` 的 `clip_low/high=0.2/0.28` ⇒ 记为 clip 需要 `Δlogp > log1.28 = +0.247` 或 `Δlogp < log0.8 = -0.223`。本次观测的目标 |Δ| 最大值 **0.123**，**没有任何一点跨过阈值** ⇒ step 1 的 `clip_frac` 预期落在 0 ~ 1e-3（不开档 0.0087，torch 副本基线 0.0007）。这是待真机验收的可证伪预测。

### 10. 代价与切换决定（未完成）

- **吞吐未测**：本次两次 `--build_traj` 墙钟 7m43s / 7m38s（差 0.9%，跨进程一致性顺带得到稳定复现）。这个数**不能**当作 batch-invariant 的开销，因为它含 ~47s 引擎初始化 + 36 点前缀重算，且没有同命令的不开档对照。要测就测**同一命令去掉 `VLLM_BATCH_INVARIANT` 的墙钟**，或者更直接：验收门那一跑的 `per-step gen 墙钟` vs torch 副本档的同一数字。
- **取舍的实质**：省下的是 GPU0 的 ~8–9G 显存 + 每步一次全序列前向；付出的是确定性 kernel 的生成开销。当前 GPU0 占用约 70G/96G，**显存并不紧张**，所以这笔账**只能靠吞吐来定**。
- **顺带观察（待训练日志确认）**：本次引擎 config dump 里是 `enable_prefix_caching=False`。若训练端同样如此，则多轮 rollout 的"续写复用前轮 KV"这条设计假设不成立——每一轮都会从头 re-prefill（retool_math 每 attempt 4 题 × 8 条 × 3 轮，代价可观）。零成本核实：

  ```bash
  grep -o "enable_prefix_caching=[A-Za-z]*" train_log*.txt | sort | uniq -c
  # 若确为 False，则 --vllm_gen_kwargs 整体替换（注意必须带上原有键）：
  #   '{"gdn_prefill_backend": "triton", "enable_prefix_caching": true}'
  ```

- **纪律**：切换训练档之前，训练一律保持 `--vllm_gen_logps=False`（且不带 `--verify_gen_logps`）。batch-invariant 档会改变采样数值（token 序列本身与不开档不同），**跨档比较不是单变量**——任何前后对比都必须同一档内进行。

## 结论与处置

1. **不开 batch-invariant 时，vLLM 的采样 logprobs 在本环境不可复现**——不是某个参数没调对，而是同一请求两次运行就会给出不同分布的 tail / 不同 token / 不同 logp。尾部（rank≥2）数值基本随机。
2. **该档下 `--vllm_gen_logps` 不能作为 retool_math 4B 的 gen_logps 来源**。用它训练会把一个随机量塞进 importance ratio，导致 clip_frac/approx_kl 被人为抬高，并偶尔爆出 10+ nat 的伪尖峰。
3. **该档下唯一同源且可复现的路是 torch 副本重算**（`vllm_gen_logps=False`）。它会：
   - 在 GPU0 多占约 8–9G（与 ref 模型共享时总占用需按 docs/04 重排）；
   - 每步做一次全序列前向；
   - 保证 gen_logps 与训练前向使用**同一个 kernel、同一个 bf16 舍入、同一个确定性路径**。
4. **训练命令去掉 `--vllm_gen_logps`，且 `--verify_gen_logps` 也一并去掉**（§8：verify 的对拍对象是 vLLM logps，没有它只会白加载 ~8G 副本）。期望地板回到 doc 基线量级：clip_frac ~0.0007，approx_kl ~5e-4。
5. **batch-invariant 档改写了这个结论的适用域**（§9）：可复现性与跨引擎口径都过关，代价只剩吞吐未测（§10）。切换与否等验收门 + 吞吐数据，**在此之前第 1–4 条就是当前纪律**。

## 仍开放的验证

- **切换档的验收门 + 吞吐**（§9C/§10）：`--vllm_gen_logps --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN --verify_gen_logps 5` 跑训练，看 step 1 的 `clip_frac` 是否落到 0~1e-3、per-step gen 墙钟相对 torch 副本档贵多少。
- **FlashInfer 档**：由于该 pod 上 FlashInfer GDN prefill JIT 会 OOM-kill（两次实锤，零成功），无法验证它是否也有同样的非确定性。理论上 FlashInfer 与 Triton 是不同 kernel 实现，不能外推。
- **其他模型 / 其他 vLLM 版本**：本结论仅限 Qwen3.5-4B + vLLM v0.19.1 + triton GDN prefill。3B 模型、非 GDN 模型、新版 vLLM 需单独验证。
- **训练端是否与诊断同档**：`grep -n "GDN prefill kernel" train_log*.txt`（应为 Triton/FLA）与 `grep -o "enable_prefix_caching=[A-Za-z]*" train_log*.txt`（§10 的顺带观察）仍建议跑一次，把训练日志与诊断数据的档位对齐。
- **batch-invariant 的其他 attention backend**：本次只验了 `FLASH_ATTN`；`TRITON_ATTN` 档未验（vLLM 的白名单允许，但"允许"不等于"同样确定"）。

## 纪律记录

- 2026-09-15 18:43 `--diff_traj` 给出 token 一致率 2.91% / max Δ=14.5 nat → 禁用 `--vllm_gen_logps` 成为显式决策。
- 2026-09-16 `VLLM_BATCH_INVARIANT=1` + `--attention_backend FLASH_ATTN`：det 字典 1/3、跨进程 token 100.00% / Δ=0、跨引擎 target max 0.123 nat（§9）。**决策未变**——切换要过验收门与吞吐关，训练仍走 torch 副本。
- 相关代码/工具保留（`vllm_logprobs_n`、`--measure lpmode`、`--diff_traj`、`--build_logprobs_n`），用于未来其他模型/backend 的复现性普查，不删除。
