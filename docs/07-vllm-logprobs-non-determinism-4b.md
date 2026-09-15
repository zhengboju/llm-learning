# Qwen3.5-4B 上 vLLM 采样 logprobs 不可复现 —— 禁用 `--vllm_gen_logps`

日期：2026-09-15  
模型：Qwen3.5-4B（多模态复合 ckpt，统一目录 `/root/Qwen3.5-4B`）  
vLLM：v0.19.1 V1 引擎，`gdn_prefill_backend=triton`  
后端：2×H20，训练用 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（EngineCore 进程内）

## 一句话结论

在这套环境上，**同一命令跑两次，vLLM 采样出来的 token 只有 2.91% 逐位置相同，对应 logp 最大差 14.5 nat**。因此 vLLM 报告的 `logprobs` 不能作为训练 gen_logps 的来源；`retool_math` 4B 训练必须回退到 torch 副本重算（`vllm_gen_logps=False`），接受约 8–9G 的额外显存开销。

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

### 6. batch-invariant 这一步尚未完成

`VLLM_BATCH_INVARIANT` 在 v0.19.1 里**确实存在**（`vllm/envs.py:78` 注册），但直接开会启动即失败：

```text
RuntimeError: VLLM batch_invariant mode requires an attention backend in
['FLASH_ATTN', 'TRITON_ATTN', 'FLASH_ATTN_MLA', 'TRITON_MLA'], but got 'None'
```

原因是 batch-invariant 的检查跑在 attention backend 解析**之前**，必须显式指定。工具已支持：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_BATCH_INVARIANT=1 PYTHONHASHSEED=0 \
  python -m rlab.diag_logps --build_traj --model_path /root/Qwen3.5-4B --providers vllm \
  --vllm_backend triton --attention_backend FLASH_ATTN --measure det --det_repeat 3 \
  --out rlab_out/diag2/det_batchinv.jsonl
```

判据：若 `top-K 字典唯一数=1/3` 且 top-1 logp 极差≈0 → 可复现性被修好，那时才值得重新评估 vLLM logps 路线（但还需过 torch 对拍这一关）；若仍 3/3 → 此路不通，torch 副本是终局。

代价提示：batch-invariant 会关掉 custom all-reduce、改用确定性 kernel（失败运行的 config 里已可见 `disable_custom_all_reduce=True`），吞吐会掉。所以即使它能修好可复现性，也是一个 **VRAM（torch 副本 8–9G）vs 吞吐** 的取舍。

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

## 结论与处置

1. **vLLM 的采样 logprobs 在本环境不可复现**——不是某个参数没调对，而是同一请求两次运行就会给出不同分布的 tail / 不同 token / 不同 logp。尾部（rank≥2）数值基本随机。
2. **因此 `--vllm_gen_logps` 不能作为 retool_math 4B 的 gen_logps 来源**。用它训练会把一个随机量塞进 importance ratio，导致 clip_frac/approx_kl 被人为抬高，并偶尔爆出 10+ nat 的伪尖峰。
3. **唯一同源且可复现的路是 torch 副本重算**（`vllm_gen_logps=False`）。它会：
   - 在 GPU0 多占约 8–9G（与 ref 模型共享时总占用需按 docs/04 重排）；
   - 每步做一次全序列前向；
   - 保证 gen_logps 与训练前向使用**同一个 kernel、同一个 bf16 舍入、同一个确定性路径**。
4. **训练命令应去掉 `--vllm_gen_logps`**，并恢复 `--verify_gen_logps N` 对拍。期望地板回到 doc 基线量级：clip_frac ~0.0007，approx_kl ~5e-4。

## 仍开放的验证

- **FlashInfer 档**：由于该 pod 上 FlashInfer GDN prefill JIT 会 OOM-kill（两次实锤，零成功），无法验证它是否也有同样的非确定性。理论上 FlashInfer 与 Triton 是不同 kernel 实现，不能外推。
- **其他模型 / 其他 vLLM 版本**：本结论仅限 Qwen3.5-4B + vLLM v0.19.1 + triton GDN prefill。3B 模型、非 GDN 模型、新版 vLLM 需单独验证。
- **训练是否真的走 triton**：`grep -n "GDN prefill kernel" train_log*.txt` 仍建议跑，以确认训练日志与诊断数据属于同一档。

## 纪律记录

- 2026-09-15 18:43 `--diff_traj` 给出 token 一致率 2.91% / max Δ=14.5 nat → 禁用 `--vllm_gen_logps` 成为显式决策。
- 相关代码/工具保留（`vllm_logprobs_n`、`--measure lpmode`、`--diff_traj`、`--build_logprobs_n`），用于未来其他模型/backend 的复现性普查，不删除。
