# Qwen3.5-4B 迁移的 GPU 显存 OOM 全记录（2026-09-11）

> 从 3B (Qwen2.5) 迁移到 4B (Qwen3.5) 的训练打通过程中，真机连续爆出 9 次
> 显存/资源类故障，共 12 个 commit 修复。本文按时间序记录每个炸点的
> **现象 → 根因 → 修复**，沉淀显存账本公式与通用方法论，避免换下一个
> 模型时把同样的账重新算一遍。
>
> 配套：迁移决策与协议层 checklist 见 `03-qwen35-4b-retool-math-checklist.md`。

## 0. TL;DR：最终定版

4B 训练命令（本机 2×H20 96G + RAM 60G 约束下的唯一可行组合）：

```bash
bash rlab/run_gsm8k.sh retool_math /root/Qwen3.5-4B-text \
    --vllm_model_path /root/Qwen3.5-4B \
    --chat_template_kwargs '{"enable_thinking": false}' \
    --round_gen_tokens 3072 \
    --gen_gpu_mem 0.30 \
    --micro_rows 1 \
    --optim_8bit \
    --difficulty_path rlab_out/difficulty_probe_4b_v4.jsonl
```

实测稳定态：GPU0 ~57G（vLLM + torch 副本 + ref）｜ GPU1 ~55-60G（8bit 优化器
方案）｜ CPU RAM ~50G 内。训练步 ~84-120s/it。

**为什么每个参数都必须存在**：见 §2 显存账本与 §4 逐项解释。去掉任何一个，
对应的炸点会原样复现（每一个都实测炸过）。

## 1. 炸点全记录（时间序）

### 加载层（非显存，但是同一条链的前置）

| # | 现象 | 根因 | 修复 | commit |
|---|---|---|---|---|
| A1 | `Qwen3_5Config.vocab_size MISSING`，torch 侧（gen_logps 副本/训练端/ref）全崩；vLLM 正常 | 官方 4B 是**原生多模态**复合 config（vocab_size 在 text_config），`AutoModelForCausalLM` 拿复合 config 喂 ForCausalLM 类 | `rlab/extract_text_model.py` 一次性抽纯文本 checkpoint（内置 logits 对拍自检） | `28562c6` |
| A2 | vLLM 反过来拒纯文本 checkpoint：processor 期望复合 config | 该版 vLLM 把 `qwen3_5_text` 也路由到多模态实现 | **分裂加载**：`--vllm_model_path` 原多模态给 vLLM、纯文本给 torch，同步走 `remap_text_to_multimodal` 键名映射（`model.X`→`model.language_model.X`，tied lm_head 丢弃——映射表经 vLLM 源码核实） | `dece268`+`4339e10` |

### 显存层

| # | 炸点 | 现象（数字） | 根因 | 修复 | commit |
|---|---|---|---|---|---|
| B1 | 生成端 gen_logps | 单次分配 **24.95G**；GPU0 vLLM 55.3G + ref 16G 共居 | 全量前向 logits (8, ~5.4k, **V=248320**) ≈ 22G + log_softmax 同尺寸；3B 时代 (8,1.7k,152k)≈4.2G 塞得下是侥幸 | `losses.forward_per_token_logps`：backbone 只产 hidden states，**时间维分块** lm_head+log_softmax+gather（logits 按位置独立，数学等价） | `88d2720` |
| B2 | GPU0 布局 | ref OOM 时 vLLM 进程实测 **68.4G**（0.45 标称 43G） | Qwen3.5 多模态实现 profiling/CUDA graph **超支 ~15G**，config 标称值≠进程实际占用 | `--gen_gpu_mem 0.30`（KV 需求仅 ~4G，混合架构注意力层占比 ~1/4，压配额不伤吞吐）+ `PYTORCH_ALLOC_CONF=expandable_segments:True` + ref chunk 256 | `830b389` |
| B3 | 训练端 forward | GPU1 **94.65G** OOM（崩在 deltanet fp32 cast，~18 层深度） | `from_pretrained` 默认 **eval 模式**，transformers 激活检查点要求 `self.training` 为真——`gradient_checkpointing_enable()` **静默失效**，backbone 全量激活保留（~46G） | `engine.module.train()`（一行；3B 时代同样没生效，只是激活小从未暴露） | `39e35a2` |
| B4 | ref 打分 | 单次分配 **20.17G** = (8, 32, 6503, 6503) 的 T² 矩阵；ref 进程缓存瞬态涨到 47.4G | head_dim=256 门控注意力在 PyTorch SDPA **回退 math 路径**物化 T²；3B T~1.7k 时仅 1.5G | `forward_per_token_logps` 加 **batch_chunk=1**（因果注意力按行独立，T² 瞬态 ÷B），gen/ref/train 三处统一 | `2c8c687` |
| B5 | 训练端 backward | **94.9G** OOM 于 checkpoint 重算（差 108M） | DS bf16 优化器全态静态 **64G**（fp32 master 16 + m/v 32 + bf16 权重/梯度 16），动态余量不足 | 先试 ZeRO-2 offload（`--zero_stage 2`） | `b543c12` |
| B6 | DS 初始化 | `pin_memory` → `cudaErrorInvalidValue` | 容器**锁页内存上限** | `offload_optimizer.pin_memory=False`（pageable 慢点能过） | `47f0ff8` |
| B7 | 训练端 step | **94.94G** OOM 于 `engine.step()`（差 90M），且 **empty_cache 无效** | 真账：静态 64G + fused optimizer step **临时分配 32G**（`p.grad.to(fp32)` 全参拷贝 16G + flatten 缓冲 16G）= **96G > 95G，数学上无解**。93G 是活张量不是缓存——empty_cache 只能还缓存块 | 单卡 stage0 无解，必须 offload（→B8） | `0c1bf92`+`47f0ff8` |
| B8 | CPU RAM | 进程被 **OS OOM-kill**（无 traceback，`Killed`），死在第 4 步 | offload 后 RAM：fp32 态 48G + **CPU step 梯度拷贝 16G** ≈ 64G > 60G 上限 | **bitsandbytes AdamW8bit**：m/v 量化 8bit（32G→8G）留 GPU，无 offload 无 RAM 依赖。GPU 静态 ~40G ✅ | `b36929c` |
| B9 | 训练动态 | backward/step 期间动态峰值 ~30G | 8 行批的检查点包+图共存（~10G）+ 重算瞬态 | `--micro_rows 1`：**按行拆 micro-backward**（前向 1 行→backward→图释放）。sample_mean 归一下 `Σ chunk_loss×(k/R)` 梯度与整批**严格等价**；其他 loss_norm fail-fast | `c1f85f5` |

时序注：B5-B8 是同一条优化器显存线的四轮迭代（offload 尝试 → pin_memory →
误诊缓存 → 真账无解 → RAM 爆 → 8bit 定案）；B9 与 B7/B8 并行（动态/静态两条线）。

## 2. 显存账本（公式表）

换模型时按此逐项重新核算，**每个公式的量级变了，对应的炸点就会复活**：

| 峰值项 | 公式 | 3B 实测 | 4B 实测 | 治理 |
|---|---|---|---|---|
| logits | `B×T×V×dtype`（log_softmax 翻倍） | 8×1.7k×152k ≈ 4.2G | 8×5.4k×248k ≈ **22G** | seq_chunk 分块（logps 按 token 独立） |
| 注意力矩阵（SDPA math 回退时） | `B×H×T²×dtype` | ≈1.5G | ≈**20G** | batch_chunk 逐行 / 换 flash-attn |
| DS bf16 stage0 静态 | `params×14B`（权重 2+梯度 2+fp32 master 4+m 4+v 4） | 42G | **56G**（进程实测 ~80G 含上下文/碎片） | offload（RAM 允许时）/ 8bit 优化器 |
| fused optimizer step 临时 | `params×8B`（fp32 梯度拷贝 4 + flatten 4） | 24G | **32G** | 同上（这是 B7"数学无解"的来源） |
| backbone 激活（无检查点） | `~L×B×T×H×c` | ~20G | **~80G** | HF gradient_checkpointing（**必须 train 模式才生效**） |
| 优化器 8bit 化后静态 | `params×10B`（权重 2+梯度 2+master 4+8bit 态 2） | — | **40G** | 本机最终方案 |
| CPU offload RAM | `params×12B`（master 4+m 4+v 4）+ step 梯度拷贝 `params×4B` | — | 48+16=**64G** | 本机 RAM 60G 不可行 |

关键换算：**bf16 训练的静态显存 ≈ params×14 字节，不是 ×4**——只算权重是
最常见的漏项；而"能不能跑"取决于**静态 + step 临时 + 动态峰值**三者之和，
缺一项都会在第 N 步炸（N 取决于缓存碎片运气，3 步还是 30 步而已）。

## 3. 走过的死路（防重蹈）

1. **ZeRO-2 offload 在本机不可行**（双重）：pin_memory 撞容器锁页上限
   （`cudaErrorInvalidValue`，可 `pin_memory=False` 绕过）；绕过后 CPU Adam
   step 仍要物化 fp32 梯度拷贝，48G 态 + 16G 临时 > 60G RAM，OS OOM-kill。
   RAM ≥ 80G 的机器此路可用（GPU 静态降到 16G，最宽裕）。
2. **`torch.cuda.empty_cache()` 治不了活张量**：B7 误诊为缓存碎片，实测
   92.98G 是"allocated by PyTorch"的活张量。empty_cache 只还"reserved but
   unallocated"；先看报错里这两个数字的比例再决定用哪招。
3. **vLLM 标称显存 ≠ 进程占用**：gpu_memory_utilization=0.45 实测 68.4G
   （多模态实现超支 ~15G）。共居卡的预算按 nvidia-smi 实测定，不按 config 抄。
4. **8bit 优化器的口径代价**：m/v 8bit 量化与 3B fp32 AdamW 不严格同口径——
   4B 实验系列内自洽，跨系列对比须声明（lr 1e-6 × 200 步教学规模下偏差可忽略）。

## 4. 最终配置逐项解释

| 参数 | 治理的炸点 | 不加会怎样 |
|---|---|---|
| `model_path=-text 目录` | A1 | torch 三处全崩 |
| `--vllm_model_path 原多模态` | A2 | vLLM 拒纯文本 checkpoint |
| `--chat_template_kwargs '{"enable_thinking": false}'` | 协议层（非显存） | thinking 烧穿单轮预算，截断 98.9%，难度分布失真 |
| `--round_gen_tokens 3072` | 协议层 + 显存（T 进入所有公式） | 1024 下截断 85%（"无代码即终局"使 max_rounds 成虚假预算） |
| `--gen_gpu_mem 0.30` | B2 | vLLM 实测超支 ~15G 挤爆 GPU0 共居 |
| `--micro_rows 1` | B9 | 动态 ~30G 叠加在静态上顶满 GPU1 |
| `--optim_8bit` | B7/B8 | fused fp32 96G>95G 数学无解；offload RAM 爆 |
| （不用 `--zero_stage 2`） | B6/B8 | 本机锁页上限 + RAM 60G 双重不可行 |

## 5. 通用方法论

1. **显存账按"峰值时刻的活张量"算**，三份账分开：静态（参数/优化器态）、
   step 瞬态（优化器物化）、逐 token 动态（激活/logits/注意力）。任何一份
   漏算都会表现为"能跑几步然后死"。
2. **报错数字先分诊**：`allocated by PyTorch` 是活张量（要改方案），
   `reserved but unallocated` 是缓存碎片（empty_cache / expandable_segments
   能治）。两者比例决定用药。
3. **"开了 X"和"X 在生效"是两回事**：激活检查点被 eval 模式静默废掉
   （B3）——大坑不报错，只在 nvidia-smi 的数字里。开了开关就打一行日志
   验证标志位。
4. **3B 能跑 ≠ 4B 能跑**：每个 O(·) 公式里 3B→4B 的放大倍数不同
   （V×1.63、T×3.2、params×1.33），倍数最高的公式先炸。
5. **torch/vLLM 双实现的模型要两头单独验证**：vLLM 能跑只证明它的实现 OK
   （A1/A2 两炸都发生在另一头）。
6. **方案穷举顺序**：同卡压缩（分块/检查点/行拆）→ 跨介质（offload）→
   改数值精度（8bit）。每一步都有代价（吞吐/RAM/口径），按代价从小到大试。
7. **修一处=修三处**：gen/ref/train 三处前向共享 `forward_per_token_logps`，
   新的分块维度（batch_chunk）一次接入三处。

## 6. 未做的后续优化（按需启用）

- **flash-attn**：head_dim 256 走 flash 核后 T² 物化消失，可撤 batch_chunk=1
  行拆（训练前向恢复 8 行批，吞吐回升）——`pip install flash-attn` +
  `_attn_implementation="flash_attention_2"`。
- **vLLM `enforce_eager`**：GPU0 若再紧，砍 CUDA graph 内存（吞吐降 20-30%）。
- **纯生成协议消融**：v4 探针样本 code_ok≈0（4B 纯 prose 直接解）——若正式
  跑确认 code 恒 0，可砍沙箱/多轮/mask 整套，单轮协议下 T 直接 = max_gen_tokens，
  上述所有分块参数随之简化。
- **bns/bnb 版本升级**：AdamW8bit 的 paged 变体（`PagedAdamW8bit`）可进一步
  抗碎片，当前未需要。
