# -*- coding: utf-8 -*-
"""rlab/config.py — 全部超参集中管理。

约定（与 docs/RL进阶学习实验规划.md 附录B 的公共实验协议一致）：
- batch32 / 300 optimizer steps / kl 0.01 / lr 1e-6 为默认对照配置；
- 任何单变量实验只改这里的一个字段，其余保持不动。

用法：
    from rlab.config import get_config
    cfg = get_config(algo="dapo", model_path="/root/Qwen2.5-3B")
"""

import copy
import os

# 各算法的默认差异项（其余超参全部继承 BASE）
ALGO_DEFAULTS = {
    # GRPO 原版：对称 clip、样本级归一化、组内 std 标准化
    "grpo":    dict(beta=0.04, clip_low=0.2, clip_high=0.2, adv_mode="group_std",
                    loss_norm="sample_mean"),
    # DAPO：clip-higher 解耦、token 级归一化、dynamic sampling（rollout 侧）、软截断惩罚
    "dapo":    dict(beta=0.04, clip_low=0.2, clip_high=0.28, adv_mode="group_std",
                    loss_norm="token_mean", dynamic_sampling=True, overlong_shaping=True),
    # Dr.GRPO：去掉 1/std 偏差（group_mean），去掉长度归一化偏差（固定常数除）
    "dr_grpo": dict(beta=0.04, clip_low=0.2, clip_high=0.2, adv_mode="group_mean",
                    loss_norm="token_const"),
    # CISPO：clip(ratio) 作 sg 权重、梯度经 logπ 流动（MiniMax-M1）；token 级
    "cispo":   dict(beta=0.04, clip_low=0.2, clip_high=0.2, adv_mode="group_std",
                    loss_norm="token_mean"),
    # GSPO：sequence 级 importance ratio 与 clip
    "gspo":    dict(beta=0.04, clip_low=0.2, clip_high=0.2, adv_mode="group_std",
                    loss_norm="seq_mean"),
    # RF++：全局基线（非组内）、per-token advantage、token 级、无 KL
    "rfpp":    dict(beta=0.0,  clip_low=0.2, clip_high=0.2, adv_mode="global_mean",
                    loss_norm="token_items"),
    # 阶段2 ReTool：多轮代码交织（loss 与 grpo 同——group_std/sample_mean，
    # 差异全在 rollout：分段轨迹 + 沙箱 + 工具段 mask 置0，见 docs/02-retool.md）
    "retool":  dict(beta=0.04, clip_low=0.2, clip_high=0.2, adv_mode="group_std",
                    loss_norm="sample_mean"),
    # 方案1：retool-math（借鉴 agentic-rl-lab/05-retool）—— DAPO-Math-17k + outcome-only
    # boxed + clip-higher（0.2/0.28）+ 参考实现的采样与组配置：
    # temperature=1.0、无 top_k（vLLM top_k=-1=全词表）、组 8 条、adv 不除 std。
    # 【2026-09-09 对齐修改，两处联动】①0.7/top_k=50 是 GSM8K 格式学习时代的遗产
    # （当时为把格式信号从 10% 抬到 27%），math 系 outcome-only 毫无格式压力，
    # 保守采样只剩副作用（压探索→组内更易全错→零方差丢弃）；②num_pre_Q 4→8 必须
    # 配 adv group_std→group_mean（参考组内减均值不除 std；4 条小 group 下除 std
    # 放大噪声，8 条才配用不除 std 的形态）。
    # 仍与参考不同的三处（报告须注明）：loss_norm 保持 sample_mean、保留 KL β=0.04、
    # Q_batch_size=1（参考每 step 8 题×8 条=64 条；我们每 upload 1 题×8 条，
    # 有效 batch=8×grad_accum4=32 样本/optimizer step，旧协议是 16）。
    # 【2026-09-12 预算重定·必读】旧 preset（round_gen_tokens=1024）+ 探针口径的
    # CLI 覆盖（--round_gen_tokens 3072）合起来是 `3×3072 = 9216 > max_context_tokens
    # = 8192`——单轮预算比全轨迹保险丝还大，于是"用满预算"物理上不可达，
    # retool_context_overlong 把 86% 的尝试整组丢掉（丢弃率 20%→90%），
    # 采样主循环空转、训练端 5 小时零产出；同时 37% 样本 clen 顶满 3072、
    # 44% 样本末段被轮长切断（trunc=1 的 acc 只有 5.0%）。
    # 定版原则：**先保证预算自洽，再谈长度控制**。约束由
    # config.validate_retool_budget() 强制（不满足直接 raise，防再次静默上线）：
    #     max_rounds × round_gen_tokens + max_prompt_length + 工具段预留
    #         ≤ max_context_tokens
    # 本 preset 代入：2×3072 + 1024 + 1×266 = 7434 ≤ 8192 ✅（余量 758）
    # 为什么是 max_rounds 2 而不是 round_gen_tokens 2048：
    #   ① 精度杀手是 trunc_final（末段被**单轮**上限切断），它只跟 round_gen_tokens
    #      有关；降到 2048 会让 trunc 更糟（探针已实测 1024 下截断 85%）。
    #   ② max_rounds 对本模型是"虚假预算"（docs/03 v4 探针：无代码即终局，
    #      有效单轨迹预算 = round_gen_tokens）——砍掉一轮不损失 prose 路径，
    #      换来的 3072 token 余量让"合法轨迹永不被 overlong 丢"。
    #   ③ 保留 TIR 最小闭环：round1 写代码 → 沙箱执行 → round2 出 final 答案。
    #   想恢复 2 次代码执行（3 轮）请改 max_rounds=3 + round_gen_tokens=2048，
    #   并接受 trunc_final 上升（本轮已有健康签名会报警）。
    "retool_math": dict(beta=0.04, clip_low=0.2, clip_high=0.28, adv_mode="group_mean",
                        loss_norm="sample_mean", data_task="dapo_math",
                        num_pre_Q=8, train_micro_batch_size_per_gpu=8,
                        temperature=1.0, top_k=-1,
                        gen_questions_per_attempt=4, sandbox_workers=8,
                        max_context_tokens=8192, round_gen_tokens=3072,
                        max_rounds=2,
                        max_gen_tokens=8192, max_prompt_length=1024,
                        # 长度控制：overlong 现在可达（预算自洽）→ 打开；再叠一个
                        # 靶向 trunc_final 的项（prose 路径唯一够得到的反向信号）。
                        overlong_shaping=True, overlong_buffer=256,
                        trunc_shaping=0.5,
                        code_w=0.0, reward_switch_step=1000000000,
                        # 【2026-09-11 eval 全灭事故】思考开关收进 preset 单点同源：
                        # Qwen3.5 默认 enable_thinking=True，eval 端 prompt 构造从
                        # config 取此值——此前只在训练 CLI 传，eval 拿到 None，
                        # 生成以 <think> 开头烧穿预算，fmt/acc 双灭（base 同灭，
                        # 证明是协议问题非权重）。训练 CLI --chat_template_kwargs
                        # 仍可覆盖；GSM8K 家族（Qwen2.5）模板忽略该上下文键。
                        chat_template_kwargs={"enable_thinking": False}),
}

BASE = dict(
    # ---- 模型与路径 ----
    model_path="/root/Qwen2.5-3B",          # base 版，与历史实验可比
    data_task="gsm8k",                       # gsm8k（阶段0/1）；阶段2/3 扩展
    out_dir="./rlab_out",
    record_path="./rlab_out/record.jsonl",   # 生成数据得分记录（analysis.py 消费）
    # 【2026-09-10 4B 探针实锤】apply_chat_template 附加 kwargs（None=不传，Qwen2.5 行为不变）。
    # Qwen3.5 系必须 {"enable_thinking": false}：thinking 模式把单轮 1024 token 预算
    # 烧在 <think> 长链上（4B 探针实测：末段截断 98.9% / 无 boxed 99.2% / 全错 484/490，
    # 换算可学带仅 1.2%——全是协议失败不是能力失败）。官方实测思考模式下"很少写代码"。
    # Qwen2.5 模板不引用该 jinja 变量，传入无副作用。
    chat_template_kwargs=None,
    # 【2026-09-11 多模态 Qwen3.5 分裂加载】vLLM 生成用的 checkpoint 路径
    # （None=model_path 同一份）。Qwen3.5-4B 官方权重是多模态复合体：vLLM 只认
    # 多模态版（纯文本 qwen3_5_text 被它路由到多模态实现崩），torch 侧只能加载
    # extract_text_model.py 抽出的纯文本版 → model_path=纯文本（torch 三处加载）、
    # vllm_model_path=原多模态，权重同步经 sync.remap_text_to_multimodal 映射键名。
    vllm_model_path=None,
    # 【2026-09-11 4B OOM】DeepSpeed zero stage（0=默认，3B 全态 ~60G 历史可比）。
    # 4B bf16 优化器全态 = fp32 master+m+v ~48G + bf16 权重/梯度 16G ≈ 64G 静态，
    # 动态（检查点包+重算瞬态+math 注意力 T²）顶满 95G——第一步 backward 差
    # 108M 都放不下。4B 传 --zero_stage 2：优化器态 offload 到 CPU RAM（~48G），
    # GPU1 静态降到 ~24G；单卡训 step 稍慢（CPU 优化器），吞吐占比小可接受。
    zero_stage=0,
    # 【2026-09-11 4B OOM】训练步按行拆 micro-backward（0=整批一次 backward，3B 不变）。
    # 背景见 train.py 训练步注释：DS bf16 优化器全态把 GPU1 逼到 ~80G 静态，本机
    # RAM 60G（offload 放不下 + DS pin_memory 撞容器锁页上限）——只能砍"8 行图
    # 共存"。sample_mean 归一下 Σ chunk_loss×(k/R) 梯度与整批严格等价；
    # 其他 loss_norm 会在 train.py fail-fast（批内归一跨 chunk 不等价）。
    micro_rows=0,
    # 【2026-09-11 4B OOM】8-bit 优化器（bitsandbytes AdamW8bit，需 pip install）。
    # 本机约束下的最后一条路：GPU 上 stage0 fused（静态 64G+step 临时 32G=96G>95G
    # 数学无解）；CPU offload RAM 60G 也爆（态 48G + step 梯度拷贝 16G 被 OOM-kill）。
    # 8bit 态 32G->8G 留 GPU：静态 ~40G + 动态 ~15G ≈ 55G ✓。代价：优化器数值
    # 与 3B 的 fp32 AdamW 不严格同口径（8bit 状态量化），4B 实验系列内自洽即可
    # （lr 1e-6 × 200 步的教学规模下偏差可忽略，需在报告里声明）。
    optim_8bit=False,

    # ---- 数据采集 ----
    Q_batch_size=1,          # 每次 rollout 的题目数（grpo_dapo 断言=1）
    num_pre_Q=4,             # 每题采样条数；H20 显存实测 4 安全
    max_prompt_length=400,   # 提示词超长直接放弃本组（防 OOM）
    max_gen_tokens=512,      # 生成长度上限
    # 【2026-09-05 格式率根因修复】temp=0.9 下 base 格式率仅 ~10%（探针 48条/组：
    # 0.9→10.4%, 0.7→27.1%, 0.6→37.5%, greedy≈49%大样本）。起头分布显示 56% 概率
    # 直接跳过 think 标签答题（'T' 开头）——格式是"窄路径"，采样温度放大跳过率。
    # 后果：训练期格式信号被高频的"非格式但对"(+1) 淹没 → group_mean/global_mean
    # (dr_grpo/rfpp) 把格式打到 eval 0%；仅 group_std 靠离群放大勉强点火
    # (cispo 训练期 2.2%→eval 63%, gspo 10.1%→71%)。
    # 老 RF++ temp=0.7+topk=-1 格式率 ~35% 故能学到 99%——统一降 0.7：
    # 与老 rf++ 可比 + 格式信号充足 + 保留探索性。eval 用 greedy，不受影响。
    temperature=0.7,
    top_p=1.0,
    # 【2026-09-04 缺口根因】HF GenerationConfig 默认 top_k=50，老脚本没显式传就用了 50；
    # vLLM SamplingParams 默认 top_k=-1（全词表采样，尾部更重、更多退化解）。
    # 这与 loss 归一化产生算法特异的交互：DAPO 的 token-mean 让长退化样本按 token 数
    # 拿到更大梯度权重（GRPO 的 sample-mean 每条样本等权，对尾部不敏感）——
    # 解释了"GRPO 跨实现复现一致、唯独 DAPO 掉 4pp"。必须与老脚本逐字对齐。
    top_k=50,
    dynamic_max_attempts_mult=5,   # dynamic sampling 尝试上限 = 需要 组数*该倍数
    # 题目级动态采样（仅 retool 家族启用，单轮路径不用以保持阶段0/1 协议可比）：
    # 连续 q_skip_streak 次产出零方差组（全错/全对，无梯度）的题跳过；
    # 过滤后池子 < q_pool_reset_floor 时全部重置（难题随模型变强重新入场）。
    # 动机：retool_math 实测丢弃率 81%（p≈5%），且动态采样的期望轨迹成本 ≈1/p
    # 与 num_pre_Q 无关——只有题目级过滤能真正削减白跑。
    q_skip_streak=2,
    q_pool_reset_floor=64,
    # 生成端每次 attempt 并采题数（2026-09-10 vLLM 利用率修复）。=1 保持旧
    # 逐题协议（GSM8K 家族可比性）；>1 时走 QuestionScheduler 队列路径（题目
    # 过滤真正生效，见 rollout.py）并按题拆分上传——训练端 micro-batch 契约
    # （=num_pre_Q 行/批）不变。retool_math=4 → vLLM 每轮并发 4×8=32
    # （旧值 8，H20 3B 严重欠利用；参考实现为 8 题×8 条=64）。
    gen_questions_per_attempt=1,

    # ---- 离线难度预探测（2026-09-10，probe_difficulty.py 配套，仅生成端消费）----
    # 训练前用 base 模型对全池每题采 k 条估计通过率（rlab/probe_difficulty.py，
    # 产出 jsonl 表）；difficulty_path 设置后生成端只保留通过率在 band 内的题。
    # 动机：丢弃率 81% 的主体是全错组 (1-p)^8——期望轨迹成本 ≈1/p 只能靠改分布
    # 切割；与在线 QuestionScheduler 互补（静态出清"base 从未做对过的题"，在线
    # 出清"当前学不动的题"，两级都不碰 loss 协议）。
    # 注意：静态过滤随模型变强会误伤（难题永久出局）——模型显著变强后用新
    # checkpoint 重跑探针即可刷新表；过期的表不自动失效，换模型训练时留意。
    difficulty_path=None,        # 探针表路径；None=不过滤（阶段0/1 与旧 run 行为不变）
    difficulty_band=(0.0, 1.0),  # 保留 n_correct/k 严格落在开区间 (lo, hi) 的题

    # ---- 训练 ----
    # 【2026-09-11 提速】torch 侧三处加载（train/gen 副本/ref_server）共用的注意力
    # 实现，默认 sdpa 与 3B 历史完全可比。Qwen3.5 的 head_dim=256 门控注意力在
    # SDPA 走 math 回退物化 T²（docs/04 B4），整条链被迫 batch_chunk=1；
    # flash_attention_2 支持大 head_dim，T² 消失后训练端可放开 --micro_rows。
    # 注意：FA2 仅支持 bf16/fp16 —— ref_server 的 fp32 ref 在该档位自动降 bf16
    # （口径：ref logps 精度略降，教学规模可忽略，报告声明）。vLLM 有独立的
    # attention backend 选择，不受本参数影响。需 pod 上 pip install flash-attn。
    attn_implementation="sdpa",
    # 【2026-09-11 放开】分块前向的"每次几行过 backbone"粒度，四处共用
    # （train 两条路径 / gen_logps 副本 / ref_server）。1 = 逐行（3B 时代为躲
    # T² 与 logits 峰而固定，历史口径）；>1 一次前向多行，减少 kernel 次数与
    # 调度开销——数学严格等价（因果注意力按行独立，CPU 对拍锁定）。
    # 显存代价：logits 峰与激活随 B 线性增长。FA2 档位 T² 已消失，4B 可试 2~4。
    # ref_server 是独立进程读不到 cfg：run_gsm8k.sh 用同一个 FWD_BATCH_CHUNK
    # 环境变量把它传给三处（CLI --fwd_batch_chunk > 环境变量 > 本默认值）。
    fwd_batch_chunk=1,
    all_steps=300,
    save_steps=100,
    gen_update_steps=16,     # 每 N 个 optimizer step 推送权重给生成端
    train_micro_batch_size_per_gpu=4,   # = Q_batch_size*num_pre_Q
    gradient_accumulation_steps=4,
    lr=1e-6,
    warmup_steps=0,

    # ---- 基础设施 ----
    gen_device=0,            # vLLM 生成 + torch gen_logps 副本所在物理卡
    gen_gpu_mem=0.45,        # 生成端 vLLM 显存占比（GPU0 = ref~7G + vLLM + 副本~7G
                             # + logits 瞬时峰~12G，0.45×96 总计 ~70G < 96G；2026-09-09
                             # 提速：旧 0.35 的 KV 池对 3B+GQA 大量闲置）
    ref_server_host="localhost",
    ref_server_port=59875,
    wandb_project="rlab",
    wandb_name=None,         # 默认 = algo 名
    use_wandb=True,

    # ---- loss 公共项（各算法 preset 会覆盖部分）----
    beta=0.04,
    clip_low=0.2,
    clip_high=0.2,
    adv_mode="group_std",    # group_std | group_mean | global_mean
    loss_norm="sample_mean", # sample_mean | token_mean | token_const | seq_mean | token_items
    dr_grpo_const=None,      # token_const 的固定常数，默认=max_gen_tokens
    dynamic_sampling=False,
    # 【2026-09-12 预算自洽后重新可达】overlong_penalty 以 completion 总长对
    # `overlong_ref_tokens()` 起坡。旧配置（3×3072=9216 > max_context_tokens=8192）
    # 下它是死开关：retool_context_overlong 会先丢弃 len(ids)+plen > 8192 的样本，
    # clen 物理封顶 ~7800 < trigger 9152 → 永远够不着。现在由
    # `validate_retool_budget()` 在 get_config 阶段强制
    # `max_rounds×round_gen_tokens + max_prompt_length + 工具段预留 ≤ max_context_tokens`，
    # 合法轨迹再也不会被丢，shaping 成为**唯一**的长度控制 → 打开它。
    overlong_shaping=False,
    overlong_buffer=64,      # DAPO 软悬崖缓冲区宽度
    # 【2026-09-12 靶向 shaping·长度膨胀的真正出口】末段被轮长上限切断
    # （trunc_final=1）的额外扣分。为什么不能只靠 overlong_shaping：completion
    # 总长惩罚够不到"单轮就结束"的 prose 轨迹（clen ≤ round_gen_tokens < trigger）。
    # 而 200 步 run 实测：trunc=1 的 458 条 acc 仅 5.0%，trunc=0 的 582 条 64.3%，
    # 44% 样本撞在轮上限上——长度膨胀杀伤精度的直接出口就在 trunc_final。另外
    # 组内"更长的那条答对率 75.4%"（corr(clen,acc)=+0.25~0.43）意味着纯 ±1 奖励
    # 会把长度当正确性的代理来强化，必须有反向项。
    # 量纲：reward 域 ±1，扣 0.5 → +1 变 +0.5、-1 变 -1.5，不改正负号、不破坏
    # group_mean 的减均值语义；评测端不用 reward，eval 口径零影响。
    # 0.0 = 关闭（其余算法协议零变化）；retool_math preset 开 0.5。
    trunc_shaping=0.0,
    # ---- 采集端反压（2026-09-12 事故）----
    # 事故形态：模型变长 → 86% 尝试被判 overlong 丢弃 → 而 overlong 既不推进
    # 题目 streak 也不触发任何熔断 → 采样主循环 `while True` 对着同一批题无限
    # 空转（log 里 18274 行 "waiting for batch..."，末尾 5 小时零产出）。
    overlong_counts_toward_skip=True,  # 超长是否计入题目级 streak（防整池原地打转）
    sampler_max_zero_yield=6,   # 连续 N 个外层轮次零产出 → fail-fast（0=关闭）
    discard_alert=0.50,         # 窗口丢弃率告警阈值（0=关闭）
    discard_abort=0.90,         # 窗口丢弃率熔断阈值（0=关闭）
    gradient_clipping=0.0,   # DeepSpeed 梯度裁剪（0=不裁剪=历史口径；4B 大 lr 建议 1.0）
    # 【减法① 2026-09-11】gen_logps 来源（仅 retool 家族有效）：
    #   False = torch 副本在拼接序列上重算（历史口径；GPU0 多占 ~8G + 每步一次全序列前向）
    #   True  = 逐轮 vLLM 采样 logprobs（SamplingParams(logprobs=0) 返回被采样 token 的
    #           logprob）。多轮下每轮请求的上下文已含之前所有工具结果，故逐轮收集拼接
    #           == 在拼接序列上重算，数学严格同义，且完全绕开 prompt_logprobs（那条
    #           路径在本环境会 hang，正是 torch 副本存在的起因）。
    # 口径变化：logps 分母由 torch kernel 换 vLLM kernel（同为 bf16）——首次启用必须
    # 用 verify_gen_logps 对拍并把最大差写进报告。
    vllm_gen_logps=False,
    verify_gen_logps=0,      # >0：前 N 组同时算两路并打印最大差（临时加载 torch 副本，验完释放）

    # ---- 系统提示（与 simple_grpo_v1 完全一致，保证可比）----
    system_prompt=(
        "You are a helpful assistant. A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. The Assistant first "
        "thinks about the reasoning process in the mind and then provides the user "
        "with the answer. The reasoning process and answer are enclosed within "
        "<think> </think> and<answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>."
    ),

    # ---- 阶段2 ReTool（代码交织多轮）----
    # max_rounds = 最多生成轮数，其中只有前 max_rounds-1 轮执行代码（最后一轮
    # 保证是 final 答案轮——末轮代码执行结果无人消费，2026-09-09 审查修复：
    # 旧版末轮执行 code_ok 还记分，模型却永远没机会读结果作答）。
    max_rounds=3,            # = 2 次代码-执行-续写 + 1 次 final 生成
    # 【2026-09-08 代码灭绝教训】280 太紧：代码块约占 80-150 token，写代码的样本
    # 极易在轮内写不完围栏 → 完整块检测不到 → 残缺结尾 → fmt=-1 且无答案——
    # "写代码"被结构性惩罚、几步内灭绝（真机两轮 code_rate=0 的根因）。
    # 400 下 2 轮代码×400+工具输出+1 轮 final 400 ≈ 1650 仍在 2200 预算内
    # （预算按全长口径计，含工具段 token，2026-09-09 修复）。
    round_gen_tokens=400,    # 每轮 assistant 段生成长度上限（控制总上下文）
    tool_result_max_chars=500,  # 沙箱输出截断长度（防输出炸弹）
    sandbox_workers=4,       # 沙箱线程池并发（subprocess 线程安全；retool_math
                             # 并采 32 条/轮，覆盖为 8 防 spawn 串行段）
    sandbox_timeout=5.0,     # 代码执行超时（秒），超时 SIGKILL 子进程
    sandbox_mem_mb=256,      # 代码内存上限（Linux RLIMIT_AS，best-effort）
    max_context_tokens=2200, # 全轨迹（prompt+各段）上限，超限整组丢弃防 OOM
    code_w=0.1,              # 代码可用率小权重（每个成功代码块 +code_w）
    reward_switch_step=256,  # 冷启动/后期奖励权重切换点（optimizer step；
                             # Auto_Program 16次权重推送*16步=256）
    reward_cold_w=(1.0, 2.0, 2.0),   # 冷启动权重 (w_acc, w_fmt, w_code)：先学格式+代码
    reward_hot_w=(2.0, 1.0, 1.0),    # 后期权重：正确性主导（与阶段1 的 2*acc+fmt 对齐）

    # ---- 可复现种子（None=旧行为不设种子；设了则抽题顺序与生成采样均可复现）----
    # 【2026-09-04 教训】dapo 同代码重跑 78.0→74.3(-3.7pp)：±2pp 噪声地板只覆盖
    # "同 checkpoint 评两次"的评测噪声，从未覆盖训练运行间方差（抽题顺序+生成采样无种子）。
    # 阶段1 起对比实验一律固定 seed，必要时双 seed 复跑。
    seed=None,
)

# 阶段2 retool 系统提示 = 基础格式提示 + 代码工具说明（复用 BASE["system_prompt"]
# 保证格式口径与阶段0/1 完全一致；新增部分零标签字面量，规避改写铁律）
# 【2026-09-08 第三/四轮教训·生成层冲突】"MAY"（可以写）在 base 上的代码采样率
# 仅 ~0.3%（greedy 实测）→ 组内 0 人写代码 → code 奖励项常数无梯度。改 MUST 后
# 代码采样有了，但 MUST"first write the computation as code, then reason"把输出
# 顺序改成"代码先行"→ base 开围栏后不再产出思考/回答标签 → 即便打分剥离代码，
# 没有标签就是没有格式 → fmt 恒 -1 信号死亡（健康检查 32 组实锤，128 样本 ~0 合规）。
# 修复：指令顺序改为"先打开思考段推理 → 计算写代码 → 继续推理 → 答案标签收尾"，
# 与格式契约（thinking 先行 / answer 收尾）对齐——代码留在思考段内，剥离后格式
# 语义天然成立，代码探索（MUST）与格式生成（先思考）两者兼得。
_RETOOL_EXTRA = (
    "\n\nYou MUST write Python code to help solve the problem. Follow this order: "
    "first open the thinking section and do your reasoning there; whenever the "
    "question involves a calculation, write that computation as code inside a "
    "fenced block like: "
    "```python\n<your code>\n```\n"
    "The environment executes your code automatically and inserts the result "
    "between [TOOL RESULT] and [/TOOL RESULT]. Read the result and continue "
    "reasoning in the thinking section, then finish with the final answer "
    "inside the required answer tags. Always finish your code block before continuing."
)
system_prompt_retool = BASE["system_prompt"] + _RETOOL_EXTRA

# 方案1：retool-math 系统提示（借鉴 agentic-rl-lab/05-retool）—— outcome-only \boxed{}
_RETOOL_MATH_SYSTEM = (
    "You solve math problems step by step with help from a Python code interpreter.\n"
    "Use the code_interpreter tool when calculation, symbolic manipulation, or enumeration helps you solve the problem accurately and quickly.\n\n"
    "How to use the code_interpreter tool:\n"
    "- Call it with Python code inside a fenced block like: ```python\n<your code>\n```\n"
    "  The environment executes your code automatically and inserts the result between [TOOL RESULT] and [/TOOL RESULT].\n"
    "- Results are captured from what your code prints with print(). Always print the values you want to see.\n"
    "- Each execution is independent: no variables, files, or state carry over between calls. Redefine everything you need in each piece of code.\n"
    "- Code must finish within a few seconds and use little memory. Do not read or write files. If you enumerate or brute-force, keep the search space small.\n"
    "- If the execution returns an error, analyze it and retry with corrected code when useful.\n\n"
    "When you have the final answer, end with exactly one line in this format:\n"
    "\\boxed{<your final answer>}\n"
    "Do not put the final answer inside the code block."
)
system_prompt_retool_math = _RETOOL_MATH_SYSTEM


def validate_retool_budget(cfg: dict) -> int:
    """多轮预算自洽校验（纯函数，CPU 可测）。返回工具段预留 token 数。

    【2026-09-12 事故固化】旧配置 `max_rounds=3 × round_gen_tokens=3072 = 9216`
    超过了全轨迹保险丝 `max_context_tokens=8192`：单轮预算比整条轨迹的丢弃线
    还大 → "合法地用满预算"物理上不可能 → `retool_context_overlong` 把 86% 的
    尝试整组丢弃 → 丢弃率 20%→90% → 采样主循环空转、训练端 5 小时零产出。
    这类错误在真实 run 里只表现为"丢弃率慢慢爬"，不报错、不崩，必须在这里
    硬拦：**任何一轮用满预算的轨迹 + 最长 prompt + 全部工具段，都必须落在
    max_context_tokens 之内。**

    工具段预留：每轮工具输出 ≤ tool_result_max_chars 字符，按最坏 2 字符/token
    （ASCII 密集 token 的保守下界）折算，加 TOOL_START/END 标记开销。
    """
    if not cfg.get("algo", "").startswith("retool"):
        return 0
    rounds = int(cfg.get("max_rounds", 1) or 1)
    per_round = int(cfg.get("round_gen_tokens", 0) or 0)
    if per_round <= 0:
        return 0
    ctx = int(cfg.get("max_context_tokens", 0) or 0)
    if ctx <= 0:
        return 0
    reserve = (rounds - 1) * (int(cfg.get("tool_result_max_chars", 0) or 0) // 2 + 16)
    need = rounds * per_round + int(cfg.get("max_prompt_length", 0) or 0) + reserve
    if need > ctx:
        per_round_max = (ctx - int(cfg.get("max_prompt_length", 0) or 0) - reserve) // rounds
        raise ValueError(
            f"[config] retool 预算不自洽：max_rounds({rounds}) × round_gen_tokens({per_round})"
            f" + max_prompt_length({cfg.get('max_prompt_length')}) + 工具段预留({reserve})"
            f" = {need} > max_context_tokens({ctx})。\n"
            f"  后果：用满预算的合法轨迹会被 retool_context_overlong 整组丢弃"
            f"（2026-09-12 事故：丢弃率 90%、采样空转、训练端零产出）。\n"
            f"  改法（三选一）：round_gen_tokens ≤ {per_round_max}"
            f"（会加剧末段截断，trunc_final 上升）；或降 max_rounds；"
            f"或抬高 max_context_tokens ≥ {need}（T 进入所有显存公式，需重算峰值）。")
    return reserve


def get_config(algo: str, **overrides) -> dict:
    """合并 BASE + 算法 preset + 显式覆盖，返回冻结配置 dict。"""
    if algo not in ALGO_DEFAULTS:
        raise KeyError(f"未知算法 {algo!r}，可选: {sorted(ALGO_DEFAULTS)}")
    cfg = copy.deepcopy(BASE)
    cfg.update(copy.deepcopy(ALGO_DEFAULTS[algo]))
    cfg["algo"] = algo
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"未知配置项 {k!r}")
        cfg[k] = v
    # retool 专用系统提示（除非用户显式覆盖）
    if algo == "retool" and "system_prompt" not in overrides:
        cfg["system_prompt"] = system_prompt_retool
    if algo == "retool_math" and "system_prompt" not in overrides:
        cfg["system_prompt"] = system_prompt_retool_math
    if cfg["wandb_name"] is None:
        cfg["wandb_name"] = f"{algo}"
    # 输出目录按算法隔离（防 grpo/dapo 的 step_N checkpoint 与 record 互相覆盖）
    if "out_dir" not in overrides:
        cfg["out_dir"] = os.path.join(cfg["out_dir"], algo)
    if "record_path" not in overrides:
        cfg["record_path"] = os.path.join(cfg["out_dir"], "record.jsonl")
    cfg["ref_server"] = f"http://{cfg['ref_server_host']}:{cfg['ref_server_port']}"
    # FWD_BATCH_CHUNK 环境变量统一驱动三处（train/gen/ref_server）——ref_server 是
    # 独立进程读不到本 cfg，只能靠共享环境变量保持全链路口径一致；显式 override 优先。
    _env_bc = os.environ.get("FWD_BATCH_CHUNK")
    if _env_bc and "fwd_batch_chunk" not in overrides:
        cfg["fwd_batch_chunk"] = int(_env_bc)
    # 多轮预算自洽（fail-fast；见 validate_retool_budget 的事故说明）
    cfg["_tool_reserve"] = validate_retool_budget(cfg)
    return cfg


def ds_config(cfg: dict) -> dict:
    """DeepSpeed 配置。ZeRO stage 0：3B 全态 ~60G < 96G，不开 offload 防 CPU OOM/-9。"""
    return {
        "train_micro_batch_size_per_gpu": cfg["train_micro_batch_size_per_gpu"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "steps_per_print": 5,
        # optim_8bit 时省略 optimizer 段——DS 由调用方传入 bitsandbytes 优化器对象
        **({} if cfg.get("optim_8bit") else
           {"optimizer": {"type": "AdamW", "params": {"lr": cfg["lr"]}}}),
        # 0.0 = 不裁剪（3B 时代历史口径，零变化）；大 lr 长跑建议 1.0
        "gradient_clipping": float(cfg.get("gradient_clipping", 0.0)),
        "bf16": {"enabled": True},
        # stage 2 + offload：fp32 优化器态驻 CPU（4B 专用；stage 0 路径零变化）。
        # 【为什么必须 offload】GPU 上无论怎么省都不够：静态 64G（fp32 master 16
        # + m/v 32 + bf16 权重/梯度 16）+ fused optimizer step 的临时分配 32G
        # （p.grad.to(fp32) 全参拷贝 + flatten 缓冲）= 96G > 95G，数学上无解。
        # 【pin_memory=False】本机容器锁页上限实测撞死（cudaErrorInvalidValue），
        # 走 pageable 内存慢一点但能过；RAM 预算 fp32 态 48G + 运行时 ~6G
        # ≈ 54G < 60G 上限。
        "zero_optimization": {
            "stage": cfg.get("zero_stage", 0),
            **({"offload_optimizer": {"device": "cpu", "pin_memory": False}}
               if cfg.get("zero_stage", 0) >= 2 else {}),
        },
    }
