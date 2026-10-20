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

# 【2026-09-25 原生协议（docs/09）】工具协议档位。
#   "fence"  = p1–p11 的 python 围栏 + [TOOL RESULT] 文本回填（逐位可复现，回退位）
#   "native" = Qwen 原生 <tool_call> 工具协议（方案 A 主线）
# 做成**运行时开关**而不是重写：协议本身才是可 A/B 的单变量（docs/09 §8）。
# 刻意**不另起 algo 名**——算法（loss/adv/打分口径）一字未改，变的是 rollout 协议；
# 新起名字会让 losses/ALGOS/eval/ref_server 五处并列分叉，而收益只是命名好听。
# 隔离由两处承担：①签名里加 -tp<native>；②新 run 必须换 out_dir（docs/09 §8）。
TOOL_PROTOCOLS = ("fence", "native")

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
                    loss_norm="sample_mean",
                    # stop 机制与 retool_math 同家族（围栏协议一致；GSM8K 历史协议
                    # 的 code_rate≈0 下 stop 无受力面，开启无行为差异风险）
                    retool_stop=True),
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
    # 【2026-09-18 方案B·轮次预算重定，为何从 2×6144 改 4×2048】三次 run（run2 /
    # p5 两版）代码路径都被 RL 压灭（code% 45→3、code_ok 56%→10%），而分层迁移
    # 分析实锤代码有真实价值（BASE 用码题 74.1% vs 纯推理 47.4% = +26.7pp）。
    # 根因不是参数（lr 1e-6 / reward ±1 / clip 0.2-0.28 已对齐官方与
    # agentic-rl-lab/05-retool），是协议结构：max_rounds=2 只有 1 次代码机会，
    # 代码写错一次整条轨迹就废、无补救空间 → RL 算出"写代码期望净收益为负"→
    # 理性压灭。参考项目（同 Qwen3.5-4B）用 max_code_calls=4/turns=6 让代码
    # "可分步改进"：turns 2.0→2.8 在学"多试几次"、code_calls 1.24→1.98 单调升、
    # Average@12 23.6→47.5、sandbox success 0.68→0.80（代码越练越能跑）。
    # 方案 B = max_rounds 4（3 次代码机会）+ round_gen_tokens 2048（不赌 4B 在
    # 1024 下的截断风险，docs/03 探针 1024 截断 85%）+ max_context_tokens 14336
    # （p5 实测档，need=4×2048+1024+798=10014 ≤ 14336 ✅，余量充足）。
    # 仍保留的差异（报告须注明）：loss_norm sample_mean、KL β=0.04、
    # Q_batch_size=1、trunc_shaping 0.0（overlong_filter 接管，硬罚归零）。
    "retool_math": dict(beta=0.04, clip_low=0.2, clip_high=0.28, adv_mode="group_mean",
                        loss_norm="sample_mean", data_task="dapo_math",
                        num_pre_Q=8, train_micro_batch_size_per_gpu=8,
                        temperature=1.0, top_k=-1,
                        gen_questions_per_attempt=4, sandbox_workers=8,
                        max_context_tokens=14336, round_gen_tokens=2048,
                        max_rounds=4,
                        max_gen_tokens=8192, max_prompt_length=1024,
                        # 【2026-09-18 stop 机制】写到代码块闭合围栏立即停（protocol.
                        # RETOOL_STOP_KWARGS）——工具结果紧跟代码回填，修复"代码→瞎猜
                        # →结果"错位与 42~55% 高截断（三轮 run 代码压灭的根因）。
                        # 关闭 = 回到 p6 旧协议（A/B 对照位）。
                        retool_stop=True,
                        # 长度控制：overlong 现在可达（预算自洽）→ 打开。
                        # 【2026-09-21 trunc_shaping 归零】overlong_filter=True 时
                        # 截断样本 adv=0（从 advantage/组统计中移除），trunc_shaping
                        # 只改截断样本的 reward 值但该值被 mask 跳过 → 零影响。
                        # DAPO 消融：硬截断罚（无论 0 还是 0.5）都是 reward noise。
                        overlong_shaping=True, overlong_buffer=256,
                        trunc_shaping=0.0,
                        # 【2026-09-21 DAPO overlong filtering】截断样本从 advantage
                        # 和组统计中移除（消融 +6 分，最稳定的长度控制组件）。
                        # 杀 NeMo-RL bug：trunc_shaping>0 时全错组+混合截断不再
                        # 产生假方差通过 group_ok（p9 step100 -8.5pp 的根因）。
                        overlong_filter=True,
                        # 【2026-09-21 尝试级 shaping（#2 风险不对称修正）】
                        # code_attempt_w>0：写出可执行代码块就给小分（不依赖
                        # code_ok）——对冲"代码路径风险不对称"的理性压灭
                        # （p6 实测 code% 50→3，outcome-only 下 RL 放弃代码）。
                        # 0 = 关闭（旧行为逐位相同，单变量 A/B 对照位）。
                        # 剂量参考：轨迹 ±1 域内 0.05×3次=+0.15，足以翻转
                        # "写代码期望净收益为负"的算术但不淹没 outcome 主信号。
                        code_attempt_w=0.0,
                        # 【2026-09-30 工具信用分配】标准成本敏感 agent RL：任务结果
                        # 仍是主奖励；每个实际执行的工具调用轮扣小成本，预算不足却
                        # 继续调用的轮只对该轮施加局部负优势。0.02/0.10 均远小于
                        # outcome ±1，先保正确性，再在同等解法中压缩调用/token。
                        tool_call_cost=0.02, tool_waste_penalty=0.10,
                        # 【2026-10-01 截断末段信用】实测零梯度 10%→12.6%、
                        # ``B_cut_mid_prose`` 占无boxed 43%：废调用已有局部负信用，
                        # 而"写散文到预算耗尽"仍整行 sw=0 → 唯一没有反向信号的长度
                        # 失控形态。与废调用惩罚同值起步（0.10，远小于 outcome ±1）。
                        trunc_tail_penalty=0.10,
                        # 训练内嵌评测：每个 checkpoint 自动跑 test+train
                        eval_during_training=True,
                        # 【2026-09-24 p10 盲窗事故】retool 多轮采样档 n=500×4 轮
                        # 15 分钟跑不完 → 旧 900s 全部 TIMEOUT → 内嵌评测形同虚设。
                        # 抬到 3600（与 eval_vllm_one.py 手动跑同预算）。
                        eval_timeout_s=3600,
                        code_w=0.0, reward_switch_step=1000000000,
                        # 【2026-09-11 eval 全灭事故】思考开关收进 preset 单点同源：
                        # Qwen3.5 默认 enable_thinking=True，eval 端 prompt 构造从
                        # config 取此值——此前只在训练 CLI 传，eval 拿到 None，
                        # 生成以 <think> 开头烧穿预算，fmt/acc 双灭（base 同灭，
                        # 证明是协议问题非权重）。训练 CLI --chat_template_kwargs
                        # 仍可覆盖；GSM8K 家族（Qwen2.5）模板忽略该上下文键。
                        chat_template_kwargs={"enable_thinking": False}),
    # =====================================================================
    # 【docs/16 Search-GRPO】多轮检索推理（agentic RL）
    # =====================================================================
    # 与 retool_math 同族（loss/adv/rollout 机制全同，见 losses.compute_loss 的
    # search_qa 分支与 ALGO_DEFAULTS 注释）：差异全在 **工具** 与 **任务/打分**：
    #   · 工具 = search（原生声明，docs/16 §3），执行器是独立 HTTP 检索服务；
    #   · 任务 = NQ/HotpotQA 式事实问答，奖励 = EM（±1），outcome-only。
    # 预算档（docs/16 §3）：检索回包比沙箱 stdout 长得多，tool_result_max_chars
    # 抬到 1500（3 段 × ~500 字符）；轮数给到 8（多跳题需要多次改写查询），单轮
    # 512（查询与短答都很短，长预算只会被"写散文"烧掉）。
    # 预算自洽（validate_retool_budget 强制）：
    #     max_rounds(8) × round_gen_tokens(512) + max_prompt_length(1024) + 预留
    #     = 4096 + 1024 + 7×766 = 10482 ≤ max_context_tokens(14336) ✅
    "search_qa": dict(beta=0.04, clip_low=0.2, clip_high=0.28, adv_mode="group_mean",
                      loss_norm="sample_mean", data_task="search_qa",
                      num_pre_Q=8, train_micro_batch_size_per_gpu=8,
                      temperature=1.0, top_k=-1,
                      gen_questions_per_attempt=4, sandbox_workers=8,
                      # 原生工具协议 + search 工具（名字驱动声明/解析/执行器三处）
                      tool_protocol="native",
                      native_tool_name="search", native_tool_param="query",
                      # 预算四件套
                      max_context_tokens=14336, round_gen_tokens=512,
                      max_rounds=8, max_prompt_length=1024, max_gen_tokens=8192,
                      # 检索回包预算：3 段 × ~500 字符（估算见 validate_retool_budget
                      # 的 //2+16 折算），抬到 1500 让整段摘要进上下文
                      tool_result_max_chars=1500,
                      retool_stop=False,      # 原生档自然停在 im_end
                      # 长度控制与 retool_math 同档（outcome-only + overlong filter）
                      overlong_shaping=True, overlong_buffer=256,
                      trunc_shaping=0.0, overlong_filter=True,
                      # outcome-only：不给检索次数/成功率的 shaping（先看基线涌现，
                      # docs/16 §3 R0 档；R1~R3 是后续单变量实验，不预置）
                      code_w=0.0, code_attempt_w=0.0, len_penalty_w=0.0,
                      tool_call_cost=0.0, tool_waste_penalty=0.0,
                      trunc_tail_penalty=0.0,
                      reward_switch_step=1000000000,
                      # eval：held-out dev（NQ dev / HotpotQA dev）由
                      # prepare_search_qa 落盘，eval_vllm_one 按 search_qa 打分
                      eval_during_training=False,
                      chat_template_kwargs={"enable_thinking": False}),
    # =====================================================================
    # 【docs/17 SWE-RL】L0：合成 bug 的仓库级修复（patch-only 单轮）
    # =====================================================================
    # L0 = HumanEvalFix 式合成 bug：单函数 + 单测现成 + **定位已给出**
    # （docs/17 §2 的难度分层第一档）。动作空间是**单轮 patch-only**：模型读
    # issue + 已定位的局部代码 + 测试，直接输出一个 diff，没有工具调用
    # （docs/17 §2.5 定版："L0 不走多轮框架，工程量比 Search-GRPO 小得多"）。
    #
    # 因此本 preset 与 search_qa/retool_math **结构上不同**：
    #   · 走**单轮**路径（不设 tool_protocol=native、不进多轮 rollout），
    #     故不需要 max_rounds/round_gen_tokens 那一套，预算就是一次生成长度；
    #   · 奖励来自**真实测试执行**（git apply + pytest），不是文本正则；
    #   · 生成长度要装得下整个 diff：L0 是单函数级改动，但模型会先"想一想"，
    #     1536 是一条 diff（通常 20~60 行）+ 简短推理的合理上限；
    #   · max_prompt_length 要大：提示里含**完整的目标文件 + 完整测试文件**
    #     （L0 的定位是给定的，这是它的定义，不是泄漏）——8 条种子实测
    #     提示约 900~2500 字符，折 token 后给 2048 有余量。
    # 预算自洽：单轮路径无轮数乘积，max_context_tokens 只需 ≥
    #     max_prompt_length + max_gen_tokens = 2048 + 1536 = 3584；给 4096。
    #
    # 【为什么 verdict 里没有 code_w/len_* 任何 shaping】L0 的 reward 是
    # F2P 硬判据（docs/17 §3）：patch 长度与"修对没修对"无关，给长度项只会
    # 引入"短 patch 骗分"的捷径；工具信用项更是完全没有受力面（单轮无工具）。
    "swe_l0": dict(beta=0.04, clip_low=0.2, clip_high=0.28, adv_mode="group_mean",
                   loss_norm="sample_mean", data_task="swe_l0",
                   num_pre_Q=8, train_micro_batch_size_per_gpu=8,
                   temperature=1.0, top_k=-1,
                   # 【刻意不设 gen_questions_per_attempt】它是**多轮**路径的吞吐杠杆
                   # （rollout 的 use_qqueue = is_retool and multi_q>1），而 L0 走单轮、
                   # 从不读它 —— 在 preset 里写一个好看的 4 只会让"调了没反应"变成
                   # 又一个静默空转（本项目最贵的一类 bug）。L0 每次 attempt 采
                   # Q_batch_size×num_pre_Q = 1×8 条轨迹；要提吞吐就抬 num_pre_Q
                   # （它同时扩组内 patch 多样性，对本任务是净收益）。
                   # 单轮 patch 生成：不声明工具、不走多轮
                   tool_protocol="fence",
                   # 【单轮的显式标注】它同时被 rollout 的 is_retool 判据与
                   # rlab.swe_l0 的判分路径读；写死 1 让"这是单轮"在配置层就成立，
                   # 而不是靠"没人传 --max_rounds"。
                   max_rounds=1,
                   max_context_tokens=4096, max_prompt_length=2048,
                   max_gen_tokens=1536,
                   # 长度控制项全部关闭（见上方"为什么没有 shaping"）
                   overlong_shaping=False, overlong_filter=False,
                   trunc_shaping=0.0, code_w=0.0, code_attempt_w=0.0,
                   len_penalty_w=0.0, tool_call_cost=0.0,
                   tool_waste_penalty=0.0, trunc_tail_penalty=0.0,
                   reward_switch_step=1000000000,
                   # 测试执行是本项目**最贵的一步**（docs/17 §1：
                   # reward 成本比 math 高 2~3 个数量级）：
                   #   · 同组相同 patch 只测一次（dedup_group_indices）
                   #   · 硬超时兜底，单条实例不许拖死 rollout
                   #   · 并发数 = 组内不同 patch 数上限（num_pre_Q），
                   #     再大没有受力面（一个组最多 8 条）
                   swe_test_timeout=60.0,
                   swe_test_mem_mb=2048,
                   swe_test_workers=8,
                   swe_work_root=None,   # 判分用的临时仓库根（None=系统临时目录）
                   # eval：held-out dev（prepare_swe_l0 按源种子分组切）
                   eval_during_training=False,
                   chat_template_kwargs={"enable_thinking": False}),
}

BASE = dict(
    # ---- 模型与路径 ----
    model_path="/root/Qwen2.5-3B",          # base 版，与历史实验可比
    data_task="gsm8k",                       # gsm8k（阶段0/1）；阶段2/3 扩展
    out_dir="./rlab_out",
    record_path="./rlab_out/record.jsonl",   # 生成数据得分记录（analysis.py 消费）
    # 【2026-10-02 轨迹全量落盘】record.jsonl 只有**统计**（acc/clen/段长/桶标志），
    # 看不到"模型到底写了什么"——B/C 桶判读的最后一步（末段是答题被掐还是写散文
    # 跑飞）必须看文本。开启后生成端把**每条轨迹**写成 `<out_dir>/traj.jsonl`：
    #   · 段级落盘（assistant/工具分段 + 每段 token 数），不是拼接文本——拼接后
    #     段边界丢失，"末段多长/写了什么"就答不了；
    #   · **含被丢弃的 attempt**（uniform 零方差 / overlong 超长）：这两类整组
    #     不落 record，正是记录口径的幸存者偏差来源，只 dump 上传组等于没治；
    #   · 工具段用**消毒后**的回包（沙箱 stdout 是注入面，落盘版不再引入原始输出）。
    # 默认关：全量文本约 50MB/16h（含丢弃 attempt 更高），pod 盘要留空间。
    # 不进 run_signature——它不改采样/奖励/loss 任何一个字节，只是观测面；
    # 进签名会让"打开诊断"意外撞 guard_ckpt_collision 或让旧 ckpt 变外来签名。
    traj_dump=False,
    # 【2026-09-10 4B 探针实锤】apply_chat_template 附加 kwargs（None=不传，Qwen2.5 行为不变）。
    # Qwen3.5 系必须 {"enable_thinking": false}：thinking 模式把单轮 1024 token 预算
    # 烧在 <think> 长链上（4B 探针实测：末段截断 98.9% / 无 boxed 99.2% / 全错 484/490，
    # 换算可学带仅 1.2%——全是协议失败不是能力失败）。官方实测思考模式下"很少写代码"。
    # Qwen2.5 模板不引用该 jinja 变量，传入无副作用。
    chat_template_kwargs=None,
    # 【2026-09-15 口径更正：不再需要分裂加载】Qwen3.5-4B 官方权重是多模态复合体，
    # 但 c991f84 之后 torch 侧可**直连**该目录（load_causal_lm 显式喂 text_config，
    # 见 model_loading），vLLM 侧本来就读它 → **统一用一份 /root/Qwen3.5-4B 即可**，
    # 本键保持 None。注意两件事互不相同：
    #   · 加载：一份目录两边都读得动（已解决）；
    #   · 权重同步：torch 内存里的参数名是纯文本布局（model.X / lm_head.*），vLLM
    #     多模态实现的参数名是 model.language_model.X → **键名映射仍然必须做**，
    #     判据由 sync.need_text_to_mm_remap 按"键名形态"给（与目录是否相同无关），
    #     生成端启动行会打印"权重同步键名映射: 开/关"自证。
    # 本键只在需要**回退**分裂加载时使用：老 transformers 缺 qwen3_5_text 前缀转换
    # 映射（A1：复合 config 喂文本类崩）时，用 extract_text_model.py 抽一份纯文本
    # ckpt 当 model_path，并把本键指向原多模态目录（映射恒开）。
    vllm_model_path=None,
    # 【2026-09-16 真统一：存盘即多模态壳】复合（多模态）模型训练时，step_N 直接按
    # vLLM 可直读的多模态格式落盘（键名 model.language_model.* + 骨架的复合 config/
    # processor），不再写"实例化那个类"的文本格式。动机：c991f84 只统一了 torch 侧
    # **加载源**，存盘产物仍是 Qwen3_5TextConfig → eval 独立进程起 vLLM 直接 TypeError
    # （A2），只能靠 materialize_mm_ckpt 补格式 = "假统一"。
    # 关掉（False）= 回到旧文本格式（Qwen2.5 口径）；此时 eval 端会自动物化兜底
    # （eval_vllm_one.py），或手工 python -m rlab.materialize_mm_ckpt。
    # 非复合模型（Qwen2.5-3B）恒走文本格式，本键不参与。
    save_mm_checkpoint=True,
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
    # 【2026-09-23 黑名单 TTL】拉黑后再采 ttl 轮题自动释放（修"训练分布单调
    # 变易"：难题随模型变强重新可学）。0 = 关闭（旧行为：streak 永久累计直到
    # floor 全量重置）。合理值 = 拉黑题预期"服刑"的 attempt 轮数（如 64）。
    q_blacklist_ttl=0,
    # 【2026-09-23 组相对长度惩罚】默认 0 = 关闭（旧行为逐位相同，单变量 A/B
    # 对照位）。MiMo Eq.4 形态：通过轨迹分位数起坡 + 通过率门槛，只罚未通过轨迹。
    # 分档奖励 code_w 不在此设：BASE 的 code_w=0.1 是 retool 家族阶段2 路径语义，
    # retool_math preset 自带 code_w=0.0（math 路径此前是死代码，2026-09-23 打通）。
    len_penalty_w=0.0,
    len_penalty_quantile=50,
    len_penalty_gate=0.25,
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
    # 【2026-09-18 max staleness】训练端丢弃"吃太旧策略"的批（off-policy 兜底）。
    # 0 = 不启用（历史行为）。>0 时：batch 的 gen_version 与当前 opt-step 差 >
    # 该值 → 丢弃该批（continue 等下一批）。合理下限 = gen_update_steps（推送周期
    # 内的批都"够新"）；设更大是容忍双队列堆积造成的额外陈旧。见 docs/05 §P3 关联。
    max_stale_opt_steps=0,
    train_micro_batch_size_per_gpu=4,   # = Q_batch_size*num_pre_Q
    gradient_accumulation_steps=4,
    lr=1e-6,
    warmup_steps=0,

    # ---- 基础设施 ----
    gen_device=0,            # vLLM 生成 + torch gen_logps 副本所在物理卡
    gen_gpu_mem=0.45,        # 生成端 vLLM 显存占比（GPU0 = ref~7G + vLLM + 副本~7G
                             # + logits 瞬时峰~12G，0.45×96 总计 ~70G < 96G；2026-09-09
                             # 提速：旧 0.35 的 KV 池对 3B+GQA 大量闲置）
    # 【2026-09-14 起跑期 OOM-kill；2026-09-16 起默认改档】vLLM 引擎参数透传，直接进
    # gen_worker 的 LLM(**kwargs)，键名用 vLLM 的**下划线**形态：默认
    # {"gdn_prefill_backend": "triton"}。
    # 【为什么默认 triton 而不是 vLLM 的 FlashInfer】Qwen3.5 的 GDN 线性注意力层在
    # vLLM 里默认走 FlashInfer GDN prefill，而它是 **JIT 现场编译**（不是 AOT 预编译件，
    # 缓存不跨 pod）。编译窗口恰好落在 ref/gen/train 三方把权重搬进主机内存的启动期：
    # nvcc/ninja 的宿主 RAM 峰值把 60G 容器顶爆，ninja 被 OOM-killer 以 SIGKILL 带走
    # （日志只有 "ninja ... died with <Signals.SIGKILL>"、零编译输出），紧接着生成端
    # 进程也被杀——**没有任何 Python traceback**，训练端只看到"生成端进程已退出"
    # （2026-09-14 事故，见 docs/05）。triton 后端不需要 nvcc，从根上消掉这段起跑期
    # 编译；代价只在 prefill，而负载是 3072 token 的 decode 为主，占比小。
    # 【训练与评测必须同档】评测端（eval_vllm_one.py）读同一份配置并把本 dict 展开进
    # LLM(**kwargs)——它构造引擎时同样会付这笔 JIT，两边不能静默分叉（否则 Δacc 里会
    # 混进 kernel 变量）。
    # 【覆盖方式】CLI --vllm_gen_kwargs 整体替换本 dict（传 '{}' = 一个引擎参数都不传，
    # 回到 vLLM 默认 FlashInfer；传 {"gdn_prefill_backend": "flashinfer"} = 显式换档）。
    # 结果进 run_signature（-vkgdn_prefill_backend=triton）：旧同配方重跑对不上，这正是
    # 要的——kernel 档确实变了。
    # 【非 GDN 模型（Qwen2.5-3B）】vLLM 里它只是不被使用的引擎参数，传了是惰性的；但它
    # 仍进签名——3B 新 run 与旧 3B checkpoint 不再同签名（刻意的：引擎参数档变了）。
    vllm_gen_kwargs={"gdn_prefill_backend": "triton"},
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
    # 【2026-09-29 长度控制修复·死开关】overlong 参考系的**显式覆盖**
    # （0 = 用 overlong_ref_tokens() 的自动值 = max_rounds×round_gen_tokens）。
    #
    # 【为什么必须能覆盖】自动值在"预算给满"的档下**必然够不着**：native_p3 实测
    # overlong_ref=10240（触发线 9984）而 avg_clen≈2811 —— 差 7000 token，长度压力
    # 实际为零（整个 reward 里没有任何一项作用于"啰嗦但不截断"的轨迹）。
    # 修法不是改公式（公式在预算自洽时是对的上限），而是把参考系从**物理上限**
    # 改成**目标长度**：>0 时直接作为起坡点，让 shaping 对啰嗦真正可达。
    # 只在 overlong_shaping=True 时有行为意义；进签名（改 reward 值域）。
    overlong_ref=0,
    # 【2026-10-19 overlong 幅度护栏】overlong_penalty 的封顶值（旧行为=1.0）。
    #
    # 【为什么必须能压】1.0 的满幅 == retool_math 正确性奖励的**全部**幅度（±1），
    # 于是这个"软护栏"在 group_mean 下有力量把「答对但写长」的轨迹压成负优势。
    # p11 实测（536 条轨迹，overlong_ref=8192/buffer=2048 → trigger 6144）：
    #   吃罚 32.6%、顶满 24.6%；被压的 correct 样本 task=+0.5~+1.14 全部翻成非正
    #   （21/321 非零 task 样本，doc15 §7 明令停线的"辅助项翻转正确性符号"）。
    # 课程把题目推进难题带后 avg_clen 3256→4683，这条从未被注意的护栏突然成为
    # 组内排序的主导项 —— p10 没暴露不是因为它安全，是因为样本够不到触发线。
    #
    # 0.5 = 「答对 + 顶满罚」仍留半个正确性幅度的正优势；答错侧方向不变（只更负）。
    # 1.0 = 旧行为逐位相同（历史 run/ckpt 口径不变）。进签名（改 reward 值域）。
    overlong_max_penalty=1.0,
    # 【2026-10-19 overlong 作用域·结构性护栏】True 时长度罚**只作用于未通过轨迹**
    # （False=旧行为，全部轨迹都罚）。
    #
    # 【为什么 cap 调小不够】p11b（cap 0.5）实测仍有 1.9% 翻转：6/8 组正确样本的
    # task 优势只有 +0.50，而 overlong(0.5)+len_eff_pen(0.1)=0.55 > 0.50。组越
    # 均衡（课程越集中在近界题）task 优势越小、越容易被长度项压翻 —— 这是
    # **结构性矛盾**，调 cap 只能逼近不能根除（再降则长度约束失效）。
    # 正解是改作用域：overlong 是护栏不是目标，职责是"答不出来时别烧预算写散文"
    # ——与 MiMo group_length_penalty 同一条原则（只罚未通过，见 reward.py:262）。
    # 开启后「答对」在任何长度下都不会被长度项翻负（结构性保证）。
    overlong_fail_only=False,
    # 【2026-10-19 正确性优先档·三件套】task_adv_guard 是这一组的总开关，开启后：
    #   ①aux_clip_alpha>0：辅助项去均值后裁剪到 ±alpha·task_adv_margin（c=1/7 组
    #     余量 0.25 → aux 限 ±0.125；c=4/4 → ±0.5）。「正确>0>错误」成为结构恒等。
    #   ②trunc_counts_as_wrong：截断/废码不再从组统计移除（写爆预算=没答完=acc−1，
    #     吃真实负梯度）。旧口径下它 sw=0、adv=0，长度约束在梯度上是免费的。
    #   ③uniform 判据改用任务维度方差：acc 全同的组直接丢（对准确率只贡献噪声）；
    #     旧判据"最终 advantage 有方差"让 aux 驱动它们上传（p10: 1000 组 uniform=0，
    #     其中 59.5% 零正确性信号）。
    # 只在 adv_mode=group_mean 下裁剪（group_std 的间距被 std 归一化，无解析余量）。
    task_adv_guard=False,
    aux_clip_alpha=0.5,
    trunc_counts_as_wrong=True,
    # 【2026-09-29 效率激励·补正向】通过轨迹内部的**组内相对**长度奖励权重
    # （0 = 关闭 = 旧行为逐位相同）。
    #
    # 【为什么需要】旧 reward 对"最短答对"零激励、且方向相反：len_penalty 只罚
    # **未通过**轨迹（reward.py:265-269），overlong 在预算档够不着，而
    # code_attempt_w/code_w 随调用次数**单调递增** → 实测答对时 0 次调用 +1.00
    # < 4 次调用 +1.40 —— 梯度指向"多烧 token"。本项补齐正向：组内**通过**轨迹
    # 按长度中位数起坡，短于中位数的拿正分。
    # **必须组内相对**：绝对长度惩罚是 run2"表面收尾"事故的根因
    # （模型学会草草收尾骗过惩罚），见 group_length_penalty 的 docstring。
    len_eff_w=0.0,
    # 【2026-10-09 正确轨迹反向效率信用】只罚**通过且长于组内通过分位数**的轨迹。
    # 与 len_eff_w 共用组内参考系：短于 ref 拿正分，长于 ref 最多扣本权重；失败
    # 轨迹仍只由 MiMo len_penalty 管，职责不重叠。0=关闭，历史行为逐位不变；
    # 推荐 0.10，reward.py 内 ratio cap=1 且保留最小正值，正确结果不翻成非正。
    len_eff_penalty_w=0.0,
    # 【2026-09-29 反激励修正】code_attempt_w/code_w 由"每次调用累加"改"一次性"
    # （用过就给固定小分，不随调用次数增长）。保留对冲"代码压灭"的作用
    # （p6 实测 code% 50→3），去掉"多调用多拿分"的方向性错误。
    # False = 旧行为逐位相同（A/B 对照位）。
    code_shaping_once=False,
    # 工具动作的 turn-level process reward（仅 retool 家族有受力面）：
    # 实际执行调用轮：A_group - tool_call_cost；预算不足的废调用轮：
    # -tool_waste_penalty。0=关闭，其他算法与历史配置逐位不变。
    tool_call_cost=0.0,
    tool_waste_penalty=0.0,
    # 【2026-10-01 截断末段信用】trunc_final 轨迹此前整行 sw=0（零梯度）——模型
    # 学不到"把预算烧光"这件事本身是坏的。本项与 tool_waste_penalty 同构：给
    # **最后一个 assistant 段**注入固定局部负优势。任务结果仍不可信（截断前可能
    # 还没作答）故继续用 sample_mask 排除出组均值，但动作本身拿到直接信用，
    # 此前有效推理/工具轮不连坐。0=关闭=历史口径（trunc 整行过滤）。
    trunc_tail_penalty=0.0,
    # 【2026-09-21 DAPO overlong filtering】截断样本（末段被轮长上限切断）从
    # advantage 和组统计中移除：组均值只算非截断、截断样本 adv=0。
    # DAPO 消融：overlong filtering +6 分（最稳定的长度控制组件）。
    # 同时杀 NeMo-RL bug：trunc_shaping>0 时全错组+混合截断不再产生假方差。
    overlong_filter=False,

    # ---- 训练内嵌评测（2026-09-21）----
    # 每个 checkpoint 保存后自动跑 test+train 评测，结果落进 step_N/eval_*.json。
    # 训练完不用再手动评测；训练中就能看到 step100 深坑或 train/test gap。
    # eval_gpu_mem=0.20 与训练 vLLM（gen_gpu_mem=0.6）共卡 GPU0，checkpoint 保存时
    # 训练暂停无竞争。retool_math preset 默认开；BASE 默认 False（GSM8K 家族不需要）。
    eval_during_training=False,
    eval_n=500,
    eval_gpu="0",
    eval_gpu_mem=0.20,
    # 【2026-09-24 内嵌评测超时（p10 盲窗事故）】每路（test/train）评测子进程的超时上限
    # 秒。retool 多轮采样档（n=500×4轮×6144 token，temp1.0）实测 15 分钟物理上跑不完，
    # 旧硬编码 900s 让 step100/200/300/400 共 8 路全部 TIMEOUT、eval_*.json 缺失——训练
    # 期间全程盲跑（2026-09-21 加内嵌评测就是为了治 p9"训完 11h 才发现深坑"的教训，被
    # 这个超时反手做成同类盲窗）。BASE 900 保持旧行为；retool_math preset 抬到 3600
    # （与 eval_vllm_one.py 的手动跑一致），并落一个"超时即写空结果 json"的哨兵——
    # 宁可留 null 让后续工具显式报"盲"，也不再静默吞掉（analysis 读到空 n 会拒绝下结论）。
    eval_timeout_s=900,
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
    #   True  = 逐轮 vLLM 采样 logprobs（SamplingParams(logprobs=N) 返回被采样 token 的
    #           logprob）。多轮下每轮请求的上下文已含之前所有工具结果，故逐轮收集拼接
    #           == 在拼接序列上重算，数学严格同义，且完全绕开 prompt_logprobs（那条
    #           路径在本环境会 hang，正是 torch 副本存在的起因）。
    # 口径变化：logps 分母由 torch kernel 换 vLLM kernel（同为 bf16）——首次启用必须
    # 用 verify_gen_logps 对拍并把最大差写进报告。
    # 【2026-09-15 真机实锤】Qwen3.5-4B + vLLM v0.19.1 + triton GDN prefill 下，
    # vLLM 采样 logprobs 不可复现：同一命令跑两次，token 逐位置一致率仅 2.91%，
    # |Δlogp| p99=5.79、max=14.5。因此该档位不能作为 gen_logps 来源，见
    # docs/07-vllm-logprobs-non-determinism-4b.md。默认值 False 保持 torch 副本口径。
    vllm_gen_logps=False,
    # 【2026-09-15 真机实锤】N 的取值决定"报哪条路"：N=0（只报被采样 token）在 vLLM
    # v0.19.1 + Qwen3.5 GDN 上**报错数**——同一 prompt/位置/token，N=0 报 -0.602，
    # N=20（raw）报 -7.40、torch 独立重算 -7.400（prefill 位单点差 6.8 nat，训练对拍
    # 的 max 12.8 同源）；N≥1（top-K 里挑被采样 token）与 torch 的差回落到 mean 0.02/
    # max 0.23。故 vllm_gen_logps 档位建议 N≥1；默认 0 只为保留 A/B 与旧 run 可复现。
    vllm_logprobs_n=0,
    # 【2026-09-15 真机】vLLM 采样确定性档：不开时同一请求**同进程背靠背**三次的 top-K
    # 字典 3/3 不同、top-1 logp 抖动 0.19 nat（det 探针）；开 VLLM_BATCH_INVARIANT=1 +
    # 显式 attention backend 后 3/3 全同（spread=0）。代价：关 custom all-reduce、
    # 用确定性 kernel，吞吐下降——是"用 vLLM logps 省 8G"能否成立的前提。见 docs/07。
    vllm_batch_invariant=False,
    vllm_attention_backend=None,   # 如 "FLASH_ATTN"；batch-invariant 要求，缺了会启动即失败
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
    # ---- 【2026-09-29 token 预算模式】轮数上限 → token 预算 + 模型自主终止 ----
    # 0 = 关闭（缺省，历史档逐位不变）。>0 时三条语义同时切换（原生协议）：
    #   ① 能否执行一次调用由**剩余 token 预算**判定，不再由"是不是末轮"判定
    #      → code_wasted 不再由"轮数用完"制造（仍会在**预算真耗尽**时发生，
    #      但那已是"确实没空间了"的真实信号，不是结构性的人为截止）；
    #   ② 被单轮上限切断（finish_reason=length）**不再是终局**，续写同一轮
    #      → A 桶（调用写到一半被切）与 B 桶（散文写到一半被切）有机会活到收尾；
    #   ③ 每次工具回包追加一行剩余额度（budget_hint）→ 模型能"看见"预算，
    #      C 桶（额度耗尽还在调用）从不可学的悬崖变成可学的判断。
    # 【为什么轮数不是预算】实测 5 轮→6 轮在固定单轮额度下几乎无改变（丢弃率
    # 45%→50%、末轮废码率 ~31%→~30% 持平），因为 answer/invalid 是**终局**，
    # 剩余轮数被整段作废——轮数从来不是"可用预算"，只是循环次数。
    # 预算不变量随之改为 max_traj_tokens + max_prompt_length ≤ max_context_tokens。
    max_traj_tokens=0,
    # 必须留在预算内、留给"最终作答"的 token 数：判定"这次调用还执行得起吗"。
    # 执行一次调用要花掉 工具回包 + 之后一轮答案 两笔，两者都装得下才执行。
    answer_reserve=0,
    # True = 工具回包追加一行剩余 token 额度（只在 max_traj_tokens>0 时有意义）
    budget_hint=False,
    # 【2026-10-20 计划预算 vs 真实上限】通告给模型的**轨迹预算数字**（0 = 用
    # max_traj_tokens 的旧行为，逐位不变）。
    #
    # 【为什么需要它】p9 §7.6 的锚机制实测：提示里写 8192、真实上限 12288
    # （"承诺紧、余量松"）比两者相等好 +5.2pp acc / +4.1pp fmt，且少花 token。
    # 但那条腿**只靠提示文本**，模型能看见预算的另外两个信道仍然漏出真实总量：
    #   ① 工具回包的 `[budget] N tokens left` = max_traj_tokens − used；
    #   ② 调用执行判据 `_can` = (max_traj_tokens − used − obs) ≥ answer_reserve。
    # p9 没暴露这个漏洞，是因为题简单（avg_clen 2602，模型在总量撞墙前就收完了）；
    # p11c 换进难度带后 avg_clen 涨到 6056~8056，B 桶（散文写到撞墙）272/272
    # 全部 clen≥0.9·cap —— 正是"模型第一次真正用上了那个 11600"。
    #
    # 【本键做什么】>0 时把**三处通告口径**统一到它：系统提示的预算数字、
    # `[budget]` 剩余额度、`_can` 执行阈值。`max_traj_tokens` 退化为纯保险丝
    # （模型看不见，只在最后兜底持有 +4096 的溢出余量）。于是"承诺紧、余量松"
    # 从"取决于题不够难"变成**结构性成立**。
    #
    # 【超出的部分会发生什么】used 越过 budget_target 后：提示行变成"0 left，
    # 别调用、立刻 boxed 收尾"，而调用一律不执行（记 code_wasted）；但**生成
    # 不终止**——保险丝给的那段余量正是留给它把 boxed 写出来。
    #
    # 约束（validate_budget_target fail-fast）：必须 ≤ max_traj_tokens（通告比
    # 保险丝还大＝纯说谎，只能换来截断）；且必须留得下一次调用
    # （budget_target − answer_reserve − obs_est > 0），否则从第 0 秒起所有调用
    # 都不执行，等于把协议废掉。进签名（改生成端行为与终止结构）。
    budget_target=0,
    # ---- 工具协议档位（docs/09-native-tool-protocol.md）----
    # "fence"（缺省）= 自造围栏 + [TOOL RESULT] 文本（p1–p11 逐位可复现）；
    # "native"       = Qwen 原生 <tool_call>（方案 A 主线，参考实现 +23.89pp 的形态）。
    # 缺省必须是 "fence"：任何未显式改档的 run（含所有历史命令）行为逐字节不变。
    tool_protocol="fence",
    # 原生调用的解析形态："auto"=逐形态试（先 <function=…> 后 JSON）。
    # 【为什么有开关】Qwen2.5 与 Qwen3.5 的 tools 渲染形态不同且不能互推
    # （本机实测：2.5 是 JSON 形态，参考实现的正则吃的是 3.5 的 <function=…> 形态）。
    # 可用 rlab/native_probe.py 在 pod 上探明后显式钉死，避免 auto 猜错时静默走错分支。
    native_tool_style="auto",
    # True = 用 "</tool_call>" 作 stop 串在调用边界硬停。默认 False（原生节奏：
    # 模型自己吐 im_end）。只在 base 冒烟发现"调用后还继续瞎写"时才打开——
    # 那种样本 parse 判 invalid → 无 boxed → reward -1，是原生协议唯一的结构性
    # 负奖励入口（docs/09 §0.1 症状③ 的对应形态）。
    native_stop_at_call=False,
    # 【docs/16 Search-GRPO】原生工具名/参数名：native 档渲染与解析都按它走。
    # 缺省 code_interpreter/code = 历史路径逐字不变；search_qa preset 改成
    # search/query（声明在 protocol.SEARCH_TOOL）。未知名字由 protocol.native_tool
    # fail-fast（静默回落 = 渲染代码工具却接检索执行器，最贵的一类空转）。
    native_tool_name="code_interpreter",
    native_tool_param="code",
    # ---- 检索服务（docs/16 §3：独立部署在训练机之外，训练侧只持 endpoint）----
    # HTTP URL（"http://host:port"）或 "local:<index_path>"（pod 本地档，省网络往返）。
    # 训练前必须探活：不通直接拒训，绝不让基础设施故障伪装成"模型全错"。
    search_endpoint=None,
    search_top_k=3,          # 每次检索返回的段落数
    search_timeout=10.0,     # 单次检索超时（秒）；检索延迟直接进 rollout 墙钟
    search_retries=1,        # 网络类失败的限次重试（不含"检索为空"——那是合法结果）
    # 可选 bearer 鉴权：**存环境变量名**而不是令牌本身（密钥绝不进配置/落盘/日志）。
    # None = 不鉴权（内网独立部署的常见形态）。
    search_auth_env=None,
    # ---- SWE-RL L0（docs/17）：patch 判分执行器 ----
    # 判分 = git apply + pytest 跑 F2P/P2P（rlab/swe_l0.py）。它是本项目**最贵**
    # 的一步（docs/17 §1：reward 成本比 math 高 2~3 个数量级），故三项都可调：
    #   · swe_test_timeout 硬上限兜底（单条实例不许挂死 rollout）；
    #   · swe_test_mem_mb 子进程内存上限（Linux RLIMIT_AS，best-effort）；
    #   · swe_test_workers 并发（= 同时执行的判分数；组内已按 patch 去重，
    #     实际并发 ≤ num_pre_Q）；
    #   · swe_work_root 判分用的临时仓库根（None=系统临时目录；pod 上若 /tmp
    #     是 tmpfs，要指到磁盘目录，否则 8 条并发各写一份仓库会吃掉内存）。
    # 数值不参与签名：它们是**资源参数**不是配方（与 gpu_mem 同类；改了不影响
    # "这份权重是哪套协议训的"）。但 swe_test_timeout 会改变"超时算不算失败"，
    # 边界上影响 reward——故它进签名（见 train.run_signature）。
    swe_test_timeout=60.0,
    swe_test_mem_mb=2048,
    swe_test_workers=8,
    swe_work_root=None,
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
    "between [TOOL RESULT] and [/TOOL RESULT]. After you close the code block "
    "with ```, stop immediately and wait for the execution result; read it, then "
    "continue reasoning in the thinking section, then finish with the final answer "
    "inside the required answer tags. Always finish your code block before continuing."
)
system_prompt_retool = BASE["system_prompt"] + _RETOOL_EXTRA

# 方案1：retool-math 系统提示（借鉴 agentic-rl-lab/05-retool）—— outcome-only \boxed{}
# 【2026-09-18 stop 配套】与 protocol.RETOOL_STOP_KWARGS 咬合：模型写完代码块
# 闭合围栏 ``` 立即被截停（stop=["```\n"]），故必须告诉它"写完代码就停、等执行
# 结果、读完再继续"——否则模型会把"被截停"理解成失败，反而抑制写代码。
_RETOOL_MATH_SYSTEM = (
    "You solve math problems step by step with help from a Python code interpreter.\n"
    "Use the code_interpreter tool when calculation, symbolic manipulation, or enumeration helps you solve the problem accurately and quickly.\n\n"
    "How to use the code_interpreter tool:\n"
    "- Call it with Python code inside a fenced block like: ```python\n<your code>\n```\n"
    "  The environment executes your code automatically and inserts the result between [TOOL RESULT] and [/TOOL RESULT].\n"
    "- After you close the code block with ```, stop immediately and wait for the execution result. It will be inserted between [TOOL RESULT] and [/TOOL RESULT]; read it, then continue reasoning from there.\n"
    "- Results are captured from what your code prints with print(). Always print the values you want to see.\n"
    "- Each execution is independent: no variables, files, or state carry over between calls. Redefine everything you need in each piece of code.\n"
    "- Code must finish within a few seconds and use little memory. Do not read or write files. If you enumerate or brute-force, keep the search space small.\n"
    "- If the execution returns an error, analyze it and retry with corrected code when useful.\n\n"
    "When you have the final answer, end with exactly one line in this format:\n"
    "\\boxed{<your final answer>}\n"
    "Do not put the final answer inside the code block."
)
system_prompt_retool_math = _RETOOL_MATH_SYSTEM

# 【2026-09-25 原生协议提示（docs/09 §2.3）】以参考实现的 SYSTEM_PROMPT 为蓝本。
# 与围栏版的差别只有一件事：**删掉所有围栏/`[TOOL RESULT]` 措辞**——原生协议下
# 工具声明由 chat template 的 tools 段提供，提示里再讲一遍围栏格式就是自相矛盾
# （docs/02 第三类灭绝：MUST 写围栏 ⊕ 格式正则锚定互相打架）。boxed 收尾保留
# （打分口径没变）。零标签字面量：本串不含任何尖括号标签字节。
_RETOOL_MATH_SYSTEM_NATIVE = (
    "You solve math problems step by step with help from a Python code interpreter.\n"
    "Use the code_interpreter tool when calculation, symbolic manipulation, or "
    "enumeration helps you solve the problem accurately and quickly.\n\n"
    "How to use the code_interpreter tool:\n"
    "- Call code_interpreter with a `code` string containing Python code. Call it at "
    "most once per assistant turn, then wait for the execution result.\n"
    "- Results are captured from what your code prints with print(). Always print the "
    "values you want to see.\n"
    "- Each execution is independent: no variables, files, or state carry over between "
    "calls. Redefine everything you need in each piece of code.\n"
    "- Code must finish within a few seconds and use little memory. Do not read or "
    "write files. If you enumerate or brute-force, keep the search space small.\n"
    "- If the execution returns an error, analyze it and retry with corrected code "
    "when useful.\n\n"
    "When you have the final answer, end with exactly one line in this format:\n"
    "\\boxed{<your final answer>}\n"
    "Do not call the tool and give the final answer in the same turn."
)
system_prompt_retool_math_native = _RETOOL_MATH_SYSTEM_NATIVE

# 【docs/16 Search-GRPO】检索问答的原生提示。与 math 版同一套原则：**不讲协议格式**
# （工具声明由 chat template 的 tools 段提供），只讲"什么时候用、怎么用、怎么收尾"。
# 零标签字面量：本串不含任何尖括号标签字节。
_SEARCH_QA_SYSTEM_NATIVE = (
    "You answer fact-seeking questions with help from a Wikipedia search tool.\n"
    "Use the search tool whenever the answer depends on a fact you are not certain "
    "about (people, dates, places, numbers, titles). Write a short keyword query — "
    "entity names work better than full sentences.\n\n"
    "How to use the search tool:\n"
    "- Call it with a `query` string. Call it at most once per assistant turn, then "
    "wait for the passages.\n"
    "- Each call returns a few passages formatted as `[n] title: text`. They may be "
    "irrelevant or incomplete: check whether they actually support an answer, and "
    "rephrase the query if not.\n"
    "- Calls are independent: nothing is remembered between calls, so repeat any "
    "context you still need.\n"
    "- If two passages conflict, prefer the more specific one and say so in your "
    "reasoning.\n\n"
    "Answer with the shortest exact answer you can support (a name, a date, a "
    "number — not a sentence). When you have it, end with exactly one line in this "
    "format:\n"
    "\\boxed{<your final answer>}\n"
    "Do not call the tool and give the final answer in the same turn."
)
system_prompt_search_qa_native = _SEARCH_QA_SYSTEM_NATIVE

# 【docs/17 SWE-RL L0】补丁修复的原生提示。与 math/search 版同一套原则：只讲
# **任务与输出形态**（issue+代码+测试 → 一个 diff），不讲协议标签（L0 无工具，
# 无模板 tools 段）。
# 【为什么在这里再强调一遍"只输出 diff"】L0 的动作空间就是一段 patch
# （docs/17 §2.5），而 4B base 会倾向"解释我怎么改"而不是"给出可 apply 的
# diff"——实测口径（docs/17 §1 第 3 条）把格式错误率列为主要瓶颈，原因就是
# 这个先验缺口。提示里把"只输出一个 diff 块、不要改测试、只动目标文件"三条
# 写死，是把格式失败率压到最低的唯一免费手段（SFT 前）。
# 零标签字面量：本串不含任何尖括号标签字节。
_SWE_L0_SYSTEM_NATIVE = (
    "You are an expert Python programmer fixing a bug in a small repository.\n"
    "You will be given a bug report, the current content of the file to fix, and "
    "the test file that defines the expected behaviour.\n\n"
    "How to answer:\n"
    "- Produce a unified diff that changes ONLY the file named in the report. Do not "
    "modify or add tests.\n"
    "- The diff must be `git apply -p1` compatible: it starts with `--- a/path` and "
    "`+++ b/path`, then one or more `@@ -start,count +start,count @@` hunks whose "
    "context lines carry one leading space, removed lines a leading `-`, added lines "
    "a leading `+`.\n"
    "- Include enough unchanged context lines around each change that the patch "
    "applies to the current file exactly as shown.\n"
    "- Keep the change minimal: fix the reported behaviour, leave working behaviour "
    "alone.\n"
    "- Output the diff inside a single fenced code block tagged `diff`, and output "
    "nothing else: no explanation before or after it.\n"
)
system_prompt_swe_l0_native = _SWE_L0_SYSTEM_NATIVE


# 原生协议的预算档（docs/09 §5.1 档 A，实跑 validate_retool_budget 核对）：
#   max_rounds(5) × round_gen_tokens(1024) + max_prompt_length(1024) + 预留(798) = 7208 ≤ 8192 ✅
# 依据：参考实现 max_code_calls=4 / max_assistant_turns=6 / 单回合 1024 / 轨迹 ≤8192。
# max_rounds=5 → 只有前 4 轮执行代码（末轮保证 final，docs/02 发现3 的语义不变），
# 恰好对应参考的 max_code_calls=4。
NATIVE_PROTOCOL_DEFAULTS = {
    "max_rounds": 5,
    "round_gen_tokens": 1024,
    "max_context_tokens": 8192,
    # 原生协议自然停在 im_end（EOS 已在 token_ids 里，docs/05 §12.3 已核对），
    # 不需要围栏路径的 stop 串。
    "retool_stop": False,
}


# 【2026-10-09 预算段】budget_hint + token 预算档下追加到系统提示末尾。
# 动机（p6/p7 真机归因）：无 boxed 的 B2 子桶（全程零调用、纯散文写到底被切）
# 从来收不到 [budget] 提示行——提示行只搭工具回包的便车，零调用 = 零提示，
# 模型对总预算完全盲。系统提示是到达 B2 的**唯一**信道；同时它把 [budget] 行的
# 语义与「调用执行判据」（回包 + 作答预留装得下才执行）预先声明，回包里的分档
# 提示（rollout.budget_hint_line）才不需要自解释。
# 门控条件与提示行的生效条件严格一致（native + max_traj_tokens>0 + budget_hint）：
# 提示里承诺了 [budget] 行而实际不注入 = 幻觉指令，比不写更糟。
_BUDGET_PROMPT_MARK = "Token budget:"
_BUDGET_PROMPT_PARA = (
    "\n\nToken budget: the whole trajectory (all your reasoning, tool calls, and "
    "their results) shares a total budget of {mtj} tokens. After each tool result, "
    "a line like \"[budget] N tokens left\" reports the remaining budget. A tool "
    "call is executed only if its result plus about {res} tokens reserved for your "
    "final answer still fits the budget; otherwise the call is discarded and the "
    "trajectory ends unscored. Plan ahead: keep each step short, and when the "
    "budget runs low, stop calling the tool and end immediately with your final "
    "\\boxed{{}} answer."
)


def budget_target_tokens(cfg: dict) -> int:
    """**通告给模型**的轨迹预算（单点真相，纯函数，CPU 可测）。0 = 未开预算档。

    【为什么必须收敛成一个函数】模型的预算认知有三个信道，三者说的数字必须
    是同一个，否则"承诺紧、余量松"立刻失效：
      ① 系统提示里的 "Token budget: N tokens"（inject_budget_prompt）；
      ② 工具回包的 `[budget] M tokens left`（rollout.py 的 `_left`）；
      ③ 调用执行判据 `_can`（rollout.py 的 `traj_budget − used − obs ≥ reserve`）。
    p11c 的事故形态就是 ①=8192 而 ②③=12288：模型一旦读到比承诺更大的剩余额度，
    就把"余量"当成"预算"花掉。把三者绑到本函数，是让这条腿**结构性成立**，
    而不是靠"题目不够难所以没被发现"。

    budget_target=0（缺省）→ 返回 max_traj_tokens，即历史行为逐位不变。"""
    _t = int(cfg.get("budget_target", 0) or 0)
    if _t > 0:
        return _t
    return int(cfg.get("max_traj_tokens", 0) or 0)


def validate_budget_target(cfg: dict) -> None:
    """budget_target 的 fail-fast（纯函数，CPU 可测）。

    【两条硬约束，都是"不报错、只是白跑"的形态，必须在第 0 秒拦】
    1. `budget_target > max_traj_tokens`：通告的数字比保险丝还大 ⇒ 提示在**说谎**，
       模型照着写必然撞 max_traj_tokens 被切（trunc_final 全红），比不开还糟。
    2. `budget_target − answer_reserve − obs_est ≤ 0`：连一次最短调用都装不下
       ⇒ 从第 0 秒起 `_can` 恒 False，每一次调用都被记 code_wasted（结构性无
       boxed），模型收到的 `[budget]` 永远是 "0 left / 别调用"，协议被整段废掉。
       这条与 docs/05 里"预算不自洽 → 丢弃率慢慢爬、训练零产出"同类：不崩，只空转。"""
    _t = int(cfg.get("budget_target", 0) or 0)
    if _t <= 0:
        return
    _mtj = int(cfg.get("max_traj_tokens", 0) or 0)
    if _mtj <= 0:
        raise ValueError(
            f"[config] budget_target({_t}) 需要 token 预算档（max_traj_tokens>0），"
            f"当前 max_traj_tokens={_mtj}。轮数档下没有「轨迹预算」这个量可通告。")
    if _t > _mtj:
        raise ValueError(
            f"[config] budget_target({_t}) > max_traj_tokens({_mtj})：通告的预算"
            f"比真实保险丝还大 = 提示在说谎，模型照着写必然撞 max_traj_tokens 被切。\n"
            f"  正确方向是**反过来的**：budget_target < max_traj_tokens（承诺紧、余量松，"
            f"p9 §7.6 实测 +5.2pp acc / +4.1pp fmt）。")
    _res = max(0, int(cfg.get("answer_reserve", 0) or 0))
    _obs = int(cfg.get("tool_result_max_chars", 500) or 500) // 2 + 16
    if _t - _res - _obs <= 0:
        raise ValueError(
            f"[config] budget_target({_t}) − answer_reserve({_res}) − 工具回包估算"
            f"({_obs}) = {_t - _res - _obs} ≤ 0：连一次最短调用都装不下 → `_can` 恒 "
            f"False，每次调用都被判 code_wasted，模型看到的永远是「0 tokens left / "
            f"别调用」，协议被整段废掉（不报错，只是全员无 boxed）。\n"
            f"  改法：budget_target ≥ answer_reserve + {_obs} + 一点余量，或降 "
            f"answer_reserve。")


def inject_budget_prompt(cfg: dict) -> bool:
    """budget 档开启时把预算段追加进 cfg["system_prompt"]（幂等，返回是否追加）。

    在 get_config 装配末尾与 probe_difficulty 的 --system_prompt_file 之后各调
    一次：两侧最终 prompt 必须逐字节相同（探针出的难度表只对同提示的训练有效）。
    追加后 system_prompt ≠ default_system_prompt → 签名自动带 -sp<hash>，
    旧 run 不会被静默复用。marker 判重：eval 回读 run_info（已含预算段）再走
    get_config 时不会叠加第二遍。

    【2026-10-20】提示里的数字改用 `budget_target_tokens(cfg)`（而不是裸的
    max_traj_tokens）：本段是模型的三个预算信道之一，它必须与 `[budget]` 行、
    `_can` 阈值说同一个数。budget_target=0 时该函数恒返回 max_traj_tokens，
    历史提示逐字节不变。"""
    if not (cfg.get("budget_hint")
            and int(cfg.get("max_traj_tokens", 0) or 0) > 0
            and cfg.get("tool_protocol") == "native"):
        return False
    sp = cfg.get("system_prompt") or ""
    if _BUDGET_PROMPT_MARK in sp:
        return False
    cfg["system_prompt"] = sp + _BUDGET_PROMPT_PARA.format(
        mtj=budget_target_tokens(cfg),
        res=int(cfg.get("answer_reserve", 0) or 0))
    return True


def default_system_prompt(algo: str, tool_protocol: str = "fence") -> str:
    """该算法 preset 的**默认**系统提示（供 run_signature 判定"提示被改过"）。

    【2026-09-17 为什么需要它】提示是探针/训练/评测的单一来源，也是难度表的
    协议的一半——表是"模型×提示×预算"三者的联合产物。改了提示却不进签名，
    就会把"新提示下探的表"和"旧提示下的 run"当成同一配方对照（与
    `vllm_gen_kwargs` 不进签名同属一类静默偏离）。

    【2026-09-25】加 tool_protocol 维：原生协议的提示与围栏版**必须不同**
    （删围栏措辞），只按 algo 取名会让"改了协议没改提示"被签名判成"无偏离"。"""
    if algo == "retool_math":
        return (system_prompt_retool_math_native if tool_protocol == "native"
                else system_prompt_retool_math)
    # 【docs/16】search_qa 只有原生档（检索工具必须由模板的 tools 段声明）
    if algo == "search_qa":
        return system_prompt_search_qa_native
    # 【docs/17】swe_l0 是单轮 patch-only：没有工具声明，提示与协议档无关
    # （tool_protocol 对它是死键——传了也不会有工具段）。
    if algo == "swe_l0":
        return system_prompt_swe_l0_native
    if algo == "retool":
        return system_prompt_retool
    return BASE["system_prompt"]


def validate_max_prompt_length(cfg: dict) -> None:
    """max_prompt_length 必须是正整数（fail-fast，纯函数，CPU 可测）。

    【2026-09-29 为什么必须查】它自本轮起可从 CLI 覆盖（此前只能改 config 源码），
    而它是**所有算法**共用的硬跳组线：rollout 采样循环里
    `if plen > cfg["max_prompt_length"]: continue` —— 池内题目的 prompt 全部超限时，
    这个 continue 会把每次 attempt 都吞掉，外层只剩"丢弃率慢慢爬"，训练端表现为
    **零产出空转**（2026-09-12 丢弃率 90% 事故的非崩溃形态：不报错、不崩，只是
    5 小时什么都没采到）。0/负数更糟：0 让每一题都被判超限。这里 fail-fast，把
    "跑了半天才发现一题没采"变成第 0 秒的错。

    上界**不设硬线**：长 prompt 该不该采是用户对数据的判断（由
    validate_retool_budget 的不变量与实测 plen 分布去约束），这里只管"能不能工作"。

    【为什么提成独立函数】get_config 之后仍有人直接改 cfg：probe_difficulty 在
    CLI 覆盖后重跑预算校验就是这个形态。校验若只写在 get_config 里，探针路径会
    绕过它 → 两条路径的"合法配置"定义漂移（本项目同类 bug 的常见形态）。
    """
    _mpl = cfg.get("max_prompt_length")
    if not isinstance(_mpl, int) or isinstance(_mpl, bool) or _mpl <= 0:
        raise ValueError(
            f"[config] max_prompt_length 必须是正整数，收到 {_mpl!r}。\n"
            f"  后果：它是采样循环的跳组线（plen > 该值即 continue 不采），0 会让"
            f"**每一题**都超限 → 采样空转、训练端零产出（不报错、不崩）。\n"
            f"  改法：GSM8K 家族用 400（BASE 默认）、retool_math 用 1024（preset）；"
            f"调小到库内多数题之下同样会空转，需配合实测的 plen 分布。")


def validate_guard_conflicts(cfg: dict) -> dict:
    """正确性优先档的**互斥项净化**（纯函数，CPU 可测）：返回被归零的键列表。

    【2026-10-19 实测冲突】tool_credit_advantages 对截断末段是**覆盖**语义
    （`val = -ttp`, 见 rollout.py:1241），不是叠加——它的设计前提是
    "截断轨迹 base_adv 恒为 0（已被 sample_mask 排除出组均值）"，即 ttp 是给
    '零信号轨迹' 补的一点局部负信用。

    而 task_adv_guard + trunc_counts_as_wrong 让截断样本以 acc=−1 进组均值，
    base_adv 变成真实组内负优势（6对2错组 = −1.5，实测净负信用 −1350）。此时
    末段覆盖会把强信号**替换**成弱信号（ttp0.2 → 净负信用 −630，砍掉 53%）：
        tcw=T, ttp=0   → −1350（正确）
        tcw=T, ttp=0.2 →  −630（自相矛盾）

    同理 tool_waste_penalty 的 `wasted` 分支也是覆盖语义，而 code_wasted 在 tcw
    下同样进组统计。故 guard 档下两者一律归零，并打印明确原因——静默保留会让
    "惩罚强度"变成不可预期的配置组合，而这正是本项目反复出事的形态。
    """
    zeroed = []
    if not cfg.get("task_adv_guard"):
        return zeroed
    if not cfg.get("trunc_counts_as_wrong", True):
        return zeroed
    for _k in ("trunc_tail_penalty", "tool_waste_penalty"):
        if float(cfg.get(_k, 0.0) or 0.0) > 0.0:
            cfg[_k] = 0.0
            zeroed.append(_k)
    return zeroed


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
    if not (cfg.get("algo", "").startswith("retool")
            or cfg.get("algo") == "search_qa"):
        return 0
    rounds = int(cfg.get("max_rounds", 1) or 1)
    per_round = int(cfg.get("round_gen_tokens", 0) or 0)
    reserve = (max(0, rounds - 1)
               * (int(cfg.get("tool_result_max_chars", 0) or 0) // 2 + 16))
    # 【2026-09-29 token 预算模式】不变量换成"整条轨迹的 token 预算 + 最长 prompt
    # ≤ 全轨迹保险丝"。此时 round_gen_tokens 退化成**单轮上限**（防一轮写飞），
    # 不再参与"轮数 × 单轮"的乘积预算——乘积口径要求"每轮都写满"，而真实轨迹
    # 远短于此（native_p3 实测 avg_clen 2811 vs 乘积 10240），那个乘积只是假想
    # 上限，用它当预算既高估了需求也掩盖了"轮数不是预算"这一事实。
    _mtj = int(cfg.get("max_traj_tokens", 0) or 0)
    if _mtj > 0:
        ctx_t = int(cfg.get("max_context_tokens", 0) or 0)
        if ctx_t <= 0:
            return 0
        _plen_t = int(cfg.get("max_prompt_length", 0) or 0)
        _need_t = _mtj + _plen_t
        if _need_t > ctx_t:
            raise ValueError(
                f"[config] retool token 预算不自洽（max_traj_tokens>0 档）："
                f"max_traj_tokens({_mtj}) + max_prompt_length({_plen_t})"
                f" = {_need_t} > max_context_tokens({ctx_t})。\n"
                f"  后果：合法用满预算的轨迹会被 retool_context_overlong 整组丢弃"
                f"（2026-09-12 事故：丢弃率 90%、采样空转、训练端零产出）。\n"
                f"  改法：max_context_tokens ≥ {_need_t}（= max_prompt_length + "
                f"max_traj_tokens），或降 max_traj_tokens。\n"
                f"  注意：token 预算档下 round_gen_tokens 只是**单轮上限**"
                f"（当前 {per_round}），不参与本不变量。")
        _check_concurrency_hint(cfg)
        _check_round_budget_hint(cfg, _mtj, per_round)
        return reserve
    if per_round <= 0:
        return 0
    ctx = int(cfg.get("max_context_tokens", 0) or 0)
    if ctx <= 0:
        return 0
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
            f"或抬高 max_context_tokens ≥ {need}（T 进入所有显存公式，需重算峰值）。\n"
            f"  （或改用 token 预算档：--max_traj_tokens >0，不变量换成 "
            f"max_traj_tokens + max_prompt_length ≤ max_context_tokens。）")
    _check_concurrency_hint(cfg)
    return reserve


def _check_concurrency_hint(cfg: dict) -> None:
    """并采题数调大而 sandbox_workers 没跟上时的提示（非致命，只在真偏离时报）。

    【2026-09-28 并发契约提示】并采题数调大而 sandbox_workers 没跟上时，多轮
    rollout 的一轮里会有代码执行排队等线程——生成端刚从"vLLM 欠利用"换成
    "沙箱成新瓶颈"。
    【为什么必须比对 preset 基线而不是直接比 并发>workers】preset 本身就是
    `gq=4 × num_pre_Q=8 = 32 > sandbox_workers=8`——那是**有意的**（不是每条
    轨迹都写代码，单次沙箱 0.1~5s 远小于轮长）。直接比大小会让提示在 shipped
    preset 上每次都喊（实测一套测试刷 44 行），正好复现 health.retool_trunc
    的教训：**检测器对已知基线叫狼来了，真信号就被"忽略习惯"淹掉**。
    现在只在"用户把并采题数抬到 preset 之上、沙箱却仍停在 preset 及以下"时报，
    即真正的"调大并发忘调沙箱"。
    """
    _preset = ALGO_DEFAULTS.get(cfg.get("algo"), {}) or {}
    _gq = int(cfg.get("gen_questions_per_attempt", 1) or 1)
    _gq0 = int(_preset.get("gen_questions_per_attempt",
                           BASE.get("gen_questions_per_attempt", 1)) or 1)
    _sw = int(cfg.get("sandbox_workers", 0) or 0)
    _sw0 = int(_preset.get("sandbox_workers", BASE.get("sandbox_workers", 0)) or 0)
    _conc = _gq * int(cfg.get("num_pre_Q", 1) or 1)
    if _sw and _gq > _gq0 and _conc > _sw and _sw <= _sw0:
        print(f"[config][提示] 并采题数已从 {_gq0} 抬到 {_gq}（并发 {_conc} 条），"
              f"但 sandbox_workers 仍是 {_sw} → 有代码的轮次里沙箱执行会排队，"
              f"可能成为新瓶颈；若生成端吞吐仍不达标，把它抬到 ≥{_conc} 再测。")


def _check_round_budget_hint(cfg: dict, mtj: int, per_round: int) -> None:
    """单轮上限小于轨迹预算时的提示（非致命）。

    【2026-09-29 为什么要有这条】token 预算档下每轮实际额度是
    `max_tokens = max(1, min(round_gen_tokens, max_traj_tokens − used))`。

    `round_gen_tokens ≥ max_traj_tokens` 时 min 恒取 `mtj − used`：撞 length 就
    意味着预算恰好用尽 → 下一轮循环顶部即判 `trunc_final=1` 出局 → **续写分支
    不可达**（`_iter_cap` 也不再是 `max_rounds`）。此时"单轮上限"与"轨迹预算"
    是同一个数，`round_gen_tokens` 不再是一个独立旋钮。

    `round_gen_tokens < max_traj_tokens` 时单轮会被切断一次、由循环续写拼接。
    已用 FakeGen 对拍：**逐 token 完全等价**（P=8/16/20/24 与 P=M 产出的 ids
    逐位相同）——上下文、采样参数、RNG 都没变，只是把一次 generate 拆成几次。
    代价仅为 generate 调用次数变多（每次重算 prefix、可观测性变差）。

    【边界情形：实测是**不稳定**的，不要依赖它】`parse_assistant` 对"调用块之后
    还有文本"判 invalid。模型写出"调用+尾随散文"时，实测（M=600，调用块 90
    token，流长 145）：
        P=10 → 执行了调用（chunk 边界恰好落在调用块末尾）
        P=20 → invalid ；P=30 → 执行了调用 ；P=40/80/200/600 → invalid
    即 P<M 是否"救回"这次调用**取决于 chunk 边界与调用块末尾是否巧合对齐**，
    不是可依赖的性质（本函数初版曾断言"续写恰好救回 tool"——实测证伪）。
    故正确读法是：P<M 会让 `invalid` 率**依赖于一个纯实现参数**，而 P≥M 是
    确定性的。推荐 P≥M；若确实要 P<M，必须开 `--native_stop_at_call`
    （在 </tool_call> 处停，尾随文本根本不产生，两取值行为一致）。
    """
    if not cfg.get("native_stop_at_call", False) and per_round and mtj > 0:
        if per_round < mtj:
            print(f"[config][提示] token 预算档下 round_gen_tokens({per_round}) < "
                  f"max_traj_tokens({mtj})，且未开 --native_stop_at_call：单轮会被"
                  f"切断并由循环续写（逐 token 等价，但 generate 次数变多；且 "
                  f"invalid 率会依赖 chunk 边界对齐，行为不确定）。"
                  f"建议把 round_gen_tokens 设为 ≥ {mtj}（单轮即用满预算，续写不可达），"
                  f"或至少开 --native_stop_at_call。")


def _validate_swe_l0_config(cfg: dict) -> dict:
    """SWE-RL L0 的配置校验（纯函数，CPU 可测）。返回一份自证 dict。

    【为什么必须 fail-fast 在这里】L0 的 reward 来自**真实测试执行**：配置错了
    不会报错，只会让每条轨迹都拿 -1（= acc 平坦 0%），而那正是本项目认定的
    系统性 bug 签名。能在第 0 秒拦下的错，绝不留到训练中期去"从曲线反推"。
    """
    out = {}
    # ① 单轮路径：多轮专属键传了也没有受力面（静默空转是本项目最贵的一类 bug）
    if (cfg.get("tool_protocol") or "fence") != "fence":
        raise ValueError(
            "[config] swe_l0 是**单轮 patch-only**（docs/17 §2.5），不支持 "
            f"tool_protocol={cfg.get('tool_protocol')!r}：它那条路径只声明工具、"
            "多轮续写，而 L0 的动作空间就是一段 diff，没有工具可调。\n"
            "  用 --tool_protocol fence（默认）跑 L0；多轮工具是 L1/L2 的事。")
    if int(cfg.get("max_rounds", 1) or 1) != 1:
        raise ValueError(
            f"[config] swe_l0 必须 max_rounds=1（单轮），当前 "
            f"{cfg.get('max_rounds')}——多轮在 L0 下没有定义：末轮才有最终 diff，"
            f"前面几轮产出无人消费，只会白烧生成预算。")
    # ② 生成长度必须装得下一条 diff
    _gen = int(cfg.get("max_gen_tokens", 0) or 0)
    if _gen < 512:
        raise ValueError(
            f"[config] swe_l0 的 max_gen_tokens={_gen} 太小：一条单函数 diff "
            f"（含上下文行）通常 300~800 token，加推理会更多。建议 ≥1024。")
    # ③ 预算自洽：单轮路径下 prompt + 生成必须在上下文保险丝内
    _ctx = int(cfg.get("max_context_tokens", 0) or 0)
    _plen = int(cfg.get("max_prompt_length", 0) or 0)
    if _ctx > 0 and _plen + _gen > _ctx:
        raise ValueError(
            f"[config] swe_l0 预算不自洽：max_prompt_length({_plen}) + "
            f"max_gen_tokens({_gen}) = {_plen + _gen} > "
            f"max_context_tokens({_ctx})。\n"
            f"  后果：用满生成预算的合法轨迹会被判超长整组丢弃 → 采样空转"
            f"（2026-09-12 事故的同类形态，只是换成了单轮路径）。\n"
            f"  改法：max_context_tokens ≥ {_plen + _gen}，或降 max_prompt_length。")
    # ④ 判分执行器的资源项必须可用（0/负数会让每条都超时或 OOM）
    _to = float(cfg.get("swe_test_timeout", 0) or 0)
    if _to <= 0:
        raise ValueError(
            f"[config] swe_test_timeout 必须 > 0（当前 {_to}）：0 会让每次测试"
            f"立即超时 → 所有轨迹 -1（acc 平坦 0% 的系统性签名）。")
    _mem = int(cfg.get("swe_test_mem_mb", 0) or 0)
    if _mem < 256:
        raise ValueError(
            f"[config] swe_test_mem_mb={_mem} 太小：跑 pytest 至少要 ~256MB "
            f"（import pytest 本身就要 80~120MB），否则每条都 OOM → 全 -1。")
    _wrk = int(cfg.get("swe_test_workers", 0) or 0)
    if _wrk <= 0:
        raise ValueError(
            f"[config] swe_test_workers 必须 > 0（当前 {_wrk}）：它是判分并发，"
            f"0 会让组内每个 patch 串行判，rollout 墙钟被测试执行主导。")
    # ⑤ 单轮路径不消费的键：传了等于没传（静默空转是本项目最贵的一类 bug）
    _gq = int(cfg.get("gen_questions_per_attempt", 1) or 1)
    if _gq > 1:
        raise ValueError(
            f"[config] swe_l0 走**单轮**路径，不读 gen_questions_per_attempt="
            f"{_gq}（rollout 的 use_qqueue 要求 is_retool，L0 恒 False）→ "
            f"传了等于没传。\n"
            f"  要提生成吞吐/组内多样性：抬 --num_pre_Q（一次 attempt 采的轨迹数），"
            f"它才真的改 vLLM 并发与组内 patch 分布。")
    # ⑥ 长度项/工具信用项在 L0 无受力面：>0 只会引入"短 patch 骗分"的捷径
    for _key in ("code_w", "code_attempt_w", "len_penalty_w", "tool_call_cost",
                 "tool_waste_penalty", "trunc_tail_penalty", "len_eff_w",
                 "len_eff_penalty_w"):
        _v = float(cfg.get(_key, 0.0) or 0.0)
        if _v > 0.0:
            raise ValueError(
                f"[config] swe_l0 不支持 {_key}={_v}：L0 的 reward 是 F2P 硬判据"
                f"（docs/17 §3），patch 长度与'修对没修对'无关，长度/工具项在这里"
                f"只会开出一条'写短 patch 骗分包'的捷径。\n"
                f"  要试 shaping 请另起 algo 名并写清它是哪个变量的单变量实验。")
    out["max_gen_tokens"] = _gen
    out["max_prompt_length"] = _plen
    out["max_context_tokens"] = _ctx
    out["swe_test_timeout"] = _to
    out["swe_test_mem_mb"] = _mem
    out["swe_test_workers"] = _wrk
    return out


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
    # 工具协议档位校验（fail-fast 在 rollout/生成端之前——生成端崩溃的表现只是
    # 训练端一句"生成端进程已退出"，错误必须留在能排查的地方）
    if cfg.get("tool_protocol") not in TOOL_PROTOCOLS:
        raise ValueError(
            f"[config] 未知 tool_protocol={cfg.get('tool_protocol')!r}，可选 "
            f"{TOOL_PROTOCOLS}（fence=p1–p11 围栏协议；native=Qwen 原生 tool_call）")
    # 【2026-09-25 静默空转防线】native 只对 retool 家族有意义——多轮 rollout
    # （multi_turn_rollout_group / collect_retool_group）才是它的消费点，单轮路径
    # 从不读它。在 grpo/dapo 上传 native = 用户以为开了原生工具协议、实际什么都没
    # 发生（本项目最贵的一类 bug：成功的表象由空动作伪造）。当场拦下。
    # 【docs/16】search_qa 是同样走多轮路径的第三个消费者，故一并放行。
    if cfg.get("tool_protocol") == "native" and not algo.startswith("retool") \
            and algo != "search_qa":
        raise ValueError(
            f"[config] tool_protocol='native' 只对 retool 家族有效（search_qa 亦可），"
            f"当前 algo={algo!r} 走单轮路径（从不读该键）→ 传了等于没传。\n"
            f"  要跑原生工具协议：--algo retool_math（或 retool / search_qa）。")
    for _key in ("tool_call_cost", "tool_waste_penalty", "trunc_tail_penalty",
                 "len_eff_penalty_w"):
        if float(cfg.get(_key, 0.0) or 0.0) < 0.0:
            raise ValueError(f"[config] {_key} 必须 >= 0，收到 {cfg.get(_key)!r}")
    # 原生协议预算档（docs/09 §5.1 档 A）。显式 override 优先（CLI 单变量微调位）。
    # 【docs/16 例外】search_qa 自带一套检索档预算（8×512/14336，回包更长、轮数更多），
    # 不能被这套为 code 工具调的默认值静默覆盖——那正是本项目最忌讳的"preset 被默认
    # 档悄悄改掉"（改完表现为预算不自洽 fail-fast，或更糟：能跑但预算口径不是你要的）。
    if cfg.get("tool_protocol") == "native" and algo != "search_qa":
        for _k, _v in NATIVE_PROTOCOL_DEFAULTS.items():
            if _k not in overrides:
                cfg[_k] = _v
    # retool 专用系统提示（除非用户显式覆盖）
    if algo == "retool" and "system_prompt" not in overrides:
        cfg["system_prompt"] = system_prompt_retool
    if algo == "retool_math" and "system_prompt" not in overrides:
        cfg["system_prompt"] = default_system_prompt(algo, cfg["tool_protocol"])
    if algo == "search_qa" and "system_prompt" not in overrides:
        cfg["system_prompt"] = default_system_prompt(algo, cfg["tool_protocol"])
    # 【docs/17】swe_l0 的提示与协议档无关（单轮、无工具声明），同样只在未显式
    # 覆盖时套 preset。
    if algo == "swe_l0" and "system_prompt" not in overrides:
        cfg["system_prompt"] = default_system_prompt(algo, cfg["tool_protocol"])
    # 预算档系统提示段（preset 与 --system_prompt_file 自定义提示都吃）：必须发生在
    # override 合并之后，且幂等（marker 判重），详见 inject_budget_prompt docstring。
    inject_budget_prompt(cfg)
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
    # vLLM 引擎参数透传的类型闸：非 dict（如 CLI 递了 JSON 数组/字符串）若原样进
    # LLM(**x)，会以 TypeError 在**生成端子进程**里炸——而生成端崩溃的表现只是训练端
    # 一句"生成端进程已退出"（见 BASE.vllm_gen_kwargs 注释里的同类盲区）。在配置层
    # fail-fast，错误才留在能排查的地方。
    _vk = cfg.get("vllm_gen_kwargs")
    if _vk is not None and not isinstance(_vk, dict):
        raise ValueError(
            f"[config] vllm_gen_kwargs 必须是 dict 或 None，收到 "
            f"{type(_vk).__name__}（{_vk!r}）；CLI 形态："
            "--vllm_gen_kwargs '{\"gdn_prefill_backend\": \"triton\"}'")
    # 【2026-09-20 micro-batch 契约推导】train_micro_batch_size_per_gpu 必须等于
    # 一个上传批的行数 = Q_batch_size × num_pre_Q（生成端按题构造 num_pre_Q 行批）。
    # 旧版把两者在 preset 里**硬编码成对**（retool_math: 8/8）却没有任何推导或
    # 校验 —— 于是 `--num_pre_Q 4`（记录在案的 OOM 回退位）只改前者，DS 仍声明
    # micro_batch=8：梯度缩放分母按 8 算而实际只有 4 行，**等效 lr 被静默腰斩**，
    # 有效 batch 也从声明的 32 变成 16。这类"配置自相矛盾但不报错"最难排查。
    _implied_mb = int(cfg.get("Q_batch_size", 1) or 1) * int(cfg.get("num_pre_Q", 1) or 1)
    if "train_micro_batch_size_per_gpu" not in overrides:
        cfg["train_micro_batch_size_per_gpu"] = _implied_mb
    elif cfg["train_micro_batch_size_per_gpu"] != _implied_mb:
        raise ValueError(
            f"[config] micro-batch 契约不符：train_micro_batch_size_per_gpu="
            f"{cfg['train_micro_batch_size_per_gpu']} 但 Q_batch_size×num_pre_Q="
            f"{cfg.get('Q_batch_size')}×{cfg.get('num_pre_Q')}={_implied_mb}。\n"
            f"  后果：DeepSpeed 按声明值缩放梯度，与实际上传行数不符 → 等效 lr 被"
            f"静默改变（--num_pre_Q 4 这类回退位最易踩）。\n"
            f"  改法：不要显式传 train_micro_batch_size_per_gpu，让它由 "
            f"Q_batch_size×num_pre_Q 推导。")
    # 【2026-09-29 prompt 上限校验·防采样空转】见 validate_max_prompt_length 的
    # 事故说明（0/负数 → 每题都跳组 → 训练端零产出空转）。
    validate_max_prompt_length(cfg)
    # 【docs/16】search_qa 的检索服务必须显式给出 endpoint（fail-fast 在启动期）：
    # 缺它就等于每条轨迹都执行失败 → 全题无答案 → 训练"跑得动但学不到"，而日志里
    # 看不到任何异常（只表现为 acc 平坦 = 本项目认定的系统性 bug 签名）。
    if cfg.get("algo") == "search_qa":
        _ep = cfg.get("search_endpoint")
        if not (isinstance(_ep, str) and _ep.strip()):
            raise ValueError(
                "[config] search_qa 必须指定检索服务 endpoint（--search_endpoint）。\n"
                "  形态：http://<host>:<port>（独立部署，docs/16 §3）或 local:<索引路径>。\n"
                "  启动索引：python -m rlab.search_server build-index <corpus.jsonl> "
                "<index.sqlite>，再 python -m rlab.search_server serve <index.sqlite> "
                "--host 0.0.0.0 --port 8000")
        if cfg.get("tool_protocol") != "native":
            raise ValueError(
                "[config] search_qa 只支持 tool_protocol='native'"
                f"（检索工具必须由 chat template 的 tools 段声明），当前 "
                f"{cfg.get('tool_protocol')!r}。")
        _tn = cfg.get("native_tool_name")
        if _tn != "search":
            raise ValueError(
                f"[config] search_qa 的 native_tool_name 必须是 'search'，当前 {_tn!r}"
                "（该键驱动模板声明/解析/执行器三处，改错即三处错配）。")
    # 多轮预算自洽（fail-fast；见 validate_retool_budget 的事故说明）
    cfg["_tool_reserve"] = validate_retool_budget(cfg)
    # 【2026-10-20】通告预算（budget_target）的两条"不报错只白跑"约束
    # （见 validate_budget_target：通告>保险丝=说谎；连一次调用都装不下=全员废码）
    validate_budget_target(cfg)
    # 【2026-10-19】正确性优先档的互斥项净化（ttp/twp 与 tcw 的覆盖语义冲突）
    _zeroed = validate_guard_conflicts(cfg)
    if _zeroed:
        cfg["_guard_zeroed"] = _zeroed
        print(f"[config][守卫] task_adv_guard + trunc_counts_as_wrong 已生效，"
              f"自动归零 {_zeroed}：tool_credit_advantages 对截断末段/废码是"
              f"**覆盖**语义（为 base_adv=0 的旧口径设计），而 tcw 下截断样本已有"
              f"真实组内负优势（实测净负信用 −1350 vs 覆盖后 −630），保留会削掉"
              f"一半惩罚。", flush=True)
    # 【docs/17 SWE-RL L0】判分执行器的 fail-fast（纯配置层，不碰文件/子进程）。
    if cfg.get("algo") == "swe_l0" or cfg.get("data_task") == "swe_l0":
        _swe = _validate_swe_l0_config(cfg)
        cfg["_swe_check"] = _swe
    # 【2026-09-18 H2 护栏】vllm_gen_logps 档位：N=0 是 docs/07 实锤的坏路径
    # （logprobs=0 只报被采样 token，在 vLLM v0.19.1 + Qwen3.5 GDN 上 prefill 位
    # 单点差 6.8 nat、与 torch 独立重算不符），正确用法需 N≥1（top-K 里挑被采样
    # token）+ batch_invariant + 显式 attention_backend 四件套。默认 False 的
    # torch 副本路径是合法的（A/B、旧 run 复现），故这里 warn 不 fail-fast——
    # 但开了 gen_logps 还停在 N=0，就是静默跑在已证伪路径上。
    if cfg.get("vllm_gen_logps") and not cfg.get("vllm_logprobs_n"):
        print("[config][警告] vllm_gen_logps=True 但 vllm_logprobs_n=0 —— docs/07 实锤的"
              "坏路径（N=0 只报被采样 token，prefill 位单点差 6.8 nat）。"
              "正确用法四件套: --vllm_gen_logps --vllm_logprobs_n 1 "
              "--vllm_batch_invariant --vllm_attention_backend FLASH_ATTN")
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
