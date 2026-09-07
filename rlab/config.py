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
}

BASE = dict(
    # ---- 模型与路径 ----
    model_path="/root/Qwen2.5-3B",          # base 版，与历史实验可比
    data_task="gsm8k",                       # gsm8k（阶段0/1）；阶段2/3 扩展
    out_dir="./rlab_out",
    record_path="./rlab_out/record.jsonl",   # 生成数据得分记录（analysis.py 消费）

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

    # ---- 训练 ----
    all_steps=300,
    save_steps=100,
    gen_update_steps=16,     # 每 N 个 optimizer step 推送权重给生成端
    train_micro_batch_size_per_gpu=4,   # = Q_batch_size*num_pre_Q
    gradient_accumulation_steps=4,
    lr=1e-6,
    warmup_steps=0,

    # ---- 基础设施 ----
    gen_device=0,            # vLLM 生成 + torch gen_logps 副本所在物理卡
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
    overlong_shaping=False,
    overlong_buffer=64,      # DAPO 软悬崖缓冲区宽度

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
    max_rounds=3,            # assistant+tool 最多成对轮数
    round_gen_tokens=280,    # 每轮 assistant 段生成长度上限（控制总上下文）
    tool_result_max_chars=500,  # 沙箱输出截断长度（防输出炸弹）
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
_RETOOL_EXTRA = (
    "\n\nYou MAY write Python code to help solve the problem. If you do, put each "
    "piece of code inside a fenced block like: ```python\n<your code>\n```\n"
    "The environment executes your code automatically and inserts the result "
    "between [TOOL RESULT] and [/TOOL RESULT]. Read the result and continue "
    "reasoning until you reach the final answer inside the required answer tags."
)
system_prompt_retool = BASE["system_prompt"] + _RETOOL_EXTRA


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
    if cfg["wandb_name"] is None:
        cfg["wandb_name"] = f"{algo}"
    # 输出目录按算法隔离（防 grpo/dapo 的 step_N checkpoint 与 record 互相覆盖）
    if "out_dir" not in overrides:
        cfg["out_dir"] = os.path.join(cfg["out_dir"], algo)
    if "record_path" not in overrides:
        cfg["record_path"] = os.path.join(cfg["out_dir"], "record.jsonl")
    cfg["ref_server"] = f"http://{cfg['ref_server_host']}:{cfg['ref_server_port']}"
    return cfg


def ds_config(cfg: dict) -> dict:
    """DeepSpeed 配置。ZeRO stage 0：3B 全态 ~60G < 96G，不开 offload 防 CPU OOM/-9。"""
    return {
        "train_micro_batch_size_per_gpu": cfg["train_micro_batch_size_per_gpu"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "steps_per_print": 5,
        "optimizer": {"type": "AdamW", "params": {"lr": cfg["lr"]}},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 0},
    }
