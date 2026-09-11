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
    # CISPO：截断保留 min(ratio, 1+eps) 梯度，token 级
    "cispo":   dict(beta=0.04, clip_low=0.2, clip_high=0.2, adv_mode="group_std",
                    loss_norm="token_mean"),
    # GSPO：sequence 级 importance ratio 与 clip
    "gspo":    dict(beta=0.04, clip_low=0.2, clip_high=0.2, adv_mode="group_std",
                    loss_norm="seq_mean"),
    # RF++：全局基线（非组内）、per-token advantage、token 级、无 KL
    "rfpp":    dict(beta=0.0,  clip_low=0.2, clip_high=0.2, adv_mode="global_mean",
                    loss_norm="token_items"),
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
    temperature=0.9,
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

    # ---- 可复现种子（None=旧行为不设种子；设了则抽题顺序与生成采样均可复现）----
    # 【2026-09-04 教训】dapo 同代码重跑 78.0→74.3(-3.7pp)：±2pp 噪声地板只覆盖
    # "同 checkpoint 评两次"的评测噪声，从未覆盖训练运行间方差（抽题顺序+生成采样无种子）。
    # 阶段1 起对比实验一律固定 seed，必要时双 seed 复跑。
    seed=None,
)


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
