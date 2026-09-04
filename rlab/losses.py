# -*- coding: utf-8 -*-
"""rlab/losses.py — 可插拔 loss 与 advantage 计算（阶段0 的核心交付）。

所有算法共享同一数据契约（见 protocol.py），只在本文件的
compute_advantages() / compute_loss() 里产生差异：

  grpo    : 对称 clip、样本级 token-mean 归一化（simple_grpo_v1/grpo_ref_split 原版）
  dapo    : clip-higher 解耦（上界放宽）+ 全 batch token 级归一化
  dr_grpo : advantage 不除 std（去 1/std 偏差）+ loss 除固定常数（去长度归一化偏差）
  cispo   : 截断方向只截上升、被截 token 保留 min(ratio, 1+eps) 梯度
  gspo    : importance ratio 与 clip 从 token 级提升到 sequence 级
  rfpp    : 全局基线 per-token advantage + token 级（num_items 全局计数）、无 KL

依赖仅 torch，可在 CPU 冒烟环境直接单测。
"""

import torch

# ---------------------------------------------------------------- 基础件 ----

def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """逐行 log_softmax + gather，控制显存峰值（沿用现有实现）。"""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def compute_advantages(rewards: torch.Tensor, group_size: int, mode: str,
                       eps: float = 1e-4) -> torch.Tensor:
    """奖励 -> advantage（生成端调用，上传前完成）。

    rewards: (G,) 展平的每条样本原始奖励，G = Q_batch_size * num_pre_Q
      group_std  : 组内 (r-mean)/std          —— GRPO/DAPO/CISPO/GSPO
      group_mean : 组内 r-mean（不除 std）     —— Dr.GRPO（去 1/std 偏差）
      global_mean: 全局 r-mean（基线不分组）    —— RF++（组内对比仍在，但基线跨组共享）
    """
    G = rewards.numel()
    assert G % group_size == 0, f"奖励数 {G} 不是组大小 {group_size} 的整数倍"
    r = rewards.float()
    if mode == "global_mean":
        return r - r.mean()
    groups = r.view(G // group_size, group_size)
    if mode == "group_mean":
        return (groups - groups.mean(dim=1, keepdim=True)).view(-1)
    if mode == "group_std":
        return ((groups - groups.mean(dim=1, keepdim=True))
                / (groups.std(dim=1, keepdim=True) + eps)).view(-1)
    raise KeyError(f"未知 adv_mode {mode!r}")


def _k3_kl(ref_logps: torch.Tensor, policy_logps: torch.Tensor) -> torch.Tensor:
    """k3 估计：exp(r-p) - (r-p) - 1，无偏低方差（沿用现有实现）。"""
    d = ref_logps - policy_logps
    return torch.exp(d) - d - 1.0


def _adv_broadcast(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) 或 (B,T) 的 advantage 统一成可与 mask 逐元素相乘的形状。"""
    if advantages.dim() == 1:
        return advantages.unsqueeze(1)          # (B,1) 对 (B,T) 广播
    if advantages.shape == mask.shape:
        return advantages
    raise ValueError(f"advantage 形状 {tuple(advantages.shape)} 与 mask {tuple(mask.shape)} 不兼容")


def _finalize(loss: torch.Tensor, per_token_loss: torch.Tensor, ratio: torch.Tensor,
              mask: torch.Tensor) -> tuple:
    """附统计量：clip 比例 / 近似 KL / 平均 ratio（全部只统计有效 token）。"""
    with torch.no_grad():
        m = mask.bool()
        clip_frac = (((ratio > 1.2) | (ratio < 0.8)) & m).sum() / m.sum().clamp(min=1)
        approx_kl = ((ratio.log() ** 2)[m]).mean()
        mean_ratio = (ratio[m]).mean()
    stats = {"loss": float(loss.item()), "clip_frac": float(clip_frac.item()),
             "approx_kl": float(approx_kl.item()), "mean_ratio": float(mean_ratio.item())}
    return loss, stats

# ------------------------------------------------------------- loss 主体 ----

def compute_loss(algo: str, policy_logps: torch.Tensor, gen_logps: torch.Tensor,
                 advantages: torch.Tensor, mask: torch.Tensor, cfg: dict,
                 ref_logps: torch.Tensor = None,
                 num_items_in_batch: torch.Tensor = None) -> tuple:
    """统一入口。

    policy_logps/gen_logps/refs : (B, T) completion 区 per-token logps（prompt 已裁掉）
    advantages                  : (B,) 标量 adv（组内已归一化）或 (B,T) per-token（rfpp）
    mask                        : (B, T) completion 有效位（pad=0），float
    num_items_in_batch          : rfpp 用——全局累积批的有效 token 总数（防梯度累积偏差）
    返回 (loss, stats)；loss 为标量 tensor（可 backward），stats 为纯 python dict。
    """
    beta = float(cfg.get("beta", 0.04))
    lo, hi = float(cfg.get("clip_low", 0.2)), float(cfg.get("clip_high", 0.2))
    adv = _adv_broadcast(advantages.to(policy_logps.device), mask)
    mask = mask.to(policy_logps.device)
    gen_logps = gen_logps.to(policy_logps.device)

    ratio = torch.exp(policy_logps - gen_logps)
    clipped = torch.clamp(ratio, 1 - lo, 1 + hi)
    pg_term = torch.min(ratio * adv, clipped * adv)   # PPO-clip 风格目标（未取负）
    kl_term = beta * _k3_kl(ref_logps.to(policy_logps.device), policy_logps) \
        if (beta > 0 and ref_logps is not None) else 0.0

    norm = cfg.get("loss_norm", "sample_mean")

    if algo in ("grpo", "dapo", "rfpp"):
        per_token_loss = -(pg_term - kl_term)
        if norm == "sample_mean":       # GRPO：样本级（每条样本 token 平均后再 batch 平均）
            loss = (per_token_loss * mask).sum(dim=1).div(mask.sum(dim=1)).mean()
        elif norm == "token_mean":      # DAPO：全 batch 按 token 归一化
            loss = (per_token_loss * mask).sum() / mask.sum()
        elif norm == "token_items":     # RF++：除以全局累积批有效 token 数
            denom = num_items_in_batch if num_items_in_batch is not None else mask.sum()
            loss = (per_token_loss * mask).sum() / denom
        else:
            raise KeyError(f"{algo} 不支持 loss_norm={norm!r}")

    elif algo == "dr_grpo":
        assert norm == "token_const"
        per_token_loss = -(pg_term - kl_term)
        const = cfg.get("dr_grpo_const") or cfg.get("max_gen_tokens", 512)
        # 除固定常数（B*max_len）而非批内真实 token 数：消除"长回答被摊薄"的长度偏差
        loss = (per_token_loss * mask).sum() / (mask.shape[0] * const)

    elif algo == "cispo":
        assert norm == "token_mean"
        # CISPO：只截上升方向；被截断 token 保留 min(ratio, 1+eps) 的梯度
        # L = -1/|e| Σ sg(1(ratio > 1+ε)) · min(ratio, 1+ε) · A
        keep = (ratio > 1 + hi).detach().float()
        per_token_loss = -(keep * torch.clamp(ratio, max=1 + hi) * adv - kl_term)
        loss = (per_token_loss * mask).sum() / mask.sum()

    elif algo == "gspo":
        assert norm == "seq_mean"
        # sequence 级比率：s_i = exp(mean_t log_ratio)，clip 也作用在 s 上
        seq_len = mask.sum(dim=1).clamp(min=1)                       # (B,)
        s = torch.exp((policy_logps - gen_logps).mul(mask).sum(dim=1) / seq_len)
        adv_seq = (adv * mask).sum(dim=1) / seq_len if adv.dim() == 2 \
            else adv.squeeze(1)
        obj = torch.min(s * adv_seq, torch.clamp(s, 1 - lo, 1 + hi) * adv_seq)
        loss = -obj.mean()
        if isinstance(kl_term, torch.Tensor):  # KL 回到 token 级按序列平均
            loss = loss + (kl_term * mask).sum(dim=1).div(mask.sum(dim=1)).mean()

    else:
        raise KeyError(f"未知算法 {algo!r}")

    return _finalize(loss, None, ratio, mask)


ALGOS = ("grpo", "dapo", "dr_grpo", "cispo", "gspo", "rfpp")
