# -*- coding: utf-8 -*-
"""rlab/diag_logps.py — gen_logps 不一致的责任方定位（离线复现 + 后端消融）。

【要回答的问题（2026-09-15 实机）】对拍出现"中位差 1e-5、尾部 max 12.8"：某一处
vLLM 说被采样 token 的 logp=-0.946（p≈0.39），torch 重算说 -13.75（p≈1e-6）。
bf16/kernel 的常规数值差（logit 级 ~0.1 nat）不可能把 0.39 变成 1e-6——那需要
~12.8 nat 的 **token 级** logit 误差。所以两路在该位置的**分布实质不同**，必有一路
的 logits 是错的。本脚本把"哪一路、哪条 kernel"用可复现的方式定下来。

【设计：测量与判读分离】
  · 测量（GPU，一个 provider 一个进程）：
      --build-traj   用**训练同口径**采样参数跑一条固定轨迹（温度/top_k/top_p/seed
                     全取 preset），记录 ids 与逐位置采样 logp（= 训练真用的 gen_logps）
      --traj-in      复用这条固定轨迹 → 所有 provider 在**同一条上下文**上比较
      --providers    vllm | torch
    每个 provider 对每个 (题, 前缀长 L) 测同一个位置：给出自己的 top-K 分布，以及
    目标 token（= 轨迹里第 L 个 token，即训练时被采样的那个）的 logp 与排名。
    vLLM 侧的 lp_target 只有在目标落在自己 top-K 内才有值——**缺失本身就是证据**
    （另一侧认为它 p=0.39，这一侧把它排在 top-K 之外）。
  · 判读（CPU，--merge）：把多个 provider 的 jsonl 合起来算成对差异矩阵 + 结论。

【为什么必须跨进程消融】两条消融轴各自钉死在进程初始化时机上：
  · vLLM 的 GDN kernel 后端是**引擎构造参数**（gdn_prefill_backend），一个进程一档；
  · transformers 走 fla 还是纯 torch 回退，是**建模模块导入时的绑定**
    （--torch-path fallback 在 load 之前打桩强制回退）。
所以一次进程 = 一个 provider；消融 = 多次运行 + --merge。轨迹靠 --traj-in 固定，
避免"换进程顺便换了前缀"这种混杂（那会让消融结果无法归因）。

【判据】同一位置、同一目标 token，比较**同引擎跨 kernel** 的差 与 **跨引擎** 的差：
  · torch:fla vs torch:fallback 差很大  → torch 侧前向路径不稳（先修它）；
  · vllm:A vs vllm:B 差很大            → vLLM 侧 GDN backend 不稳；
  · 两侧各自自洽、跨引擎差很大          → kernel 口径差是本质（gen_logps 该回 torch 副本）；
  · 全都一致                            → 不是 kernel，回到序列构造/对齐去查。
本判据不能替代真机结论：**先跑，再看 verdict() 打了哪一支**。

用法（pod，建议 GPU0 跑 vLLM/GPU1 跑 torch，避免同卡共居；**不要与训练同时跑**——
两条路都要独占显存。路径与开关照抄本次训练命令：2026-09-15 起统一用一份复合 ckpt
/root/Qwen3.5-4B，torch 侧可直连（显式喂 text_config），**不再需要 -text 分裂目录**；
若训练命令里还有 --vllm_model_path/--vllm_gen_kwargs/--chat_template_kwargs，一并抄过来）：
    # ① 建轨迹（同时测 vLLM 这一档）
    CUDA_VISIBLE_DEVICES=0 python -m rlab.diag_logps --build_traj \
        --model_path /root/Qwen3.5-4B \
        --chat_template_kwargs '{"enable_thinking": false}' --providers vllm \
        --traj_out rlab_out/diag/traj.jsonl --out rlab_out/diag/vllm_preset.jsonl
    # ② vLLM 换一档 backend（消融轴 1；训练若传过 --vllm_gen_kwargs 也要补上）
    CUDA_VISIBLE_DEVICES=0 python -m rlab.diag_logps --traj_in rlab_out/diag/traj.jsonl \
        --providers vllm --vllm_backend none --out rlab_out/diag/vllm_none.jsonl
    # ③ torch 两条前向路径（消融轴 2；不需要 vLLM，可放 GPU1）
    CUDA_VISIBLE_DEVICES=1 python -m rlab.diag_logps --traj_in rlab_out/diag/traj.jsonl \
        --model_path /root/Qwen3.5-4B --providers torch --torch_path fla \
        --out rlab_out/diag/torch_fla.jsonl
    CUDA_VISIBLE_DEVICES=1 python -m rlab.diag_logps --traj_in rlab_out/diag/traj.jsonl \
        --model_path /root/Qwen3.5-4B --providers torch --torch_path fallback \
        --out rlab_out/diag/torch_fb.jsonl
    # ④ 判读（CPU，无需 GPU）
    python -m rlab.diag_logps --merge rlab_out/diag/vllm_preset.jsonl \
        rlab_out/diag/vllm_none.jsonl rlab_out/diag/torch_fla.jsonl rlab_out/diag/torch_fb.jsonl
"""
import argparse
import itertools
import json
import os
import random
import sys

# 判据阈值（都在 logp 空间，单位 nat）
BIG_NAT = 1.0          # 单点 |Δlogp| > 1 nat 算"分布实质不同"，不是舍入
CONFIDENT_HI = -2.0    # 一侧认为 p ≥ 13.5%（"很确定"）
CONFIDENT_LO = -8.0    # 另一侧认为 p ≤ 3.4e-4（"认为不可能"）
UNSTABLE_FRAC = 0.01   # >1 nat 的点占比超过 1% 算该路径不稳
DEFAULT_LENS = "0,256,512,1024,2048,3072"


# --------------------------------------------------------------------------
# 纯函数（CPU 可测；--merge 判读只依赖这些，不 import torch/vllm）
# --------------------------------------------------------------------------
def normalize_prefix_lens(spec: str, cap: int | None = None) -> list:
    """'0,256,512' → 去重升序、去负、截到 ≤cap 的前缀长度列表。纯函数。

    **恒含 0**：L=0 是上下文只有 prompt 的最干净判据——那里没有任何多轮拼接、
    工具段或 mask 结构，若连 L=0 都出现 10 nat 级分歧，责任方只能是两套前向本身。
    cap<0/None = 不限。全非法 → 返回 [0]（调用方据此仍然能测首 token 位）。"""
    out = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            continue
        if v < 0:
            continue
        if cap is not None and cap >= 0 and v > cap:
            continue
        out.add(v)
    out.add(0)
    return sorted(out)


def topk_pairs(entry, k: int) -> list:
    """vLLM 的 logprobs dict → [(token_id, logp)] 降序、截到 k。纯函数（CPU 可测）。

    vLLM 会把"被采样 token"一并塞进这个 dict（它可能不在 top-k 内）——按 logp 排序
    后截断，语义才等于 top-k。值可能是 float，也可能是带 `.logprob` 的对象/None。"""
    if not entry:
        return []
    items = []
    for tid, lp in entry.items():
        v = getattr(lp, "logprob", lp)
        if v is None:
            continue
        try:
            items.append((int(tid), float(v)))
        except (TypeError, ValueError):
            continue
    items.sort(key=lambda x: -x[1])
    return items[:max(1, int(k))]


def rank_from_pairs(pairs: list, tid: int):
    """在给定的 [(tid, logp)] 里的 1-based 排名；不在其中 → None。纯函数。"""
    for i, (t, _) in enumerate(pairs):
        if t == tid:
            return i + 1
    return None


def confident_disagreement(lpa, lpb, hi: float = CONFIDENT_HI,
                           lo: float = CONFIDENT_LO) -> bool:
    """一侧"很确定"、另一侧"认为不可能"。纯函数。

    None（目标 token 超出该侧 top-K）按"这一侧把它排在很后面"处理——这正是 g4 的
    形态：vLLM=-0.946 / torch=-13.75（或反过来 vLLM 干脆没把它排进 top-K）。"""
    if lpa is None and lpb is None:
        return False
    if lpa is None:
        return lpb >= hi
    if lpb is None:
        return lpa >= hi
    return (lpa >= hi and lpb <= lo) or (lpb >= hi and lpa <= lo)


def compare_rows(a: dict, b: dict) -> dict:
    """两个 provider 在**同一 (题, L)** 上的记录比对。纯函数。

    返回的 target_d 就是训练端 [train][口径] 消费的那个量（|Δlogp| on 被采样 token），
    这里按前缀长 L 逐点复现，所以"随 L 增长"还是"某一处尖峰"一眼可读。"""
    pa, pb = a.get("topk") or [], b.get("topk") or []
    da, db = dict(pa), dict(pb)
    top1a = pa[0][0] if pa else None
    top1b = pb[0][0] if pb else None
    common = set(da) & set(db)
    max_d = max((abs(da[t] - db[t]) for t in common), default=None)
    lpa, lpb = a.get("lp_target"), b.get("lp_target")
    tgt_d = abs(lpa - lpb) if (lpa is not None and lpb is not None) else None
    denom = min(len(pa), len(pb))
    return {"q": a.get("q"), "L": a.get("L"), "target": a.get("target"),
            "top1_match": (top1a is not None and top1a == top1b),
            "overlap": (len(common) / denom) if denom else None,
            "max_abs_d_common": max_d, "target_d": tgt_d,
            "target_missing": (lpa is None) or (lpb is None),
            "confident": confident_disagreement(lpa, lpb),
            "lp_a": lpa, "lp_b": lpb,
            "rank_a": a.get("target_rank"), "rank_b": b.get("target_rank")}


def _pct(vals: list, q: float):
    """分位数（最近秩法，不插值）。纯函数：样本少时插值会给出骗人的平滑值。"""
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def aggregate(rows: list) -> dict:
    """同一对 provider 的所有 (题, L) 比对 → 统计量。纯函数。"""
    n = len(rows)
    tds = [r["target_d"] for r in rows if r["target_d"] is not None]
    ovs = [r["overlap"] for r in rows if r["overlap"] is not None]
    cds = [r["max_abs_d_common"] for r in rows if r["max_abs_d_common"] is not None]
    per_L = {}
    for r in rows:
        per_L.setdefault(r["L"], []).append(r)
    worst = sorted([r for r in rows if r["target_d"] is not None],
                   key=lambda r: -r["target_d"])[:3]
    # 【逐位相同 = 可疑，不是好消息】消融的意义在于"换了实现、数会变"。同引擎对若
    # **一个 bit 都不差**（35 点全 0、top-1 全同、top-20 交集无差），最可能的解释不是
    # "两条 kernel 完美一致"，而是**这一档根本没换实现**（打桩没命中/两档其实同路）。
    # 真机首次实测就撞上了：torch:fla vs torch:fallback 全 0。
    identical = bool(n) and all(
        (r["target_d"] in (None, 0.0)) and r["top1_match"]
        and (r["max_abs_d_common"] in (None, 0.0)) and (r["overlap"] in (None, 1.0))
        for r in rows)
    return {
        "n": n,
        "identical": identical,
        "top1_match_rate": (sum(1 for r in rows if r["top1_match"]) / n) if n else None,
        "overlap_mean": (sum(ovs) / len(ovs)) if ovs else None,
        "max_abs_d_common": max(cds) if cds else None,
        "target_d_mean": (sum(tds) / len(tds)) if tds else None,
        "target_d_p99": _pct(tds, 0.99),
        "max_target_d": max(tds) if tds else None,
        "frac_big": (sum(1 for v in tds if v > BIG_NAT) / len(tds)) if tds else None,
        "confident_n": sum(1 for r in rows if r["confident"]),
        "target_missing_n": sum(1 for r in rows if r["target_missing"]),
        "worst": worst,
        "per_L": {L: aggregate_flat(v) for L, v in sorted(per_L.items())},
    }


def aggregate_flat(rows: list) -> dict:
    """per-L 的轻量版（不递归 per_L/worst，避免结构套娃）。纯函数。

    带 max_abs_d_common：它是**分布级**指标（两侧 top-K 交集上最大的 token 差），
    与 target_d（只看被采样那一个 token）回答的是两个问题——
      · 分布级也差 → logits/kernel 真的不同；
      · 只有 target_d 差、分布级一致 → 更像是"报数/取样位置"错了（vLLM 报 logprob
        的那条路与算分布的那条路不一致）。
    真机 2026-09-15 的 L=0 极值（torch -7.40 vs vLLM -0.60）必须靠这一对指标分辨。"""
    tds = [r["target_d"] for r in rows if r["target_d"] is not None]
    cds = [r["max_abs_d_common"] for r in rows if r["max_abs_d_common"] is not None]
    return {"n": len(rows),
            "target_d_mean": (sum(tds) / len(tds)) if tds else None,
            "max_target_d": max(tds) if tds else None,
            "max_abs_d_common": max(cds) if cds else None,
            "confident_n": sum(1 for r in rows if r["confident"]),
            "top1_match_rate": (sum(1 for r in rows if r["top1_match"]) / len(rows))
                               if rows else None}


def traj_id(rows: list) -> str:
    """轨迹指纹（内容哈希，纯函数）：**同一轨迹才能比**这条契约的机械执行者。

    真机 2026-09-15 教训：`--build_traj` 会**覆盖** traj.jsonl，于是"旧的 vLLM 结果 +
    新的 torch 结果"会带着重叠的 (q, L) 键被 merge —— 它们其实在看**不同的 token、
    不同的上下文**，而报告里只会显示一个正常的差异数字（无声的错答案）。指纹不同
    的 provider 对必须拒绝比较。"""
    import hashlib
    h = hashlib.sha1()
    for r in sorted(rows, key=lambda x: x.get("q", -1)):
        h.update(str(r.get("q")).encode())
        h.update(b"|")
        h.update(",".join(str(t) for t in (r.get("prompt_ids") or [])).encode())
        h.update(b"|")
        h.update(",".join(str(t) for t in (r.get("ids") or [])).encode())
        h.update(b";")
    return h.hexdigest()[:12]


def engine_of(provider: str) -> str:
    """'vllm:triton' → 'vllm'。纯函数。"""
    return str(provider).split(":", 1)[0]


def _unstable(st: dict) -> bool:
    """该 provider 对是否"不稳"：有 >1 nat 的点、或 confident 分歧、或 >1% 点越线。"""
    if not st or not st.get("n"):
        return False
    return bool((st.get("max_target_d") or 0) > BIG_NAT
                or (st.get("frac_big") or 0) > UNSTABLE_FRAC
                or (st.get("confident_n") or 0) > 0)


def verdict(pair_stats: dict) -> list:
    """成对差异矩阵 → 责任方结论（纯函数，可 CPU 测试）。

    pair_stats: {"a vs b": aggregate(...)}。判据优先级：先看同引擎跨 kernel（它能把
    责任钉到某一侧），再看跨引擎（两侧各自自洽时的"口径差是本质"）。"""
    same = {k: v for k, v in pair_stats.items()
            if engine_of(k.split(" vs ")[0]) == engine_of(k.split(" vs ")[1])}
    cross = {k: v for k, v in pair_stats.items() if k not in same}
    # 逐位相同的同引擎对**不算消融证据**（见 aggregate.identical 的注释：更像"没换实现"）
    _void = {k: v for k, v in same.items() if v.get("identical")}
    same = {k: v for k, v in same.items() if k not in _void}
    # 哪些引擎**做过**内部消融（同引擎两档）——"各自自洽"这句话只对它们成立
    engines_with_internal = {engine_of(k.split(" vs ")[0]) for k in same}
    bad_same = {k: v for k, v in same.items() if _unstable(v)}
    bad_cross = {k: v for k, v in cross.items() if _unstable(v)}
    out = []
    if bad_same:
        for k, v in bad_same.items():
            eng = engine_of(k.split(" vs ")[0])
            out.append(
                f"**同引擎跨 kernel 就不一致**：{k}（max|Δlogp|={v['max_target_d']:.3g}，"
                f">1nat 占 {100 * (v['frac_big'] or 0):.2f}%，confident={v['confident_n']}）"
                f" → 责任在 **{eng} 侧的前向实现**：它自己换条 kernel 就变，"
                f"说明至少有一条路的 logits 是错的。")
        if any(engine_of(k.split(" vs ")[0]) == "torch" for k in bad_same):
            out.append("torch 侧优先动作：装 `causal_conv1d`（日志已在喊回退参考实现）、"
                       "核对 fla/tilelang 后端是否真的生效（用 rlab/probe_gdn_backend.py），"
                       "再复跑本诊断直到 torch:fla vs torch:fallback 塌到噪声。")
        if any(engine_of(k.split(" vs ")[0]) == "vllm" for k in bad_same):
            out.append("vLLM 侧优先动作：换 `--vllm_backend` 的另一档（triton / none / flashinfer）"
                       "并与这一档逐位对比；若两档都远离 torch，则 vLLM 的 GDN kernel 数值不可信。")
        if bad_cross:
            out.append("跨引擎也不一致（这是被追查的那个现象本身）："
                       + "；".join(bad_cross) + "。先修上面同引擎的那一支。")
    elif bad_cross:
        _unverified = [e for e in ("vllm", "torch") if e not in engines_with_internal]
        _detail = "；".join(f"{k}（max|Δlogp|={v['max_target_d']:.3g}，"
                            f">1nat 占 {100 * (v['frac_big'] or 0):.2f}%）"
                            for k, v in bad_cross.items())
        if _unverified:
            # 现象确认，但"各自自洽"这句话还没有证据支撑——不许把它写成结论
            out.append(f"**跨引擎不一致（现象确认），但内部消融不完整**：{_detail}"
                       f" → {'、'.join(_unverified)} 侧没做内部消融，"
                       f"**还不能断言\"两侧各自自洽\"**（见下面注意项）。")
            out.append("在此之前可先止血：gen_logps 切回 torch 副本算"
                       "（`vllm_gen_logps=False`，与训练端严格同源、clip_frac≈0）——"
                       "它不依赖本诊断的最终归属。")
        else:
            out.append(f"**两侧各自自洽、跨引擎才不一致**：{_detail}"
                       " → kernel 口径差是本质，不是某一侧写错了代码。")
            out.append("该形态下的唯一严格解：gen_logps 用 torch 副本算（`vllm_gen_logps=False`，"
                       "严格同源、clip_frac≈0）；要继续用 vLLM 路，就得把这条残差当面量化"
                       "（`[train][口径]` 的 clip_frac / approx_kl 地板）并接受它污染诊断量。")
    else:
        out.append("各 provider 在这一点上一致（无 >1 nat 分歧、无 confident 分歧）"
                   " → 不是 kernel 问题，回到**序列构造/对齐**去查：多轮工具段拼接、"
                   "mask 有效位、左 pad 剥离、logprobs 与 ids 的逐位置对齐。")
    # 【不许过度解读】"各自自洽"只对**做过内部消融**的引擎成立。实践中最常见的缺口是
    # vLLM 那一档跑不起来（FlashInfer GDN 的 JIT 会 OOM-kill 宿主进程，见 docs/04），
    # 于是只剩 torch 侧有内部对——这时不能把结论写成"kernel 口径差是本质"。
    for _e in ("vllm", "torch"):
        if _e not in engines_with_internal:
            out.append(
                f"注意：本次没有 **{_e} 侧内部消融**（同一引擎的两个 kernel/路径档），"
                f"所以\"各自自洽\"对 {_e} 侧只是**未验证**——要钉死责任方还差一档"
                + ("（torch 侧最便宜：不需要 vLLM，换 --torch_path 再跑同一轨迹）。"
                   if _e == "torch" else
                   "（vLLM 侧：换 --vllm_backend 档；若 FlashInfer JIT 把宿主 RAM 打爆，"
                   "至少把\"未验证\"如实写进结论，别当成已验证）。"))
    if not same:
        out.append("提示：本次 merge 里一个同引擎对都没有——**没有消融就无法定位责任方**，"
                   "请补跑 `--torch_path fallback` 与/或另一个 `--vllm_backend`。")
    for k, v in _void.items():
        out.append(f"⚠ **消融作废嫌疑**：{k} 逐位完全相同（n={v['n']}，target|Δ| 全 0、"
                   "top-1 全同、top-20 交集无差）。两条不同实现不可能一个 bit 都不差——"
                   "更像是**这一档根本没换实现**（打桩没命中/两档同路）。该对不能当作"
                   "\"两种 kernel 一致\"的证据；先看生成端 stdout 的 `[diag] 计数器 ...`："
                   "目标实现计数为 0 = 打桩没生效，换 --torch_path 或补 vLLM 侧那一档。")
    return out


def vllm_kwargs_for_backend(base_kwargs: dict, backend: str) -> dict:
    """按 --vllm_backend 生成引擎参数（纯函数）。

    keep = 原样用 preset 的 vllm_gen_kwargs（**不含 CLI 传的那些**——训练命令若用
    `--vllm_gen_kwargs '{"gdn_prefill_backend": "triton"}'` 传过，本脚本要用
    同名 flag 补上，否则探的不是训练那一档）；
    none = 去掉 gdn_prefill_backend（走 vLLM 默认 = FlashInfer GDN，**会 JIT 编译**，
    宿主 RAM 紧时直接 OOM-kill，见 docs/04）；
    其它 = 设成该值（triton 免 JIT，最稳）。"""
    kw = dict(base_kwargs or {})
    if backend in (None, "keep"):
        return kw
    if backend == "none":
        kw.pop("gdn_prefill_backend", None)
        return kw
    kw["gdn_prefill_backend"] = str(backend)
    return kw


# --------------------------------------------------------------------------
# torch 侧：GDN 前向路径打桩（必须在 load 之前）
# --------------------------------------------------------------------------
def _qwen_gdn_modules() -> list:
    """找出承载 Qwen3.5 GDN 前向的建模模块名（不硬编码单一版本路径）。"""
    import importlib
    names = []
    for path in ("transformers.models.qwen3_5.modeling_qwen3_5",
                 "transformers.models.qwen3_5_text.modeling_qwen3_5_text",
                 "transformers.models.qwen3_next.modeling_qwen3_next"):
        try:
            importlib.import_module(path)
            names.append(path)
        except Exception:
            continue
    for mod_name in list(sys.modules):
        if ("qwen3_5" in mod_name or "qwen3_next" in mod_name) \
                and "modeling" in mod_name and mod_name not in names:
            names.append(mod_name)
    return names


def force_torch_gdn_fallback(*, enable: bool = True) -> dict:
    """强制 transformers 走纯 torch 的 GDN 回退实现（enable=True 时），返回打桩报告。

    **必须在 load_causal_lm 之前调用**：有的版本在层 __init__ 里就把实现定死
    （此时 load 之后再打桩无效），有的版本在 forward 里现查 —— 提前打桩两种都覆盖。

    【2026-09-15 观测与干预必须分离】计数器**无条件安装**（它只是包一层调用计数，
    不改任何行为），`enable` 只控制"把 is_fla_available 打桩成 False"这一干预。
    旧版在 enable=False 时整段 early-return，于是 `--torch_path fla` 档根本没装计数器，
    标签自证只能报 `unknown`（fla 计数 0、参考实现计数 0）——**"没观测"被读成了
    "没跑实现"**，同一类假绿灯。现在任何一档都有计数，标签才有事实依据。

    打点目标覆盖两种 import 形态：建模模块自身的绑定（`from fla... import X`）与
    fla 包内的算子（函数体内 `import`）——**打桩/计数要打在真实开关上**，上一轮就是
    打在一个不存在的名字上（patched=[]）而白忙一场。

    打桩目标（enable=True）找不到 → raise：判据必须能分辨"打了桩"和"没找到地方打"。
    """
    report = {"enabled": bool(enable), "patched": [], "counters": {},
              "errors": [], "fla_module": None}
    if not enable:
        pass    # 仍然继续装计数器（见 docstring）
    mods = []
    try:
        from transformers.utils import import_utils as _iu
        mods.append(_iu)
    except Exception as e:
        report["errors"].append(f"transformers.utils.import_utils: {type(e).__name__}: {e}")
    for name in _qwen_gdn_modules():
        try:
            mods.append(sys.modules[name])
        except KeyError:
            continue
    # fla 包内的算子：transformers 若在函数体内 import，只有打在这里才拦得到
    try:
        import importlib
        fla_mod = importlib.import_module("fla.ops.gated_delta_rule")
        mods.append(fla_mod)
        report["fla_module"] = getattr(fla_mod, "__name__", "fla.ops.gated_delta_rule")
    except Exception:
        pass    # 没装 fla 是常态（本档就是要在没 fla 时证明走的是参考实现）
    if not mods:
        raise RuntimeError("[diag] 找不到任何 Qwen3.5/3-Next 建模模块——"
                           "本档的计数器装不上，标签只能是 unknown（别当有效档解读）")
    for mod in mods:
        nm = getattr(mod, "__name__", repr(mod))
        if enable and hasattr(mod, "is_fla_available"):
            if not getattr(mod, "_rlab_gdn_patched", False):
                mod._rlab_gdn_orig_is_fla = mod.is_fla_available
            mod.is_fla_available = lambda *a, **kw: False
            mod._rlab_gdn_patched = True
            report["patched"].append(f"{nm}.is_fla_available")
        # 计数器：证"哪条实现真的被走到"，而不是只证"我打了桩"
        for attr in ("torch_chunk_gated_delta_rule", "chunk_gated_delta_rule",
                     "fused_recurrent_gated_delta_rule"):
            fn = getattr(mod, attr, None)
            if callable(fn) and not getattr(fn, "_rlab_counted", False):
                box = {"n": 0}

                def _wrapped(*a, __fn=fn, __box=box, **kw):
                    __box["n"] += 1
                    return __fn(*a, **kw)

                _wrapped._rlab_counted = True
                _wrapped._rlab_orig = fn
                setattr(mod, attr, _wrapped)
                report["counters"][f"{nm}.{attr}"] = box
    if enable and not report["patched"] and not report["counters"]:
        raise RuntimeError("[diag] 建模模块里既没有 is_fla_available 也没有 GDN 算子名——"
                           "打桩无从下手，本档数据不可信")
    return report


def report_counters(report: dict):
    """打印计数器当前值（跑完 provider 后调用）。"""
    for k, box in (report.get("counters") or {}).items():
        print(f"[diag]   计数器 {k} = {box['n']}")


# --------------------------------------------------------------------------
# 轨迹（固定上下文；训练同口径采样）
# --------------------------------------------------------------------------
def _build_prompts(cfg, args):
    from transformers import AutoTokenizer

    from rlab.data import load_dapo_math_dev, load_qas
    from rlab.rollout import build_prompt

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    if cfg["data_task"] == "dapo_math" and args.split == "dev":
        QAs = load_dapo_math_dev()          # 探针不需要训练分布，dev 更干净
    else:
        QAs = load_qas(cfg["data_task"])
    rng = random.Random(args.seed)
    picked = rng.sample(QAs, min(max(1, args.n), len(QAs)))
    out = []
    for i, qa in enumerate(picked):
        text = build_prompt(qa["Q"], cfg["system_prompt"], tokenizer,
                            cfg.get("chat_template_kwargs"))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > int(cfg["max_prompt_length"]):
            print(f"[diag] 跳过第 {i} 题：prompt {len(ids)} > max_prompt_length")
            continue
        out.append({"q": i, "question": qa["Q"][:120], "prompt_ids": ids})
    if not out:
        raise RuntimeError("[diag] 没有可用 prompt（全被 max_prompt_length 拦了）")
    print(f"[diag] prompt {len(out)} 题（split={args.split}），"
          f"plen={min(len(r['prompt_ids']) for r in out)}~"
          f"{max(len(r['prompt_ids']) for r in out)} token")
    return out


def build_traj_vllm(llm, SamplingParams, cfg, rows, max_tokens):
    """用**训练同口径**采样参数跑一条固定轨迹，逐位置记录采样 logp（= 训练用 gen_logps）。

    【必须与 rollout.make_retool_sps 逐字同源，含 logprobs_mode】真机 2026-09-15 实测：
    同一 (prompt, 位置, token)，"logprobs=0 且不设 logprobs_mode"（本条旧写法）与
    "logprobs=20 + raw_logprobs"（探针）给出的 logp 可差 1.12 nat——**vLLM 自己的
    口径就差这么多**。训练端 make_retool_sps 是显式设了 raw_logprobs 的，所以轨迹
    构建也必须显式设，否则 decode 口径拿到的 vLLM 侧根本不是训练那个量。"""
    from rlab.rollout import sampled_logps_from_output

    kw = dict(n=1, temperature=cfg["temperature"], top_p=cfg["top_p"],
              top_k=cfg.get("top_k", 50), max_tokens=max_tokens,
              logprobs=0, seed=cfg.get("seed"))
    if _sp_logprobs_mode_supported(SamplingParams):
        kw["logprobs_mode"] = "raw_logprobs"     # 与 make_retool_sps 同源
    sp = SamplingParams(**kw)
    for row in rows:
        out = llm.generate([{"prompt_token_ids": row["prompt_ids"]}], sp, use_tqdm=False)[0]
        co = out.outputs[0]
        ids = list(co.token_ids)
        row["ids"] = ids
        row["logps"] = sampled_logps_from_output(co, ids)   # fail-fast 与训练同源
        row["plen"] = len(row["prompt_ids"])
        row["finish"] = getattr(co, "finish_reason", None)
        print(f"[diag] 轨迹 q={row['q']}: {len(ids)} token finish={row['finish']}"
              f" sampler(temperature={cfg['temperature']}, top_k={cfg.get('top_k', 50)},"
              f" top_p={cfg['top_p']})")
    return rows


def _read_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[diag] 写出 {len(rows)} 行 → {path}")


# --------------------------------------------------------------------------
# provider 实现
# --------------------------------------------------------------------------
def _sp_logprobs_mode_supported(SamplingParams) -> bool:
    """该 vLLM 版本的 SamplingParams 有没有 logprobs_mode 字段（老版本硬传会 TypeError）。"""
    fields = set(getattr(SamplingParams, "__struct_fields__", ()) or ()) | \
        set(getattr(SamplingParams, "__dataclass_fields__", {}) or ())
    return "logprobs_mode" in fields


def _probe_sp(SamplingParams, k: int, seed, mode: str):
    """探查请求的 SamplingParams。mode='raw' → 显式 raw_logprobs；
    mode='default' → **不传** logprobs_mode（走该 vLLM 版本的默认口径）。

    为什么要能发 default 档：真机上"logprobs=0 不设 mode"（轨迹构建）与
    "logprobs=20 + raw"（探针）对同一 (位置, token) 差到 1.12 nat——先把**口径**
    这一项单独量出来，才能判断跨引擎的 Δ 里有多少是测量差异、多少是模型差异。
    温度/截断都设成恒等（1.0/1.0/-1），让 raw 与 processed 在数学上一致，
    剩下的差就只能是实现口径。"""
    kw = dict(n=1, max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1,
              logprobs=max(1, int(k)), seed=seed)
    if mode == "raw" and _sp_logprobs_mode_supported(SamplingParams):
        kw["logprobs_mode"] = "raw_logprobs"
    return SamplingParams(**kw)


def _raw_topk_sp(SamplingParams, k: int, seed):
    """兼容旧签名：默认 raw 档。"""
    return _probe_sp(SamplingParams, k, seed, "raw")


def run_vllm(cfg, args, rows, lens):
    from vllm import LLM, SamplingParams

    from rlab.rollout import _check_vllm_gen_kwargs, _vllm_config_readback

    kw = vllm_kwargs_for_backend(cfg.get("vllm_gen_kwargs") or {}, args.vllm_backend)
    _check_vllm_gen_kwargs(kw)      # 键名错 = 静默忽略（与 gen_worker 同一闸门）
    model_path = cfg.get("vllm_model_path") or cfg["model_path"]
    print(f"[diag] vLLM provider: model={model_path} kwargs={kw or '{}'}")
    llm = LLM(model=model_path, gpu_memory_utilization=float(cfg.get("gen_gpu_mem", 0.45)),
              **kw)
    for k_ in sorted(kw):
        v, hints = _vllm_config_readback(llm, k_)
        print(f"[diag]   回读 {k_} = {v!r}"
              f"{'' if v is not None else f'｜相似键名 {hints}'}")

    tag = f"vllm:{args.vllm_backend}"
    # 口径混读护栏：探针读的是"temperature=1/无截断"的分布，而训练时 logprobs=0 拿到
    # 的是 raw logprob。若该版本没有 logprobs_mode 字段（返回后处理值）且 preset 开了
    # 采样后处理，两者就不是同一个量——先说清，别让本档数据被当成"训练口径的复现"。
    _post = (float(cfg["temperature"]) != 1.0
             or cfg.get("top_k", -1) not in (-1, None)
             or float(cfg.get("top_p", 1.0)) != 1.0)
    if _post and not _sp_logprobs_mode_supported(SamplingParams):
        print(f"[diag][警告] preset 有采样后处理（temperature={cfg['temperature']} "
              f"top_k={cfg.get('top_k')} top_p={cfg.get('top_p')}）且该 vLLM 无 "
              "logprobs_mode 字段：探针读的是 temperature=1/无截断 的分布，"
              "与训练时 logprobs=0 的口径**可能不同**——本档只用来定位'分布差在哪'，"
              "训练口径的地板仍以 --verify_gen_logps 为准", flush=True)
    if args.build_traj:
        max_tokens = args.max_traj_tokens or max(lens)
        print(f"[diag] 建轨迹：max_tokens={max_tokens}")
        rows = build_traj_vllm(llm, SamplingParams, cfg, rows, max_tokens)
        for _r in rows:                 # 轨迹出自哪一档 vLLM（decode 口径判读要用）
            _r["vllm_tag"] = tag
        if args.traj_out:
            _write_jsonl(args.traj_out, rows)
    else:
        rows = [r for r in rows if r.get("ids")]

    sp_raw = _probe_sp(SamplingParams, args.k, cfg.get("seed"), "raw")
    _tid = traj_id(rows)
    print(f"[diag] 轨迹指纹 traj_id={_tid}（merge 靠它拒绝跨轨迹比较）", flush=True)
    sp_alt = (_probe_sp(SamplingParams, args.k, cfg.get("seed"), "default")
              if args.probe_default else None)
    if sp_alt is not None:
        print("[diag] 同时测 default 口径（不传 logprobs_mode）——"
              "用于把'vLLM 自身口径差'从跨引擎 Δ 里分出来", flush=True)
    out_rows = []
    for row in rows:
        for L in lens:
            if L >= len(row["ids"]):
                continue
            ctx = list(row["prompt_ids"]) + list(row["ids"][:L])
            res = llm.generate([{"prompt_token_ids": ctx}], sp_raw, use_tqdm=False)[0]
            entry = res.outputs[0].logprobs[0]
            pairs = topk_pairs(entry, args.k)
            target = int(row["ids"][L])
            lp = dict(pairs).get(target)
            rec = {"provider": tag, "q": row["q"], "L": L, "traj_id": _tid,
                   "plen": len(row["prompt_ids"]), "target": target,
                   "lp_target": lp, "target_rank": rank_from_pairs(pairs, target),
                   "top1": pairs[0][0] if pairs else None,
                   "topk": [[t, round(v, 6)] for t, v in pairs]}
            if sp_alt is not None:
                res2 = llm.generate([{"prompt_token_ids": ctx}], sp_alt, use_tqdm=False)[0]
                p2 = topk_pairs(res2.outputs[0].logprobs[0], args.k)
                rec["lp_target_default"] = dict(p2).get(target)
                rec["top1_default"] = p2[0][0] if p2 else None
            out_rows.append(rec)
            if lp is not None and lp >= CONFIDENT_HI:
                print(f"[diag]   q={row['q']} L={L}: vLLM 认为目标 p={pow(2.718281828, lp):.3g}"
                      f"（logp={lp:.3f}，rank={rank_from_pairs(pairs, target)}）")
            elif lp is None:
                print(f"[diag]   q={row['q']} L={L}: 目标**不在 vLLM top-{args.k}** 内")
    _print_internal_mode_delta(out_rows)
    return tag, out_rows


def internal_mode_delta(rows: list) -> dict:
    """同一位置同一 token，raw 口径 vs default 口径的 |Δlogp| 统计。纯函数。

    这是**不涉及 torch** 的量：它只问"vLLM 自己两次报数一致吗"。真机实测有 1.12 nat
    的点——若这个量不可忽略，那么跨引擎 Δ 里就混着测量口径差，不能全算到模型头上。"""
    ds = [abs(r["lp_target"] - r["lp_target_default"]) for r in rows
          if r.get("lp_target") is not None and r.get("lp_target_default") is not None]
    top1_flip = sum(1 for r in rows
                    if r.get("top1") is not None and r.get("top1_default") is not None
                    and r["top1"] != r["top1_default"])
    if not ds:
        return {"n": 0}
    return {"n": len(ds), "mean": sum(ds) / len(ds), "max": max(ds),
            "frac_gt_1": sum(1 for d in ds if d > BIG_NAT) / len(ds),
            "top1_flip": top1_flip}


def _print_internal_mode_delta(rows: list):
    st = internal_mode_delta(rows)
    if not st.get("n"):
        return
    print(f"[diag] vLLM 自身口径差（raw vs default，同一位置同一 token）：n={st['n']} "
          f"mean|Δ|={st['mean']:.3g} max|Δ|={st['max']:.3g} "
          f">1nat={100 * st['frac_gt_1']:.2f}% top-1 翻转={st['top1_flip']}", flush=True)
    print("[diag] 判读：这个量若与跨引擎 Δ 同量级，则先把口径钉死（统一 logprobs_mode、"
          "统一 logprobs 参数形态）再谈'谁算错了'", flush=True)


def torch_impl_tag(requested: str, report: dict) -> tuple:
    """本档**实际**跑的是哪个实现 → (tag, 请求是否达成, 说明)。纯函数（CPU 可测）。

    【真机事故：标签会说谎】2026-09-15 实测 `--torch_path fla` 与 `--torch_path fallback`
    逐位相同；计数器显示两次跑的都是 `torch_chunk_gated_delta_rule` 840 次、fla 计数 0
    ——原因是打桩目标（`is_fla_available`）在本版建模模块里根本不存在（`patched=[]`），
    而 transformers 的 GDN 快路径还被 `causal_conv1d` 缺失挡着，于是"fla 档"其实也是
    纯 torch 参考实现。标签必须反映事实，否则 merge 会拿两个同源数据当两个档比。"""
    counters = report.get("counters") or {}
    def _hits(sel):
        return sum(box.get("n", 0) for k, box in counters.items() if sel(k))
    fb = _hits(lambda k: "torch_chunk_gated_delta_rule" in k)
    fla = _hits(lambda k: "torch_chunk_gated_delta_rule" not in k)
    impl = "fla" if fla > 0 else ("torch_ref" if fb > 0 else "unknown")
    want = {"fallback": "torch_ref"}.get(requested, requested)   # 请求名 → 实现名
    ok = (impl == want)
    why = (f"实际实现={impl}（fla 计数 {fla}、torch 参考实现计数 {fb}，"
           f"patched={report.get('patched')}）")
    return (f"torch:{impl}", ok, why)


def run_torch(cfg, args, rows, lens):
    import torch

    from rlab.model_loading import load_causal_lm
    from rlab.rollout import _assert_torch_replica_loadable

    _assert_torch_replica_loadable(cfg["model_path"])      # A1：复合 ckpt 进不了 torch
    rep = force_torch_gdn_fallback(enable=(args.torch_path == "fallback"))
    if args.torch_path == "fallback":
        print(f"[diag] torch GDN 打桩：patched={rep['patched']} counters={list(rep['counters'])}")
        if rep["errors"]:
            print(f"[diag]   打桩期间的非致命错误：{rep['errors']}")
    dev = f"cuda:{args.torch_device}" if args.torch_device is not None else "cuda"
    model = load_causal_lm(cfg["model_path"], dtype=torch.bfloat16,
                           attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model = model.to(dev).eval()
    print(f"[diag] torch provider: model={cfg['model_path']} device={dev} "
          f"path={args.torch_path} attn={cfg.get('attn_implementation', 'sdpa')}")
    print(f"[diag] 可用打点：{list(rep['counters']) or '（一个都没装上）'}；"
          f"fla 包={rep.get('fla_module')}；patched={rep['patched']}", flush=True)

    tag = f"torch:{args.torch_path}"
    _tid = traj_id(rows)
    print(f"[diag] 轨迹指纹 traj_id={_tid}（merge 靠它拒绝跨轨迹比较）", flush=True)
    out_rows = []
    for row in rows:
        for L in lens:
            if L >= len(row["ids"]):
                continue
            ctx = list(row["prompt_ids"]) + list(row["ids"][:L])
            with torch.inference_mode():
                t = torch.tensor([ctx], device=dev)
                # 与 losses.forward_per_token_logps 同源的两个取用点：
                # base_model 出 hidden states、get_output_embeddings 出 lm_head
                h = model.base_model(t).last_hidden_state[:, -1:, :]
                lp_row = model.get_output_embeddings()(h)[0, 0].float().log_softmax(-1)
            vals, idx = lp_row.topk(min(args.k, lp_row.numel()))
            pairs = [[int(i), float(v)] for v, i in zip(vals.tolist(), idx.tolist())]
            target = int(row["ids"][L])
            lp = float(lp_row[target])
            rank = int((lp_row > lp_row[target]).sum().item()) + 1
            out_rows.append({"provider": tag, "q": row["q"], "L": L, "traj_id": _tid,
                             "plen": len(row["prompt_ids"]), "target": target,
                             "lp_target": lp, "target_rank": rank,
                             "top1": pairs[0][0] if pairs else None,
                             "topk": pairs})
            print(f"[diag]   q={row['q']} L={L}: torch 对目标 logp={lp:.3f} rank={rank}"
                  f"（vLLM 采样时的 logp={row['logps'][L]:.3f}，"
                  f"Δ={abs(row['logps'][L] - lp):.3f}）")
    report_counters(rep)
    # 【标签必须反映事实】真机事故：--torch_path fla 实际跑的是纯 torch 参考实现
    # （打桩目标 is_fla_available 在本版建模模块里不存在，且 transformers 的 GDN
    # 快路径被 causal_conv1d 缺失挡着）→ 两次运行逐位相同。若照旧把它标成 torch:fla，
    # merge 会拿两个同源数据当两个档比，得出"torch 侧自洽"的反向结论。
    tag, ok, why = torch_impl_tag(args.torch_path, rep)
    print(f"[diag] 本档实现自证：{why}")
    if not ok:
        if args.torch_path == "fallback":
            raise RuntimeError(
                f"[diag] 请求 --torch_path fallback 但**没换掉实现**：{why}\n"
                "  处置：先看 patched/counters 定位真正的路由开关（本版 GDN 快路径还被 "
                "`causal_conv1d` 缺失挡着：装上它才会走 fla）；装好再重跑本档。")
        print(f"[diag][警告] 请求 {args.torch_path} 档，实际实现是 {tag}——"
              f"本次数据按**实际实现**标注（{tag}），不要当成 {args.torch_path} 档解读。\n"
              "  想真正拿到 fla 档：pod 上 `pip install causal_conv1d`（日志一直在喊它缺失）"
              "后重跑——transformers 的 GDN 快路径依赖它。")
    # 落盘前把标签改成实际实现（out_rows 里的 provider 字段）
    for _r in out_rows:
        _r["provider"] = tag
    return tag, out_rows


# --------------------------------------------------------------------------
# 口径对齐：训练期对拍量的是**逐位置被采样 logp**（decode 路），不是"前缀末位分布"。
# 两者是不同的 vLLM kernel：prefill/chunk 路 vs 带 KV 状态的 decode 路。真机实测
# 两者形态差一个数量级（diag 的 prefill 位 mean|Δ|≈0.33 nat，训练对拍 p50≈1e-5），
# 所以"prefill 位差得多"与"训练口径地板很小"可以同时成立——必须分开量。
# --------------------------------------------------------------------------
def decode_diff_stats(vllm_lps: list, torch_lps, plen: int) -> dict:
    """逐位置 |Δlogp|（vLLM 采样值 vs torch 重算），并**按 prefill/decode 分开**。

    vllm_lps: 轨迹构建时记录的逐位置采样 logp（= 训练真用的 gen_logps）
    torch_lps: torch 在整条拼接序列上的逐位置 logp（(1, T-1)，对 ids[:,1:]）
    plen: prompt 长（轨迹 token t 的 logp 取自 torch_lps[t + plen - 1]）
    返回 {"prefill": {...}, "decode": {...}, "all": {...}}——三份都用 logps_diff_shape，
    于是与训练期对拍器的形态学指标同源（frac>1 / 最差点等可直接比）。

    **为什么要分**：位置 0 的 logits 由 prefill（chunk）路算，位置 ≥1 由 decode 路算。
    真机数据（2026-09-15）：diag 只测 prefill 位 → mean|Δ|≈0.33 nat、>1nat 14%；
    训练对拍覆盖全部位置（其中 prefill 位只占 8/24000≈0.03%）→ p50≈1e-5。两者不矛盾，
    是**两个不同的 kernel 对**。分开量才能回答"到底哪条路在错"。"""
    import torch

    from rlab.rollout import logps_diff_shape   # 与训练期对拍器同源的形态学指标

    n = min(len(vllm_lps), int(torch_lps.shape[1]) - (plen - 1))
    if n <= 0:
        return {}
    gv = torch.tensor([list(vllm_lps[:n])])
    gt = torch_lps[:, plen - 1:plen - 1 + n]
    mask_all = torch.ones(1, n)
    out = {"n": n, "all": logps_diff_shape(gv, gt, mask_all)}
    for name, idx in (("prefill", [0]), ("decode", list(range(1, n)))):
        if not idx:
            out[name] = None
            continue
        m = torch.zeros(1, n)
        m[0, idx] = 1.0
        out[name] = logps_diff_shape(gv, gt, m)
    return out


def run_torch_decode(cfg, args, rows):
    """训练口径复现：逐位置被采样 logp 对比 + prefill/decode 分离（不需要 vLLM）。"""
    import torch

    from rlab.losses import forward_per_token_logps
    from rlab.model_loading import load_causal_lm
    from rlab.rollout import _assert_torch_replica_loadable

    _assert_torch_replica_loadable(cfg["model_path"])
    torch_device = f"cuda:{args.torch_device}" if args.torch_device is not None else "cuda"
    model = load_causal_lm(cfg["model_path"], dtype=torch.bfloat16,
                           attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model = model.to(torch_device).eval()
    tags = {r.get("vllm_tag") for r in rows if r.get("vllm_tag")}
    print(f"[diag] 口径=decode（逐位置被采样 logp）；轨迹来自 {tags or '未知 backend'}；"
          f"device={torch_device}")
    agg = {}
    for row in rows:
        if not row.get("logps"):
            print(f"[diag] q={row['q']} 跳过：轨迹没带 logps（建轨迹时未开 collect_logps）")
            continue
        ids = list(row["prompt_ids"]) + list(row["ids"])
        with torch.inference_mode():
            lp = forward_per_token_logps(model, torch.tensor([ids], device=torch_device),
                                         seq_chunk=512, batch_chunk=1)
        st = decode_diff_stats(row["logps"], lp.cpu(), len(row["prompt_ids"]))
        print(f"[diag] q={row['q']} n={st['n']}："
              f"prefill mean|Δ|={_fmt(st['prefill']['mean'])} max={_fmt(st['prefill']['max'])}"
              f"（最差 v={st['prefill']['worst'][0]['vllm']:.2f}/"
              f"t={st['prefill']['worst'][0]['torch']:.2f}）｜"
              f"decode mean|Δ|={_fmt(st['decode']['mean'])} p99={_fmt(st['decode']['p99'])} "
              f"max={_fmt(st['decode']['max'])} >1nat={_pctfmt(st['decode']['frac_gt_1'])}")
        for k in ("prefill", "decode"):
            agg.setdefault(k, []).append(st[k])
    print("\n[diag] ===== 训练口径汇总（位置 0 = prefill 路，位置 ≥1 = decode 路）=====")
    for k in ("prefill", "decode"):
        if k not in agg:
            continue
        ms = [s["mean"] for s in agg[k]]
        mx = max(s["max"] for s in agg[k])
        big = sum(s["n"] * s["frac_gt_1"] for s in agg[k]) / max(1, sum(s["n"] for s in agg[k]))
        print(f"[diag] {k:>7}: 位置 {sum(s['n'] for s in agg[k])}  题均 mean|Δ|="
              f"{sum(ms) / len(ms):.3g}  全局 max|Δ|={mx:.3g}  >1nat={100 * big:.2f}%")
    print("[diag] 判读：若 prefill 远差于 decode，则责任在 vLLM 的 chunk/prefill kernel"
          "（训练期对拍的中位差被占比 0.03% 的 prefill 位掩盖）；两者都差则两路都不对。")
    return out_rows_from_decode(agg, traj_id(rows))


def out_rows_from_decode(agg: dict, tid: str = None) -> list:
    """decode 口径的汇总行（供落盘留档；不参与 provider 矩阵 merge）。"""
    out = [{"provider": "torch:decode", "measure": "decode", "traj_id": tid}]
    for k, ss in agg.items():
        n = sum(s["n"] for s in ss)
        out.append({"provider": "torch:decode", "measure": "decode", "traj_id": tid,
                    "segment": k,
                    "n": n,
                    "mean_abs_d": sum(s["mean"] * s["n"] for s in ss) / max(1, n),
                    "max_abs_d": max(s["max"] for s in ss),
                    "frac_gt_1": (sum(s["n"] * s["frac_gt_1"] for s in ss) / max(1, n))})
    return out


# --------------------------------------------------------------------------
# 判读（CPU）
# --------------------------------------------------------------------------
def merge_main(args) -> int:
    providers = {}
    for path in args.merge:
        try:
            rows = _read_jsonl(path)
        except Exception as e:
            # 喂错文件（比如把 .py 当结果）要给出可行动的错，而不是一行 JSONDecodeError
            raise SystemExit(f"[diag] 读不了 {path}（--merge 只吃 --out 产出的 jsonl）："
                             f"{type(e).__name__}: {e}")
        if not rows or "provider" not in rows[0]:
            # 目录里常混着轨迹文件（rlab_out/diag/*.jsonl 一把喂进来）——跳过并说明，
            # 不要因为它把整次 merge 拦掉（真机上就卡在这一步过）
            print(f"[diag] 跳过 {path}：无 provider 字段（轨迹文件？只喂 --out 产出的结果）")
            continue
        for row in rows:
            providers.setdefault(row["provider"], {})[(row["q"], row["L"])] = row
    if len(providers) < 2:
        print(f"[diag] 只有 {list(providers)} 一个 provider——消融至少需要两个，"
              "补跑另一档再 merge（--torch-path fallback / 另一个 --vllm-backend）")
        return 2
    print(f"[diag] merge {len(providers)} 个 provider：" + "，".join(
        f"{p}({len(v)} 点)" for p, v in sorted(providers.items())))
    # 【同一轨迹才能比】traj_id 不同 = 两边的 (q, L) 键看着一样，实际是不同 token/上下文。
    # 真机教训：--build_traj 会覆盖 traj.jsonl，旧 vLLM 结果 + 新 torch 结果混着 merge
    # 会给出一个"正常"的差异数字——无声的错答案比报错危险得多。
    tids = {p: sorted({r.get("traj_id") for r in v.values() if r.get("traj_id")})
            for p, v in providers.items()}
    for p, t in sorted(tids.items()):
        if len(t) > 1:
            print(f"[diag][警告] {p} 内部混了多个轨迹（{t}）——请只喂同一次运行的文件")
    pair_stats = {}
    for a, b in itertools.combinations(sorted(providers), 2):
        if tids[a] and tids[b] and not (set(tids[a]) & set(tids[b])):
            print(f"[diag] ⚠ 跳过 {a} vs {b}：轨迹不同（{tids[a]} vs {tids[b]}）——"
                  "同一 (q,L) 键指向的 token/上下文不同，比较无意义")
            continue
        keys = sorted(set(providers[a]) & set(providers[b]))
        if not keys:
            continue
        rows = [compare_rows(providers[a][k], providers[b][k]) for k in keys]
        st = aggregate(rows)
        pair_stats[f"{a} vs {b}"] = st
        print(f"\n[diag] {a} vs {b}: n={st['n']}  top1_match="
              f"{_fmt(st['top1_match_rate'])}  overlap@{args.k}={_fmt(st['overlap_mean'])}  "
              f"max|Δ|common={_fmt(st['max_abs_d_common'])}")
        print(f"[diag]   target|Δ|: mean={_fmt(st['target_d_mean'])} p99={_fmt(st['target_d_p99'])} "
              f"max={_fmt(st['max_target_d'])}  >{BIG_NAT}nat={_pctfmt(st['frac_big'])}  "
              f"confident={st['confident_n']}  目标缺 top-K={st['target_missing_n']}")
        print("[diag]   per-L: " + "  ".join(
            f"L={L}:mean={_fmt(v['target_d_mean'])}/max={_fmt(v['max_target_d'])}"
            f"/dist={_fmt(v.get('max_abs_d_common'))}/conf={v['confident_n']}"
            for L, v in st["per_L"].items()))
        for w in st["worst"]:
            if (w["target_d"] or 0) > BIG_NAT:
                print(f"[diag]   最差点 q={w['q']} L={w['L']} target={w['target']}: "
                      f"{a}={_fmt(w['lp_a'])}(rank {w['rank_a']}) "
                      f"{b}={_fmt(w['lp_b'])}(rank {w['rank_b']}) Δ={w['target_d']:.3f}"
                      + ("  ← confident 分歧" if w["confident"] else ""))
    print("\n[diag] ===== 结论 =====")
    for line in verdict(pair_stats):
        print("  · " + line)
    return 0


def _fmt(v):
    return "None" if v is None else f"{v:.3g}"


def _pctfmt(v):
    return "None" if v is None else f"{100 * v:.2f}%"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="gen_logps 不一致的责任方定位")
    ap.add_argument("--algo", default="retool_math", help="preset 名（取采样/预算/口径）")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="CPU 判读：合并多个 provider 的 jsonl（不加载任何模型）")
    ap.add_argument("--providers", default="vllm", choices=("vllm", "torch"))
    ap.add_argument("--model_path", default=None,
                    help="torch 侧 checkpoint（Qwen3.5 用 extract_text_model 抽出的纯文本版）")
    ap.add_argument("--vllm_model_path", default=None,
                    help="vLLM 侧 checkpoint（Qwen3.5 官方多模态版；None=与 model_path 同一份）")
    ap.add_argument("--gen_gpu_mem", type=float, default=None, help="vLLM 显存占比")
    ap.add_argument("--chat_template_kwargs", default=None, help="JSON，与训练命令一致")
    ap.add_argument("--vllm_backend", default="keep",
                    help="keep=preset 原样（对照档）；none=去掉 gdn_prefill_backend；"
                         "其它值=设成该 backend（如 triton/flashinfer）")
    ap.add_argument("--vllm_gen_kwargs", default=None,
                    help="JSON。**训练命令里有这个 flag 就必须一并带上**——它与 --vllm_backend "
                         "keep 叠加，保证'探的档'就是'训练那一档'（键名错会被 vLLM 静默忽略）")
    ap.add_argument("--torch_path", default="fla", choices=("fla", "fallback"),
                    help="fallback=打桩强制走纯 torch GDN 回退实现（消融轴）")
    ap.add_argument("--measure", default="prefill", choices=("prefill", "decode"),
                    help="prefill=前缀末位分布对拍（默认，两档消融用）；"
                         "decode=逐位置被采样 logp 对拍（**训练口径**，且按 prefill/decode "
                         "位置分开统计；只走 torch 侧，不需要 vLLM）")
    ap.add_argument("--probe_default", action="store_true",
                    help="vLLM 侧同时测 default 口径（不传 logprobs_mode）并打印"
                         "'vLLM 自身口径差'——把测量口径差从跨引擎 Δ 里分出来")
    ap.add_argument("--torch_device", type=int, default=None, help="None=默认 cuda")
    ap.add_argument("--n", type=int, default=8, help="题数")
    ap.add_argument("--k", type=int, default=20, help="top-K 深度")
    ap.add_argument("--prefix_lens", default=DEFAULT_LENS)
    ap.add_argument("--split", default="dev", choices=("dev", "train"))
    ap.add_argument("--traj_in", default=None, help="复用固定轨迹（跨 provider 可比的关键）")
    ap.add_argument("--traj_out", default=None)
    ap.add_argument("--build_traj", action="store_true")
    ap.add_argument("--max_traj_tokens", type=int, default=None)
    ap.add_argument("--out", default=None, help="本 provider 的逐点结果 jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.merge is not None:
        if not args.merge:
            ap.error("--merge 后面要跟至少一个 jsonl")
        return merge_main(args)

    from rlab.config import get_config
    cfg = get_config(args.algo, use_wandb=False)
    cfg["seed"] = args.seed
    for _k in ("model_path", "vllm_model_path", "gen_gpu_mem"):
        _v = getattr(args, _k)
        if _v is not None:
            cfg[_k] = _v
    if args.chat_template_kwargs:
        cfg["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    if args.vllm_gen_kwargs:
        cfg["vllm_gen_kwargs"] = json.loads(args.vllm_gen_kwargs)
        print(f"[diag] vllm_gen_kwargs（与训练命令对齐）: {cfg['vllm_gen_kwargs']}")
    # 探针最容易的静默错误是"探了另一份权重"——路径不存在就当场停，别让它跑完再解读
    for _k in ("model_path", "vllm_model_path"):
        _p = cfg.get(_k)
        if _p and not os.path.exists(_p):
            raise SystemExit(f"[diag] {_k}={_p} 不存在——把训练命令里的路径照抄过来"
                             "（见 docs/04 §1 的命令块），别用 preset 默认值探错模型")
    print(f"[diag] 口径: torch={cfg['model_path']} vLLM="
          f"{cfg.get('vllm_model_path') or cfg['model_path']} "
          f"chat_template_kwargs={cfg.get('chat_template_kwargs')}")
    if cfg["data_task"] != "dapo_math":
        args.split = "train"
    cap = args.max_traj_tokens or int(cfg.get("round_gen_tokens", 0) or 0) or None
    lens = normalize_prefix_lens(args.prefix_lens, cap=cap)
    print(f"[diag] 前缀长（cap={cap}）: {lens}")

    if args.traj_in:
        try:
            rows = _read_jsonl(args.traj_in)
        except Exception as e:
            raise SystemExit(f"[diag] 读不了轨迹 {args.traj_in}：{type(e).__name__}: {e}")
        if not rows or "prompt_ids" not in rows[0]:
            raise SystemExit(f"[diag] {args.traj_in} 不像轨迹文件（缺 prompt_ids）")
        print(f"[diag] 复用轨迹 {args.traj_in}：{len(rows)} 题")
    elif args.build_traj and args.providers == "vllm":
        rows = _build_prompts(cfg, args)
    else:
        raise SystemExit(
            "[diag] 要么 --build-traj（且 --providers vllm），要么 --traj-in <jsonl>：\n"
            "  跨 provider 比较必须共用同一条轨迹，否则'换进程顺便换了前缀'会让结论无法归因。")

    if args.measure == "decode":
        # 训练口径：逐位置被采样 logp 对比（vLLM 侧的值来自建轨迹时记录的 logps，
        # 所以本模式**不需要 vLLM**——轨迹是哪一档建的，判读就归属哪一档）
        if args.providers == "vllm":
            raise SystemExit("[diag] --measure decode 走的是 torch 侧重算：请用 "
                             "--providers torch（vLLM 侧的逐位置 logp 已在轨迹里）")
        if not any(r.get("logps") for r in rows):
            raise SystemExit("[diag] 轨迹里没有 logps 字段——建轨迹时必须开 collect_logps"
                             "（--build_traj 的默认行为），decode 口径才有 vLLM 侧可比")
        out_rows = run_torch_decode(cfg, args, rows)
        if args.out:
            _write_jsonl(args.out, out_rows)
        return 0

    if args.providers == "vllm":
        tag, out_rows = run_vllm(cfg, args, rows, lens)
    else:
        missing = [r["q"] for r in rows if not r.get("ids")]
        if missing:
            raise SystemExit(f"[diag] 轨迹里缺 ids（题 {missing[:5]}）——torch provider "
                             "需要 --traj-in 指向建好的轨迹")
        tag, out_rows = run_torch(cfg, args, rows, lens)
    print(f"[diag] provider={tag} 测了 {len(out_rows)} 点")
    if args.out:
        _write_jsonl(args.out, out_rows)
    print("[diag] 下一步：把各 provider 的 jsonl 一起喂 --merge 看判读")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
