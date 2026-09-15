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
    return {
        "n": n,
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
    """per-L 的轻量版（不递归 per_L/worst，避免结构套娃）。纯函数。"""
    tds = [r["target_d"] for r in rows if r["target_d"] is not None]
    return {"n": len(rows),
            "target_d_mean": (sum(tds) / len(tds)) if tds else None,
            "max_target_d": max(tds) if tds else None,
            "confident_n": sum(1 for r in rows if r["confident"]),
            "top1_match_rate": (sum(1 for r in rows if r["top1_match"]) / len(rows))
                               if rows else None}


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
    """强制 transformers 走纯 torch 的 GDN 回退实现，返回打桩报告。

    **必须在 load_causal_lm 之前调用**：有的版本在层 __init__ 里就把实现定死
    （此时 load 之后再打桩无效），有的版本在 forward 里现查 —— 提前打桩两种都覆盖。

    两条 import 形态都打：建模模块自己的绑定（`from ...utils import is_fla_available`）
    与 `transformers.utils.import_utils.is_fla_available`（模块内 `import ... as` 形态）。
    找不到任何可打的目标 → raise：**判据必须能分辨"打了桩"和"没找到地方打"**，
    否则"fallback 档跑出来的数"可能根本还是 fla 算的（静默假绿灯，本项目栽过多次）。
    """
    report = {"enabled": bool(enable), "patched": [], "counters": {}, "errors": []}
    if not enable:
        return report
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
    if not mods:
        raise RuntimeError("[diag] 找不到任何 Qwen3.5/3-Next 建模模块——"
                           "--torch-path fallback 无法打桩（别把它当 fla 档跑）")
    for mod in mods:
        nm = getattr(mod, "__name__", repr(mod))
        if hasattr(mod, "is_fla_available"):
            if not getattr(mod, "_rlab_gdn_patched", False):
                mod._rlab_gdn_orig_is_fla = mod.is_fla_available
            mod.is_fla_available = lambda *a, **kw: False
            mod._rlab_gdn_patched = True
            report["patched"].append(f"{nm}.is_fla_available")
        # 计数器：证"回退实现真的被走到"，而不是只证"我打了桩"
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
    if not report["patched"] and not report["counters"]:
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
    """用**训练同口径**采样参数跑一条固定轨迹，逐位置记录采样 logp（= 训练用 gen_logps）。"""
    from rlab.rollout import sampled_logps_from_output

    sp = SamplingParams(n=1, temperature=cfg["temperature"], top_p=cfg["top_p"],
                        top_k=cfg.get("top_k", 50), max_tokens=max_tokens,
                        logprobs=0, seed=cfg.get("seed"))
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


def _raw_topk_sp(SamplingParams, k: int, seed):
    """探查"模型原始分布"的采样参数：温度 1、无截断 + 显式 raw_logprobs。

    为什么不用训练采样参数：我们要的是**分布本身**，不是被 top_k/top_p 截断后的
    后处理分布。temperature=1/top_p=1/top_k=-1 让 raw 与 processed 两种口径重合
    ——于是即便这个 vLLM 版本没有 logprobs_mode 字段，读到的也是原始分布（不留歧义）。
    max_tokens=1：只要那个位置的一行分布，不生成。
    """
    kw = dict(n=1, max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1,
              logprobs=max(1, int(k)), seed=seed)
    if _sp_logprobs_mode_supported(SamplingParams):
        kw["logprobs_mode"] = "raw_logprobs"
    return SamplingParams(**kw)


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
        if args.traj_out:
            _write_jsonl(args.traj_out, rows)
    else:
        rows = [r for r in rows if r.get("ids")]

    sp = _raw_topk_sp(SamplingParams, args.k, cfg.get("seed"))
    out_rows = []
    for row in rows:
        for L in lens:
            if L >= len(row["ids"]):
                continue
            ctx = list(row["prompt_ids"]) + list(row["ids"][:L])
            res = llm.generate([{"prompt_token_ids": ctx}], sp, use_tqdm=False)[0]
            entry = res.outputs[0].logprobs[0]
            pairs = topk_pairs(entry, args.k)
            target = int(row["ids"][L])
            lp = dict(pairs).get(target)
            out_rows.append({"provider": tag, "q": row["q"], "L": L,
                             "plen": len(row["prompt_ids"]), "target": target,
                             "lp_target": lp, "target_rank": rank_from_pairs(pairs, target),
                             "top1": pairs[0][0] if pairs else None,
                             "topk": [[t, round(v, 6)] for t, v in pairs]})
            if lp is not None and lp >= CONFIDENT_HI:
                print(f"[diag]   q={row['q']} L={L}: vLLM 认为目标 p={pow(2.718281828, lp):.3g}"
                      f"（logp={lp:.3f}，rank={rank_from_pairs(pairs, target)}）")
            elif lp is None:
                print(f"[diag]   q={row['q']} L={L}: 目标**不在 vLLM top-{args.k}** 内")
    return tag, out_rows


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

    tag = f"torch:{args.torch_path}"
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
            out_rows.append({"provider": tag, "q": row["q"], "L": L,
                             "plen": len(row["prompt_ids"]), "target": target,
                             "lp_target": lp, "target_rank": rank,
                             "top1": pairs[0][0] if pairs else None,
                             "topk": pairs})
            print(f"[diag]   q={row['q']} L={L}: torch 对目标 logp={lp:.3f} rank={rank}"
                  f"（vLLM 采样时的 logp={row['logps'][L]:.3f}，"
                  f"Δ={abs(row['logps'][L] - lp):.3f}）")
    report_counters(rep)
    return tag, out_rows


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
            raise SystemExit(f"[diag] {path} 里没有 provider 字段——不是本脚本产出的结果文件")
        for row in rows:
            providers.setdefault(row["provider"], {})[(row["q"], row["L"])] = row
    if len(providers) < 2:
        print(f"[diag] 只有 {list(providers)} 一个 provider——消融至少需要两个，"
              "补跑另一档再 merge（--torch-path fallback / 另一个 --vllm-backend）")
        return 2
    print(f"[diag] merge {len(providers)} 个 provider：" + "，".join(
        f"{p}({len(v)} 点)" for p, v in sorted(providers.items())))
    pair_stats = {}
    for a, b in itertools.combinations(sorted(providers), 2):
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
            f"/conf={v['confident_n']}" for L, v in st["per_L"].items()))
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
