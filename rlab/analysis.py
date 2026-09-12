# -*- coding: utf-8 -*-
"""rlab/analysis.py — 结果解析与对比表。

功能：
  1. 汇总 eval_v_*.json -> Markdown 对比表（含与 BASE 的差值、与噪声地板 ±2pp 的判定）；
  2. 解析 rlab record.jsonl -> 训练中 acc/format 正确率随上传批次的曲线数据。

用法：
    python -m rlab.analysis --eval-json eval_vllm_all.json [--base BASE]
    python -m rlab.analysis --record rlab_out/record.jsonl
"""
import argparse
import glob
import json
import math
import os
import time

NOISE_FLOOR_PP = 2.0   # 【已降级】仅留作历史报告对照；判定改用 ci95()/McNemar，见下
SESS_GAP_S = 120.0     # record 时间戳间隔 >120s = 新训练会话（进程重启/新 run 追加同文件）


# ---------------------------------------------- 统计口径（2026-09-12 新增）----
# 【为什么必须改】旧版 summarize_eval 用固定 ±2pp 地板判"真差异"——那是 GSM8K/N=300
# 时代的常数。实测量级：dapo_math N=500 下单臂 95%CI 就有 ±4.4pp、两臂差 ±6.2pp，
# 于是 m200−BASE 的 +5.0pp（未配对两比例 p≈0.11）会被旧逻辑直接打成"真差异"。
# 现在：①单臂/两臂 CI 一律按 n 现算；②若两模型都落了 per-item 结果（eval 端
# --dump_items，默认开），改用**同题配对的 McNemar 精确检验**——只看分歧对，功效
# 远高于独立两臂。这正是 run2 里 +5.0pp 本可能显著、却因只存聚合值而算不出来的检验。

def ci95(p: float, n: int) -> float:
    """单臂比例 p 的 95% 置信半宽（返回 pp）。n<=0 返回 nan。"""
    if not n or n <= 0:
        return float("nan")
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n) * 100.0


def diff_ci95(p1: float, n1: int, p2: float, n2: int):
    """两臂差（pp）与 95% 半宽（未配对两比例）。返回 (diff_pp, half_pp)。"""
    if not n1 or not n2:
        return (p1 - p2) * 100.0, float("nan")
    se = math.sqrt(max(p1 * (1 - p1), 0.0) / n1 + max(p2 * (1 - p2), 0.0) / n2)
    return (p1 - p2) * 100.0, 1.96 * se * 100.0


def mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar 精确二项检验（双侧 p）。b/c = 两个方向的分歧计数。"""
    n = int(b) + int(c)
    if n <= 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_counts(items_model, items_base, key: str = "qk"):
    """按题面 key 对齐两份 per-item 结果 → (b, c, matched)。

    约定 model 为被测臂、base 为基线臂：
      b = model 对 & base 错（提升方向的 discordant pair）
      c = model 错 & base 对（回退方向）
    任一臂缺 items / 无 key / 无可对齐题时返回 None（调用方回落两比例检验）。"""
    if not items_model or not items_base:
        return None
    bm = {it.get(key): it for it in items_model if it.get(key)}
    bb = {it.get(key): it for it in items_base if it.get(key)}
    if not bm or not bb:
        return None
    matched = b = c = 0
    for k in bm.keys() & bb.keys():
        am, ab = bm[k].get("acc", 0) == 1.0, bb[k].get("acc", 0) == 1.0
        matched += 1
        if am and not ab:
            b += 1
        elif ab and not am:
            c += 1
    return (b, c, matched) if matched else None


def _verdict(diff_pp: float, half_pp: float, p=None) -> str:
    """判定：有 McNemar p 用 p<0.05；否则看 CI 是否跨 0（不再用固定 ±2pp 地板）。"""
    if p is not None:
        return "显著" if p < 0.05 else "噪声内"
    if half_pp != half_pp:      # nan
        return "—"
    return "显著" if (diff_pp - half_pp > 0 or diff_pp + half_pp < 0) else "噪声内"


def summarize_eval(path: str, base_name: str = "BASE") -> str:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    models = {k: v for k, v in results.items() if not k.startswith("_")}
    base = models.get(base_name, {})
    n_base = base.get("n") or 0
    lines = []
    if n_base:
        hw = ci95(0.5, n_base)
        lines.append(f"> N={n_base} · 单臂 95%CI 最坏 ≈±{hw:.1f}pp · 两臂差 ≈±{hw * math.sqrt(2):.1f}pp "
                     f"· 判定 = CI 不跨 0（有 per-item 时改用 McNemar p<0.05）")
        lines.append("")
    lines += [f"| 模型 | acc%(±95%CI) | fmt% | code% | Δacc vs {base_name} | 检验 | 判定 |",
              "|---|---|---|---|---|---|---|"]
    for name, r in models.items():
        n = r.get("n") or 0
        acc = r.get("acc", 0.0) * 100
        hw = ci95(r.get("acc", 0.0), n)
        acc_col = f"{acc:.1f}±{hw:.1f}" if hw == hw else f"{acc:.1f}"
        fmt_col = f"{r.get('fmt', 0.0) * 100:.1f}"
        code_col = f"{r['code_rate'] * 100:.1f}" if "code_rate" in r else "—"
        if name == base_name or not base or not n or not n_base:
            delta, test, verdict = "—", "—", "—"
        else:
            d, h = diff_ci95(r.get("acc", 0.0), n, base.get("acc", 0.0), n_base)
            pc = paired_counts(r.get("items"), base.get("items"))
            if pc is not None:
                b, c, matched = pc
                p = mcnemar_exact(b, c)
                delta = f"{d:+.1f}pp"
                test = f"McNemar p={p:.3f} (b={b}/c={c}, n={matched})"
                verdict = _verdict(d, h, p)
            else:
                delta = f"{d:+.1f}±{h:.1f}pp"
                test = "两比例（无 per-item）"
                verdict = _verdict(d, h)
        lines.append(f"| {name} | {acc_col} | {fmt_col} | {code_col} | {delta} | {test} | {verdict} |")
    return "\n".join(lines)


def summarize_record(path: str, window: int = 20, clen_cap: int = 1800) -> str:
    """按 upload 批次滑动平均 acc/fmt/code 率与完成长度（retool 诊断用）。

    clen_cap ≈ max_context_tokens(2200) - 典型 prompt(~400) = 1800：接近上限
    说明轨迹在撞上下文预算（会被整组丢弃或标签被截断）——2026-09-08 第四轮
    "格式学到 75-95% 后崩回 0"的嫌疑机制，需 clen/code 趋势佐证。

    【会话拆分 2026-09-08】record.jsonl 以追加模式写入，多次训练（重启/新 run）
    会写进同一文件，且每次会话 pushes 计数归零 → 阶段(phase)列会来回振荡
    （pushes 单调递增，单会话内 phase 只能冷→热切一次）。因此用时间戳间隔
    >120s 切会话，逐会话聚合 stats——"崩盘点在哪个会话、各会话的冷热阶段"
    一眼可辨，避免把跨会话曲线误读成单次训练的动力学。"""
    accs, fmts, codes, clens, phases, sess_ids = [], [], [], [], [], []
    sess_span = {}   # sess -> [first_t, last_t]（墙钟，便于对 Shell 历史核对是哪次 run）
    with open(path, encoding="utf-8") as f:
        prev_t, sess = None, 0
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = len(rec.get("acc", []))
            if n == 0:
                continue
            t = rec.get("t")
            if prev_t is not None and t is not None and t - prev_t > SESS_GAP_S:
                sess += 1
            if t is not None:
                prev_t = t
                sess_span.setdefault(sess, [t, t])[1] = t
            accs.extend(a > 0 for a in rec["acc"])
            fmts.extend(v > 0 for v in rec["fmt"])
            codes.extend(u > 0 for u in rec.get("code_used", []))
            clens.extend(rec.get("clen", []))
            ph = rec.get("phase")
            if ph:
                phases.extend([ph] * n)
            sess_ids.extend([sess] * n)
    sess_lines = []
    if sess_ids:
        for s in sorted(set(sess_ids)):
            idx = [i for i, v in enumerate(sess_ids) if v == s]
            a = sum(accs[i] for i in idx) / len(idx) * 100
            ff = sum(fmts[i] for i in idx) / len(idx) * 100
            c = (sum(codes[i] for i in idx) / len(idx) * 100
                 if codes else float("nan"))
            lo, hi = idx[0], idx[-1] + 1
            span = sess_span.get(s)
            when = ""
            if span:
                fmt_t = lambda x: time.strftime("%m-%d %H:%M:%S", time.localtime(x))
                when = f" [{fmt_t(span[0])} ~ {fmt_t(span[1])}]"
            sess_lines.append(
                f"会话{s}: 样本{lo}~{hi}（{len(idx)}条 ≈{len(idx)/16:.0f}步）"
                f" acc={a:.1f}% fmt={ff:.1f}% code={c:.1f}%{when}")
    out = ["| 批次窗口 | acc率 | fmt率 | code率 | avg_clen | ≥90%cap | 阶段 | 会话 |",
           "|---|---|---|---|---|---|---|---|"]
    for i in range(0, len(accs), window):
        chunk_a, chunk_f, chunk_c = accs[i:i + window], fmts[i:i + window], codes[i:i + window]
        chunk_l = clens[i:i + window]
        if not chunk_a:
            continue
        code_col = f"{sum(chunk_c) / len(chunk_c) * 100:.1f}%" if chunk_c else "—"
        if chunk_l:
            avg_l = sum(chunk_l) / len(chunk_l)
            near = sum(1 for l in chunk_l if l >= 0.9 * clen_cap) / len(chunk_l)
            len_col = f"{avg_l:.0f} | {near * 100:.0f}%"
        else:
            len_col = "— | —"
        ph_col = phases[i] if i < len(phases) else "—"
        sess_col = "ABCDEFGH"[sess_ids[i]] if i < len(sess_ids) else "—"
        out.append(f"| {i}~{i + len(chunk_a)} | {sum(chunk_a) / len(chunk_a) * 100:.1f}% "
                   f"| {sum(chunk_f) / len(chunk_f) * 100:.1f}% | {code_col} "
                   f"| {len_col} | {ph_col} | {sess_col} |")
    if sess_lines:
        out.append("")
        out.append(f"== 会话拆分（时间戳间隔>{SESS_GAP_S:.0f}s = 新会话）==")
        out.extend(sess_lines)
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--base", default="BASE")
    args = ap.parse_args()
    if args.eval_json:
        print(summarize_eval(args.eval_json, args.base))
    if args.record:
        print(summarize_record(args.record))
    if not args.eval_json and not args.record:
        cands = sorted(glob.glob("eval_vllm_all*.json"))
        if cands:
            print(summarize_eval(cands[-1], args.base))
        else:
            print(" nothing to summarize（--eval-json / --record）")
