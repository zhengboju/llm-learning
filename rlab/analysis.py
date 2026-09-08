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
import os

NOISE_FLOOR_PP = 2.0   # 公共协议：±2pp 内视为噪声，>3pp 才算真差异


def summarize_eval(path: str, base_name: str = "BASE") -> str:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    base = results.get(base_name, {})
    lines = [f"| 模型 | acc | fmt | both | Δacc vs {base_name} | 判定 |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        acc, both = r["acc"] * 100, r["both"] * 100
        if name == base_name or not base:
            delta, verdict = "—", "—"
        else:
            d = acc - base["acc"] * 100
            delta = f"{d:+.1f}pp"
            verdict = ("真差异" if d > 3 else "噪声级" if d > NOISE_FLOOR_PP else "噪声内")
        lines.append(f"| {name} | {acc:.1f} | {r['fmt']*100:.1f} | {both:.1f} | {delta} | {verdict} |")
    return "\n".join(lines)


def summarize_record(path: str, window: int = 20, clen_cap: int = 1800) -> str:
    """按 upload 批次滑动平均 acc/fmt/code 率与完成长度（retool 诊断用）。

    clen_cap ≈ max_context_tokens(2200) - 典型 prompt(~400) = 1800：接近上限
    说明轨迹在撞上下文预算（会被整组丢弃或标签被截断）——2026-09-08 第四轮
    "格式学到 75-95% 后崩回 0"的嫌疑机制，需 clen/code 趋势佐证。"""
    accs, fmts, codes, clens = [], [], [], []
    out = ["| 批次窗口 | acc率 | fmt率 | code率 | avg_clen | ≥90%cap |",
           "|---|---|---|---|---|---|"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                accs.extend(a > 0 for a in rec.get("acc", []))
                fmts.extend(v > 0 for v in rec.get("fmt", []))
                codes.extend(u > 0 for u in rec.get("code_used", []))
                clens.extend(rec.get("clen", []))
            except json.JSONDecodeError:
                continue
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
        out.append(f"| {i}~{i + len(chunk_a)} | {sum(chunk_a) / len(chunk_a) * 100:.1f}% "
                   f"| {sum(chunk_f) / len(chunk_f) * 100:.1f}% | {code_col} "
                   f"| {len_col} |")
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
