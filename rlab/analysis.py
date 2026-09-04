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


def summarize_record(path: str, window: int = 20) -> str:
    """按 upload 批次滑动平均 acc/format 正确率。"""
    accs, fmts, out = [], [], ["| 批次窗口 | acc率 | fmt率 |", "|---|---|---|"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                accs.extend(a > 0 for a in rec.get("acc", []))
                fmts.extend(v > 0 for v in rec.get("fmt", []))
            except json.JSONDecodeError:
                continue
    for i in range(0, len(accs), window):
        chunk_a, chunk_f = accs[i:i + window], fmts[i:i + window]
        if chunk_a:
            out.append(f"| {i}~{i+len(chunk_a)} | {sum(chunk_a)/len(chunk_a)*100:.1f}% "
                       f"| {sum(chunk_f)/len(chunk_f)*100:.1f}% |")
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
