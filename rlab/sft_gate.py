# -*- coding: utf-8 -*-
"""rlab/sft_gate.py — SFT 验收门（docs/08 §4：不过门不准进 RL，2026-10-09）。

读 eval_vllm_one.py 的评测 json（经 analysis.read_eval_result，嵌套壳/扁平壳
都吃），按四项门槛判 PASS/FAIL：

  ① code_rate      > 0.6   代码调用率（SFT 应显著抬升 base 的 ~48%）
  ② fmt            > 0.9   boxed 收尾率（"答完就停"是否装上）
  ③ code_ok_rate/code_rate > 0.7   代码质量
  ④ acc            ≥ base  基础能力不回退（base 由 --base_eval 同档评测给出）

门槛是工程判断不是论文规定值（docs/08 §4），可用 CLI 覆盖；但**校准必须在
进 RL 之前完成**，不能边跑边放宽。

用法：
  python -m rlab.sft_gate --eval rlab_out/eval_sft.json \
      --base_eval rlab_out/eval_base.json
exit code：全过 0，任何一项不过 1（可接 shell && 链进 RL 命令）。
"""
import argparse
import sys

from rlab.analysis import read_eval_result


def check_gate(result: dict, base_acc: float, *, code_rate=0.6, fmt=0.9,
               code_ok_ratio=0.7):
    """四项门槛逐条判（纯函数，CPU 可测）。返回 (rows, all_pass)。"""
    cr = float(result.get("code_rate", 0.0) or 0.0)
    ck = float(result.get("code_ok_rate", 0.0) or 0.0)
    fm = float(result.get("fmt", 0.0) or 0.0)
    ac = float(result.get("acc", 0.0) or 0.0)
    ratio = ck / cr if cr > 0 else 0.0
    rows = [
        ("code_rate", cr, f"> {code_rate}", cr > code_rate),
        ("fmt", fm, f"> {fmt}", fm > fmt),
        ("code_ok/code_used", ratio, f"> {code_ok_ratio}",
         ratio > code_ok_ratio),
        ("acc", ac, f">= base({base_acc:.4f})", ac >= base_acc),
    ]
    return rows, all(r[3] for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="SFT 验收门（docs/08 §4）")
    ap.add_argument("--eval", required=True, help="SFT ckpt 的评测 json")
    ap.add_argument("--base_eval", required=True,
                    help="BASE 的评测 json（同档同 seed 评出，否则 acc 门无意义）")
    ap.add_argument("--code_rate", type=float, default=0.6)
    ap.add_argument("--fmt", type=float, default=0.9)
    ap.add_argument("--code_ok_ratio", type=float, default=0.7)
    args = ap.parse_args(argv)

    res = read_eval_result(args.eval)
    base = read_eval_result(args.base_eval)
    if not res or not res.get("n"):
        print(f"[gate] 评测结果不可读或 n=0：{args.eval}（n=0 是读错层级的"
              "签名，docs/08 §6）", flush=True)
        return 1
    if not base or not base.get("n"):
        print(f"[gate] BASE 评测结果不可读或 n=0：{args.base_eval}", flush=True)
        return 1
    rows, ok = check_gate(res, float(base["acc"]),
                          code_rate=args.code_rate, fmt=args.fmt,
                          code_ok_ratio=args.code_ok_ratio)
    print(f"[gate] {args.eval} (n={res['n']}) vs BASE acc "
          f"{base['acc'] * 100:.1f}% (n={base['n']})")
    for name, val, thr, passed in rows:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<18} "
              f"{val * 100:6.1f}%  （门槛 {thr}）")
    verdict = ("PASS —— 可进 RL" if ok
               else "FAIL —— 回数据/配方，不准进 RL（docs/08 §4）")
    print(f"[gate] 总判：{verdict}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
