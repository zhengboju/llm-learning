# -*- coding: utf-8 -*-
"""rlab/passrate.py — 训练期 per-question 通过率离线聚合（开发项 4，纯观测）。

【背景】record.jsonl 自 2026-09-23 起每行带 qk（题目指纹，sha1(Q)[:12]，与
eval_vllm_one.py 的 items.qk 同一算法）——训练产生的轨迹按题聚合后就是
"当前模型×当前提示×当前预算"下的在线通过率分布。它回答三个此前无法回答的
问题：
  1. band 内题占比在训练中是否漂移（模型变强 → p 上升 → band 内题越来越少）？
  2. 被 --difficulty_band 滤掉的 p≈0 题里，有多少其实已经能做对（表过期）？
  3. 被滤掉的 p≈1 题有多少（band 太宽、简单题浪费算力）？

【为什么是观测不是控制】先把"band 是否在漂移"变成可见数字，再决定要不要
做自动重探（开发项 2 的完整形态）/调 band/调 q_pool_reset_floor——不做
先斩后奏的自动控制。

用法（离线，读 record 不占 GPU）：
    python -m rlab.passrate --record rlab_out/record.jsonl \
        [--difficulty_path rlab_out/difficulty_probe_4b_v5.jsonl \
         --difficulty_band 0.25 0.75]        # 给了就输出表过期诊断
    # 可选：把在线通过率回填进难度表副本（--merge_out），生成"在线增强表"
    # 供下一段训练 --difficulty_path 使用（回写的是新文件，绝不动原表）
    python -m rlab.passrate --record ... --difficulty_path ... \
        --merge_out rlab_out/difficulty_probe_online.jsonl

【qk 冲突说明】指纹只有 12 hex（48bit），17k 题碰撞概率 ~1e-8，工程上安全；
但回填难度表时按 qk 匹配、Q 全文以**表内为准**（表行有 Q 原文，record 只有
指纹——冲突时宁可丢题不可错配）。
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def qk_of(Q: str) -> str:
    """题目指纹（与 rollout.collect_retool_group / eval_vllm_one.items 三点同源）。"""
    return hashlib.sha1(str(Q).encode("utf-8")).hexdigest()[:12]


def aggregate_by_qk(record_path: str):
    """读 record.jsonl → {qk: {"k": int, "n_correct": int, "n_rows": int}}。

    一行 = 一题的 num_pre_Q 条轨迹（acc 数组是 ±1/0 值）；同题多行跨训练时长
    全部累计（k 无上限——在线观测就是越多越准）。坏行/缺 acc/缺 qk 静默跳过
    （旧 record 没有 qk 字段，聚合结果为空是正常输出而非错误）。"""
    by_qk = {}
    with open(record_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            accs = r.get("acc")
            key = r.get("qk")
            if not isinstance(accs, list) or not accs or not key:
                continue
            d = by_qk.setdefault(key, {"k": 0, "n_correct": 0, "n_rows": 0})
            d["k"] += len(accs)
            d["n_correct"] += sum(1 for a in accs if a > 0)
            d["n_rows"] += 1
    return by_qk


def band_stats(by_qk: dict, lo: float, hi: float):
    """纯函数：在线通过率分布按 band 切片（诊断 band 漂移的主数字）。"""
    stats = {"n_questions": 0, "in_band": 0, "p_zero": 0, "p_one": 0,
             "band_out": 0, "total_rows": 0}
    for d in by_qk.values():
        if d["k"] <= 0:
            continue
        stats["n_questions"] += 1
        stats["total_rows"] += d["n_rows"]
        rate = d["n_correct"] / d["k"]
        if rate <= lo:
            stats["p_zero"] += 1
        elif rate >= hi:
            stats["p_one"] += 1
        elif lo < rate < hi:
            stats["in_band"] += 1
        else:
            stats["band_out"] += 1
    return stats


def merge_into_table(by_qk: dict, table_path: str, out_path: str,
                     min_k: int = 16):
    """把在线通过率回填进难度表副本（绝不动原表）。

    规则：在线 k>=min_k 的题覆盖表行的 k/n_correct/pass_rate（在线观测样本多、
    更新鲜，且正是"当前模型"的通过率）；在线没见过的题保留表行原值；record 里
    有指纹但表里匹配不上的题丢弃（宁缺勿错配）。新表带 online_meta 自证出处。"""
    rows, matched, kept_online = [], 0, 0
    if os.path.exists(table_path):
        with open(table_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get("Q") or not r.get("k"):
                    continue
                key = qk_of(r["Q"])
                on = by_qk.get(key)
                if on and on["k"] >= min_k:
                    r = dict(r)
                    r["k"] = on["k"]
                    r["n_correct"] = on["n_correct"]
                    r["pass_rate"] = on["n_correct"] / on["k"]
                    r["online_meta"] = {"source_rows": on["n_rows"],
                                        "note": "在线回填（passrate.py），原 k/n_correct 被覆盖"}
                    matched += 1
                    kept_online += 1
                rows.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"table_rows": len(rows), "online_matched": matched,
            "min_k": min_k, "out": out_path}


def main():
    ap = argparse.ArgumentParser(description="训练期 per-question 在线通过率聚合（纯观测）")
    ap.add_argument("--record", default="./rlab_out/record.jsonl",
                    help="训练 record.jsonl 路径")
    ap.add_argument("--difficulty_path", default=None,
                    help="可选：训练用的难度表；给了就输出'表过期诊断'"
                         "（在线已能做对/仍全错的题在 band 外各有多少）")
    ap.add_argument("--difficulty_band", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="与训练同 band（如 0.25 0.75）；--difficulty_path 给了但这个没给则报错")
    ap.add_argument("--merge_out", default=None,
                    help="可选：把在线通过率回填进难度表副本并写到该路径"
                         "（生成'在线增强表'供下一段训练 --difficulty_path 用）")
    ap.add_argument("--min_k", type=int, default=16,
                    help="回填门槛：在线样本 k>=min_k 才覆盖表行（默认 16，"
                         "低于此噪声过大）")
    args = ap.parse_args()

    if not os.path.exists(args.record):
        raise SystemExit(f"record 不存在: {args.record}")
    if args.difficulty_path and args.difficulty_band is None:
        raise SystemExit("--difficulty_path 需要配对 --difficulty_band LO HI（与训练同值）")

    by_qk = aggregate_by_qk(args.record)
    if not by_qk:
        print("聚合结果为空：record 里没有带 qk 的行（2026-09-23 之前的旧 record "
              "没有题目指纹字段；本次训练重启后新行才带）。")
        return

    lo, hi = (args.difficulty_band or (0.0, 1.0))
    s = band_stats(by_qk, lo, hi)
    print(f"\n=== 在线通过率分布（record={args.record}）===")
    print(f"  出现过的题: {s['n_questions']}  轨迹组: {s['total_rows']}")
    print(f"  band=({lo},{hi}):  band内 {s['in_band']}  p_zero {s['p_zero']}  "
          f"p_one {s['p_one']}  band_out {s['band_out']}")
    if s["n_questions"]:
        print(f"  band 内占比: {s['in_band'] / s['n_questions'] * 100:.1f}%"
              "（训练推进中重跑本命令对比该数字 = 漂移观测）")

    if args.difficulty_path:
        from rlab.data import load_difficulty_table
        tbl = load_difficulty_table(args.difficulty_path)
        # 交叉三象限：表判 p≈0 但在线已能做对（表过期→该释放）；表判 band 内但
        # 在线 p≈0（模型退化或表过于乐观）；表判 p≈1（band 太宽/简单题白占位）
        stale_releasable, table_optimistic, table_p_one = 0, 0, 0
        for q, row in tbl.items():
            key = qk_of(q)
            on = by_qk.get(key)
            if not on or on["k"] < 8:      # 在线样本太少不下结论
                continue
            table_rate = row["n_correct"] / row["k"]
            online_rate = on["n_correct"] / on["k"]
            if table_rate <= lo and online_rate > hi - (hi - lo) * 0.5 and online_rate > lo:
                stale_releasable += 1      # 表判 0，在线已非 0 → 表过期
            if lo < table_rate < hi and online_rate <= lo:
                table_optimistic += 1      # 表判可学，在线全错 → 表乐观
            if table_rate >= hi:
                table_p_one += 1
    print("\n=== 表过期诊断（表=%s，在线 k>=8 的题）===" % args.difficulty_path)
    print(f"  表判 p≈0 但在线已能做对（可释放）: {stale_releasable}")
    print(f"  表判 band 内但在线全错（表乐观/模型退化）: {table_optimistic}")
    print(f"  表判 p≈1（band 太宽，白占算力）: {table_p_one}")

    if args.merge_out:
        m = merge_into_table(by_qk, args.difficulty_path or "", args.merge_out,
                             min_k=args.min_k)
        print("=== 在线增强表已写 ===")
        print(f"  表行 {m['table_rows']} 行，其中在线通过率回填覆盖 {m['online_matched']} 行"
              f"（min_k={m['min_k']}）→ {m['out']}")
        print("  下一段训练 --difficulty_path 指向该文件即可（原表未动）")


if __name__ == "__main__":
    main()
