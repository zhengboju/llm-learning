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

【p11g 落地新增】给了 --difficulty_path 就打印**表刻度直方图**（精确 rate 分档 +
行加权的「有对有错」预期占比）；给了 --difficulty_band 再打印 **band 预演**
（调用训练端同一个 filter_qas_by_difficulty，预演日志里的 `N -> K` 与四个计数）。
--merge_out 时输入表与输出表的直方图都会打印——两者之差就是本段训练把池子刻度
推了多少（p11g §4.1）。注意 **`kept` 单独不具判别力**：band 收窄本身就能把它从
850 降到 499；判别力在 rate=1.0/0.875 两档被在线率重排的幅度与 online_meta 行数。

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


# 训练组大小：混合概率列的口径（与训练端 num_pre_Q 同值；p11g §6 已定版不改编组）。
NUM_PRE_Q = 8


def mixed_prob(rate: float, n: int = NUM_PRE_Q) -> float:
    """纯函数：通过率 rate、每组 n 条采样下"有对有错"（能产生正确性梯度）的概率。

    = 1 - rate^n - (1-rate)^n。这就是 p11g §1 用来把"池子刻度"翻译成"惰性组占比"
    的那个式子；全对/全错两组都不产生正确性梯度（`task_adv_guard` 丢弃）。"""
    return 1.0 - (rate ** n) - ((1.0 - rate) ** n)


def rate_histogram(rows) -> dict:
    """纯函数：难度表行 -> 按**精确通过率**分档的直方图 + 自证元信息。

    p11g §4.1 的判读（"两张表之间的差就是这轮训练把池子刻度推了多少"）需要它：
    探针表与在线回填表的 rate 分布之差 = 在线通过率重排的幅度。用精确的
    n_correct/k 而非等宽分箱——两边的 k 不同（探针 k=8、在线 k≈19），
    等宽分箱会把"刻度网格不同"和"分布真的移动了"搅成一个数。
    `online_rows` = 带 online_meta 的行数（这份表里有多少行真被在线率覆盖）。"""
    hist, k_counts, n_rows, online_rows = {}, {}, 0, 0
    for r in rows:
        k, nc = r.get("k"), r.get("n_correct")
        if not isinstance(k, int) or k <= 0 or not isinstance(nc, int) or not 0 <= nc <= k:
            continue
        rate = nc / k
        hist[rate] = hist.get(rate, 0) + 1
        k_counts[k] = k_counts.get(k, 0) + 1
        n_rows += 1
        if r.get("online_meta"):
            online_rows += 1
    return {"n_rows": n_rows, "hist": hist, "k_counts": k_counts,
            "online_rows": online_rows}


def band_preview(table: dict, lo: float, hi: float) -> dict:
    """纯函数：在表本身上预演训练端难度过滤（**调用同一个
    filter_qas_by_difficulty**，不重写分支）。

    为什么必须复用下游函数：预演与真机日志 `[rollout] 难度过滤 band=(lo,hi): N -> K`
    打印的是同一段逻辑的同一批计数——自己再写一遍分支，两边一旦漂移，"预演说 499、
    真机说 700"就没人说得清是表变了还是代码分叉了。候选池 = 表内全部 Q 行
    （训练池里不在表内的题按 `missing` 丢弃，与训练端同口径）。"""
    from rlab.data import filter_qas_by_difficulty
    qas = [{"Q": q} for q in table]
    kept_rows_q, st = filter_qas_by_difficulty(qas, table, lo=lo, hi=hi)
    kept_rows = [table[str(x["Q"])] for x in kept_rows_q]
    n = len(kept_rows)
    exp_mixed = (sum(mixed_prob(r["n_correct"] / r["k"]) for r in kept_rows) / n
                 if n else 0.0)
    return {**st, "expected_mixed": exp_mixed}


def format_hist(title: str, h: dict, n: int = NUM_PRE_Q) -> str:
    """纯函数：rate_histogram 结果 -> 文本（含行加权的"有对有错"预期占比）。"""
    ks = "{" + ", ".join(f"{k}: {v}" for k, v in sorted(h["k_counts"].items())) + "}"
    L = [f"=== 难度表刻度直方图（{title}）===",
         f"  行数 {h['n_rows']}（k 分布 {ks}）；带 online_meta 的在线回填行 "
         f"{h['online_rows']}"]
    if not h["n_rows"]:
        return "\n".join(L)
    L.append(f"  rate       行数    占比    n={n} 有对有错")
    for rate in sorted(h["hist"], reverse=True):
        c = h["hist"][rate]
        L.append(f"  {rate:.3f}   {c:6d}  {100.0 * c / h['n_rows']:5.1f}%   "
                 f"{mixed_prob(rate, n):.2f}")
    exp_all = sum(c * mixed_prob(r, n) for r, c in h["hist"].items()) / h["n_rows"]
    p_one = sum(c for r, c in h["hist"].items() if r >= 1.0) / h["n_rows"]
    L.append(f"  行加权「有对有错」预期 = {exp_all * 100:.1f}%（n={n}；"
             f"rate=1.0 的行占 {p_one * 100:.1f}%）——池子能供给正确性梯度的上限")
    return "\n".join(L)


def format_band_preview(title: str, pv: dict, lo: float, hi: float) -> str:
    """纯函数：band_preview 结果 -> 文本（与训练启动日志同一口径，可直接对账）。"""
    return (f"  band({lo},{hi}) 预演［{title}］: {pv['total']} -> {pv['kept']} 题"
            f"（全错 {pv['p_zero']} / 全对 {pv['p_one']} / 区间外 {pv['band_out']}"
            f" / 表中缺失 {pv['missing']}）；band 内行预期「有对有错」"
            f"{pv['expected_mixed'] * 100:.1f}%\n"
            f"    ⚠ 口径：全错/全对计的是**被排除**的行数（rate<=lo / rate>=hi），"
            f"不是留池里的；留池里的全对题数恒 0（band 已排除）。")


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
                    help="与训练同 band（如 0.25 0.75）；--difficulty_path 给了但这个没给则报错。"
                         "另外用于 band 预演：在表上跑训练端同一个 filter_qas_by_difficulty，"
                         "启动训练前就能知道日志会打印的 N -> K 与四个计数")
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
        # 【2026-10-20 修复·原三象限里两个结构上恒 0】
        # "表判 p≈0 但在线已能做对" 与 "表判 p≈1" 都要求该题**同时出现在 record 里**，
        # 而被 band 排除的题永远不进训练池 → 永远不会出现在 record → 计数必为 0。
        # 也就是说这两个 0 不是"表没过期"的证据，只是这个工具答不了那个问题；
        # 真机上它已经被当成过证据读（p11c 复盘）。恒 0 是系统性 bug 签名
        # （AGENTS.md），改成诚实的覆盖率报告 + 唯一可判的象限。
        # 要回答"被排除的题能不能用"，只能对该子集单独跑 probe_difficulty。
        table_optimistic = 0
        online_covered = 0
        for q, row in tbl.items():
            on = by_qk.get(qk_of(q))
            if not on or on["k"] < 8:      # 在线样本太少不下结论
                continue
            online_covered += 1
            table_rate = row["n_correct"] / row["k"]
            online_rate = on["n_correct"] / on["k"]
            if lo < table_rate < hi and online_rate <= lo:
                table_optimistic += 1      # 表判可学，在线全错 → 表乐观/模型退化
    print("\n=== 表诊断（表=%s，可判象限只有第二象限）===" % args.difficulty_path)
    print(f"  表内 {len(tbl)} 行，被在线观测覆盖 {online_covered} 行（在线 k>=8）")
    print(f"  表判 band 内但在线全错（表乐观/模型退化）: {table_optimistic}")
    print(f"  ⚠ 余下 {len(tbl) - online_covered} 行**从未进过训练池**（被 band 排除或"
          "未被抽到）——它们能不能用，record 答不了。原版「可释放 / 表判 p≈1」两个"
          "计数结构上恒 0，已删除（别再把它读成「表没过期」）。")
    # p11g §4.1 判读入口：输入表的刻度直方图 +（给了 band 就）在表上预演训练端过滤。
    print()
    print(format_hist(f"输入表 {args.difficulty_path}", rate_histogram(tbl.values())))
    if args.difficulty_band:
        print(format_band_preview("输入表（未回填）", band_preview(tbl, lo, hi), lo, hi))

    if args.merge_out:
        m = merge_into_table(by_qk, args.difficulty_path or "", args.merge_out,
                             min_k=args.min_k)
        print("=== 在线增强表已写 ===")
        print(f"  表行 {m['table_rows']} 行，其中在线通过率回填覆盖 {m['online_matched']} 行"
              f"（min_k={m['min_k']}）→ {m['out']}")
        print("  下一段训练 --difficulty_path 指向该文件即可（原表未动）")
        # 【写文件后复读验证】直方图不重用内存里的 rows，改为**复读刚写的文件**：
        # 既给出"训练真正会吃的那张表"的刻度分布（p11g §4.1 的两表对照），又顺带
        # 证明落盘内容可被 load_difficulty_table 解析（同下游入口，不是另写一个 parser）。
        from rlab.data import load_difficulty_table as _ldt
        _merged = _ldt(args.merge_out)
        print()
        print(format_hist(f"输出表（在线增强）{args.merge_out}",
                          rate_histogram(_merged.values())))
        if args.difficulty_band:
            print(format_band_preview("输出表（在线增强）",
                                      band_preview(_merged, lo, hi), lo, hi))


if __name__ == "__main__":
    main()
