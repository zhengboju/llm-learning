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

【p11g 阶段 2：hard 桶探索通道（doc15 阶段 D 最小版）】三段命令、四步闭环：
    # ① 导出 hard 桶题集（一行一个 qk）
    python -m rlab.passrate --difficulty_path <主表> --difficulty_band 0.1 0.8 \
        --dump_qk_set hard:rlab_out/p11g_hard_qk.txt
    # ② 只探这些题，固定 16 个**新 seed**、协议与训练同源（--from_run_info）
    python -m rlab.probe_difficulty --pool train \
        --only_qk_file rlab_out/p11g_hard_qk.txt --k 16 --seed 4242 \
        --from_run_info <本 run 的 run_info.json> --out rlab_out/<探针表>.jsonl
    # ③ 并回主表副本（绝不动原表）→ 下一轮训练的 --difficulty_path
    python -m rlab.passrate --difficulty_path <主表> --difficulty_band 0.1 0.8 \
        --merge_probe rlab_out/<探针表>.jsonl --merge_out rlab_out/<增强表>.jsonl
**绝不把探索轨迹当 on-policy 组上传**：探针只写**难度表**，训练照旧用自己的采样
（回流的是"这题现在可学了"这个刻度，不是那些轨迹）。

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
    rows = read_table_rows(table_path)
    rows, st = apply_online_merge(rows, by_qk, min_k=min_k)
    write_table_rows(rows, out_path)
    return {"table_rows": st["table_rows"], "online_matched": st["online_matched"],
            "min_k": min_k, "out": out_path}


def read_table_rows(path: str):
    """读难度表的**原始行**（保序、保全部字段）；坏行跳过。

    与 load_difficulty_table 同一有效性口径（Q 非空、k 为正整数、n_correct 合法）——
    两个入口对"什么算一行有效的表数据"必须一致，否则过滤端与回填端会各自丢不同的行。"""
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            k, nc = r.get("k"), r.get("n_correct")
            if (not r.get("Q") or not isinstance(k, int) or k <= 0
                    or not isinstance(nc, int) or not 0 <= nc <= k):
                continue
            rows.append(r)
    return rows


def write_table_rows(rows, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def apply_online_merge(rows, by_qk: dict, min_k: int = 16):
    """纯函数（CPU 可测）：在线通过率回填进表行副本 → `(rows, stats)`。

    不改入参行（逐行 `dict(r)` 后才改），所以"原表未动"是结构保证而不是靠调用方自觉。"""
    out, matched = [], 0
    for r in rows:
        on = by_qk.get(qk_of(r["Q"]))
        if on and on["k"] >= min_k:
            r = dict(r)
            r["k"] = on["k"]
            r["n_correct"] = on["n_correct"]
            r["pass_rate"] = on["n_correct"] / on["k"]
            r["online_meta"] = {"source_rows": on["n_rows"],
                                "note": "在线回填（passrate.py），原 k/n_correct 被覆盖"}
            matched += 1
        out.append(r)
    return out, {"table_rows": len(out), "online_matched": matched, "min_k": min_k}


# ---- 阶段 2：hard 桶探索通道（doc15 阶段 D 最小版，2026-10-22）----
# 三件事必须都做成**纯函数 + CPU 可测**，且"给出题集 → 只探这些题"要有断言锁死，
# 否则就是本项目反复踩的那类错：「过滤器存在且测试全绿，但真实数据流没生效」。
QK_BUCKETS = ("hard", "easy", "boundary")


def bucket_rows(rows, lo: float, hi: float, bucket: str):
    """纯函数：按 band 把表行分桶 → 该桶的行（保序）。

    判据与告警用的 `data.filter_qas_by_difficulty` **同一分支**：`rate<=lo` →
    hard（含 rate=0）、`rate>=hi` → easy（含 rate=1）、其余 → boundary。
    注意 `hard` 与日志里的 `全错`(p_zero, 仅 rate==0) 不是一回事：hard 还含
    `0<rate<=lo` 的"罕见可解"（k=16 的 1/16=0.062）——那正是要探索的人群。"""
    if bucket not in QK_BUCKETS:
        raise ValueError(f"bucket 必须是 {QK_BUCKETS} 之一，收到 {bucket!r}")
    out = []
    for r in rows:
        rate = r["n_correct"] / r["k"]
        if bucket == "hard" and rate <= lo:
            out.append(r)
        elif bucket == "easy" and rate >= hi:
            out.append(r)
        elif bucket == "boundary" and lo < rate < hi:
            out.append(r)
    return out


def parse_qk_set_spec(spec: str):
    """`BUCKET:PATH` → (bucket, path)。按**第一个冒号**切，Windows 盘符不受影响。"""
    if ":" not in spec:
        raise ValueError(f"--dump_qk_set 需要 BUCKET:PATH（如 hard:rlab_out/hard_qk.txt），"
                         f"收到 {spec!r}")
    bucket, path = spec.split(":", 1)
    bucket = bucket.strip()
    if bucket not in QK_BUCKETS:
        raise ValueError(f"--dump_qk_set 的 BUCKET 必须是 {QK_BUCKETS} 之一，收到 {bucket!r}")
    if not path.strip():
        raise ValueError(f"--dump_qk_set 缺路径：{spec!r}")
    return bucket, path.strip()


def dump_qk_set(rows, lo: float, hi: float, spec: str):
    """纯函数（除写文件外）：把某桶的题面指纹写成一行一个 qk，返回统计。

    qk 用 `qk_of`（与 rollout/eval/probe 同一算法）——探索探针靠它认题。"""
    bucket, path = parse_qk_set_spec(spec)
    sel = bucket_rows(rows, lo, hi, bucket)
    with open(path, "w", encoding="utf-8") as f:
        for r in sel:
            f.write(qk_of(r["Q"]) + "\n")
    return {"bucket": bucket, "path": path, "n": len(sel), "table_rows": len(rows)}


def apply_probe_merge(rows, probe_rows, min_k: int = 16,
                      fingerprint_keys=None):
    """纯函数（CPU 可测）：把探索探针的结果并回主表行副本 → `(rows, stats)`。

    按 **Q 全文**匹配（不用 qk：12 hex 有碰撞面，而两边都有 Q 原文可精确匹配；不用
    A 匹配：同题同答案键，加了只会引入新的错配面）。只覆盖**测量字段**
    （k/n_correct/pass_rate + 探针同时给出的质量列），原表的 `probe_meta`（协议指纹，
    代表这张表是哪个『模型×提示×预算』的产物）与 `A` 都不动；新值出处写进
    `probe_merge_meta`——"这行的刻度从哪来"必须每行可自证。

    **协议指纹不一致 → raise**（`data.PROBE_META_FINGERPRINT_KEYS`，单一出处）：
    把另一档预算/协议下探的通过率静默并进训练表，比没有课程更糟。`seed`/`k`/`pool`
    刻意不在指纹里——探索通道的全部意义就是**换 seed 重探**。"""
    if fingerprint_keys is None:
        from rlab.data import PROBE_META_FINGERPRINT_KEYS
        fingerprint_keys = PROBE_META_FINGERPRINT_KEYS
    by_q = {}
    for p in probe_rows:
        by_q.setdefault(str(p["Q"]), p)
    out, matched, skipped_k = [], 0, 0
    probe_no_meta, table_no_meta = 0, 0
    mismatched = {}
    matched_q = set()
    for r in rows:
        p = by_q.get(str(r["Q"]))
        if p is None:
            out.append(r)
            continue
        matched_q.add(str(r["Q"]))
        if p["k"] < min_k:
            skipped_k += 1
            out.append(r)
            continue
        pmeta, tmeta = p.get("probe_meta"), r.get("probe_meta")
        if tmeta and not pmeta:
            probe_no_meta += 1        # 探针行不能自证出处 → 拒绝覆盖（不是"没指纹就放行"）
            out.append(r)
            continue
        if not tmeta:
            table_no_meta += 1        # 主表行无指纹（2026-09-18 前的旧表）：无从比对，照并但计数
        else:
            for fk in fingerprint_keys:
                pv = pmeta.get(fk)
                if pv is not None and pv != tmeta.get(fk):
                    mismatched.setdefault(fk, (pv, tmeta.get(fk)))
        if mismatched:
            out.append(r)          # 先不动，下面统一 raise（避免写半张混表）
            continue
        r = dict(r)
        for f in ("k", "n_correct", "fmt_rate", "trunc_rate", "avg_clen", "avg_code_ok"):
            if f in p:
                r[f] = p[f]
        r["pass_rate"] = p["n_correct"] / p["k"]
        r["probe_merge_meta"] = {
            "k": p["k"], "n_correct": p["n_correct"],
            "seed": (pmeta or {}).get("seed"),
            "note": "hard 桶探索探针回填（passrate.py --merge_probe），原刻度被覆盖"}
        matched += 1
        out.append(r)
    if mismatched:
        raise ValueError(
            "[passrate] 探索探针表的协议指纹与主表不符，拒绝并表（静默混表比没有课程更糟）："
            + "；".join(f"{k} 探针{v[0]!r}≠主表{v[1]!r}" for k, v in mismatched.items())
            + "。用与训练同协议的 --from_run_info <run_info.json> 重跑 probe_difficulty。")
    return out, {"table_rows": len(out), "probe_rows": len(probe_rows),
                 "matched": matched, "skipped_low_k": skipped_k,
                 "probe_no_meta": probe_no_meta, "table_no_meta": table_no_meta,
                 "probe_only": len([q for q in by_q if q not in matched_q]),
                 "min_k": min_k}


def main():
    ap = argparse.ArgumentParser(description="训练期 per-question 在线通过率聚合（纯观测）")
    ap.add_argument("--record", default=None,
                    help="训练 record.jsonl 路径；不给她就跳过在线聚合（表侧读数、"
                         "--dump_qk_set 与 --merge_probe 都不需要 record）")
    ap.add_argument("--difficulty_path", default=None,
                    help="可选：训练用的难度表；给了就输出'表过期诊断'"
                         "（在线已能做对/仍全错的题在 band 外各有多少）")
    ap.add_argument("--difficulty_band", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="与训练同 band（如 0.25 0.75）；--difficulty_path 给了但这个没给则报错。"
                         "另外用于 band 预演：在表上跑训练端同一个 filter_qas_by_difficulty，"
                         "启动训练前就能知道日志会打印的 N -> K 与四个计数；"
                         "--dump_qk_set 的分桶也用它")
    ap.add_argument("--merge_out", default=None,
                    help="可选：把在线通过率/探索探针结果回填进难度表副本并写到该路径"
                         "（生成'增强表'供下一段训练 --difficulty_path 用）")
    ap.add_argument("--min_k", type=int, default=16,
                    help="回填门槛：来源样本 k>=min_k 才覆盖表行（默认 16，"
                         "低于此噪声过大）；在线率那种 k≈8 的场景显式传 8")
    ap.add_argument("--dump_qk_set", default=None, metavar="BUCKET:PATH",
                    help="【p11g 阶段2】把某桶的题面指纹按行写成 qk 文件，供探索探针"
                         "只探这些题：BUCKET ∈ hard|easy|boundary（判据与训练端同一分支）。"
                         "例：--difficulty_band 0.1 0.8 --dump_qk_set hard:rlab_out/hard_qk.txt")
    ap.add_argument("--merge_probe", default=None, metavar="PROBE_TABLE",
                    help="【p11g 阶段2】把探索探针表（probe_difficulty --only_qk_file 的产物）"
                         "并回主表副本：按 Q 全文匹配、只覆盖测量字段、保留原 probe_meta、"
                         "新值出处写进 probe_merge_meta；协议指纹不符则拒绝并表。"
                         "必须配 --merge_out（绝不动原表）")
    args = ap.parse_args()

    if args.difficulty_path and args.difficulty_band is None:
        raise SystemExit("--difficulty_path 需要配对 --difficulty_band LO HI（与训练同值）")
    for _need, _flag in ((args.dump_qk_set, "--dump_qk_set"),
                         (args.merge_probe, "--merge_probe")):
        if _need and not args.difficulty_path:
            raise SystemExit(f"{_flag} 需要 --difficulty_path（要合并/导出的那张主表）")
    if args.merge_probe and not args.merge_out:
        raise SystemExit("--merge_probe 必须配 --merge_out（并表只写新文件，绝不动原表）")

    by_qk = {}
    if args.record:
        if not os.path.exists(args.record):
            raise SystemExit(f"record 不存在: {args.record}")
        by_qk = aggregate_by_qk(args.record)
        if not by_qk:
            print("聚合结果为空：record 里没有带 qk 的行（2026-09-23 之前的旧 record "
                  "没有题目指纹字段；本次训练重启后新行才带）。")
    else:
        print("[passrate] 未给 --record：跳过在线通过率聚合"
              "（表侧读数 / --dump_qk_set / --merge_probe 照常）")
    if args.merge_out and not by_qk and not args.merge_probe:
        raise SystemExit("--merge_out 需要 --record 或 --merge_probe："
                         "否则会写出一份与输入逐位相同的表（静默无效产物）")

    lo, hi = (args.difficulty_band or (0.0, 1.0))
    if by_qk:
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
        if by_qk:
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

    if args.dump_qk_set:
        # 探索通道第一步：导出 hard 桶题集（一行一个 qk）。
        _rows = read_table_rows(args.difficulty_path)
        try:
            _d = dump_qk_set(_rows, lo, hi, args.dump_qk_set)
        except ValueError as _e:
            raise SystemExit(str(_e))
        _allb = {b: len(bucket_rows(_rows, lo, hi, b)) for b in QK_BUCKETS}
        print("\n=== 题集导出（--dump_qk_set）===")
        print(f"  {_d['bucket']}: {_d['n']} 题 / 表内 {_d['table_rows']} 行"
              f"（band=({lo},{hi})；三桶 hard {_allb['hard']} / boundary "
              f"{_allb['boundary']} / easy {_allb['easy']}）→ {_d['path']}")
        print("  下一步（探索，绝不进 on-policy 上传）："
              "python -m rlab.probe_difficulty --pool train "
              f"--only_qk_file {_d['path']} --k 16 --seed <新 seed> "
              "--from_run_info <本 run 的 run_info.json> --out rlab_out/<探针表>.jsonl")

    if args.merge_out:
        _rows = read_table_rows(args.difficulty_path)
        _hard_before = len(bucket_rows(_rows, lo, hi, "hard")) if args.difficulty_band else None
        _st_online = _st_probe = None
        if by_qk:
            _rows, _st_online = apply_online_merge(_rows, by_qk, min_k=args.min_k)
        if args.merge_probe:
            _probe_rows = read_table_rows(args.merge_probe)
            if not _probe_rows:
                raise SystemExit(f"探索探针表为空/不可读: {args.merge_probe}")
            try:
                _rows, _st_probe = apply_probe_merge(_rows, _probe_rows,
                                                     min_k=args.min_k)
            except ValueError as _e:
                raise SystemExit(str(_e))
        write_table_rows(_rows, args.merge_out)
        print("\n=== 增强表已写 ===")
        if _st_online:
            print(f"  表行 {_st_online['table_rows']} 行，其中在线通过率回填覆盖 "
                  f"{_st_online['online_matched']} 行（min_k={_st_online['min_k']}）")
        if _st_probe:
            print(f"  探索探针并回：匹配 {_st_probe['matched']} 题"
                  f"（探针表 {_st_probe['probe_rows']} 行；k<min_k({_st_probe['min_k']}) 跳过 "
                  f"{_st_probe['skipped_low_k']}；探针行无指纹跳过 {_st_probe['probe_no_meta']}；"
                  f"主表行无指纹（无从校验）{_st_probe['table_no_meta']}；"
                  f"探针里主表没有的 {_st_probe['probe_only']} 题丢弃=宁缺勿错配）")
            if _st_probe["matched"] == 0:
                print("  ⚠ 零覆盖：探针表与主表没有一题匹配上（Q 文本不同源？）"
                      "——这份输出与输入逐位相同，别当成「并表成功」")
        print(f"  → {args.merge_out}（原表未动）")
        # 【写文件后复读验证】直方图不重用内存里的 rows，改为**复读刚写的文件**：
        # 既给出"训练真正会吃的那张表"的刻度分布（p11g §4.1 的两表对照），又顺带
        # 证明落盘内容可被 load_difficulty_table 解析（同下游入口，不是另写一个 parser）。
        from rlab.data import load_difficulty_table as _ldt
        _merged = _ldt(args.merge_out)
        print()
        print(format_hist(f"输出表（增强）{args.merge_out}",
                          rate_histogram(_merged.values())))
        if args.difficulty_band:
            print(format_band_preview("输出表（增强）",
                                      band_preview(_merged, lo, hi), lo, hi))
            if _st_probe and _hard_before is not None:
                _hard_after = len(bucket_rows(_rows, lo, hi, "hard"))
                print(f"  hard 桶 {_hard_before} → {_hard_after} 题"
                      f"（{_hard_before - _hard_after} 题被新刻度抬出 hard；"
                      f"它们下一轮以当前策略重新生成常规组——探索轨迹本身不上传）")


if __name__ == "__main__":
    main()
