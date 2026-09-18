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
SESS_GAP_S = 120.0     # record 时间戳间隔 >120s = 新训练会话（旧协议：record 无 gen_version）
SESS_GAP_GV_S = 1800.0  # 新协议（有 gen_version）下的时间兜底阈值：见 summarize_record


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


# ---------------------------------------------- 代码分层分析（2026-09-18）----
# 【为什么】p5 的 step300 增益被抹平（§7.5 判据对照），且 record 的 code_ok 率
# 56% → 10% 单调崩——需要区分两种机制：
#   H1「理性压灭」：写代码的样本 acc 不高于纯推理 → 工具路径在 4B/GSM8K 上无真实
#      优势，崩塌是 RL 的最优解（无调参解，诚实结论，参照 §7.5.4 的 H2 分支）；
#   H2「激励/可行性不足」：写代码的样本 acc 更高 → 是信号问题（可救）。
# 判据 = eval per-item 里 code_used=1 vs code_used=0 两组的 acc 差（同题配对更好）。
# 数据源是 eval json 的 per-item（eval_vllm_one.py --dump_items，默认开），零训练成本。

def code_layer(items, key: str = "qk"):
    """按 code_used 分层一份 per-item 结果 → ((n_on, n_on_acc), (n_off, n_off_acc))。

    code_used>0 记"用码"（与 eval 的 code_rate 口径一致）。任一臂无 items 返回
    None。跨存档点的**同层**配对（如 model 用码层 vs BASE 用码层）由
    summarize_code_layer 单独做——单份 items 内同一题不可能既用码又不用码，
    code_layer 自身不做配对。"""
    _parts = _code_layer_items(items, key)
    if _parts is None:
        return None
    on, off = _parts
    return ((len(on), sum(1 for it in on if it.get("acc", 0) == 1.0)),
            (len(off), sum(1 for it in off if it.get("acc", 0) == 1.0)))


def _code_layer_items(items, key: str = "qk"):
    """分层后返回 (用码 item 列表, 纯推理 item 列表)；无可用 items 返回 None。"""
    if not items:
        return None
    on, off = [], []
    for it in items:
        if not it.get(key):
            continue
        (on if (it.get("code_used") or 0) > 0 else off).append(it)
    if not on and not off:
        return None
    return on, off


def _acc_col(n_acc: int, n: int) -> str:
    if not n:
        return "—"
    return f"{n_acc / n * 100:.1f}% ({n_acc}/{n})"


def summarize_code_layer(path: str, base_name: str = "BASE") -> str:
    """跨存档点输出「用码 vs 纯推理」分层 acc 表。

    读 eval json（per-item），对每个模型按 code_used 分层出 acc；再对每个模型
    与 BASE 的**同题**做 code 层配对 McNemar，回答「增益集中在哪一层」——
    code_on 层显著而 code_off 层噪声内 ⇒ 增益来自工具路径（健康）；反之
    增益全在 code_off ⇒ 工具是噪声（崩塌机制 H1 实锤，§7.5.4 判据）。"""
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    models = {k: v for k, v in results.items() if not k.startswith("_")}
    base = models.get(base_name)
    lines = ["| 模型 | 用码题 | 用码 acc | 纯推理题 | 纯推理 acc |",
             "|---|---|---:|---:|---:|"]
    for name, r in models.items():
        items = r.get("items")
        cl = code_layer(items) if items else None
        if cl is None:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        (n_on, a_on), (n_off, a_off) = cl
        lines.append(f"| {name} | {n_on} | {_acc_col(a_on, n_on)} | {n_off} | "
                     f"{_acc_col(a_off, n_off)} |")
    if not base:
        return "\n".join(lines)
    lines += ["", "**同题配对（code 层 vs BASE 同层）：**",
              "| 模型 | 层 | Δacc vs BASE | McNemar | 判定 |",
              "|---|---|---:|---|---|"]
    for name, r in models.items():
        if name == base_name:
            continue
        items, bitems = r.get("items"), base.get("items")
        if not items or not bitems:
            continue
        a_parts = _code_layer_items(items)
        b_parts = _code_layer_items(bitems)
        if not a_parts or not b_parts:
            continue
        for layer, (a_items, b_items) in (("用码", (a_parts[0], b_parts[0])),
                                          ("纯推理", (a_parts[1], b_parts[1]))):
            _b = _c = _m = 0
            bm = {it.get("qk"): it for it in a_items if it.get("qk")}
            bb = {it.get("qk"): it for it in b_items if it.get("qk")}
            for k in bm.keys() & bb.keys():
                am, ab = bm[k].get("acc", 0) == 1.0, bb[k].get("acc", 0) == 1.0
                if am and not ab:
                    _b += 1
                elif ab and not am:
                    _c += 1
                _m += 1
            if _m:
                p = mcnemar_exact(_b, _c)
                _n1, _na1 = len(a_items), sum(1 for it in a_items if it.get("acc", 0) == 1.0)
                _n2, _na2 = len(b_items), sum(1 for it in b_items if it.get("acc", 0) == 1.0)
                _d, _ = diff_ci95(_na1 / _n1, _n1, _na2 / _n2, _n2)
                lines.append(f"| {name} | {layer} | {_d:+.1f}pp | "
                             f"p={p:.3f} (b={_b}/c={_c}, n={_m}) | {_verdict(_d, 0, p)} |")
    return "\n".join(lines)


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


def _pair_col(pairs):
    """pairs = [(ref_item, target_item), ...] 同题对 → 'target acc / ref acc' 字符串。

    同题集上两个模型的 acc 直接可读：如 '62.3% (137/220) vs B 73.2% (161/220)' 表示
    这 220 题上 target 62.3%、BASE 73.2%——放弃代码的代价一目了然。"""
    if not pairs:
        return "—"
    n = len(pairs)
    t_ok = sum(1 for _, t in pairs if t.get("acc", 0) == 1.0)
    r_ok = sum(1 for r, _ in pairs if r.get("acc", 0) == 1.0)
    return (f"{t_ok / n * 100:.1f}% ({t_ok}/{n})"
            f" vs B {r_ok / n * 100:.1f}% ({r_ok}/{n})")


def summarize_code_migration(path: str, ref_name: str = "BASE") -> str:
    """分层迁移分析：以 ref_name（通常 BASE）的分层为锚，追踪各层题在其他
    存档点里的去向，**同题集上给出 target 与 ref 双 acc**。

    【2026-09-18 由来】--code-layer 只回答"每个存档点自己分层的 acc"，回答不了
    "BASE 用码的题到 step300 转纯推理后答得怎么样"——而这是"增益为何被抹平"的
    核心证据（p5：step300 -3.4pp = 丢代码 -12.7pp×228 题 + 涨推理 +4.6pp×272 题）。
    每格输出同题集上的双 acc，放弃代码的代价/改用代码的收益逐格可读。"""
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    models = {k: v for k, v in results.items() if not k.startswith("_")}
    ref = models.get(ref_name)
    if not ref or not ref.get("items"):
        return f"（{ref_name} 无 items，无法作迁移锚）"
    ref_items = {it.get("qk"): it for it in ref["items"] if it.get("qk")}
    lines = [f"**分层迁移（以 {ref_name} 分层为锚；同题集上 target vs {ref_name}）**",
             "| 目标 | BASE层 | 目标同层 (target/B同题) | 目标转另一层 (target/B同题) | 目标缺题 |",
             "|---|---|---|---:|---:|"]
    for name, r in models.items():
        if name == ref_name:
            continue
        items = {it.get("qk"): it for it in (r.get("items") or []) if it.get("qk")}
        for layer, keep_used in (("用码", True), ("纯推理", False)):
            same, trans, missing = [], [], 0
            for qk, rit in ref_items.items():
                # 只按锚层（ref）的 code_used 分组；target 的层只看是否与锚层相同。
                # 【2026-09-18】链式比较陷阱：`(code or 0) > 0 == keep_used` 会被
                # Python 解析成 `(code>0) and (0 == keep_used)`（共享中间值 0），
                # 恒假/恒真——必须显式括号 `((code or 0) > 0) == keep_used`。
                if ((rit.get("code_used") or 0) > 0) != keep_used:
                    continue
                t = items.get(qk)
                if t is None:
                    missing += 1
                elif (((t.get("code_used") or 0) > 0) == keep_used):
                    same.append((rit, t))
                else:
                    trans.append((rit, t))
            same_col = _pair_col(same)
            trans_col = _pair_col(trans)
            if layer == "用码":
                # BASE 用码题 → 目标也用码 / 转纯推理（◆ 放弃代码的归宿）
                lines.append(f"| {name} | {layer} | {same_col} | {trans_col}◆ | {missing} |")
            else:
                lines.append(f"| {name} | {layer} | {same_col} | {trans_col} | {missing} |")
    return "\n".join(lines)


def pair_eval(path_a: str, path_b: str, name_a: str, name_b: str) -> str:
    """跨 json 两两同题配对（McNemar）。

    【2026-09-13 由来】run2 的 m200（唯一显著正增益 +5.8pp p=0.006）原始 ckpt
    被 P1 覆盖、step_200_mm 合并副本又被手工删除 —— **模型已灭失**。但它的
    per-item 评测 json 还在；eval 抽题是 seed/split/n 确定的，两次评测抽到
    同一批题，qk 同题配对依旧成立。于是"活模型 p1s200 vs 死模型 m200"的
    单变量对照（同 step、唯一差异 trunc_shaping）可以靠两份 json 打完。

    用法：python -m rlab.analysis --eval-json 新.json --pair-json 旧.json
              --pair-a p1s200 --pair-b m200
    模型名先在 --eval-json 里找、找不到再找 --pair-json；两边都没有 → 列出
    可用名字（防拼错静默空表）。"""
    def load(p):
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        return {k: v for k, v in res.items() if not k.startswith("_")}

    ja, jb = load(path_a), load(path_b)

    def find(name):
        for res, path in ((ja, path_a), (jb, path_b)):
            if name in res:
                return res[name], path
        return None, None

    ma, sa = find(name_a)
    mb, sb = find(name_b)
    if not ma or not mb:
        missing = name_a if not ma else name_b
        avail = sorted(set(ja) | set(jb))
        return (f"[pair] 两个 json 里都找不到 {missing!r}。可用模型名："
                f"{', '.join(avail) if avail else '（无）'}")
    lines = [f"> [跨json配对] {name_a} ← {os.path.basename(sa)}"
             f"  vs  {name_b} ← {os.path.basename(sb)}（同题 qk 配对，模型本体无需在世）",
             "| 模型 | acc%(±95%CI) | n | Δacc(A−B) | 检验 | 判定 |",
             "|---|---|---:|---|---|---|"]
    for name, r in ((name_a, ma), (name_b, mb)):
        n = r.get("n") or 0
        acc = r.get("acc", 0.0) * 100
        hw = ci95(r.get("acc", 0.0), n)
        acc_col = f"{acc:.1f}±{hw:.1f}" if hw == hw else f"{acc:.1f}"
        lines.append(f"| {name} | {acc_col} | {n} | — | — | — |")
    na, nb = ma.get("n") or 0, mb.get("n") or 0
    if na and nb:
        d, h = diff_ci95(ma.get("acc", 0.0), na, mb.get("acc", 0.0), nb)
        pc = paired_counts(ma.get("items"), mb.get("items"))
        if pc is not None:
            b, c, matched = pc
            p = mcnemar_exact(b, c)
            test = f"McNemar p={p:.3f} (b={b}/c={c}, n={matched})"
            verdict = _verdict(d, h, p)
        else:
            test = "两比例（无 per-item 可配对）"
            verdict = _verdict(d, h)
        lines.append(f"| **A−B** | {d:+.1f}±{h:.1f}pp | {min(na, nb)} | {d:+.1f}pp | {test} | {verdict} |")
    return "\n".join(lines)


def _sess_label(sess: int) -> str:
    """会话显示标签：0..25 → A..Z，之后 S26/S27…。

    【2026-09-17 真机】旧版写死 `"ABCDEFGH"[sess_ids[i]]`，第 9 个会话直接
    `IndexError: string index out of range` —— 而 record.jsonl 是**追加写**的，
    同一 out_dir 下"验收跑 + 正式跑 + 中途重启"叠在一个文件里是常态（真机这次
    就是这么崩的：`--record` 整个不可用，正好卡在"训练期曲线是唯一判别证据"
    的时刻）。标签只用于显示，不得因为会话数多于字母表长度而拒绝出表。"""
    if 0 <= sess < 26:
        return chr(ord("A") + sess)
    return f"S{sess}"


def summarize_record(path: str, window: int = 160, clen_cap: int = 1800) -> str:
    """按 upload 批次滑动平均 acc/fmt(code/code_ok/trunc) 率与完成长度（retool 诊断用）。

    clen_cap ≈ max_context_tokens(2200) - 典型 prompt(~400) = 1800：接近上限
    说明轨迹在撞上下文预算（会被整组丢弃或标签被截断）——2026-09-08 第四轮
    "格式学到 75-95% 后崩回 0"的嫌疑机制，需 clen/code 趋势佐证。

    window 以**样本**计（1 条 record = num_pre_Q=8 样本 = 1 组 = 1 micro-step）。
    【2026-09-17 改默认 20→160】旧默认 20 样本 = 2.5 组，20 样本的二项噪声就有
    ±11pp：真机 bg1 的相邻窗口在 10% 与 70% 之间跳，趋势被噪声完全淹没（且极易
    被读成"崩了又好了"）。160 样本 = 20 组，与 docs/05 的"200 组窗口"同一量级。

    【会话拆分 2026-09-08 / 2026-09-17 加固】record.jsonl 以追加模式写入，多次
    训练（重启/新 run）会写进同一文件，且每次会话 pushes 计数归零。

    **切会话判据分档**（2026-09-17 真机定案）：
      · 有 `gen_version`（新协议）→ **只看 gen_version 回退**（新 run 从 0 重新
        计数），时间阈值放宽到 SESS_GAP_GV_S(30min) 只兜"真重启"。
        为什么不能沿用 120s：`gen_questions_per_attempt=4` 时**一次 attempt 的
        4 条记录时间戳完全相同**（4 题一次性上传），attempt 之间隔 ~3min →
        120s 判据把一次 106 组的 run 切成 **31 个"会话"**，逐会话表彻底失去意义
        （真机 bg1 的原始读数就是这个形态）。
      · 无 `gen_version`（旧协议）→ 沿用 120s 时间判据。"""
    accs, fmts, codes, oks, trs, clens, phases, sess_ids = [], [], [], [], [], [], [], []
    stales = []          # 每样本 staleness（opt-step 口径，见下；无 gen_version 时为空）
    sess_span = {}   # sess -> [first_t, last_t]（墙钟，便于对 Shell 历史核对是哪次 run）
    sess_gv = {}     # sess -> [first_genver, last_genver]
    has_gv = False
    with open(path, encoding="utf-8") as f:
        prev_t, prev_gv, sess = None, None, 0
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = len(rec.get("acc", []))
            if n == 0:
                continue
            t = rec.get("t")
            gv = rec.get("gen_version")
            _gap = SESS_GAP_GV_S if has_gv else SESS_GAP_S
            _new = (prev_t is not None and t is not None and t - prev_t > _gap)
            if not _new and isinstance(gv, int) and isinstance(prev_gv, int) and gv < prev_gv:
                _new = True   # 权重推送计数回退 = 新 run（时间判据在这种 run 里会误切）
            if _new:
                sess += 1
            if t is not None:
                prev_t = t
                sess_span.setdefault(sess, [t, t])[1] = t
            if isinstance(gv, int):
                has_gv = True
                prev_gv = gv
                sess_gv.setdefault(sess, [gv, gv])[1] = gv
            # 【2026-09-18 staleness 可观测】每样本的陈旧度（opt-step 口径，与
            # train.py [train][口径] 行同公式）。1 record = 8 样本 = 1 micro-step；
            # 本批第 i 个样本的 micro-step = (累计样本数 + i)//8。gen_version 是
            # 生成该批时权重对应的 micro-step。staleness 过高 = 训练在吃太旧的
            # 策略数据（off-policy，框架监控建议②）。
            if isinstance(gv, int):
                _gas = 4                                    # 与 config 一致（见函数尾注释）
                for _i in range(n):
                    _m = (len(accs) + _i) // 8              # 本样本的 micro-step
                    stales.append((_m // _gas) - (gv // _gas))
            accs.extend(a > 0 for a in rec["acc"])
            fmts.extend(v > 0 for v in rec["fmt"])
            codes.extend(u > 0 for u in rec.get("code_used", []))
            oks.extend(k > 0 for k in rec.get("code_ok", []))
            trs.extend(int(x) for x in rec.get("trunc_final", []))
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
            k = (sum(oks[i] for i in idx) / len(idx) * 100 if oks else float("nan"))
            tr = (sum(trs[i] for i in idx) / len(idx) * 100 if trs else float("nan"))
            cond = f"{a / ff * 100:.1f}%" if ff > 0 else "—"
            lo, hi = idx[0], idx[-1] + 1
            span = sess_span.get(s)
            when = ""
            if span:
                fmt_t = lambda x: time.strftime("%m-%d %H:%M:%S", time.localtime(x))
                when = f" [{fmt_t(span[0])} ~ {fmt_t(span[1])}]"
            gvr = sess_gv.get(s)
            gv_col = f" gen_ver={gvr[0]}..{gvr[1]}" if gvr else ""
            st_col = ""
            if stales:
                _ss = [stales[i] for i in idx if i < len(stales)]
                if _ss:
                    st_col = f" staleness均值={sum(_ss) / len(_ss):.1f}(max {max(_ss)})"
            sess_lines.append(
                f"会话{_sess_label(s)}(#{s}): 样本{lo}~{hi}（{len(idx)}条 ≈{len(idx)/8:.0f}组）"
                f" acc={a:.1f}% fmt={ff:.1f}% 条件精度={cond} code_ok={k:.1f}% trunc={tr:.1f}%"
                f"{gv_col}{st_col}{when}")
    out = ["| 样本窗口 | ≈组 | acc率 | fmt率 | 条件精度 | code率 | code_ok率 | trunc率 | avg_clen | staleness | 阶段 | 会话 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if sess_span:
        out.insert(0, f"> record 共 {len(sess_span)} 个会话（新协议按 gen_version 回退切分，"
                      f"旧协议按 >{SESS_GAP_S:.0f}s 间隔；见函数 docstring）"
                      f"——追加写文件，多会话 = 同一 out_dir 被重启/多 run 混用；"
                      f"末尾会话才是最近一次 run，且同签名重跑会覆盖 "
                      f"step_N，评测前先核 `step_N/run_info.json` 的 started。")
        out.insert(1, "")
    for i in range(0, len(accs), window):
        j = i + window
        chunk_a, chunk_f, chunk_c = accs[i:j], fmts[i:j], codes[i:j]
        chunk_k, chunk_t, chunk_l = oks[i:j], trs[i:j], clens[i:j]
        if not chunk_a:
            continue
        code_col = f"{sum(chunk_c) / len(chunk_c) * 100:.1f}%" if chunk_c else "—"
        ok_col = f"{sum(chunk_k) / len(chunk_k) * 100:.1f}%" if chunk_k else "—"
        tr_col = f"{sum(chunk_t) / len(chunk_t) * 100:.1f}%" if chunk_t else "—"
        if chunk_l:
            avg_l = sum(chunk_l) / len(chunk_l)
            near = sum(1 for l in chunk_l if l >= 0.9 * clen_cap) / len(chunk_l)
            len_col = f"{avg_l:.0f}（{near * 100:.0f}%≥{int(0.9 * clen_cap)}）"
        else:
            len_col = "—"
        # 【2026-09-18 staleness 列】窗口内样本陈旧度的均值/最大（opt-step 口径）。
        # >0 表示窗口内有样本吃到比推送周期更旧的策略（off-policy）；无 gen_version
        # 的旧 record 显示 "—"。均值反映"典型吃多旧"，最大反映"最坏吃多旧"。
        if stales and i < len(stales):
            _ch_s = stales[i:j]
            stal_col = f"{sum(_ch_s) / len(_ch_s):.1f}（max {max(_ch_s)}）" if _ch_s else "—"
        else:
            stal_col = "—"
        ph_col = phases[i] if i < len(phases) else "—"
        sess_col = _sess_label(sess_ids[i]) if i < len(sess_ids) else "—"
        # 【2026-09-17】条件精度 = acc率/fmt率 = "抽到 boxed 的轨迹里真做对的比例"。
        # 这是区分"学到数学"与"学会收尾"的唯一干净指标（health 的 surface 签名同口径）：
        # base 在预算充足时恒定 90%+，bg1 崩盘时掉到 25~38%——fmt 在涨而 acc 在掉。
        # fmt 率为 0 时给 "—"（格式死亡事件里 fmt 恒 0，除法会崩）。
        _a_rate = sum(chunk_a) / len(chunk_a)
        _f_rate = sum(chunk_f) / len(chunk_f)
        cond_col = f"{_a_rate / _f_rate * 100:.1f}%" if _f_rate > 0 else "—"
        out.append(f"| {i}~{j} | {i // 8}~{j // 8} "
                   f"| {_a_rate * 100:.1f}% "
                   f"| {_f_rate * 100:.1f}% | {cond_col} | {code_col} "
                   f"| {ok_col} | {tr_col} | {len_col} | {stal_col} | {ph_col} | {sess_col} |")
    if sess_lines:
        out.append("")
        out.append(f"== 会话拆分（新协议=gen_version 回退；旧协议=>{SESS_GAP_S:.0f}s 间隔）==")
        out.extend(sess_lines)
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--code-layer", default=None,
                    help="代码分层分析：跨存档点按 code_used 分层 acc + 与 BASE 同层配对")
    ap.add_argument("--code-migration", default=None,
                    help="分层迁移分析：以 BASE 分层为锚，追踪各层题在后续存档点里的去向"
                         "（放弃代码后答得怎么样）")
    ap.add_argument("--window", type=int, default=160,
                    help="record 曲线的窗口（单位=样本；8 样本=1 组=1 micro-step。"
                         "默认 160=20 组；调小看细节但噪声按 1/√n 放大）")
    ap.add_argument("--base", default="BASE")
    # 跨 json 两两配对（模型本体可以已灭失，只要有 per-item json）
    ap.add_argument("--pair-json", default=None,
                    help="另一份 eval json（--pair-a/--pair-b 的模型可来自任一份）")
    ap.add_argument("--pair-a", default=None, help="配对臂 A 的模型名")
    ap.add_argument("--pair-b", default=None, help="配对臂 B 的模型名")
    args = ap.parse_args()
    _primary = args.eval_json
    if args.pair_json and args.pair_a and args.pair_b:
        if not _primary:
            cands = sorted(glob.glob("eval_vllm_all*.json"))
            if not cands:
                raise SystemExit("[pair] 需要 --eval-json（主 json）")
            _primary = cands[-1]
        print(pair_eval(_primary, args.pair_json, args.pair_a, args.pair_b))
    if args.eval_json:
        print(summarize_eval(args.eval_json, args.base))
    if args.code_layer:
        print(summarize_code_layer(args.code_layer, args.base))
    if args.code_migration:
        print(summarize_code_migration(args.code_migration, args.base))
    if args.record:
        print(summarize_record(args.record, window=args.window))
    if not args.eval_json and not args.record and not args.code_layer and not args.pair_json:
        cands = sorted(glob.glob("eval_vllm_all*.json"))
        if cands:
            print(summarize_eval(cands[-1], args.base))
        else:
            print(" nothing to summarize（--eval-json / --record）")
