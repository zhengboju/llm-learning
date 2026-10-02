# -*- coding: utf-8 -*-
"""rlab/analysis.py — 结果解析与对比表。

功能：
  1. 汇总 eval_v_*.json -> Markdown 对比表（含与 BASE 的差值、与噪声地板 ±2pp 的判定）；
  2. 解析 rlab record.jsonl -> 训练中 acc/format 正确率随上传批次的曲线数据；
  3. 无 boxed 归因分解（--no-boxed-breakdown）：把"终局没给 boxed"的样本按分族 ×
     机理拆开，给出零梯度占比——判"该加单轮额度、还是该治啰嗦、还是该动提示层"；
     并给 ok 族的**末段 assistant 长度画像**（B 桶该抬 `answer_reserve` 还是治
     啰嗦；2026-10-02，输入是 rollout 的 segl/tsegl 字段）。

用法：
    python -m rlab.analysis --eval-json eval_vllm_all.json [--base BASE]
    python -m rlab.analysis --record rlab_out/record.jsonl
    python -m rlab.analysis --no-boxed-breakdown rlab_out/record.jsonl
"""
import argparse
import collections
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
    任一臂缺 items / 无 key / 无可对齐题时返回 None（调用方回落两比例检验）。

    【口径警告·2026-09-19】本函数按 acc==1.0 二值化，**只对 greedy（val_n=1）
    有效**。采样档（--val_n>1）的 per-item acc 是 Average@N ∈ [0,1]，acc==1.0
    退化成"N 条全对"——检验的是 all-N-correct 率而不是 acc（p8 实测：step100
    的"-4.6pp p=0.003 显著"是全对率之差，真实 acc 差仅 -1.5pp）。连续档一律走
    paired_mean_test，分派由 items_are_binary 判定。"""
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


def items_are_binary(*item_lists) -> bool:
    """per-item 的 acc 是否全为 0/1（greedy 档）。任一值落在开区间 → 采样档。

    分派依据：McNemar 要求二值配对结果；Average@N 的小数 acc 必须走连续配对
    检验，否则"显著"回答的是另一个问题（见 paired_counts 口径警告）。
    显式 val_n>1 标记也直接判为连续档（即使本次抽样恰好全 0/1）。"""
    for items in item_lists:
        for it in items or []:
            if int(it.get("val_n", 1) or 1) > 1:
                return False
            a = it.get("acc", 0)
            if a != 0 and a != 1:
                return False
    return True


def _norm_two_sided_p(z: float) -> float:
    """标准正态双侧 p（erfc 实现，无 scipy 依赖）。"""
    return math.erfc(abs(z) / math.sqrt(2.0))


def paired_mean_test(items_model, items_base, key: str = "qk"):
    """连续 per-item（Average@N）的同题配对均值检验 → (diff_pp, p, matched)。

    对每道题取 d_i = acc_model,i − acc_base,i（配对，消除题目难度方差），检验
    E[d]=0。用 z = mean(d)/SE 的正态近似（n≥30 足够；本项目 N=200/500 远超）。
    这是采样档 McNemar 的正确替代：McNemar 只能吃二值，而 Average@N 的信息
    （8 条里对几条）恰恰在小数部分，二值化会把它全部丢掉。"""
    if not items_model or not items_base:
        return None
    bm = {it.get(key): it for it in items_model if it.get(key)}
    bb = {it.get(key): it for it in items_base if it.get(key)}
    if not bm or not bb:
        return None
    diffs = [float(bm[k].get("acc", 0)) - float(bb[k].get("acc", 0))
             for k in bm.keys() & bb.keys()]
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0.0:
        # 所有题的差完全相同：非零差即确定性差异，零差即完全无变化
        return (mean * 100.0, (0.0 if mean != 0.0 else 1.0), n)
    se = math.sqrt(var / n)
    return (mean * 100.0, _norm_two_sided_p(mean / se), n)


def paired_test_auto(items_model, items_base, key: str = "qk"):
    """统一入口：按 items 口径自动选 McNemar（二值）或配对均值检验（连续）。

    返回 (delta_pp, p, matched, test_label)；无法配对时 None。"""
    if items_are_binary(items_model, items_base):
        pc = paired_counts(items_model, items_base, key)
        if pc is None:
            return None
        b, c, matched = pc
        p = mcnemar_exact(b, c)
        return (None, p, matched, f"McNemar p={p:.3f} (b={b}/c={c}, n={matched})")
    pm = paired_mean_test(items_model, items_base, key)
    if pm is None:
        return None
    d, p, matched = pm
    return (d, p, matched, f"配对均值 z 检验 p={p:.3f} (Δ={d:+.2f}pp, n={matched})")


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
    code_layer 自身不做配对。

    【2026-09-19】采样档下 acc/code_used 都是每题 Average@N 小数：acc 用求和
    （= 期望答对题数）而非 ==1.0 计数，否则"全 N 条对"才计分会系统性低估。"""
    _parts = _code_layer_items(items, key)
    if _parts is None:
        return None
    on, off = _parts
    return ((len(on), sum(float(it.get("acc", 0)) for it in on)),
            (len(off), sum(float(it.get("acc", 0)) for it in off)))


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


def _acc_col(n_acc: float, n: int) -> str:
    if not n:
        return "—"
    return f"{n_acc / n * 100:.1f}% ({n_acc:g}/{n})"


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
            # 【2026-09-19】同层配对也按口径分派（采样档走配对均值 z 检验）
            pt = paired_test_auto(a_items, b_items)
            if pt is None:
                continue
            _d_paired, p, _m, test = pt
            _n1 = len(a_items)
            _n2 = len(b_items)
            if not _n1 or not _n2:
                continue
            _na1 = sum(float(it.get("acc", 0)) for it in a_items)
            _na2 = sum(float(it.get("acc", 0)) for it in b_items)
            _d, _ = diff_ci95(_na1 / _n1, _n1, _na2 / _n2, _n2)
            if _d_paired is not None:
                _d = _d_paired
            lines.append(f"| {name} | {layer} | {_d:+.1f}pp | "
                         f"{test} | {_verdict(_d, 0, p)} |")
    return "\n".join(lines)


def legacy_code_rate(r: dict):
    """旧 json（metrics_version 缺失/=1）的 code 率回修 → (值, 是否回修过)。

    【2026-09-19 p8 事故】旧 eval 把 code_rate/code_ok_rate/avg_rounds 除以
    **题数** n_valid，而 code_used 是 [题][采样] 平铺（长度 n×val_n）→ 采样档
    显示值虚高 val_n 倍（真机 p8：401.5% 实为 50.2%）。

    回修判据**只认 metrics_version + eval_protocol.val_n**，不靠"值 >1 才像坏的"
    猜：真实 5% 的率在 val_n=8 下显示 40%，完全像个正常率却同样是错的。
    新 json（version≥2）原样返回，绝不二次缩放。"""
    if "code_rate" not in r:
        return None, False
    v = r.get("code_rate")
    if int(r.get("metrics_version", 1) or 1) >= 2:
        return v, False
    val_n = int((r.get("eval_protocol") or {}).get("val_n", 1) or 1)
    if val_n > 1:
        return v / val_n, True
    return v, False


def read_eval_result(path: str):
    """读单模型评测 json，返回**扁平**的 result dict（读不到/坏文件返回 {}）。

    【2026-09-25 事故·内嵌评测全 0】eval_vllm_one.py 落盘的是 **{name: result}**
    嵌套壳（`json.dump({name: result})`，这是全仓库的正典格式：eval_vllm.py 靠
    `results.update(json.load(f))` 合并、eval_merge.py 靠 `rows.items()` 遍历、
    summarize_eval 也按这个壳读）。而 train.py 的内嵌评测读它时直接
    `_r.get("acc", 0)` —— 键在壳里，取不到，于是**默认值 0 被当成真实读数**印进
    训练日志：

        [eval] step 50 test: acc=0.0% fmt=0.0% code=0.0% (n=0)

    评测其实跑成功了（exit=0，否则走 FAILED 分支），结果就在 step_N/eval_*.json
    里完好无损——只是训练日志和 analysis 的「评测」列在读一个不存在的层级。
    n=0 是这个 bug 的签名：真实评测永远 n>0（池空时 eval 端直接 RuntimeError
    退出，走的是 FAILED 分支而不是打印 n=0）。

    同一个壳还骗过了 p10 盲窗哨兵：嵌套壳里顶层没有 acc/n，判盲逻辑于是把**每一个
    评测成功的 checkpoint** 都标成"盲"——为看见盲窗加的列自己变成了假盲窗。

    两种壳都吃（哨兵历史上写的是扁平壳），统一出扁平 result。"""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    # 扁平壳（超时哨兵/老格式）：自带 acc 或 n 键
    if "acc" in raw or "n" in raw:
        return raw
    # 嵌套壳 {name: result}（eval_vllm_one.py 正典）：取第一个非 _meta 条目
    for k, v in raw.items():
        if not str(k).startswith("_") and isinstance(v, dict):
            return v
    return {}


def eval_is_blind(path: str) -> bool:
    """该 checkpoint 的内嵌评测是否"盲"（缺失 / 超时哨兵 / 结果不可读）。

    判据：文件不存在，或解出的 result 里 acc/n 任一为 None（哨兵签名）或缺失。
    绝不把"读不出来"当成"评测正常"——盲窗可见性的全部意义就在这。"""
    if not os.path.exists(path):
        return True
    r = read_eval_result(path)
    if not r:
        return True
    return r.get("n") is None or r.get("acc") is None


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
    # 旧 json 回修提示（一次性，表下方给出）
    _repaired = [k for k, v in models.items() if legacy_code_rate(v)[1]]
    # 【2026-10-01 效率两列】本轮目标 = "最少输出 token + 最少轮数 + 最优回答"，而 eval
    # 表此前只有 acc/fmt/code → 只能证明"更准"，无法回答"更省"（docs/14 §4.10）。
    # 口径：平均token = 模型自产 token（eval_vllm_one 的 avg_ans_tokens，assistant 段）；
    #      工具轮数 = avg_rounds（code_used 均值，与 code% 同源，非布尔化）。
    # **绝不把缺失当 0**：旧 json 没有 avg_ans_tokens（2026-10-01 才落盘）→ 必须显示
    # "—" 并在表下说清"效率目标无法判定"，否则一个恒 0 的列会被读成"模型不输出 token"。
    _tok_cells, _rnd_cells, _any_tok = [], [], False
    for _n, _r in models.items():
        _at = _r.get("avg_ans_tokens")
        if not isinstance(_at, (int, float)):
            # 回退：合并 json 丢了聚合键但 per-item 还在（老 json 的 ans_len 是 0 占位，
            # 故只在真 >0 时采用，免得把占位 0 平均成"零 token"）。
            _iv = [it.get("ans_len") for it in (_r.get("items") or [])
                   if isinstance(it.get("ans_len"), (int, float)) and it.get("ans_len")]
            _at = (sum(_iv) / len(_iv)) if _iv else None
        _ar = _r.get("avg_rounds")
        _tok_cells.append(f"{_at:.0f}" if isinstance(_at, (int, float)) else "—")
        _rnd_cells.append(f"{_ar:.2f}" if isinstance(_ar, (int, float)) else "—")
        if isinstance(_at, (int, float)):
            _any_tok = True
    lines += [f"| 模型 | acc%(±95%CI) | fmt% | code% | 平均token | 工具轮数 | Δacc vs {base_name} | 检验 | 判定 |",
              "|---|---|---|---|---|---|---|---|---|"]
    _ci = 0
    for name, r in models.items():
        n = r.get("n") or 0
        acc = r.get("acc", 0.0) * 100
        hw = ci95(r.get("acc", 0.0), n)
        acc_col = f"{acc:.1f}±{hw:.1f}" if hw == hw else f"{acc:.1f}"
        fmt_col = f"{r.get('fmt', 0.0) * 100:.1f}"
        _cr, _fixed = legacy_code_rate(r)
        code_col = f"{_cr * 100:.1f}{'*' if _fixed else ''}" if _cr is not None else "—"
        if name == base_name or not base or not n or not n_base:
            delta, test, verdict = "—", "—", "—"
        else:
            d, h = diff_ci95(r.get("acc", 0.0), n, base.get("acc", 0.0), n_base)
            # 【2026-09-19】按 items 口径自动分派：greedy→McNemar，采样档
            # （Average@N 小数 acc）→配对均值 z 检验。旧版无条件 McNemar 会把
            # "N 条全对率之差"报成 acc 之差（p8 step100 -4.6pp p=0.003 实为全对率）。
            pt = paired_test_auto(r.get("items"), base.get("items"))
            if pt is not None:
                d_paired, p, matched, test = pt
                # 配对均值检验直接给出配对 Δ（比未配对两比例差更准），优先采用
                if d_paired is not None:
                    d = d_paired
                delta = f"{d:+.1f}pp"
                verdict = _verdict(d, h, p)
            else:
                delta = f"{d:+.1f}±{h:.1f}pp"
                test = "两比例（无 per-item）"
                verdict = _verdict(d, h)
        lines.append(f"| {name} | {acc_col} | {fmt_col} | {code_col} | "
                     f"{_tok_cells[_ci]} | {_rnd_cells[_ci]} | {delta} | {test} | {verdict} |")
        _ci += 1
    if models and not _any_tok:
        lines += ["",
                  "> ⚠️ 本表**没有任何模型带 `avg_ans_tokens`**（该字段 2026-10-01 才在 "
                  "`eval_vllm_one.py` 落盘）→ 平均token 列恒为 “—”，"
                  "**效率目标（最少输出 token）无法判定**。两条修法："
                  "① 用新代码重跑 eval（per-item 会带 `ans_len`/`clen`）；"
                  "② 先看 record 表的 `avg_clen`——但它是**含工具回包**的 completion 全长，"
                  "与模型自产 token 不是同一口径，不能直接当输出 token 用。"]
    if _repaired:
        _vn = int((models[_repaired[0]].get("eval_protocol") or {}).get("val_n", 1) or 1)
        lines += ["",
                  f"> `*` code% 已按旧口径回修（原值除以 val_n={_vn}）："
                  f"旧 eval 把代码率除以题数而非轨迹数（n×val_n），采样档虚高 {_vn} 倍。"
                  f"受影响模型：{', '.join(_repaired)}。",
                  "> ⚠️ 这些旧 json 的 **per-item `code_used` 无法回修**"
                  "（当年只写入了前 N 条轨迹，题 N/val_n.. 全缺）→ "
                  "`--code-layer` / `--code-migration` 需用新代码重跑 eval 才可信。"]
    return "\n".join(lines)


def _pair_col(pairs):
    """pairs = [(ref_item, target_item), ...] 同题对 → 'target acc / ref acc' 字符串。

    同题集上两个模型的 acc 直接可读：如 '62.3% (137/220) vs B 73.2% (161/220)' 表示
    这 220 题上 target 62.3%、BASE 73.2%——放弃代码的代价一目了然。

    【2026-09-19】acc 用求和而非 ==1.0 计数：采样档 per-item 是 Average@N 小数，
    二值化会把"8 条里对 7 条"记成 0（系统性低估两臂 acc）。"""
    if not pairs:
        return "—"
    n = len(pairs)
    t_ok = sum(float(t.get("acc", 0)) for _, t in pairs)
    r_ok = sum(float(r.get("acc", 0)) for r, _ in pairs)
    return (f"{t_ok / n * 100:.1f}% ({t_ok:g}/{n})"
            f" vs B {r_ok / n * 100:.1f}% ({r_ok:g}/{n})")


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
        # 【2026-09-19】同 summarize_eval：按口径分派 McNemar / 配对均值 z 检验
        pt = paired_test_auto(ma.get("items"), mb.get("items"))
        if pt is not None:
            _d_paired, p, _matched, test = pt
            if _d_paired is not None:
                d = _d_paired
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


def record_clen_cap(path: str, default: int = 1800) -> tuple:
    """record.jsonl 同目录的 run_info.json 推导 clen 上限（纯函数，CPU 可测）。

    返回 (clen_cap, 来源说明)。clen 的语义是**整条轨迹 completion 全长**
    （rollout.py 的 per_sample_ids = 各段 ids 拼接，含工具段），它的物理上限是
    `max_context_tokens − max_prompt_length`（retool_context_overlong 的丢弃线
    减去 prompt 占用），而不是任何单轮预算。

    【2026-09-20 为什么必须读 run_info】旧版把 clen_cap 硬编码成 1800（2200−400
    的 3B 时代值），而 p8 的真实上限是 26400−1024 = 25376 —— 差 14 倍。后果：
    "≥0.9×cap" 那一列按 1620 判，p8 报表里显示 68%~98% 的样本"接近上限"，
    **全部是饱和噪声**，且极易被读成"长度顶满预算"。而 ts0/ts0.5 单变量对照的
    核心判据正是长度/截断轴，这一列不可信就等于判据不可信。
    docs/05 §7.5.3 声称过这个修法，但代码里一直没有落地（调用点也不传该参数）。
    """
    info_path = os.path.join(os.path.dirname(os.path.abspath(path)), "run_info.json")
    try:
        with open(info_path, encoding="utf-8") as f:
            cfg = (json.load(f) or {}).get("config") or {}
        # 【2026-09-29 token 预算档】该档下轨迹的物理上限是 max_traj_tokens（循环内
        # 逐样本夹 max_tokens + 预算耗尽即出局），比 ctx−plen 更紧也更准；用它读
        # "≥0.9×cap" 那一列才不会把"预算刚好用满"误读成"还没到丢弃线"。
        mtj = int(cfg.get("max_traj_tokens") or 0)
        if mtj > 0:
            return mtj, f"run_info(max_traj_tokens={mtj})"
        ctx = int(cfg.get("max_context_tokens") or 0)
        plen = int(cfg.get("max_prompt_length") or 0)
        if ctx > 0 and ctx - plen > 0:
            return ctx - plen, f"run_info({ctx}−{plen})"
    except (OSError, ValueError, TypeError):
        pass
    return default, f"默认{default}（无 run_info，口径存疑）"


def record_family_of(rec: dict) -> str:
    """record 单行归属族：\"ok\"（已上传训练）/\"dropped\"（零方差丢弃）。纯函数。

    【2026-09-28 为什么需要】自 2026-09-23 起 uniform（全对/全错，组内归一化后
    adv 恒 0）组**也落盘**（rollout.py 的 q_status 字段）。于是 record.jsonl 里的
    acc/fmt/trunc 是**两个分布混在一起**的：丢弃组结构性全错（reward -1、无 boxed），
    它的 acc≈0、invalid/trunc 高发，会把 OK 组的真实读数一路拖低。native_p3 实测：
    混合口径 acc=18.5% / fmt=20.0%，反推 ok 组是 acc≈60% / fmt≈65%——混读会把
    "ok 组质量正常"误读成"模型学不会"（交接文档 §3 头号判据就是分族对比）。

    旧 record（2026-09-23 前）没有 q_status 字段——当年丢弃组不落盘，**每一行
    都是上传组**，故缺字段按 "ok" 计：本函数对旧文件的分类与旧行为逐位一致。"""
    qs = rec.get("q_status")
    if qs is None:
        return "ok"
    return "ok" if qs == "ok" else "dropped"


def read_record(path: str) -> dict:
    """读 record.jsonl → 逐样本扁平数组 + 会话切分（纯函数，CPU 可测）。

    【为什么提成独立函数】`summarize_record`（曲线/分族表）与
    `summarize_no_boxed`（无 boxed 归因分解）**必须逐位同源**地切会话——切会话
    判据本身是踩过两次真机坑的（见下），两份实现一旦分叉，"会话A"在两张某表里
    就是不同样本区间，跨表对照全部失效。这里返回**原始**计数，布尔化口径由调用
    方决定（acc/fmt 是 ±1 奖励，必须 `>0`；code_used/code_wasted 是计数，不能布尔化）。

    返回键（全部按**样本**下标对齐，长度 = 落盘样本总数）：
      accs/fmts/codes/oks: list[bool]   —— 已按 `>0` 布尔化（±1 奖励口径）
      trs/cws/invs/ctxfs/cus: list[int] —— trunc_final / code_wasted /
                                           invalid_final / ctx_full / code_used
      clens: list[int]                  —— 整条轨迹 completion 全长
      segls/tsegls: list[list[int]|None] —— 该样本的 assistant / 工具段 token 数
                                           （按时间序；2026-10-02 前落盘 = None，
                                           与下标对齐，供 B 桶长度画像用）
      fams: list[str]                   —— "ok"/"dropped"（record_family_of）
      stales: list[int|None]            —— opt-step 陈旧度；丢弃族 = None 占位，
                                           无 gen_version 的文件 = 空表
      sess_ids: list[int]               —— 会话号（0 起）
      sess_span / sess_gv: dict         —— 会话 → [首, 末] 墙钟 / gen_version
      n_sess: int

    **切会话判据分档**（2026-09-17 真机定案）：
      · 有 `gen_version`（新协议）→ **只看 gen_version 回退**（新 run 从 0 重新
        计数），时间阈值放宽到 SESS_GAP_GV_S(30min) 只兜"真重启"。
        为什么不能沿用 120s：`gen_questions_per_attempt=4` 时**一次 attempt 的
        4 条记录时间戳完全相同**（4 题一次性上传），attempt 之间隔 ~3min →
        120s 判据把一次 106 组的 run 切成 **31 个"会话"**，逐会话表彻底失去意义
        （真机 bg1 的原始读数就是这个形态）。
      · 无 `gen_version`（旧协议）→ 沿用 120s 时间判据。
    """
    accs, fmts, codes, oks, trs, clens, phases, sess_ids = [], [], [], [], [], [], [], []
    cws = []             # 每样本末轮浪费的代码调用次数（2026-09-20）
    invs = []            # 每样本原生协议 invalid_final（围栏档恒 0；2026-09-28 接入）
    ctxfs = []           # 每样本 ctx_full（observation 装不下而终局；同上）
    cus = []             # 每样本 code_used（原始计数；2026-09-29 供无 boxed 分解用）
    segls = []           # 每样本 assistant 段长表（None = 该行无该字段，旧 record）
    tsegls = []          # 每样本工具段长表（同上；2026-10-02）
    fams = []            # 每样本所属族（ok/dropped，见 record_family_of；2026-09-28）
    stales = []          # 每样本 staleness（opt-step 口径，见下；无 gen_version 时为空）
    _ok_seen = 0         # 已上传（ok 族）样本累计——staleness 的 micro-step 基准
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
            # 本批第 i 个样本的 micro-step = (累计**已上传**样本数 + i)//8。
            # gen_version 是生成该批时权重对应的 micro-step。staleness 过高 =
            # 训练在吃太旧的策略数据（off-policy，框架监控建议②）。
            #
            # 【2026-09-28 伪影修复·必读】旧版这里用 `len(accs)`（**全部**已读样本）
            # 当 micro-step 基准，隐含假设"每条落盘记录都上传到训练端"。该假设被
            # 2026-09-23 的"uniform 组也落盘"**作废**：丢弃组不占 train step 却照样
            # 累加 → 基准被虚高 1/(1−丢弃率) 倍。native_p3 实锤：公式给
            # floor(7648/8/4) − floor(296/4) = 239 − 74 = **165**，与表里 max 165
            # 逐位吻合——而训练总共只推进 74 个 opt-step，**落后 165 不可能**。
            # 整列因此是伪影（真值要看训练日志的 [train][口径] 行）。
            # 修法：基准只数 ok 族样本（=真正上传的），与 train.py 的 step 同量纲。
            _fam = record_family_of(rec)
            if isinstance(gv, int):
                _gas = 4                                    # 与 config 一致
                for _i in range(n):
                    if _fam != "ok":
                        # 丢弃组不占 train step，它的"micro-step"无定义 → 记 None
                        # 占位（**必须**保持与样本下标对齐：会话行与窗口列都按
                        # 全局样本下标取切片，只给 ok 族追加会让下标整体错位）
                        stales.append(None)
                        continue
                    _m = (_ok_seen + _i) // 8               # 本样本的 micro-step
                    stales.append((_m // _gas) - (gv // _gas))
            if _fam == "ok":
                _ok_seen += n
            accs.extend(a > 0 for a in rec["acc"])
            fmts.extend(v > 0 for v in rec["fmt"])
            codes.extend(u > 0 for u in rec.get("code_used", []))
            oks.extend(k > 0 for k in rec.get("code_ok", []))
            trs.extend(int(x) for x in rec.get("trunc_final", []))
            # 【2026-09-28 原生协议诊断列接入】invalid_final / ctx_full 自 2026-09-25
            # 起就落盘，但 summarize_record 一直没读——于是交接文档的头号判据
            # （native_p3 的 invalid ≈60%）在 `--record` 表上**完全隐形**，只能手工
            # 写脚本统计。围栏档 / 旧 record 无该键 → 补 0（列显示 0.0% 而不是崩）。
            _iv = rec.get("invalid_final") or []
            invs.extend(int(x) for x in _iv)
            if len(_iv) < n:
                invs.extend([0] * (n - len(_iv)))
            _cx = rec.get("ctx_full") or []
            ctxfs.extend(int(x) for x in _cx)
            if len(_cx) < n:
                ctxfs.extend([0] * (n - len(_cx)))
            # 【2026-09-29 无 boxed 分解接入】code_used 原始计数：分解表要区分
            # "被切断但全程没调用过工具"（纯散文，提示/预算问题）与"调用前被切断"
            # ——旧表只存布尔化的 code 率（`codes`），该区分在表上不可见。
            _cu = rec.get("code_used") or []
            cus.extend(int(x) for x in _cu)
            if len(_cu) < n:
                cus.extend([0] * (n - len(_cu)))
            fams.extend([_fam] * n)
            # 末轮写代码 = 结构性无 boxed 且 trunc_final 记不到（2026-09-20）；
            # 旧 record 无该键 → 补 0，列会显示 0.0% 而不是崩
            _cw = rec.get("code_wasted") or []
            cws.extend(int(x) for x in _cw)
            if len(_cw) < n:
                cws.extend([0] * (n - len(_cw)))
            clens.extend(rec.get("clen", []))
            # 【2026-10-02 B 桶长度画像接入】segl/tsegl 是**嵌套**列表（每样本一张
            # 段长表），扁平化口径与 cus 等不同：这里必须保留嵌套结构（末段长度 =
            # 内层末位），且**缺键/短表一律补 None**——补 0 会把"旧 record 无此字段"
            # 与"该样本真没有 assistant 段"混成同一个值，画像的分母就错了。
            for _key, _dst in (("segl", segls), ("tsegl", tsegls)):
                _sv = rec.get(_key) or []
                _dst.extend([list(x) if isinstance(x, (list, tuple)) else None
                             for x in _sv])
                if len(_sv) < n:
                    _dst.extend([None] * (n - len(_sv)))
            ph = rec.get("phase")
            if ph:
                phases.extend([ph] * n)
            sess_ids.extend([sess] * n)
    return {"accs": accs, "fmts": fmts, "codes": codes, "oks": oks, "trs": trs,
            "clens": clens, "cws": cws, "invs": invs, "ctxfs": ctxfs, "cus": cus,
            "segls": segls, "tsegls": tsegls,
            "fams": fams, "stales": stales, "sess_ids": sess_ids, "phases": phases,
            "sess_span": sess_span, "sess_gv": sess_gv, "n_sess": sess + 1}


# 无 boxed 归因桶。**元组顺序 = 判定优先级 = 表列顺序**（三者必须一致，否则读表
# 的人会按列序误推优先级）。名字沿用诊断脚本里的既有叫法，便于与历史记录对照。
NO_BOXED_BUCKETS = (
    ("A_cut_mid_call", "轮长切断，且已写出调用开标记（在调用块里被截）"),
    ("C_wasted", "终局轮写了完整调用，按协议不执行（末轮废码）"),
    ("F_ctx_full", "observation 装不进预算而终局（ctx_full）"),
    ("B_cut_mid_prose", "轮长切断，且**没写出**任何调用（在散文里被截）"),
    ("D_invalid_other", "调用形态非法，但不是被切断（非截断的 invalid）"),
    ("E_clean_no_box", "干净收尾但没给 boxed（提示层/收尾问题）"),
)


def no_boxed_bucket(fmt_ok: bool, trunc: int, wasted: int, invalid: int,
                    ctx_full: int, code_used: int) -> str:
    """单个**无 boxed** 样本的失败机理归类（纯函数，CPU 可测）；有 boxed 返回 ""。

    【为什么必须有显式优先级】这些标志在数据里**会重叠**：
      · `trunc ∩ invalid` 是**主要**重叠：`trunc_final=1` 的样本里有相当一部分同时
        `invalid_final=1`（调用块写到一半被单轮上限切断——"有开标记但形态不完整"
        正是 parse_assistant 判 invalid 的判据，protocol.py 的 docstring 明说了）。
        手工统计若按朴素顺序取首个命中，**这部分会被 trunc 整额吞掉**，于是
        "在调用里被截"与"在散文里被截"这两类**处置完全不同**的样本（前者要加单轮
        额度，后者要治啰嗦）在数据里同形。
      · `trunc ∩ wasted` 是退化情形（末段被 length 切断、但切断点恰好落在
        `</tool_call>` 之后，于是仍解析成一个完整调用）。此处 **wasted 优先于
        trunc-only**：该样本确实产出了完整调用，"末轮不该调用"才是它的签名；若让
        trunc-only 先判，样本会落进 B，而 B 的定义是"**没写出任何调用**"——自相矛盾。
      · `wasted ∩ invalid` 在数据流里**不可能**同时为真：rollout 的终局分支是
        `if invalid: ... elif tool: code_wasted += 1`，两者互斥且都会终止该样本。

    ctx_full 单列成 F 而不是并进 E：它是**预算**失败（observation 进不去），与
    "干净收尾但没给框"（提示/收尾）是两回事。

    code_used 只用于子计数（B2 = 全程没调用过工具的纯散文样本），不改变桶归属。
    返回的桶名 + "B2" 由调用方各自累加（见 no_boxed_breakdown）。"""
    if fmt_ok:
        return ""
    if trunc and invalid:
        return "A_cut_mid_call"
    if wasted:
        return "C_wasted"
    if ctx_full:
        return "F_ctx_full"
    if trunc:
        return "B_cut_mid_prose"
    if invalid:
        return "D_invalid_other"
    return "E_clean_no_box"


def _credit_enabled(path: str) -> tuple:
    """该 record 对应 run 启用了哪几项局部负优势：返回 (waste_credit, trunc_credit)。

    旧 run 缺键 → 都是 False（历史口径：trunc OR code_wasted 整行 sw=0）。
    【2026-10-01】trunc_tail_penalty>0 后截断轨迹重新进 loss，零梯度人群随之缩小，
    判读必须按 run_info 分档，否则会把"已有局部信用"记成"零梯度算力"。
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(path)),
                               "run_info.json"), encoding="utf-8") as f:
            cfg = (json.load(f) or {}).get("config") or {}
        return (float(cfg.get("tool_waste_penalty", 0.0) or 0.0) > 0.0,
                float(cfg.get("trunc_tail_penalty", 0.0) or 0.0) > 0.0)
    except (OSError, ValueError, TypeError):
        return (False, False)


def record_run_config(path: str) -> dict:
    """record 同目录 run_info.json 的 config（缺失/坏文件 → {}，绝不抛给调用方）。

    【与 `_credit_enabled` 的分工】后者把两个信用开关判成布尔；本函数给的是原始
    config——B 桶长度画像要用 `answer_reserve`/`max_traj_tokens` 的**数值**（把
    "末段长度"与"作答预留"放在同一把尺子上比，正是画像的全部意义）。"""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(path)),
                               "run_info.json"), encoding="utf-8") as f:
            return (json.load(f) or {}).get("config") or {}
    except (OSError, ValueError, TypeError):
        return {}


# 末段长度分箱（上界开区间；None = 无上界）。第一箱 512 与 config.BASE 的
# round_gen_tokens 默认无关系，取的是"一次简短作答"的量级；`answer_reserve`
# 默认 1024 恰好是第一/第二箱的分界，故 1024~2048 箱就是"预留不够一点点"的形态。
SEG_LEN_BINS = ((0, 512), (512, 1024), (1024, 2048), (2048, None))
SEG_LEN_BIN_LABELS = ("<512", "512~1024", "1024~2048", "≥2048")
# 画像的行顺序 = 读表顺序：先给可控标尺（成功作答真要多少 token），再给待判桶
B_PROSE_GROUPS = (
    ("boxed", "有boxed（成功收尾·标尺）"),
    ("B_tool", "B（调过工具）"),
    ("B2", "B2（全程零调用）"),
    ("C_wasted", "C_wasted"),
    ("DE", "D+E（其他无boxed）"),
)


def _seg_pct(vals, q: float):
    """线性插值分位（与 numpy 默认口径一致）；空表 → None。纯函数。"""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return float(s[lo]) if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo)


def b_prose_profile(path: str) -> dict:
    """ok 族按**末段 assistant 长度**画像（纯函数，CPU 可测；2026-10-02）。

    回答 native_p4_trunc 遗留的悬案：`B_cut_mid_prose`（预算耗尽、在散文里被切）
    该抬 `answer_reserve`，还是该治"写散文不收尾"？

      · B 的末段长度集中在 reserve 附近（~1024）→ 模型最后一轮**在答题**、被预留
        额度掐死 → 抬 reserve 是对症的；
      · B 的末段是几千 token 的长尾（或零调用的 B2 占多数）→ 模型**在写散文**，
        抬 reserve 只是更早撞墙：B 变 C，而两者都无 boxed、reward 同为 −1，
        "无 boxed 率"看不出任何变化（必须看结构比）。

    **对照标尺** = 有 boxed 样本的末段长度：那是"一次成功作答实际要多少 token"
    的实测值。把 B 的 p50 与它并列，reserve 够不够就不再靠猜。

    分母纪律：只算 ok 族（丢弃族结构性全错，"成功作答"标尺在它身上无定义）；
    每条样本的 `near_cap` 判定用 `record_clen_cap`（token 档 = max_traj_tokens）
    ——B 在 token 档下**必然** `clen≥0.9cap`（`trunc_final` 只在预算耗尽时置位），
    该列就是这条口径的自证；另 `tool_*` 是回包的真实 token 占用（回包到底吃掉
    多少预算，只有它给的数不是估算）。

    返回 {has_segl, n_unknown_field, cap, cap_src, reserve, max_traj_tokens,
          tool_n/tool_p50/tool_p90, groups: {键: {label,n,n_len,unknown,bins,
          near_cap,p25,p50,p75}}}；旧 record（无 segl）= has_segl False。"""
    R = read_record(path)
    cfg = record_run_config(path)
    cap, cap_src = record_clen_cap(path)
    g = {k: {"n": 0, "lens": [], "unknown": 0, "near_cap": 0}
         for k, _lab in B_PROSE_GROUPS}
    tool_lens = []
    n_unknown_field = 0
    for i, fam in enumerate(R["fams"]):
        if fam != "ok":
            continue
        sl, tl = R["segls"][i], R["tsegls"][i]
        if tl:
            tool_lens.extend(int(x) for x in tl)
        if sl is None:
            n_unknown_field += 1
        bk = no_boxed_bucket(R["fmts"][i], R["trs"][i], R["cws"][i],
                             R["invs"][i], R["ctxfs"][i], R["cus"][i])
        if not bk:
            key = "boxed"
        elif bk == "B_cut_mid_prose":
            key = "B2" if R["cus"][i] == 0 else "B_tool"
        elif bk == "C_wasted":
            key = "C_wasted"
        else:
            key = "DE"
        d = g[key]
        d["n"] += 1
        if int(R["clens"][i]) >= 0.9 * cap:
            d["near_cap"] += 1
        if sl:
            d["lens"].append(int(sl[-1]))
        else:
            d["unknown"] += 1
    out = {"has_segl": any(x is not None for x in R["segls"]),
           "n_unknown_field": n_unknown_field, "cap": cap, "cap_src": cap_src,
           "reserve": int(cfg.get("answer_reserve", 0) or 0),
           "max_traj_tokens": int(cfg.get("max_traj_tokens", 0) or 0),
           "tool_n": len(tool_lens), "tool_p50": _seg_pct(tool_lens, 0.5),
           "tool_p90": _seg_pct(tool_lens, 0.9), "groups": {}}
    for k, label in B_PROSE_GROUPS:
        d = g[k]
        if not d["n"]:
            continue
        bins = [0] * len(SEG_LEN_BINS)
        for v in d["lens"]:
            for bi, (lo, hi) in enumerate(SEG_LEN_BINS):
                if v >= lo and (hi is None or v < hi):
                    bins[bi] += 1
                    break
        out["groups"][k] = {"label": label, "n": d["n"], "n_len": len(d["lens"]),
                            "unknown": d["unknown"], "bins": bins,
                            "near_cap": d["near_cap"],
                            "p25": _seg_pct(d["lens"], 0.25),
                            "p50": _seg_pct(d["lens"], 0.5),
                            "p75": _seg_pct(d["lens"], 0.75)}
    return out


def no_boxed_breakdown(path: str) -> dict:
    """无 boxed 样本的**分族 × 机理**分解（纯函数，CPU 可测）。

    返回：{n_samples, n_nobox, families: {族: {n, nobox, buckets: {桶: 计数},
                                             B2_pure_prose, zero_grad}},
           n_nobox_total, zero_grad_total}

    `zero_grad` = 该族里 `trunc_final or code_wasted` 的样本数（**不看有没有
    boxed**）——这是 F1（commit 0ffec5e）之后 sample_weight=0 的人群，即"既不贡献
    策略梯度、也不进组基线"的零梯度算力。它与"无 boxed"不是一回事：一条轨迹可以
    既有 boxed 又在末段被切断（则它有梯度但不完整），两部分在表上分开给。

    【为什么值得单列成命令】native_p3 的实测（576 样本 / 332 无 boxed）：
      ok 族 99 条无 boxed 里 C_wasted 占 71.7%（差一点的轨迹输在"末轮又调工具"），
      dropped 族 233 条里 A_cut_mid_call 41.6% + C_wasted 40.8%（硬题两头顶死），
      全局 E_clean_no_box 只有 1.3%——**"模型不会收尾"这一整类假设被这一个数否掉**
      （此前按"无 boxed ≈ 截断 + ≤6.9pp"估的残余 ~17pp 高了约 7 倍）。
    这类判读以前每次都靠临时 heredoc 重算，既慢又踩过上面的重叠陷阱（首版脚本把
    trunc∩invalid 全算进 trunc），故固化成子命令 + 测试。"""
    R = read_record(path)
    waste_credit, trunc_credit = _credit_enabled(path)
    n_tot = len(R["accs"])
    fams = R["fams"]
    res = {"n_samples": n_tot, "n_nobox": 0, "n_nobox_total": 0,
           "zero_grad_total": 0, "waste_credit": waste_credit,
           "trunc_credit": trunc_credit, "families": {}}
    for fam in ("ok", "dropped"):
        idx = [i for i, v in enumerate(fams) if v == fam]
        if not idx:
            continue
        buckets = collections.Counter()
        b2 = 0
        nobox = 0
        zg = 0
        for i in idx:
            _tr, _wa = R["trs"][i], R["cws"][i]
            _iv, _cx = R["invs"][i], R["ctxfs"][i]
            if (_tr and not trunc_credit) or (_wa and not waste_credit):
                zg += 1
            bk = no_boxed_bucket(R["fmts"][i], _tr, _wa, _iv, _cx, R["cus"][i])
            if not bk:
                continue
            nobox += 1
            buckets[bk] += 1
            if bk == "B_cut_mid_prose" and R["cus"][i] == 0:
                b2 += 1
        res["families"][fam] = {"n": len(idx), "nobox": nobox,
                                "buckets": dict(buckets),
                                "B2_pure_prose": b2, "zero_grad": zg}
        res["n_nobox_total"] += nobox
        res["zero_grad_total"] += zg
    res["n_nobox"] = res["n_nobox_total"]
    return res


def summarize_no_boxed(path: str) -> str:
    """`no_boxed_breakdown` 的 Markdown 渲染（分族 × 机理表 + 判据图例）。

    【读数纪律】分母是**落盘**样本；overlong 整组不落盘（幸存者偏差），故"无 boxed
    占比"不是全部轨迹的无 boxed 率——含超长的真值看生成端 `[rollout] 采样统计` 行。
    各桶占比的分母是**该族无 boxed 条数**（不是该族全部样本），因为问题是"失败的那
    些是怎么死的"。zero_grad 单独一行，分母是该族全部样本（sw=0 判定与有无 boxed
    无关）。"""
    B = no_boxed_breakdown(path)
    if not B["n_samples"]:
        return f"> {path} 里没有可统计的 record 行（acc 为空或文件不存在）。"
    names = [k for k, _ in NO_BOXED_BUCKETS]
    out = ["== 无 boxed 归因分解（分族 × 机理）==",
           f"> 分母 = 落盘样本 {B['n_samples']}（ok 族 + 丢弃族）；overlong 整组"
           f"**不落盘**（幸存者偏差），非全部轨迹的无 boxed 率。",
           f"> 桶判定按优先级 {'→'.join(n[0] for n in NO_BOXED_BUCKETS)}，"
           f"同一格只进一个桶（**trunc ∩ invalid 归 A**，不重复计数——"
           f"朴素顺序会让 A 全被 B 吞掉）。",
           "",
           "| 族 | 样本 | 无boxed | 占比 | " + " | ".join(names) + " |",
           "|---" * (4 + len(names)) + "|"]
    for fam, label in (("ok", "ok（已上传）"), ("dropped", "dropped（丢弃）")):
        f = B["families"].get(fam)
        if not f:
            continue
        nb = f["nobox"]
        cells = []
        for k in names:
            c = f["buckets"].get(k, 0)
            cells.append(f"{c}（{c / nb * 100:.0f}%）" if nb else "0")
        out.append(f"| {label} | {f['n']} | {nb} | {nb / f['n'] * 100:.0f}% | "
                   + " | ".join(cells) + " |")
    out.append("")
    if B.get("trunc_credit") and B.get("waste_credit"):
        _zg_rule = "trunc/废调用都已有局部负优势"
    elif B.get("trunc_credit"):
        _zg_rule = "trunc 已有末段局部负优势；末轮废码 → sw=0"
    elif B.get("waste_credit"):
        _zg_rule = "trunc → sw=0；废调用已有局部负优势"
    else:
        _zg_rule = "trunc ∪ 末轮废码 → sw=0"
    out.append(f"| 族 | 零梯度占比（{_zg_rule}） | 其中 B2 纯散文"
               "（被切断且全程没调用过工具） |")
    out.append("|---|---|---|")
    for fam, label in (("ok", "ok（已上传）"), ("dropped", "dropped（丢弃）")):
        f = B["families"].get(fam)
        if not f:
            continue
        out.append(f"| {label} | {f['zero_grad']}/{f['n']} "
                   f"（{f['zero_grad'] / f['n'] * 100:.0f}%） | {f['B2_pure_prose']} |")
    out.append("")
    # 【2026-10-02】B 桶末段长度画像：把"抬 answer_reserve"与"治啰嗦"分开的唯一
    # 读数（只有整条 clen 时，两者在数据里同形）。
    P = b_prose_profile(path)
    _res_txt = (f"answer_reserve={P['reserve']}" if P["reserve"]
                else "answer_reserve 未记录")
    out.append(f"== B 桶末段长度画像（该抬 {_res_txt}，还是该治啰嗦）==")
    if not P["has_segl"]:
        out.append("> 本 record 无 `segl`/`tsegl` 字段（2026-10-02 前落盘）→ 无法画像，"
                   "只能按 clen/trunc 两列间接推断；重新 rollout 一次即有。")
    else:
        out.append("> 只统计 ok 族（丢弃族结构性全错，「成功作答」标尺在它身上无定义）；"
                   "末段 = 该样本**最后一段** assistant 的 token 数。")
        out.append("> 读法：B 行落在 `≥2048`（或 B2 零调用占多数）→ 模型是把预算写成"
                   "散文，抬 reserve 只会把 B 换成 C（两者都无 boxed、reward 同为 −1，"
                   "「无 boxed 率」看不出变化）；B 行集中在 reserve 附近、且与"
                   "「有boxed」的 p50 同量级 → 作答预留确实不够，才该抬 reserve。")
        out.append("")
        out.append("| 组 | 样本 | 末段 p25/p50/p75 | "
                   + " | ".join(SEG_LEN_BIN_LABELS) + " | 未知 | clen≥0.9cap |")
        out.append("|---" * 9 + "|")
        for k, _lab in B_PROSE_GROUPS:
            d = P["groups"].get(k)
            if not d:
                continue
            _p = ("—" if d["p50"] is None
                  else f"{d['p25']:.0f}/{d['p50']:.0f}/{d['p75']:.0f}")
            out.append(f"| {d['label']} | {d['n']} | {_p} | "
                       + " | ".join(str(c) for c in d["bins"])
                       + f" | {d['unknown']} | {d['near_cap']}/{d['n']} |")
        if P["tool_n"]:
            out.append("")
            out.append(f"> 工具段（回包）实测 token：n={P['tool_n']} "
                       f"p50={P['tool_p50']:.0f} p90={P['tool_p90']:.0f}"
                       "（`tool_result_max_chars` 是字符口径的估算，这里是真实占用）；"
                       f"clen 上限口径 = {P['cap_src']} = {P['cap']}。")
    out.append("")
    out.append("桶含义（A/B 之分是「处置不同」：A 要加**单轮额度**，"
               "B 要治**啰嗦**）：")
    for k, desc in NO_BOXED_BUCKETS:
        out.append(f"  · `{k}` — {desc}")
    out.append("  · `B2` — B 的子集：全程 `code_used==0`，即从没调用过工具"
               "（纯散文一路写到底被截，与工具协议无关）。")
    return "\n".join(out)


def summarize_record(path: str, window: int = 160, clen_cap: int = None) -> str:
    """按 upload 批次滑动平均 acc/fmt(code/code_ok/trunc) 率与完成长度（retool 诊断用）。

    clen_cap: None = 从 record 同目录的 run_info.json 推导（见 record_clen_cap，
    = max_context_tokens − max_prompt_length）；显式传值则覆盖。接近上限说明轨迹
    在撞上下文预算（会被整组丢弃或标签被截断）。**注意这条线是"全轨迹预算"，
    末段被单轮上限切断是另一回事，看 trunc 列（retool_trunc 签名）。**

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
    if clen_cap is None:
        clen_cap, _cap_src = record_clen_cap(path)
    else:
        _cap_src = f"显式传入{clen_cap}"
    # 【2026-09-24 盲窗可见性】推导本 record 目录下训练存档点：内嵌评测结果
    # step_N/eval_test.json 缺失或为超时哨兵（acc=None）的 checkpoint 在曲线表上
    # 标 "盲"，把"没评测"摆到明处（p10 8 路全 TIMEOUT 的盲窗教训）。
    _dir = os.path.dirname(os.path.abspath(path))
    _ckpt_steps = []
    try:
        _save = None
        with open(os.path.join(_dir, "run_info.json"), encoding="utf-8") as _f:
            _ri = json.load(_f)
        _save = _ri.get("save_steps")
        if _save and int(_save) > 0:
            _all = int(_ri.get("all_steps", 0) or 0) or _save
            _ckpt_steps = sorted({s for s in range(int(_save), _all + 1, int(_save))})
    except (OSError, ValueError, TypeError):
        _ckpt_steps = []
    # 【2026-09-29 读数与分解同源】解析与切会话提为 read_record（纯函数）：无 boxed
    # 分解表必须与本表的"会话A"逐位落在同一批样本上，两份实现分叉 = 跨表对照失效。
    _R = read_record(path)
    accs, fmts, codes, oks = _R["accs"], _R["fmts"], _R["codes"], _R["oks"]
    trs, clens, phases, sess_ids = _R["trs"], _R["clens"], _R["phases"], _R["sess_ids"]
    cws, invs, ctxfs = _R["cws"], _R["invs"], _R["ctxfs"]
    cus = _R["cus"]          # 每样本 code_used 原始计数（2026-10-01 起用于 avg工具轮）
    fams, stales = _R["fams"], _R["stales"]
    sess_span, sess_gv = _R["sess_span"], _R["sess_gv"]
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
                # None = 丢弃组占位（无 micro-step 语义），统计时必须剔除
                _ss = [stales[i] for i in idx if i < len(stales) and stales[i] is not None]
                if _ss:
                    st_col = f" staleness均值={sum(_ss) / len(_ss):.1f}(max {max(_ss)})"
            # 【2026-09-28】该会话的 ok 族单独读数：acc/fmt 混合口径会被丢弃组
            # （结构性全错）拖低，判"模型学得怎么样"必须看 ok 族。无丢弃组的
            # 会话（旧协议 / 丢弃不落盘时代）不追加，会话行逐字不变。
            ok_col_s = ""
            _iok = [i for i in idx if fams and i < len(fams) and fams[i] == "ok"]
            if _iok and len(_iok) < len(idx):
                _oa = sum(accs[i] for i in _iok) / len(_iok) * 100
                _of = sum(fmts[i] for i in _iok) / len(_iok) * 100
                _oc = f"{_oa / _of * 100:.1f}%" if _of > 0 else "—"
                ok_col_s = (f" ｜ok族({len(_iok)}条): acc={_oa:.1f}% fmt={_of:.1f}%"
                            f" 条件精度={_oc}")
            sess_lines.append(
                f"会话{_sess_label(s)}(#{s}): 样本{lo}~{hi}（{len(idx)}条 ≈{len(idx)/8:.0f}组）"
                f" acc={a:.1f}% fmt={ff:.1f}% 条件精度={cond} code_ok={k:.1f}% trunc={tr:.1f}%"
                f"{gv_col}{st_col}{ok_col_s}{when}")
    out = [f"> clen 上限口径: {_cap_src} → cap={clen_cap}，"
           f"「≥{int(0.9 * clen_cap)}」列 = 接近**全轨迹**预算（撞它会被整组丢弃）；"
           f"末段被单轮上限切断请看 trunc 列。"
           f"注意 `avg_clen` 是**含工具回包**的 completion 全长，"
           f"不是模型输出 token（后者看 eval 表的「平均token」）。",
           "",
           "| 样本窗口 | ≈组 | acc率 | fmt率 | 条件精度 | code率 | code_ok率 | trunc率 | invalid率 | ctx满率 | 末轮废码率 | avg_clen | avg工具轮 | staleness | 阶段 | 会话 | 评测 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
        chunk_w = cws[i:j]
        chunk_i, chunk_x = invs[i:j], ctxfs[i:j]
        if not chunk_a:
            continue
        code_col = f"{sum(chunk_c) / len(chunk_c) * 100:.1f}%" if chunk_c else "—"
        ok_col = f"{sum(chunk_k) / len(chunk_k) * 100:.1f}%" if chunk_k else "—"
        tr_col = f"{sum(chunk_t) / len(chunk_t) * 100:.1f}%" if chunk_t else "—"
        # 【2026-09-28】原生协议两列：invalid = 有 <tool_call> 但形态不认识/调用后
        # 还跟内容；ctx满 = observation 放不下预算而终局。两者都是"终局无 boxed"
        # 的可归因入口，且与"啰嗦跑飞"在 acc/code 两列里完全同形——不单列就分不开。
        inv_col = f"{sum(chunk_i) / len(chunk_i) * 100:.1f}%" if chunk_i else "—"
        ctx_col = f"{sum(chunk_x) / len(chunk_x) * 100:.1f}%" if chunk_x else "—"
        # 末轮浪费代码率：与 trunc 互补，两者相加≈"没产出 boxed"的可归因部分
        wst_col = (f"{sum(1 for x in chunk_w if x > 0) / len(chunk_w) * 100:.1f}%"
                   if chunk_w else "—")
        if chunk_l:
            avg_l = sum(chunk_l) / len(chunk_l)
            near = sum(1 for l in chunk_l if l >= 0.9 * clen_cap) / len(chunk_l)
            len_col = f"{avg_l:.0f}（{near * 100:.0f}%≥{int(0.9 * clen_cap)}）"
        else:
            len_col = "—"
        # 【2026-10-01 平均工具轮数列】目标含"最少轮数"，而表上此前只有长度没有轮数
        # ——"省 token 是靠少调用还是靠写短"分不开。口径 = 每样本 code_used 的算术均值
        # （**原始计数**，不能像 code率 那样布尔化：那会丢掉"调了几次"的信息）。
        # 旧 record 无 code_used 键 → read_record 补了 0 占位，但那是"没有该字段"而非
        # "真的 0 次"。可算性判据与 code率 同源（chunk_c 非空 = 该窗口确实落了该键），
        # 免得在旧 record 上打印一个"平均 0.00 轮"的假读数。
        _ch_u = cus[i:j] if chunk_c else []
        rnd_col = f"{sum(_ch_u) / len(_ch_u):.2f}" if _ch_u else "—"
        # 【2026-09-18 staleness 列】窗口内样本陈旧度的均值/最大（opt-step 口径）。
        # >0 表示窗口内有样本吃到比推送周期更旧的策略（off-policy）；无 gen_version
        # 的旧 record 显示 "—"。均值反映"典型吃多旧"，最大反映"最坏吃多旧"。
        if stales and i < len(stales):
            # 窗口内剔除丢弃组占位（None）：它们的 micro-step 无定义，混进来会把
            # 均值拉向"看起来更旧"（2026-09-28 伪影修复，见读取处注释）
            _ch_s = [v for v in stales[i:j] if v is not None]
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
        # 【2026-09-24 盲窗可见性】该窗口覆盖的样本区间若有 checkpoint（save_steps
        # 间隔内第一个窗口），且其内嵌评测缺失或为超时哨兵（acc=None）→ 标 "盲"。
        # p10 的 8 路内嵌评测全 TIMEOUT 且无任何提示，训练侧盲跑 30h——本列把
        # "没评测"摆在表上，不再让盲窗隐形。
        _bl = "—"
        if _ckpt_steps and sess_ids and i < len(sess_ids):
            _w_samp = [i, j - 1]
            for _cs in _ckpt_steps:
                _cs_samp = (_cs - 1) * 8
                if _cs_samp < _w_samp[0] or _cs_samp > _w_samp[1]:
                    continue
                _es = os.path.join(_dir, f"step_{_cs}", "eval_test.json")
                # 【2026-09-25】经 read_eval_result 解壳再判：旧版直接读顶层
                # acc/n，对 eval_vllm_one.py 的 {name: result} 嵌套壳恒为 None
                # → 每个评测成功的 checkpoint 都被误标"盲"。
                if eval_is_blind(_es):
                    _bl = "盲"
                break
        out.append(f"| {i}~{j} | {i // 8}~{j // 8} "
                   f"| {_a_rate * 100:.1f}% "
                   f"| {_f_rate * 100:.1f}% | {cond_col} | {code_col} "
                   f"| {ok_col} | {tr_col} | {inv_col} | {ctx_col} | {wst_col} "
                   f"| {len_col} | {rnd_col} | {stal_col} "
                   f"| {ph_col} | {sess_col} | {_bl} |")
    # 【2026-09-28 分族统计】上表的每一列都是 ok+dropped **混合**的（丢弃组结构性
    # 全错，会把 ok 组读数一路拖低）。分族是唯一能读出"模型到底学得怎么样"的口径，
    # 也是交接文档 §3 头号判据（ok 组 acc 53.8% vs 丢弃组 2.7%）的自动化落地。
    # 只在真有 dropped 族时输出，旧 record（丢弃组不落盘）表体逐位不变。
    if fams and "dropped" in fams:
        _idx_ok = [k for k, fm in enumerate(fams) if fm == "ok"]
        _idx_dr = [k for k, fm in enumerate(fams) if fm == "dropped"]
        out.append("")
        out.append("== 分族统计（ok=已上传训练 / dropped=零方差丢弃；混读会把 ok 组读数拖低）==")
        out.append("| 族 | 条数 | ≈组 | acc率 | fmt率 | 条件精度 | code率 | code_ok率 | trunc率 | invalid率 | ctx满率 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _nm, _ix in (("ok", _idx_ok), ("dropped", _idx_dr)):
            if not _ix:
                continue
            _f = lambda _lst: (sum(_lst[k] for k in _ix) / len(_ix) * 100)
            _a = _f(accs); _fm = _f(fmts)
            _cond = f"{_a / _fm * 100:.1f}%" if _fm > 0 else "—"
            out.append(f"| {_nm} | {len(_ix)} | {len(_ix) / 8:.0f} | {_a:.1f}% "
                       f"| {_fm:.1f}% | {_cond} | {_f(codes):.1f}% | {_f(oks):.1f}% "
                       f"| {_f(trs):.1f}% | {_f(invs):.1f}% | {_f(ctxfs):.1f}% |")
        # 丢弃率（组口径）：ok 组数 / 总组数。overlong 整组不落盘（幸存者偏差），
        # 故此值 = "零方差丢弃占已落盘组"的比例，不是全部丢弃率（真值看生成端日志
        # 的 [rollout] 采样统计行）。
        _n_tot = len(_idx_ok) + len(_idx_dr)
        if _n_tot:
            out.append("")
            out.append(f"> 零方差丢弃占已落盘组 **{len(_idx_dr) / _n_tot * 100:.0f}%**"
                       f"（{len(_idx_dr)}/{_n_tot} 组）——注意 overlong 整组**不落盘**，"
                       f"故这不是全部丢弃率；含超长的真值看生成端 `[rollout] 采样统计` 行。"
                       f"ok 组数 ≈ 训练 micro-step 数，可与日志 step 数交叉核对。")
    if sess_lines:
        out.append("")
        out.append(f"== 会话拆分（新协议=gen_version 回退；旧协议=>{SESS_GAP_S:.0f}s 间隔）==")
        out.extend(sess_lines)
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--no-boxed-breakdown", default=None, metavar="RECORD",
                    help="无 boxed 归因分解：按 ok/dropped 分族，把『终局无 boxed』的样本"
                         "拆成 被截在调用里(A)/被截在散文里(B,含全程没调用过的 B2)/"
                         "预算装不下(F)/末轮废码(C)/形态非法(D)/干净收尾未给框(E)，"
                         "并给出零梯度占比（trunc ∪ 末轮废码 → sw=0）。"
                         "填 record.jsonl 路径（与 --record 同一个文件）")
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
    if args.no_boxed_breakdown:
        print(summarize_no_boxed(args.no_boxed_breakdown))
    if (not args.eval_json and not args.record and not args.code_layer
            and not args.pair_json and not args.no_boxed_breakdown):
        cands = sorted(glob.glob("eval_vllm_all*.json"))
        if cands:
            print(summarize_eval(cands[-1], args.base))
        else:
            print(" nothing to summarize（--eval-json / --record）")
