# -*- coding: utf-8 -*-
"""rlab/audit_config.py — 配置体检：把「我以为开了」变成「实测在起作用」。

【为什么必须有这个工具（2026-10-20）】本项目反复出同一类事故：**配置全对、
接线断开**。已发生的实例：
  · `--trunc_tail_penalty 0.2` 与 `--trunc_counts_as_wrong` 同开 → 被
    `validate_guard_conflicts` 静默归零（覆盖语义会削掉一半负信用）；
  · `sw` 判据 `(trunc and ttp<=0)` 在 ttp 被归零后恒真 → tcw 算出的 −1.75
    被乘 0，21% 样本零梯度（p11c 实测 563/2696）；
  · `overlong_ref=12288 + buffer=256` → trigger 12032 贴着保险丝，模型几乎
    不可能落在 256 token 宽的受罚区 → 惩罚在真实分布上是死开关；
  · `code_shaping_once=True` + code 率 98% → 全组同值 → 被组均值减掉 →
    零梯度（这个是对的，但**必须知道它此刻无效**，否则会误以为在训练效率）。
共同点：**每个组件单独看都对，接口处全错**。逐个开关推理抓不到，必须把
"真实 cfg → 真实打分层 → 真实 advantage"串起来算一遍。

【用法】
    python -m rlab.audit_config --from_run_info rlab_new/native_p11d/run_info.json
    python -m rlab.audit_config --from_run_info <dir> --json
    python -m rlab.audit_config --algo retool_math --max_traj_tokens 12288 \
        --budget_target 8192 --task_adv_guard --trunc_counts_as_wrong ...

输出四节：
  ① 生效值表（含**被自动改写**的键，如 ttp/twp 归零）
  ② 死键（设了但不生效）+ 死因 + 改法
  ③ 预算几何（通告/保险丝/overlong 触发线 在真实 clen 分布上的覆盖率）
  ④ 逐项组内施力（每项对 advantage 的实际改动量，用线上打分层实算）
"""
import argparse
import json
import os

# 各 reward 项在「组内」实际能施加的幅度参照系：正确性本身的间距恒为 2.0
# （group_mean 下正确样本 = 2(n−c)/n、错误 = −2c/n，差恒 2）。
_TASK_SPAN = 2.0


def _fmt(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        s = f"{v:g}"
        return s
    return str(v)


def effective_table(cfg: dict) -> list:
    """① 生效值表：返回 [(key, value, note)]。

    note 用来标出**被自动改写**的键——这类键最危险：用户以为设了 A，
    实际生效 B，而日志里只有一行容易被淹没的 print。
    """
    rows = []
    ttp = float(cfg.get("trunc_tail_penalty", 0.0) or 0.0)
    twp = float(cfg.get("tool_waste_penalty", 0.0) or 0.0)
    guard = bool(cfg.get("task_adv_guard"))
    tcw = bool(cfg.get("trunc_counts_as_wrong"))
    for k in ("overlong_shaping", "overlong_ref", "overlong_buffer",
              "overlong_max_penalty", "overlong_fail_only",
              "code_w", "code_attempt_w", "code_shaping_once",
              "tool_call_cost", "tool_waste_penalty",
              "trunc_shaping", "trunc_tail_penalty",
              "len_penalty_w", "len_eff_w", "len_eff_penalty_w",
              "task_adv_guard", "aux_clip_alpha", "trunc_counts_as_wrong",
              "adv_mode", "budget_target", "max_traj_tokens",
              "answer_reserve", "budget_hint"):
        note = ""
        if k in ("tool_waste_penalty", "trunc_tail_penalty") and guard and tcw:
            note = "已被 guard+tcw 自动归零（覆盖语义会削掉一半负信用）"
        rows.append((k, cfg.get(k), note))
    return rows


def dead_keys(cfg: dict) -> list:
    """② 死键检测：返回 [(key, 原因, 改法)]。

    "死"= 设了但当前配置下**对梯度没有任何影响**。判据一律是结构性的
    （谁被谁覆盖、哪条代码路径不读它），不是"这个值看起来小"。
    """
    out = []
    guard = bool(cfg.get("task_adv_guard"))
    tcw = bool(cfg.get("trunc_counts_as_wrong"))
    mtj = int(cfg.get("max_traj_tokens", 0) or 0)

    # --- guard+tcw 档：三个键被结构性覆盖 ---
    if guard and tcw:
        for k, why, how in (
            ("trunc_tail_penalty",
             "tool_credit_advantages 对截断末段是**覆盖**语义，而 tcw 让截断样本"
             "以 acc=−1 进组均值（真实负优势 −1.5）→ 覆盖会把强信号换成弱信号",
             "删掉该 flag；负信用已由 trunc_counts_as_wrong 承担"),
            ("tool_waste_penalty",
             "同上（wasted 分支也是覆盖语义）",
             "删掉该 flag；废码样本已以 acc=−1 进组均值"),
            ("overlong_filter",
             "guard+tcw 下 excl=[0]*n（不再从组统计移除截断/废码），"
             "该开关的排除逻辑整段被绕过",
             "无需处理（guard 档下它不起作用，留着不影响）"),
        ):
            v = cfg.get(k)
            if v not in (0.0, 0, False, None):
                out.append((k, why, how))

    # --- 长度效率项：与 overlong 职责重叠且实测噪音级 ---
    _ol = bool(cfg.get("overlong_shaping"))
    for k, why, how in (
        ("len_eff_penalty_w",
         "只罚「答对且长」，与 overlong(FailOnly) 的失败侧互补但幅度实测 0.031"
         "（正确性间距 2.0 的 1.6%），且经 aux 裁剪后进一步被压",
         "若目标是 token 效率，保留 len_eff_w（通过侧奖短）即可；"
         "本项可置 0"),
        ("len_penalty_w",
         "只罚「未通过且长」，与 overlong(fail_only) 完全重叠；"
         "且 pass_gate=0.25 会让 c≤2/8 的组整段不生效",
         "overlong_shaping=True 时置 0"),
    ):
        if float(cfg.get(k, 0.0) or 0.0) > 0.0 and _ol:
            out.append((k, why, how))
    # len_penalty 的两个从属参数：主权重为 0 时它们没有行为意义
    if float(cfg.get("len_penalty_w", 0.0) or 0.0) <= 0.0:
        for k in ("len_penalty_quantile", "len_penalty_gate"):
            out.append((k, "len_penalty_w=0，本项无行为意义",
                        "无需处理（纯占位）"))

    # --- retool_math 从不读的分档奖励三件套 ---
    if str(cfg.get("algo", "")).startswith("retool") or cfg.get("algo") == "search_qa":
        if cfg.get("algo") == "retool_math":
            for k in ("reward_cold_w", "reward_hot_w", "reward_switch_step"):
                out.append((k,
                            "retool_math 走 total_reward_retool_math，该函数"
                            "没有 phase 参数（冷/热分档只在围栏版 total_reward_retool"
                            " 里生效）",
                            "无需处理（preset 已设 switch_step=1e9，永久 cold）"))

    # --- 预算档相关 ---
    if mtj <= 0:
        for k in ("budget_target", "answer_reserve", "budget_hint"):
            if cfg.get(k):
                out.append((k, "未开 token 预算档（max_traj_tokens=0），"
                               "本项无行为意义",
                            "要么设 --max_traj_tokens，要么删掉该 flag"))
    return out


def budget_geometry(cfg: dict, clen_samples=None) -> dict:
    """③ 预算几何：通告 vs 保险丝 vs overlong 触发线，及在真实 clen 上的覆盖率。

    clen_samples 是实测 completion 长度样本（如 p11c 的分位数）；不给则用
    p11c 实测的典型分布做参照。**这一步专门抓"惩罚区间落在真实分布之外"**。
    """
    from rlab.config import budget_target_tokens
    from rlab.reward import overlong_ref_tokens
    mtj = int(cfg.get("max_traj_tokens", 0) or 0)
    announce = budget_target_tokens(cfg) if mtj > 0 else 0
    ref = overlong_ref_tokens(cfg) if cfg.get("overlong_shaping") else 0
    buf = int(cfg.get("overlong_buffer", 64) or 64)
    trigger = (ref - buf) if ref else 0
    if clen_samples is None:
        # p11c（native, mtj=12288, 难度带 0.125~0.625）实测
        clen_samples = [2600, 5000, 5600, 6056, 6800, 7200, 8060, 11059, 12000]
    hit = [c for c in clen_samples if trigger and c > trigger]
    return {"max_traj_tokens": mtj, "announce": announce,
            "overlong_ref": ref, "overlong_buffer": buf,
            "overlong_trigger": trigger,
            "announce_gap": (mtj - announce) if announce else 0,
            "penalty_zone_width": (mtj - trigger) if (trigger and mtj) else 0,
            "clen_samples": clen_samples,
            "clen_hit_rate": (len(hit) / len(clen_samples)) if clen_samples else 0.0,
            "clen_hit": hit}


def item_forces(cfg: dict, n=8, n_correct=4, clens=None) -> list:
    """④ 逐项组内施力：用**线上打分层**实算每项对 advantage 的改动量。

    做法：保留可复现的单变量——每轮只开一项，与"全关基线"相减，取组内
    最大绝对值。这样得到的就是"这一项真能改变排序多少"，而不是配置里
    那个看起来多大的权重数字。
    """
    from rlab.rollout import retool_score_flat
    if clens is None:
        clens = [5000 + 800 * i for i in range(n)]
    accs = [1] * n_correct + [0] * (n - n_correct)
    texts = [("x \\boxed{42}" if a else "x \\boxed{99}") for a in accs]

    def run(over, code_used=0, code_ok=0):
        c = dict(cfg)
        c.update(over)
        cs = [{"code_used": code_used, "code_ok": code_ok, "code_wasted": 0,
               "invalid_final": 0, "ctx_full": 0, "trunc_final": 0,
               "trunc_in_call": 0, "err_types": []} for _ in range(n)]
        sig = {}
        adv, *_ = retool_score_flat([{"Q": "q", "A": "42"}], texts, cs, c, 0,
                                    completion_lens=clens, signal_out=sig)
        return [float(x) for x in adv]

    off = {"overlong_shaping": False, "code_w": 0.0, "code_attempt_w": 0.0,
           "tool_call_cost": 0.0, "len_eff_w": 0.0, "len_eff_penalty_w": 0.0,
           "len_penalty_w": 0.0, "trunc_shaping": 0.0, "trunc_tail_penalty": 0.0,
           "tool_waste_penalty": 0.0, "task_adv_guard": False}
    base = run(off)
    out = []
    # (名字, 覆盖项, code_used)
    items = [
        ("overlong", dict(overlong_shaping=bool(cfg.get("overlong_shaping")),
                          overlong_ref=cfg.get("overlong_ref") or 0,
                          overlong_buffer=cfg.get("overlong_buffer"),
                          overlong_max_penalty=cfg.get("overlong_max_penalty"),
                          overlong_fail_only=cfg.get("overlong_fail_only")), 0),
        ("code_w", dict(code_w=float(cfg.get("code_w", 0) or 0),
                        code_shaping_once=bool(cfg.get("code_shaping_once"))), 2),
        ("code_attempt_w", dict(code_attempt_w=float(cfg.get("code_attempt_w", 0) or 0),
                                code_shaping_once=bool(cfg.get("code_shaping_once"))), 2),
        ("tool_call_cost", dict(tool_call_cost=float(cfg.get("tool_call_cost", 0) or 0)), 2),
        ("len_eff_w", dict(len_eff_w=float(cfg.get("len_eff_w", 0) or 0)), 0),
        ("len_eff_penalty_w", dict(len_eff_penalty_w=float(cfg.get("len_eff_penalty_w", 0) or 0)), 0),
        ("len_penalty_w", dict(len_penalty_w=float(cfg.get("len_penalty_w", 0) or 0)), 0),
    ]
    for name, ov, cu in items:
        d = dict(off)
        d.update(ov)
        a = run(d, code_used=cu, code_ok=cu)
        delta = [x - y for x, y in zip(a, base)]
        span = max((abs(v) for v in delta), default=0.0)
        note = ""
        # ⚠ 阈值用 1e-6 而不是 1e-9：adv 是 float32，常数项相减会留下 ~6e-8
        # 的舍入残差。用 1e-9 判"是否为零"会把"真零梯度"误报成"有微小施力"。
        if span < 1e-6:
            w = {"code_w": cfg.get("code_w"),
                 "code_attempt_w": cfg.get("code_attempt_w"),
                 "tool_call_cost": cfg.get("tool_call_cost"),
                 "len_eff_w": cfg.get("len_eff_w"),
                 "len_eff_penalty_w": cfg.get("len_eff_penalty_w"),
                 "len_penalty_w": cfg.get("len_penalty_w"),
                 "overlong": cfg.get("overlong_shaping")}.get(name)
            if name == "tool_call_cost":
                note = "token 级，不进组均值 → 本表测不到（见 ④b）"
            elif not w:
                note = "权重为 0（未启用）"
            else:
                note = "组内同值→被组均值减掉→零梯度"
        out.append({"item": name, "span": span,
                    "pct_of_task": span / _TASK_SPAN * 100.0, "note": note})
    out.sort(key=lambda r: -r["span"])
    return out


def token_level_forces(cfg: dict) -> list:
    """④b token 级（不进组均值）的项：tool_call_cost 等。

    它们走 `tool_credit_advantages` 覆盖**已执行调用那一轮**的基线，因此
    `retool_score_flat` 的序列级 adv 看不到它们——但它们在 loss 里真实存在。
    单列一节，免得体检表把它们误报成"零梯度"。
    """
    import torch
    from rlab.rollout import tool_credit_advantages
    out = []
    cc = float(cfg.get("tool_call_cost", 0.0) or 0.0)
    twp = float(cfg.get("tool_waste_penalty", 0.0) or 0.0)
    ttp = float(cfg.get("trunc_tail_penalty", 0.0) or 0.0)
    segs = [[{"kind": "assistant", "ids": list(range(600)),
              "tool_action": "executed"},
             {"kind": "tool", "ids": list(range(80))},
             {"kind": "assistant", "ids": list(range(600))}]]
    base = torch.tensor([1.5])
    for name, kw, on in (
        ("tool_call_cost", dict(call_cost=cc), cc > 0),
        ("tool_waste_penalty", dict(waste_penalty=twp), twp > 0),
        ("trunc_tail_penalty（末段覆盖）", dict(trunc_flags=[1],
                                              trunc_tail_penalty=ttp), ttp > 0),
    ):
        if not on:
            out.append({"item": name, "delta": 0.0, "note": "未启用"})
            continue
        a = tool_credit_advantages(base, segs, 1280, **kw)
        d = float(a[0][0]) - 1.5
        out.append({"item": name, "delta": abs(d),
                    "note": "只作用于已执行调用轮（token 级，不进组均值）"})
    return out


def audit(cfg: dict, clen_samples=None) -> dict:
    return {"effective": effective_table(cfg), "dead": dead_keys(cfg),
            "geometry": budget_geometry(cfg, clen_samples),
            "forces": item_forces(cfg),
            "token_level": token_level_forces(cfg)}


def render(rep: dict) -> str:
    out = []
    out.append("=" * 78)
    out.append("① 生效值（★ = 被自动改写：你设的值不是生效值）")
    out.append("=" * 78)
    for k, v, note in rep["effective"]:
        mark = "★" if note else " "
        out.append(f" {mark} {k:24s} = {_fmt(v):<10s} {note}")

    out.append("")
    out.append("=" * 78)
    out.append("② 死键（设了但不生效——这类才是「配置全对、接线断开」的来源）")
    out.append("=" * 78)
    if not rep["dead"]:
        out.append("  （无）")
    for k, why, how in rep["dead"]:
        out.append(f"  · {k}")
        out.append(f"      原因：{why}")
        out.append(f"      改法：{how}")

    g = rep["geometry"]
    out.append("")
    out.append("=" * 78)
    out.append("③ 预算几何（惩罚区间是否落在真实 clen 分布内）")
    out.append("=" * 78)
    out.append(f"  保险丝 max_traj_tokens = {g['max_traj_tokens']}")
    out.append(f"  通告 budget_target    = {g['announce']}"
               f"（差 {g['announce_gap']} = 模型看不见的溢出余量）")
    out.append(f"  overlong ref={g['overlong_ref']} buffer={g['overlong_buffer']}"
               f" → 触发线 {g['overlong_trigger']}")
    out.append(f"  受罚区间宽度 = 保险丝 − 触发线 = {g['penalty_zone_width']} token")
    out.append(f"  参照 clen 样本 = {g['clen_samples']}")
    out.append(f"  落在受罚区间的比例 = {g['clen_hit_rate'] * 100:.0f}%"
               f"  命中={g['clen_hit']}")
    if g["penalty_zone_width"] and g["penalty_zone_width"] < 1024:
        out.append("  ⚠ 受罚区间 < 1024 token：生成撞保险丝就整条停住，"
                   "模型几乎不可能落在里面 → 该惩罚**接近死开关**。"
                   "把 overlong_buffer 抬大（或给 overlong_ref 显式值）。")

    out.append("")
    out.append("=" * 78)
    out.append(f"④ 逐项组内施力（正确性间距 = {_TASK_SPAN:.1f}，n=8，c=4/8）")
    out.append("=" * 78)
    out.append(f"  {'项':24s} {'幅度':>8s} {'占正确性':>9s}  备注")
    for r in rep["forces"]:
        out.append(f"  {r['item']:24s} {r['span']:8.3f} "
                   f"{r['pct_of_task']:8.1f}%  {r['note']}")
    out.append("")
    out.append("  读法：<2% 基本是噪音；>10% 才是有效施力。"
               "若某项为 0，多半是「全组同值 → 被组均值减掉」。")
    out.append("")
    out.append("  ④b token 级项（走 tool_credit_advantages，不进组均值；"
               "上表测不到它们）")
    for r in rep.get("token_level", []):
        out.append(f"    {r['item']:24s} 幅度={r['delta']:.3f}  {r['note']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="训练前配置体检（生效值/死键/几何/施力）")
    ap.add_argument("--from_run_info", default=None,
                    help="已有 run 的 run_info.json（或所在目录）：直接体检那次 run")
    ap.add_argument("--algo", default="retool_math")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    # 常用覆盖（与 train 同名，便于把 run.sh 里的关键项搬过来）
    for _k, _t in (("max_traj_tokens", int), ("budget_target", int),
                   ("answer_reserve", int), ("max_context_tokens", int),
                   ("round_gen_tokens", int), ("overlong_ref", int),
                   ("overlong_buffer", int), ("tool_call_cost", float),
                   ("tool_waste_penalty", float), ("trunc_tail_penalty", float),
                   ("code_w", float), ("code_attempt_w", float),
                   ("len_eff_w", float), ("len_eff_penalty_w", float),
                   ("len_penalty_w", float), ("aux_clip_alpha", float),
                   ("overlong_max_penalty", float)):
        ap.add_argument(f"--{_k}", type=_t, default=None)
    for _k in ("budget_hint", "task_adv_guard", "trunc_counts_as_wrong",
               "overlong_shaping", "overlong_fail_only", "code_shaping_once"):
        ap.add_argument(f"--{_k}", action=argparse.BooleanOptionalAction,
                        default=None)
    args = ap.parse_args()

    from rlab.config import get_config
    ov = {}
    for k in ("max_traj_tokens", "budget_target", "answer_reserve",
              "max_context_tokens", "round_gen_tokens", "overlong_ref",
              "overlong_buffer", "tool_call_cost", "tool_waste_penalty",
              "trunc_tail_penalty", "code_w", "code_attempt_w", "len_eff_w",
              "len_eff_penalty_w", "len_penalty_w", "aux_clip_alpha",
              "overlong_max_penalty", "budget_hint", "task_adv_guard",
              "trunc_counts_as_wrong", "overlong_shaping", "overlong_fail_only",
              "code_shaping_once"):
        v = getattr(args, k, None)
        if v is not None:
            ov[k] = v
    if args.from_run_info:
        p = args.from_run_info
        if os.path.isdir(p):
            p = os.path.join(p, "run_info.json")
        with open(p, encoding="utf-8") as f:
            rc = (json.load(f) or {}).get("config") or {}
        if not rc:
            raise SystemExit(f"[audit] {p} 里没有 config 键")
        # run_info 的 config 是**训练时生效后**的完整配置（含自动归零结果），
        # 直接采用；CLI 显式给的值仍覆盖（用于"如果我改成 X 会怎样"）。
        cfg = dict(rc)
        cfg.update(ov)
        cfg["_audit_src"] = p
    else:
        cfg = get_config(args.algo, use_wandb=False,
                         tool_protocol="native", **ov)
        cfg["_audit_src"] = "get_config(" + args.algo + ")"

    rep = audit(cfg)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"体检来源: {cfg.get('_audit_src')}")
        print(render(rep))


if __name__ == "__main__":
    main()
