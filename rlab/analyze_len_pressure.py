#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计“答对但长于同组通过参考长度”的占比与 reward/adv 压力。

输入使用 record.jsonl：一行就是一个题组，组内 acc/clen/code_* 列严格对齐。
脚本从同目录 run_info.json 读取真实训练配方，复算三种形态：

  shipped : MiMo 原形态，只罚未通过且超 ref 的轨迹
  all_raw : 当前本地改法，所有超 ref 轨迹都罚，不封顶
  all_cap : 推荐反事实，所有超 ref 轨迹都罚，ratio cap=1

复算终点是 tool-credit 之前的 sequence-level group_mean advantage。工具调用成本、
废调用信用和截断末段信用是后续 segment-level 局部覆盖，不在本次长度曲面比较内。

用法：
  python -m rlab.analyze_len_pressure rlab_new/native_p5_best
  python -m rlab.analyze_len_pressure rlab_new/native_p5_best/record.jsonl
  python -m rlab.analyze_len_pressure rlab_new/native_p5_best --include-uniform
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(xs, p):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(p) / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def fmt_num(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def fmt_pct(num, den):
    return "n/a" if not den else f"{100.0 * num / den:.2f}%"


def describe(xs):
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs) if xs else None,
        "p25": percentile(xs, 25),
        "p50": percentile(xs, 50),
        "p75": percentile(xs, 75),
        "p90": percentile(xs, 90),
        "max": max(xs) if xs else None,
    }


def quantile_ref(lens, rewards, quantile):
    passed = sorted(float(l) for l, r in zip(lens, rewards) if float(r) > 0.0)
    if not passed:
        return None, 0
    b = min(max(int(quantile), 1), 100) / 100.0
    k = int(math.ceil(b * (len(passed) - 1)))
    return passed[k], len(passed)


def overlong_ref_tokens(cfg):
    explicit = int(cfg.get("overlong_ref", 0) or 0)
    if explicit > 0:
        return explicit
    max_traj = int(cfg.get("max_traj_tokens", 0) or 0)
    ctx = int(cfg.get("max_context_tokens", 0) or 0)
    plen = int(cfg.get("max_prompt_length", 0) or 0)
    usable = ctx - plen if ctx > plen else 0
    if max_traj > 0:
        return min(max_traj, usable) if usable else max_traj
    cap = int(cfg.get("max_rounds", 1) or 1) * int(cfg.get("round_gen_tokens", 0) or 0)
    return min(cap, usable) if usable else cap


def overlong_penalty(clen, ref, buffer):
    if ref <= 0 or buffer <= 0:
        return 0.0
    trigger = ref - buffer
    if clen <= trigger:
        return 0.0
    return min((clen - trigger) / buffer, 1.0)


def base_rewards(row, cfg):
    acc = [float(x) for x in row["acc"]]
    clen = [int(x) for x in row["clen"]]
    n = len(acc)
    cu = [int(x) for x in row.get("code_used", [0] * n)]
    ck = [int(x) for x in row.get("code_ok", [0] * n)]
    trunc = [int(x) for x in row.get("trunc_final", [0] * n)]

    once = bool(cfg.get("code_shaping_once", False))
    attempt_w = float(cfg.get("code_attempt_w", 0.0) or 0.0)
    code_w = float(cfg.get("code_w", 0.0) or 0.0)
    max_rounds = max(1, int(cfg.get("max_rounds", 8) or 8))
    trunc_w = float(cfg.get("trunc_shaping", 0.0) or 0.0)
    ol_enabled = bool(cfg.get("overlong_shaping", False))
    ol_ref = overlong_ref_tokens(cfg)
    ol_buffer = int(cfg.get("overlong_buffer", 64) or 64)

    out = []
    for a, length, used, ok, tf in zip(acc, clen, cu, ck, trunc):
        r = a
        if ol_enabled:
            r -= overlong_penalty(length, ol_ref, ol_buffer)
        if trunc_w > 0 and tf:
            r -= trunc_w
        used_n, ok_n = min(used, max_rounds), min(ok, max_rounds)
        if attempt_w > 0 and used_n > 0:
            r += attempt_w if once else used_n * attempt_w
        if code_w > 0 and ok_n > 0:
            r += code_w if once else ok_n * code_w
        out.append(float(r))
    return out


def apply_len_penalty(rewards, lens, weight, quantile, gate, mode, cap=1.0):
    rewards = [float(x) for x in rewards]
    ref, n_pass = quantile_ref(lens, rewards, quantile)
    active = ref is not None and n_pass / max(1, len(rewards)) > gate
    if weight <= 0 or not active or ref <= 0:
        return rewards, ref, n_pass, active

    out = []
    for r, length in zip(rewards, lens):
        ratio = max(0.0, float(length) / ref - 1.0)
        if mode == "shipped":
            p = weight * ratio if r <= 0.0 else 0.0
        elif mode == "all_raw":
            p = weight * ratio
        elif mode == "all_cap":
            p = weight * min(ratio, float(cap))
        else:
            raise ValueError(f"unknown penalty mode: {mode}")
        out.append(r - p)
    return out, ref, n_pass, active


def apply_eff_bonus(rewards, lens, weight, quantile=50, cap=1.0):
    rewards = [float(x) for x in rewards]
    ref, n_pass = quantile_ref(lens, rewards, quantile)
    if weight <= 0 or n_pass < 2 or ref is None or ref <= 0:
        return rewards
    out = []
    for r, length in zip(rewards, lens):
        if r > 0.0 and 0 < length < ref:
            out.append(r + weight * min(ref / length - 1.0, float(cap)))
        else:
            out.append(r)
    return out


def group_mean_adv(rewards, mask):
    valid = [r for r, m in zip(rewards, mask) if m]
    mean = statistics.fmean(valid) if valid else 0.0
    return [(r - mean) if m else 0.0 for r, m in zip(rewards, mask)]


def replay(row, cfg, mode, cap, eff_weight=None):
    lens = [int(x) for x in row["clen"]]
    r0 = base_rewards(row, cfg)
    r1, ref, n_pass, active = apply_len_penalty(
        r0, lens,
        float(cfg.get("len_penalty_w", 0.0) or 0.0),
        int(cfg.get("len_penalty_quantile", 50) or 50),
        float(cfg.get("len_penalty_gate", 0.25) or 0.25),
        mode, cap,
    )
    ew = float(cfg.get("len_eff_w", 0.0) or 0.0) if eff_weight is None else eff_weight
    r2 = apply_eff_bonus(r1, lens, ew)

    n = len(r2)
    trunc = [int(x) for x in row.get("trunc_final", [0] * n)]
    wasted = [int(x) for x in row.get("code_wasted", [0] * n)]
    if bool(cfg.get("overlong_filter", False)):
        mask = [not (tf or wf) for tf, wf in zip(trunc, wasted)]
    else:
        mask = [True] * n
    return {
        "reward": r2,
        "adv": group_mean_adv(r2, mask),
        "mask": mask,
        "ref": ref,
        "n_pass": n_pass,
        "active": active,
    }


def resolve_paths(target, run_info_arg):
    p = Path(target)
    if p.is_dir():
        record = p / "record.jsonl"
        run_info = Path(run_info_arg) if run_info_arg else p / "run_info.json"
    else:
        record = p
        run_info = Path(run_info_arg) if run_info_arg else p.parent / "run_info.json"
    if not record.is_file():
        raise SystemExit(f"record 不存在: {record}")
    if not run_info.is_file():
        raise SystemExit(f"run_info 不存在: {run_info}（可用 --run-info 指定）")
    return record, run_info


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    cfg = dict(info.get("config") or {})
    for key, value in info.items():
        if key != "config" and key not in cfg:
            cfg[key] = value
    return cfg, info


def print_dist(title, xs):
    d = describe(xs)
    print(f"{title}: n={d['n']} mean={fmt_num(d['mean'])} p25={fmt_num(d['p25'])} "
          f"p50={fmt_num(d['p50'])} p75={fmt_num(d['p75'])} "
          f"p90={fmt_num(d['p90'])} max={fmt_num(d['max'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="out_dir 或 record.jsonl")
    ap.add_argument("--run-info", default=None)
    ap.add_argument("--include-uniform", action="store_true",
                    help="把 q_status=uniform 组也纳入（默认只分析真实上传的 ok 组）")
    ap.add_argument("--cap", type=float, default=1.0,
                    help="all_cap 形态的 ratio cap（默认 1.0）")
    ap.add_argument("--len-penalty-w", type=float, default=None,
                    help="覆盖 run_info 的 len_penalty_w，仅用于反事实")
    ap.add_argument("--len-eff-w", type=float, default=None,
                    help="覆盖 run_info 的 len_eff_w")
    args = ap.parse_args()

    record_path, run_info_path = resolve_paths(args.target, args.run_info)
    cfg, info = load_cfg(run_info_path)
    if args.len_penalty_w is not None:
        cfg["len_penalty_w"] = args.len_penalty_w
    if args.len_eff_w is not None:
        cfg["len_eff_w"] = args.len_eff_w
    if cfg.get("adv_mode", "group_mean") != "group_mean":
        raise SystemExit(f"本脚本只支持 adv_mode=group_mean，收到 {cfg.get('adv_mode')!r}")

    lpw = float(cfg.get("len_penalty_w", 0.0) or 0.0)
    lew = float(cfg.get("len_eff_w", 0.0) or 0.0)
    print("=== 配方自证 ===")
    print(f"record={record_path}")
    print(f"run_info={run_info_path}")
    print(f"signature={info.get('signature')}")
    print(f"len_penalty_w={lpw} quantile={cfg.get('len_penalty_quantile')} "
          f"gate={cfg.get('len_penalty_gate')} len_eff_w={lew}")
    print(f"overlong={cfg.get('overlong_shaping')} ref={overlong_ref_tokens(cfg)} "
          f"buffer={cfg.get('overlong_buffer')} filter={cfg.get('overlong_filter')}")
    print(f"code_attempt_w={cfg.get('code_attempt_w')} code_w={cfg.get('code_w')} "
          f"once={cfg.get('code_shaping_once')} cap={args.cap}")
    print("注意：adv 是 tool-credit 之前的 sequence-level group_mean；"
          "工具调用成本/废调用/截断末段信用是后续 segment-level 局部覆盖。\n")

    groups = samples = correct = valid_correct = affected = 0
    gate_active_groups = invalid_rows = 0
    status_counts = {}
    trunc_failed = trunc_correct = 0
    ratio, adv_old, adv_raw, adv_cap = [], [], [], []
    rew_old, rew_raw, rew_cap = [], [], []
    delta_eff, delta_raw, delta_cap = [], [], []
    raw_reward_flip = cap_reward_flip = raw_adv_flip = cap_adv_flip = 0
    would_ununiform_raw = would_ununiform_cap = uniform_groups = 0

    with open(record_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            status = row.get("q_status", "ok")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "ok" and not (args.include_uniform and status == "uniform"):
                continue
            acc, lens = row.get("acc"), row.get("clen")
            if not isinstance(acc, list) or not isinstance(lens, list) or len(acc) != len(lens):
                invalid_rows += 1
                continue
            n = len(acc)
            for key in ("code_used", "code_ok", "trunc_final", "code_wasted"):
                value = row.get(key)
                if value is not None and (not isinstance(value, list) or len(value) != n):
                    raise SystemExit(f"line {line_no}: {key} 与 acc 长度不一致")

            old = replay(row, cfg, "shipped", args.cap)
            old_no_eff = replay(row, cfg, "shipped", args.cap, eff_weight=0.0)
            raw = replay(row, cfg, "all_raw", args.cap)
            capped = replay(row, cfg, "all_cap", args.cap)
            groups += 1
            samples += n
            gate_active_groups += int(old["active"])
            if status == "uniform":
                uniform_groups += 1
                would_ununiform_raw += int(any(abs(x) > 1e-8 for x in raw["adv"]))
                would_ununiform_cap += int(any(abs(x) > 1e-8 for x in capped["adv"]))

            acc_f = [float(x) for x in acc]
            correct += sum(x > 0 for x in acc_f)
            for a, tf in zip(acc_f, row.get("trunc_final", [0] * n)):
                if int(tf):
                    if a > 0:
                        trunc_correct += 1
                    else:
                        trunc_failed += 1

            ref = old["ref"]
            for i, (a, length, valid) in enumerate(zip(acc_f, lens, old["mask"])):
                if a <= 0:
                    continue
                if valid:
                    valid_correct += 1
                if not (valid and old["active"] and ref and length > ref):
                    continue
                affected += 1
                ratio.append(float(length) / ref)
                adv_old.append(old["adv"][i])
                adv_raw.append(raw["adv"][i])
                adv_cap.append(capped["adv"][i])
                rew_old.append(old["reward"][i])
                rew_raw.append(raw["reward"][i])
                rew_cap.append(capped["reward"][i])
                delta_eff.append(old["adv"][i] - old_no_eff["adv"][i])
                delta_raw.append(raw["adv"][i] - old["adv"][i])
                delta_cap.append(capped["adv"][i] - old["adv"][i])
                raw_reward_flip += int(old["reward"][i] > 0 >= raw["reward"][i])
                cap_reward_flip += int(old["reward"][i] > 0 >= capped["reward"][i])
                raw_adv_flip += int(old["adv"][i] >= 0 > raw["adv"][i])
                cap_adv_flip += int(old["adv"][i] >= 0 > capped["adv"][i])

    print("=== 人群规模（默认只看真实上传的 q_status=ok 组） ===")
    print(f"groups={groups} samples={samples} correct={correct} "
          f"valid_correct={valid_correct} bad_rows={invalid_rows}")
    print(f"status_counts={status_counts}")
    print(f"len_penalty gate 生效组={gate_active_groups}/{groups} "
          f"({fmt_pct(gate_active_groups, groups)})")
    print(f"目标人群(acc>0、未过滤、clen>同组通过ref)={affected}，"
          f"占有效正确={fmt_pct(affected, valid_correct)}，占全部样本={fmt_pct(affected, samples)}")
    print_dist("目标人群 clen/ref", ratio)
    print(f"截断失败={trunc_failed}，截断但正确={trunc_correct}（均被 sample_mask 排除，"
          "length reward 改值不产生任务梯度）\n")

    print("=== 当前 shipped 形态：是否已有隐式压力 ===")
    print_dist("目标人群 current adv", adv_old)
    neg = sum(x < 0 for x in adv_old)
    neutral = sum(abs(x) < 0.02 for x in adv_old)
    print(f"adv<0: {neg}/{affected} ({fmt_pct(neg, affected)})")
    print(f"|adv|<0.02: {neutral}/{affected} ({fmt_pct(neutral, affected)})")
    print_dist("len_eff 对目标人群的 adv 增量（通常应<=0）", delta_eff)
    print("解释：目标样本不拿 eff bonus，但短正确样本会抬高组均值；若该增量接近 0，"
          "说明隐式压力确实太弱。\n")

    print("=== 反事实比较 ===")
    print_dist("all_raw（罚全部、不封顶）目标 adv", adv_raw)
    print_dist("all_raw 相对 shipped 的 adv 变化", delta_raw)
    print(f"reward +→非正: {raw_reward_flip}/{affected}，adv 非负→负: {raw_adv_flip}/{affected}")
    print_dist(f"all_cap（罚全部、ratio cap={args.cap:g}）目标 adv", adv_cap)
    print_dist("all_cap 相对 shipped 的 adv 变化", delta_cap)
    print(f"reward +→非正: {cap_reward_flip}/{affected}，adv 非负→负: {cap_adv_flip}/{affected}")
    print_dist("shipped 目标 reward", rew_old)
    print_dist("all_raw 目标 reward", rew_raw)
    print_dist("all_cap 目标 reward", rew_cap)
    if args.include_uniform:
        print(f"uniform 组={uniform_groups}；all_raw 会变为非零adv={would_ununiform_raw}，"
              f"all_cap 会变为非零adv={would_ununiform_cap}")

    print("\n=== 判读 ===")
    if affected == 0 or affected / max(1, valid_correct) < 0.05:
        print("目标人群 <5%：不建议改 reward 语义；优先保留 MiMo shipped 形态。")
    elif neg / affected >= 0.70 and neutral / affected < 0.30:
        print("目标人群多数已拿负 adv：len_eff 隐式压力已工作；先调剂量，不建议改语义。")
    else:
        print("目标人群占比可观且当前压力弱：可以考虑 all_cap，禁止直接上 all_raw。")
    if raw_reward_flip:
        print("all_raw 已把正确 reward 翻为非正：这是硬否决信号。")
    if cap_reward_flip:
        print("警告：all_cap 仍翻 reward，说明其它 shaping 已将 reward 压近 0；需降 weight/cap。")


if __name__ == "__main__":
    main()
