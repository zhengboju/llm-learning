# -*- coding: utf-8 -*-
"""rlab/sigstats.py — 正确性信号观测聚合（doc15 阶段B，2026-10-15）。

回答一个问题：**进入训练的更新里，有多少是"这条对、那条错"的正确性对比信号？**
uploaded≠正确性混合组（group_ok 判的是含辅助/局部项的最终 advantage，全对/全错
组可靠辅助项进训练），所以只看 uploaded 数/总 reward 回答不了，必须从 record 的
观测列（task_mask/adv_task/adv_aux/adv_local_mag，2026-10-15 起由 rollout 落盘）
聚出三个量：

- M_raw      ：已完成且可评分（ok+uniform）的组中，0<n_correct<n 的比例
- M_task     ：应用任务 mask（overlong_filter 排除截断/废调用）后仍有对有错的比例
- M_uploaded ：已上传（ok）组中 M_task 的比例；另列全对/全错但有辅助梯度的组

同时给成本口径：生成 token / optimizer update、生成 token / 正确性混合上传组。

用法：
    python -m rlab.sigstats --record rlab_new/native_p10_best/record.jsonl \
        [--traj rlab_new/native_p10_best/traj.jsonl] [--gas 8] [--window 200]

gas（gradient_accumulation_steps）用于把 ok 组数换算成 optimizer updates；
不传则只报组数不报 update 口径。旧 record 无信号列的组按"不可判"计入分母
并在输出中显式标注 n/a，绝不静默当 0。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _n_correct(acc):
    return sum(1 for a in acc if a and a > 0)


def group_signal(rows):
    """纯函数（CPU 可测）：record 行（每行一组）-> 信号/成本聚合 dict。

    rows 每项：{q_status, acc, clen, task_mask?, adv_task?, adv_aux?,
    adv_local_mag?, attempt_id?, qk?, gen_version?}。"""
    g = {"groups": len(rows), "ok": 0, "uniform": 0, "other_status": 0,
         "scored": 0,            # 有 acc 数组（可判 M_raw）
         "m_raw": 0,             # 0<n_correct<n（未应用任务 mask）
         "with_sig": 0,          # 有 task_mask（可判 M_task）
         "m_task": 0,            # mask 后仍 0<correct<valid
         "valid_samples_sum": 0, "valid_samples_n": 0,
         "up_scored": 0, "up_task_mixed": 0,
         "up_allcorrect_aux": 0, "up_allwrong_aux": 0,
         "up_aux_mag_sum": 0.0, "up_local_mag_sum": 0.0,
         "flip_pos_to_nonpos": 0, "flip_neg_to_nonneg": 0, "task_nonzero": 0,
         "tok_ok": 0, "tok_uniform": 0,
         "with_attempt_id": 0}
    for r in rows:
        st = r.get("q_status")
        acc = r.get("acc") or []
        clen = r.get("clen") or []
        tok = sum(int(x) for x in clen)
        if st == "ok":
            g["ok"] += 1
            g["tok_ok"] += tok
        elif st == "uniform":
            g["uniform"] += 1
            g["tok_uniform"] += tok
        else:
            g["other_status"] += 1
            continue
        if r.get("attempt_id") is not None:
            g["with_attempt_id"] += 1
        if acc:
            g["scored"] += 1
            if 0 < _n_correct(acc) < len(acc):
                g["m_raw"] += 1
        tm = r.get("task_mask")
        if tm:
            g["with_sig"] += 1
            valid = sum(1 for m in tm if m)
            corr = sum(1 for a, m in zip(acc, tm) if m and a and a > 0)
            g["valid_samples_sum"] += valid
            g["valid_samples_n"] += len(tm)
            task_mixed = 0 < corr < valid
            if task_mixed:
                g["m_task"] += 1
            if st == "ok":
                g["up_scored"] += 1
                aux = r.get("adv_aux") or []
                loc = r.get("adv_local_mag") or []
                if task_mixed:
                    g["up_task_mixed"] += 1
                elif corr == valid and valid > 0:
                    g["up_allcorrect_aux"] += 1   # 全对仍上传 → 辅助/局部项驱动
                elif corr == 0:
                    g["up_allwrong_aux"] += 1     # 全错仍上传 → 辅助/局部项驱动
                g["up_aux_mag_sum"] += sum(abs(float(x)) for x in aux)
                g["up_local_mag_sum"] += sum(float(x) for x in loc)
                # 符号翻转红线：正确性优势为正被辅助项压成非正（或反向）。
                at = r.get("adv_task") or []
                for t, a in zip(at, aux):
                    t, a = float(t), float(a)
                    if t == 0.0:
                        continue
                    g["task_nonzero"] += 1
                    if t > 0 and t + a <= 0:
                        g["flip_pos_to_nonpos"] += 1
                    elif t < 0 and t + a >= 0:
                        g["flip_neg_to_nonneg"] += 1
    return g


def _pct(a, b):
    return "n/a" if not b else f"{100.0 * a / b:.1f}%"


def report(g, gas=0, title=""):
    """group_signal 结果 -> 文本报告（纯函数）。"""
    L = []
    if title:
        L.append(f"## {title}")
    L.append(f"组数 {g['groups']}（ok {g['ok']} / uniform {g['uniform']} / "
             f"其他 {g['other_status']}）；带 attempt_id {_pct(g['with_attempt_id'], g['groups'])}")
    L.append(f"M_raw      = {_pct(g['m_raw'], g['scored'])}"
             f"（{g['m_raw']}/{g['scored']} 可评分组 0<n_correct<n）")
    if g["with_sig"]:
        L.append(f"M_task     = {_pct(g['m_task'], g['with_sig'])}"
                 f"（{g['m_task']}/{g['with_sig']} 任务 mask 后仍混合；"
                 f"组均有效样本 {g['valid_samples_sum'] / max(1, g['with_sig']):.2f}）")
        L.append(f"M_uploaded = {_pct(g['up_task_mixed'], g['up_scored'])}"
                 f"（{g['up_task_mixed']}/{g['up_scored']} 上传组任务混合；"
                 f"全对仍传 {g['up_allcorrect_aux']} / 全错仍传 {g['up_allwrong_aux']}）")
        L.append(f"  上传组 |adv_aux| 总和 {g['up_aux_mag_sum']:.1f}，"
                 f"局部项幅度总和 {g['up_local_mag_sum']:.1f}")
        L.append(f"  符号翻转：正确性为正被压成非正 {g['flip_pos_to_nonpos']} / "
                 f"为负被抬成非负 {g['flip_neg_to_nonneg']}"
                 f"（task 非零样本 {g['task_nonzero']}）——非零即触发 doc15 §7 停线")
    else:
        L.append("M_task/M_uploaded = n/a（record 无 task_mask 列，"
                 "需 2026-10-15 后的 rollout 落盘）")
    tok_all = g["tok_ok"] + g["tok_uniform"]
    L.append(f"生成 token：ok {g['tok_ok']} + uniform {g['tok_uniform']} = {tok_all}"
             f"（不含 prompt_overlong/overlong/proto_mismatch——它们不进 record，"
             f"全口径请补 --traj）")
    if gas:
        updates = g["ok"] / gas
        L.append(f"optimizer updates ≈ {updates:.1f}（ok 组 {g['ok']} / GAS {gas}）；"
                 f"token/update ≈ {tok_all / max(1e-9, updates):.0f}；"
                 f"token/任务混合上传组 ≈ "
                 f"{tok_all / max(1, g['up_task_mixed']):.0f}")
    return "\n".join(L)


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue   # 追加写可能留截断行，与 load_difficulty_table 同口径
    return rows


def main():
    ap = argparse.ArgumentParser(description="正确性信号观测聚合（doc15 阶段B）")
    ap.add_argument("--record", required=True, help="record.jsonl 路径")
    ap.add_argument("--traj", default=None,
                    help="可选 traj.jsonl：补 overlong/proto_mismatch 的组数与 token")
    ap.add_argument("--gas", type=int, default=0,
                    help="gradient_accumulation_steps（把 ok 组数换算成 updates）")
    ap.add_argument("--window", type=int, default=200,
                    help="最近 N 个 ok 组的滑窗（0=只看全程）")
    args = ap.parse_args()

    rows = _read_jsonl(args.record)
    ok_rows = [r for r in rows if r.get("q_status") == "ok"]
    print(report(group_signal(rows), gas=args.gas, title="全程"))
    if args.window and len(ok_rows) > args.window:
        # 截尾口径：从倒数第 window 个 ok 组在记录流中的位置起，之后所有行
        # （含其间交织的 uniform）都进窗口。
        _first = ok_rows[-args.window]
        _start = next(i for i, r in enumerate(rows) if r is _first)
        print()
        print(report(group_signal(rows[_start:]), gas=args.gas,
                     title=f"最近 {args.window} 个 ok 组窗口"))
    if args.traj:
        tr = _read_jsonl(args.traj)
        by_st, tok_by_st = {}, {}
        for r in tr:
            st = r.get("status", "?")
            by_st[st] = by_st.get(st, 0) + 1
            tok_by_st[st] = tok_by_st.get(st, 0) + int(r.get("clen") or 0)
        tot = sum(tok_by_st.values())
        print()
        print("## traj 全口径（含丢弃轨迹）")
        print(f"轨迹数 {len(tr)}（{by_st}）；总生成 token {tot}，"
              f"按状态 {tok_by_st}")


if __name__ == "__main__":
    main()
