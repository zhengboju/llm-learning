# -*- coding: utf-8 -*-
"""rlab/sft_data.py — SFT 冷启动数据生成（docs/08 阶段①，2026-10-09 重定位版）。

与 docs/08 初版的差别：目标是**预算感知收尾**（治 p6/p7 无 boxed 三桶：
C_wasted 终局废码 / B_call 调用拖到末刻 / B2 纯散文写满），不是"教写代码"
（原生协议下代码行为本就在分布里）。三类样本：

  · real_tool   —— dump 里"用码 & 答对 & 未截断"的真实轨迹（行为主体）；
  · zero_call   —— "零调用 & 答对"的轨迹（B2 的反例：不用工具也能干净收尾）；
  · low_budget  —— **合成**：取 real_tool 轨迹，在最后一个工具回包处截断，
                  把该回包的 [budget] 行改写成强警档（必然装不下下一次调用），
                  后接一条简短收尾 assistant（boxed 答案取自原轨迹自身）。
                  这是 C_wasted/B_call 的直接正例："预算不够 → 不再调用，
                  立刻把已有结论 boxed 出来"。

两条铁律的落点：
  · 预算行文本只从 rollout.budget_hint_line 派生（与线上注入同一函数，
    字节级一致——SFT 教的提示形态就是 RL 时会看到的形态）；
  · 系统提示经 get_config + inject_budget_prompt 装配（与训练/probe 同一
    入口），每行落 sp_sha 供事后核对"数据提示 == RL 提示"。

dump 行只有 text/len 没有 ids（traj_dump_row 的设计是诊断面），token 化在
sft.py 的 encode_example 里经 render_chat_ids 增量渲染完成——序列契约
（prompt + 逐段增量）与 rollout 同源，本文件只处理文本层。

用法：
  python -m rlab.sft_data \
      --dump rlab_new/native_p6_best/traj.jsonl --dump rlab_new/native_p8_best/traj.jsonl \
      --out rlab_out/sft_budget_v2.jsonl \
      --from_run_info rlab_new/native_p9_best/run_info.json

选样（2026-10-11 新增，服务"交卷率已近饱和、该碰解题力"）：
  · `--max_ok N`（N < num_pre_Q）开启**近界选样**：只保留该题答对条数 ≤N 的题，
    排除已掌握的题（信号饱和）。这是 rejection sampling / STaR，唯一能真实抬高
    `P(correct)` 的选样口径。先用 pass-rate 直方图（下面那行打印）判断值不值得开。
  · `--no_prefer_short`：关闭"偏好短轨迹"。该偏好在长度已压住后只会优先剔除
    长轨迹（=难题的正确解），纯伤害解题力。
  · `--max_per_q 4 --synth_per_q 2`：放宽每题信号量。
"""
import argparse
import hashlib
import json
import random
import re
import sys

from rlab.reward import extract_last_boxed
from rlab.rollout import budget_hint_line

# 旧的单行预算提示与新分档三种形态都以 "[budget] " 开头、同行收尾；
# 改写 = 解析剩余额度 N，用同一函数重吐当前档位的文案（幂等：新形态再改写
# 仍得新形态）。
_BUDGET_RE = re.compile(r"\n?\[budget\] (?:Only )?(\d+) tokens left[^\n]*")

# 低预算合成样本的收尾模板。零格式标签字面量；boxed 答案取自原轨迹自身
# （该轨迹已验证答对），不是凭空编造。
_WRAP_UP = ("The token budget is nearly exhausted, so I stop reasoning here. "
            "Based on the results obtained above, the final answer is "
            "\\boxed{%s}.")

_SOURCES = ("real_tool", "zero_call", "low_budget")


def load_dump_rows(paths):
    """逐行读 dump jsonl（坏行跳过并计数，绝不因此中断）。"""
    n_bad = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
    if n_bad:
        print(f"[sft_data] 跳过 {n_bad} 行坏 json", flush=True)


def row_ok(row) -> bool:
    """示范轨迹的入选判据：已上传 + 答对 + 未截断（截断轨迹没有完整 boxed）。"""
    return (row.get("status") == "ok" and (row.get("acc") or 0) > 0
            and not int(row.get("trunc_final") or 0))


def split_turns(row):
    """dump 段表 → turns 列表；结构非法（不交替/不以 assistant 开头）返回 None。

    assistant 文本 strip（与 rollout 里 messages 的口径一致：chat template 会
    strip assistant 内容，存 stripped 形态让 encode 的 prefix 校验可过）。"""
    segs = row.get("segs") or []
    if not segs:
        return None
    turns = []
    for i, s in enumerate(segs):
        kind = s.get("kind")
        if kind == "assistant":
            if i % 2 != 0:
                return None
            turns.append({"kind": "assistant",
                          "text": (s.get("text") or "").strip(),
                          "len": int(s.get("len") or 0)})
        elif kind == "tool":
            if i % 2 != 1:
                return None
            turns.append({"kind": "tool",
                          "body": (s.get("body") or "").strip(),
                          "len": int(s.get("len") or 0)})
        else:
            return None
    if not turns or turns[0]["kind"] != "assistant":
        return None
    if not all(t["text"] for t in turns if t["kind"] == "assistant"):
        return None
    return turns


def parse_budget_left(body: str):
    """从回包 body 的 [budget] 行解析剩余额度；无该行返回 None。"""
    m = _BUDGET_RE.search(body or "")
    return int(m.group(1)) if m else None


def rewrite_budget_lines(turns, call_need: int) -> int:
    """把每个工具回包里的 [budget] 行改写成 budget_hint_line 的当前档位文案。

    返回改写的行数。无论旧文案是哪一代（单行/分档），重写后都与线上注入
    字节一致——SFT 分布与 RL 分布的提示行同一条生产线。"""
    n = 0
    for t in turns:
        if t["kind"] != "tool":
            continue
        left = parse_budget_left(t["body"])
        if left is None:
            continue
        # 替换串必须走 lambda：budget_hint_line 含 \boxed 等反斜杠序列，
        # re.sub 的字符串替换形态会把 \b 解析成退格符（字节污染）。
        t["body"] = _BUDGET_RE.sub(
            lambda _m, _l=left: budget_hint_line(_l, call_need),
            t["body"], count=1)
        n += 1
    return n


def final_boxed(turns):
    """轨迹的最终 boxed 答案（最后一个 assistant 段）；无则 None。"""
    for t in reversed(turns):
        if t["kind"] == "assistant":
            return extract_last_boxed(t["text"][-300:])
    return None


def synthesize_low_budget(turns, call_need: int, rng: random.Random):
    """在**最后一个工具回包**处截断并合成低预算收尾（cut=last 的信息完备口径）。

    为什么只切最后一刀：截在此处时全部工具结果已在上下文里，原轨迹的最终
    答案在信息上可达——合成样本教的是"把已有结论压缩成收尾"，而不是"没算
    完就猜"。更早的切点会教后者，宁可不要。

    返回新 turns（前缀深拷贝语义无共享可变引用）或 None（无工具段可切）。"""
    last_tool = max((i for i, t in enumerate(turns) if t["kind"] == "tool"),
                    default=None)
    if last_tool is None:
        return None
    answer = final_boxed(turns)
    if not answer:
        return None
    out = [dict(t) for t in turns[:last_tool + 1]]
    # 强警档：剩余额度 < 一次调用开销 → "再调用必然不执行" 文案
    low_left = max(1, call_need // 2)
    body = out[-1]["body"]
    if parse_budget_left(body) is not None:
        body = _BUDGET_RE.sub(lambda _m: budget_hint_line(low_left, call_need),
                              body, count=1)
    else:
        body += budget_hint_line(low_left, call_need)
    out[-1]["body"] = body
    out.append({"kind": "assistant", "text": _WRAP_UP % answer, "len": 0})
    return out


def compose_system_prompt(system_prompt_file=None, max_traj_tokens=8192,
                          answer_reserve=1024, tool_result_max_chars=500,
                          run_info_path=None):
    """与训练/probe 同一入口装配系统提示（含预算段自动注入）。

    返回 (sp, sha6, call_need, src)。优先级：--system_prompt_file >
    --from_run_info > native preset。

    【run_info 路径的注意点】历史 run_info 写在预算段功能之前，其
    config.system_prompt **不含**预算段；这里经 inject_budget_prompt（幂等）
    补注入——结果与未来 RL run 用同一提示文件 + 同预算参数时逐字节一致
    （同一入口、同一函数，不是两处各写一遍）。call_need 也从 run_info 的
    tool_result_max_chars/answer_reserve 推导，少传三个易错参数。"""
    from rlab.config import get_config, inject_budget_prompt
    src = "preset"
    if system_prompt_file:
        with open(system_prompt_file, encoding="utf-8") as f:
            base_sp = f.read().strip()
        src = f"file:{system_prompt_file}"
    elif run_info_path:
        with open(run_info_path, encoding="utf-8") as f:
            ri = json.load(f)
        rc = ri.get("config") or {}
        if not rc.get("system_prompt"):
            raise ValueError(f"[sft_data] {run_info_path} 里没有 config.system_prompt")
        base_sp = rc["system_prompt"]
        max_traj_tokens = int(rc.get("max_traj_tokens") or max_traj_tokens)
        answer_reserve = int(rc.get("answer_reserve") or answer_reserve)
        tool_result_max_chars = int(rc.get("tool_result_max_chars")
                                    or tool_result_max_chars)
        src = f"run_info:{run_info_path}"
    else:
        base_sp = None
    call_need = tool_result_max_chars // 2 + 16 + answer_reserve
    if base_sp is None:
        ov = dict(use_wandb=False, tool_protocol="native",
                  max_traj_tokens=int(max_traj_tokens),
                  answer_reserve=int(answer_reserve),
                  max_context_tokens=int(max_traj_tokens) + 1024,
                  budget_hint=True)
        sp = get_config("retool_math", **ov)["system_prompt"]
    else:
        c = {"system_prompt": base_sp, "budget_hint": True,
             "max_traj_tokens": int(max_traj_tokens),
             "answer_reserve": int(answer_reserve),
             "tool_protocol": "native"}
        inject_budget_prompt(c)
        sp = c["system_prompt"]
    return sp, hashlib.sha1(sp.encode("utf-8")).hexdigest()[:6], call_need, src


def build_dataset(rows, *, call_need, max_per_q=2, synth_per_q=1,
                  max_examples=0, seed=42, min_ok=1, max_ok=0,
                  prefer_short=True):
    """dump 行 → SFT 样本（纯函数，CPU 可测）。返回 (samples, stats)。

    【2026-10-11 选样口径改版】三处新参数，服务"交卷率已接近饱和、该碰解题力"：

      · `min_ok`/`max_ok`：按**该题答对的轨迹条数**（pass rate 的绝对值，不是
        比例）筛题。`max_ok=0` 表示不限上限。默认 `min_ok=1, max_ok=0` 与旧行为
        等价（能进 samples 的题必然已有 ≥1 条正确轨迹）。
        **设 `max_ok=N<num_pre_Q` = rejection sampling（STaR 机制）**：只保留
        "模型不是每次都做对"的近界题，排除已掌握（ok=num_pre_Q，信号饱和）与
        零正确（无正例可用）两端。这是唯一能真实抬高 `P(correct)` 的选样方式。
      · `prefer_short`：旧行为是每题**按 clen 升序取前 max_per_q 条**——长度早已
        压住（SFT −17%、RL −21%），该偏好现在只会**优先剔除长轨迹（=难题的正确
        解）**，纯伤害解题力。置 False 改为在候选里**均匀随机抽**（seeded）。
      · `stats["pass_hist"]`：每题答对条数的直方图（key=ok 数，value=题数）。
        它是"该不该开近界选样"的判据：若 `ok=num_pre_Q` 占绝大多数 ⇒ 池子已
        饱和、近界题太少，先别指望 ② 路线的收益。
    """
    rng = random.Random(seed)
    by_q = {}
    # 每题两条计数：全部落盘 attempt 数、其中答对条数（pass rate 的分子/分母）。
    n_att, n_ok_q = {}, {}
    stats = {"rows": 0, "ok": 0, "bad_struct": 0, "no_boxed": 0,
             "rewritten_budget_lines": 0}
    for row in rows:
        stats["rows"] += 1
        _qk0 = row.get("qk") or hashlib.sha1(
            str(row.get("Q", "")).encode("utf-8")).hexdigest()[:12]
        n_att[_qk0] = n_att.get(_qk0, 0) + 1
        if (row.get("acc") or 0) > 0:
            n_ok_q[_qk0] = n_ok_q.get(_qk0, 0) + 1
        if not row_ok(row):
            continue
        stats["ok"] += 1
        turns = split_turns(row)
        if turns is None:
            stats["bad_struct"] += 1
            continue
        if final_boxed(turns) is None:
            stats["no_boxed"] += 1
            continue
        stats["rewritten_budget_lines"] += rewrite_budget_lines(turns, call_need)
        qk = _qk0
        clen = int(row.get("clen") or 0)
        source = "real_tool" if int(row.get("code_used") or 0) > 0 else "zero_call"
        by_q.setdefault(qk, []).append(
            {"qk": qk, "Q": row.get("Q"), "A": row.get("A"),
             "source": source, "clen": clen, "turns": turns})
    # ---- 近界筛选（按该题答对条数）----
    # 直方图覆盖**所有落盘题目**（含 ok=0——它们不在 n_ok_q 里，漏了就看不到
    # "整组全错"这一端，而那正是 pass rate 分布的另一半）。
    stats["pass_hist"] = {}
    for _qk0 in n_att:
        _n = n_ok_q.get(_qk0, 0)
        stats["pass_hist"][_n] = stats["pass_hist"].get(_n, 0) + 1
    stats["pass_hist_num_pre_Q"] = max(n_att.values()) if n_att else 0
    _band_q = 0
    if min_ok > 1 or max_ok > 0:
        _keep = {q for q, n in n_ok_q.items()
                 if n >= int(min_ok) and (int(max_ok) <= 0 or n <= int(max_ok))}
        _band_q = len(by_q) - len([q for q in by_q if q in _keep])
        by_q = {q: v for q, v in by_q.items() if q in _keep}
    stats["band_skipped_questions"] = _band_q
    samples = []
    for qk, cands in by_q.items():
        if prefer_short:
            cands.sort(key=lambda c: c["clen"])
            kept = cands[:max(1, int(max_per_q))]
        else:
            # 均匀随机抽（不按长度偏好）：长轨迹 = 难题的正确解，不该被系统性剔除
            k = min(len(cands), max(1, int(max_per_q)))
            kept = rng.sample(cands, k)
        samples.extend(kept)
        n_synth = 0
        for c in kept:
            if c["source"] != "real_tool" or n_synth >= synth_per_q:
                continue
            sv = synthesize_low_budget(c["turns"], call_need, rng)
            if sv is not None:
                samples.append({**c, "source": "low_budget", "turns": sv})
                n_synth += 1
    rng.shuffle(samples)
    if max_examples > 0:
        samples = samples[:max_examples]
    stats["n_questions"] = len(by_q)
    stats["n_samples"] = len(samples)
    stats["per_source"] = {s: sum(1 for x in samples if x["source"] == s)
                           for s in _SOURCES}
    return samples, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="SFT 冷启动数据生成（docs/08 阶段①）")
    ap.add_argument("--dump", action="append", required=True,
                    help="traj.jsonl 路径，可多次（多 run 混合）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--system_prompt_file", default=None,
                    help="必须与后续 RL 训练同一份（缺省 = native preset）")
    ap.add_argument("--from_run_info", default=None,
                    help="从 run_info.json 取系统提示与预算三件套（推荐：p6/p7 的"
                         " run_info；其提示不含预算段，经 inject_budget_prompt"
                         " 幂等补注入后与后续 RL 同配方逐字节一致）。"
                         "优先级低于 --system_prompt_file")
    ap.add_argument("--max_traj_tokens", type=int, default=8192)
    ap.add_argument("--answer_reserve", type=int, default=1024)
    ap.add_argument("--tool_result_max_chars", type=int, default=500,
                    help="与线上同值：推导 call_need = chars//2+16+answer_reserve")
    ap.add_argument("--max_per_q", type=int, default=2)
    ap.add_argument("--synth_per_q", type=int, default=1)
    ap.add_argument("--min_ok", type=int, default=1,
                    help="按该题答对条数筛题的下限（1=不限，能进样本的题必然≥1）")
    ap.add_argument("--max_ok", type=int, default=0,
                    help="按该题答对条数筛题的上限（0=不限）。设 <num_pre_Q 即"
                         "开启近界选样（rejection sampling / STaR）：排除已掌握的"
                         "题（信号饱和），只留'不是每次都对'的题")
    ap.add_argument("--no_prefer_short", action="store_true",
                    help="关闭'偏好短轨迹'。长度已压住（SFT −17%%/RL −21%%），"
                         "该偏好现在只会优先剔除长轨迹=难题的正确解，纯伤害解题力")
    ap.add_argument("--max_examples", type=int, default=0, help="0=不限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    sp, sp_sha, call_need, sp_src = compose_system_prompt(
        args.system_prompt_file, args.max_traj_tokens, args.answer_reserve,
        args.tool_result_max_chars, args.from_run_info)
    samples, stats = build_dataset(load_dump_rows(args.dump),
                                   call_need=call_need,
                                   max_per_q=args.max_per_q,
                                   synth_per_q=args.synth_per_q,
                                   max_examples=args.max_examples,
                                   seed=args.seed,
                                   min_ok=args.min_ok,
                                   max_ok=args.max_ok,
                                   prefer_short=not args.no_prefer_short)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(
                {"qk": s["qk"], "Q": s["Q"], "A": s["A"], "source": s["source"],
                 "system_prompt": sp, "sp_sha": sp_sha,
                 "turns": [{"kind": t["kind"],
                            **({"text": t["text"]} if t["kind"] == "assistant"
                               else {"body": t["body"]})}
                           for t in s["turns"]]},
                ensure_ascii=False) + "\n")
    print(f"[sft_data] 读 {stats['rows']} 行 dump → ok {stats['ok']} → "
          f"{stats['n_samples']} 条样本 / {stats['n_questions']} 题", flush=True)
    print(f"[sft_data] 分类: {stats['per_source']}；结构非法跳过 "
          f"{stats['bad_struct']}，无 boxed 跳过 {stats['no_boxed']}，"
          f"预算行改写 {stats['rewritten_budget_lines']} 行", flush=True)
    # 【pass-rate 直方图】决定"要不要开近界选样（--max_ok）"的唯一判据：
    # 若 ok=num_pre_Q 的题占绝大多数 ⇒ 池子已饱和、近界题太少，② 路线收益有限。
    _ph = stats.get("pass_hist") or {}
    if _ph:
        _np = stats.get("pass_hist_num_pre_Q") or max(_ph)
        _tot = sum(_ph.values())
        _sat = _ph.get(_np, 0)
        _zero = _ph.get(0, 0)
        _band = sum(v for k, v in _ph.items() if 1 <= k < _np)
        print(f"[sft_data] pass-rate 直方图（每题答对条数 / 该题数，共 {_tot} 题；"
              f"每题 attempts≈{_np}）:", flush=True)
        print("           " + "  ".join(
            f"{k}→{_ph[k]}" for k in sorted(_ph)), flush=True)
        print(f"[sft_data]   已掌握(ok={_np}) {_sat}/{_tot} = "
              f"{_sat / max(1, _tot) * 100:.1f}% ｜ 全错(ok=0) {_zero} ｜ "
              f"**近界(1≤ok<{_np}) {_band}** "
              f"= {_band / max(1, _tot) * 100:.1f}%", flush=True)
        if _band < 0.2 * _tot:
            print("[sft_data]   ⇒ 近界题 <20%：`--max_ok` 近界选样收益有限"
                  "（池子已饱和），优先走交卷率路线", flush=True)
        else:
            print(f"[sft_data]   ⇒ 近界题充足，可试 `--max_ok {_np - 1}`"
                  "（rejection sampling，只保留不是每次都对的题）", flush=True)
    if stats.get("band_skipped_questions"):
        print(f"[sft_data] 近界筛选丢掉 {stats['band_skipped_questions']} 题"
              f"（min_ok={args.min_ok}, max_ok={args.max_ok}）", flush=True)
    print(f"[sft_data] 系统提示 sp{sp_sha}（来源 {sp_src}，含预算段）已写入每行；"
          f"后续 RL 的签名必须出现同一 -sp{sp_sha}", flush=True)
    if stats["per_source"]["low_budget"] == 0:
        print("[sft_data][警告] low_budget 合成 0 条——治 C_wasted/B_call 的"
              "正例缺失，检查 dump 里是否有 real_tool 轨迹", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
