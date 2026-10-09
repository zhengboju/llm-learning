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
      --dump rlab_new/native_p6_best/traj.jsonl --dump rlab_new/native_p7_best/traj.jsonl \
      --out rlab_out/sft_budget_v1.jsonl \
      --system_prompt_file rlab/prompts/retool_math_concise.txt \
      --max_traj_tokens 8192 --answer_reserve 1024
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
                          answer_reserve=1024):
    """与训练/probe 同一入口装配系统提示（含预算段自动注入），返回 (sp, sha6)。"""
    from rlab.config import get_config
    ov = dict(use_wandb=False, tool_protocol="native",
              max_traj_tokens=int(max_traj_tokens),
              answer_reserve=int(answer_reserve),
              max_context_tokens=int(max_traj_tokens) + 1024,
              budget_hint=True)
    if system_prompt_file:
        with open(system_prompt_file, encoding="utf-8") as f:
            ov["system_prompt"] = f.read().strip()
    sp = get_config("retool_math", **ov)["system_prompt"]
    return sp, hashlib.sha1(sp.encode("utf-8")).hexdigest()[:6]


def build_dataset(rows, *, call_need, max_per_q=2, synth_per_q=1,
                  max_examples=0, seed=42):
    """dump 行 → SFT 样本（纯函数，CPU 可测）。返回 (samples, stats)。

    每题（qk）限额且**偏好更短**的轨迹——简洁正是要教的行为，长轨迹留给
    RL 自己探索。low_budget 每题至多 synth_per_q 条，从入选的 real_tool
    轨迹派生。"""
    rng = random.Random(seed)
    by_q = {}
    stats = {"rows": 0, "ok": 0, "bad_struct": 0, "no_boxed": 0,
             "rewritten_budget_lines": 0}
    for row in rows:
        stats["rows"] += 1
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
        qk = row.get("qk") or hashlib.sha1(
            str(row.get("Q", "")).encode("utf-8")).hexdigest()[:12]
        clen = int(row.get("clen") or 0)
        source = "real_tool" if int(row.get("code_used") or 0) > 0 else "zero_call"
        by_q.setdefault(qk, []).append(
            {"qk": qk, "Q": row.get("Q"), "A": row.get("A"),
             "source": source, "clen": clen, "turns": turns})
    samples = []
    for qk, cands in by_q.items():
        cands.sort(key=lambda c: c["clen"])
        kept = cands[:max(1, int(max_per_q))]
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
    ap.add_argument("--max_traj_tokens", type=int, default=8192)
    ap.add_argument("--answer_reserve", type=int, default=1024)
    ap.add_argument("--tool_result_max_chars", type=int, default=500,
                    help="与线上同值：推导 call_need = chars//2+16+answer_reserve")
    ap.add_argument("--max_per_q", type=int, default=2)
    ap.add_argument("--synth_per_q", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=0, help="0=不限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    call_need = args.tool_result_max_chars // 2 + 16 + args.answer_reserve
    sp, sp_sha = compose_system_prompt(args.system_prompt_file,
                                       args.max_traj_tokens,
                                       args.answer_reserve)
    samples, stats = build_dataset(load_dump_rows(args.dump),
                                   call_need=call_need,
                                   max_per_q=args.max_per_q,
                                   synth_per_q=args.synth_per_q,
                                   max_examples=args.max_examples,
                                   seed=args.seed)
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
    print(f"[sft_data] 系统提示 sp{sp_sha}（含预算段）已写入每行；"
          f"后续 RL 的签名必须出现同一 -sp{sp_sha}", flush=True)
    if stats["per_source"]["low_budget"] == 0:
        print("[sft_data][警告] low_budget 合成 0 条——治 C_wasted/B_call 的"
              "正例缺失，检查 dump 里是否有 real_tool 轨迹", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
