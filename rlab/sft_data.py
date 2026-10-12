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
  · 固定 k 的旧 RL dump 可用 `--max_rate 0.875` 做近界选样；
  · `sft_collect` 的广覆盖 dump 是自适应采样（成功题1次、失败题多次），尝试数由
    结果决定，禁止据此估通过率/使用 `--max_rate`；它靠广题面 + 失败题追加探索
    提供纠错轨迹。
    **口径必须用比例**：合并 dump 后每题 attempts 8~40 不等，绝对条数不可比
    （`ok=10` 在 40 次里是 25%，在 8 次里是全对）。
  · `--ref_src N`：pass-rate 的参考 run（多 run 合并时**必须限定**，否则同一题
    被反复采样、统计全假）。缺省自动取行数最多的 run 并打印报出。
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


def load_dump_rows(paths, tag=False):
    """逐行读 dump jsonl（坏行跳过并计数，绝不因此中断）。

    `tag=True` 时产出 `(src_idx, row)`，`src_idx` = `paths` 下标（= 哪个 run）。
    调用方存入行内私有键 `_src`——pass-rate 必须**按 run 分开算**（见
    build_dataset 的 `ref_src`），样本构造只显式取字段，`_src` 不外泄到 jsonl。"""
    n_bad = 0
    for _idx, p in enumerate(paths):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                if tag:
                    row["_src"] = _idx
                yield row
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
                  max_examples=0, seed=42, min_rate=0.0, max_rate=1.0,
                  prefer_short=True, ref_src=None, min_att=4,
                  keep_unknown=True, expected_sp_sha=None, synth_prob=1.0):
    """dump 行 → SFT 样本（纯函数，CPU 可测）。返回 (samples, stats)。

    【2026-10-11 选样口径改版】服务"交卷率已接近饱和、该碰解题力"：

      · `min_rate`/`max_rate`：按**该题在参考 run 内的通过率**筛题。
        **必须用比例而非绝对条数**：合并 dump 后每题 attempts 是 8~40 不等
        （num_pre_Q × 被采样到的次数），同一个 `ok=10` 在 attempts=40 里是 25%
        （近界、该留）、在 attempts=8 里是全对（该剔）——绝对阈值跨题不可比，
        2026-10-11 实机验证过这一点。
        **`max_rate < 1.0` 即近界选样**（rejection sampling / STaR）：排除已掌握
        的题（信号饱和），只留"不是每次都做对"的题。这是唯一能真实抬高
        `P(correct)` 的选样口径。
      · `prefer_short`：旧行为是每题按 clen 升序取前 max_per_q 条——长度早已
        压住（SFT −17%、RL −21%），该偏好现在只会优先剔除长轨迹（=难题的正确
        解），纯伤害解题力。置 False 改为均匀随机抽（seeded）。
      · `ref_src`：**pass-rate 的参考 run 下标**（`_src`）。多 run dump 合并后
        同一 qk 会被反复采样，必须限定一个 run 才能得到可比口径。
        **⚠ 缺省 `None` 时取行数最多的 run 并在打印里报出用的是谁**——绝不在
        混合池上统计（那会得到"已掌握 0%、近界 92%" 这种全假读数，实机踩过）。
      · `min_att`：参考 run 内 attempts < min_att 的题不进直方图/不参与筛选
        （样本不足时通过率不可信）。
      · `keep_unknown`（近界筛选时才有意义）：参考 run 里**没采到**的题
        （通过率未知）**默认保留**——筛选目的是剔除"有证据表明已掌握"的题，
        而"未知" ≠ "已掌握"，静默丢掉它们是最糟的默认。实机形态：741 题里
        341 题（46%）不在 p9 的 400 题内。置 False = 严格模式（只留参考 run 里
        有实测通过率的题），会被 stats["band_skipped_unknown"] 单独计数。
      · `stats["pass_hist"]`：按**换算到 num_pre_Q 刻度**的通过率分档
        （`round(rate × num_pre_Q)`，key 0..num_pre_Q）。这样不同 attempts 的题
        可比，也和 RL record 里"8 条里对几条"的直觉一致。
    """
    rng = random.Random(seed)
    if not 0.0 <= float(synth_prob) <= 1.0:
        raise ValueError(f"synth_prob 必须在 [0,1]，收到 {synth_prob}")
    by_q = {}
    # 每题**分 run** 计数：n_att[src][qk]、n_ok[src][qk]。混合池上算通过率是
    # 错的（不同 run 是不同策略，且同一题被重复采了 N 遍）。
    n_att, n_ok_q = {}, {}
    stats = {"rows": 0, "ok": 0, "bad_struct": 0, "no_boxed": 0,
             "rewritten_budget_lines": 0, "adaptive_collect": False}
    for row in rows:
        stats["rows"] += 1
        if row.get("collect_phase") in ("first", "retry"):
            stats["adaptive_collect"] = True
        _row_sp = row.get("sp_sha")
        if row.get("collect_phase") in ("first", "retry") and not _row_sp:
            raise ValueError(
                f"[sft_data] collector dump 第 {stats['rows']} 行缺 sp_sha；"
                "无法证明生成提示与SFT编码提示一致，拒绝混入")
        if expected_sp_sha and _row_sp and _row_sp != expected_sp_sha:
            raise ValueError(
                f"[sft_data] dump 第 {stats['rows']} 行按 sp{_row_sp} 生成，"
                f"本次 SFT 将按 sp{expected_sp_sha} 编码；提示错档，拒绝混入")
        _qk0 = row.get("qk") or hashlib.sha1(
            str(row.get("Q", "")).encode("utf-8")).hexdigest()[:12]
        _src = row.get("_src", 0)
        n_att.setdefault(_src, {})
        n_ok_q.setdefault(_src, {})
        n_att[_src][_qk0] = n_att[_src].get(_qk0, 0) + 1
        if (row.get("acc") or 0) > 0:
            n_ok_q[_src][_qk0] = n_ok_q[_src].get(_qk0, 0) + 1
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
             "source": source, "clen": clen, "turns": turns,
             "collect_phase": row.get("collect_phase")})
    # ---- 参考 run 选择 + 通过率（比例口径）----
    if stats["adaptive_collect"] and (min_rate > 0.0 or max_rate < 1.0):
        raise ValueError(
            "[sft_data] collector dump 是自适应采样：首轮成功题1次、失败题3~5次，"
            "attempts 由结果决定，不能据此做 --min_rate/--max_rate 近界筛选；"
            "请移除 rate 过滤，近界率只用独立 fixed-k run/probe")
    stats["src_counts"] = {k: sum(v.values()) for k, v in n_att.items()}
    if ref_src is None and n_att:
        ref_src = max(stats["src_counts"], key=lambda k: stats["src_counts"][k])
    stats["ref_src"] = ref_src
    _ra = n_att.get(ref_src, {})
    _ro = n_ok_q.get(ref_src, {})
    _scale = max(_ra.values()) if _ra else 0
    rate = {}
    for _qk0, _a in _ra.items():
        rate[_qk0] = _ro.get(_qk0, 0) / _a if _a else 0.0
    # 直方图：按换算刻度分档（att 不同也可比，且与 num_pre_Q=8 的直觉一致）
    stats["pass_hist"] = {}
    _scale_q = 8 if _scale else 0
    for _qk0, _a in _ra.items():
        if _a < int(min_att):
            continue
        _b = min(_scale_q, int(round(rate[_qk0] * _scale_q)))
        stats["pass_hist"][_b] = stats["pass_hist"].get(_b, 0) + 1
    stats["pass_hist_scale"] = _scale_q
    stats["pass_hist_num_pre_Q"] = _scale
    _band_q = _unknown_dropped = 0
    if min_rate > 0.0 or max_rate < 1.0:
        _keep = set()
        for _q in by_q:
            _a = _ra.get(_q, 0)
            if _a < int(min_att):
                # 【通过率未知的题】参考 run 里没采到（或样本不足）。
                # 缺省**保留**：筛选的目的是剔除"有证据表明已掌握"的题，而"未知"
                # ≠ "已掌握"。静默丢掉它们是最糟的默认（实机：741 题里 341 题
                # 不在参考 run 内，占 46%）。`--drop_unknown` 可改为严格模式。
                if keep_unknown:
                    _keep.add(_q)
                else:
                    _unknown_dropped += 1
                continue
            _r = rate[_q]
            if _r >= float(min_rate) - 1e-9 and _r <= float(max_rate) + 1e-9:
                _keep.add(_q)
        _band_q = len([q for q in by_q if q not in _keep]) - _unknown_dropped
        by_q = {q: v for q, v in by_q.items() if q in _keep}
    stats["band_skipped_questions"] = _band_q
    stats["band_skipped_unknown"] = _unknown_dropped
    # 入样且通过率未知的题数（诚实记账：这部分未经过近界验证）。
    # **无条件按最终 by_q 统计**——严格模式下自然为 0，默认模式下自然计数，
    # 且两种模式下覆盖率打印用的是同一个量。
    stats["n_unknown_q"] = sum(1 for _q in by_q
                               if _ra.get(_q, 0) < int(min_att))
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
            if (c["source"] != "real_tool" or n_synth >= synth_per_q
                    or (float(synth_prob) < 1.0
                        and rng.random() >= float(synth_prob))):
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
    stats["per_collect_phase"] = {
        p: sum(1 for x in samples
               if x.get("collect_phase") == p and x["source"] != "low_budget")
        for p in ("first", "retry")}
    return samples, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="SFT 冷启动数据生成（docs/08 阶段①）")
    ap.add_argument("--dump", action="append", required=True,
                    help="traj.jsonl 路径，可多次（多 run 混合）。**顺序有语义**："
                         "pass-rate 的参考 run 按此下标索引（--ref_src），"
                         "建议把'下个 RL 起点最接近的那个 run'放最后并显式指定")
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
    ap.add_argument("--synth_prob", type=float, default=1.0,
                    help="每个可合成候选生成 low_budget 样本的概率；默认1保持旧行为。"
                         "广覆盖集建议0.5，使 low_budget 约占最终样本1/3")
    ap.add_argument("--min_rate", type=float, default=0.0,
                    help="按参考 run 内该题**通过率**筛题的下限（0=不限）")
    ap.add_argument("--max_rate", type=float, default=1.0,
                    help="按参考 run 内该题**通过率**筛题的上限（1=不限）。"
                         "设 <1 即开启近界选样（rejection sampling / STaR）：排除"
                         "已掌握的题（信号饱和），只留'不是每次都做对'的题。"
                         "**必须用比例**：合并 dump 后每题 attempts 8~40 不等，"
                         "绝对条数阈值跨题不可比")
    ap.add_argument("--ref_src", type=int, default=None,
                    help="pass-rate 的参考 run 下标（--dump 的给出顺序，0 起）。"
                         "None=自动取行数最多的那个 run 并在打印里报出。"
                         "**多 run 合并 dump 时必须限定一个 run**，否则该题被"
                         "重复采 N 遍，统计出的通过率全假")
    ap.add_argument("--min_att", type=int, default=4,
                    help="参考 run 内该题 attempts 下限（低于此不进直方图/不参与"
                         "近界筛选：样本不足时通过率不可信）")
    ap.add_argument("--drop_unknown", action="store_true",
                    help="近界筛选时丢掉'参考 run 里没采到'的题（通过率未知）。"
                         "**默认不丢**——筛选是剔除有证据表明已掌握的题，而'未知'"
                         "≠'已掌握'；静默丢掉是最糟的默认（实机：741 题里 341 题"
                         "不在参考 run 内，占 46%%）")
    ap.add_argument("--no_prefer_short", action="store_true",
                    help="关闭'偏好短轨迹'。长度已压住（SFT −17%%/RL −21%%），"
                         "该偏好现在只会优先剔除长轨迹=难题的正确解，纯伤害解题力")
    ap.add_argument("--max_examples", type=int, default=0, help="0=不限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    sp, sp_sha, call_need, sp_src = compose_system_prompt(
        args.system_prompt_file, args.max_traj_tokens, args.answer_reserve,
        args.tool_result_max_chars, args.from_run_info)
    samples, stats = build_dataset(load_dump_rows(args.dump, tag=True),
                                   call_need=call_need,
                                   max_per_q=args.max_per_q,
                                   synth_per_q=args.synth_per_q,
                                   max_examples=args.max_examples,
                                   seed=args.seed,
                                   min_rate=args.min_rate,
                                   max_rate=args.max_rate,
                                   prefer_short=not args.no_prefer_short,
                                   ref_src=args.ref_src,
                                   min_att=args.min_att,
                                   keep_unknown=not args.drop_unknown,
                                   expected_sp_sha=sp_sha,
                                   synth_prob=args.synth_prob)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(
                {"qk": s["qk"], "Q": s["Q"], "A": s["A"], "source": s["source"],
                 "collect_phase": s.get("collect_phase"),
                 "system_prompt": sp, "sp_sha": sp_sha,
                 "turns": [{"kind": t["kind"],
                            **({"text": t["text"]} if t["kind"] == "assistant"
                               else {"body": t["body"]})}
                           for t in s["turns"]]},
                ensure_ascii=False) + "\n")
    print(f"[sft_data] 读 {stats['rows']} 行 dump → ok {stats['ok']} → "
          f"{stats['n_samples']} 条样本 / {stats['n_questions']} 题", flush=True)
    print(f"[sft_data] 分类: {stats['per_source']}；采集来源(不含合成): "
          f"{stats.get('per_collect_phase')}；结构非法跳过 "
          f"{stats['bad_struct']}，无 boxed 跳过 {stats['no_boxed']}，"
          f"预算行改写 {stats['rewritten_budget_lines']} 行", flush=True)
    # 【覆盖率】直方图只覆盖参考 run 内实测的题；入样题里不在其中的部分通过率
    # **未知**。多 run 合并时该差额可到一半（实机：741 入样 vs 参考 run 内 400
    # 题 ⇒ 46% 未知），必须显式报出——否则会被误当成"全部题都按通过率筛过了"。
    _unk = stats.get("n_unknown_q", 0)
    _nq = stats.get("n_questions", 0)
    if _unk and _nq and not stats.get("adaptive_collect"):
        print(f"[sft_data] ⚠ 入样 {_nq} 题中有 {_unk} 题"
              f"（{_unk / max(1, _nq) * 100:.0f}%）通过率**未知**（不在参考 run "
              f"内）→ 未经过近界验证；只留有实测通过率的题请加 --drop_unknown",
              flush=True)
    # 【pass-rate 直方图】决定"要不要开近界选样（--max_rate）"的唯一判据。
    # 两条口径纪律（都是 2026-10-11 实机踩出来的）：
    #   ① **限定参考 run**：混合多 run 后同一题被反复采样，合并统计全假；
    #   ② **用比例不用绝对条数**：每题 attempts 是 8~40 不等，`ok=10` 在 40 次里
    #      是 25%（近界、该留）、在 8 次里是全对（该剔）——绝对阈值跨题不可比。
    # 直方图即按比例换算到 num_pre_Q=8 刻度，与本项目"8 条里对几条"的直觉一致。
    _ph = stats.get("pass_hist") or {}
    if stats.get("adaptive_collect"):
        print("[sft_data] 自适应 collector dump：首轮成功题1次、失败题3~5次，"
              "attempts 由结果决定；本表不把直方图解释为通过率，也禁止 --max_rate。",
              flush=True)
        _ph = {}
    if _ph:
        _sc = stats.get("pass_hist_scale") or 8
        _tot = sum(_ph.values())
        _sat = _ph.get(_sc, 0)
        _zero = _ph.get(0, 0)
        _band = sum(v for k, v in _ph.items() if 0 < k < _sc)
        _ref = stats.get("ref_src")
        print(f"[sft_data] pass-rate 直方图（参考 run #{_ref} 内，按通过率换算到 "
              f"{_sc} 刻度；共 {_tot} 题，该 run 每题最多 attempts "
              f"{stats.get('pass_hist_num_pre_Q')}）:", flush=True)
        print("           " + "  ".join(
            f"{k}/{_sc}→{_ph[k]}" for k in sorted(_ph)), flush=True)
        # 档位语义（必须说清：档是四舍五入的结果，不是精确的 0/1）
        #   key = round(rate × scale) ⇒ key=k 表示 rate ∈ [(k-0.5)/scale, (k+0.5)/scale)
        #   档 0 = 通过率 < 0.5/scale（近全错，非"精确 0"）；档 scale = ≥(scale-0.5)/scale
        print(f"[sft_data]   近全错档(0，通过率<{0.5 / _sc:.1%}) {_zero}/{_tot} = "
              f"{_zero / max(1, _tot) * 100:.1f}% ｜ 已掌握档({_sc}，通过率≥"
              f"{(_sc - 0.5) / _sc:.1%}) {_sat}/{_tot} = "
              f"{_sat / max(1, _tot) * 100:.1f}% ｜ **近界(中间档) {_band}** "
              f"= {_band / max(1, _tot) * 100:.1f}%", flush=True)
        print(f"[sft_data]   各 run 行数（参考 run 取行数最多者，可用 --ref_src "
              f"显式指定）: {stats.get('src_counts')}", flush=True)
        if _band < 0.2 * _tot:
            print("[sft_data]   ⇒ 近界题 <20%：近界选样收益有限（池子已饱和），"
                  "优先走交卷率路线", flush=True)
        else:
            print(f"[sft_data]   ⇒ 近界题充足（{_band} 题），可试 "
                  f"`--max_rate 0.875`（=7/8，只留不是每次都对的题）", flush=True)
    if stats.get("band_skipped_questions") or stats.get("band_skipped_unknown"):
        print(f"[sft_data] 近界筛选（min_rate={args.min_rate}, "
              f"max_rate={args.max_rate}）：按通过率剔除 "
              f"{stats.get('band_skipped_questions', 0)} 题，丢掉'通过率未知'"
              f"（参考 run 未采到）{stats.get('band_skipped_unknown', 0)} 题"
              f"（--drop_unknown 才丢）", flush=True)
    # 【通过率覆盖率】诚实记账：多少题是**真的**按通过率筛过的
    # （打印在 build_dataset 的直方图段已做过一次，这里不重复）
    print(f"[sft_data] 系统提示 sp{sp_sha}（来源 {sp_src}，含预算段）已写入每行；"
          f"后续 RL 的签名必须出现同一 -sp{sp_sha}", flush=True)
    if stats["per_source"]["low_budget"] == 0:
        print("[sft_data][警告] low_budget 合成 0 条——治 C_wasted/B_call 的"
              "正例缺失，检查 dump 里是否有 real_tool 轨迹", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
