# -*- coding: utf-8 -*-
"""rlab/probe_difficulty.py — 离线难度预探针（训练前一次性估计每题通过率）。

背景（2026-09-10）：retool_math 丢弃率 81% 的主体是全错组 (1-p)^8；题目级
QuestionScheduler 只能出清"在线观察到连续零方差"的题（每题先烧 streak×n 条
轨迹才出局），离线探针在训练开始前用 k 条样本把 p≈0 / p≈1 的题一次性出清——
这是"期望轨迹成本 ≈1/p 只能靠改分布切割"的直接解法（与在线调度互补，两级
过滤都不碰 loss 协议）。

产物：jsonl 表，每题一行 {Q, A, k, n_correct, pass_rate, fmt_rate, trunc_rate,
avg_clen, avg_code_ok}。训练时 `--difficulty_path <该文件>` 即启用过滤
（保留 pass_rate ∈ (0,1)，difficulty_band 可调，见 config.py）。

判别统计（决定"丢弃率高该修 prompt 还是修数据"）：
  - fmt_rate 低（大量轨迹无 \\boxed{}）→ 协议/prompt 失败主导，few-shot、
    轮数预算等生成层手段有救；
  - fmt_rate 高但 pass_rate 低（规规矩矩答完就是错）→ 能力失败主导，
    prompt 无杠杆，靠难度过滤 + 在线调度 + 更强基座。

用法（pod，单卡；16.5k 题 × k=4 全量约数小时，建议先用 --max_questions 试跑）：
    CUDA_VISIBLE_DEVICES=0 python -m rlab.probe_difficulty \
        --model_path /root/Qwen2.5-3B --k 4 \
        --out rlab_out/difficulty_probe.jsonl
    # 断点续跑：同 --out 重跑即自动跳过已有题（逐题追加写，崩溃可续）

    bash rlab/run_gsm8k.sh retool_math /root/Qwen2.5-3B \
        --difficulty_path rlab_out/difficulty_probe.jsonl
"""
import argparse
import json
import os
import random
import sys
import time

# vLLM 环境必须在 import vllm 之前设置（与 rollout.py 同款：RPC 模式权重同步 stall 的教训）
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 允许 `python rlab/probe_difficulty.py` 直接跑（sys.path[0] 会变成 rlab/ 目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def aggregate_rows(rows: list) -> list:
    """纯函数（CPU 可测）：逐轨迹记录 -> 逐题通过率统计。

    rows 每项：{Q, A, acc, fmt, trunc, clen}（acc=±1，fmt=±1 有无 boxed，
    trunc=末段是否被轮长切断，clen=轨迹全长 token）。
    返回每题 {Q, A, k, n_correct, pass_rate, fmt_rate, trunc_rate,
    avg_clen, avg_code_ok}，顺序与 rows 中题首次出现顺序一致。"""
    order, by_q = [], {}
    for r in rows:
        q = str(r["Q"])
        if q not in by_q:
            by_q[q] = {"A": r.get("A"), "acc": [], "fmt": [], "trunc": [],
                       "clen": [], "code_ok": []}
            order.append(q)
        d = by_q[q]
        d["acc"].append(int(r["acc"] > 0))
        d["fmt"].append(int(r["fmt"] > 0))
        d["trunc"].append(int(bool(r["trunc"])))
        d["clen"].append(int(r["clen"]))
        d["code_ok"].append(int(r.get("code_ok", 0)))

    out = []
    for q in order:
        d = by_q[q]
        k = len(d["acc"])
        nc = sum(d["acc"])
        out.append({"Q": q, "A": d["A"], "k": k, "n_correct": nc,
                    "pass_rate": nc / k,
                    "fmt_rate": sum(d["fmt"]) / k,
                    "trunc_rate": sum(d["trunc"]) / k,
                    "avg_clen": sum(d["clen"]) / k,
                    "avg_code_ok": sum(d["code_ok"]) / k})
    return out


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def summarize(all_rows: list, per_q: list) -> str:
    """纯函数：全量判别统计 + 难度分布直方（probe 的结论输出）。"""
    n = len(all_rows)
    no_boxed = sum(1 for r in all_rows if r["fmt"] <= 0) / max(1, n)
    trunc = sum(1 for r in all_rows if r["trunc"]) / max(1, n)
    p0 = sum(1 for r in per_q if r["n_correct"] == 0)
    p1 = sum(1 for r in per_q if r["n_correct"] == r["k"])
    mid = len(per_q) - p0 - p1
    lines = [
        f"[probe] 题数 {len(per_q)} × k={per_q[0]['k'] if per_q else 0} 轨迹 {n} 条",
        f"[probe] 难度分布: 全错(0/4类) {p0} | 可学(0<rate<1) {mid} | 全对 {p1}",
        f"        -> 训练池预期 ≈{mid} 题（丢弃率主体 = 全错组 (1-p)^8，p≈0 题出清即切割）",
        f"[probe] 判别统计: 无 boxed {_fmt_pct(no_boxed)}（协议/prompt 失败签名，"
        f"高则先修生成层）/ 末段截断 {_fmt_pct(trunc)}",
    ]
    if no_boxed < 0.2 and p0 > len(per_q) * 0.5:
        lines.append("[probe] 判读: 无 boxed 少 + 全错题多 → 能力失败主导，"
                     "prompt 层无杠杆，直接启用 --difficulty_path 过滤")
    elif no_boxed >= 0.2:
        lines.append("[probe] 判读: 无 boxed 占比高 → 协议/prompt 失败显著，"
                     "先做 few-shot 示范 / 放宽轮数预算再评估过滤收益")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="离线难度预探针（retool_math 训练池）")
    ap.add_argument("--model_path", default="/root/Qwen2.5-3B",
                    help="用哪个模型探（base 或某 checkpoint；换模型需重探）")
    ap.add_argument("--k", type=int, default=4, help="每题采样轨迹数（4 的估计噪声大，预算够建议 8）")
    ap.add_argument("--out", default="rlab_out/difficulty_probe.jsonl",
                    help="通过率表输出路径（追加写，重跑自动跳过已有题）")
    ap.add_argument("--questions_per_wave", type=int, default=16,
                    help="每波并采题数（vLLM 并发 = 题数×k）")
    ap.add_argument("--max_questions", type=int, default=0,
                    help="最多探多少题（0=全池）；打乱后取前 N，试跑/抽样用")
    ap.add_argument("--data_task", default=None,
                    help="覆盖数据集（默认取 retool_math preset 的 dapo_math）")
    ap.add_argument("--temp", type=float, default=None,
                    help="覆盖采样温度（默认与训练一致 = retool_math 1.0）")
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chat_template_kwargs", default=None,
                    help='JSON dict 透传 apply_chat_template；Qwen3.5 系必传 '
                         '\'{"enable_thinking": false}\'（不关 thinking 会烧穿单轮预算）')
    # 预算三件套（4B 探针 v2 实锤：3 轮×1024 下截断 86.2%，可学带被预算压瘪——
    # 探针必须先过"轨迹完整"关，难度分布才有意义；对齐参考 6 轮×1024 重测）
    ap.add_argument("--round_gen_tokens", type=int, default=None,
                    help="覆盖单轮生成预算（默认取 preset；截断率高时放宽）")
    ap.add_argument("--max_rounds", type=int, default=None,
                    help="覆盖工具轮数上限（默认取 preset；参考实现 6 轮）")
    ap.add_argument("--max_context_tokens", type=int, default=None,
                    help="覆盖总上下文上限（默认取 preset）")
    ap.add_argument("--dump_samples", type=int, default=0,
                    help="额外把前 N 条截断轨迹 + 前 3 条正常轨迹的原文落盘到 "
                         "<out>.samples.jsonl（截断率高时定位 token 去向用）")
    args = ap.parse_args()

    from rlab.config import get_config
    cfg = get_config("retool_math", model_path=args.model_path, use_wandb=False,
                     seed=args.seed)
    if args.data_task:
        cfg["data_task"] = args.data_task
    if args.temp is not None:
        cfg["temperature"] = args.temp
    if args.chat_template_kwargs:
        cfg["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    if args.round_gen_tokens is not None:
        cfg["round_gen_tokens"] = args.round_gen_tokens
    if args.max_rounds is not None:
        cfg["max_rounds"] = args.max_rounds
    if args.max_context_tokens is not None:
        cfg["max_context_tokens"] = args.max_context_tokens

    from rlab.data import load_qas, load_difficulty_table
    from rlab.reward import overlong_ref_tokens, total_reward_retool_math
    from rlab.rollout import build_prompt, multi_turn_rollout_group

    QAs = load_qas(cfg["data_task"])
    # 断点续跑：已探过的题跳过（表逐题追加写，崩溃/中断不丢进度）
    done = load_difficulty_table(args.out) if os.path.exists(args.out) else {}
    todo = [x for x in QAs if str(x["Q"]) not in done]
    if done:
        print(f"[probe] 续跑: 表中已有 {len(done)} 题，剩余 {len(todo)} 题")
    rng = random.Random(args.seed)
    rng.shuffle(todo)   # --max_questions 截断时取到的是随机前缀，不是数据集顺序
    if args.max_questions > 0:
        todo = todo[:args.max_questions]
    print(f"[probe] 模型 {args.model_path} | k={args.k} | temp={cfg['temperature']} "
          f"| 预算 {cfg['max_rounds']}轮×{cfg['round_gen_tokens']}tok"
          f"(ctx {cfg['max_context_tokens']}) | 本轮探 {len(todo)} 题（全池 {len(QAs)}）")
    if not todo:
        print("[probe] 无剩余题，直接输出统计")
        per_q = list(load_difficulty_table(args.out).values())
        print(summarize([], per_q))
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    vllm_gen = LLM(model=cfg["model_path"], gpu_memory_utilization=args.gpu_mem)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, "a", encoding="utf-8")
    fout_samp = None
    if args.dump_samples > 0:
        samp_path = args.out + ".samples.jsonl"
        fout_samp = open(samp_path, "w", encoding="utf-8")
        print(f"[probe] 样本落盘 -> {samp_path}")
    n_dump_trunc, n_dump_ok = 0, 0   # 截断轨迹是定位对象；正常轨迹留几条做对照
    k, wq = args.k, args.questions_per_wave
    all_rows, t0 = [], time.time()
    n_traj = 0   # 全局轨迹计数（seed 盐：防不同波次复采同轨迹）
    for w0 in range(0, len(todo), wq):
        wave = todo[w0:w0 + wq]
        group_prompts, rows = [], []
        for x in wave:
            p = build_prompt(x["Q"], cfg["system_prompt"], tokenizer,
                             cfg.get("chat_template_kwargs"))
            group_prompts.extend([p] * k)   # 每题扩成 k 条独立轨迹（与训练扩样同构）
        sps = [SamplingParams(n=1, temperature=cfg["temperature"],
                              max_tokens=cfg["round_gen_tokens"], top_p=cfg["top_p"],
                              top_k=cfg["top_k"], seed=args.seed * 1000003 + n_traj + j)
               for j in range(len(group_prompts))]
        n_traj += len(group_prompts)
        segs, texts, code_stats = multi_turn_rollout_group(
            vllm_gen, sps, tokenizer, group_prompts, cfg)
        for qi, x in enumerate(wave):
            for j in range(k):
                idx = qi * k + j
                asst_text = "".join(s["text"] for s in segs[idx] if s["kind"] == "assistant")
                clen = sum(len(s["ids"]) for s in segs[idx])   # 全长口径（与训练一致）
                sc = total_reward_retool_math(
                    x["A"], asst_text, code_ok=code_stats[idx]["code_ok"],
                    completion_len=clen, max_gen_tokens=overlong_ref_tokens(cfg),
                    overlong_buffer=cfg["overlong_buffer"],
                    overlong_shaping=cfg.get("overlong_shaping", False))
                trunc = code_stats[idx]["trunc_final"]
                rows.append({"Q": x["Q"], "A": x["A"], "acc": sc["acc"],
                             "fmt": sc["format"], "trunc": trunc,
                             "clen": clen, "code_ok": code_stats[idx]["code_ok"]})
                if fout_samp is not None:
                    dump = (trunc == 1 and n_dump_trunc < args.dump_samples) or \
                           (trunc == 0 and n_dump_ok < 3)
                    if dump:
                        n_dump_trunc += (trunc == 1)
                        n_dump_ok += (trunc == 0)
                        fout_samp.write(json.dumps(
                            {"Q": x["Q"], "A": x["A"], "acc": sc["acc"], "trunc": trunc,
                             "clen": clen, "code_ok": code_stats[idx]["code_ok"],
                             "n_segs": len(segs[idx]),
                             "segs_kind": [s["kind"] for s in segs[idx]],
                             "text": texts[idx]}, ensure_ascii=False) + "\n")
                        fout_samp.flush()
        if fout_samp is not None and n_dump_trunc >= args.dump_samples and n_dump_ok >= 3:
            fout_samp.close()
            fout_samp = None   # 额度收满即停写，防止全量落盘
        per_q = aggregate_rows(rows)
        for r in per_q:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
        all_rows.extend(rows)
        nc_sum = sum(r["n_correct"] for r in per_q)
        print(f"[probe] 波次 {w0 // wq + 1}/{(len(todo) + wq - 1) // wq}: "
              f"{len(wave)} 题正确 {nc_sum}/{len(per_q) * k} 条 | "
              f"累计 {w0 + len(wave)} 题 | {time.time() - t0:.0f}s", flush=True)

    fout.close()
    print(summarize(all_rows, aggregate_rows(all_rows)))
    print(f"[probe] 表已写出: {args.out}\n"
          f"[probe] 训练启用: bash rlab/run_gsm8k.sh retool_math <model> "
          f"--difficulty_path {args.out}")


if __name__ == "__main__":
    main()
