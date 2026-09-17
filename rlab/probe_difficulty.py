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

【2026-09-17 档位铁律】探针必须与训练的**采样档**一致，否则表描述的是另一个分布：
  · 预算/提示/温度：默认自动取 retool_math preset；**训练时用 CLI 覆盖过
    --round_gen_tokens/--max_rounds/--max_context_tokens，探针必须传同一组值**
    （CLI 覆盖后 validate_retool_budget 会重跑校验，不自洽直接 raise）。
  · thinking 开关：preset 已内置 enable_thinking=False，探针自动继承。
  · 引擎档：默认取 preset 的 {"gdn_prefill_backend":"triton"}（不传 = 掉回 FlashInfer
    GDN JIT，本 pod 会无 traceback 被 SIGKILL）。训练若开了确定性档，探针要同开：
        --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN
    环境里若**遗留**了 VLLM_BATCH_INVARIANT=1（训练 run 的 export），探针会自动按
    确定性档对齐并在缺 backend 时于引擎构造前 fail-fast；不想开就 `unset` 它。
  · 模型：必须与训练起点同权重（换基座/换 ckpt 要重探；旧表不自动失效）。
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


def shard_items(items: list, index: int, count: int) -> list:
    """把（已打乱、已按 --max_questions 截断的）题目列表切成 count 片，返回第 index 片。

    【2026-09-17 多卡并行探针】全量表在定版预算（6144/14336）下实测 ~11 s/题 →
    16.7k 题要 ~50h 单卡。两卡各跑一片可减半，而分片正确性的**不变量**是
    "各片互斥 且 并集 = 原集合"——丢题（表有洞 → 训练池被静默缩小）与重题
    （白烧算力）都是不报错的静默错误，所以这里做成纯函数并锁进测试。

    为什么按 `items[index::count]` 切：两个进程用同一 seed 打乱同一全池 → 各自的
    todo 逐元素相同 → 余数切片天然互斥、并集等于原集合，且顺序确定（与"连续切"
    等价，但对 --max_questions/续跑造成的 todo 差异更鲁棒）。合并方式：`cat` 即可
    （load_difficulty_table 按题面建 dict，行序无关）。"""
    if count < 1:
        raise ValueError(f"shard_count 必须 ≥1，收到 {count}")
    if not (0 <= index < count):
        raise ValueError(f"shard_index 必须落在 [0,{count})，收到 {index}")
    return list(items)[index::count]


def probe_plan(QAs: list, done: dict, *, seed: int = 42, max_questions: int = 0,
               shard_index: int = 0, shard_count: int = 1):
    """→ `(order, todo)`：本片的固定参考顺序（**与 done 无关**）+ 本片真正待探的题。

    **顺序铁律：先打乱 → max_questions → 切片 → 最后才按 done 过滤。**
    反过来（先按 done 过滤、再打乱切片）在**续跑**时会让 todo 变短 → 打乱后的切片
    整体错位 → 本片跑去探另一片的题：分片的"互斥/并集 = 原集合"不变量在续跑路径上
    失效（首跑完全正常，只有中断恢复才踩——最难发现的那类）。
    【2026-09-17 真机】50h 全量表第一次分片跑到 ~60% 才发现这个顺序问题，
    所以它必须是纯函数 + 有反证测试。"""
    rng = random.Random(seed)
    order = list(QAs)
    rng.shuffle(order)          # --max_questions 截断时取到的是随机前缀，不是数据集顺序
    if max_questions > 0:
        order = order[:max_questions]
    if shard_count > 1:
        order = shard_items(order, shard_index, shard_count)
    todo = [x for x in order if str(x["Q"]) not in done]
    return order, todo


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
    ap.add_argument("--shard_index", type=int, default=0,
                    help="多卡并行的分片序号（0-based；默认 0 = 不分片）")
    ap.add_argument("--shard_count", type=int, default=1,
                    help="总分片数（两卡各跑一片：0/2 与 1/2，最后 `cat` 合并；"
                         "各片 --out 必须不同，合并后才是完整表）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system_prompt_file", default=None,
                    help="用文件内容整体替换系统提示（**必须与训练同源**：难度表是"
                         "『模型×提示×预算』三者的联合产物，提示不同就是另一张表）")
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
    # 【2026-09-17 真机】档位三件套必须与训练一致（"探针与训练同口径"铁律）：
    # 旧版探针既不传 cfg 的 vllm_gen_kwargs（连 preset 的 gdn_prefill_backend=triton
    # 都没生效 → 会落回 FlashInfer GDN JIT → 本 pod 两次实锤的无 traceback SIGKILL），
    # 也没有 attention backend 入口 —— 环境里遗留的 VLLM_BATCH_INVARIANT=1 会让引擎
    # 启动即 RuntimeError（真机实锤）。
    ap.add_argument("--vllm_gen_kwargs", default=None,
                    help='JSON dict **整体替换** 引擎参数（与 train.py 同语义；默认取 '
                         'preset 的 {"gdn_prefill_backend": "triton"}）。注意整体替换：'
                         "只想加键时要把 triton 一并写回")
    ap.add_argument("--vllm_batch_invariant", action="store_true",
                    help="与训练同档：开 VLLM_BATCH_INVARIANT=1（**必须同时给 "
                         "--vllm_attention_backend**，否则引擎启动即失败）。"
                         "环境里若已继承该变量，探针会自动按确定性档对齐")
    ap.add_argument("--vllm_attention_backend", default=None,
                    help="显式 attention backend（如 FLASH_ATTN）；确定性档必需")
    args = ap.parse_args()

    from rlab.config import get_config, validate_retool_budget
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
    if args.system_prompt_file:
        # 提示是协议的一半：本探针出的表只对"同提示 + 同预算 + 同模型"的训练有效。
        # 打印指纹，便于与训练启动行的 `signature=...-sp<hash6>` 逐字对上。
        import hashlib
        with open(args.system_prompt_file, encoding="utf-8") as f:
            cfg["system_prompt"] = f.read().strip()
        _sp_sha = hashlib.sha1(cfg["system_prompt"].encode("utf-8")).hexdigest()[:6]
        print(f"[probe] 系统提示替换为 {args.system_prompt_file}"
              f"（{len(cfg['system_prompt'])} 字符，sp{_sp_sha}）"
              f"—— 训练必须传同一个 --system_prompt_file，签名里应出现 -sp{_sp_sha}")
    # 【2026-09-12】CLI 覆盖后必须重跑预算校验：探针的价值就在于"描述训练时的
    # 采样分布"，若探针预算几何与训练不一致（或不自洽），整张难度表都是另一个
    # 分布下的产物。get_config 里的校验发生在 override 之前，拦不住这里。
    cfg["_tool_reserve"] = validate_retool_budget(cfg)

    from rlab.data import load_qas, load_difficulty_table
    from rlab.reward import overlong_ref_tokens, total_reward_retool_math
    from rlab.rollout import build_prompt, multi_turn_rollout_group

    QAs = load_qas(cfg["data_task"])
    # 断点续跑：已探过的题跳过（表逐题追加写，崩溃/中断不丢进度）。
    # 【2026-09-17】分片 × 续跑的顺序交给 probe_plan 纯函数（切片与 done 无关），
    # 否则续跑时切片错位 → 两片互相探对方的题（重叠浪费 + 不变量失效）。
    done = load_difficulty_table(args.out) if os.path.exists(args.out) else {}
    order, todo = probe_plan(QAs, done, seed=args.seed,
                             max_questions=args.max_questions,
                             shard_index=args.shard_index,
                             shard_count=args.shard_count)
    if args.shard_count > 1:
        print(f"[probe] 分片 {args.shard_index}/{args.shard_count}: 本片 {len(order)} 题"
              f"（各片 --out 不同，最后 cat 合并成完整表）")
    if done:
        print(f"[probe] 续跑: 本片表中已有 {len(done)} 题，本片剩余 {len(todo)} 题")
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
    from rlab.rollout import (attention_backend_kwargs, batch_invariant_guard,
                              gdn_backend_missing)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    # ---- 档位对齐（与 train/rollout 同一套 helper，不另写一份判定）----
    _gk = dict(cfg.get("vllm_gen_kwargs") or {})
    if args.vllm_gen_kwargs:
        _gk = json.loads(args.vllm_gen_kwargs)
        if not isinstance(_gk, dict):
            raise SystemExit("[probe] --vllm_gen_kwargs 必须是 JSON dict（整体替换语义）")
    _env_bi = str(os.environ.get("VLLM_BATCH_INVARIANT", "")).strip().lower() not in (
        "", "0", "false")
    _bi = bool(args.vllm_batch_invariant) or _env_bi
    if _env_bi and not args.vllm_batch_invariant:
        print(f"[probe] 检测到环境里继承的 VLLM_BATCH_INVARIANT="
              f"{os.environ.get('VLLM_BATCH_INVARIANT')!r}（多半是训练 run 留下的 export）"
              f"——探针按**确定性档**对齐（与训练同档是铁律）。不想开就先 "
              f"`unset VLLM_BATCH_INVARIANT` 再跑。", flush=True)
    batch_invariant_guard(_bi, args.vllm_attention_backend)   # 缺 backend 在引擎构造前拦下
    if _bi:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    if args.vllm_attention_backend:
        _gk.update(attention_backend_kwargs(args.vllm_attention_backend))
    _vllm_path = cfg.get("vllm_model_path") or cfg["model_path"]
    if gdn_backend_missing(_vllm_path, _gk):
        print("[probe][警告] 引擎参数里没有 gdn_prefill_backend → Qwen3.5 的 GDN prefill "
              "会落到 FlashInfer JIT 现场编译（本 pod 两次实锤：ninja 打爆宿主 RAM → "
              "进程被 SIGKILL、**无 traceback**）。preset 默认已含 triton；"
              "`--vllm_gen_kwargs` 是整体替换，别把 triton 写丢。", flush=True)
    print(f"[probe] vLLM 引擎参数: {_gk}"
          f"{'｜确定性档 VLLM_BATCH_INVARIANT=1' if _bi else ''}"
          f"｜model={_vllm_path}", flush=True)
    vllm_gen = LLM(model=_vllm_path, gpu_memory_utilization=args.gpu_mem, **_gk)

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
