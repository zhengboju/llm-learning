# -*- coding: utf-8 -*-
"""广覆盖 SFT 轨迹采集：先每题一次，失败题再追加探索。

职责刻意只到标准 ``traj.jsonl``：数据筛选、低预算合成仍交给
``rlab.sft_data``，避免复制两套 SFT 样本口径。
"""
import argparse
import hashlib
import json
import os
import time


def qa_key(q, a):
    return hashlib.sha1((str(q) + "\0" + str(a)).encode("utf-8")).hexdigest()


def collect_seed(base_seed, qakey, attempt_idx, shard_index=0):
    """题×尝试×分片的稳定独立 seed；续跑不依赖处理顺序。"""
    raw = f"{int(base_seed)}:{int(shard_index)}:{qakey}:{int(attempt_idx)}"
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def load_collect_state(paths, expected_sp_sha=None, expected_model=None,
                       expected_protocol_sha=None, expected_plan_sha=None):
    """从一个或多个 collector dump 恢复进度；重复合并的行不会重复计数。"""
    state, seen = {}, set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if row.get("collect_phase") not in ("first", "retry"):
                    raise RuntimeError(
                        f"[sft_collect] 状态文件 {path} 含非 collector 行；"
                        "--state/--out 只能复用 sft_collect 产物")
                required = {"sp_sha": expected_sp_sha,
                            "collector_model": expected_model,
                            "protocol_sha": expected_protocol_sha,
                            "plan_sha": expected_plan_sha}
                missing = [k for k, v in required.items() if v and not row.get(k)]
                if missing:
                    raise RuntimeError(
                        f"[sft_collect] 状态文件 {path} 的 collector 行缺 {missing}；"
                        "无法证明续跑同档，拒绝复用")
                if expected_sp_sha and row.get("sp_sha") \
                        and row["sp_sha"] != expected_sp_sha:
                    raise RuntimeError(
                        f"[sft_collect] 状态文件 {path} 混入提示 sp{row['sp_sha']}，"
                        f"当前是 sp{expected_sp_sha}")
                if expected_model and row.get("collector_model") \
                        and row["collector_model"] != expected_model:
                    raise RuntimeError(
                        f"[sft_collect] 状态文件 {path} 混入模型 "
                        f"{row['collector_model']}，当前是 {expected_model}")
                if expected_protocol_sha and row.get("protocol_sha") \
                        and row["protocol_sha"] != expected_protocol_sha:
                    raise RuntimeError(
                        f"[sft_collect] 状态文件 {path} 混入协议 "
                        f"{row['protocol_sha']}，当前是 {expected_protocol_sha}")
                if expected_plan_sha and row.get("plan_sha") \
                        and row["plan_sha"] != expected_plan_sha:
                    raise RuntimeError(
                        f"[sft_collect] 状态文件 {path} 混入采集计划 "
                        f"{row['plan_sha']}，当前是 {expected_plan_sha}")
                key = row.get("qa_key") or qa_key(row.get("Q", ""), row.get("A", ""))
                uid = row.get("collect_uid")
                if uid and uid in seen:
                    continue
                if uid:
                    seen.add(uid)
                s = state.setdefault(key, {"attempts": 0, "success": False,
                                           "seen": False, "terminal": False})
                s["seen"] = True
                status = row.get("status")
                if status == "prompt_overlong":
                    s["terminal"] = True
                    continue
                idx = row.get("attempt_idx")
                if isinstance(idx, int) and idx >= 0:
                    s["attempts"] = max(s["attempts"], idx + 1)
                else:
                    s["attempts"] += 1
                if (status == "ok" and (row.get("acc") or 0) > 0
                        and (row.get("fmt") or 0) > 0
                        and not int(row.get("trunc_final") or 0)):
                    s["success"] = True
    return state


def build_first_plan(order, state):
    """首采严格只补从未见过的题。"""
    return [(x, 1) for x in order
            if not state.get(qa_key(x["Q"], x["A"]), {}).get("seen")]


def build_retry_plan(order, state, retry_min=2, retry_max=4):
    """首轮失败题先补到 retry_min，仍失败者再逐次补到 retry_max。"""
    if retry_min < 0 or retry_max < retry_min:
        raise ValueError("retry_max 必须 >= retry_min >= 0")
    out = []
    for x in order:
        s = state.get(qa_key(x["Q"], x["A"]), {})
        if not s.get("seen") or s.get("success") or s.get("terminal"):
            continue
        retries = max(0, int(s.get("attempts", 0)) - 1)
        if retries < retry_min:
            out.append((x, retry_min - retries))
        elif retries < retry_max:
            out.append((x, 1))
    return out


def clean_training_pool(qas, dev_rows):
    """held-out fail-closed，并跳过同题多答案，避免 qk 合并冲突监督。"""
    dev_q = {str(x["Q"]).strip() for x in dev_rows}
    overlap = [x for x in qas if str(x["Q"]).strip() in dev_q]
    if overlap:
        raise RuntimeError(
            f"[sft_collect] 训练池与 held-out(dev) 重叠 {len(overlap)} 行；"
            "拒绝采集，先重跑 prepare_dapo_math")
    answers = {}
    for x in qas:
        answers.setdefault(str(x["Q"]), set()).add(str(x["A"]))
    conflict_q = {q for q, aa in answers.items() if len(aa) > 1}
    clean = [x for x in qas if str(x["Q"]) not in conflict_q]
    return clean, {"n_input": len(qas), "n_dev": len(dev_rows),
                   "n_conflict_q": len(conflict_q), "n_clean": len(clean)}


def load_protocol_cfg(proto_from, model_path, seed):
    from rlab.config import (get_config, inject_budget_prompt,
                             validate_max_prompt_length, validate_retool_budget)
    cfg = get_config("retool_math", use_wandb=False)
    path = proto_from
    if os.path.isdir(path):
        path = os.path.join(path, "run_info.json")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    src = info.get("config") or {}
    if not src:
        raise RuntimeError(f"[sft_collect] {path} 没有 config")
    cfg.update(src)
    # collector 直接由目标 checkpoint 生成，不存在训练侧“base 壳启动后再同步权重”。
    cfg["model_path"] = model_path
    cfg["vllm_model_path"] = model_path
    cfg["seed"] = int(seed)
    cfg["use_wandb"] = False
    inject_budget_prompt(cfg)
    validate_max_prompt_length(cfg)
    cfg["_tool_reserve"] = validate_retool_budget(cfg)
    sp_sha = hashlib.sha1(cfg["system_prompt"].encode("utf-8")).hexdigest()[:6]
    return cfg, sp_sha, path


def make_dump_row(x, segs, stats, cfg, *, attempt_idx, seed, phase, sp_sha,
                  protocol_sha, plan_sha, model_path):
    from rlab.reward import overlong_ref_tokens, total_reward_retool_math
    from rlab.rollout import traj_dump_row
    text = "".join(s["text"] for s in segs if s["kind"] == "assistant")
    clen = sum(len(s.get("ids") or ()) for s in segs)
    sc = total_reward_retool_math(
        x["A"], text, code_ok=stats.get("code_ok", 0),
        completion_len=clen, max_gen_tokens=overlong_ref_tokens(cfg),
        overlong_buffer=cfg["overlong_buffer"],
        overlong_shaping=cfg.get("overlong_shaping", False))
    status = "proto_mismatch" if stats.get("proto_mismatch") else "ok"
    qakey = qa_key(x["Q"], x["A"])
    row = traj_dump_row(
        segs, Q=x["Q"], A=x["A"],
        qk=hashlib.sha1(str(x["Q"]).encode("utf-8")).hexdigest()[:12],
        status=status, stats=stats, acc=sc["acc"], fmt=sc["format"], clen=clen)
    row.update({"qa_key": qakey, "attempt_idx": int(attempt_idx),
                "collect_phase": phase, "collect_seed": int(seed),
                "collect_uid": f"{qakey}:{int(attempt_idx)}:{int(seed)}",
                "sp_sha": sp_sha, "protocol_sha": protocol_sha,
                "plan_sha": plan_sha, "collector_model": model_path})
    return row


def _write_meta(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="广覆盖 SFT 轨迹采集")
    ap.add_argument("--model_path", required=True,
                    help="直接供 vLLM 加载的 checkpoint（纯文本 Qwen3.5 请先物化）")
    ap.add_argument("--proto_from", required=True,
                    help="含 run_info.json 的目录/文件；协议、提示和预算全部从这里回读")
    ap.add_argument("--out", required=True, help="标准 traj.jsonl，支持断点续跑")
    ap.add_argument("--phase", choices=("first", "retry", "all"), default="all")
    ap.add_argument("--state", action="append", default=[],
                    help="retry 时的完整首采 dump；分片先 cat 后传，可多次")
    ap.add_argument("--max_questions", type=int, default=4000)
    ap.add_argument("--retry_min", type=int, default=2)
    ap.add_argument("--retry_max", type=int, default=4)
    ap.add_argument("--questions_per_wave", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_mem", type=float, default=0.78)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    args = ap.parse_args(argv)

    if args.phase == "all" and args.shard_count != 1:
        raise SystemExit("[sft_collect] 多分片禁止 --phase all：先所有卡跑 first，"
                         "合并状态后再跑 retry，才能保证全局覆盖优先")
    if args.phase == "retry" and not args.state:
        raise SystemExit("[sft_collect] retry 必须用 --state 传完整首采 dump")
    if args.questions_per_wave < 1:
        raise SystemExit("--questions_per_wave 必须 >=1")

    cfg, sp_sha, proto_path = load_protocol_cfg(
        args.proto_from, args.model_path, args.seed)
    protocol_fields = {k: cfg.get(k) for k in (
        "tool_protocol", "native_tool_style", "native_stop_at_call",
        "temperature", "top_p", "top_k", "round_gen_tokens", "max_rounds",
        "max_traj_tokens", "max_context_tokens", "max_prompt_length",
        "answer_reserve", "tool_result_max_chars", "budget_hint",
        "vllm_gen_kwargs", "vllm_batch_invariant", "vllm_attention_backend")}
    protocol_fields["collector_gpu_mem"] = float(args.gpu_mem)
    protocol_fields["sp_sha"] = sp_sha
    protocol_sha = hashlib.sha1(json.dumps(
        protocol_fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    if args.phase in ("retry", "all") and float(cfg.get("temperature", 0) or 0) <= 0:
        raise SystemExit("[sft_collect] retry 要求 temperature>0；贪心重试只会复制同一路径")

    from rlab.data import load_dapo_math_dev, load_qas
    from rlab.probe_difficulty import probe_plan
    dev = load_dapo_math_dev()  # 缺失直接终止，绝不以训练池冒充 held-out
    qas, pool_stats = clean_training_pool(load_qas("dapo_math"), dev)
    full_order, _ = probe_plan(qas, {}, seed=args.seed,
                               max_questions=args.max_questions)
    order, _ = probe_plan(qas, {}, seed=args.seed,
                          max_questions=args.max_questions,
                          shard_index=args.shard_index,
                          shard_count=args.shard_count)
    pool_hash = hashlib.sha1("\n".join(
        qa_key(x["Q"], x["A"]) for x in full_order).encode("utf-8")).hexdigest()[:12]
    dev_hash = hashlib.sha1("\n".join(sorted(
        hashlib.sha1(str(x["Q"]).encode("utf-8")).hexdigest()
        for x in dev)).encode("utf-8")).hexdigest()[:12]
    plan_fields = {"seed": int(args.seed), "max_questions": int(args.max_questions),
                   "shard_count": int(args.shard_count), "pool_hash": pool_hash,
                   "dev_hash": dev_hash}
    plan_sha = hashlib.sha1(json.dumps(
        plan_fields, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    state_paths = list(args.state)
    if os.path.exists(args.out):
        state_paths.append(args.out)
    state = load_collect_state(state_paths, expected_sp_sha=sp_sha,
                               expected_model=args.model_path,
                               expected_protocol_sha=protocol_sha,
                               expected_plan_sha=plan_sha)
    if args.phase == "retry":
        missing = [x for x in full_order
                   if not state.get(qa_key(x["Q"], x["A"]), {}).get("seen")]
        if missing:
            raise SystemExit(
                f"[sft_collect] retry 状态缺 {len(missing)}/{len(full_order)} 道首采题；"
                "先完成并合并所有 first 分片，拒绝提前把预算花在重试上")

    print(f"[sft_collect] phase={args.phase}｜全局 {len(full_order)} 题｜"
          f"分片 {args.shard_index}/{args.shard_count}={len(order)} 题｜"
          f"sp{sp_sha} proto={protocol_sha} plan={plan_sha}｜"
          f"pool={pool_hash} dev={dev_hash}")
    print(f"[sft_collect] pool {pool_stats}｜model={args.model_path}｜proto={proto_path}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from rlab.protocol import (NATIVE_BAD_WORDS, NATIVE_CALL_STOP,
                               RETOOL_STOP_KWARGS, initial_messages,
                               native_special_ban_words)
    from rlab.rollout import (_check_vllm_gen_kwargs, attention_backend_kwargs,
                              batch_invariant_guard, build_prompt, build_prompt_ids,
                              gdn_backend_missing, multi_turn_rollout_group,
                              traj_dump_row)

    native = (cfg.get("tool_protocol") or "fence") != "fence"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    gen_kwargs = dict(cfg.get("vllm_gen_kwargs") or {})
    bi = bool(cfg.get("vllm_batch_invariant"))
    attn = cfg.get("vllm_attention_backend")
    batch_invariant_guard(bi, attn)
    if bi:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    if attn:
        gen_kwargs.update(attention_backend_kwargs(attn))
    if gdn_backend_missing(args.model_path, gen_kwargs):
        print("[sft_collect][警告] Qwen3.5 未显式使用 gdn_prefill_backend=triton，"
              "可能触发 FlashInfer JIT 和宿主 RAM 峰值", flush=True)
    _check_vllm_gen_kwargs(gen_kwargs)
    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_mem,
              **gen_kwargs)

    def prompt_info(x):
        msgs = initial_messages(cfg["system_prompt"], x["Q"]) if native else None
        text = build_prompt(x["Q"], cfg["system_prompt"], tokenizer,
                            cfg.get("chat_template_kwargs"), tools=native)
        plen = (len(build_prompt_ids(msgs, tokenizer,
                                     cfg.get("chat_template_kwargs"), tools=True))
                if native else len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        return text, msgs, plen

    def sampling(seed):
        kw = dict(n=1, temperature=cfg["temperature"],
                  max_tokens=cfg["round_gen_tokens"], top_p=cfg["top_p"],
                  top_k=cfg.get("top_k", 50), seed=seed)
        if native:
            if cfg.get("native_stop_at_call"):
                kw.update({"stop": [NATIVE_CALL_STOP],
                           "include_stop_str_in_output": True})
            kw["bad_words"] = (list(NATIVE_BAD_WORDS)
                               + native_special_ban_words(tokenizer))
        elif cfg.get("retool_stop"):
            kw.update(RETOOL_STOP_KWARGS)
        return SamplingParams(**kw)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, "a", encoding="utf-8")
    counters = {"written": 0, "correct": 0, "prompt_overlong": 0,
                "proto_mismatch": 0, "first": 0, "retry": 0}
    t0 = time.time()

    def run_plan(plan, phase):
        for w0 in range(0, len(plan), args.questions_per_wave):
            wave = plan[w0:w0 + args.questions_per_wave]
            requests = []
            for x, n_new in wave:
                key = qa_key(x["Q"], x["A"])
                start = int(state.get(key, {}).get("attempts", 0))
                text, msgs, plen = prompt_info(x)
                if plen > int(cfg["max_prompt_length"]):
                    if not state.get(key, {}).get("seen"):
                        row = traj_dump_row([], Q=x["Q"], A=x["A"],
                            qk=hashlib.sha1(str(x["Q"]).encode("utf-8")).hexdigest()[:12],
                            status="prompt_overlong")
                        row.update({"qa_key": key, "attempt_idx": -1,
                                    "collect_phase": phase, "sp_sha": sp_sha,
                                    "protocol_sha": protocol_sha,
                                    "plan_sha": plan_sha,
                                    "collector_model": args.model_path,
                                    "collect_uid": f"{key}:prompt_overlong"})
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        state[key] = {"attempts": 0, "success": False,
                                      "seen": True, "terminal": True}
                        counters["written"] += 1
                        counters["prompt_overlong"] += 1
                    continue
                for j in range(n_new):
                    attempt = start + j
                    sd = collect_seed(args.seed, key, attempt, args.shard_index)
                    requests.append((x, attempt, sd, text, msgs))
            if not requests:
                fout.flush()
                continue
            prompts = [r[3] for r in requests]
            messages = [r[4] for r in requests] if native else None
            sps = [sampling(r[2]) for r in requests]
            segs, _, stats = multi_turn_rollout_group(
                llm, sps, tokenizer, prompts, cfg,
                prompts_messages=messages)
            for r, sg, st in zip(requests, segs, stats):
                x, attempt, sd = r[:3]
                row = make_dump_row(x, sg, st, cfg, attempt_idx=attempt,
                                    seed=sd, phase=phase, sp_sha=sp_sha,
                                    protocol_sha=protocol_sha, plan_sha=plan_sha,
                                    model_path=args.model_path)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                key = row["qa_key"]
                ss = state.setdefault(key, {"attempts": 0, "success": False,
                                            "seen": True, "terminal": False})
                ss["seen"] = True
                ss["attempts"] = max(ss["attempts"], attempt + 1)
                if (row["status"] == "ok" and row["acc"] > 0
                        and row["fmt"] > 0 and not row["trunc_final"]):
                    ss["success"] = True
                    counters["correct"] += 1
                counters["written"] += 1
                counters[phase] += 1
                counters["proto_mismatch"] += int(row["status"] == "proto_mismatch")
            fout.flush()
            print(f"[sft_collect] {phase} 波 {w0 // args.questions_per_wave + 1}/"
                  f"{(len(plan) + args.questions_per_wave - 1) // args.questions_per_wave}: "
                  f"写 {counters['written']}，本进程正确 {counters['correct']}，"
                  f"{time.time() - t0:.0f}s", flush=True)

    if args.phase in ("first", "all"):
        run_plan(build_first_plan(order, state), "first")
    if args.phase in ("retry", "all"):
        while True:
            plan = build_retry_plan(order, state, args.retry_min, args.retry_max)
            if not plan:
                break
            run_plan(plan, "retry")
    fout.close()

    final_state = load_collect_state(
        list(args.state) + [args.out], expected_sp_sha=sp_sha,
        expected_model=args.model_path, expected_protocol_sha=protocol_sha,
        expected_plan_sha=plan_sha)
    covered = sum(bool(final_state.get(qa_key(x["Q"], x["A"]), {}).get("seen"))
                  for x in order)
    success = sum(bool(final_state.get(qa_key(x["Q"], x["A"]), {}).get("success"))
                  for x in order)
    meta = {"model_path": args.model_path, "proto_from": proto_path,
            "phase": args.phase, "seed": args.seed, "sp_sha": sp_sha,
            "protocol_sha": protocol_sha, "plan_sha": plan_sha,
            "plan": plan_fields,
            "pool_hash": pool_hash, "dev_hash": dev_hash, "pool": pool_stats,
            "max_questions": args.max_questions,
            "shard_index": args.shard_index, "shard_count": args.shard_count,
            "retry_min": args.retry_min, "retry_max": args.retry_max,
            "questions_in_shard": len(order), "covered": covered,
            "questions_with_success": success, "this_run": counters,
            "protocol": protocol_fields}
    _write_meta(args.out + ".meta.json", meta)
    print(f"[sft_collect] 完成：本片覆盖 {covered}/{len(order)}，有正确轨迹 "
          f"{success}/{len(order)}；写 {args.out}，审计 {args.out}.meta.json")


if __name__ == "__main__":
    main()
