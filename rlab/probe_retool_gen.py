# -*- coding: utf-8 -*-
"""rlab/probe_retool_gen.py — 诊断 base 在 retool 提示下的真实生成（零标签字面量）。

背景（2026-09-08 第三/四轮）：健康检查报"fmt 最近 32 组恒为 -1.0"= 128 个样本里
~0 个格式合规。怀疑 MUST"先写代码"在生成层掐死了格式（模型开围栏后不再产出
思考/回答标签），而不是打分层问题。本探针直接打印 base 的多轮原始生成 + 新旧两套
打分口径的逐样本判定，一次看清：模型写了什么、格式差在哪、剥离有没有用。

用法（pod，约 1-2 分钟）：
    CUDA_VISIBLE_DEVICES=0 python rlab/probe_retool_gen.py --n 6

输出每段：
    [assistant] 模型生成文本（repr 截断）；[tool] 沙箱返回
    fmt_raw   = 旧打分（不剥离代码）格式判定 ±1
    fmt_strip = 新打分（剥离代码块后）格式判定 ±1
"""
import argparse
import json
import random
import subprocess

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="抽样题数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="/root/Qwen2.5-3B")
    ap.add_argument("--gpu_mem", type=float, default=0.22)
    ap.add_argument("--round_tokens", type=int, default=400, help="每轮 assistant 段上限")
    ap.add_argument("--max_rounds", type=int, default=3)
    args = ap.parse_args()

    # ---- 版本自检：确认跑的是不是含剥离修复的代码 ----
    print("== git head ==")
    try:
        print(subprocess.check_output(["git", "log", "-1", "--oneline"],
                                      text=True).strip())
    except Exception as e:  # pragma: no cover
        print(f"  (git 不可用: {e})")
    import rlab.reward as R
    print(f"== rlab.reward.strip_code_blocks 存在: {hasattr(R, 'strip_code_blocks')} ==")

    from rlab.config import system_prompt_retool
    print("== system_prompt_retool（前 400 字符）==")
    print(repr(system_prompt_retool[:400]))

    # ---- 数据（与评测同源：modelscope gsm8k test, seed 抽样）----
    from modelscope.msdatasets import MsDataset
    from rlab.data import _patch_verification_mode
    _patch_verification_mode()
    ds = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test",
                        trust_remote_code=True)
    test_data = [{"Q": x["question"], "A": x["answer"]} for x in ds]
    random.seed(args.seed)
    sample = random.sample(test_data, args.n)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt_retool},
         {"role": "user", "content": item["Q"]}], tokenize=False,
        add_generation_prompt=True) for item in sample]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2600, dtype="bfloat16")
    from rlab.rollout import multi_turn_rollout_group
    from rlab.reward import reward_format, strip_code_blocks
    sp = SamplingParams(temperature=0, max_tokens=args.round_tokens)
    cfg = {"max_rounds": args.max_rounds, "sandbox_timeout": 5.0,
           "sandbox_mem_mb": 256, "tool_result_max_chars": 500}
    segs, full, code_stats = multi_turn_rollout_group(llm, sp, tokenizer, prompts, cfg)

    stats = {"fmt_raw": 0, "fmt_strip": 0, "code": 0, "ok": 0}
    for i, (item, segs_i, cs) in enumerate(zip(sample, segs, code_stats)):
        raw = "".join(s["text"] for s in segs_i if s["kind"] == "assistant")
        stripped = strip_code_blocks(raw)
        fr, fs = reward_format(raw), reward_format(stripped)
        stats["fmt_raw"] += fr > 0
        stats["fmt_strip"] += fs > 0
        stats["code"] += cs["code_used"] > 0
        stats["ok"] += cs["code_ok"] > 0
        print(f"\n===== 样本 {i} | 题: {item['Q'][:70]}...")
        print(f"      code_used={cs['code_used']} code_ok={cs['code_ok']} "
              f"| fmt_raw={fr:.0f}（不剥离） fmt_strip={fs:.0f}（剥离后）")
        for j, s in enumerate(segs_i):
            print(f"--- 段{j} [{s['kind']}] ({len(s['text'])} 字符) ---")
            print(repr(s["text"][:600]))

    print("\n== 汇总 ==")
    print(json.dumps(stats, indent=2))
    print(f"旧打分(不剥离) fmt 率: {stats['fmt_raw']}/{args.n} = "
          f"{stats['fmt_raw'] / args.n * 100:.1f}%")
    print(f"新打分(剥离后) fmt 率: {stats['fmt_strip']}/{args.n} = "
          f"{stats['fmt_strip'] / args.n * 100:.1f}%")
    print(f"代码调用率: {stats['code']}/{args.n} = {stats['code'] / args.n * 100:.1f}%"
          f"  成功率: {stats['ok']}/{args.n}")


if __name__ == "__main__":
    main()
