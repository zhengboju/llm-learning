# -*- coding: utf-8 -*-
"""rlab/probe_retool_gen.py — 诊断 base 在 retool 提示下的真实生成（零标签字面量）。

背景（2026-09-08 第三/四轮）：健康检查 32 组 fmt 恒 -1 = 128 样本 ~0 格式合规；
round-4 eval 仍 fmt≈0、code_rate≈2%。怀疑 base+3B 在"写代码"压力下要么代码写得太
稀、要么一写代码格式就崩——两者都让 code/fmt 奖励近乎常数、无梯度。本探针把 base
在三种提示下的多轮原始生成 + 新旧打分逐样本判定一次打全，直接回答三件事：
  ① 提示本身在不在起作用（版本自检 git head + strip 存在性）；
  ② 代码是不是完整的可执行围栏（code_used 才有），还是开了围栏写不完；
  ③ 剥离代码后格式到底差在哪（没标签 / 标签顺序乱 / 被截断）。

用法（pod，约 2-3 分钟）：
    CUDA_VISIBLE_DEVICES=0 python rlab/probe_retool_gen.py --n 8 --show_prompt new

输出：
    --show_prompt new  打印"新提示"下每个样本的逐段原始文本（repr 截断）
    三种提示各输出一组汇总：fmt_raw / fmt_strip / code_rate / code_ok_rate
"""
import argparse
import json
import random
import subprocess

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="每种提示抽样题数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="/root/Qwen2.5-3B")
    ap.add_argument("--gpu_mem", type=float, default=0.22)
    ap.add_argument("--round_tokens", type=int, default=400, help="每轮 assistant 段上限")
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--show_prompt", default="new",
                    choices=["new", "old", "plain", "none"],
                    help="打印哪个提示下样本的逐段原始文本")
    args = ap.parse_args()

    # ---- 版本自检 ----
    print("== git head ==")
    try:
        print(subprocess.check_output(["git", "log", "-1", "--oneline"],
                                      text=True).strip())
    except Exception as e:  # pragma: no cover
        print(f"  (git 不可用: {e})")
    import rlab.reward as R
    print(f"== rlab.reward.strip_code_blocks 存在: {hasattr(R, 'strip_code_blocks')} ==")

    from rlab.config import BASE, system_prompt_retool
    # 旧 MUST（代码先行，70c92ff 前的措辞；零标签字面量）与 plain（无代码指令）对照
    _OLD_EXTRA = (
        "\n\nYou MUST write Python code to help solve the problem: when the question "
        "involves any calculation, first write the computation as code, then reason "
        "from the result. Put each piece of code inside a fenced block like: "
        "```python\n<your code>\n```\n"
        "The environment executes your code automatically and inserts the result "
        "between [TOOL RESULT] and [/TOOL RESULT]. Read the result and continue "
        "reasoning until you reach the final answer inside the required answer tags. "
        "Always finish your code block before continuing."
    )
    prompts_spec = {
        "plain": BASE["system_prompt"],
        "old": BASE["system_prompt"] + _OLD_EXTRA,
        "new": system_prompt_retool,
    }
    print(f"== 三路提示（{args.n} 题/路, seed={args.seed}, greedy, round_tokens={args.round_tokens}）==")

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
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2600, dtype="bfloat16")
    from rlab.rollout import multi_turn_rollout_group
    from rlab.reward import reward_format, strip_code_blocks
    sp = SamplingParams(temperature=0, max_tokens=args.round_tokens)
    cfg = {"max_rounds": args.max_rounds, "sandbox_timeout": 5.0,
           "sandbox_mem_mb": 256, "tool_result_max_chars": 500}

    overall = {}
    for pname, sp_text in prompts_spec.items():
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": sp_text},
             {"role": "user", "content": item["Q"]}], tokenize=False,
            add_generation_prompt=True) for item in sample]
        segs, full, code_stats = multi_turn_rollout_group(
            llm, sp, tokenizer, prompts, cfg)
        st = {"fmt_raw": 0, "fmt_strip": 0, "code": 0, "ok": 0, "rounds": 0}
        for i, (item, segs_i, cs) in enumerate(zip(sample, segs, code_stats)):
            raw = "".join(s["text"] for s in segs_i if s["kind"] == "assistant")
            fr, fs = reward_format(raw), reward_format(strip_code_blocks(raw))
            st["fmt_raw"] += fr > 0
            st["fmt_strip"] += fs > 0
            st["code"] += cs["code_used"] > 0
            st["ok"] += cs["code_ok"] > 0
            st["rounds"] += cs["code_used"]
            if pname == args.show_prompt:
                print(f"\n===== [{pname}] 样本 {i} | 题: {item['Q'][:70]}...")
                print(f"      code_used={cs['code_used']} code_ok={cs['code_ok']} "
                      f"| fmt_raw={fr:.0f}（不剥离） fmt_strip={fs:.0f}（剥离后）")
                for j, s in enumerate(segs_i):
                    print(f"--- 段{j} [{s['kind']}] ({len(s['text'])} 字符) ---")
                    print(repr(s["text"][:600]))
        overall[pname] = st
        n = args.n
        print(f"\n== [{pname}] 汇总 ==")
        print(f"  旧打分(不剥离) fmt: {st['fmt_raw']}/{n} = {st['fmt_raw'] / n * 100:.1f}%")
        print(f"  新打分(剥离后) fmt: {st['fmt_strip']}/{n} = {st['fmt_strip'] / n * 100:.1f}%")
        print(f"  代码调用率: {st['code']}/{n} = {st['code'] / n * 100:.1f}%"
              f"  成功: {st['ok']}/{n}  总轮次: {st['rounds']}")

    print("\n== 三路对照表（关键输出）==")
    for pname, st in overall.items():
        print(f"  {pname:<6} fmt_raw={st['fmt_raw']/args.n*100:5.1f}%  "
              f"fmt_strip={st['fmt_strip']/args.n*100:5.1f}%  "
              f"code={st['code']/args.n*100:5.1f}%  code_ok={st['ok']/args.n*100:5.1f}%  "
              f"rounds={st['rounds']}")


if __name__ == "__main__":
    main()
