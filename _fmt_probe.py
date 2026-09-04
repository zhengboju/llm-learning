# -*- coding: utf-8 -*-
"""_fmt_probe.py — base 格式率探针（修复版 2）

背景：rlab 训练期 record.jsonl 显示格式率仅 0.5~10%（temp=0.9 采样），
而 greedy 评测 BASE fmt≈49.3%。本探针量化 base 模型在不同采样参数下
的格式率，回答两个问题：
  1) 是温度 0.9 还是 top_k 毁掉了格式？
  2) 降温度（0.7/0.6）能否把格式率抬回可学习水平？

关键设计：
  - 标签/正则/系统提示一律从 rlab.reward / rlab.config 导入，
    字节与训练、评测完全一致，杜绝"聊天界面改写标签"的伪影（教训：
    _base_sampling_probe.py 曾被界面改写而误报 0%）；
  - 修掉上一版 temp=0 时 n=4 的崩溃（vLLM greedy 必须 n=1）；
  - 输出"起头 Top5 字符"辅助判断模型没格式时在输出什么。

用法（训练机，先 git pull / scp 同步本文件，不要从聊天界面复制粘贴）：
  python _fmt_probe.py --engine vllm      # rlab 同款引擎 vLLM
  python _fmt_probe.py --engine hf        # 历史 grpo/dapo 同款 HF generate，作对照
"""
import argparse
import os
import re
import sys

from transformers import AutoTokenizer

# 从 rlab 导入已验证正确的字节（防界面改写）
try:
    from rlab.reward import _FORMAT_RE
    from rlab.config import BASE
except ImportError:
    sys.exit("[错误] 请在 simple_GRPO 仓库根目录运行（需要 rlab 包）")

PATTERN = re.compile(_FORMAT_RE, re.DOTALL)   # ^...$ 已在 _FORMAT_RE 内
SYSTEM_PROMPT = BASE["system_prompt"]

# 字节自检：正则与提示里必须真实含有四个标签（任一缺失 = 文件被界面改写）
_TAGS = (" thinking", " response", "<answer>", "</answer>")
for _b in _TAGS:
    if _b not in PATTERN.pattern or _b not in SYSTEM_PROMPT:
        print(f"[自检失败] 缺少真实标签字节 {_b!r}（文件被界面改写？用 git/scp 同步本文件）")
        sys.exit(1)
print("[自检通过] 正则含真实标签字节:", PATTERN.pattern)

QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "A train travels 120 km in 2 hours. At the same speed, how far does it travel in 5 hours?",
    "A store sells apples at 3 for $2. How much do 12 apples cost?",
    "A rectangle is 6 cm wide and 9 cm long. What is its area in square centimeters?",
    "A bike costs $240. It is on sale for 25% off. What is the sale price?",
    "Three friends split a $45 dinner bill evenly. How much does each pay?",
    "A tank fills at 30 liters per minute. How long to fill 750 liters?",
    "A book has 300 pages. Tom reads 20 pages a day. How many days to finish?",
    "A garden has 4 rows of 7 tulips and 3 rows of 5 roses. How many flowers total?",
    "A box holds 6 cans. How many boxes for 42 cans?",
    "A recipe needs 2 cups of flour for 12 cookies. How many cups for 30 cookies?",
]

MODEL = "/root/Qwen2.5-3B"


def build_metrics(answers):
    n = max(1, len(answers))
    fmt = sum(1 for a in answers if PATTERN.match(a))
    starts = {}
    for a in answers:
        k = a[:1] if a else "<EMPTY>"
        starts[k] = starts.get(k, 0) + 1
    top = sorted(starts.items(), key=lambda x: -x[1])[:5]
    return fmt / n, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("vllm", "hf"), default="vllm")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--questions", type=int, default=12)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ps = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
        for q in QUESTIONS[: args.questions]
    ]

    if args.engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, gpu_memory_utilization=0.3)
        for temp, tk in ((0.9, 50), (0.9, -1), (0.7, 50), (0.7, -1), (0.6, 50)):
            out = llm.generate(
                ps, SamplingParams(n=args.n, temperature=temp,
                                   max_tokens=512, top_k=tk), use_tqdm=False)
            ans = [z.text for o in out for z in o.outputs]
            r, top = build_metrics(ans)
            print(f"[vllm] temp={temp} top_k={tk}: fmt={r*100:.1f}%  起头Top5={top}")
        # greedy（vLLM 要求 temp=0 时 n=1）
        out = llm.generate(
            ps, SamplingParams(n=1, temperature=0.0, max_tokens=512), use_tqdm=False)
        ans = [o.outputs[0].text for o in out]
        r, top = build_metrics(ans)
        print(f"[vllm] greedy(temp=0): fmt={r*100:.1f}%  起头Top5={top}")
    else:
        import torch
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        from transformers import AutoModelForCausalLM, GenerationConfig
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa").to(dev).eval()
        for temp in (0.9, 0.7, 0.0):
            gcfg = GenerationConfig(
                max_new_tokens=512, do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_k=50, top_p=1.0, pad_token_id=tok.pad_token_id)
            ans = []
            with torch.inference_mode():
                for p in ps:
                    enc = tok(p, return_tensors="pt").to(dev)
                    o = model.generate(**enc, generation_config=gcfg)
                    ans.append(tok.decode(o[0][enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True))
            r, top = build_metrics(ans)
            print(f"[hf] temp={temp} top_k=50: fmt={r*100:.1f}%  起头Top5={top}")


if __name__ == "__main__":
    main()