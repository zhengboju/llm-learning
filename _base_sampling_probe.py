# -*- coding: utf-8 -*-
# 一次性诊断脚本（pod 上跑）：BASE@训练采样参数 的格式基线
# 修掉上一版只测 outputs[0] 的 bug：遍历全部 n=4 条；20 题×4=80 条置信度够判读。
#
# 【防改写】标签经聊天界面/渲染器拷贝会被静默替换成普通文字（AGENTS.local.md 教训），
# 本脚本自带字节自检：system_prompt 与评分正则里若没有真实标签字节直接中止。
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

_TAGS = (" thinking  response", "<answer>", "</answer>")

tok = AutoTokenizer.from_pretrained("/root/Qwen2.5-3B")
llm = LLM(model="/root/Qwen2.5-3B", gpu_memory_utilization=0.3)

sysp = ("You are a helpful assistant. A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. The Assistant first "
        "thinks about the reasoning process in the mind and then provides the user "
        "with the answer. The reasoning process and answer are enclosed within "
        " thinking  response and<answer> </answer> tags, respectively, i.e., "
        " thinking reasoning process here  response<answer> answer here </answer>.")
for tag in _TAGS:
    if tag not in sysp:
        raise SystemExit(f"[自检失败] system_prompt 里缺少真实标签字节 {tag!r} —— 文件被改写，中止")

qs = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "A deep-sea monster rises up once every 8 years to eat 2 fish, but a village offers it 3 fish this time. How many fish will it have eaten after 8 more years?",
    "Ken had fifty pairs of shoes. If he sold half of his pairs, how many individual shoes did he remain with?",
    "Lana puts an 8% sales tax on a $12.50 shirt. How much does she end up paying?",
    "At the market, a bunch of bananas costs $3.50 and a carton of eggs costs $2. If John buys two bunches and a carton, how much does he pay?",
    "A library has 3 floors, each with 250 books. If 180 books are checked out, how many remain?",
    "A train travels 120 km in 2 hours. At the same speed, how far does it travel in 5 hours?",
    "A rectangle is 6 cm wide and 9 cm long. What is its area in square centimeters?",
    "A store sells apples at 3 for $2. How much do 12 apples cost?",
    "Tom has 5 more marbles than Sam. Sam has 12. How many do they have together?",
    "A recipe needs 2 cups of flour for 12 cookies. How many cups for 30 cookies?",
    "A bike costs $240. It is on sale for 25% off. What is the sale price?",
    "A tank fills at 30 liters per minute. How long to fill 750 liters?",
    "Sue reads 5 pages a night for 2 weeks. How many pages did she read?",
    "A box holds 6 cans. How many boxes for 42 cans?",
    "Three friends split a $45 dinner bill evenly. How much does each pay?",
    "A phone battery loses 20% charge per hour idle. After 2 hours, what percent remains?",
    "A garden has 4 rows of 7 tulips and 3 rows of 5 roses. How many flowers total?",
    "A ticket costs $9.50; a family buys 4 tickets and pays with $50. What is the change?",
]

prompts = [tok.apply_chat_template(
    [{"role": "system", "content": sysp}, {"role": "user", "content": q}],
    tokenize=False, add_generation_prompt=True) for q in qs]

sp = SamplingParams(n=4, temperature=0.9, top_p=1.0, top_k=50, max_tokens=512, seed=42)
outs = llm.generate(prompts, sp, use_tqdm=False)

pat = re.compile(r"^ thinking.*? response<answer>.*?</answer>$", re.DOTALL | re.VERBOSE)
if " response<answer>" not in pat.pattern:
    raise SystemExit("[自检失败] 评分正则缺少标签字节，中止")
n_fmt = n_total = 0
per_prompt = []
for o in outs:
    cnt = sum(1 for z in o.outputs if pat.match(z.text))
    per_prompt.append((cnt, len(o.outputs)))
    n_fmt += cnt; n_total += len(o.outputs)
print(f"BASE@temp0.9/top_k50/n4/seed42 格式率 = {n_fmt}/{n_total} = {n_fmt/n_total:.1%}")
print("  每题(条数):", per_prompt)
for o in outs[:2]:
    for z in o.outputs[:4]:
        print("  ---"); print(z.text[:120].replace("\n", " "))