# -*- coding: utf-8 -*-
# 对比 base (Qwen2.5-3B) 和 GRPO 训练后的 step_200
# 用法: python eval_step200.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

base_path = "/root/Qwen2.5-3B"
tuned_path = "./step_200"

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within  thinking  response and<answer> </answer> tags, respectively, i.e.,  thinking reasoning process here  response<answer> answer here </answer>."""

questions = [
    # 挑几条训练日志里见过的题 + 一条没见过的
    "A store has 20 units of a certain product, 5 of which are defective. Customer A buys 3, Customer B buys some, Customer C buys 7, and all non-defective units are sold. How many did Customer B buy?",
    "A gecko eats 70 crickets. On day one she eats 30% of the crickets. On day two she eats 6 less crickets than day one. How many crickets does she eat on day three?",
    "If 3 cats catch 3 rats in 3 minutes, how long will it take 100 cats to catch 100 rats?",
]

def build_prompt(q):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)

for name, path in [("BASE", base_path), ("GRPO_step200", tuned_path)]:
    print(f"\n{'='*70}\n  {name}  ({path})\n{'='*70}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").cuda().eval()
    for q in questions:
        text = build_prompt(q)
        ids = tokenizer(text, return_tensors="pt").to("cuda")
        plen = ids["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=300,
                                 do_sample=True, temperature=0.9, num_return_sequences=2,
                                 pad_token_id=tokenizer.pad_token_id)
        print(f"\nQ: {q}")
        for i in range(2):
            ans = tokenizer.decode(out[i][plen:], skip_special_tokens=True)
            print(f"  --- 样本{i+1} ---\n  {ans}")
    del model, tokenizer
    torch.cuda.empty_cache()
