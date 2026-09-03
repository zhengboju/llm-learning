# -*- coding: utf-8 -*-
# 1分钟复现：串行逐题 vs 批量(左pad)生成是否一致（定位 eval 全 0.000 的根因）
# 用法: python debug_batch.py   （单卡跑1分钟，BASE，只看前2题）
import random
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

base_path = "/root/Qwen2.5-3B"
print("transformers:", transformers.__version__)

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

# 取与 eval 完全相同的 seed=42 前2题
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test", trust_remote_code=True)
test_data = [{"Q": x["question"], "A": x["answer"]} for x in ds]
random.seed(42)
sample = random.sample(test_data, 300)[:2]
print(f"题目数={len(sample)}，第1题GT={sample[0]['A'].split('####')[-1].strip()}")

tok = AutoTokenizer.from_pretrained(base_path)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
print("pad_id=", tok.pad_token_id, "eos_id=", tok.eos_token_id)
model = AutoModelForCausalLM.from_pretrained(
    base_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").cuda().eval()
print("sliding_window=", getattr(model.config, "sliding_window", None),
      "attn_impl=", getattr(model.config, "_attn_implementation", None))


def build(q):
    return tok.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


prompts = [build(item["Q"]) for item in sample]

# ---- (a) 串行逐题（老路径，之前能跑出67%的那个） ----
serial_out = []
with torch.inference_mode():
    for p in prompts:
        ids = tok(p, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        serial_out.append(out[0][ids["input_ids"].shape[1]:].tolist())

# ---- (b) 批量左pad（新路径） ----
tok.padding_side = "left"
with torch.inference_mode():
    enc = tok(prompts, return_tensors="pt", padding=True)
    Lmax = enc["input_ids"].shape[1]
    print("各题真实长度:", enc["attention_mask"].sum(dim=1).tolist(), "Lmax=", Lmax)
    # 校验：batch 行真实部分必须与串行 ids 逐字一致
    for i, p in enumerate(prompts):
        s_ids = tok(p, return_tensors="pt")["input_ids"][0].tolist()
        Li = len(s_ids)
        b_ids = enc["input_ids"][i][Lmax - Li:].tolist()
        print(f"  题{i}: 串行len={Li}，batch行一致={s_ids == b_ids}")
    out = model.generate(input_ids=enc["input_ids"].to("cuda"),
                         attention_mask=enc["attention_mask"].to("cuda"),
                         max_new_tokens=64, do_sample=False,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    batch_out = [out[i][Lmax:].tolist() for i in range(len(prompts))]

for i in range(len(prompts)):
    print(f"\n===== 题{i} =====")
    print("串行前15ids:", serial_out[i][:15])
    print("批量前15ids:", batch_out[i][:15])
    print("前15一致:", serial_out[i][:15] == batch_out[i][:15])
    print("串行解码:", tok.decode(serial_out[i], skip_special_tokens=True)[:200])
    print("批量解码:", tok.decode(batch_out[i], skip_special_tokens=True)[:200])
