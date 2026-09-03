# -*- coding: utf-8 -*-
# 诊断：DAPO step_200 在 GSM8K test 上 greedy 生成的格式失败原因分类
# 顺带交叉验证 eval_gsm8k_test.py 的口径：本脚本的 strict fmt 应≈ eval 的 35.7%
# 用法: CUDA_VISIBLE_DEVICES=0 python diag_format_fail.py --model /root/llm-learning/simple_grpo_v1/step_200 --n 100
import re, random, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify, ExprExtractionConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/root/llm-learning/simple_grpo_v1/step_200")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ===== 与训练(grpo_ref_split.py)逐字一致的正则和提示词 =====
SYSTEM_PROMPT = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
STRICT = re.compile(r"^<think>.*?</think><answer>.*?</answer>$", re.DOTALL | re.VERBOSE)
RELAXED = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL | re.VERBOSE)

def reward_correct(answer, ground_truth):
    nums = re.findall(r'\d+\.\d+|\d+/\d+|\d+', answer)
    if len(nums) == 0: return 0.0
    try:
        ans = parse(nums[-1], extraction_config=[ExprExtractionConfig()])
        gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
        return 1.0 if verify(ans, gt) else 0.0
    except Exception:
        return 0.0

# ===== 数据（与 eval_gsm8k_test.py 相同的加载与抽样）=====
print("[1/3] 加载 GSM8K test ...")
test_data = None
try:
    from modelscope.msdatasets import MsDataset
    ds = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test", trust_remote_code=True)
    test_data = [{"Q": x["question"], "A": x["answer"]} for x in ds]
    print(f"  modelscope: {len(test_data)} 题")
except Exception as e:
    print(f"  modelscope 失败: {e}")
if test_data is None:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    test_data = [{"Q": x["question"], "A": x["answer"].split("####")[-1].strip()} for x in ds]
    print(f"  datasets: {len(test_data)} 题")

random.seed(args.seed)
sample = random.sample(test_data, min(args.n, len(test_data)))

# ===== 生成（与 eval 相同：greedy, max_new_tokens=512, skip_special_tokens）=====
print(f"[2/3] 加载 {args.model} 生成 {len(sample)} 题 ...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").cuda().eval()

outputs = []
with torch.inference_mode():
    for i, item in enumerate(sample):
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": item["Q"]}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to("cuda")
        plen = ids["input_ids"].shape[1]
        out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                             num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
        ans = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
        gt = item["A"].split("####")[-1].strip()
        outputs.append({"ans": ans, "gt": gt})
        if (i + 1) % 20 == 0: print(f"  {i+1}/{len(sample)}")

# ===== 分类统计 =====
def categorize(a):
    """返回失败原因列表（可能多个）"""
    reasons = []
    if "reasoning process here" in a.lower():
        reasons.append("占位符抄写")
    to, tc = a.count("<think>"), a.count("</think>")
    ao, ac = a.count("<answer>"), a.count("</answer>")
    if not a.startswith("<think>"):
        reasons.append(f"开头非<think>={a[:24]!r}")
    if (to, tc) != (1, 1): reasons.append(f"think标签数={to}/{tc}")
    if (ao, ac) != (1, 1): reasons.append(f"answer标签数={ao}/{ac}")
    m = re.search(r"</think>(\s*)<answer>", a)
    if m and m.group(1):
        reasons.append(f"</think>与<answer>间有空白={m.group(1)!r}")
    if not a.rstrip().endswith("</answer>"):
        reasons.append(f"结尾非</answer>={a[-48:]!r}")
    return reasons

print("\n[3/3] 分类结果 =====")
n = len(outputs)
strict_n = sum(1 for o in outputs if STRICT.match(o["ans"]))
relax_n  = sum(1 for o in outputs if RELAXED.match(o["ans"]))
acc_n    = sum(reward_correct(o["ans"], o["gt"]) for o in outputs)
print(f"总数={n}  严格格式(训练口径)={strict_n/n*100:.1f}%  "
      f"宽松格式(允许空白)={relax_n/n*100:.1f}%  acc={acc_n/n*100:.1f}%")
print(f"（对照：eval_gsm8k_test.py 测得 fmt=35.7% —— 若本脚本 strict 值接近，说明 pod eval 干净）\n")

cat_count = {}
fails = []
for o in outputs:
    if not STRICT.match(o["ans"]):
        rs = categorize(o["ans"])
        fails.append((o["ans"], rs))
        for r in rs:
            key = r.split("=")[0].split(":")[0]
            cat_count[key] = cat_count.get(key, 0) + 1

print("失败原因分布（一条样本可计入多个原因）:")
for k, v in sorted(cat_count.items(), key=lambda x: -x[1]):
    print(f"  {k:<28} {v} ({v/(n-strict_n)*100:.0f}% of 失败样本)")

print(f"\n失败样本示例（前 {min(6, len(fails))} 条，head/tail 用 repr 显示不可见字符）:")
for ans, rs in fails[:6]:
    print(f"\n--- 原因: {rs}")
    print(f"  HEAD: {ans[:60]!r}")
    print(f"  TAIL: {ans[-90:]!r}")
