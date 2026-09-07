# -*- coding: utf-8 -*-
# vLLM 单模型评测器：continuous batching(300题一次全喂)+prefix caching，单模型约1-2分钟，GPU util 80%+
# 与 eval_gsm8k_test.py 完全同口径：seed=42 抽样、greedy、math_verify 评分、相同的 system_prompt
# 一个进程只评一个模型（vLLM显存随进程退出干净释放）；多模型=多进程并行，同卡几个进程就各给 --gpu_mem≈1/N
# 用法: CUDA_VISIBLE_DEVICES=0 python eval_vllm_one.py --name dapo200 --model /path/step_200 --n 300 --gpu_mem 0.26 --out eval_v_dapo200.json
import argparse, json, random, re
from transformers import AutoTokenizer
from math_verify import parse, verify, ExprExtractionConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="模型/checkpoint 路径")
parser.add_argument("--name", default=None, help="表内显示名，默认取路径末2段")
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default=None, help="结果json路径，默认 eval_vllm_<name>.json")
parser.add_argument("--gpu_mem", type=float, default=0.26, help="vLLM显存占比(占总显存)；同卡并行N个进程就各给≈1/N")
parser.add_argument("--max_len", type=int, default=1280, help="prompt(~400)+生成(≤512)上限，1280足够")
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--split", default="test", choices=["test", "train"], help="test=held-out(默认)；train=训练集内抽样(过拟合诊断：train高test低=过优化实锤)")
parser.add_argument("--show", type=int, default=0, help="打印前N个原始回答")
parser.add_argument("--retool", action="store_true", help="阶段2：多轮代码交织评测（生成→执行代码→续写→最终答案），并记录代码调用率")
parser.add_argument("--max_rounds", type=int, default=3, help="--retool 时最多代码-执行轮数")
parser.add_argument("--round_tokens", type=int, default=280, help="--retool 时每轮 assistant 段生成长度上限")
args = parser.parse_args()

name = args.name or "_".join(args.model.rstrip("/").split("/")[-2:])
out_path = args.out or f"eval_vllm_{name}.json"

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


if args.retool:
    from rlab.config import system_prompt_retool
    system_prompt = system_prompt_retool


def reward_correct(answer, ground_truth):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0: return 0.0
    try:
        ans = parse(nums[-1], extraction_config=[ExprExtractionConfig()], parsing_timeout=None)
        gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()], parsing_timeout=None)
        return 1.0 if verify(ans, gt, timeout_seconds=None) else 0.0
    except Exception:
        try:
            gt_nums = re.findall(pattern, ground_truth.replace(",", ""))
            if gt_nums and abs(float(nums[-1]) - float(gt_nums[-1])) < 1e-6:
                return 1.0
        except Exception:
            pass
        return 0.0


def reward_format(answer):
    if "reasoning process here" in answer.lower():
        return 0.0
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.0 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else 0.0


# ---------- GSM8K split：仅使用 ModelScope（pod HF 网络不通，禁止回落） ----------
print(f"[1/3] 加载 GSM8K {args.split} split（ModelScope）...")
try:
    from modelscope.msdatasets import MsDataset
    from rlab.data import _patch_verification_mode

    # 兼容 modelscope 旧版向 datasets>=3 传已删除的 verification_mode
    _patch_verification_mode()

    ds = MsDataset.load("modelscope/gsm8k", subset_name="main",
                        split=args.split, trust_remote_code=True)
    if len(ds) == 0:
        raise RuntimeError(f"ModelScope GSM8K {args.split} split 为空")
    test_data = [{"Q": x["question"], "A": x["answer"]} for x in ds]
    print(f"  modelscope gsm8k [{args.split}]: {len(test_data)} 题")
except Exception as e:
    raise RuntimeError(
        f"ModelScope GSM8K {args.split} split 加载失败；"
        "已禁止回落 Hugging Face（pod HF 网络不通）。") from e

random.seed(args.seed)
sample = random.sample(test_data, min(args.n, len(test_data)))
print(f"  固定 seed={args.seed}，抽 {len(sample)} 题")

# ---------- 建 prompt（HF tokenizer 的 chat template，与训练/旧eval同一条文本路径） ----------
tokenizer = AutoTokenizer.from_pretrained(args.model)
prompts = [tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt},
     {"role": "user", "content": item["Q"]}], tokenize=False, add_generation_prompt=True)
    for item in sample]

# ---------- vLLM 批量生成（300题一次全喂，引擎内部连续批处理） ----------
print(f"[2/3] vLLM 生成并评分 ... {name}: {args.model}")
from vllm import LLM, SamplingParams
llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
          max_model_len=args.max_len, dtype="bfloat16")

code_used = code_ok = None
if args.retool:
    # 阶段2：多轮代码交织（贪心）——复用 rollout 的多轮生成逻辑（含沙箱执行）
    from rlab.rollout import multi_turn_rollout_group
    sp_mt = SamplingParams(temperature=0, max_tokens=args.round_tokens)
    mt_cfg = {"max_rounds": args.max_rounds, "sandbox_timeout": 5.0,
              "sandbox_mem_mb": 256, "tool_result_max_chars": 500}
    _segs, _full, code_stats = multi_turn_rollout_group(
        llm, sp_mt, tokenizer, prompts, mt_cfg)
    # 打分只用模型自己的 assistant 段拼接文本：全文含 prompt（格式正则 ^ 锚定
    # 必败 → fmt 恒 0，2026-09-08 实测）也含沙箱输出（"最后一个数字"会变成
    # 工具 stdout，把工具结果当模型答案发信用）。
    answers = ["".join(s["text"] for s in segs_i if s["kind"] == "assistant")
               for segs_i in _segs]
    code_used = [s["code_used"] for s in code_stats]
    code_ok = [s["code_ok"] for s in code_stats]
else:
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens))
    answers = [o.outputs[0].text for o in outs]
    code_used = code_ok = [0] * len(answers)

acc, fmt, both, n_valid = 0.0, 0.0, 0.0, 0
for i, (item, ans) in enumerate(zip(sample, answers)):
    if len(ans.strip()) == 0:
        continue
    n_valid += 1
    gt = item["A"].split("####")[-1].strip()
    a = reward_correct(ans, gt)
    f = reward_format(ans)
    acc += a; fmt += f; both += (a == 1.0 and f == 1.0)
    if i < args.show:
        print(f"  [a={a:.0f} f={f:.0f}] {ans[:400]}")

result = {"acc": acc / n_valid if n_valid else 0, "fmt": fmt / n_valid if n_valid else 0,
          "both": both / n_valid if n_valid else 0, "n": n_valid}
if args.retool and n_valid:
    result["code_rate"] = sum(1 for u in code_used if u > 0) / n_valid
    result["code_ok_rate"] = sum(1 for k in code_ok if k > 0) / n_valid
    result["avg_rounds"] = sum(code_used) / n_valid
print(f"\n[3/3] {name}（GSM8K {args.split}，N={len(sample)}）")
print(f"{name:<16}{result['acc']*100:>9.1f}%{result['fmt']*100:>9.1f}%{result['both']*100:>9.1f}%{result['n']:>10}")
if args.retool and n_valid:
    print(f"{name:<16}代码调用率 {result['code_rate']*100:.1f}%  成功率 {result['code_ok_rate']*100:.1f}%  平均轮次 {result['avg_rounds']:.2f}")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({name: result}, f, indent=2, ensure_ascii=False)
print(f"结果已存 {out_path}")
