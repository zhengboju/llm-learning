# -*- coding: utf-8 -*-
# vLLM 单模型评测器：continuous batching(300题一次全喂)+prefix caching，单模型约1-2分钟，GPU util 80%+
# 与 eval_gsm8k_test.py 完全同口径：seed=42 抽样、greedy、math_verify 评分、相同的 system_prompt
# 一个进程只评一个模型（vLLM显存随进程退出干净释放）；多模型=多进程并行，同卡几个进程就各给 --gpu_mem≈1/N
# 用法: CUDA_VISIBLE_DEVICES=0 python eval_vllm_one.py --name dapo200 --model /path/step_200 --n 300 --gpu_mem 0.26 --out eval_v_dapo200.json
import argparse
import json
import random
import os
import re
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="模型/checkpoint 路径")
parser.add_argument("--name", default=None, help="表内显示名，默认取路径末2段")
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default=None, help="结果json路径，默认 eval_vllm_<name>.json")
parser.add_argument("--gpu_mem", type=float, default=0.26, help="vLLM显存占比(占总显存)；同卡并行N个进程就各给≈1/N")
parser.add_argument("--max_len", type=int, default=None, help="prompt+全轨迹上限；None=自动（retool 用 400+2200，其余 1280）")
parser.add_argument("--max_tokens", type=int, default=None, help="单轮/单次生成长度；None=自动取配置")
parser.add_argument("--split", default="test", choices=["test", "train"], help="test=held-out(默认)；train=训练集内抽样(过拟合诊断：train高test低=过优化实锤)")
parser.add_argument("--show", type=int, default=0, help="打印前N个原始回答")
parser.add_argument("--retool", action="store_true", help="阶段2：多轮代码交织评测（兼容旧 flag，等价 --algo retool）")
parser.add_argument("--algo", type=str, default=None, help="算法名：grpo/dapo/retool/retool_math 等；指定后自动决定 prompt/预算/奖励口径。未指定时由 --retool 推断")
parser.add_argument("--eval_task", type=str, default=None, choices=["gsm8k", "dapo_math"], help="评测数据集；None=自动（retool_math→dapo_math，其余→gsm8k）")
parser.add_argument("--max_rounds", type=int, default=None, help="--retool 时最多代码-执行轮数；None=取训练配置")
parser.add_argument("--round_tokens", type=int, default=None, help="--retool 时每轮 assistant 段生成长度上限；None=取训练配置")
args = parser.parse_args()

# ---- algo 推断（兼容旧 --retool） ----
if args.algo is None:
    args.algo = "retool" if args.retool else "grpo"
# 归一化：retool_math 必须带下划线
if args.algo == "retool-math":
    args.algo = "retool_math"
is_retool_family = args.algo in ("retool", "retool_math")
if args.eval_task is None:
    args.eval_task = "dapo_math" if args.algo == "retool_math" else "gsm8k"

# ---- 预算/轮次自动对齐训练配置 ----
from rlab.config import get_config as _get_config, BASE as _BASE_CFG

try:
    _rcfg = _get_config(args.algo)
except KeyError:
    # 未知 algo（如直接传 base_path），回落 BASE
    _rcfg = dict(_BASE_CFG)

if is_retool_family:
    if args.round_tokens is None:
        args.round_tokens = _rcfg.get("round_gen_tokens", 400)
    if args.max_len is None:
        args.max_len = _rcfg.get("max_prompt_length", 400) + _rcfg.get("max_context_tokens", 2200)
    if args.max_rounds is None:
        args.max_rounds = _rcfg.get("max_rounds", 3)
    if args.max_tokens is None:
        args.max_tokens = _rcfg.get("max_gen_tokens", 512)
else:
    if args.max_len is None:
        args.max_len = 1280
    if args.max_tokens is None:
        args.max_tokens = _rcfg.get("max_gen_tokens", 512)
    if args.max_rounds is None:
        args.max_rounds = _rcfg.get("max_rounds", 3)

name = args.name or "_".join(args.model.rstrip("/").split("/")[-2:])
out_path = args.out or f"eval_vllm_{name}.json"

# ---- system_prompt 对齐训练 ----
if args.algo == "retool_math":
    from rlab.config import system_prompt_retool_math
    system_prompt = system_prompt_retool_math
elif args.algo == "retool":
    from rlab.config import system_prompt_retool
    system_prompt = system_prompt_retool
else:
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
 The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

# ---- 奖励口径复用 rlab.reward（单点真相，不再 дублировать） ----
from rlab.reward import (
    strip_code_blocks as _strip_code_blocks,
    extract_last_boxed as _extract_last_boxed,
    _MATH_BOXED_WINDOW,
)
from math_verify import parse, verify, ExprExtractionConfig

def _mv_parse(text: str):
    return parse(text, extraction_config=[ExprExtractionConfig()], parsing_timeout=None)

def reward_correct_gsm8k(answer: str, ground_truth: str) -> float:
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return 0.0
    try:
        ans = _mv_parse(nums[-1])
        gt = _mv_parse(ground_truth)
        return 1.0 if verify(ans, gt, timeout_seconds=None) else 0.0
    except Exception:
        try:
            gt_nums = re.findall(pattern, ground_truth.replace(",", ""))
            if gt_nums and abs(float(nums[-1]) - float(gt_nums[-1])) < 1e-6:
                return 1.0
        except Exception:
            pass
        return 0.0

def reward_correct_boxed_eval(answer: str, ground_truth: str) -> float:
    boxed = _extract_last_boxed(answer[-_MATH_BOXED_WINDOW:])
    if boxed is None:
        return 0.0
    try:
        ans = _mv_parse(f"${boxed.strip()}$")
        gt = _mv_parse(f"${ground_truth.strip()}$")
        return 1.0 if verify(ans, gt, timeout_seconds=None) else 0.0
    except Exception:
        return 0.0

def reward_format_gsm8k(answer: str) -> float:
    if "reasoning process here" in answer.lower():
        return 0.0
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.0 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else 0.0

def reward_format_boxed(answer: str) -> float:
    return 1.0 if _extract_last_boxed(answer[-_MATH_BOXED_WINDOW:]) is not None else 0.0

# ---------- 数据集加载 ----------
print(f"[1/3] 加载 {args.eval_task} {args.split} split ...")
if args.eval_task == "gsm8k":
    try:
        from modelscope.msdatasets import MsDataset
        from rlab.data import _patch_verification_mode
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
elif args.eval_task == "dapo_math":
    # 优先本地 jsonl（prepare_dapo_math 产物），失败回落 MsDataset/HF
    local_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlab", "datasets", "dapo_math", "dev.jsonl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlab", "datasets", "dapo_math", "train.jsonl"),
        "./rlab/datasets/dapo_math/dev.jsonl",
        "./rlab/datasets/dapo_math/train.jsonl",
    ]
    test_data = None
    for cand in local_candidates:
        if os.path.exists(cand):
            rows = []
            with open(cand, encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        r=json.loads(line)
                        q=r.get("question") or r.get("Q")
                        a=r.get("answer") or r.get("A")
                        if q and a:
                            rows.append({"Q": q, "A": a})
                    except Exception:
                        continue
            if rows:
                test_data = rows
                print(f"  本地 dapo_math {cand}: {len(test_data)} 题")
                break
    if test_data is None:
        try:
            from rlab.data import load_dapo_math_train
            rows = load_dapo_math_train()
            test_data = rows
            print(f"  rlab.data DAPO-Math-17k: {len(test_data)} 题")
        except Exception as e:
            raise RuntimeError(f"DAPO-Math 加载失败: {e}") from e
else:
    raise ValueError(f"未知 eval_task {args.eval_task}")

random.seed(args.seed)
sample = random.sample(test_data, min(args.n, len(test_data)))
print(f"  固定 seed={args.seed}，抽 {len(sample)} 题  algo={args.algo} eval_task={args.eval_task} max_len={args.max_len} round_tokens={args.round_tokens}")

# ---------- 建 prompt ----------
tokenizer = AutoTokenizer.from_pretrained(args.model)
prompts = [tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt},
     {"role": "user", "content": item["Q"]}], tokenize=False, add_generation_prompt=True)
    for item in sample]

# ---------- vLLM 批量生成 ----------
print(f"[2/3] vLLM 生成并评分 ... {name}: {args.model}")
from vllm import LLM, SamplingParams
llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
          max_model_len=args.max_len, dtype="bfloat16")

code_used = code_ok = None
if is_retool_family:
    from rlab.rollout import multi_turn_rollout_group
    sp_mt = SamplingParams(temperature=0, max_tokens=args.round_tokens)
    mt_cfg = {"max_rounds": args.max_rounds, "sandbox_timeout": 5.0,
              "sandbox_mem_mb": 256, "tool_result_max_chars": 500}
    _segs, _full, code_stats = multi_turn_rollout_group(
        llm, sp_mt, tokenizer, prompts, mt_cfg)
    answers = ["".join(s["text"] for s in segs_i if s["kind"] == "assistant")
               for segs_i in _segs]
    # 打分域：retool 去代码块后再判，retool_math 直接 boxed（代码块不影响 boxed 提取，但为一致仍可剥离）
    if args.algo == "retool_math":
        # boxed 提取已只看末300字符，代码块残留不干扰；为与训练一致不剥离也行，但剥离更干净
        answers = [_strip_code_blocks(a) for a in answers]
    else:
        answers = [_strip_code_blocks(a) for a in answers]
    code_used = [s["code_used"] for s in code_stats]
    code_ok = [s["code_ok"] for s in code_stats]
else:
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens))
    answers = [o.outputs[0].text for o in outs]
    code_used = code_ok = [0] * len(answers)

# ---------- 评分 ----------
acc, fmt, both, n_valid = 0.0, 0.0, 0.0, 0
for i, (item, ans) in enumerate(zip(sample, answers)):
    if len(ans.strip()) == 0:
        continue
    n_valid += 1
    # ground_truth 归一：gsm8k 带 ####，dapo 直接答案
    if args.eval_task == "gsm8k":
        gt = item["A"].split("####")[-1].strip()
        a = reward_correct_gsm8k(ans, gt)
        f = reward_format_gsm8k(ans)
    else:
        gt = str(item["A"]).strip()
        a = reward_correct_boxed_eval(ans, gt)
        f = reward_format_boxed(ans)
    acc += a; fmt += f; both += (a == 1.0 and f == 1.0)
    if i < args.show:
        print(f"  [a={a:.0f} f={f:.0f}] {ans[:500]}")

result = {"acc": acc / n_valid if n_valid else 0, "fmt": fmt / n_valid if n_valid else 0,
          "both": both / n_valid if n_valid else 0, "n": n_valid,
          "algo": args.algo, "eval_task": args.eval_task}
if is_retool_family and n_valid:
    result["code_rate"] = sum(1 for u in code_used if u > 0) / n_valid
    result["code_ok_rate"] = sum(1 for k in code_ok if k > 0) / n_valid
    result["avg_rounds"] = sum(code_used) / n_valid
print(f"\n[3/3] {name}（{args.eval_task} {args.split}，N={len(sample)} algo={args.algo}）")
print(f"{name:<16}{result['acc']*100:>9.1f}%{result['fmt']*100:>9.1f}%{result['both']*100:>9.1f}%{result['n']:>10}")
if is_retool_family and n_valid:
    print(f"{name:<16}代码调用率 {result['code_rate']*100:.1f}%  成功率 {result['code_ok_rate']*100:.1f}%  平均轮次 {result['avg_rounds']:.2f}")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({name: result}, f, indent=2, ensure_ascii=False)
print(f"结果已存 {out_path}")
