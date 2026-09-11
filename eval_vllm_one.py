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

# 【spawn 递归引爆防护】vLLM V1 默认用 multiprocessing spawn 启动 EngineCore 子进程，
# 子进程会按 spawn 语义重新执行本模块顶层代码——本脚本沿用"顶层直线流程"风格没有
# __main__ 保护，子进程会再次跑到 LLM() 触发 _check_not_importing_main RuntimeError
# （2026-09-11 eval 4B checkpoint 实测）。进程内引擎（=0）与训练端 run_gsm8k.sh 的
# 既有配置完全一致，评测为一次性批量生成无性能损失。必须在 vLLM 首次初始化前生效。
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="模型/checkpoint 路径")
parser.add_argument("--name", default=None, help="表内显示名，默认取路径末2段")
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default=None, help="结果json路径，默认 eval_vllm_<name>.json")
parser.add_argument("--gpu_mem", type=float, default=0.26, help="vLLM显存占比(占总显存)；同卡并行N个进程就各给≈1/N")
parser.add_argument("--max_len", type=int, default=None, help="prompt+全轨迹上限；None=自动（retool 用 400+2200，其余 1280）")
parser.add_argument("--max_tokens", type=int, default=None, help="单轮/单次生成长度；None=自动取配置")
parser.add_argument("--split", default="test", choices=["test", "train"],
                    help="test=held-out（dapo_math=dev.jsonl；gsm8k=test split）；"
                         "train=训练池内抽样（过拟合诊断：train高test低=过优化实锤）")
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
    # 【2026-09-09 审查修复·split 强制生效】test=dev.jsonl(held-out)，train=训练池。
    # 旧版两个 split 都回落全量 17k train pool（训练/评测同池污染）且 dev 缺失时
    # 静默用训练池充当 held-out——现在 dev 缺失直接报错。
    from rlab.data import load_dapo_math_dev, load_dapo_math_train
    if args.split == "test":
        test_data = load_dapo_math_dev()   # 缺失时 FileNotFoundError 带指引
    else:
        test_data = load_dapo_math_train()
    print(f"  dapo_math [{args.split}]: {len(test_data)} 题")
else:
    raise ValueError(f"未知 eval_task {args.eval_task}")

random.seed(args.seed)
_n_take = min(args.n, len(test_data))
if _n_take < args.n:
    # 【2026-09-09 审查修复】n 大于池子时旧版静默缩水：dev=50 时请求 300 实评 50，
    # 二项噪声 ±6.5pp 却当 300 题的精度用。现在大字告警 + 写进结果 json。
    _noise = 1.96 * (0.5 ** 0.5) / (_n_take ** 0.5) * 100
    print(f"  [警告] 请求 n={args.n} 但池仅 {len(test_data)} 题 → 实际评测 {_n_take} 题"
          f"（最坏二项噪声 ±{_noise:.1f}pp，结论慎读）")
sample = random.sample(test_data, _n_take)
print(f"  固定 seed={args.seed}，抽 {len(sample)} 题  algo={args.algo} eval_task={args.eval_task} "
      f"max_len={args.max_len} round_tokens={args.round_tokens}")

# ---------- 建 prompt ----------
# 【2026-09-11 eval 4B 全灭事故】chat_template_kwargs 必须与训练端单点同源：
# Qwen3.5 默认 enable_thinking=True，eval 未传开关时生成以 <think> 开头，贪心解码
# 烧穿轮预算也出不了 </think>/boxed -> fmt/acc 双灭（base 与 step200 同为 2%——
# 炸的是协议不是权重）。从 rlab config 取训练同款（retool_math preset 已带
# {"enable_thinking": false}），Qwen2.5 家族模板忽略多余上下文键，无副作用。
tokenizer = AutoTokenizer.from_pretrained(args.model)
_ctkw = _rcfg.get("chat_template_kwargs")
print(f"  chat_template_kwargs={_ctkw}（与训练端 config 单点同源）")
prompts = [tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt},
     {"role": "user", "content": item["Q"]}], tokenize=False, add_generation_prompt=True,
    chat_template_kwargs=_ctkw)
    for item in sample]

# 【2026-09-09 审查修复·prompt 长度防线】训练端 plen>max_prompt_length 跳组，eval
# 旧版没有任何防线：长题多轮 ctx 增长后撞 vLLM max_model_len → 整个 eval 进程崩溃。
# 规则与训练对齐：prompt + 生成预算 + 工具段余量 > max_len 的题剔除（训练端同规则
# 根本采不到这些题，剔除后口径反而更一致），剔除数进结果 json。
_gen_budget = (args.max_rounds * args.round_tokens + 512) if is_retool_family \
    else (args.max_tokens + 64)
_kept, _dropped_long = [], 0
for _item, _p in zip(sample, prompts):
    _pl = len(tokenizer(_p, add_special_tokens=False)["input_ids"])
    if _pl + _gen_budget > args.max_len:
        _dropped_long += 1
        continue
    _kept.append((_item, _p))
if _dropped_long:
    print(f"  [警告] {_dropped_long} 题 prompt+生成预算超 max_len={args.max_len}，已剔除"
          "（与训练端跳组规则对齐）")
if not sample:
    raise RuntimeError(
        f"所有抽中题目的 prompt+生成预算都超 max_len={args.max_len}，无题可评；"
        "请调大 --max_len 或检查数据")
sample = [it for it, _ in _kept]
prompts = [p for _, p in _kept]

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
# 【2026-09-09 审查修复】空答案计入分母记 0 分——旧版 `if len(ans.strip())==0: continue`
# 把"只写代码没写答案/输出为空"的样本剔出分母，模型退化时反而美化 acc。
acc, fmt, both, n_valid = 0.0, 0.0, 0.0, 0
for i, (item, ans) in enumerate(zip(sample, answers)):
    n_valid += 1
    if len(ans.strip()) == 0:
        continue    # 空答案：计入分母，acc/fmt 记 0
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
          "n_requested": args.n, "n_dropped_long": _dropped_long,
          "algo": args.algo, "eval_task": args.eval_task, "split": args.split}
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
