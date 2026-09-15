# -*- coding: utf-8 -*-
# vLLM 单模型评测器：continuous batching(300题一次全喂)+prefix caching，单模型约1-2分钟，GPU util 80%+
# 与 eval_gsm8k_test.py 完全同口径：seed=42 抽样、greedy、math_verify 评分、相同的 system_prompt
# 一个进程只评一个模型（vLLM显存随进程退出干净释放）；多模型=多进程并行，同卡几个进程就各给 --gpu_mem≈1/N
# 用法: CUDA_VISIBLE_DEVICES=0 python eval_vllm_one.py --name dapo200 --model /path/step_200 --n 300 --gpu_mem 0.26 --out eval_v_dapo200.json
import argparse
import atexit
import hashlib
import json
import random
import os
import re
import shutil
import tempfile
from transformers import AutoTokenizer

# 【spawn 递归引爆防护】vLLM V1 默认用 multiprocessing spawn 启动 EngineCore 子进程，
# 子进程会按 spawn 语义重新执行本模块顶层代码——本脚本沿用"顶层直线流程"风格没有
# __main__ 保护，子进程会再次跑到 LLM() 触发 _check_not_importing_main RuntimeError
# （2026-09-11 eval 4B checkpoint 实测）。进程内引擎（=0）与训练端 run_gsm8k.sh 的
# 既有配置完全一致，评测为一次性批量生成无性能损失。必须在 vLLM 首次初始化前生效。
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="模型/checkpoint 路径")
parser.add_argument("--mm_base", default=None,
                    help="纯文本 Qwen3.5 ckpt 的多模态骨架目录（A2 自动物化用）；"
                         "None=先查该 ckpt 的 run_info.json 里训练时的 model_path，"
                         "再查 rlab 配置。")
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
# 【2026-09-12】per-item 落盘默认开：run2 评完只留聚合计数，导致 m200−BASE 的 +5.0pp
# 无法做同题配对的 McNemar（未配对检验 p≈0.11、白丢功效），事后无法补救。
parser.add_argument("--dump_items", action=argparse.BooleanOptionalAction, default=True,
                    help="落盘 per-item 明细（默认开，供 analysis.py 做配对检验/分层）；--no-dump_items 关闭")
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
# 【2026-09-11 eval 4B 全灭事故·二次修复】prompt 构造直接复用训练端 build_prompt
# （单点同源，调用形态物理上不可能分叉）。根因：Qwen3.5 模板支持 enable_thinking
# 开关，但只认"直接 kwarg 透传"形态——训练端 build_prompt 一直用 **kwargs 直接
# 透传所以训练正常；eval 曾两次全灭：①没传开关；②用 chat_template_kwargs={...}
# 包裹形态传，被当前 transformers 版本静默忽略（pod 实测两形态 prompt 分叉：
# 直接 kwarg -> <think>\n\n</think>，包裹 -> <think>\n）。生成以未闭合 <think>
# 开头烧穿轮预算，fmt/acc 双灭（base 同灭 = 炸协议非权重）。
from rlab.rollout import build_prompt as _build_prompt
tokenizer = AutoTokenizer.from_pretrained(args.model)
_ctkw = _rcfg.get("chat_template_kwargs")
print(f"  chat_template_kwargs={_ctkw}（经训练端 build_prompt 单点同源）")
prompts = [_build_prompt(item["Q"], system_prompt, tokenizer, _ctkw) for item in sample]
# fail-fast：请求关思考但模板没响应（如 transformers 版本行为变化），立刻告警
if _ctkw and _ctkw.get("enable_thinking") is False \
        and prompts[0].rstrip().endswith("<think>"):
    print("  [警告] 模板未响应 enable_thinking=False，生成仍将以 <think> 开头！"
          "结果会接近全灭，请检查 transformers 版本行为")

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

# ---------- 起引擎前的显存前置检查（fail-fast 在 vLLM init_device 之前）----------
def _mem_shortfall(gpu_mem: float, free: int, total: int):
    """纯函数：显存请求 vs 当前空闲。不够 → 返回带改法的说明；够 → None。"""
    want = gpu_mem * total
    if want <= free:
        return None
    used = total - free
    hint = max(0.05, (free / total) * 0.9) if total else 0.0
    return (
        f"GPU 空闲显存不够：请求 gpu_mem={gpu_mem:.2f} → {want / 2**30:.1f} GiB，"
        f"但只剩 {free / 2**30:.1f}/{total / 2**30:.1f} GiB"
        f"（{used / 2**30:.1f} GiB 被别的进程占着，常见是训练端/上一次 eval 没退干净）。\n"
        f"  修法：降 --gpu_mem 到 ≈{hint:.2f}（留 10% 余量）；或换空闲卡 --gpus；"
        "或等训练结束。\n  看占用者：nvidia-smi")


def _gpu_mem_preflight(gpu_mem: float, device: int = 0) -> None:
    """【2026-09-16 真机】--gpus 0,1 时 GPU1 被训练占着（空闲 35.4/95 GiB），默认
    gpu_mem=0.78 → 白等一次 materialize+引擎初始化，最后只拿到 vLLM 一行 ValueError
    （"Free memory ... less than desired GPU memory utilization"）。提前把三个数
    （空闲/总量/请求）与改法打出来。判不了就不拦（无 CUDA / 老 torch）。"""
    try:
        import torch
        free, total = torch.cuda.mem_get_info(device)
    except Exception as exc:
        print(f"  [GPU][警告] 显存前置检查跳过（{type(exc).__name__}: {exc}）")
        return
    print(f"  [GPU] 空闲 {free / 2**30:.1f}/{total / 2**30:.1f} GiB；"
          f"请求 gpu_mem={gpu_mem:.2f} → {gpu_mem * total / 2**30:.1f} GiB")
    msg = _mem_shortfall(gpu_mem, free, total)
    if msg:
        raise RuntimeError(msg)


# ---------- B：旧 ckpt 自动物化（vLLM A2 兜底）----------
# 2026-09-16 起新 step_N 存盘即多模态壳（train.save_checkpoint），vLLM 直读；这里管
# 两类历史产物：① 09-16 之前存下的 step_N（Qwen3_5TextConfig）；② -text 纯文本目录。
# vLLM 对它们会在 processor 构造阶段直接 TypeError（A2），先物化到临时目录再起引擎。
def _needs_mm_materialize(model_path: str) -> bool:
    """True = 纯文本 Qwen3.5 目录（vLLM 的 A2 崩法），需要先物化成多模态壳。

    复合目录（config 有 text_config）→ False；非 Qwen3.5（Qwen2.5 无多模态路由）
    → False。判不出来也返回 False：宁可不物化，不可错物化（把好模型评成别的）。
    """
    try:
        from rlab.model_loading import resolve_load_config
        if resolve_load_config(model_path)[1]:
            return False
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            mt = str(json.load(f).get("model_type", ""))
        return mt.startswith("qwen3_5")
    except Exception:
        return False


def _find_mm_base(model_path: str, cli_mm_base, cfg):
    """找可用的多模态骨架目录：--mm_base > ckpt 的 run_info.json > rlab 配置。

    只接受**真复合目录**（resolve_load_config 判有 text_config）——模型路径的默认值
    （如 /root/Qwen2.5-3B）在 Qwen3.5 评测里是错的，必须由判据挡掉，不能盲信配置。
    """
    from rlab.model_loading import resolve_load_config
    cands = [cli_mm_base]
    try:
        with open(os.path.join(model_path, "run_info.json"), encoding="utf-8") as f:
            _ri = json.load(f)
        cands.append(_ri.get("model_path"))
        cands.append((_ri.get("config") or {}).get("vllm_model_path"))
    except Exception:
        pass
    cands.append(cfg.get("vllm_model_path") or cfg.get("model_path"))
    for c in cands:
        if not c or not os.path.isdir(c):
            continue
        try:
            if resolve_load_config(c)[1]:
                return c
        except Exception:
            continue
    return None


_model_for_vllm = args.model
if _needs_mm_materialize(args.model):
    _mm_base = _find_mm_base(args.model, args.mm_base, _rcfg)
    if not _mm_base:
        raise RuntimeError(
            f"{args.model} 是纯文本 Qwen3.5 checkpoint（vLLM 的 A2：Qwen3_5TextConfig "
            "被路由到多模态实现，processor 构造直接 TypeError），但找不到可用的多模态骨架。\n"
            "  修法一：显式给骨架 --mm_base /root/Qwen3.5-4B\n"
            "  修法二：先手工物化再评：\n"
            f"    python -m rlab.materialize_mm_ckpt --text_ckpt {args.model} \\\n"
            f"        --mm_base /root/Qwen3.5-4B --out {args.model}_mm")
    _mm_tmp = tempfile.mkdtemp(prefix="_eval_mm_",
                               dir=os.path.dirname(os.path.abspath(args.model)) or ".")
    from rlab.materialize_mm_ckpt import materialize_mm_checkpoint
    print(f"  [物化] {args.model} 是纯文本 ckpt（vLLM A2）→ 多模态壳（临时）: {_mm_tmp}")
    materialize_mm_checkpoint(args.model, _mm_base, _mm_tmp)
    atexit.register(shutil.rmtree, _mm_tmp, ignore_errors=True)
    _model_for_vllm = _mm_tmp

# ---------- vLLM 批量生成 ----------
print(f"[2/3] vLLM 生成并评分 ... {name}: {args.model}")
from vllm import LLM, SamplingParams
# 【与训练同档】引擎参数从 rlab 配置取（BASE 默认 {"gdn_prefill_backend": "triton"}）：
# 评测端构造 vLLM 时同样会触发 GDN prefill 的 FlashInfer JIT 现场编译（宿主 RAM 紧时
# ninja 被 SIGKILL、无 traceback），不设档就会与训练静默分叉——Δacc 里混进 kernel 变量。
_vllm_kwargs = dict(_rcfg.get("vllm_gen_kwargs") or {})
if _vllm_kwargs:
    print(f"  vLLM 引擎参数（与训练同一份配置）: {_vllm_kwargs}")
_gpu_mem_preflight(args.gpu_mem)
llm = LLM(model=_model_for_vllm, gpu_memory_utilization=args.gpu_mem,
          max_model_len=args.max_len, dtype="bfloat16", **_vllm_kwargs)

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
items = []      # per-item 明细（--dump_items，默认开）：供 analysis.py 配对检验/分层
for i, (item, ans) in enumerate(zip(sample, answers)):
    n_valid += 1
    a = f = 0.0     # 空答案计入分母记 0 分（见上方审查修复注释）
    if len(ans.strip()) > 0:
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
    if args.dump_items:
        items.append({
            # 题面指纹：跨模型对齐用（同 seed/split 下同题同 key）——McNemar 的配对键。
            # run2 缺的正是这个键，导致 +5.0pp 只能做未配对检验（p≈0.11）。
            "qk": hashlib.sha1(str(item["Q"]).encode("utf-8")).hexdigest()[:12],
            "acc": a, "fmt": f,
            "code_used": int(code_used[i]) if code_used and i < len(code_used) else 0,
            "code_ok": int(code_ok[i]) if code_ok and i < len(code_ok) else 0,
            "ans_len": len(ans), "empty": 0 if ans.strip() else 1,
        })
    if i < args.show:
        print(f"  [a={a:.0f} f={f:.0f}] {ans[:500]}")

result = {"acc": acc / n_valid if n_valid else 0, "fmt": fmt / n_valid if n_valid else 0,
          "both": both / n_valid if n_valid else 0, "n": n_valid,
          "n_requested": args.n, "n_dropped_long": _dropped_long,
          "algo": args.algo, "eval_task": args.eval_task, "split": args.split,
          # 【2026-09-12 审计缺口补齐】旧版不记 model_path：多模型同表时事后无法核对
          # "这一行评的到底是哪个 checkpoint"（本文件 docstring 自己就在警告同名覆盖）。
          "model_path": args.model,
          "eval_protocol": {"temperature": 0, "greedy": True, "seed": args.seed,
                            "max_rounds": args.max_rounds, "round_tokens": args.round_tokens,
                            "max_tokens": args.max_tokens, "max_len": args.max_len}}
if args.dump_items:
    result["items"] = items
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
