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
# 【2026-09-18 采样评测】参考项目（agentic-rl-lab/05-retool）的增益全在采样下显现
# （Average@N，temp 1.0 / top_p 0.7）；greedy 会把多轮代码行为测成灭绝（docs §7.5.2
# mode vs mixture 已记录）。--val_n>1 时每题采样 val_n 条（per-item 记首条，
# 聚合 acc=Average@N），采样参数默认对齐参考（temp 1.0 / top_p 0.7）。
parser.add_argument("--val_n", type=int, default=1,
                    help="每题采样数（>1 启用 Average@N 采样评测，参考项目口径）；默认 1=greedy")
parser.add_argument("--temperature", type=float, default=None,
                    help="采样温度；None=val_n>1 时取 1.0（参考项目），否则 0（greedy）")
parser.add_argument("--top_p", type=float, default=None,
                    help="采样 top_p；None=val_n>1 时取 0.7（参考项目），否则 1.0")
parser.add_argument("--proto_from", default=None,
                    help="协议来源目录（含 run_info.json）；None=从 --model 自己的 run_info 回读。"
                         "供 BASE 等无 run_info 的裸模型复用『被测 checkpoint 的训练协议』，"
                         "保证 Δacc 同档（2026-09-17）。")
# 【2026-09-20 采样档可复现性】确定性档开关：默认 None=从 run_info 回读训练档
# （采样评测不开则同权重重跑漂移 ~2pp，实测 BASE 63.1→61.1）。两个键必须成对，
# 只开 batch_invariant 不给 backend 会在引擎构造时 RuntimeError（已 fail-fast 前拦）。
parser.add_argument("--vllm_batch_invariant", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="确定性档（VLLM_BATCH_INVARIANT=1）；None=随训练 run_info。"
                         "采样评测（--val_n>1）强烈建议开，否则跨 run 不可比")
parser.add_argument("--vllm_attention_backend", default=None,
                    help="显式 attention backend（FLASH_ATTN/TRITON_ATTN）；"
                         "None=随训练 run_info。确定性档必需（与上一项成对）")
args = parser.parse_args()

# ---- 从 checkpoint 回读训练协议（run_info.json）----
# 【2026-09-17 对齐缺口】此前 eval 的预算/提示一律取 preset 默认：训练用
# `--system_prompt_file`/`--round_gen_tokens 6144` 等 CLI 覆盖时，eval 会静默落到
# 默认 3072/默认提示——"评测一个 checkpoint"实际测的是第三种协议，Δacc 无法自证。
# 现在：CLI 显式传参 > run_info.json 的训练 config > preset 默认，并在偏离时醒目告警。
def _load_run_cfg(model_path: str):
    """读 ckpt 的 run_info.json['config']（训练时的完整 cfg）；缺失/损坏 → None。

    注意顶层另有 signature/git_head 等身份字段，但协议本体在 config 里（write_run_info
    落盘的是 cfg 全量）。None 视为"出处不明的旧 ckpt"，回落 preset 默认。"""
    try:
        with open(os.path.join(model_path, "run_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        cfg = info.get("config") or {}
        if not isinstance(cfg, dict):
            return None
        return {"signature": info.get("signature"), "config": cfg}
    except (OSError, ValueError):
        return None

_run = _load_run_cfg(args.proto_from) if args.proto_from else _load_run_cfg(args.model)
_run_cfg = (_run["config"] if _run else {}) or {}
_proto_src = args.proto_from or args.model

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
# 【2026-09-17】run_info 的训练 config 优先于 preset：CLI 显式传参仍覆盖（见下）。
if _run_cfg:
    _rcfg = {**_rcfg, **_run_cfg}

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
from rlab.config import default_system_prompt as _default_system_prompt
# 【2026-09-17】训练用 --system_prompt_file 时，run_info.config.system_prompt 是文件内容；
# eval 端回读同一内容，并用 run_info 的 signature 里的 `-sp<hash>` 校验哈希。
_sp_run = _run_cfg.get("system_prompt")
if args.algo == "retool_math":
    from rlab.config import system_prompt_retool_math
    _sp_preset = system_prompt_retool_math
elif args.algo == "retool":
    from rlab.config import system_prompt_retool
    _sp_preset = system_prompt_retool
else:
    _sp_preset = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
 The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

if _sp_run and _sp_run != _default_system_prompt(args.algo):
    system_prompt = _sp_run
    _sp_src = "run_info(config.system_prompt)"
else:
    system_prompt = _sp_preset
    _sp_src = "preset 默认"
_sp_sha = hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:6]
_sp_sig = ""
if _run and _run.get("signature"):
    import re as _re
    _m = _re.search(r"-sp([0-9a-f]{6})", _run["signature"])
    if _m:
        _sp_sig = _m.group(1)
if _sp_sig and _sp_sha != _sp_sig:
    print(f"[eval][警告] system_prompt 与训练签名不符：训练 -sp{_sp_sig} ≠ eval -sp{_sp_sha}\n"
          f"  （eval 用 {_sp_src}；若训练是 --system_prompt_file 且文件已变，请核对）")

# ---- 评测协议来源自证（预算/提示/采样）----
print(f"[eval] 协议来源: run_info={'有' if _run else '无（preset 默认）'} "
      f"(src={_proto_src}) | "
      f"algo={args.algo} eval_task={args.eval_task} | "
      f"round_tokens={args.round_tokens} max_len={args.max_len} max_rounds={args.max_rounds} | "
      f"system_prompt={_sp_src} sp_sha={_sp_sha}")
# 【2026-09-20 回落告警】任何模型读不到 run_info 就会**静默**用 preset 默认协议跑。
# 事故形态（本次实测）：`--skip_base --models "baseA=/root/Qwen3.5-4B,baseB=..."`
# —— 裸模型目录没有 run_info，而调度器的同档兜底（eval_vllm.py 的 BASE_PROTO）
# 只认名字恰好是 "BASE" 的那一项，于是两个 base 双双回落 preset：
# round_tokens 6144→2048、ctx 26400→14336、sp 3aac5d→72078f。
# 结果 fmt 从 ~70% 掉到 34.4%、acc 31.3% —— 看起来像"模型很差"，实际是**测了
# 另一个协议**。此前只有 BASE 缺 run_info 且 tuned 里也没有时才告警（那条分支
# 在"根本没有名为 BASE 的项"时压根不进），所以这次全程没有任何提示。
# 现在：只要回落 preset 就醒目告警，并直接给出旁路（--proto_from）。
if not _run:
    print(f"\n[eval][警告] {args.model} 读不到 run_info.json → 协议**回落 preset 默认**：\n"
          f"    round_tokens={args.round_tokens} max_rounds={args.max_rounds} "
          f"max_len={args.max_len} sp_sha={_sp_sha}\n"
          f"  若被测对象是用别的预算/提示训出来的，这些数与它不同档，Δacc 无意义\n"
          f"  （典型签名：fmt 相对同档基线掉一半 = 轮预算装不下推理）。\n"
          f"  → 显式指定协议来源: --proto_from <某个带 run_info 的 step_N 目录>\n", flush=True)

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
    # 【2026-09-22 分布对齐修复】训练和评测用同一套难度过滤。
    # 此前 train split 从 run_info 回读 difficulty_path/band 过滤，而 test split
    # （dev.jsonl）不过滤 → 两个 split 难度分布不同 → train eval 偏高（p9 假信号：
    # train +9.2pp vs test +0.8pp）。
    # 正确做法：probe 同时覆盖训练池 + dev 集（见 probe_difficulty.py），eval 两端
    # 用同一张表过滤到同一 difficulty_band → 同分布 → 公平对比且不被 p≈0 题稀释。
    from rlab.data import (load_dapo_math_dev, load_dapo_math_train,
                           load_difficulty_table, filter_qas_by_difficulty)
    if args.split == "test":
        test_data = load_dapo_math_dev()   # 缺失时 FileNotFoundError 带指引
    else:
        test_data = load_dapo_math_train()
    # 两个 split 都用训练端同一套难度过滤（difficulty_path/band 从 run_info 回读）
    _dp = (_run_cfg.get("difficulty_path") or "")
    if _dp:
        _lo, _hi = _run_cfg.get("difficulty_band", (0.0, 1.0))
        if not isinstance(_lo, (int, float)) or not isinstance(_hi, (int, float)):
            _lo, _hi = 0.0, 1.0
        try:
            _tbl = load_difficulty_table(_dp)
        except OSError:
            _tbl = {}
        if _tbl:
            test_data, _dstat = filter_qas_by_difficulty(test_data, _tbl, lo=_lo, hi=_hi)
            print(f"  [eval][{args.split}] 难度过滤 band=({_lo},{_hi}): "
                  f"{_dstat['total']} -> {_dstat['kept']} 题"
                  f"（p_zero {_dstat['p_zero']} / p_one {_dstat['p_one']} / "
                  f"band_out {_dstat['band_out']} / missing {_dstat['missing']}）")
        else:
            print(f"  [eval][警告] 难度表读不到（{_dp}）→ 按 {args.split} 全量池抽")
    else:
        print(f"  [eval][警告] 训练 run_info 无 difficulty_path → "
              f"{args.split} split 按全量池抽（含 p≈0 题，效果被稀释）")
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
# 【2026-09-20 口径对齐】旧版只有 max_len 这一道线，实际阈值 = max_len − 生成预算
# = 27424 − (4×6144+512) = 2336，而训练端是 `plen > max_prompt_length` = 1024：
# 1024 < plen ≤ 2336 的题**训练端永远采不到、eval 端照评**，Δacc 混进了分布外题。
# 现在两条线都判：先按训练端的 max_prompt_length（同源回读），再保留 max_len 兜底
# （防撞 vLLM max_model_len 崩进程）。
_max_plen = int(_rcfg.get("max_prompt_length") or 0)
_gen_budget = (args.max_rounds * args.round_tokens + 512) if is_retool_family \
    else (args.max_tokens + 64)
_kept, _dropped_long, _dropped_plen = [], 0, 0
for _item, _p in zip(sample, prompts):
    _pl = len(tokenizer(_p, add_special_tokens=False)["input_ids"])
    if _max_plen and _pl > _max_plen:
        _dropped_plen += 1          # 训练端同规则跳组 → 分布内一致性
        continue
    if _pl + _gen_budget > args.max_len:
        _dropped_long += 1          # 兜底：防多轮 ctx 撞 max_model_len
        continue
    _kept.append((_item, _p))
if _dropped_plen:
    print(f"  [对齐] {_dropped_plen} 题 prompt>{_max_plen}(max_prompt_length)，已剔除"
          f"（训练端同规则跳组，这些题不在训练分布内）")
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
# 【2026-09-20 采样档可复现性】确定性档必须与训练同步接线——否则采样评测不可复现：
# 实测同权重/同 --seed/同 --n 重跑 BASE 漂移 63.1→61.1（-2.0pp），而 3pp 级效应正
# 埋在这个地板里。抽题与轨迹 seed 本来就是确定的（random.seed(--seed) 之后
# random.sample 与 randrange 同属一条流，已实测同 seed/同 n 逐位复现），真正的
# 抖动源是 vLLM 侧：批调度 + bf16 归约顺序（docs/07 已定案）。训练端 gen_worker
# 走的正是 VLLM_BATCH_INVARIANT=1 + 显式 attention backend 这一对（缺一即失效，
# batch_invariant_guard 会 fail-fast），eval 此前只取 vllm_gen_kwargs、把这两个
# 键静默丢掉 = 评测与训练不同档。CLI 可显式覆盖，默认从 run_info 回读训练档。
from rlab.rollout import attention_backend_kwargs as _attn_kw
from rlab.rollout import batch_invariant_guard as _bi_guard

_bi = _rcfg.get("vllm_batch_invariant") if args.vllm_batch_invariant is None \
    else args.vllm_batch_invariant
_attn_be = args.vllm_attention_backend or _rcfg.get("vllm_attention_backend")
_bi_guard(bool(_bi), _attn_be)      # 开了确定性档却缺 backend → 构造前 raise
if _bi:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    print("  [确定性档] VLLM_BATCH_INVARIANT=1（与训练同档；关 custom all-reduce、"
          "改用确定性 kernel，吞吐有代价）")
if _attn_be:
    _vllm_kwargs.update(_attn_kw(_attn_be))
    print(f"  [确定性档] attention backend={_attn_be} → {_vllm_kwargs}")
if not _bi and args.val_n > 1:
    print("  [警告] 采样评测（--val_n>1）未开确定性档：同权重重跑会有 ~2pp 漂移"
          "（实测 BASE 63.1→61.1），3pp 级效应无法与噪声区分。"
          "训练档若已开，请确认 run_info 可读；或显式传 "
          "--vllm_batch_invariant --vllm_attention_backend FLASH_ATTN")
_gpu_mem_preflight(args.gpu_mem)
llm = LLM(model=_model_for_vllm, gpu_memory_utilization=args.gpu_mem,
          max_model_len=args.max_len, dtype="bfloat16", **_vllm_kwargs)

code_used = code_ok = None
# 【2026-09-18 采样评测】--val_n>1 时每题采样 val_n 条（Average@N，参考项目口径）。
# greedy（val_n=1）保持历史行为逐位不变：retool 家族贪心（temp=0）→ 多轮生成；
# 非 retool 贪心 → 单轮生成。采样档（val_n>1）：
#   · 非 retool：直接 llm.generate(n=val_n)，聚合 acc = 平均正确率（Average@N）
#   · retool：多轮生成每样本独立 seed（参考项目逐条独立采样），聚合同理
_sampling = args.val_n > 1
if _sampling:
    _temp = args.temperature if args.temperature is not None else 1.0
    # 【2026-09-21 top_p 与训练对齐】旧版硬编码 0.7（参考项目口径），但训练端
    # retool_math 用 top_p=1.0。采样评测用 0.7 会截断训练分布的高尾 token →
    # 测的是不同采样分布下的表现。从 run_info config 回读 top_p，无则回落 0.7。
    _topp_default = _rcfg.get("top_p") if _rcfg else None
    _topp_default = _topp_default if _topp_default is not None else 0.7
    _topp = args.top_p if args.top_p is not None else _topp_default
    print(f"  [采样评测] val_n={args.val_n} temperature={_temp} top_p={_topp}（Average@N，"
          f"top_p 来源: {'CLI' if args.top_p is not None else 'run_info' if _rcfg else '默认0.7'}）")
else:
    _temp = 0.0
    _topp = 1.0
if is_retool_family:
    from rlab.rollout import multi_turn_rollout_group
    from rlab.protocol import RETOOL_STOP_KWARGS as _STOP_KW
    # 【2026-09-18 stop 机制】评测与训练同节奏铁律：stop 从训练 config 回读
    # （_rcfg 即 --proto_from 对齐的那份），旧 ckpt（无 retool_stop 键）不带 stop
    # → 评测行为与训练严格一致，新旧协议不混测。
    _stop = dict(_STOP_KW) if _rcfg.get("retool_stop") else {}
    if _sampling:
        # 每条轨迹独立请求 + 独立 seed（与训练同形态；vLLM 同 seed 会生成相同轨迹）
        import random as _rnd
        _base = _rnd.randrange(1 << 30)
        sp_mt = [SamplingParams(temperature=_temp, top_p=_topp,
                                max_tokens=args.round_tokens, seed=_base + k,
                                **_stop)
                 for k in range(len(prompts) * args.val_n)]
    else:
        sp_mt = SamplingParams(temperature=0, max_tokens=args.round_tokens, **_stop)
    mt_cfg = {"max_rounds": args.max_rounds, "sandbox_timeout": 5.0,
              "sandbox_mem_mb": 256, "tool_result_max_chars": 500}
    _probe_prompts = [p for p in prompts for _ in range(args.val_n)] if _sampling else prompts
    _segs, _full, code_stats = multi_turn_rollout_group(
        llm, sp_mt, tokenizer, _probe_prompts, mt_cfg)
    answers = ["".join(s["text"] for s in segs_i if s["kind"] == "assistant")
               for segs_i in _segs]
    # 打分域：retool_math 剥离代码块（与训练端 total_reward_retool_math 同口径）
    answers = [_strip_code_blocks(a) for a in answers]
    code_used = [s["code_used"] for s in code_stats]
    code_ok = [s["code_ok"] for s in code_stats]
else:
    if _sampling:
        outs = llm.generate(prompts, SamplingParams(temperature=_temp, top_p=_topp,
                                                    max_tokens=args.max_tokens,
                                                    n=args.val_n))
        answers = [o.text for out in outs for o in out.outputs]
        code_used = code_ok = [0] * len(answers)
    else:
        outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens))
        answers = [o.outputs[0].text for o in outs]
        code_used = code_ok = [0] * len(answers)

# ---------- 评分 ----------
# 【2026-09-09 审查修复】空答案计入分母记 0 分——旧版 `if len(ans.strip())==0: continue`
# 把"只写代码没写答案/输出为空"的样本剔出分母，模型退化时反而美化 acc。
# 【2026-09-18 采样口径】--val_n>1 时 answers 是 [题][采样] 平铺（每题 val_n 条）：
# 聚合 acc = Average@N（每题 val_n 条平均），per-item 记每题平均 acc/fmt。
acc, fmt, both, n_valid = 0.0, 0.0, 0.0, 0
items = []      # per-item 明细（--dump_items，默认开）：供 analysis.py 配对检验/分层
for i, item in enumerate(sample):
    n_valid += 1
    a_avg = f_avg = 0.0
    for k in range(args.val_n):
        ans = answers[i * args.val_n + k] if _sampling else answers[i]
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
        a_avg += a / args.val_n
        f_avg += f / args.val_n
    acc += a_avg; fmt += f_avg; both += (a_avg == 1.0 and f_avg == 1.0)
    if args.dump_items:
        # 【2026-09-19 修复·采样档索引错位】code_used/code_ok 是 [题][采样] 平铺
        # （长度 n*val_n），旧版直接取 code_used[i] 只覆盖前 200 条轨迹 = 题 0..24
        # 的全部采样，题 25.. 全部缺失 → 代码分层/迁移分析口径全错。
        # 现按题聚合 val_n 条（与 acc/fmt 的 Average@N 同口径）。
        if _sampling:
            _sl = slice(i * args.val_n, (i + 1) * args.val_n)
            _cu = sum(code_used[_sl]) / args.val_n if code_used else 0.0
            _ck = sum(code_ok[_sl]) / args.val_n if code_ok else 0.0
        else:
            _cu = float(code_used[i]) if code_used and i < len(code_used) else 0.0
            _ck = float(code_ok[i]) if code_ok and i < len(code_ok) else 0.0
        items.append({
            # 题面指纹：跨模型对齐用（同 seed/split 下同题同 key）——McNemar 的配对键。
            # run2 缺的正是这个键，导致 +5.0pp 只能做未配对检验（p≈0.11）。
            "qk": hashlib.sha1(str(item["Q"]).encode("utf-8")).hexdigest()[:12],
            "acc": a_avg, "fmt": f_avg,
            # 采样档为每题均值（float）；greedy 档为 0/1 计数（int 语义不变）
            "code_used": _cu, "code_ok": _ck,
            "val_n": args.val_n,
            "ans_len": 0, "empty": 0,
        })
    if i < args.show:
        print(f"  [a={a_avg:.2f} f={f_avg:.2f}]")

result = {"acc": acc / n_valid if n_valid else 0, "fmt": fmt / n_valid if n_valid else 0,
          "both": both / n_valid if n_valid else 0, "n": n_valid,
          # 【2026-09-19】指标口径版本：2 = code_rate/code_ok_rate/avg_rounds 以
          # **轨迹数**（n×val_n）为除数、per-item 的 code_used 按题聚合。
          # 缺此键或 =1 的旧 json 是 p8 事故档（除数=题数 → 采样档虚高 val_n 倍），
          # analysis.py 按本键决定是否做 legacy 回修，绝不靠"看起来像不像率"猜。
          "metrics_version": 2,
          "n_requested": args.n, "n_dropped_long": _dropped_long, "n_dropped_plen": _dropped_plen,
          "algo": args.algo, "eval_task": args.eval_task, "split": args.split,
          # 【2026-09-12 审计缺口补齐】旧版不记 model_path：多模型同表时事后无法核对
          # "这一行评的到底是哪个 checkpoint"（本文件 docstring 自己就在警告同名覆盖）。
          "model_path": args.model,
          "eval_protocol": {"temperature": _temp, "top_p": _topp, "greedy": not _sampling,
                            "val_n": args.val_n, "seed": args.seed,
                            "max_rounds": args.max_rounds, "round_tokens": args.round_tokens,
                            "max_tokens": args.max_tokens, "max_len": args.max_len,
                            # 【2026-09-20】确定性档落盘：采样评测的跨 run 可比性前提。
                            # 缺这两项的旧 json（或 batch_invariant=false）不可与新
                            # json 直接比 Δacc——同权重漂移可达 2pp。
                            "vllm_batch_invariant": bool(_bi),
                            "vllm_attention_backend": _attn_be,
                            # 【2026-09-20】协议出处落盘：False = 回落 preset 默认
                            # （读不到 run_info）。事后核对"这一行测的是哪个协议"
                            # 的唯一凭据——本次 baseA/baseB 事故正是因为回落不留痕。
                            "proto_from_run_info": bool(_run),
                            "proto_src": _proto_src,
                            "max_prompt_length": _max_plen,
                            "system_prompt_sha": _sp_sha}}
if args.dump_items:
    result["items"] = items
if is_retool_family and n_valid:
    # 【2026-09-19 修复·采样档除数】code_used 长度 = n_valid*val_n（每题 val_n 条轨迹）
    # 旧版除以 n_valid 导致 val_n=8 时显示值是真实值的 8 倍（p8: 401.5% 实为 50.2%）
    _denom = len(code_used) if code_used else 1
    result["code_rate"] = sum(1 for u in code_used if u > 0) / _denom
    result["code_ok_rate"] = sum(1 for k in code_ok if k > 0) / _denom
    result["avg_rounds"] = sum(code_used) / _denom
print(f"\n[3/3] {name}（{args.eval_task} {args.split}，N={len(sample)} algo={args.algo}）")
print(f"{name:<16}{result['acc']*100:>9.1f}%{result['fmt']*100:>9.1f}%{result['both']*100:>9.1f}%{result['n']:>10}")
if is_retool_family and n_valid:
    print(f"{name:<16}代码调用率 {result['code_rate']*100:.1f}%  成功率 {result['code_ok_rate']*100:.1f}%  平均轮次 {result['avg_rounds']:.2f}")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({name: result}, f, indent=2, ensure_ascii=False)
print(f"结果已存 {out_path}")
