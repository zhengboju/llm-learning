# -*- coding: utf-8 -*-
"""rlab/rollout.py — 生成端 worker（vLLM 采样 + torch gen_logps 副本 + 上传）。

职责（阶段0 范围 = 单轮 GSM8K 式任务；阶段2 多轮工具调用在此文件扩展）：
  1. vLLM 批量采样每题 num_pre_Q 条；
  2. torch 模型副本前向算 gen_logps（vLLM prompt_logprobs 路径在本环境会 hang——实测教训）；
  3. rlab.reward 打分 -> rlab.losses.compute_advantages 归一化；
  4. protocol 打包上传 ref_server；
  5. mp.Queue 收训练端权重，rlab.sync 同步进 vLLM（fail-fast）。

上传契约：
  非 rfpp : [meta, merged_ids, advantages(B,), gen_logps, acc, fmt]
  rfpp    : [meta, merged_ids, raw_rewards(B,), gen_logps, acc, fmt]
            —— advantage 由 ref_server 的 macro-batch 路径计算（eos-mask + 反向
               cumsum 信用分配 + 全局标准化 + num_items），是 RF++ 算法定义的一部分

【工程教训内置】
- 清除 DeepSpeed 分布式环境变量，避免 vLLM 子进程冲突；
- gen_logps 副本与 vLLM 必须同权重（同步顺序：先 vLLM 后 torch 副本）；
- 同步失败 raise，绝不静默吞掉（生成器冻结在 base 的废跑教训）；
- VLLM_ENABLE_V1_MULTIPROCESSING=0 等 env 在本进程内、import vllm 前主动 setdefault。

独立运行（分进程模式）：
    python -m rlab.rollout --algo dapo --gen_device 0
"""

import json
import os
import queue as _queue
import random
import time

# vLLM 相关环境变量必须在实际 import vllm 之前设置（教训：RPC 模式权重同步会 stall）
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # EngineCore 进程内运行，无 RPC 序列化
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from torch.nn.utils.rnn import pad_sequence

from rlab.config import get_config
from rlab.data import load_qas
from rlab.health import HealthMonitor as _HealthMonitor
from rlab.health import weight_fingerprint as _weight_fingerprint
from rlab.losses import compute_advantages, get_per_token_logps
from rlab.protocol import (TOOL_END, TOOL_START, encode_batch, extract_python_blocks,
                           make_bytes_list, segment_mask_from_spans, tensor_to_bytes)
from rlab.reward import (reward_phase, total_reward, total_reward_math,
                           total_reward_retool, total_reward_retool_math)
from rlab.sandbox import run_code
from rlab.sync import sync_weights_into_vllm

# 清除分布式环境变量（gen worker 进程内 vLLM 不允许看到 DeepSpeed 的 WORLD_SIZE 等）
_DEEPSPEED_ENV_KEYS = [
    "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK",
    "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
    "GROUP_WORLD_SIZE", "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_COMM_ID", "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
]


def build_prompt(question: str, system_prompt: str, tokenizer) -> str:
    """单轮 prompt 模板。阶段2 多轮工具调用时替换本函数。"""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)


def group_ok(scores: torch.Tensor) -> bool:
    """组内有区分度才可用于训练（全同组 advantage 恒 0，白占训练配额）。"""
    return (scores.max() - scores.min()).item() >= 1e-4


def multi_turn_rollout_group(vllm_gen, sampling_params, tokenizer, prompts_text, cfg,
                             code_runner=run_code):
    """阶段2 ReTool：代码交织多轮生成（一组样本并行走）——token id 续写版。

    【2026-09-09 修复·生成/训练同序列契约】续写一律走 token id（vLLM
    prompt_token_ids），每段直接采用 vLLM 采样返回的 token_ids：旧版给 vLLM
    整串**文本**续写（内部整串 tokenize），而训练端 retool_build_batch 是分段
    tokenize 拼接——实测 Qwen2.5 tokenizer 下 assistant 段尾 "```" 接工具段头
    "\n" 时整串合并为单 token 13874、分段则是两个 token（典型轨迹 whole=67 vs
    split=68），所有"以代码围栏结尾"的样本（恰是触发工具调用的样本）边界必
    错位 → gen_logps 基线失真、采样分布≠训练序列。token id 续写后生成/训练/
    mask 三方共用同一序列，text 只用于围栏提取/沙箱/打分。

    对每组样本：生成一段 → 检测 python 围栏代码块 → 有则沙箱执行 → 结果按
    TOOL_START/TOOL_END 回填 → 续生成下一轮；本轮无代码块则该样本结束（后续
    应给出最终答案）。最多 cfg['max_rounds'] 轮，其中**只有前 max_rounds-1 轮
    执行代码**：最后一轮即使写了代码也不执行不回填——循环已结束，执行结果
    永远无人消费，只会白烧沙箱并给 code_ok 记无效分（2026-09-09 审查发现3）。

    sampling_params: 单个 SamplingParams（所有请求共用，eval 贪心用）或与
      prompts_text 等长的列表——**训练时必须是列表且每样本 seed 不同**：
      同题 num_pre_Q 条是独立请求，共用 seed 会让 vLLM 生成 n 条完全相同的
      轨迹（组内零方差 → group_ok 永假 → 无限重采）。

    返回 (segs, full_text, code_stats)：
      segs:      list[list[dict]] —— 每个样本一段段的
                 {"kind": "assistant"|"tool", "text": str, "ids": list[int]}
                 （ids = 该段真实 token 序列，assistant 段即 vLLM 采样 token）
      full_text: list[str] —— 每个样本的完整轨迹文本（prompt+全部段，日志用）
      code_stats: list[{"code_used": int, "code_ok": int, "trunc_final": int}]
                  （trunc_final=1 表示末个 assistant 段被轮长上限切断）
    """
    n = len(prompts_text)
    # 每条请求的无 pad prompt token（与批量左 pad prompt_ids 同源：去 pad 即得）
    ctx_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts_text]
    segs = [[] for _ in range(n)]
    code_stats = [{"code_used": 0, "code_ok": 0} for _ in range(n)]
    active = list(range(n))            # 还在"代码-执行-续写"循环里的样本
    n_rounds = int(cfg.get("max_rounds", 3))
    for _rnd in range(n_rounds):
        if not active:
            break
        is_final_round = (_rnd == n_rounds - 1)
        if isinstance(sampling_params, list):
            sps = [sampling_params[i] for i in active]
        else:
            sps = sampling_params
        outs = vllm_gen.generate([{"prompt_token_ids": ctx_ids[i]} for i in active],
                                 sps, use_tqdm=False)
        new_ids_map, results, exec_jobs = {}, {}, []
        for i, o in zip(active, outs):
            new_ids = list(o.outputs[0].token_ids)
            new_text = o.outputs[0].text
            # finish_reason（"length"=被轮长上限切断，"stop"=自然停；FakeGen 等测试
            # 替身无此属性时按 None≈stop 处理——末段截断是 retool 答案被切的直接签名，
            # 训练期不记录就永远看不见（2026-09-09 审查发现6）
            fin = getattr(o.outputs[0], "finish_reason", None)
            segs[i].append({"kind": "assistant", "text": new_text, "ids": new_ids,
                            "finish_reason": fin})
            new_ids_map[i] = new_ids
            blocks = extract_python_blocks(new_text)
            if not blocks:
                continue              # 本轮无代码块 → 样本结束，等待最终答案
            if is_final_round:
                continue              # 最后一轮：不执行代码（结果无人消费，见 docstring）
            code = blocks[-1]          # 执行最后一个完整代码块（最新计算意图；
                                        # Auto_Program 原版取第一个——并非一致，是有意改进）
            code_stats[i]["code_used"] += 1
            exec_jobs.append((i, code))

        # 沙箱并行执行（2026-09-09 提速）：run_code 是 subprocess，线程池并发安全。
        # 旧版逐个串行：每个 subprocess 启动 ~0.1-0.3s，一条 5s 超时的死循环代码
        # 会让同轮其余样本全部干等——4 条并行最多省 ~4x 的沙箱墙钟时间。
        if exec_jobs:
            workers = max(1, int(cfg.get("sandbox_workers", 4)))
            def _run(pair):
                i, code = pair
                return i, code_runner(code, timeout=cfg.get("sandbox_timeout", 5.0),
                                      mem_mb=cfg.get("sandbox_mem_mb", 256),
                                      max_chars=cfg.get("tool_result_max_chars", 500))
            if len(exec_jobs) > 1 and workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(workers, len(exec_jobs))) as ex:
                    done = list(ex.map(_run, exec_jobs))
            else:
                done = [_run(p) for p in exec_jobs]
            for i, res in done:
                code_stats[i]["code_ok"] += int(res["ok"])
                tool_text = TOOL_START + res["display"] + TOOL_END
                tool_ids = tokenizer(tool_text, add_special_tokens=False)["input_ids"]
                segs[i].append({"kind": "tool", "text": tool_text, "ids": tool_ids})
                results[i] = tool_ids
        # 下一轮：只有执行过代码的样本续写（扩展上下文）；其余样本就此定格
        next_active = [i for i in active if i in results]
        for i in next_active:
            ctx_ids[i] = ctx_ids[i] + new_ids_map[i] + results[i]
        active = next_active

    full_text = [p + "".join(s["text"] for s in segs_i)
                 for p, segs_i in zip(prompts_text, segs)]
    # 末段截断标记：最后一个 assistant 段 finish_reason=="length" → final 答案
    # 很可能被 round_gen_tokens 切断（acc 必 -1 的直接原因，训练期可见）
    for i in range(n):
        last_a = next((s for s in reversed(segs[i]) if s["kind"] == "assistant"), None)
        code_stats[i]["trunc_final"] = int(bool(last_a) and last_a.get("finish_reason") == "length")
    return segs, full_text, code_stats


def retool_build_batch(prompt_ids, segs, plen, pad_token_id):
    """由分段轨迹构造 merged_ids + 工具段 mask（assistant=1/tool=0/pad=0）。

    【2026-09-09 修复】直接采用各段生成时记录的 token ids（assistant 段 =
    vLLM 采样 token_ids、工具段 = 分段 tokenize），**不再对段文本重新
    tokenize**——生成/训练必须逐 token 同一条序列：文本往返与段边界 BPE
    合并都会破坏该契约（见 multi_turn_rollout_group docstring）。mask 由
    assistant 区间纯函数给出——工具返回 token 不进 loss 的契约。"""
    per_sample_ids, masks = [], []
    for segs_i in segs:
        ids, spans = [], []
        for seg in segs_i:
            t = list(seg["ids"])
            start = len(ids); ids.extend(t); end = len(ids)
            if seg["kind"] == "assistant":
                spans.append((start, end))
        per_sample_ids.append(ids)
        masks.append(segment_mask_from_spans(len(ids), spans))
    output_ids = pad_sequence([torch.tensor(t) for t in per_sample_ids],
                              batch_first=True, padding_value=pad_token_id)
    mask = pad_sequence(masks, batch_first=True, padding_value=0.0)
    n = output_ids.shape[0]
    Qrep = prompt_ids.repeat(1, n).view(-1, plen)
    merged_ids = torch.cat([Qrep, output_ids], dim=1)
    return merged_ids, mask, per_sample_ids


def retool_context_overlong(per_sample_ids, plen, max_context_tokens):
    """逐样本全长 token 预算检查（模块级纯函数，CPU 可测）。任一样本超限即超。

    【2026-09-09 修复·两级】
    ①全长口径（assistant+工具段一起计）——旧版用 mask.sum() 只数 assistant
      token，工具段（每轮 ≤500 字符 ≈150-200 token，3 轮 ≈600）被漏算：防 OOM
      防线可被超出 ~25%，且与 eval max_len（max_prompt_length+max_context_tokens）
      联动错位——训练放行的最长轨迹会撞 eval 的 vLLM max_model_len。
    ②逐样本口径——旧版按组均值（total > max*b）判定，单条超长样本可被组内短
      样本均摊掩盖而放行；但 padded batch 按【最长样本】计激活显存，均值检查
      挡不住单条尖峰。改为任一样本 len(ids)+plen > max_context_tokens 即整组丢弃。"""
    return any(len(t) + plen > max_context_tokens for t in per_sample_ids)


def retool_score_flat(inputs, asst_texts, code_stats, cfg, steps_elapsed,
                      completion_lens=None):
    """阶段2 打分（模块级纯函数，CPU 可测）。

    索引契约（2026-09-08 真机 IndexError 教训）：asst_texts/code_stats 必须是
    Q_batch_size × num_pre_Q 条——即每道题先扩成 num_pre_Q 条独立轨迹再进
    multi_turn_rollout_group，idx = i*num_pre_Q + j 一一对应。

    打分文本 = 模型自己的 assistant 段拼接（**不含工具段**，2026-09-08 真机
    fmt=0.0% 教训）：全文含 prompt → 格式正则 ^ 锚定必败 → fmt 恒为常数 →
    组内归一化后 fmt 梯度信号彻底死亡；全文还含沙箱输出 → "最后一个数字"
    变成工具 stdout，把工具结果当模型答案发信用。工具段只作上下文，也绝不
    参与打分——与"工具 token 不进 loss"是同一条原则在奖励端的投影。

    completion_lens（可选）：每条轨迹的 token 全长（collect_retool_group 传入），
    供 math 分支的 overlong shaping 使用——不传则 shaping 静默无效
    （2026-09-09 审查发现12：旧版调用点从不传 completion_len，开了
    overlong_shaping 也不生效）。

    返回 (adv, acc_s, fmt_s, code_used, code_ok, phase)。"""
    phase = reward_phase(steps_elapsed, cfg["reward_switch_step"])
    rewards, acc_s, fmt_s, cu, ck = [], [], [], [], []
    n = cfg["num_pre_Q"]
    assert len(asst_texts) == len(inputs) * n, \
        f"轨迹数 {len(asst_texts)} != 题数{len(inputs)}×num_pre_Q{n}（检查是否漏了扩样）"
    is_math = cfg.get("data_task") in ("dapo_math", "dapo-math-17k", "math_dapo")
    for i, inp in enumerate(inputs):
        for j in range(n):
            idx = i * n + j
            if is_math:
                sc = total_reward_retool_math(
                    inp["A"], asst_texts[idx], code_ok=code_stats[idx]["code_ok"],
                    completion_len=(completion_lens[idx] if completion_lens is not None else 0),
                    max_gen_tokens=cfg["max_gen_tokens"],
                    overlong_buffer=cfg["overlong_buffer"],
                    overlong_shaping=cfg.get("overlong_shaping", False))
            else:
                sc = total_reward_retool(
                    inp["A"], asst_texts[idx], code_ok=code_stats[idx]["code_ok"],
                    phase=phase, code_w=cfg["code_w"],
                    cold_w=cfg["reward_cold_w"], hot_w=cfg["reward_hot_w"])
            rewards.append(sc["reward"]); acc_s.append(sc["acc"])
            fmt_s.append(sc["format"]); cu.append(code_stats[idx]["code_used"])
            ck.append(code_stats[idx]["code_ok"])
    rewards = torch.tensor(rewards, dtype=torch.float32)
    adv = compute_advantages(rewards, n, cfg["adv_mode"])
    return (adv, torch.tensor(acc_s), torch.tensor(fmt_s), cu, ck, phase)


def gen_worker(Q, cfg: dict):
    """生成端主入口（由 train.py rank0 spawn，或分进程模式独立运行）。

    Q: mp.Queue，训练端每 gen_update_steps 步 put 一次 state_dict；独立模式传 None。
    """
    for key in _DEEPSPEED_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gen_device"])
    torch.cuda.set_device(0)
    print(f"[rollout] generation worker on GPU {cfg['gen_device']}")

    import requests
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    # 显存占比 0.35→0.45（2026-09-09 提速）：GPU0 = ref(~7G) + vLLM + torch副本(~7G)
    # + gen_logps logits 瞬时峰(~6-12G)，0.45×96=43G 总计 ~70G < 96G，安全。
    # 3B+GQA 的 KV 极小（每条 5k token 才 ~370MB），旧 0.35 的 KV 池大量闲置——
    # 多给 vLLM 显存主要扩大 continuous batching 的调度余量。
    vllm_gen = LLM(model=cfg["model_path"],
                   gpu_memory_utilization=float(cfg.get("gen_gpu_mem", 0.45)))
    # torch 副本：只用它前向算 gen_logps（vLLM prompt_logprobs 路径 hang 的教训）
    gen_torch = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa").cuda().eval()
    print("[rollout] torch gen_logps 副本已加载")

    sampling_params = SamplingParams(n=cfg["num_pre_Q"], temperature=cfg["temperature"],
                                     max_tokens=cfg["max_gen_tokens"], top_p=cfg["top_p"],
                                     top_k=cfg.get("top_k", 50),
                                     seed=cfg.get("seed"))
    # 阶段2 retool：每题先扩成 num_pre_Q 条独立轨迹再进 multi_turn_rollout_group，
    # 每条一个独立请求（n=1）且 seed 各不相同——共用 seed 会让同题各条生成完全
    # 相同的轨迹（组内零方差 → group_ok 永假 → 无限重采）。单段长度上限
    # round_gen_tokens，总上下文由 max_rounds × 单段约束。

    # 可复现种子：抽题顺序(random) + 生成采样(vLLM SamplingParams.seed)。
    # 对比实验固定 seed 后可按"更新数配对"做单变量比较（同 seed 下 vLLM 采样可复现）。
    seed = cfg.get("seed")
    if seed is not None:
        import random as _random
        _random.seed(seed)
        torch.manual_seed(seed)
        print(f"[rollout] 已固定训练种子 seed={seed}（抽题顺序 + vLLM 采样）")

    QAs = load_qas(cfg["data_task"])
    print(f"[rollout] 数据集 {cfg['data_task']} 共 {len(QAs)} 题")
    ref_server = cfg["ref_server"]
    pushes = [0]   # 权重推送次数（每 gen_update_steps 优化步一次；近似 optimizer step）
    last_fp = [None]   # 上次推送的权重指纹（两次相同 = 训练端权重没在变）
    health = _HealthMonitor()

    def try_update_model():
        nonlocal pushes
        if Q is None:
            return
        try:
            state_dict = Q.get_nowait()
        except _queue.Empty:
            return
        print("[rollout] recving new model ...")
        try:
            # 顺序强制：先 vLLM 后 torch 副本，两者必须保持同一份权重
            path = sync_weights_into_vllm(vllm_gen, state_dict)
            gen_torch.load_state_dict(
                {k: v.to(torch.bfloat16) for k, v in state_dict.items()})
            print(f"[rollout] model updated via {path}, {len(state_dict)} tensors")
            pushes[0] += 1            # 权重推送计数（用于冷启动/后期奖励切换）
            # 权重指纹（float64，位级敏感）：两次推送指纹完全相同 = 训练端权重
            # 位级未变（优化器未步进/更新全被 bf16 舍入吞掉）。float32 求和会在
            # 3e8 元素上分辨率 ~0.5，淹没 bf16 单权重翻转 ~1e-4 → 假阳性（教训）。
            fp = _weight_fingerprint(state_dict)
            if fp == last_fp[0]:
                print("[健康检查] 本次推送权重指纹（float64）与上次完全相同 → 训练端"
                      "权重位级未变；若连续 2+ 次推送均如此，判定权重冻结，停止排查"
                      "训练端优化器路径", flush=True)
            last_fp[0] = fp
            del state_dict
        except Exception:
            import traceback
            traceback.print_exc()
            raise RuntimeError("[rollout] weight sync failed -> gen worker abort (fail-fast)")

    def compute_gen_logps(merged_ids: torch.Tensor, plen: int) -> torch.Tensor:
        # 已知妥协（阶段0 遗留，如实记录）：前向不传 attention_mask，左 pad 区
        # token 参与 attention——但训练端 policy 前向与 ref_server 前向同样不传，
        # 三方一致的偏差在 ratio（policy/gen）中抵消；教学规模实测可用。
        with torch.inference_mode():
            mids = merged_ids.to(gen_torch.device)
            logits = gen_torch(mids).logits
            return get_per_token_logps(logits[:, :-1, :], mids[:, 1:])[:, plen - 1:].cpu()

    def score_group(inputs, answers, completion_lens):
        """打分。返回 (scores, acc_s, fmt_s)。

        scores = advantage（非 rfpp，已组内归一化）或原始 reward（rfpp）。
        """
        rewards, acc_s, fmt_s = [], [], []
        n = cfg["num_pre_Q"]
        for i, inp in enumerate(inputs):
            for j, a in enumerate(answers[i * n:(i + 1) * n]):
                if cfg.get("data_task") in ("dapo_math", "dapo-math-17k", "math_dapo"):
                    sc = total_reward_math(inp["A"], a,
                                           completion_len=completion_lens[i * n + j],
                                           max_gen_tokens=cfg["max_gen_tokens"],
                                           overlong_buffer=cfg["overlong_buffer"],
                                           overlong_shaping=cfg["overlong_shaping"])
                else:
                    sc = total_reward(inp["A"], a, w_acc=2.0,
                                      completion_len=completion_lens[i * n + j],
                                      max_gen_tokens=cfg["max_gen_tokens"],
                                      overlong_buffer=cfg["overlong_buffer"],
                                      overlong_shaping=cfg["overlong_shaping"])
                rewards.append(sc["reward"]); acc_s.append(sc["acc"]); fmt_s.append(sc["format"])
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if cfg["algo"] == "rfpp":
            return rewards, torch.tensor(acc_s), torch.tensor(fmt_s)  # 原始分直传，服务端算 advantage
        adv = compute_advantages(rewards, cfg["num_pre_Q"], cfg["adv_mode"])
        return adv, torch.tensor(acc_s), torch.tensor(fmt_s)

    # ------------------- 阶段2 retool 专用：打分 / 打包 -------------------
    def retool_score_group(inputs, full_texts, code_stats, completion_lens=None):
        """多段轨迹打分（委托模块级 retool_score_flat，索引契约见其 docstring）。"""
        return retool_score_flat(inputs, full_texts, code_stats, cfg,
                                 steps_elapsed=pushes[0] * cfg["gen_update_steps"],
                                 completion_lens=completion_lens)

    def collect_retool_group(inputs, prompts_text, prompt_ids, plen, seed_salt=0):
        """多轮 rollout → 打分 → 上传就绪数据。超长/全同组返回 None（重采）。

        每题扩成 num_pre_Q 条独立轨迹（独立请求 + 独立 seed）——这是组内
        对比的前提，也是 2026-09-08 真机 IndexError（轨迹数<打分索引）的根因。
        seed_salt：随重采次数递增的盐——旧版丢组重采后复用同一批 seed，同题
        再现时会生成完全相同的轨迹、必然再被丢（2026-09-09 审查修复）。"""
        n = cfg["num_pre_Q"]
        group_prompts = [p for p in prompts_text for _ in range(n)]   # Q*n 条
        seed0 = cfg.get("seed")
        sps = [SamplingParams(n=1, temperature=cfg["temperature"],
                              max_tokens=cfg.get("round_gen_tokens", 400),
                              top_p=cfg["top_p"], top_k=cfg.get("top_k", 50),
                              seed=(seed0 + seed_salt + k if seed0 is not None else None))
               for k in range(len(group_prompts))]
        segs, full_texts, code_stats = multi_turn_rollout_group(
            vllm_gen, sps, tokenizer, group_prompts, cfg)
        # 打分文本 = assistant 段拼接（工具段不参与 acc/fmt，见 retool_score_flat）
        asst_texts = ["".join(s["text"] for s in segs_i if s["kind"] == "assistant")
                      for segs_i in segs]
        merged_ids, mask, per_sample_ids = retool_build_batch(
            prompt_ids, segs, plen, tokenizer.pad_token_id)
        if mask.shape[1] == 0:
            return None
        completion_lens = [len(t) for t in per_sample_ids]
        # 真实上下文 token 数上限检查（防 OOM：逐样本全长口径，任一超限整组
        # 丢弃重采——2026-09-09 审查修复，旧版组均值口径会放行单条超长尖峰）
        if retool_context_overlong(per_sample_ids, plen, cfg["max_context_tokens"]):
            print(f"[rollout] 轨迹超长（逐样本全长口径，max={cfg['max_context_tokens']}/条），"
                  "整组丢弃重采")
            return None
        adv, acc_s, fmt_s, cu, ck, phase = retool_score_group(
            inputs, asst_texts, code_stats, completion_lens=completion_lens)
        if not group_ok(adv):
            return None
        gen_logps = compute_gen_logps(merged_ids, plen)
        return {"merged": merged_ids, "mask": mask, "gen_logps": gen_logps,
                "adv": adv, "acc": acc_s, "fmt": fmt_s,
                "cu": cu, "ck": ck, "phase": phase, "clen": completion_lens,
                "trunc": [int(s["trunc_final"]) for s in code_stats]}

    # ------------------------- 采样主循环 -------------------------
    os.makedirs(os.path.dirname(os.path.abspath(cfg["record_path"])), exist_ok=True)
    fout = open(cfg["record_path"], "a", encoding="utf-8")
    uploaded_total = 0
    is_retool = cfg["algo"] in ("retool", "retool_math")
    rollout_seq = [0]   # 全局递增的 rollout 计数（丢组重采的 seed 盐，防同 seed 复采）
    samp_stats = {"attempts": 0, "dropped": 0}   # 丢弃重采可见性（丢弃率是速度杀手）
    while True:
        try_update_model()
        # dynamic sampling（DAPO 机制2）：全同组不占配额，继续采直到攒够 Q_batch_size 组
        need = cfg["Q_batch_size"]
        groups = []   # 单轮: (inputs, prompt_text, prompt_ids, ans_ids, adv, acc, fmt, plen)
                      # retool: (plen, prompt_ids, ready_dict)
        attempts = 0
        max_attempts = need * cfg["dynamic_max_attempts_mult"]
        while len(groups) < need and attempts < max_attempts:
            attempts += 1
            inputs = random.sample(QAs, cfg["Q_batch_size"])
            prompts_text = [build_prompt(x["Q"], cfg["system_prompt"], tokenizer) for x in inputs]
            prompt_ids = tokenizer(prompts_text, return_tensors="pt", padding=True,
                                   padding_side="left", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            if plen > cfg["max_prompt_length"]:
                continue
            if is_retool:
                # 阶段2：多轮代码交织 → 打分 → 上传就绪（mask 已按段边界算好）
                ready = collect_retool_group(inputs, prompts_text, prompt_ids, plen,
                                             seed_salt=rollout_seq[0])
                rollout_seq[0] += 1
                if ready is None:
                    continue  # 全同组/超长：重采（seed 盐已递增，不会复采同轨迹）
                groups.append((plen, prompt_ids, ready))
                continue
            voutputs = vllm_gen.generate(prompts_text, sampling_params, use_tqdm=False)
            answers, ans_token_ids = [], []
            for v in voutputs:
                for z in v.outputs:
                    answers.append(z.text); ans_token_ids.append(list(z.token_ids))
            completion_lens = [len(t) for t in ans_token_ids]

            adv, acc_s, fmt_s = score_group(inputs, answers, completion_lens)
            if not group_ok(adv):
                continue  # 全同组：重采（对 rfpp 即原始 reward 全同；语义一致）

            groups.append((inputs, prompts_text, prompt_ids, ans_token_ids, adv, acc_s, fmt_s, plen))

        # 丢弃重采可见性：outcome-only/难题下"全错组整组白跑"是最大的隐形时间黑洞，
        # 丢弃率高就该调 num_pre_Q/难度分布，而不是让它悄悄吃掉一半 GPU 时间
        samp_stats["attempts"] += attempts
        samp_stats["dropped"] += attempts - len(groups)
        for g in groups:
            if is_retool:
                plen, prompt_ids, r = g
                meta = {"plen": plen, "algo": cfg["algo"], "has_mask": 1}
                xdata = encode_batch(meta, r["merged"], r["adv"], r["gen_logps"],
                                     r["mask"], r["acc"], r["fmt"])
                requests.post(f"{ref_server}/upload", data=xdata)
                uploaded_total += 1
                fout.write(json.dumps({
                    "t": time.time(), "algo": cfg["algo"],
                    "acc": r["acc"].tolist(), "fmt": r["fmt"].tolist(),
                    "clen": r["clen"], "code_used": r["cu"], "code_ok": r["ck"],
                    "trunc_final": r["trunc"],
                    "phase": r["phase"]}, ensure_ascii=False) + "\n")
                health.observe(r["acc"].tolist(), r["fmt"].tolist(), r["clen"], r["cu"],
                               r["trunc"])
                continue
            inputs, prompts_text, prompt_ids, ans_token_ids, adv, acc_s, fmt_s, plen = g
            tensor_list = [torch.tensor(t) for t in ans_token_ids]
            output_ids = pad_sequence(tensor_list, batch_first=True,
                                      padding_value=tokenizer.pad_token_id)
            n = output_ids.shape[0]
            Qrep = prompt_ids.repeat(1, n).view(-1, plen)
            merged_ids = torch.cat([Qrep, output_ids], dim=1)
            gen_logps = compute_gen_logps(merged_ids, plen)

            meta = {"plen": plen, "algo": cfg["algo"]}
            xdata = encode_batch(meta, merged_ids, adv, gen_logps, acc_s, fmt_s)
            requests.post(f"{ref_server}/upload", data=xdata)
            uploaded_total += 1
            fout.write(json.dumps({
                "t": time.time(), "algo": cfg["algo"],
                "acc": acc_s.tolist(), "fmt": fmt_s.tolist(),
                "clen": [len(t) for t in ans_token_ids]}, ensure_ascii=False) + "\n")
            health.observe(acc_s.tolist(), fmt_s.tolist(),
                           [len(t) for t in ans_token_ids])
        if uploaded_total % 10 == 0:
            fout.flush()
        if uploaded_total and uploaded_total % 16 == 0:
            _a, _d = samp_stats["attempts"], samp_stats["dropped"]
            print(f"[rollout] 采样统计: 累计尝试 {_a} 次 / 有效上传 {uploaded_total} 组"
                  f"（丢弃率 {_d / max(1, _a) * 100:.0f}%，全同组/超长整组重采）", flush=True)
        # 训练期健康检查：窗口签名告警（fmt 恒定/没有学习/退化/截断/代码信号缺失）
        # retool 家族的 clen 上限按"轮数×每轮预算"计——旧版用
        # max_context_tokens-max_prompt_length（8192-1024=7168），而轨迹实际上限
        # ≈3×1024+工具段，永远摸不到 0.95×上限，trunc 签名形同虚设
        # （2026-09-09 审查发现6；真正的截断签名另见 trunc_final/retool_trunc）
        health.maybe_check(
            retool=is_retool,
            max_clen=(cfg["max_rounds"] * cfg.get("round_gen_tokens", 400))
            if is_retool else cfg["max_gen_tokens"])


def main():
    import argparse
    ap = argparse.ArgumentParser(description="rlab 生成端独立运行（分进程模式）")
    ap.add_argument("--algo", required=True,
                    choices=("grpo", "dapo", "dr_grpo", "cispo", "gspo", "rfpp", "retool", "retool_math"))
    ap.add_argument("--gen_device", type=int, default=0)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--port", type=int, default=59875)
    args = ap.parse_args()
    cfg = get_config(args.algo, gen_device=args.gen_device, ref_server_port=args.port)
    if args.model_path:
        cfg["model_path"] = args.model_path
    gen_worker(None, cfg)


if __name__ == "__main__":
    main()
