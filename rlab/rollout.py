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
from rlab.losses import compute_advantages, get_per_token_logps
from rlab.protocol import (TOOL_END, TOOL_START, encode_batch, extract_python_blocks,
                           make_bytes_list, segment_mask_from_spans, tensor_to_bytes)
from rlab.reward import reward_phase, total_reward, total_reward_retool
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
    """阶段2 ReTool：代码交织多轮生成（一组样本并行走）。

    对每组样本：生成一段 → 检测 python 围栏代码块 → 有则沙箱执行 → 结果按
    TOOL_START/TOOL_END 回填 → 续生成下一轮；本轮无代码块则该样本结束（后续
    应给出最终答案）。最多 cfg['max_rounds'] 轮。

    sampling_params: 单个 SamplingParams（所有请求共用，eval 贪心用）或与
      prompts_text 等长的列表——**训练时必须是列表且每样本 seed 不同**：
      同题 num_pre_Q 条是独立请求，共用 seed 会让 vLLM 生成 n 条完全相同的
      轨迹（组内零方差 → group_ok 永假 → 无限重采）。

    返回 (segs, full_text, code_stats)：
      segs:      list[list[dict]] —— 每个样本一段段的 {"kind": "assistant"|"tool", "text"}
      full_text: list[str] —— 每个样本的完整轨迹文本（prompt+全部段）
      code_stats: list[{"code_used": int, "code_ok": int}]
    """
    n = len(prompts_text)
    segs = [[] for _ in range(n)]
    code_stats = [{"code_used": 0, "code_ok": 0} for _ in range(n)]
    ctxs = list(prompts_text)          # 每轮续写的完整上下文
    active = list(range(n))            # 还在"代码-执行-续写"循环里的样本
    for _rnd in range(int(cfg.get("max_rounds", 3))):
        if not active:
            break
        if isinstance(sampling_params, list):
            sps = [sampling_params[i] for i in active]
        else:
            sps = sampling_params
        outs = vllm_gen.generate([ctxs[i] for i in active], sps, use_tqdm=False)
        new_text = {i: o.outputs[0].text for i, o in zip(active, outs)}
        results = {}
        for i in active:
            segs[i].append({"kind": "assistant", "text": new_text[i]})
            blocks = extract_python_blocks(new_text[i])
            if not blocks:
                continue              # 本轮无代码块 → 样本结束，等待最终答案
            code = blocks[-1]          # 执行最后一个完整代码块（与 Auto_Program 一致）
            code_stats[i]["code_used"] += 1
            res = code_runner(code, timeout=cfg.get("sandbox_timeout", 5.0),
                              mem_mb=cfg.get("sandbox_mem_mb", 256),
                              max_chars=cfg.get("tool_result_max_chars", 500))
            code_stats[i]["code_ok"] += int(res["ok"])
            tool_text = TOOL_START + res["display"] + TOOL_END
            segs[i].append({"kind": "tool", "text": tool_text})
            results[i] = tool_text
        # 下一轮：只有执行过代码的样本续写（扩展上下文）；其余样本就此定格
        next_active = [i for i in active if i in results]
        for i in next_active:
            ctxs[i] = ctxs[i] + new_text[i] + results[i]
        active = next_active

    full_text = [p + "".join(s["text"] for s in segs_i)
                 for p, segs_i in zip(prompts_text, segs)]
    return segs, full_text, code_stats


def retool_score_flat(inputs, asst_texts, code_stats, cfg, steps_elapsed):
    """阶段2 打分（模块级纯函数，CPU 可测）。

    索引契约（2026-09-08 真机 IndexError 教训）：asst_texts/code_stats 必须是
    Q_batch_size × num_pre_Q 条——即每道题先扩成 num_pre_Q 条独立轨迹再进
    multi_turn_rollout_group，idx = i*num_pre_Q + j 一一对应。

    打分文本 = 模型自己的 assistant 段拼接（**不含工具段**，2026-09-08 真机
    fmt=0.0% 教训）：全文含 prompt → 格式正则 ^ 锚定必败 → fmt 恒为常数 →
    组内归一化后 fmt 梯度信号彻底死亡；全文还含沙箱输出 → "最后一个数字"
    变成工具 stdout，把工具结果当模型答案发信用。工具段只作上下文，也绝不
    参与打分——与"工具 token 不进 loss"是同一条原则在奖励端的投影。

    返回 (adv, acc_s, fmt_s, code_used, code_ok, phase)。"""
    phase = reward_phase(steps_elapsed, cfg["reward_switch_step"])
    rewards, acc_s, fmt_s, cu, ck = [], [], [], [], []
    n = cfg["num_pre_Q"]
    assert len(asst_texts) == len(inputs) * n, \
        f"轨迹数 {len(asst_texts)} != 题数{len(inputs)}×num_pre_Q{n}（检查是否漏了扩样）"
    for i, inp in enumerate(inputs):
        for j in range(n):
            idx = i * n + j
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
    vllm_gen = LLM(model=cfg["model_path"], gpu_memory_utilization=0.35)
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
            del state_dict
        except Exception:
            import traceback
            traceback.print_exc()
            raise RuntimeError("[rollout] weight sync failed -> gen worker abort (fail-fast)")

    def compute_gen_logps(merged_ids: torch.Tensor, plen: int) -> torch.Tensor:
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
    def retool_score_group(inputs, full_texts, code_stats):
        """多段轨迹打分（委托模块级 retool_score_flat，索引契约见其 docstring）。"""
        return retool_score_flat(inputs, full_texts, code_stats, cfg,
                                 steps_elapsed=pushes[0] * cfg["gen_update_steps"])

    def retool_build_batch(prompt_ids, segs, plen):
        """由分段轨迹构造 merged_ids + 工具段 mask（assistant=1/tool=0/pad=0）。

        分段 tokenize（add_special_tokens=False）后累计得到 completion 与每样本
        assistant 区间；mask 由区间纯函数给出——工具返回 token 不进 loss 的契约。"""
        per_sample_ids, masks = [], []
        for segs_i in segs:
            ids, spans = [], []
            for seg in segs_i:
                t = tokenizer(seg["text"], add_special_tokens=False)["input_ids"]
                start = len(ids); ids.extend(t); end = len(ids)
                if seg["kind"] == "assistant":
                    spans.append((start, end))
            per_sample_ids.append(ids)
            masks.append(segment_mask_from_spans(len(ids), spans))
        output_ids = pad_sequence([torch.tensor(t) for t in per_sample_ids],
                                  batch_first=True, padding_value=tokenizer.pad_token_id)
        mask = pad_sequence(masks, batch_first=True, padding_value=0.0)
        n = output_ids.shape[0]
        Qrep = prompt_ids.repeat(1, n).view(-1, plen)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        return merged_ids, mask, per_sample_ids

    def collect_retool_group(inputs, prompts_text, prompt_ids, plen):
        """多轮 rollout → 打分 → 上传就绪数据。超长/全同组返回 None（重采）。

        每题扩成 num_pre_Q 条独立轨迹（独立请求 + 独立 seed）——这是组内
        对比的前提，也是 2026-09-08 真机 IndexError（轨迹数<打分索引）的根因。"""
        n = cfg["num_pre_Q"]
        group_prompts = [p for p in prompts_text for _ in range(n)]   # Q*n 条
        seed0 = cfg.get("seed")
        sps = [SamplingParams(n=1, temperature=cfg["temperature"],
                              max_tokens=cfg.get("round_gen_tokens", 280),
                              top_p=cfg["top_p"], top_k=cfg.get("top_k", 50),
                              seed=(seed0 + k if seed0 is not None else None))
               for k in range(len(group_prompts))]
        segs, full_texts, code_stats = multi_turn_rollout_group(
            vllm_gen, sps, tokenizer, group_prompts, cfg)
        # 打分文本 = assistant 段拼接（工具段不参与 acc/fmt，见 retool_score_flat）
        asst_texts = ["".join(s["text"] for s in segs_i if s["kind"] == "assistant")
                      for segs_i in segs]
        merged_ids, mask, per_sample_ids = retool_build_batch(prompt_ids, segs, plen)
        if mask.shape[1] == 0:
            return None
        # 真实上下文 token 数上限检查（防 OOM：超长整组丢弃重采）
        total_toks = int(mask.sum().item()) + plen * mask.shape[0]
        if total_toks > cfg["max_context_tokens"] * mask.shape[0]:
            print(f"[rollout] 轨迹超长 total={total_toks} > "
                  f"{cfg['max_context_tokens']*mask.shape[0]}，整组丢弃重采")
            return None
        completion_lens = [len(t) for t in per_sample_ids]
        adv, acc_s, fmt_s, cu, ck, phase = retool_score_group(
            inputs, asst_texts, code_stats)
        if not group_ok(adv):
            return None
        gen_logps = compute_gen_logps(merged_ids, plen)
        return {"merged": merged_ids, "mask": mask, "gen_logps": gen_logps,
                "adv": adv, "acc": acc_s, "fmt": fmt_s,
                "cu": cu, "ck": ck, "phase": phase,
                "clen": completion_lens}

    # ------------------------- 采样主循环 -------------------------
    os.makedirs(os.path.dirname(os.path.abspath(cfg["record_path"])), exist_ok=True)
    fout = open(cfg["record_path"], "a", encoding="utf-8")
    uploaded_total = 0
    is_retool = cfg["algo"] == "retool"
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
                ready = collect_retool_group(inputs, prompts_text, prompt_ids, plen)
                if ready is None:
                    continue  # 全同组/超长：重采
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
                    "phase": r["phase"]}, ensure_ascii=False) + "\n")
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
        if uploaded_total % 10 == 0:
            fout.flush()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="rlab 生成端独立运行（分进程模式）")
    ap.add_argument("--algo", required=True,
                    choices=("grpo", "dapo", "dr_grpo", "cispo", "gspo", "rfpp", "retool"))
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
