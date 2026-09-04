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
from rlab.protocol import encode_batch, make_bytes_list, tensor_to_bytes
from rlab.reward import total_reward
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
                                     seed=cfg.get("seed"))

    # 可复现种子：抽题顺序(random) + 生成采样(vLLM SamplingParams.seed)。
    # 【2026-09-04 教训】训练运行间方差可达 ±3pp+（dapo 同代码重跑 78.0→74.3），
    # 对比实验必须固定 seed 才能按"更新数配对"做单变量比较。
    seed = cfg.get("seed")
    if seed is not None:
        import random as _random
        _random.seed(seed)
        torch.manual_seed(seed)
        print(f"[rollout] 已固定训练种子 seed={seed}（抽题顺序 + vLLM 采样）")

    QAs = load_qas(cfg["data_task"])
    print(f"[rollout] 数据集 {cfg['data_task']} 共 {len(QAs)} 题")
    ref_server = cfg["ref_server"]

    def try_update_model():
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

    # ------------------------- 采样主循环 -------------------------
    os.makedirs(os.path.dirname(os.path.abspath(cfg["record_path"])), exist_ok=True)
    fout = open(cfg["record_path"], "a", encoding="utf-8")
    uploaded_total = 0
    while True:
        try_update_model()
        # dynamic sampling（DAPO 机制2）：全同组不占配额，继续采直到攒够 Q_batch_size 组
        need = cfg["Q_batch_size"]
        groups = []                      # 每元素: (inputs, prompt_text, prompt_ids, ans_ids, adv, acc, fmt, plen)
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

        for (inputs, prompts_text, prompt_ids, ans_token_ids, adv, acc_s, fmt_s, plen) in groups:
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
                "acc": acc_s.tolist(), "fmt": fmt_s.tolist()}, ensure_ascii=False) + "\n")
        if uploaded_total % 10 == 0:
            fout.flush()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="rlab 生成端独立运行（分进程模式）")
    ap.add_argument("--algo", required=True,
                    choices=("grpo", "dapo", "dr_grpo", "cispo", "gspo", "rfpp"))
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
