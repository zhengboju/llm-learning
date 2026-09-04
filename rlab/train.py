# -*- coding: utf-8 -*-
"""rlab/train.py — DeepSpeed 训练端主程序（loss 可插拔）。

用法（2×H20 推荐部署，三进程）：
    # 1. 打分服务器（0 号卡，与 vLLM 共卡）
    python -m rlab.ref_server --model_path /root/Qwen2.5-3B --port 59875 [--mode rfpp]
    # 2. 训练端（单卡 ZeRO-0；如需双卡训练改 deepspeed --num_gpus 2，stage 建议 0/2）
    deepspeed --num_gpus 1 rlab/train.py --algo dapo --model_path /root/Qwen2.5-3B
    #    训练端 rank0 自动 spawn 生成 worker（共驻 0 号卡，gpu_mem 0.35）

必带环境变量（run_gsm8k.sh 已内置）：
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
"""

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlab.config import ds_config, get_config
from rlab.losses import ALGOS, compute_loss, get_per_token_logps
from rlab.protocol import decode_batch


def get_batch(ref_server):
    import requests
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b"empty":
            return None
    except Exception:
        return None
    return decode_batch(r)


def run_training(cfg, args):
    import deepspeed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    deepspeed.init_distributed()

    # rank0 在 spawn 出 gen worker 之后再加载训练模型，避免 fork 时的 CUDA 上下文污染
    gen_proc = None
    Q = None
    if dist.get_rank() == 0:
        print("\n[train] START vLLM generation worker...\n")
        mp.set_start_method("spawn", force=True)
        Q = mp.Queue()
        gen_proc = mp.Process(target=_spawn_gen, args=(Q, cfg), daemon=True)
        gen_proc.start()

    def _ensure_gen_alive():
        """fail-fast：生成端进程死亡 = 权重/数据链路已断，继续等只会空转
        （vLLM 启动 OOM 等故障曾表现为训练端无限 'waiting for batch'）。"""
        if gen_proc is not None and not gen_proc.is_alive():
            raise RuntimeError(
                "[train] 生成端进程已退出（见其 traceback，常见原因：显存不足/"
                "权重同步失败）-> 训练端中止。检查 run_gsm8k.sh 的卡位与显存编排。")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config(cfg), model=model, model_parameters=model.parameters())
    pad_id = tokenizer.pad_token_id

    wandb_run = None
    if cfg["use_wandb"] and dist.get_rank() == 0:
        try:
            import wandb
            wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=False)
            wandb_run = wandb.init(project=cfg["wandb_project"], name=cfg["wandb_name"],
                                   config={k: v for k, v in cfg.items()})
        except Exception as e:
            print(f"[train] wandb 不可用（{e}），继续训练不记录")
    totals = {"num": 0, "acc": 0.0, "fmt": 0.0}

    from tqdm import tqdm
    progress = tqdm(range(1, cfg["all_steps"] + 1)) if dist.get_rank() == 0 \
        else range(1, cfg["all_steps"] + 1)

    for step in progress:
        batch = get_batch(cfg["ref_server"])
        while batch is None:
            _ensure_gen_alive()
            if dist.get_rank() == 0:
                print("waiting for batch...")
            time.sleep(3)
            batch = get_batch(cfg["ref_server"])
        _ensure_gen_alive()

        plen = batch["plen"]
        inputs = batch["inputs"].to(engine.device)
        advantages = batch["advantages"].to(engine.device)
        gen_logps = batch["gen_logps"].to(engine.device)
        ref_logps = batch["refs"].to(engine.device)

        logits = engine(inputs).logits[:, :-1, :]
        per_token_logps = get_per_token_logps(logits, inputs[:, 1:])[:, plen - 1:]
        mask = (inputs[:, plen:] != pad_id).float()

        loss, stats = compute_loss(
            cfg["algo"], per_token_logps, gen_logps, advantages, mask, cfg,
            ref_logps=ref_logps,
            num_items_in_batch=batch.get("num_items_in_batch"))
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
            n = inputs.shape[0]
            totals["num"] += n
            if "acc_scores" in batch:
                totals["acc"] += float((batch["acc_scores"] > 0).sum())
                totals["fmt"] += float((batch["format_scores"] > 0).sum())
            if wandb_run is not None:
                log = {"loss": stats["loss"], "clip_frac": stats["clip_frac"],
                       "approx_kl": stats["approx_kl"], "mean_ratio": stats["mean_ratio"],
                       "acc_correct_ratio": totals["acc"] / totals["num"],
                       "format_correct_ratio": totals["fmt"] / totals["num"]}
                wandb_run.log(log, step=step)

        if step % cfg["gen_update_steps"] == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print("[train] sending latest state_dict ...")
                Q.put(engine.module.state_dict())
                print("[train] send state_dict ok!")
            dist.barrier()

        if step % cfg["save_steps"] == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                save_name = os.path.join(cfg["out_dir"], f"step_{step}")
                os.makedirs(save_name, exist_ok=True)
                sd = engine.module.state_dict()
                sd = type(sd)({k: v.cpu() for k, v in sd.items()})
                engine.module.save_pretrained(save_name, state_dict=sd)
                tokenizer.save_pretrained(save_name)
                print(f"[train] saved -> {save_name}")
            dist.barrier()


def _spawn_gen(Q, cfg):
    """子进程入口：必须走模块顶层可寻址的函数（spawn pickle 约束）。"""
    from rlab.rollout import gen_worker
    gen_worker(Q, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--steps", type=int, default=None, help="覆盖 all_steps")
    ap.add_argument("--save_steps", type=int, default=None)
    ap.add_argument("--gen_update_steps", type=int, default=None)
    ap.add_argument("--num_pre_Q", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gen_device", type=int, default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-log", action="store_true", help="关闭 wandb")
    ap.add_argument("--seed", type=int, default=None,
                    help="固定训练种子（抽题顺序+生成采样），阶段1 起对比实验必带")
    ap.add_argument("--local_rank", type=int, default=0)  # deepspeed 传入
    args = ap.parse_args()

    overrides = {}
    if args.model_path: overrides["model_path"] = args.model_path
    if args.steps: overrides["all_steps"] = args.steps
    if args.save_steps: overrides["save_steps"] = args.save_steps
    if args.gen_update_steps: overrides["gen_update_steps"] = args.gen_update_steps
    if args.num_pre_Q: overrides["num_pre_Q"] = args.num_pre_Q
    if args.out_dir: overrides["out_dir"] = args.out_dir
    if args.gen_device is not None: overrides["gen_device"] = args.gen_device
    if args.port is not None: overrides["ref_server_port"] = args.port
    if args.no_log: overrides["use_wandb"] = False
    if args.seed is not None: overrides["seed"] = args.seed

    cfg = get_config(args.algo, **overrides)
    print("[train] config:", json.dumps(cfg, ensure_ascii=False, indent=2, default=str))
    run_training(cfg, args)


if __name__ == "__main__":
    main()
