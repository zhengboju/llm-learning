# -*- coding: utf-8 -*-
"""rlab/ref_server.py — 打分中转服务器（带 ref 模型），双模式。

passthrough 模式（默认，GRPO/DAPO/Dr.GRPO/CISPO/GSPO 用）：
  upload: [meta, inputs, rewards/adv, gen_logps, acc, fmt]
  计算 ref per-token logps 后转发，get 返回：
          [meta, inputs, rewards/adv, refs, gen_logps, acc, fmt]

rfpp 模式（RF++ 专用，算法定义的一部分）：
  攒满 macro_step（= grad_accum × gpu_num）个 upload 组成一个 macro batch：
    1. 每条算 ref logps；
    2. per-token reward = (r·eos_mask − β·KL)·completion_mask；
       eos_mask：reward 只放在每条轨迹最后一个有效 token 上（整体 reward → token 级）；
    3. macro batch 内对 per-token reward 标准化；
    4. 反向 cumsum → per-token advantage（序列后段奖励回传前段）；
    5. advantage 再做 macro batch 标准化；num_items_in_batch = 有效 token 数 / macro_step。
  get 返回：[meta(+num_items_in_batch), inputs, rewards, refs, gen_logps, advantages, acc, fmt]

数学部分抽成了纯函数（rfpp_process_macro / passthrough_repack / get_eos_mask /
masked_mean_std），可在 CPU 冒烟测试中直接单测（test_ref_server_cpu.py）。

启动：
    python -m rlab.ref_server --model_path /root/Qwen2.5-3B --port 59875 [--mode rfpp]
"""

import argparse
import json
import queue
import threading

import torch

from rlab.protocol import bytes_list_to_list, bytes_to_tensor, make_bytes_list, tensor_to_bytes


def get_per_token_logps(model, input_ids):
    logits = model(input_ids).logits
    logits = logits[:, :-1, :]
    ids = input_ids[:, 1:]
    out = []
    for row_logits, row_ids in zip(logits, ids):
        log_probs = row_logits.log_softmax(dim=-1)
        out.append(torch.gather(log_probs, dim=1, index=row_ids.unsqueeze(1)).squeeze(1))
    return torch.stack(out)


def get_eos_mask(completion_mask):
    """每条轨迹最后一个有效 token 位置 = 1（RF++ 把序列级 reward 放在这里）。"""
    seq_len = completion_mask.size(1)
    rev = torch.flip(completion_mask, dims=[1])
    last_rev = rev.argmax(dim=1)
    last = seq_len - 1 - last_rev
    eos = torch.zeros_like(completion_mask)
    eos.scatter_(1, last.unsqueeze(1), 1)
    return eos


def masked_mean_std(tensors, masks, eps=1e-5):
    """跨样本（形状可不同）在有效位上的 mean / std（unbiased=False + eps）。"""
    valid = torch.cat([t[m.bool()].flatten() for t, m in zip(tensors, masks)])
    return valid.mean().item(), valid.std(unbiased=False).item() + eps


def passthrough_repack(d, refs):
    """passthrough 单条处理：refs 插到第 3 位，extras 依序保留。
    返回 bytes parts 列表（不含 make_bytes_list 包装）。"""
    parts = [json.dumps(d["base"]).encode(), tensor_to_bytes(d["inputs"]),
             tensor_to_bytes(d["rewards"]), tensor_to_bytes(refs)]
    parts.extend(tensor_to_bytes(t) for t in d["extras"])
    return parts


def rfpp_process_macro(items, beta, pad_id):
    """RF++ macro-batch 数学（纯函数，可单测）。

    items: list of dict(base, inputs, rewards, refs, gen_logps, acc, fmt)
           refs/gen_logps 形状 (B,T)，rewards (B,)，inputs (B, plen+T)
    返回 list of (base_meta, parts) —— parts 为 bytes 列表（含 advantages 与 acc/fmt）。
    """
    per_token_rewards, cmasks = [], []
    for it in items:
        plen = it["base"]["plen"]
        cmask = (it["inputs"][:, plen:] != pad_id).int()
        eos = get_eos_mask(cmask)
        kl = it["gen_logps"] - it["refs"]
        ptr = (it["rewards"].unsqueeze(1).expand(kl.shape) * eos - beta * kl) * cmask
        per_token_rewards.append(ptr)
        cmasks.append(cmask)

    # reward 标准化 -> 反向 cumsum（信用回传）-> advantage 标准化
    r_mean, r_std = masked_mean_std(per_token_rewards, cmasks)
    advs = []
    for ptr, cmask in zip(per_token_rewards, cmasks):
        norm = ((ptr - r_mean) / r_std) * cmask
        advs.append(torch.flip(torch.cumsum(torch.flip(norm, dims=(1,)), dim=1), dims=(1,)))
    a_mean, a_std = masked_mean_std(advs, cmasks)
    valid_num = sum(int(m.sum().item()) for m in cmasks)

    outputs = []
    for it, adv, cmask in zip(items, advs, cmasks):
        it["base"]["num_items_in_batch"] = valid_num / len(items)
        adv = ((adv - a_mean) / a_std) * (adv != 0)   # pad 位归零（原版靠训练端 mask 兜底）
        parts = [json.dumps(it["base"]).encode(), tensor_to_bytes(it["inputs"]),
                 tensor_to_bytes(it["rewards"]), tensor_to_bytes(it["refs"]),
                 tensor_to_bytes(it["gen_logps"]), tensor_to_bytes(adv),
                 tensor_to_bytes(it["acc"]), tensor_to_bytes(it["fmt"])]
        outputs.append((it["base"], parts))
    return outputs


def run_server(model_path, port, mode="passthrough", beta=0.04, grad_accum=4,
               device="cuda", attn_implementation="sdpa"):
    from bottle import Bottle, request
    from bottle import run as bottle_run
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        _attn_implementation=attn_implementation).to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    print(f"[ref_server] mode={mode} device={device} port={port}", flush=True)

    macro_step = grad_accum          # 单 GPU 部署：macro batch = grad_accum 个 micro upload
    raw_queue, result_queue = queue.Queue(), queue.Queue()
    app = Bottle()

    @app.route("/upload", method="POST")
    def do_upload():
        dd = bytes_list_to_list(request.body.read())
        data = {"base": json.loads(dd[0])}
        data["inputs"] = bytes_to_tensor(dd[1])
        data["rewards"] = bytes_to_tensor(dd[2])
        data["extras"] = [bytes_to_tensor(x) for x in dd[3:]]   # gen_logps, acc, fmt...
        raw_queue.put(data)
        print(f"[ref_server] receive {data['inputs'].shape}", flush=True)
        return b"tensor"

    @app.route("/get", method="GET")
    def do_get():
        if result_queue.empty():
            return b"empty"
        return result_queue.get()

    threading.Thread(
        target=lambda: bottle_run(app, host="0.0.0.0", port=port, server="tornado"),
        daemon=True).start()

    if mode == "passthrough":
        while True:
            d = raw_queue.get()
            plen = d["base"]["plen"]
            with torch.inference_mode():
                refs = get_per_token_logps(ref_model, d["inputs"].to(device))
            result_queue.put(make_bytes_list(passthrough_repack(d, refs[:, plen - 1:].cpu())))
    elif mode == "rfpp":
        while True:
            items = []
            for _ in range(macro_step):
                d = raw_queue.get()
                plen = d["base"]["plen"]
                with torch.inference_mode():
                    refs = get_per_token_logps(ref_model, d["inputs"].to(device))
                items.append({"base": d["base"], "inputs": d["inputs"],
                              "rewards": d["rewards"], "refs": refs[:, plen - 1:].cpu(),
                              "gen_logps": d["extras"][0],
                              "acc": d["extras"][1], "fmt": d["extras"][2]})
            for base, parts in rfpp_process_macro(items, beta, tokenizer.pad_token_id):
                result_queue.put(make_bytes_list(parts))
    else:
        raise KeyError(f"未知 mode {mode!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--port", type=int, default=59875)
    ap.add_argument("--mode", default="passthrough", choices=("passthrough", "rfpp"))
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--device", default="cuda", help="cuda（训练机默认）或 cpu（集成测试）")
    args = ap.parse_args()
    run_server(args.model_path, args.port, args.mode, args.beta, args.grad_accum,
               args.device)
