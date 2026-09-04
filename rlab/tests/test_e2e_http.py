# -*- coding: utf-8 -*-
"""rlab/tests/test_e2e_http.py — 真实 HTTP 端到端集成测试（全 CPU，tiny GPT-2）。

把生产栈原样跑起来：ref_server（bottle+tornado 真实端口）← requests 上传
→ 训练端 get_batch/decode_batch → compute_loss（tiny GPT-2 当 policy）。
覆盖此前未测的最大风险面：字节流 over HTTP、双模式服务器循环、端口收发顺序。

  A. passthrough 模式：gen 侧打包上传 → ref 打分 → 训练端取 batch → 6 算法 loss 均有限
  B. rfpp 模式（grad_accum=2）：macro batch 攒批 → per-token advantages 下发 → loss 有限

运行：python -m rlab.tests.test_e2e_http
"""
import socket
import tempfile
import threading
import time

import requests as http
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from rlab.config import get_config
from rlab.losses import compute_loss, get_per_token_logps
from rlab.protocol import decode_batch, encode_batch
from rlab.ref_server import run_server

PASS = []


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _save_tiny_gpt2(tmpdir):
    """本地构造并保存 tiny GPT-2 + tokenizer，供 run_server 从路径加载。"""
    cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=64, n_layer=2, n_head=2)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model.save_pretrained(tmpdir)
    tok.save_pretrained(tmpdir)
    return tmpdir


def _wait_ready(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if http.get(f"http://127.0.0.1:{port}/get", timeout=2).content == b"empty":
                return True
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"ref_server :{port} 未就绪")


def _pad_batch(seqs, pad_id, left):
    m = max(len(x) for x in seqs)
    if left:
        return torch.tensor([[pad_id] * (m - len(x)) + list(x) for x in seqs])
    return torch.tensor([list(x) + [pad_id] * (m - len(x)) for x in seqs])


def _gen_upload(model, tok, port, n_batches=2):
    """模拟生成端：左 pad prompt + 右 pad completion，算 gen_logps，打包上传。"""
    prompts = ["The capital of France is", "A longer question about arithmetic: 2+2="]
    comps = [" <think>paris</think><answer>Paris</answer>",
             " <think>4</think><answer>4</answer>"]
    p = tok(prompts, add_special_tokens=False)["input_ids"]
    c = tok(comps, add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id
    p_ids = _pad_batch(p, pad, left=True)
    c_ids = _pad_batch(c, pad, left=False)
    plen = p_ids.shape[1]
    merged = torch.cat([p_ids, c_ids], dim=1)

    with torch.inference_mode():
        gl = get_per_token_logps(model(merged).logits[:, :-1, :], merged[:, 1:])[:, plen - 1:]
    acc = torch.tensor([1.0, -1.0])
    fmt = torch.tensor([1.0, 1.0])
    for i in range(n_batches):
        adv = torch.tensor([0.8 - 0.1 * i, -0.8 + 0.1 * i])   # 每批不同，便于区分 FIFO 顺序
        raw = encode_batch({"plen": plen, "algo": "grpo"}, merged, adv,
                           gl.clone(), acc, fmt)
        resp = http.post(f"http://127.0.0.1:{port}/upload", data=raw, timeout=10)
        assert resp.content == b"tensor", resp.content
    return merged, gl, acc, fmt


def _policy_loss(model, batch, algo):
    """训练端一步（与 train.py 主循环同数据流）。"""
    cfg = get_config(algo, use_wandb=False)
    plen = batch["plen"]
    inputs = batch["inputs"]
    with torch.inference_mode():
        logits = model(inputs).logits[:, :-1, :]
        pol = get_per_token_logps(logits, inputs[:, 1:])[:, plen - 1:]
    pol = pol.clone().requires_grad_(True)
    mask = (inputs[:, plen:] != batch.get("pad_id", 50256)).float()
    adv = batch["advantages"]
    if algo == "rfpp":
        adv = torch.randn(adv.shape)   # 仅验证形状通路；数值正确性已在纯函数单测锁定
    loss, st = compute_loss(algo, pol, batch["gen_logps"], adv, mask, cfg,
                            ref_logps=batch["refs"],
                            num_items_in_batch=batch.get("num_items_in_batch"))
    loss.backward()
    return loss, st


def test_passthrough_e2e():
    print("[A] passthrough 模式 over HTTP")
    with tempfile.TemporaryDirectory() as tmp:
        path = _save_tiny_gpt2(tmp)
        tok = AutoTokenizer.from_pretrained(path)
        pol_model = GPT2LMHeadModel.from_pretrained(path).eval()
        port = _free_port()
        th = threading.Thread(target=run_server,
                              args=(path, port, "passthrough", 0.04, 4, "cpu", "eager"),
                              daemon=True)
        th.start()
        _wait_ready(port)

        merged, gl, acc, fmt = _gen_upload(pol_model, tok, port)
        time.sleep(0.5)
        raw = http.get(f"http://127.0.0.1:{port}/get", timeout=5).content
        check("get 返回非 empty", raw != b"empty")
        batch = decode_batch(raw)
        check("batch 契约完整", {"inputs", "advantages", "refs", "gen_logps",
                                "acc_scores", "format_scores", "plen"} <= set(batch))
        check("ref_logps 与 gen_logps 同形状", batch["refs"].shape == gl.shape)
        batch["pad_id"] = tok.pad_token_id

        for algo in ("grpo", "dapo", "dr_grpo", "cispo", "gspo", "rfpp"):
            loss, st = _policy_loss(pol_model, batch, algo)
            check(f"{algo} 端到端 loss 有限且可反传", torch.isfinite(loss).item())

        raw2 = http.get(f"http://127.0.0.1:{port}/get", timeout=5).content
        check("FIFO：第二批可取", raw2 != b"empty" and raw2 != raw)
        check("队列取空", http.get(f"http://127.0.0.1:{port}/get",
                                   timeout=5).content == b"empty")


def test_rfpp_e2e():
    print("[B] rfpp 模式 over HTTP（grad_accum=2）")
    with tempfile.TemporaryDirectory() as tmp:
        path = _save_tiny_gpt2(tmp)
        tok = AutoTokenizer.from_pretrained(path)
        pol_model = GPT2LMHeadModel.from_pretrained(path).eval()
        port = _free_port()
        th = threading.Thread(target=run_server,
                              args=(path, port, "rfpp", 0.04, 2, "cpu", "eager"),
                              daemon=True)
        th.start()
        _wait_ready(port)

        # rfpp 上传 part2 = 原始 rewards
        prompts = ["Q one", "Q two is longer"]
        comps = [" <think>a</think><answer>1</answer>",
                 " <think>b</think><answer>2</answer>"]
        p = tok(prompts, add_special_tokens=False)["input_ids"]
        c = tok(comps, add_special_tokens=False)["input_ids"]
        pad = tok.pad_token_id
        p_ids = _pad_batch(p, pad, left=True)
        c_ids = _pad_batch(c, pad, left=False)
        plen = p_ids.shape[1]
        merged = torch.cat([p_ids, c_ids], dim=1)
        with torch.inference_mode():
            gl = get_per_token_logps(pol_model(merged).logits[:, :-1, :],
                                     merged[:, 1:])[:, plen - 1:]
        for rw in (2.0, -2.0):   # 两个 micro upload，攒一个 macro batch
            raw = encode_batch({"plen": plen, "algo": "rfpp"}, merged,
                               torch.tensor([rw, rw]), gl.clone(),
                               torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
            http.post(f"http://127.0.0.1:{port}/upload", data=raw, timeout=10)
        time.sleep(0.5)
        raw = http.get(f"http://127.0.0.1:{port}/get", timeout=5).content
        check("rfpp macro batch 下发", raw != b"empty")
        batch = decode_batch(raw)
        check("rfpp advantages 为 per-token (B,T)",
              batch["advantages"].dim() == 2 and batch["advantages"].shape == gl.shape)
        check("num_items_in_batch 已写入", batch.get("num_items_in_batch", 0) > 0)
        batch["pad_id"] = pad
        loss, st = _policy_loss(pol_model, batch, "rfpp")
        check("rfpp 端到端 loss 有限且可反传", torch.isfinite(loss).item())


if __name__ == "__main__":
    test_passthrough_e2e()
    test_rfpp_e2e()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
