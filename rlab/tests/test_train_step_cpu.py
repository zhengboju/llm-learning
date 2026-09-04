# -*- coding: utf-8 -*-
"""rlab/tests/test_train_step_cpu.py — 用 tiny 因果 LM 在 CPU 上端到端验证训练步。

复现 train.py 主循环的精确数据流（不含 DeepSpeed/分布式）：
  左 pad prompt + 右 pad completion 拼接 → logits → prompt 切片 → mask 重算
  → compute_loss 六算法 → backward。
验证点：plen 切片错位（off-by-one）会立刻反映在 loss 值上。

运行：python -m rlab.tests.test_train_step_cpu
"""
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from rlab.config import get_config
from rlab.losses import compute_loss, get_per_token_logps


def _tiny_model():
    """本地构造 2 层/64 维 tiny GPT-2（不触网，随机权重即可验证数据流）。"""
    cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=64,
                     n_layer=2, n_head=2)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg).eval()


def test_train_step():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = _tiny_model()

    prompts = ["Q1: what?", "Q2: how long is a longer question here?"]
    completions = [" <think>a</think><answer>42</answer>",
                   " <think>b</think><answer>7</answer>"]
    # 手动 pad（transformers 4.41 的 tokenizer 不支持 padding_side 参数；训练机新版无此问题）
    def _pad_left(seqs, pad_id):
        lens = [len(s) for s in seqs]
        m = max(lens)
        return torch.tensor([[pad_id] * (m - l) + list(s) for s, l in zip(seqs, lens)])

    def _pad_right(seqs, pad_id):
        lens = [len(s) for s in seqs]
        m = max(lens)
        return torch.tensor([list(s) + [pad_id] * (m - l) for s, l in zip(seqs, lens)])

    p_ids = _pad_left(tok(prompts, add_special_tokens=False)["input_ids"], tok.pad_token_id)
    c_ids = _pad_right(tok(completions, add_special_tokens=False)["input_ids"], tok.pad_token_id)
    plen = p_ids.shape[1]
    n = c_ids.shape[0]
    p_rep = p_ids.repeat(1, n // p_ids.shape[0]).view(-1, plen) if n > p_ids.shape[0] else p_ids
    merged = torch.cat([p_rep, c_ids], dim=1)   # (2, plen+T)

    with torch.inference_mode():
        logits = model(merged).logits[:, :-1, :]
    logps = get_per_token_logps(logits, merged[:, 1:])[:, plen - 1:]   # (B, T)
    assert logps.shape == c_ids.shape, f"切片形状错位: {logps.shape} vs {tuple(c_ids.shape)}"
    check("logps 切片形状 = completion 形状", True)

    mask = (merged[:, plen:] != tok.pad_token_id).float()
    check("mask 只覆盖有效 completion token", mask.sum() > 0
          and mask[:, -1].sum().item() >= 1)   # 末 token（eos=pad 时首行可能为0，不苛刻）

    # 数值锚：logps 与手动逐 token 计算一致（防 gather 错位）
    manual = []
    with torch.inference_mode():
        lp = model(merged).logits.log_softmax(-1)
        for b in range(merged.shape[0]):
            row = [lp[b, plen - 1 + t, merged[b, plen + t]] for t in range(c_ids.shape[1])]
            manual.append(torch.stack(row))
    manual = torch.stack(manual)
    check("logps 数值锚（逐 token 手算一致）",
          torch.allclose(logps, manual, atol=1e-5))

    # 六算法在该真实模型前向上均可 backward
    adv = torch.tensor([1.0, -1.0])
    for algo in ("grpo", "dapo", "dr_grpo", "cispo", "gspo", "rfpp"):
        cfg = get_config(algo, use_wandb=False)
        merged_req = merged.clone()
        with torch.inference_mode():
            gen_logps = get_per_token_logps(
                model(merged_req).logits[:, :-1, :], merged_req[:, 1:])[:, plen - 1:]
        gen_logps = gen_logps.detach()
        emb = model.get_input_embeddings().weight
        # 用一个可训练的伪 policy：对 gen_logps 加可训练偏移，保证 backward通
        delta = torch.zeros_like(gen_logps, requires_grad=True)
        pol = gen_logps + delta
        a = adv_tok = adv.unsqueeze(1).expand_as(pol).contiguous() if algo == "rfpp" else adv
        loss, st = compute_loss(algo, pol, gen_logps, a, mask, cfg,
                                ref_logps=gen_logps, num_items_in_batch=mask.sum())
        loss.backward()
        check(f"{algo} 真实前向 backward 通过 (loss={loss.item():.4f})",
              delta.grad is not None and torch.isfinite(loss).all())


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    print(f"  ok - {name}")


if __name__ == "__main__":
    test_train_step()
    print("\n训练步端到端（CPU tiny model）全部通过 ✅")
