# -*- coding: utf-8 -*-
"""rlab/sft.py — SFT 冷启动训练（docs/08 阶段②，2026-10-09 落地）。

输入是 rlab/sft_data.py 产出的 jsonl（文本层样本）。本文件负责两件**必须**
与 RL 同源的事：

  ① token 化（encode_example）：经 render_chat_ids 增量渲染——
     prompt → (assistant + 结束符 | tool observation 增量)×N，
     与 rollout/build_next_prompt 同一条生产线；每一段重渲染都做前缀校验
     （文本口径，同校验①的容忍级），不过就丢该样本并计数，绝不产出错位序列。
  ② mask：labels 只在 assistant token（含收尾 im_end——"答完就停"也要拿
     梯度，docs/05 §12.3 的 EOS 契约）上等于 input_ids，prompt/工具段/pad
     全 -100。工具 token 进 loss = 教模型预测沙箱输出（docs/08 §3.4 已实锤
     的数值后果），这条没有例外。

训练循环刻意不引 DeepSpeed：SFT 是单次半天级任务，plain torch + bf16 +
AdamW（可选 8bit）+ grad accum 足够；少一层抽象少一类"配置标称≠进程实测"。

用法（pod）：
  python -m rlab.sft --data rlab_out/sft_budget_v1.jsonl \
      --model_path /root/Qwen3.5-4B --out_dir rlab_out/sft_budget_v1 \
      --lr 1e-5 --epochs 2 --optim_8bit
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- 编码 ----

def encode_example(tokenizer, system_prompt: str, question: str, turns: list,
                   chat_template_kwargs: dict | None = None) -> dict:
    """一条 SFT 样本 → (input_ids, labels)（模块级，MockTok/真 tokenizer 可测）。

    序列 = render(system+user+生成头) + Σ assistant(comp+closing差额)
           + Σ tool(observation 增量)。closing 用占位 "x" 定位（与
    build_next_prompt 同一技巧：增量与 assistant 内容无关已实测）。
    每一步重渲染都做**前缀校验**：模板一旦改写历史（think 重构/档不一致），
    抛 ValueError 由调用方丢样计数——宁可丢数据，不训错位序列。
    """
    from rlab.protocol import (encoded_text_tokens, initial_messages,
                               make_call_id, render_chat_ids,
                               suffix_prefix_overlap, tool_message)
    ctkw = chat_template_kwargs

    def _render(msgs, gen):
        return render_chat_ids(tokenizer, msgs, gen, ctkw)

    def _closing(msgs):
        """assistant 段的结束符（占位法，含校验②的同构断言）。"""
        base = _render(msgs, True)
        end_r = _render([*msgs, {"role": "assistant", "content": "x"}], False)
        ph = encoded_text_tokens(tokenizer, "x")
        if end_r[:len(base) + len(ph)] != [*base, *ph]:
            raise ValueError("[sft] 模板无法定位 assistant 结束边界"
                             "（占位校验失败，同 build_next_prompt 校验②）")
        return end_r[len(base) + len(ph):]

    msgs = initial_messages(system_prompt, question)
    ids = _render(msgs, True)
    labels = [-100] * len(ids)
    n_call = 0
    for turn in turns:
        if turn["kind"] == "assistant":
            text = (turn.get("text") or "").strip()
            base = _render(msgs, True)
            if base != ids:
                raise ValueError("[sft] 历史重渲染 != 增量拼接（前缀校验失败）"
                                 f"：{len(base)} vs {len(ids)} token")
            comp = encoded_text_tokens(tokenizer, text)
            closing = _closing(msgs)
            seg = [*comp, *closing[suffix_prefix_overlap(comp, closing):]]
            ids += seg
            labels += list(seg)          # assistant 含收尾 im_end：进 loss
            msgs.append({"role": "assistant", "content": text})
        else:
            n_call += 1
            obs = tool_message(make_call_id(0, 0, n_call),
                               (turn.get("body") or "").strip())
            end_r = _render([*msgs, {"role": "assistant", "content": "x"}], False)
            nxt = _render([*msgs, {"role": "assistant", "content": "x"}, obs],
                          True)
            if nxt[:len(end_r)] != end_r:
                raise ValueError("[sft] 加入 tool observation 后模板改写历史"
                                 "（校验③ 同构）")
            delta = nxt[len(end_r):]
            ids += delta
            labels += [-100] * len(delta)  # 工具段：永不进 loss
            msgs.append(obs)
    return {"input_ids": ids, "labels": labels,
            "n_tokens": len(ids),
            "n_assistant_tokens": sum(1 for x in labels if x != -100)}


# ---------------------------------------------------------------- 损失 ----

def masked_lm_loss_sum(logits: torch.Tensor, labels: torch.Tensor):
    """掩码交叉熵的**未归一化**和（纯函数，CPU 可测；供分块聚合）。

    左移一位（位置 t 预测 t+1）、labels=-100 忽略、logits 升 fp32 算
    log_softmax（bf16 logits 直接 CE 的精度坑）。返回 (loss_sum, n_tok)：
    分块时各块求和再统一除，保证 chunk 边界不影响归一化口径。

    注意：分块调用时每块内部左移会丢掉块边界上的一个预测对（logits[b-1]
    →labels[b]）——T/1024 量级，确定性地少算而不是算错，可接受。"""
    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]
    n = int((shift_labels != -100).sum())
    if n == 0:
        return logits.sum() * 0.0, 0      # 保持计算图连通
    loss_sum = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                               shift_labels.reshape(-1),
                               ignore_index=-100, reduction="sum")
    return loss_sum, n


def chunked_sft_loss(model, input_ids, attention_mask, labels, chunk=1024):
    """backbone 一次前向 + lm_head 分块算 loss（全量 logits 的显存是 T×V×4B，
    4B 模型 T=9216 时 ≈9GB，分块后峰值 ≈ chunk×V×4B ≈1GB）。

    返回 (loss, n_tok)：token 均值口径（掩码后），与该 micro-batch 的
    有效 token 数一起返回供日志聚合成全局 token-mean。"""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    head = model.get_output_embeddings()
    h = decoder(input_ids=input_ids,
                attention_mask=attention_mask).last_hidden_state
    total, n_tok = None, 0
    for s in range(0, input_ids.size(1), chunk):
        ls, n = masked_lm_loss_sum(head(h[:, s:s + chunk, :]),
                                   labels[:, s:s + chunk])
        total = ls if total is None else total + ls
        n_tok += n
    return (total / max(1, n_tok)) if total is not None else None, n_tok


# ---------------------------------------------------------------- 批次 ----

def collate_right_pad(examples, pad_id: int):
    """右 pad 成批（训练口径）：input_ids 补 pad_id，labels 补 -100。"""
    T = max(len(e["input_ids"]) for e in examples)
    B = len(examples)
    input_ids = torch.full((B, T), int(pad_id), dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(e["labels"], dtype=torch.long)
        attn[i, :n] = 1
    return input_ids, labels, attn


# ---------------------------------------------------------------- 主流程 ----

def load_sft_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="SFT 冷启动训练（docs/08 阶段②）")
    ap.add_argument("--data", required=True, help="rlab.sft_data 产出的 jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="4B 全参 SFT 常规区间 1e-5~2e-5；不要照抄 ReTool 论文句里"
                         "的 1e-6（那是 RL 段超参，docs/08 §3.2 归属辨析）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=9216,
                    help="mtj 8192 + prompt 1024；超长样本丢弃并计数（不截断——"
                         "截断轨迹 = 没有完整收尾的负示范，正是要治的形态）")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--loss_chunk", type=int, default=1024)
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="关闭梯度检查点（默认开：T=9216 全层激活约 78GB，"
                         "H20-96G 必 OOM；开后激活降到约 3GB，时长 +30%）")
    ap.add_argument("--optim_8bit", action="store_true",
                    help="bitsandbytes AdamW8bit（与 RL 同一显存策略）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 数据：读 + 提示一致性断言 + 编码 ----
    rows = load_sft_jsonl(args.data)
    assert rows, f"[sft] {args.data} 为空"
    sps = {r.get("system_prompt") for r in rows}
    assert len(sps) == 1, (f"[sft] 数据里混了 {len(sps)} 份系统提示"
                           "（sft_data 单次产出的文件不该出现）")
    sp_sha = rows[0].get("sp_sha")
    print(f"[sft] {len(rows)} 条样本，系统提示 sp{sp_sha}"
          "（与后续 RL 的 -sp 签名必须逐字一致）", flush=True)

    from transformers import AutoTokenizer
    from rlab.model_loading import load_causal_lm
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    ctkw = {"enable_thinking": False}
    t0 = time.time()
    encoded, dropped = [], {"overlong": 0, "encode_err": 0}
    for i, r in enumerate(rows):
        try:
            e = encode_example(tokenizer, r["system_prompt"], r["Q"],
                               r["turns"], ctkw)
        except ValueError as ex:
            dropped["encode_err"] += 1
            if dropped["encode_err"] <= 3:
                print(f"[sft] 样本 {i} 编码失败（丢样）：{ex}", flush=True)
            continue
        if e["n_tokens"] > args.max_seq_len:
            dropped["overlong"] += 1
            continue
        encoded.append(e)
        if (i + 1) % 200 == 0:
            print(f"[sft] 编码 {i + 1}/{len(rows)}", flush=True)
    assert encoded, "[sft] 编码后零样本——检查数据与 tokenizer/模板档位"
    n_tok = sum(e["n_tokens"] for e in encoded)
    n_atok = sum(e["n_assistant_tokens"] for e in encoded)
    print(f"[sft] 编码完成 {time.time() - t0:.0f}s：{len(encoded)} 条可用，"
          f"丢 {dropped}；总 token {n_tok / 1e6:.2f}M，"
          f"assistant 占 {n_atok / max(1, n_tok) * 100:.1f}%", flush=True)

    # ---- 模型 / 优化器 ----
    model = load_causal_lm(args.model_path, dtype=torch.bfloat16,
                           attn_implementation="flash_attention_2")
    model.cuda().train()
    if not args.no_grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if args.optim_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    n_update = math.ceil(
        len(encoded) * args.epochs / (args.micro_bs * args.grad_accum))
    warmup = max(1, int(n_update * args.warmup_ratio))

    def lr_at(update):
        if update < warmup:
            return args.lr * (update + 1) / warmup
        p = (update - warmup) / max(1, n_update - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    # ---- 训练循环 ----
    os.makedirs(args.out_dir, exist_ok=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    step = update = 0
    t1 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(encoded)))
        random.shuffle(order)
        for b0 in range(0, len(order), args.micro_bs):
            batch = [encoded[j] for j in order[b0:b0 + args.micro_bs]]
            ids, labels, attn = collate_right_pad(batch, pad_id)
            loss, nb = chunked_sft_loss(model, ids.cuda(), attn.cuda(),
                                        labels.cuda(), chunk=args.loss_chunk)
            (loss / args.grad_accum).backward()
            step += 1
            if step % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    args.max_grad_norm)
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(update)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % args.log_every == 0:
                    el = time.time() - t1
                    print(f"[sft] ep{ep} update {update}/{n_update} "
                          f"loss {loss.item():.4f} lr {lr_at(update):.2e} "
                          f"gnorm {gn:.2f} tok/s "
                          f"{sum(e['n_tokens'] for e in encoded) * (ep + step * args.micro_bs / len(encoded)) / max(1., el):.0f}",
                          flush=True)
        print(f"[sft] epoch {ep} 完成（update {update}/{n_update}）", flush=True)

    # ---- 存盘（vLLM 可直读的 HF 目录）+ 自证信息 ----
    out = os.path.join(args.out_dir, "final")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    info = {"data": os.path.abspath(args.data), "model_path": args.model_path,
            "sp_sha": sp_sha, "n_examples": len(encoded), "dropped": dropped,
            "n_tokens": n_tok, "n_assistant_tokens": n_atok,
            "args": vars(args), "updates": update,
            "wall_s": round(time.time() - t1, 1)}
    with open(os.path.join(args.out_dir, "sft_info.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[sft] 完成 → {out}（update {update}，{time.time() - t1:.0f}s）",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
