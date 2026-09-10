# -*- coding: utf-8 -*-
"""rlab/extract_text_model.py — 从多模态 Qwen3.5 checkpoint 抽取纯文本模型。

背景（2026-09-11）：Qwen3.5-4B 官方 checkpoint 是原生多模态
（architectures=Qwen3_5ForConditionalGeneration，config 顶层为 text_config+
vision_config 复合体，vocab_size 在 text_config 里）。rlab 全链路按纯文本
causal LM 设计（与 Qwen2.5 同款）：AutoModelForCausalLM 对复合 config 会拿到
ForCausalLM 类但喂进复合 config → Qwen3_5Config.vocab_size MISSING 直接崩。

不选"让 rlab 适配多模态包装"的原因：权重名多一层 language_model 前缀会打崩
sync.py 的 vLLM 权重同步；vision tower 权重白占 DeepSpeed 优化器显存；ref 模型/
gen_logps 副本/训练端三处都要改。一次性抽取纯文本 checkpoint 是总成本最低解。

用法（pod，CPU 即可，内存 ~20G，约 10 分钟）：
    python -m rlab.extract_text_model --src /root/Qwen3.5-4B --dst /root/Qwen3.5-4B-text

产出 /dst（纯文本 Qwen3_5ForCausalLM + tokenizer + chat template）。之后探针/
训练/评测的 model_path 一律用 --dst；脚本内置 logits 对拍自检（全模型 vs 抽取
模型同输入 logits allclose），不通过会显式报错而不是静默产出坏 checkpoint。
"""

import argparse
import os

import torch


def _find_text_backbone(full):
    """定位多模态包装内的纯文本主干（transformers 5.x: model.language_model）。"""
    m = full.model
    for name in ("language_model", "text_model", "model"):
        sub = getattr(m, name, None)
        if sub is not None and hasattr(sub, "layers"):
            print(f"[extract] 文本主干: full.model.{name} ({type(sub).__name__})")
            return sub
    raise RuntimeError(
        f"未找到文本主干（model 下属性: {list(dict(full.model.named_children()))}）")


def main():
    ap = argparse.ArgumentParser(description="多模态 Qwen3.5 -> 纯文本 checkpoint")
    ap.add_argument("--src", required=True, help="多模态 checkpoint 路径")
    ap.add_argument("--dst", required=True, help="输出纯文本 checkpoint 路径")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5 import (Qwen3_5ForCausalLM,
                                             Qwen3_5ForConditionalGeneration)

    cfg = AutoConfig.from_pretrained(args.src)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        raise RuntimeError("config 无 text_config——不是多模态复合体，无需抽取")
    print(f"[extract] text_config: vocab_size={text_cfg.vocab_size} "
          f"layers={text_cfg.num_hidden_layers} tie={text_cfg.tie_word_embeddings}")

    print("[extract] 加载多模态全模型（内存峰值 ~2x 模型大小）...")
    full = Qwen3_5ForConditionalGeneration.from_pretrained(args.src, dtype=dtype)
    causal = Qwen3_5ForCausalLM(text_cfg).to(dtype)

    backbone = _find_text_backbone(full)
    missing, unexpected = causal.model.load_state_dict(
        backbone.state_dict(), strict=True)
    print(f"[extract] 主干权重复制完成 missing={missing} unexpected={unexpected}")
    if not text_cfg.tie_word_embeddings:
        causal.lm_head.load_state_dict(full.lm_head.state_dict(), strict=True)
    else:
        causal.tie_weights()   # lm_head 与 embedding 共享，绑一下即可

    # ---- 自检：同输入 logits 对拍（bf16 噪声内一致才产出）----
    print("[extract] logits 对拍自检...")
    tok = AutoTokenizer.from_pretrained(args.src)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        lg_full = full(input_ids=ids).logits[0, -1].float()
        lg_causal = causal(input_ids=ids).logits[0, -1].float()
    max_diff = (lg_full - lg_causal).abs().max().item()
    same_top = int(lg_full.argmax()) == int(lg_causal.argmax())
    print(f"[extract] max|Δlogits|={max_diff:.4f} argmax 一致={same_top} "
          f"(top={tok.decode(int(lg_causal.argmax()))!r})")
    if max_diff > 0.1 or not same_top:
        raise RuntimeError(f"对拍不通过（max_diff={max_diff}, same_top={same_top}）"
                           "——抽取有误，不产出 checkpoint")

    # ---- 落盘：模型 + tokenizer（含 chat template，enable_thinking 靠它）----
    print(f"[extract] 保存 -> {args.dst}")
    causal.generation_config = full.generation_config   # eos 列表等停止语义要带走
    causal.save_pretrained(args.dst)
    AutoTokenizer.from_pretrained(args.src).save_pretrained(args.dst)
    # README 等杂项不拷贝；确认产物
    print("[extract] 产物:", sorted(os.listdir(args.dst)))
    print("[extract] 完成。后续 model_path 一律用", args.dst)


if __name__ == "__main__":
    main()
