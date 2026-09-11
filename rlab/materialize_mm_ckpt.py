# -*- coding: utf-8 -*-
"""rlab/materialize_mm_ckpt.py — 把微调的纯文本 checkpoint 物化成多模态壳 checkpoint。

【为什么存在】4B 训练产出是纯文本 checkpoint（Qwen3_5TextConfig），而 vLLM 只认
多模态 Qwen3.5（A2 教训：qwen3_5_text 被路由到多模态实现，processor 拿复合
config 类型检查直接 TypeError）。训练端运行时同步用 remap_text_to_multimodal
在内存里现做键名映射绕开；eval 是独立 vLLM 进程，绕不开——需要一份落盘的
多模态格式 checkpoint。

做法：以原始多模态 checkpoint 为骨架（视觉塔/projector 等非语言键原样保留），
把纯文本 checkpoint 的张量按 remap_text_to_multimodal 同一张映射表
（"model.X" -> "model.language_model.X"，tied lm_head 丢弃）替换进对应位置。
映射表与训练端运行时同步严格同源，杜绝两套映射漂移。

自检（fail-fast）：
  1. 映射后每个文本键必须命中骨架键（新键 = 布局漂移，禁止静默写入）；
  2. 骨架中 language_model 键必须被全部覆盖（漏一个 = 该层用回 base 旧权重，
     评测静默失真）；
  3. 非语言键逐键比对保留张量与 base 位级一致（防索引/分片错位）。

用法（pod 上）：
    python -m rlab.materialize_mm_ckpt \
        --text_ckpt ./rlab_out/retool_math/step_200 \
        --mm_base /root/Qwen3.5-4B \
        --out ./rlab_out/retool_math/step_200_mm
    # 之后 eval 用 step_200_mm 参与 vLLM 评测
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from rlab.sync import remap_text_to_multimodal


def _load_safetensors_dir(path: str) -> dict[str, torch.Tensor]:
    """加载目录下全部 safetensors 分片为 {key: tensor}（不依赖 transformers）。"""
    sd: dict[str, torch.Tensor] = {}
    shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"{path} 下没有 .safetensors 文件")
    for f in shards:
        sd.update(load_file(os.path.join(path, f)))
    return sd


def merge_text_into_mm(
    text_sd: dict[str, torch.Tensor],
    mm_base_sd: dict[str, torch.Tensor],
    lm_prefix: str = "model.language_model.",
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """核心合并（纯函数，可 CPU 单测）。

    text_sd: 纯文本 checkpoint（"model.X" / "lm_head.X" 键名）
    mm_base_sd: 原始多模态骨架（"model.language_model.X" + 视觉键）
    返回 (merged_sd, stats)；任何自检不过直接 raise，不产出半成品。
    """
    # 1) 文本张量按运行时同步同一映射表改名（tied lm_head 丢弃）
    remapped = dict(remap_text_to_multimodal(list(text_sd.items())))

    # 2) 骨架分类：language_model 键用文本张量替换，其余（视觉等）原样保留
    merged: dict[str, torch.Tensor] = {}
    stats = {"text_substituted": 0, "base_kept": 0}
    covered = set()
    for key, base_t in mm_base_sd.items():
        if key.startswith(lm_prefix):
            if key not in remapped:
                raise KeyError(
                    f"[materialize] 骨架键 {key!r} 在文本 checkpoint 中无对应"
                    "——布局漂移，禁止用 base 旧权重静默补位")
            t = remapped[key]
            # dtype 对齐骨架（文本端可能是 fp32 master；vLLM 按 config dtype 读）
            merged[key] = t.to(base_t.dtype) if t.dtype != base_t.dtype else t
            covered.add(key)
            stats["text_substituted"] += 1
        else:
            merged[key] = base_t
            stats["base_kept"] += 1

    # 3) 自检：language_model 键全覆盖（漏 = 该层静默用回旧权重）
    missing = sorted(set(remapped) - covered)
    if missing:
        raise KeyError(
            f"[materialize] {len(missing)} 个文本键未命中骨架（前3: {missing[:3]}）"
            "——映射表与骨架不一致，先核对再产出")

    # 4) 自检：非语言键位级一致（分片错位会在这里现形）
    for key in mm_base_sd:
        if not key.startswith(lm_prefix) and merged[key] is not mm_base_sd[key]:
            raise AssertionError(f"[materialize] 非语言键 {key!r} 未按原样保留")

    return merged, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text_ckpt", required=True, help="微调的纯文本 checkpoint 目录")
    ap.add_argument("--mm_base", required=True, help="原始多模态 checkpoint 目录（骨架）")
    ap.add_argument("--out", required=True, help="输出目录（多模态壳 checkpoint）")
    args = ap.parse_args()

    print(f"[materialize] 加载文本 checkpoint: {args.text_ckpt}")
    text_sd = _load_safetensors_dir(args.text_ckpt)
    print(f"[materialize] 加载多模态骨架: {args.mm_base}")
    mm_sd = _load_safetensors_dir(args.mm_base)

    merged, stats = merge_text_into_mm(text_sd, mm_sd)
    print(f"[materialize] 合并完成: 替换 {stats['text_substituted']} 个语言键 / "
          f"保留 {stats['base_kept']} 个非语言键 / 文本端丢弃 tied lm_head")

    os.makedirs(args.out, exist_ok=True)
    # 权重单分片落盘（4B bf16 ~8G）；骨架的 index/旧分片一律不拷贝防冲突
    save_file(merged, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    # 非权重文件（config/tokenizer/processor/chat_template）全从骨架复制——
    # vLLM 的多模态路由与 processor 依赖这些文件，文本端目录里没有
    for f in os.listdir(args.mm_base):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        src = os.path.join(args.mm_base, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, f))
    print(f"[materialize] 已产出 {args.out}（config/processor 继承骨架）")


if __name__ == "__main__":
    main()
