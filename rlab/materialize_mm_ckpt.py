# -*- coding: utf-8 -*-
"""rlab/materialize_mm_ckpt.py — 把微调的纯文本 checkpoint 物化成多模态壳 checkpoint。

【2026-09-16 起：新 ckpt 不再需要这一步】训练端存盘已直接产出多模态壳
（`train.save_checkpoint` → `write_mm_checkpoint`，产物 vLLM/torch 都直读）——
"训练的输入"与"评测的输入"终于是同一份格式。本模块保留两件用途：
  ① **旧 ckpt 补格式**：09-16 之前存下的 `step_N` 仍是纯文本（Qwen3_5TextConfig）；
  ② **eval 自动兜底**：单独评一个纯文本目录时（如 `--model .../Qwen3.5-4B-text`），
     进程内调 `materialize_mm_checkpoint` 物化到临时目录再起 vLLM，免手工。

【为什么它曾经必须存在】4B 训练产出是纯文本 checkpoint（Qwen3_5TextConfig），而 vLLM
只认多模态 Qwen3.5（A2 教训：qwen3_5_text 被路由到多模态实现，processor 拿复合
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

用法（pod 上，仅旧 ckpt 需要）：
    python -m rlab.materialize_mm_ckpt \
        --text_ckpt ./rlab_out/retool_math/step_200 \
        --mm_base /root/Qwen3.5-4B \
        --out ./rlab_out/retool_math/step_200_mm
    # 之后 eval 用 step_200_mm 参与 vLLM 评测（新 ckpt 直接指 step_200 即可）
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


LM_PREFIX = "model.language_model."


def read_mm_key_index(mm_base: str) -> set:
    """只读 safetensors **头部**拿骨架的全部键名（不加载任何张量，内存零代价）。

    用途：骨架模式（只留非语言键）下仍要保住"文本键必须命中骨架"这条自检——
    否则多出来的键（布局漂移/多余 buffer）会被原样写进产物，vLLM 加载时才发现。
    """
    from safetensors import safe_open

    keys: set[str] = set()
    shards = sorted(f for f in os.listdir(mm_base) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"{mm_base} 下没有 .safetensors 文件")
    for f in shards:
        with safe_open(os.path.join(mm_base, f), framework="pt") as fh:
            keys.update(fh.keys())
    return keys


def load_mm_skeleton(mm_base: str, lm_prefix: str = LM_PREFIX) -> dict:
    """流式读多模态骨架，只保留**非语言键**（视觉塔 / projector 等）。

    【为什么流式、为什么不复用 merge_text_into_mm】base 的 ~8G 语言权重对"存盘"
    毫无用处：训练进程的 CPU 内存已被 ZeRO offload 占去大半（本机上限 60G），
    把整模型读进来是自找 OOM。逐分片读、读完即弃，峰值只有单个分片，驻留 ~1G。
    """
    out: dict[str, torch.Tensor] = {}
    shards = sorted(f for f in os.listdir(mm_base) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"{mm_base} 下没有 .safetensors 文件")
    for f in shards:
        sd = load_file(os.path.join(mm_base, f))
        for k, v in sd.items():
            if not k.startswith(lm_prefix):
                out[k] = v
        del sd
    return out


def merge_text_into_skeleton(
    text_sd: dict[str, torch.Tensor],
    skeleton_sd: dict[str, torch.Tensor],
    lm_prefix: str = LM_PREFIX,
    dtype=None,
    base_keys=None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """把文本 state_dict 盖到**已剔语言键的骨架**上（纯函数，可 CPU 单测）。

    与 `merge_text_into_mm` 的分工：那个吃完整 base（含语言键），能做"骨架语言键
    全覆盖"自检，代价是整模型进内存；这个吃 `load_mm_skeleton` 的产物，省内存，
    那条自检由 `base_keys`（`read_mm_key_index` 只读头部拿到）补回来。
    """
    remapped = dict(remap_text_to_multimodal(list(text_sd.items())))
    bad = sorted(k for k in remapped if not k.startswith(lm_prefix))
    if bad:
        raise KeyError(
            f"[materialize] 映射后出现非语言键 {bad[:3]}——映射表与骨架前缀不一致，"
            "先核对再产出")
    if base_keys is not None:
        unknown = sorted(set(remapped) - set(base_keys))
        if unknown:
            raise KeyError(
                f"[materialize] {len(unknown)} 个文本键在骨架里没有对应（前3: "
                f"{unknown[:3]}）——布局漂移，禁止写出 vLLM 不认识的键"
                "（旧版靠『骨架键全覆盖』自检拦，骨架模式用只读头部补回这条判据）")
    if dtype is not None:
        remapped = {k: (t.to(dtype) if t.dtype != dtype else t)
                    for k, t in remapped.items()}
    merged = dict(skeleton_sd)
    merged.update(remapped)
    return merged, {"text_substituted": len(remapped), "base_kept": len(skeleton_sd)}


def _write_state_dict_and_files(merged: dict, mm_base: str, out: str) -> None:
    """落盘权重 + 从骨架复制全部非权重文件（config/processor/tokenizer/chat_template）。

    非权重文件必须来自骨架：vLLM 的多模态路由与 processor 依赖它们，纯文本目录里
    没有。先清掉 out 里已有的 safetensors/index——防"先文本后多模态"的重复存盘在
    同一目录里留下残片，被加载器 glob 到（同名重跑护栏放行的正是同签名目录）。
    """
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out, f))
    save_file(merged, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(mm_base):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        src = os.path.join(mm_base, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, f))


def write_mm_checkpoint(state_dict: dict, mm_base: str, out: str, *,
                        skeleton: dict | None = None, dtype=None,
                        base_keys: set | None = None) -> dict:
    """把（文本布局）state_dict 落成 vLLM 可直读的多模态壳目录。返回 stats。

    skeleton 传 `load_mm_skeleton` 的缓存可省掉每次存盘重读 base；None 则现读。
    base_keys 同理可用 `read_mm_key_index` 缓存（只读头部，代价可忽略，默认现读）。
    """
    skel = skeleton if skeleton is not None else load_mm_skeleton(mm_base)
    keys = base_keys if base_keys is not None else read_mm_key_index(mm_base)
    merged, stats = merge_text_into_skeleton(state_dict, skel, dtype=dtype,
                                             base_keys=keys)
    _write_state_dict_and_files(merged, mm_base, out)
    return stats


def materialize_mm_checkpoint(text_ckpt: str, mm_base: str, out: str) -> dict:
    """目录级物化：纯文本 ckpt 目录 + 骨架目录 -> vLLM 可直读目录。返回 stats。

    `main()`（离线 CLI）与 eval 的自动兜底共用这一条路径。
    """
    print(f"[materialize] 加载文本 checkpoint: {text_ckpt}")
    text_sd = _load_safetensors_dir(text_ckpt)
    print(f"[materialize] 读多模态骨架（非语言键）: {mm_base}")
    return write_mm_checkpoint(text_sd, mm_base, out)


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

    _write_state_dict_and_files(merged, args.mm_base, args.out)
    print(f"[materialize] 已产出 {args.out}（config/processor 继承骨架）")


if __name__ == "__main__":
    main()
