# -*- coding: utf-8 -*-
"""rlab/protocol.py — 训练端↔生成端 的 batch 契约与字节编解码。

字节流格式（沿用本项目 ref_server 协议，bytes_list）：
  [0] json meta   : {"plen": int, "num_items_in_batch": int(可选), ...}
  [1] merged_ids  : (B, plen+T) 左 pad prompt + 右 pad completion
  [2] advantages  : (B,) 已在生成端按 adv_mode 归一化（或 (B,T) per-token，RF++）
  [3] refs        : (B, T) ref 模型 per-token logps（ref_server 补充）
  [4] gen_logps   : (B, T) 生成时 policy 的 per-token logps（torch 副本算）
  [5] acc_scores  : (B,) 正确性原始分（仅记录/监控用，不进 loss）
  [6] format_scores: (B,) 格式原始分（同上）

mask 约定：completion 区 pad token 位置在训练端由 inputs!=pad 重算，
因此本协议不需要显式传 mask —— 工具段 mask（阶段2）届时才加入 meta。
"""

import io
import json

import torch


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def bytes_to_tensor(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)


def make_bytes_list(blist) -> bytes:
    buf = io.BytesIO()
    buf.write(len(blist).to_bytes(4, "big"))
    for b in blist:
        buf.write(len(b).to_bytes(4, "big"))
        buf.write(b)
    return buf.getvalue()


def bytes_list_to_list(b: bytes):
    buf = io.BytesIO(b)
    num = int.from_bytes(buf.read(4), "big")
    out = []
    for _ in range(num):
        l = int.from_bytes(buf.read(4), "big")
        out.append(buf.read(l))
    return out


def encode_batch(meta: dict, merged_ids: torch.Tensor, advantages: torch.Tensor,
                 *extra_tensors: torch.Tensor) -> bytes:
    """生成端打包。extra_tensors 依序为 gen_logps / acc_scores / format_scores。"""
    parts = [json.dumps(meta).encode(), tensor_to_bytes(merged_ids),
             tensor_to_bytes(advantages)]
    parts.extend(tensor_to_bytes(t) for t in extra_tensors)
    return make_bytes_list(parts)


def decode_batch(raw: bytes) -> dict:
    """训练端解包（ref_server /get 返回）。布局由 meta['algo'] 决定：

    passthrough 输出（GRPO 家族）:
      [meta, inputs, advantages, refs, gen_logps, acc_scores, format_scores]
    rfpp 输出（多一个服务端算好的 per-token advantages 段）:
      [meta, inputs, raw_rewards, refs, gen_logps, advantages(B,T), acc_scores, format_scores]
    """
    dd = bytes_list_to_list(raw)
    data = json.loads(dd[0])
    data["inputs"] = bytes_to_tensor(dd[1])
    data["rewards"] = bytes_to_tensor(dd[2])
    data["refs"] = bytes_to_tensor(dd[3])
    data["gen_logps"] = bytes_to_tensor(dd[4])
    if data.get("algo") == "rfpp":
        data["advantages"] = bytes_to_tensor(dd[5])
        if len(dd) >= 7:
            data["acc_scores"] = bytes_to_tensor(dd[6])
        if len(dd) >= 8:
            data["format_scores"] = bytes_to_tensor(dd[7])
    else:
        data["advantages"] = data["rewards"]      # GRPO 家族：上传的就是归一化 advantage
        if len(dd) >= 6:
            data["acc_scores"] = bytes_to_tensor(dd[5])
        if len(dd) >= 7:
            data["format_scores"] = bytes_to_tensor(dd[6])
    return data
