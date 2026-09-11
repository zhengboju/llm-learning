# -*- coding: utf-8 -*-
"""rlab/protocol.py — 训练端↔生成端 的 batch 契约与字节编解码。

字节流格式（沿用本项目 ref_server 协议，bytes_list）：
  [0] json meta   : {"plen": int, "num_items_in_batch": int(可选), ...}
  [1] merged_ids  : (B, plen+T) prompt + 右 pad completion
                    （阶段0/1 单轮批的 prompt 段含左 pad；阶段2 retool 逐题用
                    rollout.strip_left_pad 剥掉左 pad 后建批，plen=本题真实 prompt
                    长——左 pad 会同时污染注意力键与位置编码，见 docs/02）
  [2] advantages  : (B,) 已在生成端按 adv_mode 归一化（或 (B,T) per-token，RF++）
  [3] refs        : (B, T) ref 模型 per-token logps（ref_server 补充）
  [4] gen_logps   : (B, T) 生成时 policy 的 per-token logps（torch 副本算）
  [5] acc_scores  : (B,) 正确性原始分（仅记录/监控用，不进 loss）
  [6] format_scores: (B,) 格式原始分（同上）

mask 约定：
- 阶段0/1（单轮）：completion 区 pad 位由训练端 inputs!=pad 重算，协议不传 mask。
- 阶段2（retool 多段工具轨迹）：meta 带 "has_mask":1，extras 首槽为 (B,T) 0/1 完成掩码
  （assistant token=1 / 工具返回段=0 / pad=0）。训练端直接采用，不再自行重算——
  工具返回 token 不进 loss 是 TIR 的核心契约，必须由生成端按段边界精确给出。
"""

import io
import json
import re

import torch

# ---------------------------------------------------------------- 阶段2 常量 ----
# 工具段起止标记：用纯文本方括号，不用尖括号/think/response 字节
# （零标签字面量铁律——含标签字节的文本经聊天管道会被改写成普通英文单词）。
# 多段轨迹里 assistant 生成的 token 进 loss，工具返回段（TOOL_START..TOOL_END）
# 只作上下文、mask 置 0 不进 loss——这是 TIR 训练最易错的点（见 tests/test_retool_cpu.py）。
TOOL_START = "\n[TOOL RESULT]\n"
TOOL_END = "\n[/TOOL RESULT]"

# python 围栏代码块提取（```python ... ```，DOTALL 跨行）
_PY_FENCE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

# 特殊 token 字面量（<|im_end|> / <|endoftext|> 等，Qwen 系通用形态）
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def sanitize_tool_text(text: str) -> str:
    """沙箱输出拼回模型上下文前的无害化消毒（对齐 agentic-rl-lab/05-retool 教训：
    "tool 返回不消毒会污染 observation 结构"）。

    沙箱 stdout 是模型经 print() 间接可控的通道——若模型打印 <|im_end|> 类
    特殊 token 字面量，分段 tokenize 时会被 Qwen tokenizer 还原成真的 special
    token id，往训练序列注入非模型生成的 EOS/边界 token（gen_logps/mask/训练
    全被污染）；打印 [TOOL RESULT] 字面量则可伪造嵌套工具边界混淆上下文。
    两类字节一律剥除——与 mask 契约同一条原则：凡"非模型生成但要拼进模型
    上下文"的字节流，都是训练序列的信任边界。"""
    return _SPECIAL_TOKEN_RE.sub("", text).replace(TOOL_START, "").replace(TOOL_END, "")


def extract_python_blocks(text: str):
    """返回文本里所有完整 ```python``` 代码块（去围栏与首尾空白）。"""
    return [m.group(1).strip() for m in _PY_FENCE_RE.finditer(text)]


def segment_mask_from_spans(total: int, assistant_spans) -> torch.Tensor:
    """纯函数：由 completion 总长 T 与 assistant 生成区间 [(s,e),...]（左闭右开）
    构造 (T,) 0/1 mask——assistant token=1，工具段/其余=0。
    阶段2 最易错点：工具返回 token 若置 1，其不可信的 logps（策略对沙箱输出
    的困惑度/KL）会污染 loss，甚至产生假梯度信号。"""
    m = torch.zeros(total)
    for s, e in assistant_spans:
        m[s:e] = 1.0
    return m


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
    """生成端打包。extra_tensors 依序：单轮为 gen_logps / acc_scores / format_scores；
    retool（meta["has_mask"]=1）为 gen_logps / mask / acc_scores / format_scores。"""
    parts = [json.dumps(meta).encode(), tensor_to_bytes(merged_ids),
             tensor_to_bytes(advantages)]
    parts.extend(tensor_to_bytes(t) for t in extra_tensors)
    return make_bytes_list(parts)


def decode_batch(raw: bytes) -> dict:
    """训练端解包（ref_server /get 返回）。布局由 meta['algo'] 决定：

    passthrough 输出（GRPO 家族）:
      [meta, inputs, advantages, refs, gen_logps, acc_scores, format_scores]
      retool（meta['has_mask']=1）:
      [meta, inputs, advantages, refs, gen_logps, mask, acc_scores, format_scores]
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
        if data.get("has_mask"):
            data["mask"] = bytes_to_tensor(dd[5])
            if len(dd) >= 7:
                data["acc_scores"] = bytes_to_tensor(dd[6])
            if len(dd) >= 8:
                data["format_scores"] = bytes_to_tensor(dd[7])
        else:
            if len(dd) >= 6:
                data["acc_scores"] = bytes_to_tensor(dd[5])
            if len(dd) >= 7:
                data["format_scores"] = bytes_to_tensor(dd[6])
    return data
