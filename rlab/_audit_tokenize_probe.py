# -*- coding: utf-8 -*-
"""审查探针：retool 分段 tokenize vs 整串 tokenize 的一致性（零字面量 fence/标签）。

背景：multi_turn_rollout_group 续写时给 vLLM 的是【整串】上下文（vLLM 内部
整体 tokenize），而训练端 retool_build_batch 是【分段】tokenize 再拼 ids。
若两者产出 token 序列不同 → 训练序列与真实生成序列错位（gen_logps/ratio/mask
边界全部失真）。test_trajectory_logps 只测了"同一序列上前向等价"，没测本项。
"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("D:/tmp/qwen_tok_audit/Qwen/Qwen2___5-3B")
F = chr(96) * 3  # fence

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlab.protocol import TOOL_START, TOOL_END

def seg_ids(t):
    return tok(t, add_special_tokens=False)["input_ids"]

# 典型段边界：assistant 尾 ↔ tool 头；tool 尾 ↔ 下轮 assistant 头
cases = [
    # assistant 段以代码围栏结尾（最常见：写完代码停笔）
    (f"The code is:\n{F}python\nprint(2+2)\n{F}", TOOL_START + "4" + TOOL_END),
    (f"let me compute\n{F}python\nx=1\nprint(x)\n{F}", TOOL_START + "1" + TOOL_END),
    # assistant 段以普通文字结尾
    ("reasoning here ... some text", TOOL_START + "stdout: 42" + TOOL_END),
    # tool 尾 → 下轮 assistant 头（各种开头）
    (TOOL_START + "4" + TOOL_END, " So the answer is 4."),
    (TOOL_START + "Error! timeout" + TOOL_END, " The result is"),
    (TOOL_START + "4" + TOOL_END, "\nSo the answer is 4."),
    # 三段连拼
    ("A" * 5, TOOL_START + "4" + TOOL_END + " next"),
]

bad = 0
for a, b in cases:
    whole = seg_ids(a + b)
    split = seg_ids(a) + seg_ids(b)
    tag = "OK " if whole == split else "MISMATCH"
    if whole != split:
        bad += 1
        k = 0
        while k < min(len(whole), len(split)) and whole[k] == split[k]:
            k += 1
        print(f"[{tag}] ...{a[-30:]!r} + {b[:20:]!r}")
        print(f"    whole[{k-2}:{k+3}] = {whole[max(0,k-2):k+3]}  decode={tok.decode(whole[max(0,k-2):k+3])!r}")
        print(f"    split[{k-2}:{k+3}] = {split[max(0,k-2):k+3]}  decode={tok.decode(split[max(0,k-2):k+3])!r}")
    else:
        print(f"[{tag}] {a[-20:]!r} + {b[:15:]!r}")
print(f"\n结果: {bad}/{len(cases)} 个边界不一致")

# ===== 探针2：token -> text -> token 往返恒等性（vLLM 取 text 再重 tokenize）=====
import random
random.seed(0)
vocab = list(range(151645, 151936))  # 末段普通 token 区
rt_bad = 0
N = 2000
for _ in range(N):
    ids = [random.randrange(1000, 150000) for _ in range(random.randrange(5, 40))]
    text = tok.decode(ids)
    back = seg_ids(text)
    if back != ids:
        rt_bad += 1
        if rt_bad <= 3:
            print(f"往返不等: n_ids={len(ids)} n_back={len(back)}")
print(f"探针2 往返: {rt_bad}/{N} 不恒等")
