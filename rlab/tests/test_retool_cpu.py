# -*- coding: utf-8 -*-
"""rlab/tests/test_retool_cpu.py — 阶段2（ReTool 代码交织 RL）CPU 验收测试。

覆盖（本阶段最重要的学习点是【工具 token mask 错/对】——B 组直接对拍）：
  A. 代码块提取（extract_python_blocks）
  B. 【mask 错/对对照 A/B】：同一条含工具段的轨迹，mask 正确（工具段=0）vs
     mask 错误（工具段=1）→ loss/梯度/被 KL 污染的差异数值实锤
  C. 沙箱（run_code）：成功/异常/超时/空代码/输出截断
  D. 阶段2 奖励（acc+fmt+code 组合、冷启动/后期权重切换）
  E. 协议：mask 槽位 roundtrip + 向后兼容（无 has_mask → 旧布局）
  F. config：retool preset 与阶段2 超参
  G. tiny GPT2 全序列前向 == 逐段前缀前向（logps 对齐，省去按段拼接）
     + 轨迹 mask 下工具段梯度为 0

运行：python -m rlab.tests.test_retool_cpu
"""
import os
import sys
import tempfile
import threading

import torch

from rlab.config import get_config
from rlab.losses import compute_loss, get_per_token_logps
from rlab.protocol import (TOOL_END, TOOL_START, decode_batch, encode_batch,
                           extract_python_blocks, segment_mask_from_spans)
from rlab.reward import (_FORMAT_RE, reward_code, reward_phase,
                         total_reward_retool)
from rlab.sandbox import run_code

PASS = []


def fmt_answer(num="72"):
    """从 _FORMAT_RE 字面量 roundtrip 构造一个必然格式正确的答案串。
    铁律：零标签字面量——格式标签字节一律从 rlab 导入/派生，绝不手写
    （手写标签经聊天管道会被改写，且显示层会把标签渲染成误导形态）。"""
    body = _FORMAT_RE.replace("^", "").replace("$", "")
    p = body.split(".*?")          # [" thinking", " response<answer>", "</answer>"]
    assert len(p) == 3, f"unexpected _FORMAT_RE shape: {body!r}"
    return p[0] + "x" + p[1] + num + p[2]


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


# ---------------------------------------------------------------- A. 代码提取 ----
def test_extract():
    print("[A] extract_python_blocks")
    t1 = "before\n```python\nx = 1 + 1\nprint(x)\n```\nafter"
    bl = extract_python_blocks(t1)
    check("单代码块提取", len(bl) == 1 and bl[0] == "x = 1 + 1\nprint(x)")
    t2 = "```python\nprint(1)\n```\nthinking\n```python\nprint(2)\n```"
    check("多代码块提取（全返回）", [b for b in extract_python_blocks(t2)] == ["print(1)", "print(2)"])
    check("无代码块", extract_python_blocks("no fence here") == [])
    check("未闭合围栏不提取", extract_python_blocks("```python\nx=1") == [])


# ------------------------------------------- B. mask 错/对对照 A/B（学习点） ----
def test_mask_ab():
    print("[B] 工具 token mask 错/对 对照 A/B")
    cfg = get_config("retool", use_wandb=False, beta=0.04)
    # 轨迹（2 样本，T=6）：
    #   s0: [assistant*3 | TOOL*2 | pad]   工具段 = 沙箱输出（策略从未生成它）
    #   s1: [assistant*4 | pad | pad]
    gen = torch.tensor([
        [-1.0, -1.0, -1.0, -8.0, -8.0, 0.0],   # 工具 token：策略对沙箱输出极困惑（logp -8）
        [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0],
    ])
    pol = gen.clone().requires_grad_(True)      # ratio=1，便于手算
    ref = torch.tensor([                         # 工具 token 上 ref（base）也困惑但没那么深
        [-1.0, -1.0, -1.0, -2.0, -2.0, 0.0],
        [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0],
    ])
    adv = torch.tensor([1.0, -1.0])
    mask_correct = torch.tensor([
        [1, 1, 1, 0, 0, 0],    # 工具段=0：正确
        [1, 1, 1, 1, 0, 0],
    ])
    mask_wrong = torch.tensor([
        [1, 1, 1, 1, 1, 0],    # 工具段=1：错误（把沙箱输出当模型生成）
        [1, 1, 1, 1, 0, 0],
    ])

    # --- 正确 mask：只算 assistant token
    loss_c, _ = compute_loss("grpo", pol, gen, adv, mask_correct, cfg, ref_logps=ref)
    # 手算：s0 = -(1*1 - 0)= -1 的 3 token 均值 = -1；s1 = -(-1)= +1 的 4 token 均值 = +1；
    # sample_mean batch 平均 = 0
    check("正确 mask loss = 0（纯 assistant，KL=0）", abs(loss_c.item()) < 1e-5)

    pol2 = gen.clone().requires_grad_(True)
    lc2, _ = compute_loss("grpo", pol2, gen, adv, mask_correct, cfg, ref_logps=ref)
    lc2.backward()
    check("正确 mask：s0 工具 token 梯度严格为 0",
          bool(pol2.grad[0, 3:5].abs().sum() == 0))

    # --- 错误 mask：工具 token 进 loss → KL 污染 + 假梯度
    loss_w, _ = compute_loss("grpo", pol, gen, adv, mask_wrong, cfg, ref_logps=ref)
    # 工具 token 上 KL = exp(ref-pol)- (ref-pol) -1 = exp(-2-(-8)) - 6 - 1 ≈ 396/个，
    # β=0.04 后 s0 样本均值被推高约 5.7，batch 均值从 0 → ~3.4（对比 beta=0 时同值）
    check("错误 mask：loss 被工具段 KL 污染（明显大于正确 mask）",
          loss_w.item() > loss_c.item() + 1.0)

    pol3 = gen.clone().requires_grad_(True)
    lw2, _ = compute_loss("grpo", pol3, gen, adv, mask_wrong, cfg, ref_logps=ref)
    lw2.backward()
    check("错误 mask：工具 token 梯度非零（假信号实锤）",
          bool(pol3.grad[0, 3:5].abs().sum() > 0))

    # --- 即使 beta=0（无 KL），错误 mask 仍在工具 token 上制造梯度
    cfg0 = get_config("retool", use_wandb=False, beta=0.0)
    pol4 = gen.clone().requires_grad_(True)
    l0w, _ = compute_loss("grpo", pol4, gen, adv, mask_wrong, cfg0, ref_logps=ref)
    l0w.backward()
    check("错误 mask（beta=0）：工具 token 梯度仍非零",
          bool(pol4.grad[0, 3:5].abs().sum() > 0))


# ---------------------------------------------------------------- C. 沙箱 ----
def test_sandbox():
    print("[C] run_code 沙箱")
    ok = run_code("x = 2 + 3\nprint(x)", timeout=5)
    check("成功：stdout=5", ok["ok"] and ok["display"].strip() == "5")
    err = run_code("raise ValueError('boom')", timeout=5)
    check("异常：返回 Error! 摘要", (not err["ok"]) and err["display"].startswith("Error!"))
    tm = run_code("while True:\n    pass", timeout=1)
    check("死循环：超时 SIGKILL", tm["timed_out"] and (not tm["ok"]))
    em = run_code("   ", timeout=5)
    check("空代码：返回空错误", not em["ok"])
    big = run_code("print('A' * 10000)", timeout=5, max_chars=100)
    check("输出截断（防输出炸弹）", len(big["display"]) <= 100)
    clean = run_code("import sys; print(sys.executable)", timeout=5)
    check("子进程真实执行（非本进程 exec）", clean["ok"] and "python" in clean["display"].lower())


# ------------------------------------------------------------ D. 阶段2 奖励 ----
def test_reward_retool():
    print("[D] 阶段2 奖励与冷启动切换")
    good = fmt_answer("72")   # 由 _FORMAT_RE 派生，保证格式正则与计数双通过
    # cold 权重 (1,2,2)：acc=+1, fmt=+1, code_ok=2 → 1 + 2 + 2*(2*0.1)=3.4
    sc = total_reward_retool("72", good, code_ok=2, phase="cold", code_w=0.1)
    check("cold 正确 + 2 个成功代码 = 3.4", abs(sc["reward"] - 3.4) < 1e-6)
    # hot 权重 (2,1,1)：acc=+1, fmt=+1, code_ok=1 → 2 + 1 + 0.1 = 3.1
    sc2 = total_reward_retool("72", good, code_ok=1, phase="hot", code_w=0.1)
    check("hot 正确 + 1 个成功代码 = 3.1", abs(sc2["reward"] - 3.1) < 1e-6)
    sc3 = total_reward_retool("99", good, code_ok=0, phase="hot", code_w=0.1)
    check("hot 答错 = 2*(-1)+1+0 = -1", abs(sc3["reward"] - (-1.0)) < 1e-6)
    check("reward_code 单块成功 = 0.1", abs(reward_code(1, 0.1) - 0.1) < 1e-9)
    check("reward_code 全失败 = 0", reward_code(0, 0.1) == 0.0)
    check("reward_phase 冷（<switch）", reward_phase(100, 256) == "cold")
    check("reward_phase 热（≥switch）", reward_phase(256, 256) == "hot")
    check("retool 格式正则仍兼容（复用阶段0/1 口径）",
          sc["format"] == 1.0 and sc["acc"] == 1.0)


def test_strip_code_scoring():
    """【2026-09-08 第三轮真机教训】打分域 = 剥离代码块后的回答文本。

    MUST 提示让模型"代码先行"（围栏开局），而 _FORMAT_RE ^ 锚定要求以思考标签
    开头——冲突导致所有代码先行样本 fmt 结构性失败（训练 fmt 恒 -1 死亡、
    eval 精确 0/300）。修复：acc/fmt 一律在 strip_code_blocks 后的文本上判。
    本组把这些契约锁死（零标签字面量：格式串一律经 fmt_answer/_FORMAT_RE 派生）。"""
    print("[D2] 打分域：剥离代码块（MUST 代码先行 vs 格式锚定冲突）")
    from rlab.reward import reward_format_retool, strip_code_blocks
    good = fmt_answer("72")   # 由 _FORMAT_RE 派生的合法格式串
    code_first = "```python\nprint(70+2)\n```\n\n" + good   # 模型典型开头：代码先行
    check("剥离代码块：围栏与内容整体移除", strip_code_blocks(code_first) == good)
    check("代码先行 + 尾随合法格式 → 格式 1.0（修复前必 -1）",
          reward_format_retool(code_first) == 1.0)
    p = _FORMAT_RE.replace("^", "").replace("$", "").split(".*?")
    mid = p[0] + "x```python\nprint(1)\n```y" + p[1] + "72" + p[2]
    check("代码夹在思考与答案之间 → 格式 1.0", reward_format_retool(mid) == 1.0)
    check("未闭合围栏不剥离 → 结构仍不合格 -1",
          reward_format_retool("```python\nx = 1\n" + good) == -1.0)
    check("只有代码没有答案标签 → 格式 -1",
          reward_format_retool("```python\nprint(1)\n```") == -1.0)
    # acc 取数域：代码在答案标签之后，其数字不得当"模型答案"
    code_after = good + "\n```python\nprint(99)\n```"
    sc = total_reward_retool("72", code_after, code_ok=1, phase="hot", code_w=0.1)
    check("代码在答案后（剥离后取 72；不剥离会取到 99）", sc["acc"] == 1.0)
    sc2 = total_reward_retool("72", code_first, code_ok=1, phase="hot", code_w=0.1)
    check("代码先行 + 正确答案：acc/fmt 双 1", sc2["acc"] == 1.0 and sc2["format"] == 1.0)
    sc3 = total_reward_retool("72", good, code_ok=0, phase="hot", code_w=0.1)
    check("无代码文本剥离是恒等（阶段0/1 口径不变）",
          abs(sc3["reward"] - 3.0) < 1e-9 and sc3["format"] == 1.0)


# ---------------------------------------------------------------- E. 协议 ----
def test_protocol_mask():
    print("[E] 协议 mask 槽位与向后兼容")
    meta = {"plen": 5, "algo": "retool", "has_mask": 1}
    ids = torch.randint(0, 100, (2, 11))
    adv = torch.tensor([0.5, -0.5])
    gl = torch.randn(2, 6)
    mask = torch.tensor([[1., 1., 0., 0., 1., 0.], [1., 1., 1., 1., 0., 0.]])
    acc = torch.tensor([1.0, -1.0]); fmt = torch.tensor([1.0, -1.0])
    raw = encode_batch(meta, ids, adv, gl, mask, acc, fmt)
    # 模拟 passthrough ref_server：插 refs 到第 3 位
    from rlab.protocol import bytes_list_to_list, make_bytes_list, tensor_to_bytes
    dd = bytes_list_to_list(raw)
    refs = torch.randn(2, 6)
    out = make_bytes_list([dd[0], dd[1], dd[2], tensor_to_bytes(refs),
                           dd[3], dd[4], dd[5], dd[6]])
    d = decode_batch(out)
    check("retool mask roundtrip", torch.equal(d["mask"], mask))
    check("retool 契约完整", {"mask", "inputs", "advantages", "refs", "gen_logps",
                            "acc_scores", "format_scores"} <= set(d))

    # 向后兼容：无 has_mask → 旧布局（acc/fmt 紧随 gen_logps）
    raw2 = encode_batch({"plen": 5, "algo": "grpo"}, ids, adv, gl, acc, fmt)
    dd2 = bytes_list_to_list(raw2)
    out2 = make_bytes_list([dd2[0], dd2[1], dd2[2], tensor_to_bytes(refs),
                            dd2[3], dd2[4], dd2[5]])
    d2 = decode_batch(out2)
    check("旧布局向后兼容（无 mask）", "mask" not in d2
          and torch.equal(d2["acc_scores"], acc) and torch.equal(d2["format_scores"], fmt))


# ------------------------------------------------------------ F. config ----
def test_config_retool():
    print("[F] config retool preset")
    cfg = get_config("retool", use_wandb=False)
    check("retool 复用 grpo loss（group_std/sample_mean）",
          cfg["adv_mode"] == "group_std" and cfg["loss_norm"] == "sample_mean")
    check("retool beta=0.04（有 KL）", cfg["beta"] == 0.04)
    check("retool 阶段2 超参（round=400：280 会截断代码围栏→写代码被结构性惩罚）",
          cfg["max_rounds"] == 3 and cfg["round_gen_tokens"] == 400
          and cfg["reward_switch_step"] == 256)
    check("retool 系统提示含代码工具说明（MUST 冷启动：MAY 采样率仅~0.3%进不了分布）",
          "```python" in cfg["system_prompt"]
          and "[TOOL RESULT]" in cfg["system_prompt"]
          and "You MUST write Python code" in cfg["system_prompt"])


# --------------------------------- G. tiny GPT2：logps 对齐 + mask 排除 ----
def _save_tiny_gpt2(tmpdir):
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=64, n_layer=2, n_head=2)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model.save_pretrained(tmpdir)
    tok.save_pretrained(tmpdir)
    return tmpdir


def test_trajectory_logps():
    print("[G] tiny GPT2 全序列前向 == 逐段前缀前向（logps 对齐）")
    from transformers import AutoTokenizer, GPT2LMHeadModel
    with tempfile.TemporaryDirectory() as tmp:
        path = _save_tiny_gpt2(tmp)
        tok = AutoTokenizer.from_pretrained(path)
        model = GPT2LMHeadModel.from_pretrained(path).eval()

        prompt = "Question: 2+2="
        asst1 = " Let me compute.\n```python\nprint(2+2)\n```"
        tool = TOOL_START + "4" + TOOL_END
        asst2 = " So the answer is 4."
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a1 = tok(asst1, add_special_tokens=False)["input_ids"]
        tl = tok(tool, add_special_tokens=False)["input_ids"]
        a2 = tok(asst2, add_special_tokens=False)["input_ids"]
        plen = len(p_ids)
        merged = torch.tensor([p_ids + a1 + tl + a2])
        # assistant 区间（completion 坐标）
        spans = [(plen - 1, plen - 1 + len(a1)), (plen - 1 + len(a1) + len(tl),
                                                  plen - 1 + len(a1) + len(tl) + len(a2))]

        # 全序列前向的逐 token logp
        with torch.inference_mode():
            lp_full = get_per_token_logps(model(merged).logits[:, :-1, :],
                                          merged[:, 1:])[0]  # (L-1,)

        # 逐段前缀前向：assistant1 段的 logp 由 [prompt+asst1] 前向得到（因果性）
        with torch.inference_mode():
            pre1 = torch.tensor([p_ids + a1])
            lp_pre1 = get_per_token_logps(model(pre1).logits[:, :-1, :],
                                          pre1[:, 1:])[0][plen - 1:]     # (len(a1),)
            pre2 = torch.tensor([p_ids + a1 + tl + a2])
            lp_pre2 = get_per_token_logps(model(pre2).logits[:, :-1, :],
                                          pre2[:, 1:])[0][plen - 1:]     # (len(a1)+len(tl)+len(a2),)
            # assistant2 段的 logp = pre2 前向在 [len(a1)+len(tl) : ] 处
            lp_a2 = lp_pre2[len(a1) + len(tl):]

        s1, e1 = spans[0]
        s2, e2 = spans[1]
        check("assistant1 段 logp：全序列 == 逐段前缀",
              torch.allclose(lp_full[s1:e1], lp_pre1, atol=1e-5))
        check("assistant2 段 logp：全序列 == 逐段前缀",
              torch.allclose(lp_full[s2:e2], lp_a2, atol=1e-5))

        # 工具段在 mask=0 → 不产生梯度；错误 mask → 有梯度
        T = len(a1) + len(tl) + len(a2)
        mask_c = segment_mask_from_spans(T, [(s1 - (plen - 1), e1 - (plen - 1)),
                                             (s2 - (plen - 1), e2 - (plen - 1))]).unsqueeze(0)
        gen_lp = lp_full[plen - 1:].unsqueeze(0).detach()
        cfg = get_config("retool", use_wandb=False, beta=0.0)
        pol = lp_full[plen - 1:].unsqueeze(0).clone().requires_grad_(True)
        loss, _ = compute_loss("grpo", pol, gen_lp, torch.tensor([1.0]),
                               mask_c, cfg, ref_logps=gen_lp)
        loss.backward()
        tool_slice = slice(len(a1), len(a1) + len(tl))
        check("轨迹 mask：工具段梯度为 0（整条轨迹验证）",
              bool(torch.all(pol.grad[0, tool_slice] == 0)))
        check("轨迹 mask：assistant 段有梯度",
              bool(pol.grad[0, :len(a1)].abs().sum() > 0)
              and bool(pol.grad[0, len(a1) + len(tl):].abs().sum() > 0))


# --------------------------------- H. 多轮循环 + 打分索引契约（FakeGen） ----
def test_multi_rollout_and_scoring():
    print("[H] multi_turn_rollout_group 多轮循环 + 同序列契约 + retool_score_flat")
    from rlab.rollout import (multi_turn_rollout_group, retool_build_batch,
                              retool_context_overlong, retool_score_flat)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token          # H 组内部需要 padding
    tok.padding_side = "left"

    def tids(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    class _C:
        def __init__(self, text):
            self.text = text
            self.token_ids = tids(text)    # 模拟 vLLM：token_ids 与 text 对应

    class _O:
        def __init__(self, text): self.outputs = [_C(text)]

    class FakeGen:
        """按轮次回放 canned 文本（带 token_ids），记录每轮收到的 prompt_token_ids。"""
        def __init__(self, rounds):
            self.rounds = rounds
            self.r = 0
            self.seen = []   # list[list[list[int]]] —— 每轮收到的 token id prompt

        def generate(self, prompts, sps, use_tqdm=False):
            self.seen.append([list(p["prompt_token_ids"]) for p in prompts])
            texts = self.rounds[self.r]
            self.r += 1
            assert len(texts) == len(prompts)
            return [_O(t) for t in texts]

    # 1 题 × num_pre_Q=4 条独立轨迹（真机扩样形态：**同一 prompt 重复 4 份**，
    # retool_build_batch 的契约是 Q 行 prompt × Q*n 条轨迹）
    prompts = ["PA", "PA", "PA", "PA"]
    r1 = ["```python\nprint(6*7)\n```", "no code",
          "```python\nprint(2+3)\n```", "none"]
    r2 = ["```python\nprint(99)\n```", "final text here"]       # 仅 active=[0,2]
    r3 = ["```python\nprint(1)\n```"]                           # 仅 active=[0]——末轮仍写代码！
    fg = FakeGen([r1, r2, r3])
    cfg_mt = {"max_rounds": 3, "sandbox_timeout": 5.0,
              "sandbox_mem_mb": 256, "tool_result_max_chars": 500}
    sps = [object() for _ in prompts]   # 每样本独立请求参数（FakeGen 不检查内容）
    segs, full_texts, code_stats = multi_turn_rollout_group(fg, sps, tok, prompts, cfg_mt)

    check("轨迹数 = 组内样本数 (4)", len(full_texts) == 4 and len(segs) == 4)
    # 【发现1 修复锁】生成端必须收到 token id prompt（不再收整串文本）
    check("生成端收到 prompt_token_ids（token id 续写）",
          len(fg.seen[0]) == 4 and all(isinstance(x, list) for x in fg.seen[0]))
    check("第2轮只续写执行过代码的样本 (0,2)", len(fg.seen[1]) == 2)
    check("第3轮只续写第2轮又执行了代码的样本 (0)", len(fg.seen[2]) == 1)
    # 【发现1 修复锁·核心契约】续写上下文 = 前轮 prompt+assistant ids+工具段 ids
    # 逐 token 拼接（生成序列 == 训练序列；旧版整串文本续写时 BPE 跨界合并
    # "```"+"`\n" 会让两序列分叉——实测 whole=67 vs split=68）
    tool42 = tids(TOOL_START + "42" + TOOL_END)   # print(6*7) 的沙箱输出
    expect_r2 = tids(prompts[0]) + tids(r1[0]) + tool42
    check("续写上下文 = prompt+assistant+工具段 ids 逐 token 拼接",
          fg.seen[1][0] == expect_r2)    # 【发现3 修复锁】末轮（第 max_rounds 轮）写了代码也不执行：
    # code_used 停在 2、segs 末尾无第 3 个 tool 段（修复前为 3/3 + 多一个 tool 段）
    check("s0 段序列 [a,tool,a,tool,a]（末轮代码不执行→无第3个tool）",
          [s["kind"] for s in segs[0]] ==
          ["assistant", "tool", "assistant", "tool", "assistant"])
    check("s1/s3 无代码即结束 [a]",
          [s["kind"] for s in segs[1]] == ["assistant"]
          and [s["kind"] for s in segs[3]] == ["assistant"])
    check("code_used/ok 统计正确（末轮代码不计入）",
          code_stats[0] == {"code_used": 2, "code_ok": 2}
          and code_stats[1] == {"code_used": 0, "code_ok": 0}
          and code_stats[2] == {"code_used": 1, "code_ok": 1})
    check("工具段内容 = 沙箱 stdout",
          "42" in segs[0][1]["text"] and "5" in segs[2][1]["text"])
    check("每段带 ids（assistant 段 ids = 生成 token）",
          segs[0][0]["ids"] == tids(r1[0]) and segs[0][1]["ids"] == tool42)

    # 【发现1 修复锁】retool_build_batch：per_sample_ids = 各段 ids 拼接（含工具段），
    # merged 序列与生成序列逐 token 一致；mask 工具段=0。
    # 契约：prompt_ids 是 Q=1 行（一道题），segs 是 Q*n=4 条（扩样后共享该 prompt）
    prompt_ids = tok([prompts[0]], return_tensors="pt", padding=True,
                     add_special_tokens=False)["input_ids"]
    plen = prompt_ids.shape[1]
    merged, mask, per_sample_ids = retool_build_batch(
        prompt_ids, segs, plen, tok.pad_token_id)
    check("per_sample_ids = 各段 ids 拼接（含工具段，与生成同序列）",
          per_sample_ids[0] == tids(r1[0]) + tool42 + tids(r2[0])
          + tids(TOOL_START + "99" + TOOL_END) + tids(r3[0]))
    check("merged 形状 = (B, plen+T) 且 T=max 完成长", merged.shape[0] == 4
          and merged.shape[1] == plen + max(len(t) for t in per_sample_ids))
    m0 = mask[0].tolist()
    a1, tl1 = len(tids(r1[0])), len(tool42)
    check("mask：assistant 段=1 / 工具段=0（s0 首两段）",
          m0[:a1] == [1.0] * a1 and m0[a1:a1 + tl1] == [0.0] * tl1)

    # 【发现2 修复锁】超长检查按全长（assistant+工具段）计——旧 mask.sum 口径漏算工具段
    # （[[0]*5 是"长 5 的轨迹"，不是 token 值 5）
    check("超长检查：全长口径（含工具段）触发",
          retool_context_overlong([[0] * 5, [0] * 5], 3, 7))      # 5+5+3*2=16 > 7*2=14
    check("超长检查：预算内不触发",
          not retool_context_overlong([[0] * 5, [0] * 5], 3, 10))  # 16 < 20
    check("超长检查：工具段撑爆预算（纯 assistant 口径会漏放行）",
          retool_context_overlong([[0] * 10, [0] * (10 + 6)], 3, 15))  # 10+16+6=32 > 30，工具段6是关键

    # 打分索引契约：asst_texts 必须 = 题数 × num_pre_Q（真机 IndexError 的回归锁）
    cfg2 = get_config("retool", use_wandb=False)
    asst_texts = ["".join(s["text"] for s in segs_i if s["kind"] == "assistant")
                  for segs_i in segs]
    inputs = [{"Q": "q", "A": "42"}]
    adv, acc_s, fmt_s, cu, ck, phase = retool_score_flat(
        inputs, asst_texts, code_stats, cfg2, steps_elapsed=0)
    check("score_flat 输出长度 = Q*n=4", adv.shape[0] == 4 and len(cu) == 4)
    check("cu/ck 与 code_stats 对齐", cu == [2, 0, 1, 0] and ck == [2, 0, 1, 0])
    check("group_std 组内和≈0", abs(float(adv.sum())) < 1e-4)
    check("phase cold（steps_elapsed=0 < 256）", phase == "cold")
    # 打分文本必须是 assistant 拼接而非全文：全文上 reward_format 必败
    # （fmt 恒常数 → 组内归一化后信号死亡，2026-09-08 真机 fmt=0.0% 根因）
    check("asst 文本不含 prompt 前缀", all(not t.startswith("P") for t in asst_texts))
    check("asst 文本不含工具段", all("[TOOL RESULT]" not in t for t in asst_texts))
    from rlab.reward import reward_format
    check("全文上 reward_format 必败（对照实锤）",
          all(reward_format(p + t) == -1.0 for p, t in zip(prompts, asst_texts)))
    try:
        retool_score_flat(inputs, asst_texts[:1], code_stats[:1], cfg2, 0)
        ok_flag = False
    except AssertionError:
        ok_flag = True
    check("漏扩样（轨迹数=题数）直接 AssertionError", ok_flag)


# --------------------------------- I. 训练期健康检查（窗口签名） ----
def test_health_monitor():
    print("[I] rlab.health 窗口签名（历史 bug 的训练期探测）")
    from rlab.health import HealthMonitor, window_check

    def mk(n, acc_fn, fmt_fn, clen=100.0, code_rate=0.5):
        return [{"acc": acc_fn(i), "fmt": fmt_fn(i), "clen": clen,
                 "code_rate": code_rate} for i in range(n)]

    # 签名①：fmt 恒为常数（retool 打分域 bug 的签名）
    hist = mk(40, lambda i: 0.3 + 0.1 * (i % 2), lambda i: -1.0)
    codes = {c for c, _ in window_check(hist)}
    check("fmt 恒常数 → fmt_const", "fmt_const" in codes and "acc_const" not in codes)

    # 签名②：100 组后格式率仍 <25%（温度混杂事件签名）
    hist = mk(110, lambda i: 0.3 + 0.1 * (i % 2), lambda i: -1.0)
    codes = {c for c, _ in window_check(hist)}
    check("fmt 低位持续 → fmt_low + fmt_const", {"fmt_low", "fmt_const"} <= codes)

    # 对照：fmt 恒 +1.0（格式学满）是健康收敛，不是信号死亡
    hist = mk(40, lambda i: 0.3 + 0.1 * (i % 2), lambda i: 1.0)
    codes = {c for c, _ in window_check(hist)}
    check("fmt 恒 +1（学满）不误报 fmt_const", "fmt_const" not in codes)

    # 签名③a：256 组净提升 <1pp（零梯度/没有学习签名）——acc 有波动但均值不动
    hist = mk(256, lambda i: 0.5 + (0.05 if i % 2 else -0.05), lambda i: 0.5)
    codes = {c for c, _ in window_check(hist)}
    check("长期平坦 → flat", "flat" in codes and "acc_const" not in codes)

    # 签名③b：开局 0.5 → 尾窗 0.3（rfpp 退化签名）
    def acc_decline(i):
        return 0.5 if i < 128 else (0.3 + (0.04 if i % 2 else -0.04))
    hist = mk(256, acc_decline, lambda i: 0.9)
    codes = {c for c, _ in window_check(hist)}
    check("单调下滑 → decline（不误报 flat）", "decline" in codes and "flat" not in codes)

    # 签名④：completion 顶满上限（截断坍缩）
    hist = mk(40, lambda i: 0.3, lambda i: 0.9, clen=99.0)
    codes = {c for c, _ in window_check(hist, max_clen=100.0)}
    check("长度顶满 → trunc", "trunc" in codes)

    # 签名⑤：retool 128 组后 code_rate 恒 0
    hist = mk(130, lambda i: 0.3, lambda i: 0.9, code_rate=0.0)
    codes = {c for c, _ in window_check(hist, retool=True)}
    check("代码信号未出现 → no_code", "no_code" in codes)

    # 健康 曲线：acc 上升 / fmt 从低位学到高位 → 无致命告警
    def acc_rise(i):
        return -0.2 + 1.0 * i / 300
    def fmt_rise(i):
        # 学满后带微小抖动（真实训练 fmt 率在 99~100% 间抖动，不会是精确常数）
        return -0.4 + 1.3 * min(1.0, i / 150) + (0.01 if i % 2 else -0.01)
    hist = mk(300, acc_rise, fmt_rise, clen=120.0, code_rate=0.4)
    codes = {c for c, _ in window_check(hist, retool=True, max_clen=200.0)}
    check("健康曲线 → 无任何告警", codes == set())

    # 32 组不足 → 不检查；Monitor 同一告警只报一次
    check("样本不足 32 组 → 不告警", window_check(mk(31, lambda i: 0.0, lambda i: 0.0)) == [])
    m = HealthMonitor()
    for _ in range(64):
        m.observe([-1.0, 1.0], [-1.0, -1.0], [100, 100])
    import io
    import contextlib
    buf1, buf2 = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf1):
        m.maybe_check(max_clen=200)
    with contextlib.redirect_stdout(buf2):
        m.maybe_check(max_clen=200)
    check("Monitor 触发告警一次",
          "[健康检查]" in buf1.getvalue() and buf2.getvalue() == "")

    # 权重指纹：float64 位级敏感——单权重翻转一个 bf16 ULP 必须被识别
    from rlab.health import weight_fingerprint
    sd1 = {"a": torch.randn(2048).bfloat16(),
           "b": torch.randn(1 << 21).bfloat16()}
    fp1 = weight_fingerprint(sd1)
    sd2 = {k: v.clone() for k, v in sd1.items()}
    check("权重未变 → 指纹相同", weight_fingerprint(sd2) == fp1)
    bits = sd2["b"][123:124].view(torch.int16)   # 位模式 +1 = 翻转一个 bf16 ULP
    bits += 1
    check("单权重翻转 1 ULP → 指纹必变", weight_fingerprint(sd2) != fp1)


# --------------------------------- J. 静态未定义名检查（运行时 NameError 防线） ----
def test_pyflakes_undefined():
    print("[J] pyflakes 静态检查：gen_worker 内部只有运行时才执行，import 冒烟测不出"
          "未定义名（_health 别名事故教训）")
    try:
        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
    except ImportError:
        print("  skip: pyflakes 未安装（pip install pyflakes 后本检查生效）")
        return
    import contextlib
    import io
    files = ["rlab/rollout.py", "rlab/train.py", "rlab/health.py", "rlab/config.py",
             "rlab/protocol.py", "rlab/reward.py", "rlab/losses.py", "rlab/sync.py",
             "rlab/sandbox.py", "rlab/analysis.py", "rlab/probe_retool_gen.py",
             "eval_vllm_one.py", "eval_vllm.py"]
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        for f in files:
            checkPath(f, Reporter(buf, buf))
    undefined = [l for l in buf.getvalue().splitlines() if "undefined name" in l]
    check("无未定义名", undefined == [])
    if undefined:
        for l in undefined:
            print(f"  !! {l}")


if __name__ == "__main__":
    test_extract()
    test_mask_ab()
    test_sandbox()
    test_reward_retool()
    test_strip_code_scoring()
    test_protocol_mask()
    test_config_retool()
    test_trajectory_logps()
    test_multi_rollout_and_scoring()
    test_health_monitor()
    test_pyflakes_undefined()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
    sys.exit(0)
