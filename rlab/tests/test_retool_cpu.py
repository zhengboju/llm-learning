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
    check("retool 阶段2 超参", cfg["max_rounds"] == 3 and cfg["round_gen_tokens"] == 280
          and cfg["reward_switch_step"] == 256)
    check("retool 系统提示含代码工具说明", "```python" in cfg["system_prompt"]
          and "[TOOL RESULT]" in cfg["system_prompt"])


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
    print("[H] multi_turn_rollout_group 多轮循环 + retool_score_flat 索引契约")
    from rlab.rollout import multi_turn_rollout_group, retool_score_flat

    class _C:
        def __init__(self, text): self.text = text

    class _O:
        def __init__(self, text): self.outputs = [_C(text)]

    class FakeGen:
        """按轮次回放 canned 文本，记录每轮收到的 prompts。"""
        def __init__(self, rounds):
            self.rounds = rounds
            self.r = 0
            self.seen = []

        def generate(self, prompts, sps, use_tqdm=False):
            self.seen.append(list(prompts))
            texts = self.rounds[self.r]
            self.r += 1
            assert len(texts) == len(prompts)
            return [_O(t) for t in texts]

    # 1 题 × num_pre_Q=4 条独立轨迹（修复后的扩样形态）
    prompts = ["P0", "P1", "P2", "P3"]
    r1 = ["```python\nprint(6*7)\n```", "no code",
          "```python\nprint(2+3)\n```", "none"]
    r2 = ["```python\nprint(99)\n```", "final text here"]       # 仅 active=[0,2]
    r3 = ["done"]                                               # 仅 active=[0]
    fg = FakeGen([r1, r2, r3])
    cfg_mt = {"max_rounds": 3, "sandbox_timeout": 5.0,
              "sandbox_mem_mb": 256, "tool_result_max_chars": 500}
    sps = [object() for _ in prompts]   # 每样本独立请求参数（FakeGen 不检查内容）
    segs, full_texts, code_stats = multi_turn_rollout_group(fg, sps, None, prompts, cfg_mt)

    check("轨迹数 = 组内样本数 (4)", len(full_texts) == 4 and len(segs) == 4)
    check("第2轮只续写执行过代码的样本 (0,2)", len(fg.seen[1]) == 2)
    check("第3轮只续写第2轮又执行了代码的样本 (0)", len(fg.seen[2]) == 1)
    check("续写上下文含首轮文本+工具输出(42)",
          "42" in fg.seen[1][0] and "[TOOL RESULT]" in fg.seen[1][0])
    check("s0 段序列 [a,tool,a,tool,a]",
          [s["kind"] for s in segs[0]] ==
          ["assistant", "tool", "assistant", "tool", "assistant"])
    check("s1/s3 无代码即结束 [a]",
          [s["kind"] for s in segs[1]] == ["assistant"]
          and [s["kind"] for s in segs[3]] == ["assistant"])
    check("code_used/ok 统计正确",
          code_stats[0] == {"code_used": 2, "code_ok": 2}
          and code_stats[1] == {"code_used": 0, "code_ok": 0}
          and code_stats[2] == {"code_used": 1, "code_ok": 1})
    check("工具段内容 = 沙箱 stdout",
          "42" in segs[0][1]["text"] and "5" in segs[2][1]["text"])

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


if __name__ == "__main__":
    test_extract()
    test_mask_ab()
    test_sandbox()
    test_reward_retool()
    test_protocol_mask()
    test_config_retool()
    test_trajectory_logps()
    test_multi_rollout_and_scoring()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
    sys.exit(0)
