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
import json
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


def test_config_retool_math():
    print("[F2] config retool_math preset（对齐 agentic-rl-lab/05-retool 采样与组配置）")
    cfg = get_config("retool_math", use_wandb=False)
    check("retool_math 采样 temperature=1.0（GSM8K 的 0.7 是格式学习遗产，math 无格式压力）",
          cfg["temperature"] == 1.0)
    check("retool_math 无 top_k（vLLM top_k=-1=全词表，与参考一致）", cfg["top_k"] == -1)
    check("retool_math 组 8 条（参考 group_size=8，micro batch 同步 8）",
          cfg["num_pre_Q"] == 8 and cfg["train_micro_batch_size_per_gpu"] == 8)
    check("retool_math adv 不除 std（参考组内减均值，group_mean）",
          cfg["adv_mode"] == "group_mean")
    # 【2026-09-10 训练变慢修复锁】并采 4 题（vLLM 并发 4×8=32）+ 题目过滤走
    # QuestionScheduler 队列路径；GSM8K 家族保持 1（旧逐题协议可比性）
    check("retool_math 并采 4 题 + 沙箱并发 8",
          cfg["gen_questions_per_attempt"] == 4 and cfg["sandbox_workers"] == 8)
    check("BASE 默认并采 1 题 + 沙箱并发 4（GSM8K 家族协议不变）",
          get_config("retool", use_wandb=False)["gen_questions_per_attempt"] == 1
          and get_config("retool", use_wandb=False)["sandbox_workers"] == 4)
    # 联动锁：num_pre_Q=8 必须配 group_mean（两处一起改，缺一即错）
    from rlab.losses import compute_advantages
    from rlab.rollout import group_ok
    r = torch.tensor([1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    adv = compute_advantages(r, 8, cfg["adv_mode"])
    check("group_mean：8 条组 adv=r-mean（形状/除零安全）",
          adv.shape == (8,) and bool(((adv - (r - r.mean())).abs() < 1e-6).all()))
    check("group_mean 下非均匀组 group_ok 照常通过", bool(group_ok(adv)))
    check("group_mean 下全同组 group_ok 照常拒绝（零方差丢弃语义不变）",
          not bool(group_ok(compute_advantages(torch.ones(8), 8, cfg["adv_mode"]))))
    # BASE 隔离锁：retool_math preset 不得污染其他算法
    g = get_config("grpo", use_wandb=False)
    check("BASE 隔离：grpo 仍是 0.7/top_k=50/4 条/group_std",
          g["temperature"] == 0.7 and g["top_k"] == 50
          and g["num_pre_Q"] == 4 and g["adv_mode"] == "group_std")


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
          code_stats[0] == {"code_used": 2, "code_ok": 2, "trunc_final": 0}
          and code_stats[1] == {"code_used": 0, "code_ok": 0, "trunc_final": 0}
          and code_stats[2] == {"code_used": 1, "code_ok": 1, "trunc_final": 0})
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
    # （[[0]*5 是"长 5 的轨迹"，不是 token 值 5）。【2026-09-09 改逐样本口径】任一
    # 样本 len+plen 超预算即超——组均值口径会放行单条超长尖峰（padded batch 按最长算）
    check("超长检查：全长口径（含工具段）触发",
          retool_context_overlong([[0] * 5, [0] * 5], 3, 7))       # 逐样本 5+3=8 > 7
    check("超长检查：预算内不触发",
          not retool_context_overlong([[0] * 5, [0] * 5], 3, 10))  # 8 < 10
    check("超长检查：工具段撑爆预算（纯 assistant 口径会漏放行）",
          retool_context_overlong([[0] * 10, [0] * (10 + 6)], 3, 15))  # 19 > 15
    check("超长检查：逐样本口径抓住组均值漏放行的单条尖峰",
          retool_context_overlong([[0] * 18, [0] * 2], 2, 10))     # 20>10；旧组均值 (20+4)/2=12≤20 会放行
    check("超长检查：逐样本口径——短样本不救长样本",
          not retool_context_overlong([[0] * 5, [0] * 5], 2, 7))   # 7 ≤ 7 恰好放行

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


# --------------------------------- K. retool_math 审查修复回归锁（2026-09-09） ----
def test_retool_math_fixes():
    print("[K] retool_math 审查修复回归锁：math 打分域剥离 / retool_trunc / dev 剔除")
    from rlab.reward import total_reward_retool_math
    from rlab.health import window_check

    # 打分域统一：码内 boxed 不得覆盖最终答案（训练端剥离口径与 eval 端一致，
    # 修复前训练端 rfind 会取到码内 \boxed 导致同轨迹两端判定漂移）
    ans_good = "reasoning ... \\boxed{42}"
    ans_code_last = ans_good + "\n```python\nprint('\\boxed{99}')\n```"
    check("math 打分域：码内 boxed 不覆盖最终答案（与无码版同判）",
          total_reward_retool_math("42", ans_good)["acc"] == 1.0
          and total_reward_retool_math("42", ans_code_last)["acc"] == 1.0)
    ans_wrong = "reasoning \\boxed{42}\n```python\nprint(1)\n```"
    check("math 打分域：剥离不影响最终答案对错判定",
          total_reward_retool_math("42", ans_wrong)["acc"] == 1.0
          and total_reward_retool_math("43", ans_wrong)["acc"] == -1.0)

    # retool_trunc 签名：末段截断率 >20% 告警；无截断不误报
    hist = [{"acc": 0.3, "fmt": 0.9, "clen": 300.0, "code_rate": 0.5,
             "trunc_rate": 0.5} for _ in range(40)]
    codes = {c for c, _ in window_check(hist, retool=True)}
    check("retool 末段截断 >20% → retool_trunc", "retool_trunc" in codes)
    hist_ok = [{"acc": 0.3, "fmt": 0.9, "clen": 300.0, "code_rate": 0.5,
                "trunc_rate": 0.0} for _ in range(40)]
    codes = {c for c, _ in window_check(hist_ok, retool=True)}
    check("retool 末段未截断 → 不误报 retool_trunc", "retool_trunc" not in codes)
    codes = {c for c, _ in window_check(hist_ok, retool=False)}
    check("非 retool 不查 retool_trunc（旧 hist 无 trunc_rate 字段也兼容）",
          "retool_trunc" not in codes)

    # dev 剔除纯函数：训练池与 held-out 必须不相交（同池污染修复的契约锁）
    from rlab.data import dapo_exclude_dev
    pool = [{"Q": "q1", "A": "1"}, {"Q": "q2", "A": "2"}, {"Q": "q3", "A": "3"}]
    dev = [{"question": "q2", "answer": "2"}]
    kept = dapo_exclude_dev(pool, dev)
    check("dev 剔除：dev 题从训练池移除", [r["Q"] for r in kept] == ["q1", "q3"])
    check("dev 剔除：dev 为空 → 训练池不变", len(dapo_exclude_dev(pool, [])) == 3)
    check("dev 剔除：匹配按题面全文（部分匹配不误删）",
          [r["Q"] for r in dapo_exclude_dev(pool, [{"question": "q", "answer": "0"}])]
          == ["q1", "q2", "q3"])

    # 题目级动态采样（丢弃率 81% 根因修复的契约锁）
    from rlab.rollout import filter_question_pool
    QAs = [{"Q": f"q{i}", "A": "1"} for i in range(10)]
    q_stat = {f"q{i}": 2 for i in range(5)}          # 5 题连败到阈值
    cand, reset = filter_question_pool(QAs, q_stat, streak_max=2, floor=3)
    check("题目过滤：达到 streak 阈值的题被跳过",
          not reset and {c["Q"] for c in cand} == {f"q{i}" for i in range(5, 10)})
    q_stat2 = {f"q{i}": 2 for i in range(9)}          # 只剩 1 题 < floor=3
    cand2, reset2 = filter_question_pool(QAs, q_stat2, 2, 3)
    check("题目过滤：池子低于下限 → 全量重置（难题重新入场）",
          reset2 and len(cand2) == 10)
    cand3, reset3 = filter_question_pool(QAs, {}, 2, 3)
    check("题目过滤：无统计 → 全池可用", not reset3 and len(cand3) == 10)


# --------------------- L. QuestionScheduler（题目过滤死代码修复） ---------------------
def test_question_scheduler():
    print("[L] QuestionScheduler：队列走池 + 同题重试 + 拉黑 + floor 重置")
    from rlab.rollout import QuestionScheduler

    class _NoShuffle:   # 确定性 rng：shuffle 恒等（顺序可预期）
        def shuffle(self, x): pass

    QAs = [{"Q": f"q{i}", "A": "1"} for i in range(8)]
    sched = QuestionScheduler(QAs, streak_max=2, floor=4, rng=_NoShuffle())
    drawn = sched.draw(4)
    check("draw 顺序走池", [q["Q"] for q in drawn] == ["q0", "q1", "q2", "q3"])
    sched.report(drawn[0], "uniform")
    check("uniform 一次：streak=1 且插回队首（下一 draw 最先重试同题）",
          sched.q_stat["q0"] == 1 and sched.draw(1)[0]["Q"] == "q0")
    sched.report({"Q": "q0", "A": "1"}, "uniform")
    d2 = sched.draw(4)
    check("uniform 达标：拉黑（后续 draw 不再出现 q0）",
          sched.q_stat["q0"] == 2 and "q0" not in [q["Q"] for q in d2])
    q1 = {"Q": "q1", "A": "1"}
    sched.report(q1, "ok")
    check("ok 清零 streak", sched.q_stat["q1"] == 0)
    sched.report(q1, "uniform")
    sched.report(q1, "overlong")
    check("overlong 不计 streak（uniform 后超长不会误拉黑）", sched.q_stat["q1"] == 1)
    check("blacklisted_count 只算达标（拉黑）题", sched.blacklisted_count() == 1)
    # 拉黑到候选 < floor → draw 自动全量重置（难题重新入场）
    for i in range(2, 8):
        for _ in range(2):
            sched.report({"Q": f"q{i}", "A": "1"}, "uniform")
    d3 = sched.draw(4)
    check("候选低于 floor → draw 自动重置（q_stat 清空、全池重新入场）",
          len(d3) == 4 and sched.q_stat == {})


# ------------- M. collect_retool_group：多题并采 + 按题拆分（不连坐） -------------
def test_collect_retool_group_split():
    print("[M] collect_retool_group：多题并采（vLLM 并发=题数×n）+ 按题拆分")
    from rlab.rollout import collect_retool_group
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
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
        def __init__(self, rounds):
            self.rounds = rounds; self.r = 0
            self.batch_sizes = []
        def generate(self, prompts, sps, use_tqdm=False):
            self.batch_sizes.append(len(prompts))
            texts = self.rounds[self.r]; self.r += 1
            assert len(texts) == len(prompts)
            return [_O(t) for t in texts]

    cfg = get_config("retool", use_wandb=False)   # num_pre_Q=4 / max_rounds=3 / ctx 2200
    good72, good99 = fmt_answer("72"), fmt_answer("99")
    # 2 题 × 4 条：q0 混合结果（对/错各半 → 有梯度）；q1 无码全错（零方差 → uniform）
    # q1 文本刻意放长（第二段用小 max_context 验证超长按题隔离）
    r1 = [good72, good99, good72, good99,
          "filler " * 60, "filler " * 60, "filler " * 60, "filler " * 60]
    fg = FakeGen([r1])
    qs = [{"Q": f"q{i}", "A": "72"} for i in range(2)]
    prompts_text = [f"prompt{i}" for i in range(2)]
    prompt_ids = tok(prompts_text, return_tensors="pt", padding=True,
                     add_special_tokens=False)["input_ids"]   # padding_side 已设 left
    plen = prompt_ids.shape[1]
    sps = [object() for _ in range(8)]   # 每轨迹独立请求参数（FakeGen 不检查内容）
    gl_calls = []

    def fake_gl(merged, plen_):
        gl_calls.append(merged.shape[0])
        return torch.zeros(merged.shape[0], merged.shape[1] - plen_)

    results = collect_retool_group(fg, tok, cfg, fake_gl, qs, prompts_text,
                                   prompt_ids, plen, sps, steps_elapsed=0)
    check("一次并采 2 题全部轨迹（vLLM 单轮 batch = 题数×n = 8）",
          fg.batch_sizes == [8])
    check("返回 per-question 结果（每题一项）", len(results) == 2)
    check("q0 混合 → ok；q1 全错 → uniform（零方差按题判定）",
          results[0]["status"] == "ok" and results[1]["status"] == "uniform")
    ok = results[0]
    check("ok 项：merged 4 行、adv/acc/fmt/clen/trunc 形状正确",
          ok["merged"].shape[0] == 4 and ok["adv"].shape[0] == 4
          and len(ok["clen"]) == 4 and len(ok["trunc"]) == 4
          and ok["merged"].shape[1] == plen + max(ok["clen"]))
    check("ok 项：gen_logps 每题独立一次（uniform 题不算，省 GPU0 前向）",
          gl_calls == [4])
    check("ok 项：plen 记入（上传 meta 用）", ok["plen"] == plen)

    # 按题拆分·超长不连坐：q1 轨迹更长，压低 max_context → q1 overlong、q0 照常 ok
    # （旧整批口径 = 整组丢弃，多题并采下会放大丢弃损失）
    q0_comp_max = max(len(tids(t)) for t in r1[:4])
    q1_comp_min = min(len(tids(t)) for t in r1[4:])
    check("前提：q1 轨迹 token 数确比 q0 长", q1_comp_min > q0_comp_max)
    cfg2 = dict(cfg)
    cfg2["max_context_tokens"] = plen + q0_comp_max   # 严格 > 判超长：q0 恰好放行
    results2 = collect_retool_group(FakeGen([r1]), tok, cfg2, fake_gl, qs,
                                    prompts_text, prompt_ids, plen, sps, steps_elapsed=0)
    check("超长按题隔离：q0 ok / q1 overlong", 
          results2[0]["status"] == "ok" and results2[1]["status"] == "overlong")
    check("丢弃项不含上传字段（uniform/overlong 零上传成本）",
          "merged" not in results[1] and "merged" not in results2[1])


# ------------- N. 离线难度预探测过滤 + 探针聚合（2026-09-10） -------------
def test_difficulty_filter():
    print("[N1] filter_qas_by_difficulty：band 过滤口径（丢弃率 81% 的静态出清层）")
    from rlab.data import filter_qas_by_difficulty, load_difficulty_table
    qas = [{"Q": f"q{i}", "A": "72"} for i in range(6)]
    # k=4：q0 全错(0)、q1 半对(2)、q2 全对(4)、q3 一条对(1)、q4 三条对(3)、q5 不在表
    table = {"q0": {"k": 4, "n_correct": 0}, "q1": {"k": 4, "n_correct": 2},
             "q2": {"k": 4, "n_correct": 4}, "q3": {"k": 4, "n_correct": 1},
             "q4": {"k": 4, "n_correct": 3}}
    kept, st = filter_qas_by_difficulty(qas, table)
    check("默认 band (0,1)：全错/全对/缺失题全部出清，保留 1<=nc<=k-1",
          [x["Q"] for x in kept] == ["q1", "q3", "q4"])
    check("stats 口径齐全（p_zero/p_one/missing/kept/total）",
          st == {"kept": 3, "p_zero": 1, "p_one": 1, "band_out": 0,
                 "missing": 1, "total": 6})
    kept2, st2 = filter_qas_by_difficulty(qas, table, lo=0.0, hi=0.5)
    check("窄 band (0,0.5)：通过率 0.5/0.75 的题都落在开区间外（band_out=2）",
          [x["Q"] for x in kept2] == ["q3"] and st2["band_out"] == 2
          and st2["p_one"] == 1)
    kept3, st3 = filter_qas_by_difficulty(qas, table, lo=0.0, hi=0.0)
    check("空 band：结果为空但 missing/总数口径不变（供 gen_worker fail-fast）",
          kept3 == [] and st3["total"] == 6 and st3["missing"] == 1)

    # 表加载：坏行/非法行静默跳过（探针逐题追加写，崩溃可能留截断行）
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"Q": "q0", "k": 4, "n_correct": 1}\n')
            f.write('{"Q": "bad1", "k": 4}\n')            # 缺 n_correct
            f.write('{"Q": "bad2", "k": 0, "n_correct": 0}\n')   # k<=0
            f.write('{"Q": "bad3", "k": 4, "n_correct": 5}\n')   # nc>k
            f.write('{"Q": "bad4", "k": 4, "n_correct": -1}\n')  # nc<0
            f.write('{"Q": "q1", "k": 8, "n_correct": 3, "fmt_rate": 0.9}\n')  # 合法
            f.write('{"Q": "trunc", "k": 4, "n_cor')      # 截断行
        tbl = load_difficulty_table(p)
    check("表加载：合法行收录、非法/截断行静默跳过",
          set(tbl) == {"q0", "q1"} and tbl["q1"]["k"] == 8)
    print()


def test_probe_aggregate():
    print("[N2] probe_difficulty.aggregate_rows/summarize：逐轨迹 -> 逐题统计")
    from rlab.probe_difficulty import aggregate_rows, summarize
    rows = [
        {"Q": "a", "A": "7", "acc": 1, "fmt": 1, "trunc": 0, "clen": 100, "code_ok": 1},
        {"Q": "a", "A": "7", "acc": -1, "fmt": 1, "trunc": 1, "clen": 200, "code_ok": 0},
        {"Q": "b", "A": "9", "acc": -1, "fmt": -1, "trunc": 0, "clen": 50, "code_ok": 0},
        {"Q": "b", "A": "9", "acc": -1, "fmt": -1, "trunc": 1, "clen": 60, "code_ok": 2},
    ]
    per_q = aggregate_rows(rows)
    check("题顺序 = 首次出现顺序", [r["Q"] for r in per_q] == ["a", "b"])
    a, b = per_q
    check("a: k=2 / n_correct=1 / pass_rate=0.5 / fmt_rate=1.0 / trunc_rate=0.5",
          a["k"] == 2 and a["n_correct"] == 1 and a["pass_rate"] == 0.5
          and a["fmt_rate"] == 1.0 and a["trunc_rate"] == 0.5)
    check("b: fmt_rate=0（无 boxed）→ 协议失败签名可见", b["fmt_rate"] == 0.0)
    check("avg 口径：a.avg_clen=150 / b.avg_code_ok=1.0",
          a["avg_clen"] == 150 and b["avg_code_ok"] == 1.0)
    s = summarize(rows, per_q)
    check("summarize：含难度分布与判别统计（无 boxed 率 = 2/4 = 50%）",
          "难度分布" in s and "50.0%" in s and isinstance(s, str))
    print()


# ------------- O. 沙箱加固：auto-print + 工具输出消毒（2026-09-10，对齐 05-retool） -------------
def test_sandbox_hardening():
    print("[O1] auto_print：没写 print 的代码末行纯表达式自动补 print（verl 官方技巧）")
    from rlab.sandbox import auto_print
    c, p = auto_print("1+1")
    check("纯表达式 1+1 → print(1+1)", c == "print(1+1)" and p is True)
    c, p = auto_print("x = 1\nx")
    check("多行代码末行是表达式 x → print(x)，赋值行不动", c == "x = 1\nprint(x)" and p)
    check("已含 print 的代码原样返回", auto_print("x=1\nprint(x)") == ("x=1\nprint(x)", False))
    check("末行是赋值 → 不包裹（打印无意义对象）",
          auto_print("x = compute()") == ("x = compute()", False))
    check("末行是块开头/控制流 → 不包裹（防语法错误）",
          auto_print("for i in range(3):") == ("for i in range(3):", False)
          and auto_print("if x > 0:") == ("if x > 0:", False))
    check("空代码 → 原样返回", auto_print("   ") == ("   ", False))
    check("多行+末行表达式（真实形态）：调用代码补在最后一行",
          auto_print("import math\nmath.sqrt(2)")[1] is True)

    ok = run_code("1+1", timeout=5)
    check("run_code 集成：忘 print 的表达式代码能拿到 stdout（display='2'）",
          ok["ok"] and ok["display"] == "2" and ok["auto_printed"] == 1)
    ok2 = run_code("x = 2 + 3\nprint(x)", timeout=5)
    check("run_code 集成：已有 print 不触发（auto_printed=0）",
          ok2["ok"] and ok2["auto_printed"] == 0)

    print("[O2] sanitize_tool_text：特殊 token / 工具标记字面量剥除（注入封堵）")
    from rlab.protocol import sanitize_tool_text
    im_end = chr(60) + "|im_end|" + chr(62)   # 零标签字面量：<|im_end|> 不手写裸字节
    check("特殊 token 字面量剥除", im_end not in sanitize_tool_text("res " + im_end + " val"))
    check("工具标记字面量剥除（防伪造嵌套边界）",
          TOOL_START not in sanitize_tool_text("x" + TOOL_START + "y")
          and TOOL_END not in sanitize_tool_text("x" + TOOL_END + "y"))
    check("正常输出原样通过", sanitize_tool_text("42\n[1, 2, 3]") == "42\n[1, 2, 3]")
    check("混合：两种注入同现时全部剥除",
          sanitize_tool_text("a" + im_end + TOOL_END + "b") == "ab")
    print()


# --------------------------------- J. 静态未定义名检查（运行时 NameError 防线） ----
def test_chat_template_kwargs():
    print("[P] chat_template_kwargs 透传：Qwen3.5 系需 enable_thinking=false"
          "（4B 探针实测不关 thinking：截断 98.9%/无 boxed 99.2%，协议失败非能力失败）")
    from rlab.rollout import build_prompt

    seen = {}

    class FakeTok:
        """记录 apply_chat_template 收到 kwargs 的最小 stub（模板行为在真机验收）。"""

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            seen["msgs"], seen["kw"] = msgs, kw
            return "PROMPT"

    tok = FakeTok()
    p = build_prompt("Q1", "SYS", tok)
    check("默认不传附加 kwargs（Qwen2.5 路径零变化）",
          seen["kw"] == {} and p == "PROMPT")
    check("消息结构 system+user",
          seen["msgs"] == [{"role": "system", "content": "SYS"},
                           {"role": "user", "content": "Q1"}])
    build_prompt("Q1", "SYS", tok, {"enable_thinking": False})
    check("kwargs 透传模板上下文", seen["kw"] == {"enable_thinking": False})
    cfg = get_config("retool_math", use_wandb=False)
    check("retool_math preset 带思考开关（2026-09-11 eval 全灭事故后收进 preset "
          "单点同源，训练/eval 共用；旧契约'preset 必须 None'已随事故作废）",
          cfg["chat_template_kwargs"] == {"enable_thinking": False})
    check("GSM8K 家族（grpo）仍无该键（Qwen2.5 行为零变化）",
          get_config("grpo", use_wandb=False)["chat_template_kwargs"] is None)
    # CLI 以 JSON 字符串进 overrides（train.py / probe_difficulty.py 同一解析形态）
    check("CLI JSON 反序列化形态",
          json.loads('{"enable_thinking": false}') == {"enable_thinking": False})


def test_split_load_remap():
    print("[Q] 分裂加载键名映射：多模态 vLLM + 纯文本 torch（Qwen3.5 实锤）")
    from rlab.sync import remap_text_to_multimodal, sync_weights_into_vllm

    sd = [("model.embed_tokens.weight", "t0"),
          ("model.layers.0.self_attn.q_proj.weight", "t1"),
          ("model.norm.weight", "t2"),
          ("lm_head.weight", "t3")]
    out = dict(remap_text_to_multimodal(sd))
    check("model.* -> model.language_model.*（HF ForConditionalGeneration 布局，"
          "vLLM AutoWeightsLoader+hf_to_vllm_mapper 已核实吃这个形态）",
          out["model.language_model.embed_tokens.weight"] == "t0"
          and out["model.language_model.layers.0.self_attn.q_proj.weight"] == "t1"
          and out["model.language_model.norm.weight"] == "t2")
    check("tied lm_head 丢弃（Qwen3.5-4B tie=True，原 checkpoint 无此键，"
          "torch 侧是共享张量重复键）", "lm_head.weight" not in out)
    out2 = dict(remap_text_to_multimodal(sd, drop_tied_lm_head=False))
    check("非 tied 目标可保留 lm_head（参数化退路）", out2.get("lm_head.weight") == "t3")
    check("张量对象原样搬运（不 copy 数据）",
          all(isinstance(t, str) for t in out.values()))
    try:
        remap_text_to_multimodal([("visual.weight", "t")])
        check("未知键名 fail-fast", False)
    except KeyError:
        check("未知键名 fail-fast（静默漏同步=生成端旧权重）", True)
    cfg = get_config("retool_math", use_wandb=False)
    check("BASE 默认 vllm_model_path=None（单 checkpoint 路径零变化）",
          cfg["vllm_model_path"] is None)
    check("BASE 默认 zero_stage=0（3B 路径零变化；4B 用 --zero_stage 2 offload）",
          cfg["zero_stage"] == 0)
    from rlab.config import ds_config as _ds
    check("stage 2 配置带 offload_optimizer(cpu)，stage 0 不带",
          "offload_optimizer" in _ds({**cfg, "zero_stage": 2})["zero_optimization"]
          and "offload_optimizer"
          not in _ds({**cfg, "zero_stage": 0})["zero_optimization"])


def test_chunked_logps():
    print("[R] 分块 logps：forward_per_token_logps == 全量前向（4B logits 峰 OOM 修复）")
    from rlab.losses import forward_per_token_logps
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=64, n_layer=2,
                     n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg).eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("The capital of France is Paris and Rome is old",
              return_tensors="pt").input_ids
    with torch.inference_mode():
        ref = get_per_token_logps(model(ids).logits[:, :-1, :], ids[:, 1:])
        chunked = forward_per_token_logps(model, ids, seq_chunk=3, batch_chunk=1)
    check("分块 == 全量（seq_chunk=3 + batch_chunk=1 覆盖双维非对齐边界）",
          torch.allclose(ref, chunked, atol=1e-5))
    model.config.use_cache = False
    model.train()   # dropout 已置 0，train 模式确定性与 eval 一致
    out = forward_per_token_logps(model, ids, seq_chunk=3, use_checkpoint=True)
    check("checkpoint 分块 == 全量（grad 路径数值等价）",
          torch.allclose(ref, out.detach(), atol=1e-5))
    out.sum().backward()
    # 【2026-09-11 放开 batch_chunk】批量前向与逐行严格等价（多行输入的跨行独立性）
    ids2 = torch.cat([ids, ids], dim=0)          # 2 行
    with torch.inference_mode():
        ref2 = get_per_token_logps(model(ids2).logits[:, :-1, :], ids2[:, 1:])
        bc2 = forward_per_token_logps(model, ids2, seq_chunk=3, batch_chunk=2)
        bc_mixed = forward_per_token_logps(model, ids2, seq_chunk=3, batch_chunk=1)
    check("batch_chunk=2（整块）== batch_chunk=1（逐行）== 全量前向（三路同值）",
          torch.allclose(ref2, bc2, atol=1e-5) and torch.allclose(ref2, bc_mixed, atol=1e-5))
    check("checkpoint 反向梯度可达（lm_head/embed 均有 grad）",
          model.lm_head.weight.grad is not None
          and model.transformer.wte.weight.grad is not None)


def test_alloc_conf_ipc():
    """【S】expandable_segments 与 CUDA IPC 互斥的环境契约（源码级检查）。

    4B 首跑死在第 16 步权重同步：训练进程在 expandable_segments:True 下分配的
    state_dict 张量过 mp.Queue，跨进程共享走 pidfd_open（容器内核不支持）->
    生成端反序列化 RuntimeError。契约：train.py 顶层强制 False（两个变量名），
    gen_worker 入口改回 True（GPU0 碎片治理仍需要），脚本全局 export 不动。
    环境行为无法在 CPU 测试进程验证，退化为源码契约断言。"""
    print("[S] allocator env 契约：expandable_segments 与 CUDA IPC 互斥的分层配置")
    with open("rlab/train.py", encoding="utf-8") as f:
        train_src = f.read()
    with open("rlab/rollout.py", encoding="utf-8") as f:
        rollout_src = f.read()
    with open("rlab/run_gsm8k.sh", encoding="utf-8") as f:
        sh_src = f.read()
    check("train.py 顶层对两个 allocator 变量名强制 expandable_segments:False",
          'for _alloc_k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):' in train_src
          and 'os.environ[_alloc_k] = "expandable_segments:False"' in train_src)
    check("gen_worker 入口改回 True（生成端碎片治理仍需要；且在 set_device 前生效）",
          'os.environ[_alloc_k] = "expandable_segments:True"' in rollout_src
          and rollout_src.index('expandable_segments:True')
          < rollout_src.index("torch.cuda.set_device"))
    check("run_gsm8k.sh 全局 export 保留（ref_server 独立进程只能靠 shell 环境拿到 True）",
          "expandable_segments:True" in sh_src)
    check("train.py 的 False 覆盖发生在 import torch 之前（allocator 首次解析 env 前生效）",
          train_src.index('expandable_segments:False') < train_src.index("import torch"))
    check("gen_worker 收侧无需特殊处理（经典 cudaIpcMemHandle 路径不用 pidfd，"
          "关键只在发送端张量在普通段分配）", True)


def test_attn_impl():
    """【T】attn_implementation 提速开关：三处 torch 加载点统一接线（FA2 提速）。"""
    print("[T] attn_implementation：三处 torch 加载点统一接线（FA2 消 T² math 回退）")
    cfg = get_config("retool_math")
    check("BASE 默认 sdpa（3B 历史口径零变化）", cfg["attn_implementation"] == "sdpa")
    with open("rlab/train.py", encoding="utf-8") as f:
        train_src = f.read()
    with open("rlab/rollout.py", encoding="utf-8") as f:
        rollout_src = f.read()
    with open("rlab/ref_server.py", encoding="utf-8") as f:
        ref_src = f.read()
    with open("rlab/run_gsm8k.sh", encoding="utf-8") as f:
        sh_src = f.read()
    check("train.py 加载点走 cfg（无硬编码 sdpa）",
          '_attn_implementation=cfg.get("attn_implementation", "sdpa")' in train_src)
    check("rollout.py gen 副本加载点走 cfg",
          '_attn_implementation=cfg.get("attn_implementation", "sdpa")' in rollout_src)
    check("ref_server.py FA2 档位自动降 bf16（FA2 不支持 fp32）",
          'torch.bfloat16 if attn_implementation == "flash_attention_2"' in ref_src)
    check("run_gsm8k.sh 把 ATTN_IMPL 注入 ref_server 与 train 两处（手动传参可覆盖："
          "注入 flag 在 \"$@\" 之前，argparse 后者胜）",
          sh_src.count('--attn_implementation "$ATTN_IMPL"') == 2
          and '--attn_implementation "$ATTN_IMPL" "$@"' in sh_src)
    check("train.py CLI choices 含 flash_attention_2",
          '"flash_attention_2"' in train_src and "--attn_implementation" in train_src)


def test_materialize_mm():
    print("[U] 多模态壳物化：纯文本 checkpoint -> vLLM 可评的多模态格式"
          "（A2 在 eval 路复现：vLLM 拒 Qwen3_5TextConfig）")
    from rlab.materialize_mm_ckpt import merge_text_into_mm

    t_emb = torch.randn(4, 3)
    t_norm = torch.randn(4)
    t_vis = torch.randn(2, 2)
    text_sd = {"model.embed_tokens.weight": t_emb, "model.norm.weight": t_norm,
               "lm_head.weight": torch.randn(4, 3)}   # tied 重复键
    mm_sd = {"model.language_model.embed_tokens.weight": torch.randn(4, 3),
             "model.language_model.norm.weight": torch.randn(4, dtype=torch.bfloat16),
             "model.visual.patch_embed.weight": t_vis}
    merged, stats = merge_text_into_mm(text_sd, mm_sd)
    check("语言键被文本张量替换（与训练端同步同一映射表，杜绝两套映射漂移）",
          merged["model.language_model.embed_tokens.weight"] is t_emb)
    check("dtype cast 键数值守恒（bf16 舍入容差内；t.to() 产生新对象按值验）",
          torch.allclose(merged["model.language_model.norm.weight"].float(),
                         t_norm, rtol=0.05, atol=0.05))
    check("tied lm_head 丢弃（与 remap_text_to_multimodal 行为一致）",
          "lm_head.weight" not in merged and "model.lm_head.weight" not in merged)
    check("视觉键原样保留（骨架非语言键不动）",
          merged["model.visual.patch_embed.weight"] is t_vis)
    check("dtype 对齐骨架（文本 fp32 master -> 骨架 bf16）",
          merged["model.language_model.norm.weight"].dtype == torch.bfloat16)
    check("stats 计数正确（替换 2 / 保留 1）",
          stats["text_substituted"] == 2 and stats["base_kept"] == 1)
    # 布局漂移双向 fail-fast：骨架键缺文本对应 / 文本键未命中骨架
    try:
        merge_text_into_mm({"model.embed_tokens.weight": t_emb}, mm_sd)
        check("骨架键缺文本对应 fail-fast", False)
    except KeyError:
        check("骨架键缺文本对应 fail-fast（禁止 base 旧权重静默补位）", True)
    try:
        merge_text_into_mm({**text_sd, "model.dummy.weight": t_norm}, mm_sd)
        check("文本键未命中骨架 fail-fast", False)
    except KeyError:
        check("文本键未命中骨架 fail-fast（新键 = 布局漂移）", True)
    # 纯视觉骨架 + 文本 ckpt = 语言键无处落位，同样 fail-fast（不许静默丢层）
    try:
        merge_text_into_mm({"model.norm.weight": t_norm}, {"model.visual.x": t_vis})
        check("纯视觉骨架 fail-fast", False)
    except KeyError:
        check("纯视觉骨架 fail-fast（语言键无处落位=配置错误）", True)


def test_eval_spawn_guard():
    print("[V] eval spawn 递归引爆防护：vLLM V1 spawn 子进程重执行 eval_vllm_one.py "
          "顶层 LLM() -> _check_not_importing_main（2026-09-11 4B eval 实测）")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "eval_vllm_one.py"), encoding="utf-8").read()
    guard = 'os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")'
    check("eval_vllm_one.py 顶部设 VLLM_ENABLE_V1_MULTIPROCESSING=0"
          "（与 run_gsm8k.sh 训练端同款，进程内引擎不走 spawn）",
          guard in src)
    check("守卫在 vLLM import 之前（spawn 发生在 LLM() 初始化，env 须先于其生效）",
          src.index(guard) < src.index("from vllm import"))


def test_eval_thinking_switch():
    print("[W] Qwen3.5 思考开关单点同源：eval 全灭事故（base/step200 同为 2%）——"
          "eval 未传 enable_thinking=False，<think> 烧穿预算，炸协议不炸权重")
    cfg_m = get_config("retool_math", use_wandb=False)
    check("retool_math preset 带 chat_template_kwargs={'enable_thinking': False}"
          "（训练/eval 单点同源，CLI 仍可覆盖）",
          cfg_m.get("chat_template_kwargs") == {"enable_thinking": False})
    check("BASE 保持 None（Qwen2.5 家族零变化）",
          get_config("grpo", use_wandb=False).get("chat_template_kwargs") is None)
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "eval_vllm_one.py"), encoding="utf-8").read()
    # 【二次修复】包裹形态 chat_template_kwargs={...} 会被 pod 上 transformers 版本
    # 静默忽略（实测：直接 kwarg -> <think>\n\n</think>，包裹 -> <think>\n），
    # eval 必须复用训练端 build_prompt 函数本身，而非自己拼 apply_chat_template
    check("eval_vllm_one.py 复用训练端 build_prompt（调用形态单点同源，"
          "包裹形态 chat_template_kwargs= 会被 transformers 静默忽略——二次全灭教训）",
          "from rlab.rollout import build_prompt" in src
          and "_build_prompt(item[\"Q\"], system_prompt, tokenizer, _ctkw)" in src)
    check("eval_vllm_one.py 不再自带 apply_chat_template 调用（杜绝形态分叉复发）",
          "apply_chat_template" not in src)
    check("eval_vllm_one.py 有 enable_thinking 未生效的 fail-fast 告警",
          "模板未响应 enable_thinking=False" in src)


def test_overlong_ref_and_opt_cli():
    print("[X] overlong 参考系修复 + 优化超参 CLI：retool 多轮总预算 ≠ 单轮 max_gen_tokens")
    from rlab.reward import overlong_ref_tokens, overlong_penalty, total_reward_math
    # 参考系：retool 家族 = max_rounds × round_gen_tokens（CLI 覆盖后同步生效）
    cfg_rm = get_config("retool_math", use_wandb=False, round_gen_tokens=3072)
    check("retool_math 参考系 = max_rounds × round_gen_tokens = 9216（非单轮 8192）",
          overlong_ref_tokens(cfg_rm) == 3 * 3072)
    check("preset 默认（round_gen_tokens=1024）参考系 = 3072",
          overlong_ref_tokens(get_config("retool_math", use_wandb=False)) == 3 * 1024)
    check("单轮路径（grpo）参考系仍 = max_gen_tokens",
          overlong_ref_tokens(get_config("grpo", use_wandb=False))
          == get_config("grpo", use_wandb=False)["max_gen_tokens"])
    # 行为：合法用满预算（2 轮满 3072 + 末轮 3008 = 9152，落在 trigger 9152 内）
    # 做对的轨迹不该被误伤；旧口径（8192）下同一轨迹被扣满 → +1 抹成 0 / -1 压成 -2
    _full_legit = 2 * 3072 + 3008
    r_now = total_reward_math("72", "answer is \\boxed{72}",
                              completion_len=_full_legit,
                              max_gen_tokens=overlong_ref_tokens(cfg_rm),
                              overlong_shaping=True)["reward"]
    r_old = total_reward_math("72", "answer is \\boxed{72}",
                              completion_len=_full_legit, max_gen_tokens=8192,
                              overlong_shaping=True)["reward"]
    check("合法满预算(9152)+答对：新参考系不罚（+1），旧参考系被抹平（0）",
          abs(r_now - 1.0) < 1e-6 and abs(r_old - 0.0) < 1e-6)
    r_wrong_old = total_reward_math("72", "answer is \\boxed{99}",
                                    completion_len=_full_legit, max_gen_tokens=8192,
                                    overlong_shaping=True)["reward"]
    check("旧参考系还会加倍惩罚答错（-1 → -2），新参考系下为 -1",
          abs(r_wrong_old - (-2.0)) < 1e-6
          and abs(total_reward_math("72", "answer is \\boxed{99}",
                                    completion_len=_full_legit,
                                    max_gen_tokens=overlong_ref_tokens(cfg_rm),
                                    overlong_shaping=True)["reward"] - (-1.0)) < 1e-6)
    check("真超预算才线性扣分（trigger 9152 + 32 → 0.5）",
          abs(overlong_penalty(9152 + 32, 9216, 64) - 0.5) < 1e-6)
    # CLI 入口：新三件套被 train.py 接收并落到 overrides
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()
    for flag, key in (('"--lr"', 'overrides["lr"] = args.lr'),
                      ('"--beta"', 'overrides["beta"] = args.beta'),
                      ('"--overlong_shaping"', 'overrides["overlong_shaping"] = True')):
        check(f"train.py CLI {flag} 存在且映射到 {key}",
              flag in src and key in src)
    # 接线：rollout retool 打分路径与探针都用同一参考系函数
    ro = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "rollout.py"), encoding="utf-8").read()
    pb = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "probe_difficulty.py"), encoding="utf-8").read()
    check("rollout retool 打分用 overlong_ref_tokens（不再传单轮 max_gen_tokens）",
          "_ol_ref = overlong_ref_tokens(cfg)" in ro
          and 'max_gen_tokens=cfg["max_gen_tokens"]' not in ro.split("def retool_score_flat")[1].split("def ")[0])
    check("probe_difficulty 与训练同口径（overlong_ref_tokens）",
          "max_gen_tokens=overlong_ref_tokens(cfg)" in pb)
    # 【2026-09-11 审查】overlong 在 retool 当前预算几何下不可达（clen 被
    # max_context_tokens 封顶 ~7800 < trigger 9152），注释必须写明防误导
    cfg_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "config.py"), encoding="utf-8").read()
    check("config 注释声明 overlong 当前不可达（防后人误以为它在抑制截断）",
          "当前配置下不可达" in cfg_src)


def test_grad_clip_and_run_info():
    print("[Y] 梯度裁剪入口 + checkpoint 自证配方（10 小时长跑的provenance）")
    from rlab.config import ds_config
    cfg_g = get_config("retool_math", use_wandb=False)
    check("ds_config 默认 gradient_clipping=0.0（历史口径零变化）",
          ds_config(cfg_g)["gradient_clipping"] == 0.0)
    check("设 gradient_clipping=1.0 能进 DS 配置",
          ds_config(get_config("retool_math", use_wandb=False,
                               gradient_clipping=1.0))["gradient_clipping"] == 1.0)
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()
    check("train.py CLI --grad_clip 映射到 overrides（float 透传）",
          '"--grad_clip"' in src and 'overrides["gradient_clipping"] = args.grad_clip' in src)
    check("run_info 落盘完整 cfg（lr/beta/grad_clip/GAS 可回溯）",
          '"config": {k: v for k, v in cfg.items()}' in src
          and "default=str" in src)
    # get_config 拒绝未知键的契约仍成立（新键必须先在 BASE 注册）
    try:
        get_config("retool_math", use_wandb=False, gradient_clipping=1.0)
        ok = True
    except KeyError:
        ok = False
    check("gradient_clipping 已在 BASE 注册（未知键 fail-fast 契约未被绕过）", ok)


def test_fwd_batch_chunk():
    print("[Z] 分块前向粒度放开：三处同口径（train/gen/ref_server），默认 1 不变历史口径")
    import inspect
    import os as _os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 默认值：不动 3B 与 200 步 run 的口径
    _os.environ.pop("FWD_BATCH_CHUNK", None)
    check("preset 默认 fwd_batch_chunk=1（历史口径零变化）",
          get_config("retool_math", use_wandb=False)["fwd_batch_chunk"] == 1)
    # 环境变量驱动（ref_server 读不到 cfg，靠它保持全链路一致）
    _os.environ["FWD_BATCH_CHUNK"] = "2"
    try:
        check("FWD_BATCH_CHUNK 环境变量可驱动（供 ref_server 同口径）",
              get_config("retool_math", use_wandb=False)["fwd_batch_chunk"] == 2)
        check("显式 override 优先于环境变量",
              get_config("retool_math", use_wandb=False,
                         fwd_batch_chunk=4)["fwd_batch_chunk"] == 4)
    finally:
        _os.environ.pop("FWD_BATCH_CHUNK", None)
    # 四处接线
    tr = open(os.path.join(root, "rlab", "train.py"), encoding="utf-8").read()
    ro = open(os.path.join(root, "rlab", "rollout.py"), encoding="utf-8").read()
    rs = open(os.path.join(root, "rlab", "ref_server.py"), encoding="utf-8").read()
    sh = open(os.path.join(root, "rlab", "run_gsm8k.sh"), encoding="utf-8").read()
    check("train.py 两条训练路径都用 _fbc（无残留 batch_chunk=1 硬编码）",
          "batch_chunk=_fbc" in tr and "batch_chunk=1" not in tr
          and '"--fwd_batch_chunk"' in tr
          and 'overrides["fwd_batch_chunk"] = args.fwd_batch_chunk' in tr)
    check("rollout gen_logps 副本读 cfg.fwd_batch_chunk",
          'batch_chunk=max(1, int(cfg.get("fwd_batch_chunk", 1) or 1))' in ro)
    check("ref_server 有 --batch_chunk 并接到两处调用点",
          '"--batch_chunk"' in rs and rs.count("batch_chunk=batch_chunk") == 2)
    check("run_gsm8k.sh 用 FWD_BATCH_CHUNK 驱动三处（export + 传 ref_server）",
          "export FWD_BATCH_CHUNK=${FWD_BATCH_CHUNK:-1}" in sh
          and '--batch_chunk "$FWD_BATCH_CHUNK"' in sh)
    # e2e 测试按位置传 run_server(path, port, mode, beta, grad_accum, device, attn)——
    # 新参数必须追加在末尾，插在中间会静默错位（device 收到 "cpu" 之类的字符串）
    from rlab.ref_server import run_server
    params = list(inspect.signature(run_server).parameters)
    check("run_server 前 6 个位置参数保持不变（e2e 位置传参契约）",
          params[:6] == ["model_path", "port", "mode", "beta", "grad_accum", "device"]
          and params[-1] == "batch_chunk")


def test_vllm_gen_logps():
    print("[AA] 减法①：gen_logps 改走 vLLM 逐轮采样 logprobs（提取/拼接/对拍闸门）")
    from rlab.rollout import (LogpsVerifier, collect_retool_group, gen_logps_from_segs,
                              multi_turn_rollout_group, sampled_logps_from_output)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def tids(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    class _LP:
        def __init__(self, v): self.logprob = v

    class _C:
        def __init__(self, text, with_lp=True):
            self.text = text
            self.token_ids = tids(text)
            self.logprobs = ([{i: _LP(-0.1 * (k + 1))}
                              for k, i in enumerate(self.token_ids)] if with_lp else None)

    class _O:
        def __init__(self, c): self.outputs = [c]

    # ---- 提取：happy path + 三种错位 fail-fast ----
    ids = tids("hello")
    out = _C("hello")
    got = sampled_logps_from_output(out, ids)
    check("提取被采样 token 的 logprob（逐位置对齐）",
          len(got) == len(ids) and abs(got[0] - (-0.1)) < 1e-9
          and abs(got[-1] - (-0.1 * len(ids))) < 1e-9)

    def _expect_raise(fn, exc, why):
        try:
            fn()
        except exc:
            return True
        except Exception as e:
            print(f"  !! 期望 {exc.__name__}，实得 {type(e).__name__}: {e}")
            return False
        print(f"  !! 期望 {exc.__name__}，实得无异常")
        return False

    check("无 logprobs（未开 logprobs=0）→ RuntimeError",
          _expect_raise(lambda: sampled_logps_from_output(_C("hi", with_lp=False),
                                                          tids("hi")), RuntimeError, "no-lp"))
    check("logprobs 条数 ≠ token 数 → ValueError",
          _expect_raise(lambda: sampled_logps_from_output(out, ids + [1]),
                        ValueError, "len-mismatch"))
    bad = _C("hi")
    bad.token_ids = bad.token_ids + [999]
    check("被采样 token 不在该位置 logprobs 里 → ValueError",
          _expect_raise(lambda: sampled_logps_from_output(bad, bad.token_ids),
                        ValueError, "missing-key"))

    # ---- 拼接：assistant 段取真实 logps，工具段置 0，右补 0 ----
    segs2 = [[{"kind": "assistant", "ids": [1, 2], "logps": [-0.5, -1.5]},
              {"kind": "tool", "ids": [3, 4, 5]}],
             [{"kind": "assistant", "ids": [6], "logps": [-2.0]}]]
    t = gen_logps_from_segs(segs2)
    check("拼接：(B,T) 形状 + 工具段填 0 + pad 填 0 + bf16（与 torch 路同 dtype）",
          tuple(t.shape) == (2, 5) and t.dtype == torch.bfloat16
          and [round(float(x), 3) for x in t[0]] == [-0.5, -1.5, 0.0, 0.0, 0.0]
          and [round(float(x), 3) for x in t[1]] == [-2.0, 0.0, 0.0, 0.0, 0.0])
    check("assistant 段缺 logps → ValueError（采样时没开 collect_logps 的签名）",
          _expect_raise(lambda: gen_logps_from_segs([[{"kind": "assistant", "ids": [1]}]]),
                        ValueError, "no-logps"))

    # ---- 对拍闸门：预算内才要求 torch 重算，验完关闸（副本才可安全释放）----
    fin = []
    v = LogpsVerifier(2, on_finish=lambda: fin.append(1))
    gv = torch.zeros(1, 3, dtype=torch.bfloat16)
    gt = torch.zeros(1, 3, dtype=torch.bfloat16)
    m = torch.ones(1, 3)
    gt[0, 1] = 0.25
    v(gv, gt, m)
    check("对拍器：第 1 组后 wants 仍为真、记录最大差",
          v.wants() and abs(v.max_diff - 0.25) < 1e-6 and fin == [])
    v(gv, gt, m)
    check("对拍器：预算用尽后 wants 转假 + on_finish 恰好触发一次（释放副本）",
          (not v.wants()) and fin == [1])

    # ---- collect_retool_group 接线：vLLM 路不再调用 torch 重算 ----
    cfg = get_config("retool", use_wandb=False)
    cfg["vllm_gen_logps"] = True
    class FakeGen:
        def __init__(self, rounds):
            self.rounds = rounds; self.r = 0
        def generate(self, prompts, sps, use_tqdm=False):
            texts = self.rounds[self.r]; self.r += 1
            return [_O(_C(t)) for t in texts]

    good72, good99 = fmt_answer("72"), fmt_answer("99")
    r1 = [good72, good99, good72, good99]
    qs = [{"Q": "q0", "A": "72"}]
    prompts_text = ["prompt0"]
    prompt_ids = tok(prompts_text, return_tensors="pt", padding=True,
                     add_special_tokens=False)["input_ids"]
    plen = prompt_ids.shape[0 + 1]
    sps = [object() for _ in range(4)]
    gl_calls = []

    def fake_gl(merged, plen_):
        gl_calls.append(merged.shape[0])
        return torch.zeros(merged.shape[0], merged.shape[1] - plen_)

    ok = collect_retool_group(FakeGen([r1]), tok, cfg, fake_gl, qs, prompts_text,
                              prompt_ids, plen, sps)[0]
    check("vLLM 路：torch 重算零调用（副本真的省掉了）", gl_calls == [])
    clen0 = ok["clen"][0]
    exp = [-0.1 * (k + 1) for k in range(clen0)]
    check("vLLM 路：gen_logps == 采样 logprob（逐位置），形状与 mask 一致",
          tuple(ok["gen_logps"].shape) == tuple(ok["mask"].shape)
          and all(abs(float(ok["gen_logps"][0][k]) - exp[k]) < 1e-2 for k in range(clen0)))

    # 带对拍器：只在前 1 组触发 torch 重算，之后自动关闸
    cfg_v = dict(cfg)
    v2 = LogpsVerifier(1, on_finish=lambda: None)
    collect_retool_group(FakeGen([r1]), tok, cfg_v, fake_gl, qs, prompts_text,
                         prompt_ids, plen, sps, verify_logps=v2)
    check("带对拍器：恰好 1 次 torch 重算（预算内）", gl_calls == [4])
    collect_retool_group(FakeGen([r1]), tok, cfg_v, fake_gl, qs, prompts_text,
                         prompt_ids, plen, sps, verify_logps=v2)
    check("对拍预算用尽后：不再触发 torch 重算（副本释放后不会炸）", gl_calls == [4])

    # ---- 采样侧缺 logprobs 时 fail-fast（而不是静默产出错位 gen_logps）----
    class FakeGenNoLP:
        def generate(self, prompts, sps, use_tqdm=False):
            return [_O(_C(t, with_lp=False)) for t in r1]
    check("collect_logps=True 但 vLLM 没回 logprobs → RuntimeError",
          _expect_raise(lambda: multi_turn_rollout_group(
              FakeGenNoLP(), sps, tok, prompts_text * 4,
              {"max_rounds": 3, "sandbox_timeout": 5.0, "sandbox_mem_mb": 256,
               "tool_result_max_chars": 500}, collect_logps=True), RuntimeError, "nologp"))

    # ---- 配置/CLI 接线 ----
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tr = open(os.path.join(root, "rlab", "train.py"), encoding="utf-8").read()
    ro = open(os.path.join(root, "rlab", "rollout.py"), encoding="utf-8").read()
    check("preset 默认 False（历史口径零变化）+ CLI 双开关存在",
          get_config("retool_math", use_wandb=False)["vllm_gen_logps"] is False
          and '"--vllm_gen_logps"' in tr and '"--verify_gen_logps"' in tr
          and 'overrides["vllm_gen_logps"] = True' in tr)
    check("gen_worker 仅在需要时加载 torch 副本（含对拍窗口）",
          "if (not _use_vllm_logps) or _verify_budget > 0:" in ro)
    check("SamplingParams 在 vLLM 路带 logprobs=0（+ logprobs_mode 字段探测）",
          'kw["logprobs"] = 0' in ro and '"logprobs_mode" in _sp_fields' in ro)
    # 回归锁：权重同步处曾残留 gen_torch 悬空引用（pyflakes 抓到），无副本时必须跳过
    check("无 torch 副本时权重同步跳过它（不留悬空引用）",
          "gen_torch" not in ro and "if _torch_holder[0] is not None:" in ro)


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
             "rlab/probe_difficulty.py", "rlab/extract_text_model.py",
             "rlab/materialize_mm_ckpt.py",
             "rlab/ref_server.py",
             "rlab/data.py", "rlab/prepare_dapo_math.py",
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
    test_config_retool_math()
    test_trajectory_logps()
    test_multi_rollout_and_scoring()
    test_health_monitor()
    test_retool_math_fixes()
    test_question_scheduler()
    test_collect_retool_group_split()
    test_difficulty_filter()
    test_probe_aggregate()
    test_sandbox_hardening()
    test_chat_template_kwargs()
    test_split_load_remap()
    test_chunked_logps()
    test_alloc_conf_ipc()
    test_attn_impl()
    test_materialize_mm()
    test_eval_spawn_guard()
    test_eval_thinking_switch()
    test_overlong_ref_and_opt_cli()
    test_grad_clip_and_run_info()
    test_fwd_batch_chunk()
    test_vllm_gen_logps()
    test_pyflakes_undefined()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
    sys.exit(0)
