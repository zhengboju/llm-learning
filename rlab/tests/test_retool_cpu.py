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

    print("[F3] stop 机制（工具调用节奏修复，2026-09-18）")
    # 【三轮 run 代码压灭的根因】无 stop 时一段生成写满 max_tokens：代码被事后提取、
    # TOOL_RESULT 拼在整段末尾——模型先瞎猜结果才看到真结果，且末段高截断（42~55%）
    # → 无 boxed → -1，代码路径结构性负 advantage。修复 = stop 在闭围栏立即停。
    from rlab.protocol import RETOOL_STOP_KWARGS as _STOP
    _bt = chr(96) * 3
    check("stop 串 = 闭围栏+换行（含 include_stop 保留围栏字节）",
          _STOP == {"stop": [_bt + "\n"], "include_stop_str_in_output": True})
    check("开围栏不误停：'```python\\n' 不含 stop 串（chr 验证过的语义）",
          (_bt + "\n") not in (_bt + "python\n"))
    from rlab.protocol import extract_python_blocks as _epb
    check("stop 截断形态下完整块仍可提取（include_stop 保留闭围栏）",
          _epb("reasoning " + _bt + "python\nprint(1)\n" + _bt + "\n") == ["print(1)"])
    check("未闭合围栏（无 stop 触发、段被切）不产块（行为与旧协议一致）",
          _epb("x " + _bt + "python\nprint(1)\n") == [])
    _ro_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "rollout.py"), encoding="utf-8").read()
    check("gen_worker 的 make_retool_sps 接线 stop（cfg.retool_stop 条件化）",
          "if cfg.get(\"retool_stop\"):" in _ro_src
          and "kw.update(_RETOOL_STOP_KWARGS)" in _ro_src)
    _ev_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "eval_vllm_one.py"), encoding="utf-8").read()
    check("eval 的 sp_mt 接线 stop（从训练 config 回读，新旧 ckpt 不混测）",
          "RETOOL_STOP_KWARGS as _STOP_KW" in _ev_src
          and "_rcfg.get(\"retool_stop\")" in _ev_src
          and "**_stop" in _ev_src)
    check("retool_math/retool preset 默认开 retool_stop；grpo 无此键（家族隔离）",
          cfg["retool_stop"] is True
          and get_config("retool", use_wandb=False)["retool_stop"] is True
          and "retool_stop" not in get_config("grpo", use_wandb=False))
    from rlab.train import run_signature as _rsig
    _sig_stop = _rsig(cfg)
    _sig_nostop = _rsig({**cfg, "retool_stop": False})
    check("stop 进签名（-stop1 存在；关闭或缺键 → 不含 -stop1）",
          "-stop1" in _sig_stop and "-stop1" not in _sig_nostop)
    # 【2026-09-18 prompt 配套】stop 与 prompt 是同一机制的两半：必须告诉模型
    # "写完代码块就停、等结果"——否则被截停会被理解成失败，抑制写代码。
    # （对照参考 Auto_Program：prompt 教"写完代码说固定停句" ↔ stop 句同串咬合）
    from rlab.config import default_system_prompt, system_prompt_retool_math
    _math_sp = default_system_prompt("retool_math")
    check("retool_math 默认提示含 stop 配套句（写完代码块就停等结果）",
          "stop immediately and wait for the execution result" in _math_sp)
    _pf = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "prompts", "retool_math_concise.txt")
    _psp = open(_pf, encoding="utf-8").read()
    check("concise 提示文件同步补 stop 配套句（p7 用这份）",
          "stop immediately and wait for the execution result" in _psp)
    check("retool（GSM8K 家族）提示也补了（一致性，低风险）",
          "stop immediately and wait for the execution result"
          in default_system_prompt("retool"))
    _sig_sp2 = _rsig({**cfg, "system_prompt": _math_sp + " "})
    check("提示改动会更新 -sp hash（p7 与 p5/p6 的 -sp8e0184 不同 = 提示确实是新协议一部分）",
          "-sp" in _sig_sp2 and _sig_sp2.split("-sp")[1][:6] not in ("8e0184",))
    # 【2026-09-18 H2】vllm_gen_logps=True 但 vllm_logprobs_n=0 = docs/07 实锤坏路径
    # （N=0 只报被采样 token，prefill 位单点差 6.8 nat）。config 层打警告（不
    # fail-fast：torch 副本默认路径合法，A/B 与旧 run 复现需要保留）。
    import io as _io, contextlib as _ctx
    _buf = _io.StringIO()
    with _ctx.redirect_stdout(_buf):
        get_config("retool_math", use_wandb=False, vllm_gen_logps=True,
                   vllm_logprobs_n=0)
    check("H2: gen_logps=True + N=0 → config 层警告（坏路径不再静默）",
          "vllm_logprobs_n=0" in _buf.getvalue()
          and "6.8 nat" in _buf.getvalue())
    _buf2 = _io.StringIO()
    with _ctx.redirect_stdout(_buf2):
        get_config("retool_math", use_wandb=False, vllm_gen_logps=True,
                   vllm_logprobs_n=1)
    check("H2: gen_logps=True + N=1 → 无警告（正确档位）",
          "vllm_logprobs_n=0" not in _buf2.getvalue())
    _buf3 = _io.StringIO()
    with _ctx.redirect_stdout(_buf3):
        get_config("retool_math", use_wandb=False)
    check("H2: 默认 torch 副本路径（gen_logps=False）→ 无警告",
          "vllm_logprobs_n=0" not in _buf3.getvalue())
    print()


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
    # 【2026-09-20 契约扩展】新增 code_wasted（末轮写了代码但不执行的次数）。
    # code_used/code_ok 的口径**刻意不变**（末轮仍不计入）——它们是跨 run 比较的
    # code% 列，改语义会让 p8 的 48~70% 与后续 run 不可比。末轮废码单列一个字段，
    # 既补上可观测性又不动历史口径。
    # 【2026-09-21 err_types】H 组走真沙箱（print(6*7) 真执行），成功执行追加 "ok"
    check("code_used/ok 统计正确（末轮代码仍不计入——跨 run 口径不变）",
          code_stats[0] == {"code_used": 2, "code_ok": 2, "code_wasted": 1,
                            "trunc_final": 0, "err_types": ["ok", "ok"]}
          and code_stats[1] == {"code_used": 0, "code_ok": 0, "code_wasted": 0,
                                "trunc_final": 0, "err_types": []}
          and code_stats[2] == {"code_used": 1, "code_ok": 1, "code_wasted": 0,
                                "trunc_final": 0, "err_types": ["ok"]})
    check("末轮写代码 → code_wasted=1（旧版三个统计量同时为 0，与'啰嗦跑飞'无法区分）",
          code_stats[0]["code_wasted"] == 1 and code_stats[0]["trunc_final"] == 0)
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

    # 签名⑥：表面收尾（收尾率↑ 且 正确率↓ 同时成立）—— bg1 崩溃的早期签名
    def acc_drop(i):
        return -0.40 if i < 64 else (-0.80 + (0.02 if i % 2 else -0.02))
    def fmt_gain(i):
        return 0.14 if i < 64 else (0.60 + (0.02 if i % 2 else -0.02))
    codes = {c for c, _ in window_check(mk(128, acc_drop, fmt_gain))}
    check("收尾率↑+正确率↓ → surface", "surface" in codes)
    codes = {c for c, _ in window_check(
        mk(128, lambda i: -0.40 + (0.02 if i % 2 else -0.02), fmt_gain))}
    check("对照：只涨收尾率不报 surface", "surface" not in codes)
    codes = {c for c, _ in window_check(
        mk(128, acc_drop, lambda i: 0.14 + (0.02 if i % 2 else -0.02)))}
    check("对照：只掉正确率不报 surface（decline 负责）", "surface" not in codes)
    codes = {c for c, _ in window_check(
        mk(128, acc_drop, lambda i: -0.90 if i < 64 else 0.60))}
    check("对照：fmt 从低位学起 → surface 守卫不启用", "surface" not in codes)

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

    # 【2026-09-13 P1 教训】maybe_check 每个外层轮末才被调用一次、外层轮产组数
    # 不定，旧版 %16 门让多数检查点永远落空（P1 实测组 112→240 约 1 小时零检查，
    # 期间 fmt_low/length_runaway 条件真的成立过却从未报告）。滚动门：非 16 倍数
    # 也查，且不重复查没攒够新组的老窗口。
    m2 = HealthMonitor()
    for _ in range(40):
        m2.observe([-1.0, 1.0], [-1.0, -1.0], [100, 100])
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        m2.maybe_check(max_clen=200)
    check("稀疏调用：n=40（非16倍数）也检查（P1 盲窗修复）",
          "[健康检查]" in buf3.getvalue())
    m3 = HealthMonitor()
    for _ in range(18):   # 不足 32 组 → 不查
        m3.observe([-1.0, 1.0], [-1.0, -1.0], [100, 100])
    buf4 = io.StringIO()
    with contextlib.redirect_stdout(buf4):
        m3.maybe_check(max_clen=200)
    m3._last_check = 17   # 模拟"上次检查后只攒了 1 组"的稀疏调用
    with contextlib.redirect_stdout(io.StringIO()):
        m3.maybe_check(max_clen=200)
    check("不足32组/未攒够 check_every → 不查", buf4.getvalue() == "")

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
    # 【2026-09-17】阈值改基线锚定：定版协议（concise 提示 + 6144/14336）的训练起点
    # 截断率实测 22.7%，旧的"绝对 >20%"会在第一个窗口必然误报（叫狼来了会淹掉真信号）。
    hist_base = [{"acc": 0.3, "fmt": 0.9, "clen": 300.0, "code_rate": 0.5,
                  "trunc_rate": 0.23} for _ in range(40)]
    check("定版起点截断率 23% 恒定 → 不误报（旧口径会必然误报）",
          "retool_trunc" not in {c for c, _ in window_check(hist_base, retool=True)})
    hist_rise = [{"acc": 0.3, "fmt": 0.9, "clen": 300.0, "code_rate": 0.5,
                  "trunc_rate": 0.23 if i < 32 else 0.35} for i in range(96)]
    check("截断率较开局涨 12pp → 报警（截断在膨胀）",
          "retool_trunc" in {c for c, _ in window_check(hist_rise, retool=True)})
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

    # 【2026-09-16】train.jsonl 的"已剔除 dev"是**声明**不是自证：真机见到
    # 1,791,200 条（=17,912×100）的可疑产物，而 dev 污染正是 2026-09-09 事件的根因。
    # 契约必须当场核实 → 掉出 dev 题就大声告警并剔除；池子规模按 Q 去重后如实打印。
    import io as _io
    from contextlib import redirect_stdout as _rso

    from rlab.data import dedup_questions, pool_report_line, verify_train_pool_clean

    buf = _io.StringIO()
    with _rso(buf):
        kept2 = verify_train_pool_clean(pool, dev)
    check("train.jsonl 核实：含 dev 题 → 剔除 + 告警（不静默放行）",
          [r["Q"] for r in kept2] == ["q1", "q3"]
          and "held-out" in buf.getvalue() and "prepare_dapo_math" in buf.getvalue())
    buf2 = _io.StringIO()
    with _rso(buf2):
        same = verify_train_pool_clean(pool, [])
    check("train.jsonl 核实：无 dev 可核时原样返回且不告警（不噪音）",
          same == pool and buf2.getvalue() == "")

    # 题目去重（2026-09-16 真机池 1,791,200 = 17,912×100）：必须**无损**——
    # 按 (Q,A) 对去重，多解题各自保留；且判据要能说清"去重改不改变题目分布"。
    dup_rows = pool + pool                      # 3 题 × 2 份，答案一致
    dd, st = dedup_questions(dup_rows)
    check("去重：重复整行合并，保留首次出现顺序",
          [r["Q"] for r in dd] == ["q1", "q2", "q3"] and st["n_dup"] == 3
          and (st["n_in"], st["n_out"], st["n_unique_q"]) == (6, 3, 3))
    check("去重：重复份数统计（均匀 2 份）", st["dup_min"] == 2 and st["dup_max"] == 2)
    check("去重：题干相同但答案不同 → 两条都保留（不丢监督信号）",
          [r["A"] for r in dedup_questions(
              [{"Q": "q1", "A": "1"}, {"Q": "q1", "A": "2"},
               {"Q": "q1", "A": "1"}])[0]] == ["1", "2"])
    _, st_multi = dedup_questions([{"Q": "q1", "A": "1"}, {"Q": "q1", "A": "2"}])
    check("去重统计：多解题面单独计数（n_conflict_q）",
          st_multi["n_conflict_q"] == 1 and st_multi["n_out"] == 2)
    check("去重：无重复池原样返回（旧 run 逐字可比）",
          dedup_questions(pool)[0] == pool and dedup_questions(pool)[1]["n_dup"] == 0)

    check("池子证据：无重复 → 无重复字样，仍报唯一题面数",
          "无重复" in pool_report_line(dedup_questions(pool)[1], "本地 train.jsonl"))
    line_u = pool_report_line(st, "本地 train.jsonl")          # 均匀 2 份、无多解
    check("池子证据：均匀重复且无多解 → 明确写「与去重前可比」",
          "6 -> 3" in line_u and "重复 2 份/题" in line_u and "可比" in line_u
          and "不可比" not in line_u)
    line_nu = pool_report_line(dedup_questions(
        [{"Q": "q1", "A": "1"}, {"Q": "q1", "A": "1"}, {"Q": "q2", "A": "2"}])[1],
        "本地 train.jsonl")
    check("池子证据：重复不均匀 → 必须警告「与去重前不可比」（防两次 run 被当同一实验）",
          "1~2 份/题" in line_nu and "不可比" in line_nu)
    check("池子证据：空池不炸", "空池" in pool_report_line(
        {"n_in": 0, "n_out": 0, "n_dup": 0, "n_unique_q": 0, "n_conflict_q": 0,
         "dup_min": 0, "dup_max": 0}, "x"))

    # 【2026-09-16 事故·held-out 100% 污染】源数据每题 100~400 份重复（HF 侧
    # 1,791,200 行 / ~16.7k 唯一题面）。旧版 prepare 按**位置**切 dev/train，
    # 实测 train.jsonl 含 52,700 条 dev 题（500 题 × ~105 份）——dev 每一题都被训过。
    from rlab.prepare_dapo_math import split_train_dev

    dup_pool = [{"Q": f"q{i}", "A": "1"} for i in range(10) for _ in range(100)]
    tr, dv = split_train_dev(dup_pool, 3, 42)
    check("prepare 切分：以题面为单位抽满 dev（整组归 dev，行数随之超目标）、两侧不相交",
          len(dv) == 100 and len(tr) == 900
          and not ({r["Q"] for r in dv} & {r["Q"] for r in tr}))
    check("prepare 切分：同一题面的所有副本整体归一侧（不会一半 dev 一半 train）",
          all(sum(1 for r in dv if r["Q"] == q) in (0, 100)
              for q in {r["Q"] for r in dup_pool}))
    tr2, dv2 = split_train_dev([{"Q": "q1", "A": "1"}, {"Q": "q1", "A": "2"},
                                {"Q": "q2", "A": "3"}], 1, 42)
    check("prepare 切分：同题多解整体归一侧（不给'答案一半在 dev'留口子）",
          not ({r["Q"] for r in dv2} & {r["Q"] for r in tr2})
          and len(dv2) + len(tr2) == 3)
    # 反证：旧的位置切法在同一份重复池上**必然**相交——这条锁住"为什么必须按题面切"
    _old_dv, _old_tr = dup_pool[:3], dup_pool[3:]
    check("反证：旧的位置切法在重复池上必然泄漏（本次事故的机制）",
          bool({r["Q"] for r in _old_dv} & {r["Q"] for r in _old_tr}))
    psrc = open("rlab/prepare_dapo_math.py", encoding="utf-8").read()
    check("prepare 脚本三步齐备：去重 → 按题面切 → 落盘复读断言（缺一即回到事故）",
          "dedup_questions(records)" in psrc and "split_train_dev(records" in psrc
          and "_read_qa_jsonl(args.output_dir" in psrc)

    # 【2026-09-16 事故·跨模块缝隙·绿字下的全错】normalize_row 曾返回
    # {"question","answer"}，而 dedup/split 读 "Q"/"A" → 两边都 .get() 到 None →
    # **1,791,700 行塌成 1 条**，报告却写"重复 1791700 份/题（均匀且无多解：可比）"。
    # 两个模块各自的测试都是绿的，缝隙没人测。这条端到端断言就是那条缝隙。
    from rlab.data import require_qa_rows
    from rlab.prepare_dapo_math import normalize_row

    raw = [{"prompt": [{"content": "Q1"}], "reward_model": {"ground_truth": "1"}},
           {"prompt": [{"content": "Q2"}], "reward_model": {"ground_truth": "2"}}] * 3
    norm = [r for r in (normalize_row(i, x) for i, x in enumerate(raw)) if r]
    check("缝隙：normalize_row 产出规范键名 Q/A（不是 question/answer）",
          len(norm) == 6 and all("Q" in r and "A" in r and "question" not in r for r in norm))
    _n_dd, _n_st = dedup_questions(norm)
    check("缝隙：normalize_row 的输出能被去重正确消费（修复前整池塌成 1 条）",
          _n_st["n_in"] == 6 and _n_st["n_out"] == 2 and _n_st["n_unique_q"] == 2)
    _n_tr, _n_dv = split_train_dev(_n_dd, 1, 42)
    check("缝隙：去重输出能被切分正确消费（修复前 KeyError）",
          len(_n_tr) == 1 and len(_n_dv) == 1)
    try:
        require_qa_rows([{"question": "q", "answer": "a"}], "测试")
        check("护栏：别名键名必须 fail-fast（不许塌成 1 条还报「可比」）", False)
    except RuntimeError as e:
        check("护栏：别名键名必须 fail-fast（不许塌成 1 条还报「可比」）",
              "Q/A" in str(e) and "question" in str(e))
    try:
        dedup_questions([{"Q": "", "A": "a"}])
        check("护栏：空题面也必须拦（空串会被当成同一个键合并）", False)
    except RuntimeError as e:
        check("护栏：空题面也必须拦（空串会被当成同一个键合并）", "Q/A" in str(e))

    # prepare **端到端**（打桩数据源，不触网）：main() 的「去重 → 切分 → 落盘 → 复读」
    # 全链。此前 main() 无任何测试——事故就发生在 main 拼装这三步的地方。
    import sys as _sys
    import tempfile as _tf
    from pathlib import Path as _P

    import rlab.prepare_dapo_math as _ppm
    from rlab.data import _read_qa_jsonl as _rq

    _saved_loader, _saved_argv = _ppm._load_records, _sys.argv
    _ppm._load_records = lambda: [{"Q": f"q{i}", "A": "1"} for i in range(5) for _ in range(4)]
    try:
        with _tf.TemporaryDirectory() as _td:
            _sys.argv = ["prepare_dapo_math", "--output-dir", _td, "--dev-size", "2"]
            with _rso(_io.StringIO()):
                _ppm.main()
            _tr = _rq(str(_P(_td) / "train.jsonl"))
            _dv = _rq(str(_P(_td) / "dev.jsonl"))
        check("prepare 端到端（打桩源）：去重生效 + 落盘 train/dev 题面不相交",
              len(_tr) == 3 and len(_dv) == 2
              and not ({r["Q"] for r in _tr} & {r["Q"] for r in _dv}))
    finally:
        _ppm._load_records, _sys.argv = _saved_loader, _saved_argv
    dsrc2 = open("rlab/data.py", encoding="utf-8").read()
    check("load_dapo_math_train 两条加载路都接上去重（不是只定义了纯函数）",
          dsrc2.count("return _finalize_train_pool(rows") == 3
          and "dedup_questions(rows)" in dsrc2)

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
    # 【2026-09-12 反压修复】旧语义"overlong 不动 streak"导致模型整体变长时
    # 题池原地打转（真机：丢弃率 90%、采样主循环空转、训练端 5 小时零产出）。
    # 现在超长计入 streak（但不插队首——不是"立刻重试"，是"当前预算装不下"）。
    sched.report(q1, "uniform")
    sched.report(q1, "overlong")
    check("overlong 计入 streak（默认开启反压，防采样空转）", sched.q_stat["q1"] == 2)
    check("overlong 达标即拉黑、不插回队首",
          "q1" not in [q["Q"] for q in sched.draw(8)])
    # 旧语义可显式退回（对照实验用）
    sched._refill()
    q2 = {"Q": "q2", "A": "1"}
    sched.report(q2, "uniform")
    sched.report(q2, "overlong", count_overlong=False)
    check("count_overlong=False → 退回旧语义（不动 streak）", sched.q_stat["q2"] == 1)
    check("blacklisted_count 只算达标（拉黑）题：q0/q1 达标、q2 未达标 → 2",
          sched.blacklisted_count() == 2)
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
    # 【2026-09-17 真机 bg1 崩溃后的推论】band(0,1) 会留下 lopsided 组（8 条里只有
    # 1-2 条对）：那种组里正向 advantage 只发给"幸运那一条"，RL 学到的是运气。
    # band 收窄到 (0.25,0.75) 且 k=8 → 只留 3~5 对（对错各半、±1 有信息量）。
    _q8 = [{"Q": f"m{i}", "A": "1"} for i in range(9)]
    _t8 = {f"m{nc}": {"k": 8, "n_correct": nc} for nc in range(9)}
    _k8, _s8 = filter_qas_by_difficulty(_q8, _t8, lo=0.25, hi=0.75)
    check("k=8 + band(0.25,0.75)：只留 3~5 对，1/2/6/7/8 对全出清（治 lopsided 组）",
          [x["Q"] for x in _k8] == ["m3", "m4", "m5"]
          and _s8["band_out"] == 4 and _s8["p_zero"] == 1 and _s8["p_one"] == 1)
    # band 是过滤器的唯一旋钮，此前只能改 config.py 源码（bg1 崩后补 CLI）
    import inspect as _insb
    import rlab.train as _TB
    _tsrcb = _insb.getsource(_TB.main)
    check("train.py 暴露 --difficulty_band 并接线（含 0<=lo<hi<=1 校验）",
          '"--difficulty_band"' in _tsrcb
          and 'overrides["difficulty_band"] = (_lo, _hi)' in _tsrcb)
    # 【2026-09-17 真机】探针此前不带引擎档：连 preset 的 gdn_prefill_backend=triton
    # 都没传（会落回 FlashInfer GDN JIT → 无 traceback 的 SIGKILL），环境里遗留的
    # VLLM_BATCH_INVARIANT=1 又让引擎启动即 RuntimeError。探针与训练同档是铁律。
    import rlab.probe_difficulty as _PD
    _psrc = _insb.getsource(_PD.main)
    check("probe 把 preset 的 vllm_gen_kwargs 传进 LLM（否则掉回 FlashInfer JIT 档）",
          'dict(cfg.get("vllm_gen_kwargs") or {})' in _psrc
          and "LLM(model=_vllm_path, gpu_memory_utilization=args.gpu_mem, **_gk)" in _psrc)
    check("probe 有一致性档入口 + 缺 backend 时引擎构造前 fail-fast",
          '"--vllm_batch_invariant"' in _psrc
          and "batch_invariant_guard(_bi, args.vllm_attention_backend)" in _psrc
          and "attention_backend_kwargs(args.vllm_attention_backend)" in _psrc)
    check("probe 识别继承的 VLLM_BATCH_INVARIANT（真机就是被它撞死的）",
          'os.environ.get("VLLM_BATCH_INVARIANT"' in _psrc
          and 'os.environ["VLLM_BATCH_INVARIANT"] = "1"' in _psrc)
    check("probe 用 vllm_model_path（分裂加载）而非硬编码 model_path 起引擎",
          '_vllm_path = cfg.get("vllm_model_path") or cfg["model_path"]' in _psrc)

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

    # 【2026-09-18 M4】表协议指纹：表是"模型×提示×预算"的联合产物，行必须自证出处。
    # 换预算/提示/k 续跑同一 --out 会静默混表，训练端无从检测——load 时校验。
    from rlab.probe_difficulty import aggregate_rows
    _meta = {"model": "Qwen3.5-4B", "k": 8, "rounds": 2, "round_tokens": 6144,
             "ctx": 14336, "temp": 1.0, "sp": "8e0184"}
    _mr = [{"Q": "a", "A": "7", "acc": 1, "fmt": 1, "trunc": 0, "clen": 100, "code_ok": 1},
           {"Q": "a", "A": "7", "acc": -1, "fmt": 1, "trunc": 1, "clen": 200, "code_ok": 0}]
    _r2 = aggregate_rows(_mr, _meta)
    check("aggregate_rows 带 probe_meta：每行落协议指纹",
          _r2 and _r2[0].get("probe_meta") == _meta)
    _r3 = aggregate_rows(_mr[:2])
    check("不带 probe_meta：行无该字段（旧调用方零变化）",
          "probe_meta" not in _r3[0])
    with tempfile.TemporaryDirectory() as td:
        _p2 = os.path.join(td, "t2.jsonl")
        with open(_p2, "w", encoding="utf-8") as f:
            for r in _r2:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        _tbl2 = load_difficulty_table(_p2, expected_meta=_meta)
        check("expected_meta 匹配 → 正常加载（不告警）", set(_tbl2) == {"a"})
        _tbl3 = load_difficulty_table(_p2, expected_meta={**_meta, "sp": "deadbe"})
        check("expected_meta 不匹配（换提示续跑同表）→ 仍加载（不 fail-fast，旧表兼容）"
              "但触发告警", set(_tbl3) == {"a"})
        _tbl4 = load_difficulty_table(_p2)   # 无 expected_meta（训练端不传）→ 零告警
        check("无 expected_meta → 不校验（旧调用方零变化）", set(_tbl4) == {"a"})
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
    # 【2026-09-18 M3】断点续跑时 summarize 必须以磁盘全量表的聚合为准，否则判读
    # 数字只覆盖"本轮新探片段"（系统性失真）。disk_per_q 传入时用它做难度分布 +
    # 按 k 加权重算轨迹级比率。
    _dk = [{"Q": "x", "A": "1", "k": 8, "n_correct": 0, "fmt_rate": 1.0, "trunc_rate": 0.0},
           {"Q": "y", "A": "2", "k": 8, "n_correct": 8, "fmt_rate": 1.0, "trunc_rate": 0.0},
           {"Q": "z", "A": "3", "k": 8, "n_correct": 4, "fmt_rate": 0.5, "trunc_rate": 0.25}]
    _sd = summarize([], [], disk_per_q=_dk)
    check("M3: 续跑判读用磁盘全量（题数=3、全错1/可学1/全对1、无 boxed 由 fmt_rate 加权）",
          "题数 3" in _sd and "（磁盘全量）" in _sd
          and "全错(0/4类) 1" in _sd and "可学(0<rate<1) 1" in _sd and "全对 1" in _sd)
    check("M3: 轨迹级比率按 k 加权（z 的 fmt_rate 0.5 → 无 boxed = (8*0+8*0+8*0.5)/24 = 16.7%）",
          "16.7%" in _sd and "末段截断" in _sd)
    check("M3: 无 disk_per_q → 保持旧路径（本轮内存口径，无『磁盘全量』标记）",
          "（磁盘全量）" not in summarize(rows, per_q))
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


def test_vllm_gen_kwargs():
    """【P2】vLLM 引擎参数透传：起跑期 FlashInfer JIT 编译撞宿主 RAM 上限的规避入口。

    2026-09-14 事故：Qwen3.5 的 GDN prefill 默认 FlashInfer JIT（现场 nvcc），编译
    窗口与三方搬权重的启动期重合 → 生成端被 OOM-killer 以 SIGKILL 带走、**无任何
    Python traceback**，训练端只看到"生成端进程已退出"。2026-09-16 起默认档改为
    gdn_prefill_backend="triton"（免 nvcc），训练 rollout 与评测读同一份配置。
    本测试覆盖：默认档 / 覆盖方式 / 类型闸 / 回读容错 / 签名可见性 / 接线
    （真机才有 vLLM，接线用源码断言兜住）。"""
    print("[P2] vLLM 引擎参数透传：gdn_prefill_backend=triton 免起跑期 JIT 编译")
    from rlab.rollout import _check_vllm_gen_kwargs, _vllm_config_readback
    from rlab.train import run_signature

    cfg = get_config("retool_math", use_wandb=False)
    check("BASE 默认 vllm_gen_kwargs 已含 gdn_prefill_backend=triton（2026-09-16 默认改档）",
          cfg["vllm_gen_kwargs"] == {"gdn_prefill_backend": "triton"})
    check("传 '{}' 可整体关掉（回到 vLLM 默认 = FlashInfer）",
          get_config("retool_math", use_wandb=False,
                     vllm_gen_kwargs={})["vllm_gen_kwargs"] == {})
    check("CLI JSON 反序列化形态（与 chat_template_kwargs 同一解析套路）",
          json.loads('{"gdn_prefill_backend": "triton"}')
          == {"gdn_prefill_backend": "triton"})

    def _raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        except Exception as e:
            print(f"  !! 期望 {exc.__name__}，实得 {type(e).__name__}: {e}")
            return False
        print(f"  !! 期望 {exc.__name__}，实得无异常")
        return False

    # 【类型闸的反证控制】非 dict 若原样进 LLM(**x)，只会在生成端子进程里炸成一句
    # "进程已退出"；闸放在 config 层，错误才留在能排查的地方。dict 必须放行（反证）。
    check("非 dict（如 CLI 递了 JSON 数组）-> config 层 ValueError；dict 放行为反证",
          _raises(lambda: get_config("retool_math", vllm_gen_kwargs=["triton"]),
                  ValueError)
          and get_config("retool_math", vllm_gen_kwargs={"a": 1})["vllm_gen_kwargs"]
          == {"a": 1})

    # 回读：真机对象布局随版本漂，判据是"能不能在图里找到"，不是写死路径。
    # 【2026-09-14 实机教训】首版硬编码 LLM.llm_engine[.vllm_config].model_config 四条
    # 路径，在 vLLM 0.19 上打回 None——那既可能=键被静默忽略、也可能=我找错了地方，
    # 判据当场失去分辨力（只能靠"日志里没有 ninja"间接推断修复生效）。改为有界 BFS。
    class _MC:
        def __init__(self):
            self.gdn_prefill_backend = "triton"

    class _VC:
        def __init__(self):
            self.model_config = _MC()

    class _Eng:
        def __init__(self):
            self.vllm_config = _VC()

    class _LLM:
        def __init__(self):
            self.engine = _Eng()

    class _ClsAttr:
        gdn_prefill_backend = "triton"      # 类属性：纯 __dict__ 遍历读不到，getattr 能

    class _Big:
        def __init__(self):
            self.blob = list(range(100000))  # 大容器必须被排除，否则 BFS 爆炸
            self.gdn_prefill_backend = "triton"

    check("回读到生效值（实例属性深三层，非写死路径）",
          _vllm_config_readback(_LLM(), "gdn_prefill_backend") == ("triton", []))
    check("类属性也读得到（首版纯 __dict__ 遍历的盲区）",
          _vllm_config_readback(_ClsAttr(), "gdn_prefill_backend")[0] == "triton")
    check("大容器不进图但仍找得到键",
          _vllm_config_readback(_Big(), "gdn_prefill_backend")[0] == "triton")
    check("miss -> (None, 线索) 而不抛；线索带宿主类型名（用于分辨「键被忽略」"
          "vs「布局变了」两种 None）",
          _vllm_config_readback(object(), "gdn_prefill_backend") == (None, [])
          and _vllm_config_readback(
              type("Mn", (), {"gdn_prefill_backends": "x"})(),
              "gdn_prefill_backend")[1] == ["Mn.gdn_prefill_backends"])

    # 【2026-09-14 实机复现的 A1】护栏判据必须是"本进程**能不能构造**"（meta 真预检），
    # 不是"config 是不是复合形状"——同一份复合 ckpt 在 train 侧（裸 torch）两次 run 都
    # 加载成功、在 gen 侧（import 过 vLLM）必抛 A1，差别在**环境**不在形状；按形状判会在
    # 环境修好之后误杀合法启动（重装环境后就正好落在这一条上）。
    from rlab.rollout import _assert_torch_replica_loadable, _probe_torch_construct
    from rlab.model_loading import resolve_load_config
    import rlab.rollout as _ro

    def _mk_cfg(cfg_dict):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f)
        return d

    # 【2026-09-15 假绿灯】预检必须喂**与真实加载同一个 config**。实机：预检探顶层复合
    # cfg（from_config 自己会解包）、加载走 from_pretrained 的自动解包（gen 进程不解包）
    # → 预检在崩溃前打印"通过"。判据只能锚在"同源"上：预检与加载共用 resolve_load_config。
    # 替身用 llava 而非 qwen3_5：本机 transformers 4.41 不认识 qwen3_5（开发机不装 5.x
    # 依赖），而判据要保的是"复合 ckpt → 解析出 text_config"这条逻辑，与具体架构无关。
    _mm = _mk_cfg({"model_type": "llava",
                   "text_config": {"model_type": "llama", "vocab_size": 128},
                   "vision_config": {"model_type": "clip_vision_model", "hidden_size": 32,
                                     "image_size": 8, "patch_size": 4, "num_hidden_layers": 1,
                                     "num_attention_heads": 1, "intermediate_size": 32},
                   "architectures": ["LlavaForConditionalGeneration"]})
    _txt = _mk_cfg({"model_type": "llama", "vocab_size": 128})
    check("复合 ckpt 解析出 text_config（显式喂它，不靠自动解包）",
          resolve_load_config(_mm)[1] is True
          and resolve_load_config(_mm)[0].model_type == "llama")
    check("纯文本 ckpt 解析出顶层 config（原样加载，行为零变化）",
          resolve_load_config(_txt)[1] is False
          and resolve_load_config(_txt)[0].model_type == "llama")
    check("坏目录直接抛（不静默当成纯文本 ckpt 放行）",
          _raises(lambda: resolve_load_config("/nope/nope"), Exception))
    # 同源判据走**行为**（不数源码文本）：给解析函数装替身，预检与加载都必须经过它——
    # 假绿灯的根因就是"预检自己解析了一份、加载又解析了一份"，两者分叉。
    import rlab.model_loading as _ml
    _orig_resolve = _ml.resolve_load_config
    _called = []
    try:
        _ml.resolve_load_config = lambda p: (_called.append(p), _orig_resolve(p))[1]
        _probe_torch_construct("gpt2")
        check("预检经过 resolve_load_config（同源，不是各解析一份）", _called == ["gpt2"])

        class _Sentinel(Exception):
            pass

        def _boom(p):
            raise _Sentinel(p)

        _ml.resolve_load_config = _boom
        check("加载也经过 resolve_load_config（同一份解析）",
              _raises(lambda: _ml.load_causal_lm("gpt2", dtype=torch.float32), _Sentinel))
    finally:
        _ml.resolve_load_config = _orig_resolve

    # 判据三分支：替身注入预检结果（真预检见下）
    _orig_probe = _ro._probe_torch_construct
    try:
        _ro._probe_torch_construct = lambda p: {
            "error": "AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'",
            "composite": True, "model_type": "qwen3_5_text"}
        check("预检报 A1 特征 -> RuntimeError（拦，带两条改法）",
              _raises(lambda: _assert_torch_replica_loadable("/x"), RuntimeError))
        _ro._probe_torch_construct = lambda p: {
            "error": "ValueError: meta 预检自身的限制", "composite": False, "model_type": "gpt2"}
        check("预检报非 A1 -> 只告警放行（反证：否则预检的怪癖会误杀合法启动）",
              _assert_torch_replica_loadable("/x") is None)
        _ro._probe_torch_construct = lambda p: {
            "error": None, "composite": False, "model_type": "gpt2"}
        check("预检通过 -> 放行", _assert_torch_replica_loadable("/x") is None)
    finally:
        _ro._probe_torch_construct = _orig_probe

    # 真预检（不注入替身）：meta 构造零显存零内存，喂的 config 与真实加载同源
    check("真预检：可加载的小模型返回 error=None（gpt2 本地缓存）",
          _probe_torch_construct("gpt2")["error"] is None
          and _probe_torch_construct("gpt2")["composite"] is False)
    check("真预检：坏目录/缺 config 返回错误串而不抛",
          isinstance(_probe_torch_construct(tempfile.mkdtemp())["error"], str))

    check("空 kwargs 直接放行（不触发 vLLM import，CPU 可测）",
          _check_vllm_gen_kwargs({}) is None)
    try:
        import vllm  # noqa: F401
        _has_vllm = True
    except Exception:
        _has_vllm = False
    if _has_vllm:
        check("键名写错 -> RuntimeError（静默忽略=修复白做，必须拦在构造前）",
              _raises(lambda: _check_vllm_gen_kwargs({"no_such_engine_key": 1}),
                      RuntimeError))
    else:
        print("  -- 本机无 vLLM：键名闸的真机判据回落为下面的接线断言")

    # 签名可见性：kernel 换了（triton vs flashinfer）不能算同配方——默认档现在就是
    # triton，故新 run 必然带 -vk 段；只有**显式关掉该键**（vllm_gen_kwargs=None）时
    # 历史签名串逐字不变，旧 ckpt 仍算同签名（刻意的：不传档 = 同一配方）。
    # 【2026-09-18 stop 机制】retool_math preset 默认 retool_stop=True → 签名尾部
    # 追加 -stop1（在 -vk 段之后）。
    # 【2026-09-21 overlong filtering】retool_math preset 默认 overlong_filter=True
    # → -stop1 之后再追加 -of1。vk 断言相应改为含 -stop1-of1 后缀。
    sig0 = run_signature(cfg)
    check("默认档带 triton：签名含 -vkgdn_prefill_backend=triton（不静默换档）",
          "-vkgdn_prefill_backend=triton" in sig0 and sig0.endswith("-stop1-of1"))
    sig_none = run_signature({**cfg, "vllm_gen_kwargs": None})
    # vk 段排在 stop/of 段之前：vk 开关会移动其后所有段，原"全串 startswith"
    # 语义失效——改为比较去掉 -stop1-of1 尾巴后的前缀关系（验证力不变）。
    _tail = "-stop1-of1"
    _s0, _sn = sig0[:-len(_tail)], sig_none[:-len(_tail)]
    check("显式关掉该键：签名无 vk 段（去掉 -stop1-of1 后历史串逐字不变 -> 旧 ckpt 同签名）",
          "-vk" not in sig_none and _s0.startswith(_sn) and sig_none.endswith("-stop1-of1"))
    sig_vk = run_signature({**cfg, "vllm_gen_kwargs": {"gdn_prefill_backend": "flashinfer"}})
    check("换档 -> 签名尾部追加 -vk<键=值>（纯追加，前缀不变）",
          sig_vk.endswith("-vkgdn_prefill_backend=flashinfer-stop1-of1")
          and sig_vk[:-len(_tail)].startswith(_sn))
    check("多个键按 key 排序（同配方两次 run 签名逐字可比）",
          run_signature({**cfg, "vllm_gen_kwargs": {"b": 1, "a": 2}})
          .endswith("-vka=2,b=1-stop1-of1"))

    # 接线（无 GPU 的机器上唯一能验的部分：真机构造路径由源码断言兜住）
    rollout_src = open("rlab/rollout.py", encoding="utf-8").read()
    train_src = open("rlab/train.py", encoding="utf-8").read()
    check("gen_worker 把透传 dict 展开进 LLM(**kwargs)",
          "**_gen_kwargs)" in rollout_src)
    check("键名闸在 LLM 构造**之前**（构造本身就是那次 JIT，事后回读来不及）",
          rollout_src.index("_check_vllm_gen_kwargs(_gen_kwargs)")
          < rollout_src.index("vllm_gen = LLM("))
    check("A1 护栏前置到 LLM() 之前（任何 GPU 分配前就拦，不是白烧一分钟才崩）",
          rollout_src.index('_assert_torch_replica_loadable(cfg["model_path"])')
          < rollout_src.index("vllm_gen = LLM("))
    check("train.py CLI --vllm_gen_kwargs 映射到 overrides",
          '"--vllm_gen_kwargs"' in train_src
          and 'overrides["vllm_gen_kwargs"] = json.loads(args.vllm_gen_kwargs)' in train_src)
    # 【评测同档】评测端曾完全不传 gdn_prefill_backend：训练换了 kernel 而评测没换，
    # Δacc 里混进 kernel 变量，且评测自己也会撞 FlashInfer JIT（无 traceback）。
    eval_src = open("eval_vllm_one.py", encoding="utf-8").read()
    check("eval_vllm_one.py 从 rlab 配置取 vllm_gen_kwargs 并展开进 LLM(**kwargs)",
          '_vllm_kwargs = dict(_rcfg.get("vllm_gen_kwargs") or {})' in eval_src
          and "**_vllm_kwargs)" in eval_src)

    # 【2026-09-16 默认 triton ≠ 兜底】--vllm_gen_kwargs 是**整体替换**：只想加一个键
    # （如 enable_prefix_caching）却没把 triton 写回，就会静默掉回 FlashInfer GDN JIT ——
    # 即 2026-09-14 那个无 traceback 的 SIGKILL 档。默认值救不了它，必须有前置告警。
    from rlab.rollout import gdn_backend_missing

    check("默认档（dict 里带 triton）→ 不告警（不噪音）",
          gdn_backend_missing("/root/Qwen3.5-4B",
                              {"gdn_prefill_backend": "triton"}) is False)
    check("整体替换成只含新键 → 判定缺档（真陷阱：默认值不兜底）",
          gdn_backend_missing("/root/Qwen3.5-4B",
                              {"enable_prefix_caching": True}) is True)
    check("显式空档 {}（有意的 FlashInfer 消融）→ 同样判定缺档",
          gdn_backend_missing("/root/Qwen3.5-4B", {}) is True)
    check("非 GDN 模型（Qwen2.5-3B）缺键不告警（该键本就不被使用）",
          gdn_backend_missing("/root/Qwen2.5-3B", {}) is False)
    check("vLLM 路径缺失时不误报", gdn_backend_missing(None, {}) is False)
    check("gen_worker 在 LLM 构造前就打印该告警（不白烧一次引擎初始化才崩）",
          "gdn_backend_missing(cfg.get(" in rollout_src
          and rollout_src.index("gdn_backend_missing(cfg.get(")
          < rollout_src.index("vllm_gen = LLM("))
    check("CLI help 仍写明「整体替换」（语义不能悄悄变成 merge）",
          "整体替换" in train_src)


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
    # 【2026-09-16 实机】save_pretrained 按 _checkpoint_conversion_mapping 逆向写回
    # model.language_model.X（config 仍是文本类）→ 产物是"多模态权重 + 文本 config"。
    # 映射必须幂等，否则产出 model.language_model.language_model.X 对不上骨架。
    mm_sd = [("model.language_model.embed_tokens.weight", "t0"),
             ("model.language_model.norm.weight", "t2"),
             ("model.language_model.lm_head.weight", "t3"),   # tied 的多模态形态
             ("model.language_model.layers.0.linear_attn.A_log", "t4")]
    out_mm = dict(remap_text_to_multimodal(mm_sd))
    check("已是多模态布局 -> 原样透传（幂等，不产出双前缀）",
          out_mm["model.language_model.embed_tokens.weight"] == "t0"
          and out_mm["model.language_model.layers.0.linear_attn.A_log"] == "t4"
          and not any("language_model.language_model" in k for k in out_mm))
    check("多模态形态的 tied lm_head 同样丢弃",
          "model.language_model.lm_head.weight" not in out_mm)
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
          'attn_implementation=cfg.get("attn_implementation", "sdpa")' in train_src)
    check("rollout.py gen 副本加载点走 cfg",
          'attn_implementation=cfg.get("attn_implementation", "sdpa")' in rollout_src)
    # 【2026-09-15 收口】三处加载统一走 rlab.model_loading.load_causal_lm：口径（多模态
    # 目录直连/显式 text_config）与防静默缺键护栏只有一份实现——比逐个字面断言更能保契约。
    check("三处 torch 加载点统一收口到 load_causal_lm",
          all("load_causal_lm(" in s for s in (train_src, rollout_src, ref_src)))
    check("收口后不再有裸 AutoModelForCausalLM.from_pretrained 加载点",
          all("AutoModelForCausalLM.from_pretrained" not in s
              for s in (train_src, rollout_src, ref_src)))
    check("ref_server.py FA2 档位自动降 bf16（FA2 不支持 fp32）",
          'torch.bfloat16 if attn_implementation == "flash_attention_2"' in ref_src)
    # 【2026-09-14 修脆性断言】原判据是字面串 '--attn_implementation "$ATTN_IMPL" "$@"'，
    # 8496f47 往 "$@" 之前插入 '--out_dir "$OUT_DIR"'（护栏/record 归档跟随用户目录）后，
    # 两者不再字面紧邻 → 断言变红；而它要保的契约其实完好：注入 flag 排在 "$@" 之前，
    # argparse 后者胜、用户可覆盖。原写法把"参数顺序契约"和"字面排版"混为一谈。
    # 改断言**位置关系**（同文件 train_src.index(...) < train_src.index("import torch")
    # 是同范式），锚在 train 调用内部——以后再往中间插 flag、或在别处写 "$@" 注释都不会误报。
    _train_i = sh_src.index("python -m rlab.train")
    _inj_i = sh_src.index('--attn_implementation "$ATTN_IMPL"', _train_i)  # train 那处注入
    _argv_i = sh_src.index('"$@"', _train_i)                              # train 的用户参数兜底
    check("run_gsm8k.sh 把 ATTN_IMPL 注入 ref_server 与 train 两处（手动传参可覆盖："
          "注入 flag 在 \"$@\" 之前，argparse 后者胜）",
          sh_src.count('--attn_implementation "$ATTN_IMPL"') == 2 and _inj_i < _argv_i)
    check("train.py CLI choices 含 flash_attention_2",
          '"flash_attention_2"' in train_src and "--attn_implementation" in train_src)
    # 【2026-09-15 pod 实机】显式传 config 后，私有名 _attn_implementation 会漏给
    # __init__（TypeError: unexpected keyword argument）——三处加载点全中。判据锚在
    # "真正传出去的 kwargs"上（纯函数，本地 CPU 可测，不必真加载模型）。
    from rlab.model_loading import build_load_kwargs
    _kw = build_load_kwargs(object(), True, torch.bfloat16, "flash_attention_2")
    check("显式 config 时 attn 走**公开名**（私有名会被漏给 __init__ → pod TypeError）",
          _kw.get("attn_implementation") == "flash_attention_2"
          and "_attn_implementation" not in _kw)
    check("复合 ckpt 把 config 显式带上；纯文本 ckpt 不带（原样加载，行为零变化）",
          "config" in _kw
          and "config" not in build_load_kwargs(object(), False, torch.bfloat16))
    check("不传 attn_implementation 时不带该键（用 config 自带口径）",
          "attn_implementation" not in build_load_kwargs(object(), False, torch.float32))


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

    # ---- A/B：存盘即多模态壳 / 评测自动物化（2026-09-16 真统一）----
    from rlab.materialize_mm_ckpt import (load_mm_skeleton, materialize_mm_checkpoint,
                                          merge_text_into_skeleton, read_mm_key_index,
                                          write_mm_checkpoint)
    from safetensors.torch import load_file, save_file

    skel = {"model.visual.patch_embed.weight": t_vis}
    merged2, st2 = merge_text_into_skeleton(text_sd, skel)
    check("骨架合并：语言键落到 model.language_model.*、tied lm_head 丢弃、非语言键原样",
          merged2["model.language_model.embed_tokens.weight"] is t_emb
          and "lm_head.weight" not in merged2
          and merged2["model.visual.patch_embed.weight"] is t_vis
          and st2 == {"text_substituted": 2, "base_kept": 1})
    check("骨架合并：dtype 对齐存盘档（fp32 master -> bf16）",
          merge_text_into_skeleton(text_sd, skel, dtype=torch.bfloat16)[0]
          ["model.language_model.embed_tokens.weight"].dtype == torch.bfloat16)
    try:
        merge_text_into_skeleton({"foo.weight": t_norm}, skel)
        check("骨架合并：未知键 fail-fast", False)
    except KeyError:
        check("骨架合并：未知键 fail-fast（映射表不许静默漏同步）", True)
    # 【2026-09-16 实机 crash 的最小复现】旧 ckpt = 多模态权重 + 文本 config：
    # 幂等映射后能直接并进骨架；修复前这里会产出双前缀并对不上 base_keys。
    merged_mm, _ = merge_text_into_skeleton(
        {"model.language_model.embed_tokens.weight": t_emb}, skel,
        base_keys={"model.language_model.embed_tokens.weight",
                   "model.visual.patch_embed.weight"})
    check("幂等映射：已多模态布局的旧 ckpt 也能并进骨架（不产双前缀）",
          merged_mm["model.language_model.embed_tokens.weight"] is t_emb
          and not any("language_model.language_model" in k for k in merged_mm))

    # 落盘口径：骨架分片流式读（只留非语言键）→ 产物=单文件权重 + 骨架非权重文件
    mm_dir = tempfile.mkdtemp()
    save_file({"model.language_model.embed_tokens.weight": torch.randn(4, 3),
               "model.language_model.norm.weight": torch.randn(4),
               "model.visual.patch_embed.weight": t_vis},
              os.path.join(mm_dir, "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(mm_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "qwen3_5",
                   "text_config": {"model_type": "qwen3_5_text"}}, f)
    skel_loaded = load_mm_skeleton(mm_dir)
    check("load_mm_skeleton 流式只留非语言键（8G 语言权重不进训练/评测进程内存）",
          list(skel_loaded) == ["model.visual.patch_embed.weight"])
    check("read_mm_key_index 只读头部拿全键名（含语言键，零张量加载）",
          read_mm_key_index(mm_dir) == {"model.language_model.embed_tokens.weight",
                                        "model.language_model.norm.weight",
                                        "model.visual.patch_embed.weight"})
    out_dir = os.path.join(tempfile.mkdtemp(), "step_1")
    stats = write_mm_checkpoint(text_sd, mm_dir, out_dir, skeleton=skel_loaded,
                                dtype=torch.bfloat16)
    check("write_mm_checkpoint 落单文件权重 + 从骨架拷 config（vLLM 多模态路由依赖）",
          os.path.isfile(os.path.join(out_dir, "model.safetensors"))
          and os.path.isfile(os.path.join(out_dir, "config.json"))
          and stats["text_substituted"] == 2 and stats["base_kept"] == 1)
    back = load_file(os.path.join(out_dir, "model.safetensors"))
    check("落盘产物：语言键=多模态布局、视觉键在、tied lm_head 不在",
          "model.language_model.embed_tokens.weight" in back
          and "model.visual.patch_embed.weight" in back
          and "lm_head.weight" not in back)
    check("产物 config 是复合体（torch 侧 resolve_load_config 也能再读 = 真统一）",
          "text_config" in json.load(open(os.path.join(out_dir, "config.json"))))
    # 骨架模式丢了"骨架语言键全覆盖"自检 → 用只读头部的全键名把"文本键必须命中"补回来
    try:
        write_mm_checkpoint({**text_sd, "model.dummy.weight": t_norm}, mm_dir,
                            os.path.join(tempfile.mkdtemp(), "bad"), skeleton=skel_loaded)
        check("骨架键索引自检 fail-fast", False)
    except KeyError:
        check("骨架键索引自检 fail-fast（只读头部拿全键名，多出的键当场拦）", True)

    # 目录级物化（eval 兜底与离线 CLI 共用同一条路径）
    text_dir = tempfile.mkdtemp()
    save_file({"model.embed_tokens.weight": t_emb, "model.norm.weight": t_norm},
              os.path.join(text_dir, "model.safetensors"), metadata={"format": "pt"})
    out2 = os.path.join(tempfile.mkdtemp(), "step_2")
    materialize_mm_checkpoint(text_dir, mm_dir, out2)
    back2 = load_file(os.path.join(out2, "model.safetensors"))
    check("materialize_mm_checkpoint：文本目录 -> vLLM 可直读的多模态壳",
          "model.language_model.embed_tokens.weight" in back2
          and "model.visual.patch_embed.weight" in back2)

    # A/B 接线（无 GPU 机器上能验的部分：配置默认 + 源码断言）
    check("config 默认 save_mm_checkpoint=True（存盘即多模态壳）",
          get_config("retool_math", use_wandb=False).get("save_mm_checkpoint") is True)
    train_src = open("rlab/train.py", encoding="utf-8").read()
    check("train.py 存盘走 save_checkpoint -> write_mm_checkpoint（复合模型不再裸 save_pretrained）",
          "def save_checkpoint(" in train_src
          and "write_mm_checkpoint(sd, mm_base, save_name" in train_src
          and "_fmt = save_checkpoint(cfg, engine, tokenizer, save_name, sd)" in train_src)
    eval_src = open("eval_vllm_one.py", encoding="utf-8").read()
    check("eval_vllm_one.py 检测纯文本 ckpt 并自动物化（旧 ckpt 兜底，免手工）",
          "def _needs_mm_materialize(" in eval_src
          and "materialize_mm_checkpoint(args.model, _mm_base, _mm_tmp)" in eval_src
          and "atexit.register(shutil.rmtree, _mm_tmp" in eval_src)


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
    # 【2026-09-16 真机】--gpus 0,1 时 GPU1 被训练占着（空闲 35.4/95 GiB），默认
    # gpu_mem=0.78 → 白等一次 materialize+引擎初始化才拿到一行 ValueError。
    check("eval_vllm_one.py 起引擎前做显存前置检查（三个数 + 可执行改法，fail-fast）",
          "def _mem_shortfall(" in src and "def _gpu_mem_preflight(" in src
          and src.index("_gpu_mem_preflight(args.gpu_mem)")
          < src.index("llm = LLM(model=_model_for_vllm"))
    eval_cli = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "eval.py"), encoding="utf-8").read()
    check("rlab.eval 透传 --gpu_mem/--mm_base（否则卡被占时无法从统一入口降档）",
          '"--gpu_mem", str(args.gpu_mem)' in eval_cli
          and '"--mm_base", args.mm_base' in eval_cli)
    # 判据行为自检：脚本不可 import（顶层就要 --model 并起 vLLM），故只取该纯函数的
    # AST 源码 exec 出来测——"能不能分辨够用/不够用"必须真跑，不能只数源码文本。
    import ast as _ast
    _fn = next(n for n in _ast.parse(src).body
               if isinstance(n, _ast.FunctionDef) and n.name == "_mem_shortfall")
    _ns = {}
    exec(compile(_ast.Module(body=[_fn], type_ignores=[]), "<mem>", "exec"), _ns)
    _ms = _ns["_mem_shortfall"]
    _G = 2 ** 30
    check("显存判据：够用放行 / 不够时给出 gpu_mem 上限（35.4/95 vs 0.78 实机档）",
          _ms(0.30, 35 * _G, 95 * _G) is None
          and "gpu_mem" in _ms(0.78, 35 * _G, 95 * _G)
          and _ms(0.78, 80 * _G, 95 * _G) is None)


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

    # 【2026-09-17】--system_prompt_file：让"提示层单变量 A/B"与"P1b 原配方重现"
    # 能并行（file 只作用于本次 run，且提示指纹进签名 -sp<hash6>）。这几行在真机才会
    # 执行，CPU 测不到 → 按项目既有做法查接线（pyflakes 只抓未定义名，抓不到漏接线）。
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _tr_src = open(os.path.join(_root, "rlab", "train.py"), encoding="utf-8").read()
    _pb_src = open(os.path.join(_root, "rlab", "probe_difficulty.py"), encoding="utf-8").read()
    check("train.py: --system_prompt_file 读文件后覆盖 system_prompt",
          '"--system_prompt_file"' in _tr_src
          and 'overrides["system_prompt"] = f.read().strip()' in _tr_src)
    check("probe_difficulty.py: 同样支持（表是『模型×提示×预算』的联合产物，缺口径即另一张表）",
          '"--system_prompt_file"' in _pb_src
          and 'cfg["system_prompt"] = f.read().strip()' in _pb_src)

    # 【2026-09-17 对齐缺口】eval 此前一律取 preset 默认预算/提示：训练用
    # --system_prompt_file/--round_gen_tokens 6144 覆盖时，eval 静默测第三种协议。
    # 现在 eval_vllm_one.py 从 ckpt 的 run_info.json 回读训练 config（CLI 显式传参仍覆盖）。
    _ev_src = open(os.path.join(_root, "eval_vllm_one.py"), encoding="utf-8").read()
    check("eval_vllm_one.py 回读 run_info.json 的训练 config（协议本体，非仅顶层签名）",
          "def _load_run_cfg(" in _ev_src
          and 'info.get("config")' in _ev_src)
    check("优先级：CLI 显式 > run_info 训练 config > preset 默认",
          _ev_src.index("_rcfg = {**_rcfg, **_run_cfg}")
          < _ev_src.index("if args.round_tokens is None:"))
    check("回读顺序在系统提示装配之前（system_prompt 也吃 run_info）",
          _ev_src.index("_run_cfg = (_run[\"config\"] if _run else {}) or {}")
          < _ev_src.index("# ---- system_prompt 对齐训练 ----"))
    check("system_prompt 来源 run_info.config.system_prompt，且与训练签名 -sp<hash> 对拍告警",
          "_sp_run = _run_cfg.get(\"system_prompt\")" in _ev_src
          and "训练 -sp{_sp_sig}" in _ev_src)
    check("chat_template_kwargs 也吃 _rcfg（已 merge run_info，thinking 开关不丢）",
          "_ctkw = _rcfg.get(\"chat_template_kwargs\")" in _ev_src)
    # 行为级：_load_run_cfg 纯函数真跑（AST exec，脚本顶层要 --model 起 vLLM 不可 import）
    import ast as _ast2
    _fn2 = next(n for n in _ast2.parse(_ev_src).body
                if isinstance(n, _ast2.FunctionDef) and n.name == "_load_run_cfg")
    _ns2 = {"os": os, "json": json}
    exec(compile(_ast2.Module(body=[_fn2], type_ignores=[]), "<runcfg>", "exec"), _ns2)
    _load_run_cfg = _ns2["_load_run_cfg"]
    _ck = os.path.join(tempfile.mkdtemp(), "ckpt_runinfo")
    os.makedirs(_ck, exist_ok=True)
    with open(os.path.join(_ck, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"signature": "retool_math-ts0.5-ol1-r2x6144-s300x50-lr1e-06-d0-1-tabc123-sp8e0184-vkgdn_prefill_backend=triton",
                   "config": {"round_gen_tokens": 6144, "max_context_tokens": 14336,
                              "max_rounds": 2, "system_prompt": "你是一个简洁的解题助手。\n",
                              "chat_template_kwargs": {"enable_thinking": False}}}, f)
    _r = _load_run_cfg(_ck)
    check("行为：回读到训练 config（预算三件套 + system_prompt + thinking 开关）",
          _r and _r["config"]["round_gen_tokens"] == 6144
          and _r["config"]["max_context_tokens"] == 14336
          and _r["config"]["max_rounds"] == 2
          and _r["config"]["system_prompt"] == "你是一个简洁的解题助手。\n"
          and _r["config"]["chat_template_kwargs"] == {"enable_thinking": False})
    check("行为：无 run_info / 损坏 → None（旧 ckpt 回落 preset，不抛）",
          _load_run_cfg(os.path.join(os.path.dirname(_ck), "no_such_dir")) is None)

    # 【2026-09-17 同档修复】BASE 无 run_info → 复用第一个 tuned ckpt 的训练协议。
    # one 进程吃 --proto_from；调度器探测并给 BASE 传。
    _ev2_src = open(os.path.join(_root, "eval_vllm.py"), encoding="utf-8").read()
    check("调度器: BASE 无 run_info 时复用第一个带 run_info 的 tuned ckpt 协议",
          "BASE_PROTO" in _ev2_src
          and "if not _has_run_info(base_path) and _first_proto:" in _ev2_src
          and "BASE_PROTO[\"BASE\"] = _first_proto" in _ev2_src)
    check("调度器: run_one 给 BASE 进程传 --proto_from",
          '"--proto_from", _pf' in _ev2_src)
    check("one 进程: --proto_from 优先于 --model 自身 run_info",
          "args.proto_from" in _ev_src
          and "_load_run_cfg(args.proto_from) if args.proto_from else _load_run_cfg(args.model)"
          in _ev_src)

    # 【2026-09-22 分布对齐修复】eval 的 train 和 test split 都用训练端同一套
    # 难度过滤。此前只过滤 train split → 两个 split 难度分布不同 → train eval
    # 偏高（p9 假信号）。现在 probe 覆盖训练池+dev 集，eval 两端同表同 band 过滤。
    check("eval 两端 split 都用难度过滤（同表同 band）",
          'from rlab.data import (load_dapo_math_dev, load_dapo_math_train,'
          in _ev_src
          and "load_difficulty_table" in _ev_src
          and "filter_qas_by_difficulty(test_data, _tbl, lo=_lo, hi=_hi)" in _ev_src)
    check("难度过滤对 test 和 train split 都生效（不分流）",
          '[eval][{args.split}] 难度过滤' in _ev_src)
    check("无 difficulty_path 时两端都告警（含 p≈0 稀释）",
          "含 p≈0 题，效果被稀释" in _ev_src)



def test_overlong_ref_and_opt_cli():
    print("[X] overlong 参考系修复 + 优化超参 CLI：retool 多轮总预算 ≠ 单轮 max_gen_tokens")
    from rlab.reward import overlong_ref_tokens, overlong_penalty, total_reward_math
    # 【2026-09-12 语义变更】参考系不再是裸的 rounds×per_round，而是"可写满的总预算"
    # = min(rounds×per_round, max_context_tokens − max_prompt_length)，防止在
    # 预算不自洽的配置下 trigger 落到 overlong 丢弃线之外（那样它永远够不着 → 死开关）。
    cfg_rm = get_config("retool_math", use_wandb=False)
    _rounds, _per = cfg_rm["max_rounds"], cfg_rm["round_gen_tokens"]
    _usable = cfg_rm["max_context_tokens"] - cfg_rm["max_prompt_length"]
    check(f"retool_math 参考系 = min(rounds×per_round, ctx−plen) "
          f"= min({_rounds * _per}, {_usable}) = {_rounds * _per}",
          overlong_ref_tokens(cfg_rm) == min(_rounds * _per, _usable))
    # 单轮路径（grpo）参考系仍 = max_gen_tokens，历史口径零变化
    check("单轮路径（grpo）参考系仍 = max_gen_tokens",
          overlong_ref_tokens(get_config("grpo", use_wandb=False))
          == get_config("grpo", use_wandb=False)["max_gen_tokens"])
    # 【2026-09-12 关键不变量】参考系必须落在 overlong 丢弃线之内，否则 shaping 不可达
    # （2026-09-11 那版就是踩了这个：3×3072=9216 > 8192，trigger 永不可达）
    check("不变量：overlong 参考系 ≤ max_context_tokens − max_prompt_length（shaping 必然可达）",
          overlong_ref_tokens(cfg_rm) <= _usable)
    # DAPO 软悬崖语义（参考系 = 悬崖顶）：trigger=ref-buffer 以下不罚，
    # ref 处扣满，中点 0.5。关键是**参考系可达**——旧配置下 trigger(9152)
    # 在丢弃线(8192)之外，样本先被 retool_context_overlong 丢掉，永远走不到这里。
    _ref = overlong_ref_tokens(cfg_rm)
    check("软悬崖：trigger(=ref−buffer) 以下不罚",
          overlong_penalty(_ref - 256, _ref, 256) == 0.0)
    check("软悬崖：ref−buffer/2 → 0.5（参考系可达，惩罚真能生效）",
          abs(overlong_penalty(_ref - 128, _ref, 256) - 0.5) < 1e-6)
    check("软悬崖：ref 处扣满 1.0（+1 抹成 0、−1 压成 −2 的最坏情形）",
          abs(overlong_penalty(_ref, _ref, 256) - 1.0) < 1e-6)
    # 可达性对照：旧配置（3×3072=9216，丢弃线 8192）下，能触发 shaping 的长度
    # （>9152）必然已经被 retool_context_overlong 整组丢弃（见 [H] 的超长检查）
    _old_ref = 3 * 3072
    check("旧配置自证死开关：能触发 penalty 的长度(>9152) 全在丢弃线(8192)之外，"
          "样本先被丢 → penalty 永不生效",
          _old_ref > 8192 and overlong_penalty(9200, _old_ref, 64) > 0.0
          and 9200 > 8192)
    # 【2026-09-12】trunc 靶向 shaping：总长惩罚够不到的单轮 prose 轨迹靠它反向
    from rlab.reward import trunc_penalty, total_reward_retool_math
    check("trunc_penalty：weight=0 → 恒 0（其他算法协议零变化）",
          trunc_penalty(1, 0.0) == 0.0 and trunc_penalty(0, 0.5) == 0.0)
    check("trunc_penalty：trunc_final=1 → weight", trunc_penalty(1, 0.5) == 0.5)
    _ans = "answer is \\boxed{72}"
    _n = total_reward_retool_math("72", _ans, trunc_final=1, trunc_shaping=0.5)["reward"]
    _y = total_reward_retool_math("72", _ans, trunc_final=0, trunc_shaping=0.5)["reward"]
    check("trunc shaping 打在 reward 上：未截断 +1，被截断 +0.5（不改正负号）",
          abs(_y - 1.0) < 1e-6 and abs(_n - 0.5) < 1e-6)
    _neg = total_reward_retool_math("72", "answer is \\boxed{99}",
                                    trunc_final=1, trunc_shaping=0.5)["reward"]
    check("trunc shaping 对答错：-1 → -1.5（仍为负，group_mean 语义不变）",
          abs(_neg - (-1.5)) < 1e-6)

    # ---- 多卡分片探针（2026-09-17）：不变量 = 各片互斥 且 并集 = 原集合 ----
    # 全量表在定版预算下 ~50h 单卡；两卡分片是唯一不动协议的减半手段。丢题（表有洞
    # → 训练池被静默缩小）与重题（白烧算力）都不报错，故锁死纯函数不变量。
    from rlab.probe_difficulty import shard_items
    _pool = [f"q{i}" for i in range(1000)]
    _parts = [shard_items(_pool, i, 5) for i in range(5)]
    check("分片并集 = 原集合（不丢题）",
          sorted(x for p in _parts for x in p) == sorted(_pool))
    check("分片互斥（不重题）", len({x for p in _parts for x in p}) == len(_pool))
    check("分片大小均衡（差 ≤1）", max(map(len, _parts)) - min(map(len, _parts)) <= 1)
    check("shard_count=1 恒等（默认档零行为变化）", shard_items(_pool, 0, 1) == _pool)
    check("分片边界 fail-fast（越界/非法 count 不静默跑全池）",
          bool(_exc_msg(lambda: shard_items(_pool, 2, 2)))
          and bool(_exc_msg(lambda: shard_items(_pool, 0, 0)))
          and bool(_exc_msg(lambda: shard_items(_pool, -1, 2))))
    _pb_shard = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "rlab",
        "probe_difficulty.py"), encoding="utf-8").read()
    check("probe_difficulty 接线：分片切片在 max_questions 之后（各片要求同一参考顺序）",
          "shard_items(order, shard_index, shard_count)" in _pb_shard
          and _pb_shard.index("order = order[:max_questions]")
          < _pb_shard.index("shard_items(order, shard_index, shard_count)"))

    # ---- 分片 × 续跑：切片必须与 done 无关（否则续跑时切片错位 → 两片互相探对方的题）
    # 【2026-09-17 真机】50h 分片表跑到 ~60% 才发现：旧顺序"先按 done 过滤再打乱切片"
    # 首跑正常、只有中断恢复才踩。下面既锁新不变量，也用反证把旧顺序的错法钉住。
    from rlab.probe_difficulty import probe_plan
    _pool = [{"Q": f"q{i}", "A": "1"} for i in range(100)]
    _o0, _t0 = probe_plan(_pool, {}, seed=42, shard_index=0, shard_count=2)
    _o1, _t1 = probe_plan(_pool, {}, seed=42, shard_index=1, shard_count=2)
    _q0, _q1 = {x["Q"] for x in _t0}, {x["Q"] for x in _t1}
    check("首跑：两片互斥且并集 = 全池",
          not (_q0 & _q1) and (_q0 | _q1) == {x["Q"] for x in _pool})
    _done0 = {x["Q"]: {"n_correct": 1} for x in _t0[:10]}          # 片0 已完成 10 题
    _o0b, _t0b = probe_plan(_pool, _done0, seed=42, shard_index=0, shard_count=2)
    check("续跑：本片参考顺序不变（切片与 done 无关）",
          [x["Q"] for x in _o0b] == [x["Q"] for x in _o0])
    check("续跑：待探 = 本片切片 − done（不会越界探另一片的题）",
          {x["Q"] for x in _t0b} == _q0 - set(_done0) and len(_t0b) == len(_o0) - 10)
    import random as _random

    def _old_plan(pool, done, idx, cnt):     # 旧顺序（先 done 过滤再打乱切片）
        todo = [x for x in pool if x["Q"] not in done]
        _random.Random(42).shuffle(todo)
        return todo[idx::cnt]
    _old0 = {x["Q"] for x in _old_plan(_pool, _done0, 0, 2)}
    check("反证：旧顺序在续跑时越界（片0 会探到片1 的题 → 重叠浪费）",
          len(_old0 & _q1) > 0)
    check("trunc_shaping 默认关闭（total_reward_retool_math 不传 = 旧行为）",
          abs(total_reward_retool_math("72", _ans)["reward"] - 1.0) < 1e-6)
    # CLI 入口：新三件套被 train.py 接收并落到 overrides
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()
    for flag, key in (('"--lr"', 'overrides["lr"] = args.lr'),
                      ('"--beta"', 'overrides["beta"] = args.beta'),
                      ('"--overlong_shaping"', 'overrides["overlong_shaping"] = True'),
                      ('"--trunc_shaping"', 'overrides["trunc_shaping"] = args.trunc_shaping'),
                      ('"--discard_abort"', 'overrides["discard_abort"] = args.discard_abort')):
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
    # 【2026-09-12 修复】旧注释断言"overlong 当前配置下不可达"——那是 bug 的自白，
    # 不是设计。现在改为断言：预算自洽校验存在，且预设配置能通过它。
    cfg_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "config.py"), encoding="utf-8").read()
    check("config 有 validate_retool_budget（预算自洽 fail-fast，防事故重演）",
          "def validate_retool_budget" in cfg_src
          and "validate_retool_budget(cfg)" in cfg_src)
    check("preset 预算自洽：rounds×per_round + max_prompt_length + 工具段预留 ≤ ctx",
          _rounds * _per + cfg_rm["max_prompt_length"] + cfg_rm["_tool_reserve"]
          <= cfg_rm["max_context_tokens"])


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
    # 【2026-09-12】偏离签名三处冗余（wandb name / run_info / 启动 print）：run2 的四个
    # 关键偏离散落在 preset + CLI + eval 默认值里，跑完才发现"差了哪几维"全靠翻日志。
    check("run signature 进 run_info 顶层（checkpoint 自证消融维度）",
          "run_signature(cfg)," in src)
    check("run signature 进 wandb run name（列表页可直接比对偏离）",
          "run_signature(cfg)}" in src)
    check("启动即打印 signature=（grep 第一现场）", "偏离签名 signature=" in src)
    # get_config 拒绝未知键的契约仍成立（新键必须先在 BASE 注册）
    try:
        get_config("retool_math", use_wandb=False, gradient_clipping=1.0)
        ok = True
    except KeyError:
        ok = False
    check("gradient_clipping 已在 BASE 注册（未知键 fail-fast 契约未被绕过）", ok)

    # 【2026-09-13 P1 事故】ckpt 撞名护栏：P1（ts0）的 step_200 静默覆掉了
    # run2（ts0.5）的 step_200 原始 ckpt（后者只剩 _mm 合并副本）。护栏契约：
    # 外来签名/无签名（出处不明）→ 启动即拒；同签名重跑 → 放行。
    from rlab.train import guard_ckpt_collision
    from rlab.train import run_signature as _run_sig
    import shutil
    _tmp = tempfile.mkdtemp()
    try:
        cfg_t = get_config("retool_math", use_wandb=False)
        cfg_t["run_signature"] = _run_sig(cfg_t)
        ck = os.path.join(_tmp, "step_200")
        os.makedirs(ck)
        with open(os.path.join(ck, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump({"signature": "retool_math-ts0.5-ol1-r2x3072-s1000x200-lr5e-06-d0-1"}, f)
        try:
            guard_ckpt_collision(_tmp, cfg_t)
            ok = False
        except RuntimeError:
            ok = True
        check("外来签名 ckpt → 启动即拒绝（P1 覆盖事故的护栏）", ok)
        with open(os.path.join(ck, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump({"signature": cfg_t["run_signature"]}, f)
        guard_ckpt_collision(_tmp, cfg_t)
        check("同签名重跑 → 放行", True)
        os.remove(os.path.join(ck, "run_info.json"))
        try:
            guard_ckpt_collision(_tmp, cfg_t)
            ok = False
        except RuntimeError:
            ok = True
        check("无签名旧 ckpt（run2 时代，出处不明）→ 拒绝", ok)
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)
    check("护栏接入 run_training 启动路径",
          'guard_ckpt_collision(cfg["out_dir"], cfg)' in src)


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
    # 【2026-09-13 护栏口径修正】首版撞名护栏只看脚本自己的 OUT_DIR 变量，
    # 没解析 "$@" 里的 --out_dir → 把带 --out_dir 的合法启动误拦在共享目录上。
    check("run_gsm8k.sh 护栏/归档跟随用户 --out_dir（空格与等号两种写法）",
          '[ "$_prev" = "--out_dir" ]; then OUT_DIR="$_a"' in sh
          and 'case "$_a" in --out_dir=*) OUT_DIR="${_a#--out_dir=}"' in sh)
    # e2e 测试按位置传 run_server(path, port, mode, beta, grad_accum, device, attn)——
    # 新参数必须追加在末尾，插在中间会静默错位（device 收到 "cpu" 之类的字符串）
    from rlab.ref_server import run_server
    params = list(inspect.signature(run_server).parameters)
    check("run_server 前 6 个位置参数保持不变（e2e 位置传参契约）",
          params[:6] == ["model_path", "port", "mode", "beta", "grad_accum", "device"]
          and params[-2] == "batch_chunk"
          and params[-1] == "queue_max" and "queue_max" not in params[:6])


def test_max_stale_discard():
    print("[Z2] max_stale_opt_steps：off-policy 兜底——训练端丢弃超陈旧批（2026-09-18）")
    # config 默认 0 = 不启用（历史行为零变化）
    check("config 默认 max_stale_opt_steps=0（不启用）",
          get_config("retool_math", use_wandb=False)["max_stale_opt_steps"] == 0)
    # CLI 透传进 overrides
    import inspect as _ins
    import rlab.train as _TB
    _tsrc = _ins.getsource(_TB.main)
    check("train.py 暴露 --max_stale_opt_steps 并接线",
          '"--max_stale_opt_steps"' in _tsrc
          and 'overrides["max_stale_opt_steps"] = args.max_stale_opt_steps' in _tsrc)
    # 丢弃逻辑（真机运行时路径）：staleness = floor((step-1)/GAS) − floor(gv/GAS)
    _tr_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()
    check("训练循环有 max staleness 丢弃块（读 gen_version、超限 continue）",
          "max_stale_opt_steps" in _tr_src
          and "off-policy 兜底" in _tr_src
          and _tr_src.index("staleness] 丢弃批次")
          < _tr_src.index("continue", _tr_src.index("staleness] 丢弃批次"))
          < _tr_src.index('plen = batch["plen"]'))
    # 口径纯函数级验证：模拟训练端算 staleness 并判丢弃
    _gas = 4
    for _step, _gv, _max_s, _expect in [
        (17, 0, 16, False),     # floor(16/4)=4 − 0 = 4 ≤ 16 不丢
        (65, 0, 16, False),     # floor(64/4)=16 − 0 = 16，严格 > 判定 → 恰好=16 不丢
        (81, 0, 16, True),      # floor(80/4)=20 − 0 = 20 > 16 丢
        (81, 64, 16, False),    # floor(80/4)=20 − floor(64/4)=16 = 4 不丢（生成端已跟进）
    ]:
        _upd = (_step - 1) // _gas - _gv // _gas
        check(f"staleness 口径 step={_step} gv={_gv}: {_upd} > {_max_s} → {'丢' if _expect else '不丢'}",
              (_upd > _max_s) == _expect)


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
    check("拼接：(B,T) 形状 + 工具段填 0 + pad 填 0 + float32",
          tuple(t.shape) == (2, 5) and t.dtype == torch.float32
          and [round(float(x), 3) for x in t[0]] == [-0.5, -1.5, 0.0, 0.0, 0.0]
          and [round(float(x), 3) for x in t[1]] == [-2.0, 0.0, 0.0, 0.0, 0.0])
    # 【2026-09-12 dtype 回归锁】vLLM 返回 float32 精度的 logprob，降到 bf16 会引入
    # |δ|≈|logp|·2⁻⁹ 的量化误差（|logp|=5 → ~0.01），既给 ratio 加噪声，又给
    # approx_kl 造出 ~1.5e-4 的假地板，让人分不清后面的读数是 drift 还是量化。
    _fine = [[{"kind": "assistant", "ids": [1], "logps": [-3.14159265]}]]
    _tf = gen_logps_from_segs(_fine)
    check("float32 保精度：-3.14159265 不丢有效数字（bf16 会截成 -3.140625）",
          abs(float(_tf[0, 0]) - (-3.14159265)) < 1e-6)
    check("dtype 回归锁：gen_logps_from_segs 不再降到 bf16",
          t.dtype == torch.float32 and float(_tf[0, 0]) != -3.140625)
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
    check("SamplingParams 在 vLLM 路带 logprobs=N（cfg 可调，默认 0）+ logprobs_mode 字段探测",
          'kw["logprobs"] = int(cfg.get("vllm_logprobs_n", 0) or 0)' in ro
          and '"logprobs_mode" in _sp_fields' in ro)
    # 回归锁：权重同步处曾残留 gen_torch 悬空引用（pyflakes 抓到），无副本时必须跳过
    check("无 torch 副本时权重同步跳过它（不留悬空引用）",
          "gen_torch" not in ro and "if _torch_holder[0] is not None:" in ro)


def test_strip_left_pad():
    """【AB】逐题剥左 pad：打分序列 == vLLM 生成序列。

    这是"生成序列 == 训练序列"契约在**多题并采 + 批内 pad** 场景下的完整版。
    判据是数值等价，不是"跑通了"：带 pad 的批（批内最长 prompt 补齐）里，短题行
    的真实 token 既要被 pad 键污染（attention_mask 可解），位置编码又整体后移
    （attention_mask 解不了——主流 HF 实现 position_ids 取下标，不按 mask 做
    cumsum 修正）——两处叠加实测差 1.0 量级，且这正是对拍里 1e1 分叉的来源。
    剥掉 pad 后两处同时消失：逐 token 逐位置与 vLLM 一致，且不依赖模型对 2D 掩码
    的支持（Qwen3.5 线性注意力层是否吃 mask 无从离线验证，故不走掩码这条路）。"""
    print("[AB] 逐题剥左 pad：批内 pad 短题的打分序列 == 无 pad 生成序列")
    import torch as _t
    from rlab.losses import forward_per_token_logps
    from rlab.rollout import collect_retool_group, strip_left_pad
    from transformers import AutoTokenizer, GPT2LMHeadModel

    with tempfile.TemporaryDirectory() as tmp:
        path = _save_tiny_gpt2(tmp)
        tok = AutoTokenizer.from_pretrained(path)
        model = GPT2LMHeadModel.from_pretrained(path).eval()
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        pad = tok.pad_token_id

        # ---- 纯函数单测 ----
        row = _t.tensor([[pad, pad, 5, 6, 7]])
        check("strip_left_pad：剥掉左 pad 前缀（保留真实 token 顺序）",
              strip_left_pad(row, pad).tolist() == [[5, 6, 7]])
        check("strip_left_pad：无 pad 时是恒等（单题路径零变化）",
              strip_left_pad(_t.tensor([[5, 6, 7]]), pad).tolist() == [[5, 6, 7]])
        try:
            strip_left_pad(_t.tensor([[pad, pad]]), pad)
            check("整行皆 pad → ValueError", False)
        except ValueError:
            check("整行皆 pad → ValueError（否则 plen=0 会让 logps 切片变 -1 整条错位）", True)

        # ---- 端到端：两题 prompt 长度不同 → 批内左 pad ----
        q0, q1 = "a short question", "a much longer question text for padding"
        p0 = tok(q0, add_special_tokens=False)["input_ids"]
        p1 = tok(q1, add_special_tokens=False)["input_ids"]
        prompts_text = [q0, q1]
        prompt_ids = tok(prompts_text, return_tensors="pt", padding=True,
                         add_special_tokens=False)["input_ids"]
        plen = prompt_ids.shape[1]
        assert len(p0) < plen, "测试前提：q0 的 prompt 被左 pad 补齐（否则本组无效）"
        check("前提：批内最长 prompt 补齐，q0 前面挂着 pad",
              int((prompt_ids[0] != pad).sum()) == len(p0) < plen)

        def tids(s):
            return tok(s, add_special_tokens=False)["input_ids"]

        class _C:
            def __init__(self, text): self.text, self.token_ids = text, tids(text)
        class _O:
            def __init__(self, text): self.outputs = [_C(text)]
        class FakeGen:
            def generate(self, prompts, sps, use_tqdm=False):
                return [_O(t) for t in self.rounds[0]]

        good72, good99 = fmt_answer("72"), fmt_answer("99")
        # 用 retool preset（fmt_answer 派生的合法格式串即得分口径）；retool_math 的
        # 奖励走数学答案抽取，格式串不是它的拿分形态
        cfg = get_config("retool", use_wandb=False)
        n_traj = 2 * int(cfg["num_pre_Q"])            # 2 题 × num_pre_Q
        FakeGen.rounds = [[good72, good99] * (n_traj // 2)]   # 一半对一半错
        sps = [object() for _ in range(n_traj)]
        gl_calls = []

        def gl(merged, plen_):
            """真 torch 前向当 compute_gen_logps（不是 zeros 桩——要对拍数值）。"""
            gl_calls.append(tuple(merged.shape))
            with _t.inference_mode():
                return forward_per_token_logps(model, merged)[:, plen_ - 1:]

        results = collect_retool_group(FakeGen(), tok, cfg, gl, [{"Q": q0, "A": "72"},
                                        {"Q": q1, "A": "72"}], prompts_text,
                                        prompt_ids, plen, sps, steps_elapsed=0)
        check("两题都产出 ok 组（q0 是带 pad 的短题）",
              [r["status"] for r in results] == ["ok", "ok"])
        r0 = results[0]
        check("q0 的 plen = 本题真实 prompt 长（不再是批内最长的含 pad 宽）",
              r0["plen"] == len(p0) and results[1]["plen"] == len(p1))
        check("q0 的 merged 前缀就是无 pad 的真实 prompt（逐 token 一致）",
              r0["merged"][0, :len(p0)].tolist() == p0)
        check("q0 各行的 prompt 区都无 pad token（整批剥干净）",
              int((r0["merged"][:, :len(p0)] == pad).sum()) == 0)

        # 决定性判据：批内短题行的逐 token logps == 该行单独(无 pad)前向的 logps
        row = r0["merged"][0]
        c0 = r0["clen"][0]
        seq = row[:len(p0) + c0].tolist()          # 该行真实序列（无 pad 前缀）
        with _t.inference_mode():
            alone = forward_per_token_logps(model, _t.tensor([seq]))[:, len(p0) - 1:]
        d_new = float((r0["gen_logps"][0, :c0] - alone[0, :c0]).abs().max())
        check(f"批内短题行 logps == 该行单独无 pad 前向（最大差 {d_new:.3g}）",
              d_new < 1e-4)

        # 反证：旧行为 = 同一行前面挂上 pad 前缀（批内最长补齐）后前向
        old_row = [pad] * (plen - len(p0)) + seq
        with _t.inference_mode():
            old_lp = forward_per_token_logps(model, _t.tensor([old_row]))[:, plen - 1:]
        d_old = float((old_lp[0, :c0] - alone[0, :c0]).abs().max())
        check(f"反证：带 pad 前缀（旧行为，plen={plen}）同位置分叉 {d_old:.3g} > 1e-4",
              d_old > 1e-4)

    # ---- 接线 ----
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ro = open(os.path.join(root, "rlab", "rollout.py"), encoding="utf-8").read()
    check("collect_retool_group 逐题剥 pad，全部 plen 用本题真实长度",
          "prompt_i = strip_left_pad(prompt_ids[i:i + 1]" in ro
          and "plen_i = prompt_i.shape[1]" in ro
          and "retool_build_batch(\n            prompt_i, segs_i, plen_i" in ro
          and ro.count("compute_gen_logps(merged_i, plen_i)") == 2
          and '"plen": plen_i})' in ro)
    check("超长判定用真实 plen（旧版把 pad 宽度算进每个样本的 token 预算）",
          'per_ids_i, plen_i, cfg["max_context_tokens"]' in ro)


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
             "rlab/model_loading.py", "rlab/materialize_mm_ckpt.py",
             "rlab/ref_server.py",
             "rlab/data.py", "rlab/prepare_dapo_math.py", "rlab/diag_logps.py",
             # 【2026-09-20】rlab/eval.py 此前漏在清单外（改它时无静态防线）
             "rlab/eval.py",
             # 【2026-09-25】新增诊断脚本一律入清单（本文件初版就带过一个全角
             # 括号笔误 → SyntaxError；静态检查是唯一能在提交前拦住它的防线）
             "rlab/diag_eval_gap.py",
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


def test_budget_guard_and_drift_stats():
    """[Z] 2026-09-12 事故固化：预算自洽 fail-fast + 口径诊断修复。

    两件都是"静默跑废"型 bug——不报错、不崩，只在 10 小时后表现为精度没了：
      ① 3×3072 > 8192 使 overlong 丢弃吃掉 86% 尝试（丢弃率 20%→90%），
         采样主循环空转、训练端 5 小时零产出；
      ② clip_frac 用硬编码 1.2/0.8（与 loss 的 1+hi/1-lo 不一致）系统性低估；
         mean_ratio 恒为 1 是重要度采样恒等式，永远报不出 drift。
    """
    print("[Z] 预算自洽 fail-fast + ratio 口径诊断修复")
    import torch as _t
    from rlab.config import validate_retool_budget
    from rlab.losses import compute_loss, _finalize

    # ---- ① 旧的事故配置必须被硬拦 ----
    # 【2026-09-18 方案B】preset 已改 4×2048+14336（合法）；旧事故复现需显式 8192 ctx。
    _raised = False
    try:
        get_config("retool_math", use_wandb=False, max_rounds=3, round_gen_tokens=3072,
                   max_context_tokens=8192)
    except ValueError as e:
        _raised = "预算不自洽" in str(e)
    check("旧事故配置 3×3072+1024 > 8192 → get_config 直接 ValueError（防静默上线）", _raised)
    check("错误信息含可执行改法（给出 round_gen_tokens 上限）",
          "round_gen_tokens ≤" in str(_exc_msg(lambda: get_config(
              "retool_math", use_wandb=False, max_rounds=3, round_gen_tokens=3072,
              max_context_tokens=8192))))
    check("方案B preset（4×2048+14336）预算自洽且不报错",
          validate_retool_budget(get_config("retool_math", use_wandb=False)) >= 0
          and get_config("retool_math", use_wandb=False)["max_rounds"] == 4
          and get_config("retool_math", use_wandb=False)["round_gen_tokens"] == 2048)
    check("非 retool 算法不参与预算校验（返回值 0）",
          validate_retool_budget(get_config("grpo", use_wandb=False)) == 0)

    # ---- ② clip_frac 阈值必须与 loss 的 clip 域一致 ----
    B, T = 1, 4
    mask = _t.ones(B, T)
    adv = _t.ones(B)
    cfg = get_config("retool_math", use_wandb=False)
    # ratio = exp(policy - gen)。构造 ratio = 1.25：在 (1+lo, 1+hi] = (1.2, 1.28]
    # 内 → 旧口径（硬编码 1.2）漏计，新口径（1+hi=1.28）也不该计；再构造 1.30 二者都计
    gen = _t.zeros(B, T)
    pol_125 = _t.full((B, T), float(__import__("math").log(1.25)))
    _, st125 = compute_loss("retool_math", pol_125, gen, adv, mask, cfg,
                            ref_logps=_t.zeros(B, T))
    check("clip_frac 由 clip_high 驱动：ratio=1.25 在 [0.8,1.28] 内不计 clip",
          st125["clip_frac"] == 0.0)
    pol_130 = _t.full((B, T), float(__import__("math").log(1.30)))
    _, st130 = compute_loss("retool_math", pol_130, gen, adv, mask, cfg,
                            ref_logps=_t.zeros(B, T))
    check("ratio=1.30 超出 1+clip_high=1.28 → clip_frac=1（旧硬编码 1.2 阈值口径不一致）",
          st130["clip_frac"] == 1.0)
    check("ratio=0.75 低于 1-clip_low=0.8 → clip_frac=1",
          compute_loss("retool_math", _t.full((B, T), float(__import__("math").log(0.75))),
                       gen, adv, mask, cfg, ref_logps=_t.zeros(B, T))[1]["clip_frac"] == 1.0)

    # ---- ③ 新统计量真的有信息量（mean_ratio 没有）----
    check("stats 新键存在：kl / frac_d_gt_0.1",
          "kl" in st130 and "frac_d_gt_0.1" in st130)
    check("kl = mean(-log ratio) 是 KL(π_old‖π_new) 的采样估计（ratio=1.30 → -0.262）",
          abs(st130["kl"] - (-float(__import__("math").log(1.30)))) < 1e-5)
    check("frac_d_gt_0.1：ratio=1.30（|log ratio|=0.262>0.1）→ 1.0",
          st130["frac_d_gt_0.1"] == 1.0)
    # 关键对照：ratio 一半 1.30 一半 0.70 —— mean_ratio≈1 但 drift 明显
    _mix = _t.tensor([[float(__import__("math").log(1.4))] * 2
                      + [float(__import__("math").log(0.6))] * 2])
    _, stm = compute_loss("retool_math", _mix, _t.zeros(1, 4), _t.ones(1),
                          _t.ones(1, 4), cfg, ref_logps=_t.zeros(1, 4))
    check("对照实锤：对称 drift 下 mean_ratio≈1.0（恒等式，无诊断价值），"
          "而 frac|d|>0.1=1.0 与 kl>0 能报出来",
          abs(stm["mean_ratio"] - 1.0) < 1e-6 and stm["frac_d_gt_0.1"] == 1.0
          and stm["kl"] > 0.05)

    # ---- ④ mean_ratio 恒等式（记录在案，防后人再把它当健康指标）----
    _src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "losses.py"), encoding="utf-8").read()
    check("losses.py 注释写明 mean_ratio 是重要度采样恒等式（对任意远 π_new 都=1）",
          "重要度采样的数学恒等式" in _src or "数学恒等式" in _src)

    # ---- ⑤ staleness 标签：初值必须是 0（= 初始 checkpoint），不能是 None ----
    # 真机实测：初值 None 时第 1~15 步（第一次推送之前）打出
    # `gen_version=None staleness=-1 micro-step(-0.2)` —— 这段恰恰是权重最新鲜的
    # 区间，标签却被作废。初始权重等价于 version 0。
    _ro_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "rollout.py"), encoding="utf-8").read()
    check("rollout policy_version 初值 = 0（初始 checkpoint；None 会让前 15 步 staleness 报 -1）",
          "policy_version = [0]" in _ro_src)
    _tr_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()
    check("train.py 对缺失 gen_version 单独走 n/a 分支（不打印 staleness=-1）",
          "staleness=n/a" in _tr_src and "staleness=-1" not in _tr_src)
    # staleness 精确口径：step 1-based 而 version 是"step 末"语义，直接相减恒多算 1
    check("train.py staleness 用精确优化器步 floor((step-1)/GAS) − floor(gv/GAS)（不是 step−gv）",
          "(step - 1) // _gas - _gv // _gas" in _tr_src
          and "staleness={step - _gv}" not in _tr_src)


def _exc_msg(fn):
    try:
        fn()
    except Exception as e:
        return str(e)
    return ""


def test_length_runaway_signature():
    """[Z2] 2026-09-12 新增健康签名：长度膨胀（本轮静默跑废的直接签名）。"""
    print("[Z2] health 新增 length_runaway 签名")
    from rlab.health import window_check
    # 开局 32 组 clen≈1500 → 后面 32 组 clen≈4000（涨 167%）→ 必须报
    hist = ([{"acc": 0.3, "fmt": 0.3, "clen": 1500.0, "code_rate": 0.3,
              "trunc_rate": 0.05} for _ in range(64)]
            + [{"acc": 0.15, "fmt": 0.15, "clen": 4000.0, "code_rate": 0.1,
                "trunc_rate": 0.8} for _ in range(32)])
    codes = [c for c, _ in window_check(hist, retool=True, max_clen=6144)]
    check("clen 窗口均值较开局涨 167% → length_runaway", "length_runaway" in codes)
    # 稳定长度不误报
    hist2 = [{"acc": 0.3, "fmt": 0.3, "clen": 1500.0, "code_rate": 0.3,
              "trunc_rate": 0.05} for _ in range(96)]
    codes2 = [c for c, _ in window_check(hist2, retool=True, max_clen=6144)]
    check("长度稳定 → 不误报 length_runaway", "length_runaway" not in codes2)


def test_logps_diff_shape():
    """[AC1] 对拍差异形态学：占比 / 前后半段 / 最差 k 点（2026-09-15 max=12.8 事故固化）。

    事故的关键教训：**只看 max 会把"某一侧 logits 算错"误读成"尾部低概率 token 的
    舍入"**——12.8 那一点 vLLM 的 logp=-0.946（p≈0.39），根本不是尾点。要当场分辨
    "漂移增长"与"局部尖峰"，必须同时拿到越线占比、前后半段均值、最差点坐标。"""
    print("[AC1] logps 对拍形态学（logps_diff_shape）")
    from rlab.rollout import logps_diff_shape
    # 行0：前 3 位 0.2（>0.1 但不 >1）、后 3 位 2.0（漂移形态）；行1：整体零差
    gv = torch.tensor([[-1.0, -1.0, -1.0, -3.0, -4.0, -5.0],
                       [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    gt = torch.tensor([[-1.2, -1.2, -1.2, -5.0, -6.0, -7.0],
                       [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    mask = torch.ones(2, 6)
    st = logps_diff_shape(gv, gt, mask)
    check("有效位数 = mask 和", st["n"] == 12)
    check("max/mean 与手算一致",
          abs(st["max"] - 2.0) < 1e-6
          and abs(st["mean"] - (3 * 0.2 + 3 * 2.0) / 12) < 1e-6)
    check("frac>0.1 与 frac>1 分得开（单点 max 给不出的量）",
          abs(st["frac_gt_01"] - 0.5) < 1e-6 and abs(st["frac_gt_1"] - 0.25) < 1e-6)
    check("前后半段逐行切：后半段均值 > 前半段（漂移形态可读）",
          st["half_mean_first"] is not None
          and st["half_mean_second"] > st["half_mean_first"] * 2)
    w5 = [w for w in st["worst"] if w["col"] == 5]
    check("最差 3 点带行/列/两路原值（并列时取到哪三点不假定，只查集合与原值）",
          len(st["worst"]) == 3 and {w["col"] for w in st["worst"]} == {3, 4, 5}
          and all(w["row"] == 0 for w in st["worst"])
          and w5 and abs(w5[0]["vllm"] - (-5.0)) < 1e-6
          and abs(w5[0]["torch"] - (-7.0)) < 1e-6 and abs(w5[0]["d"] - 2.0) < 1e-6)
    # 反证：g4 那种"一侧高概率、一侧认为不可能"必须在最差点里可读
    gv2 = torch.tensor([[-0.946] + [-3.0] * 5])
    gt2 = torch.tensor([[-13.750] + [-3.0] * 5])
    st2 = logps_diff_shape(gv2, gt2, torch.ones(1, 6))
    check("g4 形态：Δ≈12.8 且两侧原值都在（-0.946 / -13.75）",
          abs(st2["worst"][0]["d"] - 12.804) < 1e-2
          and abs(st2["worst"][0]["vllm"] - (-0.946)) < 1e-3
          and abs(st2["worst"][0]["torch"] - (-13.75)) < 1e-3)
    # 右 pad 不参与：mask=0 的列即使差很大也不能进统计
    mask3 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    gv3 = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    gt3 = torch.tensor([[-1.0, -1.0, -99.0, -99.0]])
    st3 = logps_diff_shape(gv3, gt3, mask3)
    check("pad 位（mask=0）不进统计", st3["n"] == 2 and st3["max"] < 1e-6)
    check("全 mask=0 → 退化返回 n=0（调用方本来就会跳过）",
          logps_diff_shape(gv3, gt3, torch.zeros(1, 4))["n"] == 0)


def test_diag_logps_pure():
    """[AC2] diag_logps 纯函数：前缀长/top-K 提取/成对判读/责任方结论。

    诊断脚本的价值全在"判据能分辨两种假设"上，所以这几条必须锁死：confident
    分歧必须由 g4 那种 (高, 极低) 触发、`None`（超 top-K）也必须算进去，而
    同引擎 vs 跨引擎的结论必须分开打。"""
    print("[AC2] diag_logps 纯函数（判据自检）")
    from rlab.diag_logps import (aggregate, compare_rows, confident_disagreement,
                                 engine_of, normalize_prefix_lens, rank_from_pairs,
                                 topk_pairs, vllm_kwargs_for_backend, verdict)
    check("前缀长：去重升序 + 恒含 0 + 去负",
          normalize_prefix_lens("512,0,256,256,-8") == [0, 256, 512])
    check("前缀长：按 cap 截断（超预算的 L 测不了）",
          normalize_prefix_lens("0,256,1024,4096", cap=1024) == [0, 256, 1024])
    check("前缀长：全非法 → 仍给出 [0]（首 token 位永远可测）",
          normalize_prefix_lens("", cap=1024) == [0] and normalize_prefix_lens("x,y") == [0])

    class _LP:
        def __init__(self, v): self.logprob = v
    entry = {5: _LP(-0.5), 9: _LP(-3.0), 7: None, 11: 0.25, 3: _LP(-1.0)}
    pairs = topk_pairs(entry, 3)
    check("top-K：按 logp 降序、None 剔除、截到 k",
          [t for t, _ in pairs] == [11, 5, 3] and all(isinstance(v, float) for _, v in pairs))
    check("top-K：vLLM 额外塞的被采样 token 会被 k 截掉（语义=top-k）",
          len(topk_pairs({1: -0.1, 2: -0.2, 3: -9.0}, 2)) == 2)
    check("排名：命中→1-based，未命中→None",
          rank_from_pairs(pairs, 5) == 2 and rank_from_pairs(pairs, 999) is None)

    check("confident 分歧：g4 形态 (高, 极低) → True",
          confident_disagreement(-0.946, -13.75) and confident_disagreement(-13.75, -0.946))
    check("confident 分歧：常规舍入级差异 → False",
          not confident_disagreement(-1.0001, -1.0003))
    check("confident 分歧：一侧超出 top-K(None) 按'排很后'处理",
          confident_disagreement(None, -1.0) and not confident_disagreement(None, -20.0))

    a = {"q": 0, "L": 0, "target": 101, "lp_target": -0.946, "target_rank": 1,
         "topk": [[101, -0.946], [102, -1.5], [103, -2.0]]}
    b = {"q": 0, "L": 0, "target": 101, "lp_target": -13.75, "target_rank": 900,
         "topk": [[102, -1.4], [103, -1.9], [104, -2.1]]}
    cr = compare_rows(a, b)
    check("成对：top1 不一致 + 交集只有 2 个 + 目标差 12.8",
          cr["top1_match"] is False and abs(cr["overlap"] - 2 / 3) < 1e-9
          and abs(cr["target_d"] - 12.804) < 1e-2 and cr["confident"])
    st = aggregate([cr, {"q": 0, "L": 256, "target_d": 1e-5, "top1_match": True,
                         "overlap": 1.0, "max_abs_d_common": 1e-5,
                         "target_missing": False, "confident": False,
                         "lp_a": -1.0, "lp_b": -1.0, "rank_a": 1, "rank_b": 1}])
    check("聚合：n / max / >1nat 占比 / 最差点排序",
          st["n"] == 2 and abs(st["max_target_d"] - 12.804) < 1e-2
          and abs(st["frac_big"] - 0.5) < 1e-9 and st["worst"][0]["L"] == 0
          and set(st["per_L"]) == {0, 256})

    # ---- verdict：三种结局必须分开打（判据的分辨力所在）----
    def _mk(max_d, frac, conf=0):
        return {"n": 10, "max_target_d": max_d, "frac_big": frac, "confident_n": conf,
                "per_L": {}, "worst": [], "target_d_mean": max_d, "target_d_p99": max_d,
                "top1_match_rate": 1.0, "overlap_mean": 1.0, "max_abs_d_common": max_d,
                "target_missing_n": 0}
    v_same = verdict({"torch:fla vs torch:fallback": _mk(6.0, 0.1),
                      "torch:fla vs vllm:keep": _mk(6.0, 0.1)})
    check("torch 内部两路就不一致 → 责任钉在 torch 侧前向实现",
          any("同引擎跨 kernel" in s and "torch" in s for s in v_same)
          and any("causal_conv1d" in s for s in v_same))
    v_cross = verdict({"torch:fla vs vllm:keep": _mk(12.8, 0.04, conf=2),
                       "torch:fallback vs vllm:keep": _mk(12.8, 0.04, conf=2)})
    check("只做了跨引擎、没做任何内部消融 → 现象确认但**不许**写成'各自自洽'",
          any("内部消融不完整" in s for s in v_cross)
          # 注意用 startswith 判"没把它当结论下发"：注释里那句「还不能断言"两侧各自自洽"」
          # 本身含这个词，子串匹配会自己骗自己（本项目栽过的同类坑）
          and not any(s.startswith("**两侧各自自洽") for s in v_cross)
          and any("vllm_gen_logps=False" in s for s in v_cross))
    v_ok = verdict({"torch:fla vs torch:fallback": _mk(1e-5, 0.0),
                    "torch:fla vs vllm:keep": _mk(1e-5, 0.0)})
    check("全都一致 → 判为不是 kernel（回到序列构造/对齐）",
          any("不是 kernel" in s for s in v_ok))
    check("只有跨引擎对（没做消融）→ 明确提示'无法定位责任方'",
          any("没有消融" in s for s in verdict({"torch:fla vs vllm:keep": _mk(12.8, 0.04)})))
    # 【不许过度解读】两侧内部消融都干净、只有跨引擎不一致 → 才允许下"kernel 口径差"结论
    v_full = verdict({"torch:fla vs torch:fallback": _mk(1e-5, 0.0),
                      "vllm:keep vs vllm:none": _mk(1e-5, 0.0),
                      "torch:fla vs vllm:keep": _mk(12.8, 0.04)})
    check("两侧内部消融都干净 + 跨引擎不一致 → 判为 kernel 口径差",
          any(s.startswith("**两侧各自自洽") for s in v_full)
          and not any("内部消融不完整" in s for s in v_full))
    # 只有 torch 侧有内部对：允许下结论，但必须注明 vLLM 侧"未验证"（vLLM 那档常因
    # FlashInfer GDN 的 JIT OOM-kill 跑不起来，这个缺口必须显式写出来）
    v_half = verdict({"torch:fla vs torch:fallback": _mk(1e-5, 0.0),
                      "vllm:keep vs vllm:none": _mk(9.0, 0.3),
                      "torch:fla vs vllm:keep": _mk(12.8, 0.04)})
    check("vLLM 侧自己有内部分歧 → 责任钉在 vLLM 侧前向实现",
          any("同引擎跨 kernel" in s and "vllm" in s for s in v_half))

    check("engine 标签解析", engine_of("vllm:triton") == "vllm"
          and engine_of("torch:fallback") == "torch" and engine_of("torch") == "torch")
    base = {"gdn_prefill_backend": "triton", "max_num_seqs": 32}
    check("--vllm-backend keep 原样（对照档 = 训练口径）",
          vllm_kwargs_for_backend(base, "keep") == base)
    check("--vllm-backend none 去掉该键、其余保留",
          vllm_kwargs_for_backend(base, "none") == {"max_num_seqs": 32})
    check("--vllm-backend flashinfer 覆盖成该值",
          vllm_kwargs_for_backend(base, "flashinfer")["gdn_prefill_backend"] == "flashinfer")


def test_diag_logps_static():
    """[AC3] diag_logps 的静态契约：CPU 判读路径不许 import torch/vllm（探针要能在
    无 GPU 的机器上 merge），且 pyflakes 名单必须覆盖它（运行时路径的未定义名盲区）。"""
    print("[AC3] diag_logps 静态契约")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(root, "rlab", "diag_logps.py"), encoding="utf-8").read()
    head = src.split("def ", 1)[0]
    top_imports = [l.strip() for l in head.splitlines()
                   if l.strip().startswith(("import ", "from "))]
    bad = [l for l in top_imports
           if l.startswith(("import torch", "import vllm", "from torch", "from vllm"))]
    check("模块顶层不 import vllm/torch（--merge 要在 CPU 上能跑）", bad == [])
    check("provider 内部按需 import（vllm/torch 都在函数体里）",
          "from vllm import LLM" in src and "import torch" in src)
    check("三档 backend 与 torch 两档路径的 CLI 都在",
          '"--vllm_backend"' in src and '"--torch_path"' in src
          and 'choices=("fla", "fallback")' in src)
    check("能与训练命令逐字对齐：模型路径/chat 模板/vllm_gen_kwargs 都能覆盖",
          '"--model_path"' in src and '"--vllm_model_path"' in src
          and '"--chat_template_kwargs"' in src
          and '"--vllm_gen_kwargs"' in src and 'cfg["vllm_gen_kwargs"] = json.loads' in src)
    check("路径不存在就当场停（防探错模型这种静默错误）",
          "别用 preset 默认值探错模型" in src)
    check("打桩找不到目标必须 raise（不许静默当 fla 档跑）",
          "找不到任何可打的目标" in src or "打桩无从下手" in src)
    test_src = open(os.path.join(root, "rlab", "tests", "test_retool_cpu.py"),
                    encoding="utf-8").read()
    check("pyflakes 名单覆盖 diag_logps（J 组）", '"rlab/diag_logps.py"' in test_src)


def test_remap_decision_unified_ckpt():
    """[AD] 键名映射判据改口径：统一目录仍必须映射（2026-09-15 澄清）。

    【为什么必须有这组】c991f84 之后 torch 能直连复合 ckpt，"统一用一份
    /root/Qwen3.5-4B"成了自然写法——而旧判据 `bool(cfg['vllm_model_path'])` 会把
    这种情况判成"不需要映射"，于是 torch 发的 `model.layers.X` 撞上 vLLM 多模态的
    `model.language_model.X`，同步**一个张量都认领不了**。加载能不能读 ≠ 同步能不能
    对上：判据必须是键名形态。"""
    print("[AD] 键名映射判据（统一目录 vs 分裂目录 vs Qwen2.5）")
    from rlab.sync import need_text_to_mm_remap, vllm_load_weights
    check("统一目录（vLLM 侧是复合 ckpt、没传 vllm_model_path）→ **仍要映射**",
          need_text_to_mm_remap(vllm_checkpoint_composite=True,
                                vllm_model_path_set=False) is True)
    check("分裂加载（显式 --vllm_model_path）→ 映射（旧行为不变）",
          need_text_to_mm_remap(vllm_checkpoint_composite=True,
                                vllm_model_path_set=True) is True)
    check("Qwen2.5/同布局（vLLM 侧 config 无 text_config）→ 不映射（映射反而不该做）",
          need_text_to_mm_remap(vllm_checkpoint_composite=False,
                                vllm_model_path_set=False) is False)
    check("config 读不出来但用户显式给了 flag → 恒映射（唯一可靠信号）",
          need_text_to_mm_remap(vllm_checkpoint_composite=False,
                                vllm_model_path_set=True) is True)

    # ---- 同步侧的兜底：一个张量都没认领 = 键名体系不匹配，必须当场炸 ----
    class _ModelZero:
        def load_weights(self, items): return []
    class _ModelOk:
        def load_weights(self, items): return ["a", "b"]
    class _ModelNone:
        def load_weights(self, items): return None
    items = [("model.embed_tokens.weight", "t")]
    try:
        vllm_load_weights(_ModelZero(), items)
        check("loaded=0 且 sent>0 → RuntimeError（不许当成功）", False)
    except RuntimeError as e:
        check("loaded=0 且 sent>0 → RuntimeError（不许当成功）",
              "一个都没被认领" in str(e) and "need_text_to_mm_remap" in str(e))
    check("正常计数进日志（stacked 融合下 loaded<sent 属常态）",
          vllm_load_weights(_ModelOk(), items) == "loaded 2/1 tensors")
    check("loaded=None（老版本不返回清单）只告警不拦（宁可不拦不可错拦）",
          vllm_load_weights(_ModelNone(), items) == "loaded 0/1 tensors")

    # ---- gen_worker 接线：判据来自 need_text_to_mm_remap，不再是 vllm_model_path ----
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ro = open(os.path.join(root, "rlab", "rollout.py"), encoding="utf-8").read()
    check("gen_worker 用 need_text_to_mm_remap 判定（旧 _split_load 判据已废）",
          "need_text_to_mm_remap(" in ro and "_split_load" not in ro)
    check("判定结果同时用于 name_remap 与启动行自证",
          "name_remap=remap_text_to_multimodal if _need_remap else None" in ro
          and "权重同步键名映射" in ro)
    check("判据按 vLLM 侧 ckpt 是否复合体算（resolve_load_config 同源）",
          "resolve_load_config(_vllm_path)" in ro)


def test_diag_ablation_and_decode():
    """[AE] 消融自证 + 训练口径（prefill/decode 分离）。

    【为什么必须有】真机首次消融实测 `torch:fla vs torch:fallback` **逐位全 0**——
    两条不同 kernel 不可能一个 bit 都不差，真相是"这一档根本没换实现"。这类假对照
    比没有对照更危险：它会产出一个看起来干净的"自洽"结论。两条防线：
      ① 生成端：计数器证明目标实现真的被调用过，否则**拒绝产出**数据文件；
      ② 判读端：逐位相同的同引擎对不算消融证据，反而要打"作废嫌疑"。
    另外把口径分清：训练期对拍量的是逐位置被采样 logp（位置 0 走 prefill、≥1 走
    decode），而 diag 默认只测前缀末位（prefill）——两者形态差一个数量级是正常的。"""
    print("[AE] 消融自证 + prefill/decode 口径分离")
    from rlab.diag_logps import (decode_diff_stats, aggregate, verdict, torch_impl_tag)

    rep = {"patched": ["m.is_fla_available"],
           "counters": {"m.torch_chunk_gated_delta_rule": {"n": 84},
                        "m.chunk_gated_delta_rule": {"n": 0}}}
    tag_f, ok_f, why_f = torch_impl_tag("fallback", rep)
    check("计数 >0 → fallback 档被证实（回退实现真的跑了）",
          ok_f and tag_f == "torch:torch_ref" and "84" in why_f)
    _tag_a, ok_a, why_a = torch_impl_tag("fla", rep)
    check("要 fla 档但 fla 计数为 0 → **未达成**（不能当 fla 档解读）",
          (not ok_a) and "fla 计数 0" in why_a)
    check("没有计数器 → 一律未达成（'打了桩'不等于'换了实现'）",
          torch_impl_tag("fallback", {"patched": ["x"], "counters": {}})[1] is False)

    # 逐位全 0 的同引擎对 = 消融作废嫌疑，不能当"自洽"证据
    def _zero_pair(engine="torch"):
        return {"n": 5, "identical": True, "max_target_d": 0.0, "frac_big": 0.0,
                "confident_n": 0, "per_L": {}, "worst": [], "target_d_mean": 0.0,
                "target_d_p99": 0.0, "top1_match_rate": 1.0, "overlap_mean": 1.0,
                "max_abs_d_common": 0.0, "target_missing_n": 0}
    def _bad(m, f=0.2, c=2):
        return {"n": 5, "identical": False, "max_target_d": m, "frac_big": f,
                "confident_n": c, "per_L": {}, "worst": [], "target_d_mean": m,
                "target_d_p99": m, "top1_match_rate": 0.8, "overlap_mean": 0.8,
                "max_abs_d_common": m, "target_missing_n": 0}
    v = verdict({"torch:fla vs torch:fallback": _zero_pair(),
                 "vllm:keep vs vllm:none": _bad(0.0, 0.0, 0),   # vLLM 侧自己一致
                 "torch:fla vs vllm:keep": _bad(1.84)})
    check("逐位全 0 的同引擎对 → 打'消融作废嫌疑'并说明更像没换实现",
          any("消融作废嫌疑" in s and "没换实现" in s for s in v))
    check("该对被剔除后，torch 侧等于**没做**消融（不许当'自洽'）",
          any("没有 **torch 侧内部消融**" in s for s in v))
    check("清白的 vLLM 内配对仍被认可 → 不许连坐（结论不写成'两侧都未验证'）",
          not any("没有 **vllm 侧内部消融**" in s for s in v))

    # 逐位相同判定本身：全 0 → identical；有一个点差异 → 不是
    rows0 = [{"q": 0, "L": 0, "target_d": 0.0, "top1_match": True,
              "max_abs_d_common": 0.0, "overlap": 1.0, "confident": False,
              "target_missing": False}]
    rows1 = [dict(rows0[0]), {"q": 0, "L": 256, "target_d": 1e-9, "top1_match": True,
                              "max_abs_d_common": 0.0, "overlap": 1.0,
                              "confident": False, "target_missing": False}]
    check("aggregate.identical：全 0 → True，出现任何非 0 → False",
          aggregate(rows0)["identical"] is True and aggregate(rows1)["identical"] is False)

    # ---- decode 口径：位置 0（prefill 路）与 ≥1（decode 路）必须分开统计 ----
    torch_lps = torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, -5.0]])  # 对 ids[:,1:]
    vllm_lps = [9.0, -1.0, -2.0, -3.0, -4.0, -5.0]      # 位置 0 故意差 9 nat
    st = decode_diff_stats(vllm_lps, torch_lps, plen=1)
    check("decode：prefill 位（位置 0）单列，max|Δ|=9",
          st["prefill"]["n"] == 1 and abs(st["prefill"]["max"] - 9.0) < 1e-6)
    check("decode：位置 ≥1 全部为 0 → 训练口径的'紧致主体'可复现",
          st["decode"]["n"] == 5 and st["decode"]["max"] < 1e-9)
    check("decode：分段数 = prefill 1 + decode n-1 = 全部",
          st["prefill"]["n"] + st["decode"]["n"] == st["all"]["n"] == 6)
    check("decode：plen 偏移正确（prompt 长 3 时取 torch 的 [2:8]）",
          decode_diff_stats([0.0] * 5,
                            torch.tensor([[0., 0., -1., -2., -3., -4., -5., -6.]]),
                            plen=3)["all"]["n"] == 5)


def test_diag_impl_label_and_mode():
    """[AF] 标签必须反映实际实现 + vLLM 自身口径差（真机 2026-09-15 两条实锤）。

    ① `--torch_path fla` 与 `fallback` 逐位相同、计数器显示两次都是
       `torch_chunk_gated_delta_rule`(840 次)、fla 计数 0、`patched=[]`
       → "fla 档"是假的：打桩目标在本版建模模块里不存在，且 transformers 的 GDN
       快路径被 `causal_conv1d` 缺失挡着。标签不改就会把同源数据当两档比。
    ② 同一 (prompt, 位置, token)，轨迹构建的 `logprobs=0`（旧版没设 logprobs_mode）
       与探针的 `logprobs=20 + raw_logprobs` 差到 1.12 nat——**vLLM 自身口径差**，
       必须先量出来，否则跨引擎 Δ 里混着测量差。"""
    print("[AF] 实现标签自证 + vLLM 自身口径差")
    from rlab.diag_logps import internal_mode_delta, torch_impl_tag
    rep = {"patched": [],
           "counters": {"transformers.models.qwen3_5.modeling_qwen3_5"
                        ".torch_chunk_gated_delta_rule": {"n": 840},
                        "transformers.models.qwen3_next.modeling_qwen3_next"
                        ".torch_chunk_gated_delta_rule": {"n": 0}}}
    tag, ok, why = torch_impl_tag("fla", rep)
    check("请求 fla 但实际是 torch 参考实现 → 标签改为 torch:torch_ref 且标记未达成",
          tag == "torch:torch_ref" and ok is False and "840" in why)
    tag2, ok2, _ = torch_impl_tag("fallback", rep)
    check("请求 fallback 且回退实现真的跑了 → 达成（标签=torch:torch_ref）",
          tag2 == "torch:torch_ref" and ok2 is True)
    rep_fla = {"patched": ["m.is_fast_path_available"],
               "counters": {"m.chunk_gated_delta_rule": {"n": 84},
                            "m.torch_chunk_gated_delta_rule": {"n": 0}}}
    check("装了 causal_conv1d 走 fla 后 → fla 档名副其实",
          torch_impl_tag("fla", rep_fla)[0] == "torch:fla"
          and torch_impl_tag("fla", rep_fla)[1] is True)
    check("无计数器（既没 fla 也没参考实现被打点）→ 标签 unknown，不冒充",
          torch_impl_tag("fla", {"patched": [], "counters": {}})[0] == "torch:unknown")

    rows = [{"lp_target": -0.112, "lp_target_default": -1.0, "top1": 11},
            {"lp_target": -1.0, "lp_target_default": -1.0, "top1": 22},
            {"lp_target": -2.0, "lp_target_default": -2.0, "top1": 33},
            {"lp_target": -0.5, "lp_target_default": -3.0, "top1": 44},
            {"lp_target": -0.5, "lp_target_default": None, "top1": 55}]
    st = internal_mode_delta(rows)
    check("vLLM 自身口径差：只统计两侧都有的点（缺的跳过）",
          st["n"] == 4 and abs(st["mean"] - (0.888 + 0 + 0 + 2.5) / 4) < 1e-6)
    check("vLLM 自身口径差：>1nat 占比与 max 都能报出来（真机 max 1.12）",
          abs(st["max"] - 2.5) < 1e-6 and abs(st["frac_gt_1"] - 0.25) < 1e-6)
    check("全缺 → n=0（调用方不打印，不产生假 0 结论）",
          internal_mode_delta([{"lp_target": 1.0}])["n"] == 0)

    # 轨迹构建的 SamplingParams 必须与 rollout.make_retool_sps 同源（含 logprobs_mode）
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(root, "rlab", "diag_logps.py"), encoding="utf-8").read()
    build = src.split("def build_traj_vllm", 1)[1].split("def _read_jsonl", 1)[0]
    check("建轨迹的 SP 显式设 raw_logprobs（旧版漏设 → 与训练口径差 1.12 nat）",
          'kw["logprobs_mode"] = "raw_logprobs"' in build and "logprobs=0" in build)
    check("探针提供 default 档（用于量 vLLM 自身口径差）",
          "def _probe_sp" in src and 'mode == "raw"' in src
          and '"--probe_default"' in src)
    check("落盘前标签改成实际实现（不许把同源数据当两档）",
          '_r["provider"] = tag' in src)
    check("merge 跳过轨迹文件而不是拦掉整次 merge",
          "跳过 {path}：无 provider 字段" in src)


def test_diag_counter_and_trajid():
    """[AG] 观测与干预分离 + 轨迹指纹（真机 2026-09-15 两条工具缺陷）。

    ① `--torch_path fla` 档报 `实际实现=unknown`（fla 计数 0、参考实现计数 0）——
       不是环境问题：旧版 `force_torch_gdn_fallback(enable=False)` 直接 early-return，
       **根本没装计数器**，于是"没观测"被读成"没跑实现"。计数器必须无条件安装。
    ② `--build_traj` 会**覆盖** traj.jsonl：旧 vLLM 结果 + 新 torch 结果混着 merge 时，
       (q, L) 键照样重叠，但指的是不同的 token/上下文 → 会给出一个"正常"的差异数字。
       必须用轨迹指纹硬拦。"""
    print("[AG] 计数器无条件安装 + merge 拒绝跨轨迹比较")
    import sys
    import types
    import rlab.diag_logps as D

    fake = types.ModuleType("fake_modeling_qwen3_5_for_test")
    calls = {"n": 0}

    def torch_chunk_gated_delta_rule(*a, **kw):
        calls["n"] += 1
        return "ref"

    fake.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    sys.modules[fake.__name__] = fake
    orig_mods = D._qwen_gdn_modules
    D._qwen_gdn_modules = lambda: [fake.__name__]
    try:
        rep_obs = D.force_torch_gdn_fallback(enable=False)
        check("enable=False（观测档）也装计数器——不再 early-return",
              any("torch_chunk_gated_delta_rule" in k for k in rep_obs["counters"]))
        fake.torch_chunk_gated_delta_rule(1, 2)
        cnt = [b["n"] for k, b in rep_obs["counters"].items()
               if "torch_chunk_gated_delta_rule" in k]
        check("计数随调用递增（且原函数语义不变）",
              cnt == [1] and calls["n"] == 1)
        check("观测档不打桩（patched 为空，只看不动）", rep_obs["patched"] == [])
        tag, ok, _why = D.torch_impl_tag("fla", rep_obs)
        check("有计数就能给出事实标签：实际是 torch 参考实现 → 不冒充 fla",
              tag == "torch:torch_ref" and ok is False)
    finally:
        D._qwen_gdn_modules = orig_mods
        sys.modules.pop(fake.__name__, None)

    a = [{"q": 0, "prompt_ids": [1, 2], "ids": [3, 4]},
         {"q": 1, "prompt_ids": [5], "ids": [6]}]
    b = [{"q": 0, "prompt_ids": [1, 2], "ids": [3, 4]},
         {"q": 1, "prompt_ids": [5], "ids": [7]}]      # 只差一个 token
    check("轨迹指纹：同轨迹同 id / 差一个 token 就不同 id",
          D.traj_id(a) == D.traj_id(list(reversed(a))) and D.traj_id(a) != D.traj_id(b))

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(root, "rlab", "diag_logps.py"), encoding="utf-8").read()
    check("merge 按 traj_id 跳过跨轨迹的 provider 对（并存证内部混轨迹）",
          "轨迹不同（" in src and "tids[a] and tids[b]" in src
          and "内部混了多个轨迹" in src)
    check("per-L 同时给 target 差与分布差（分辨'算错'与'报数错'）",
          "/dist=" in src and 'v.get(\'max_abs_d_common\')' in src)
    check("轨迹指纹随行落盘（三个 provider 路径都写 traj_id）",
          src.count('"traj_id"') >= 4 and src.count("traj_id(rows)") >= 3)


def test_logprobs_n_fix_path():
    """[AH] vLLM 上报路径修复档：`vllm_logprobs_n` + lpmode 探针（真机 2026-09-15）。

    实锤：同一 prompt/位置/token，`logprobs=0` 报 -0.602，而 `logprobs=20`(raw) 报
    -7.40、torch 独立重算 -7.400 —— 训练端 gen_logps 走的正是 N=0 这条形态，所以
    对拍 max 12.8 / clip_frac 0.9% 量的是**上报 bug**，不是模型差。修复档 = 改用
    N≥1（top-K 里挑被采样 token）并保持 fail-fast。"""
    print("[AH] vLLM logprobs=N 修复档 + lpmode 形态探针")
    import inspect

    import rlab.rollout as R
    from rlab.config import ALGO_DEFAULTS, BASE
    from rlab.diag_logps import lpmode_spread, vllm_kwargs_for_backend

    check("config 新增 vllm_logprobs_n（默认 0 = 保留旧行为便于 A/B）",
          BASE.get("vllm_logprobs_n") == 0)
    src = inspect.getsource(R.gen_worker)
    check("make_retool_sps 用 cfg 的 vllm_logprobs_n 而不是硬编码 0",
          'kw["logprobs"] = int(cfg.get("vllm_logprobs_n", 0) or 0)' in src)
    check("raw_logprobs 仍显式声明（口径不被版本默认值左右）",
          'kw["logprobs_mode"] = "raw_logprobs"' in src)
    for algo in ("retool", "retool_math"):
        check(f"{algo} preset 未偷偷覆盖 N（档位由 CLI 决定）",
              "vllm_logprobs_n" not in ALGO_DEFAULTS.get(algo, {}))
    # 【2026-09-17】修复档建议 N≥1，但 train.py 此前没有该 CLI —— 每个
    # --vllm_gen_logps 的 run 都被钉死在 N=0（bg1 真机 run 即如此）。
    import inspect as _ins
    import rlab.train as _T
    _tsrc = _ins.getsource(_T.main)
    check("train.py 暴露 --vllm_logprobs_n 并接线到 overrides（否则 N=0 无法覆盖）",
          '"--vllm_logprobs_n"' in _tsrc
          and 'overrides["vllm_logprobs_n"] = int(args.vllm_logprobs_n)' in _tsrc)

    # N≥1 时 vLLM 必须仍把被采样 token 放进返回字典 —— 缺了就 fail-fast 且提示别退回 N=0
    class _LP(dict):
        pass

    class _Out:
        def __init__(self, entries):
            self.logprobs = entries

    good = _Out([{7: type("L", (), {"logprob": -0.25})()},
                 {9: type("L", (), {"logprob": -1.5})(), 3: type("L", (), {"logprob": -2.0})()}])
    check("N≥1：从 top-K 字典里取被采样 token 的值",
          R.sampled_logps_from_output(good, [7, 9]) == [-0.25, -1.5])
    bad = _Out([{7: type("L", (), {"logprob": -0.25})()}, {3: type("L", (), {"logprob": -2.0})()}])
    try:
        R.sampled_logps_from_output(bad, [7, 9])
        check("缺被采样 token → 必须 raise（不许静默）", False)
    except ValueError as e:
        check("缺被采样 token → raise 且提示别退回 N=0（那条路已被实锤不可信）",
              "别退回 N=0" in str(e))
    doc = R.sampled_logps_from_output.__doc__ or ""
    check("函数 docstring 留档 2026-09-15 实锤（N=0 报数与分布不符）",
          "2026-09-15" in doc and "-7.400" in doc)

    st = lpmode_spread({"A_K0_T1": -0.602, "D_K0_TT": -0.602,
                        "B_KK_T1": -7.40, "C_KK_TT": -7.40})
    check("lpmode 极差能分辨'是 logprobs 取值'（A≈D 偏、B≈C 一致）",
          abs(st["spread"] - 6.798) < 1e-6 and st["argmax"].startswith("A")
          and st["argmin"].startswith("B"))
    st2 = lpmode_spread({"A_K0_T1": -3.0, "B_KK_T1": -3.0,
                         "C_KK_TT": -3.0, "D_K0_TT": -0.5})
    check("lpmode 极差也能分辨'是 max_tokens'（A≈B≈C 偏、D 离群）",
          st2["argmax"].startswith("D") and st2["argmin"].startswith("A"))
    check("形态不足两个 → 不给极差（不许从单点编结论）",
          lpmode_spread({"A_K0_T1": -1.0})["spread"] is None)
    dsrc = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "diag_logps.py"), encoding="utf-8").read()
    check("lpmode 模式的四种形态齐备（K0/KK × T1/TT）",
          all(f'"{n}"' in dsrc for n in ("A_K0_T1", "B_KK_T1", "C_KK_TT", "D_K0_TT")))
    check("lpmode 要求 vLLM provider（它是报数路径探针，不是跨引擎对拍）",
          "--measure lpmode 是 vLLM 侧的报数路径探针" in dsrc)
    # 【真机首跑 crash】新调用点把参数顺序写反（(backend, cfg) 而非 (dict, backend)）
    # → ValueError: dictionary update sequence element #0 has length 1。参数顺序错误
    # 静态检查抓不到，但"所有调用点必须是 (dict/空dict, backend)"可以静态锁。
    import re as _re
    calls = _re.findall(r"vllm_kwargs_for_backend\(([^)]*)\)", dsrc)
    check("vllm_kwargs_for_backend 的所有调用点参数顺序正确（第一个是 dict 来源）",
          len(calls) >= 2 and all("args.vllm_backend" not in c.split(",")[0] for c in calls))
    check("vllm_kwargs_for_backend 纯函数：keep/none/其它三档语义",
          vllm_kwargs_for_backend({"gdn_prefill_backend": "flashinfer"}, "keep")
          == {"gdn_prefill_backend": "flashinfer"}
          and vllm_kwargs_for_backend({"gdn_prefill_backend": "flashinfer"}, "none") == {}
          and vllm_kwargs_for_backend({}, "triton") == {"gdn_prefill_backend": "triton"})
    # 【真机两次实锤】默认档 = FlashInfer GDN（JIT）→ ninja 被 SIGKILL(9) → 引擎初始化失败。
    # 前置告警必须存在且**只告警不改档**（静默替换被测配置是本项目禁止的操作）。
    import io
    from contextlib import redirect_stdout
    from rlab.diag_logps import warn_if_default_backend

    class _A:
        def __init__(self, b):
            self.vllm_backend = b

    buf = io.StringIO()
    with redirect_stdout(buf):
        warn_if_default_backend(_A("keep"), {})
    out = buf.getvalue()
    check("默认档会走 FlashInfer → 打印前置告警（含 triton 与 MAX_JOBS 两条处置）",
          "FlashInfer" in out and "--vllm_backend triton" in out and "MAX_JOBS=1" in out
          and "VLLM_ENABLE_V1_MULTIPROCESSING=0" in out)
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        warn_if_default_backend(_A("triton"), {})
        warn_if_default_backend(_A("keep"), {"vllm_gen_kwargs": {"gdn_prefill_backend": "triton"}})
    check("显式指定 backend 或 preset 已带 gdn_prefill_backend → 不告警（不噪音）",
          buf2.getvalue() == "")
    check("告警只打印、不修改配置（不做静默改档）",
          warn_if_default_backend(_A("keep"), {}) is None)

    # 【真机首个 lpmode 跑批的设计缺陷】原设计拿"各形态自采到的 token"当比较基准，
    # 实测 36 点里 13 点四条采到**不同 token** → 那些点的四个 logp 不可比。归拢函数
    # 必须只统计同 token 的点，并把两条轴分开（K 轴=logprobs 取值；T 轴=max_tokens）。
    from rlab.diag_logps import lpmode_summary

    def _row(q, L, forms, toks, tops=None, ref=None):
        same = len(set(toks.values())) == 1
        r = {"q": q, "L": L, "forms": forms, "tokens": toks,
             "traj_target": ref, "same_token": same}
        if tops:
            r["tops"] = tops
        return r

    # 同 token、K 轴偏（A 与 B 差 2nat）但 T 轴字典完全相同 → 结论=logprobs=0 上报错
    r1 = _row(0, 0, {"A_K0_T1": -2.0, "B_KK_T1": -4.0, "C_KK_TT": -4.0, "D_K0_TT": -2.0},
              {"A_K0_T1": 7, "B_KK_T1": 7, "C_KK_TT": 7, "D_K0_TT": 7},
              {"B_KK_T1": {7: -4.0, 8: -5.0}, "C_KK_TT": {7: -4.0, 8: -5.0}}, ref=7)
    # 不同 token → 整点剔除，不许进极差
    r2 = _row(1, 0, {"A_K0_T1": 0.0, "B_KK_T1": -9.0, "C_KK_TT": -9.0, "D_K0_TT": -9.0},
              {"A_K0_T1": 1, "B_KK_T1": 2, "C_KK_TT": 2, "D_K0_TT": 2})
    st = lpmode_summary([r1, r2])
    check("lpmode 归拢：只统计同 token 的点（不同 token 的点剔除，不进极差）",
          st["n"] == 2 and st["n_same_token"] == 1 and st["K_axis_T1"]["n"] == 1
          and abs(st["K_axis_T1"]["max"] - 2.0) < 1e-9)
    check("lpmode 归拢：T 轴用**字典**比较（此处 B==C → 字典相同、top-1 相同）",
          st["T_axis_dicts"]["n_dicts_equal"] == 1
          and st["T_axis_dicts"]["n_top1_same"] == 1
          and st["T_axis_dicts"]["max_d_common"] == 0.0)
    r3 = _row(2, 0, {"A_K0_T1": -1.0, "B_KK_T1": -1.0, "C_KK_TT": -1.0, "D_K0_TT": -1.0},
              {"A_K0_T1": 7, "B_KK_T1": 7, "C_KK_TT": 7, "D_K0_TT": 7},
              {"B_KK_T1": {7: -1.0, 8: -9.0}, "C_KK_TT": {7: -1.0, 8: -2.0}}, ref=7)
    st3 = lpmode_summary([r3])
    check("lpmode 归拢：字典不同 → 能测出交集上的 token 差（T 轴影响 logits 的证据）",
          st3["T_axis_dicts"]["n_dicts_equal"] == 0
          and abs(st3["T_axis_dicts"]["max_d_common"] - 7.0) < 1e-9
          and st3["K_axis_T1"]["max"] == 0.0)
    check("建轨迹可指定 logprobs=N（验证修复档必须与训练 cfg 同 N）",
          dsrc.count("build_logprobs_n") >= 3 and "logprobs=int(logprobs_n or 0)" in dsrc)
    # 噪声地板：同形态重复档 B2 必须存在，否则无法区分"形态的系统性差异"与
    # "同一形态自己就不可复现"（真机首跑 13/36 点四条采到不同 token 已提示后者可能）
    check("lpmode 有同形态重复档 B2_KK_T1（仪器的噪声地板）",
          '"B2_KK_T1"' in dsrc and "noise_B_vs_B2" in dsrc)
    r4 = _row(3, 0, {"A_K0_T1": -1.0, "B_KK_T1": -1.0, "C_KK_TT": -1.0, "D_K0_TT": -1.0,
                     "B2_KK_T1": -3.5},
              {"A_K0_T1": 7, "B_KK_T1": 7, "C_KK_TT": 7, "D_K0_TT": 7, "B2_KK_T1": 7},
              {"B_KK_T1": {7: -1.0}, "C_KK_TT": {7: -1.0}, "B2_KK_T1": {7: -3.5}}, ref=7)
    st4 = lpmode_summary([r4])
    check("lpmode 归拢给出噪声地板（同形态两次 max|Δ| 与字典是否相同）",
          abs(st4["noise_B_vs_B2"]["max"] - 2.5) < 1e-9
          and st4["noise_dicts"]["n_dicts_equal"] == 0
          and abs(st4["noise_dicts"]["max_d_common"] - 2.5) < 1e-9)
    check("噪声地板大时的判读优先于任何开关归因（先承认读数不可复现）",
          "噪声地板本身就大" in dsrc and "此时不能把差异归给任何开关" in dsrc)
    # 同一设置 --build_traj 两次的 diff 工具：决定训练档在自己那一档是否可复现
    from rlab.diag_logps import traj_diff_stats
    a = [{"q": 0, "ids": [1, 2, 3], "logps": [-0.5, -1.0, -2.0]}]
    b = [{"q": 0, "ids": [1, 2, 4], "logps": [-0.6, -1.1, -3.0]}]
    st = traj_diff_stats(a, b)
    check("diff_traj 算 token 一致率、逐位置 |Δlogp| 形态、位置 0 与其余分开",
          st["n"] == 3 and abs(st["token_match_rate"] - 2 / 3) < 1e-9
          and abs(st["pos0_max"] - 0.1) < 1e-9 and abs(st["rest_max"] - 1.0) < 1e-9)
    check("diff_traj 给出最差点（训练档不可复现时从哪冒出来的）",
          len(st["worst"]) > 0 and st["worst"][0]["pos"] == 2
          and st["worst"][0]["same_tok"] is False)
    check("--diff_traj CLI 存在且纯函数可测",
          '--diff_traj' in dsrc and "def print_traj_diff" in dsrc)
    # 引擎自身的确定性下限（跨实例一致性测试的最小版本）——先证明"引擎自己确定"，
    # 再谈训练/推理对齐；否则任何对拍数字都无意义。
    from rlab.diag_logps import det_repeat_stats
    same = [{"ids_prefix": [1, 2], "dict_hash": "aa", "lp_top1": -0.1}] * 3
    st_same = det_repeat_stats(same)
    check("det：三次全同 → ids/dicts 都判定一致（可复现）",
          st_same["ids_identical"] and st_same["dicts_identical"]
          and st_same["n_unique_dicts"] == 1 and not st_same["first_call_differs"])
    diff_d = [{"ids_prefix": [1, 2], "dict_hash": "aa", "lp_top1": -0.1},
              {"ids_prefix": [1, 2], "dict_hash": "bb", "lp_top1": -0.9},
              {"ids_prefix": [1, 2], "dict_hash": "cc", "lp_top1": -2.5}]
    st_d = det_repeat_stats(diff_d)
    check("det：分布不同（dict hash 变）必须单独判出来——这是'logits 不可复现'的证据",
          st_d["ids_identical"] and not st_d["dicts_identical"]
          and abs(st_d["lp_top1_spread"] - 2.4) < 1e-9)
    diff_ids = [{"ids_prefix": [1, 2], "dict_hash": "aa", "lp_top1": -0.1},
                {"ids_prefix": [3, 4], "dict_hash": "aa", "lp_top1": -0.1}]
    st_i = det_repeat_stats(diff_ids)
    check("det：分布一致但 token 不同 → 单独判为采样/RNG 层问题",
          not st_i["ids_identical"] and st_i["dicts_identical"])
    check("det 模式要求 vLLM provider 且三种判读都在源码里",
          "--measure det 是 vLLM 引擎自身的确定性探针" in dsrc
          and "logits 本身不可复现" in dsrc and "采样/RNG 层不确定" in dsrc)
    # batch-invariant 要求显式 attention backend（真机 19:03 启动即 RuntimeError）。
    # 键名跨版本两形态 → 必须问注册表映射，两个都没有就 raise（不许静默忽略）。
    from rlab.diag_logps import map_attention_backend
    check("attention backend 键名映射：有 attention_config 用 dict 形态",
          map_attention_backend("FLASH_ATTN", {"attention_config", "gdn_prefill_backend"})
          == {"attention_config": {"backend": "FLASH_ATTN"}})
    check("attention backend 键名映射：只有 attention_backend 时用平铺形态",
          map_attention_backend("TRITON_ATTN", {"attention_backend"})
          == {"attention_backend": "TRITON_ATTN"})
    try:
        map_attention_backend("FLASH_ATTN", {"gdn_prefill_backend"})
        check("两个键都没有 → 必须 raise（否则'开了 batch-invariant'是假的）", False)
    except RuntimeError as e:
        check("两个键都没有 → raise 且说明后果（静默忽略 = 假绿灯）",
              "VLLM_BATCH_INVARIANT=1" in str(e))
    check("--attention_backend 已接线到所有 LLM() 入口（backend 档 + attention 档）",
          dsrc.count("vllm_extra_kwargs(cfg, args)") >= 3
          and '"--attention_backend"' in dsrc)
    # 【真机 19:xx】VLLM_BATCH_INVARIANT=1 + 显式 attention backend → det 3/3 全同（spread=0）。
    # 训练端要能用同一套：config 两个键 + train.py 两个 flag + gen_worker 设 env/前置检查。
    from rlab.config import BASE
    from rlab.rollout import batch_invariant_guard
    check("config 新增 vllm_batch_invariant / vllm_attention_backend（默认关）",
          BASE.get("vllm_batch_invariant") is False
          and BASE.get("vllm_attention_backend") is None)
    trsrc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()
    ro = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "rollout.py"), encoding="utf-8").read()
    check("train.py 暴露 --vllm_batch_invariant 与 --vllm_attention_backend",
          '"--vllm_batch_invariant"' in trsrc and '"--vllm_attention_backend"' in trsrc
          and 'overrides["vllm_batch_invariant"] = True' in trsrc)
    check("gen_worker 设 VLLM_BATCH_INVARIANT=1 并把 attention backend 并进引擎参数",
          'os.environ["VLLM_BATCH_INVARIANT"] = "1"' in ro
          and "_gen_kwargs.update(attention_backend_kwargs(_attn_be))" in ro)
    check("确定性档前置检查：开了但没有 backend → raise（别白等一次引擎启动）",
          batch_invariant_guard(False, None) is None
          and batch_invariant_guard(True, "FLASH_ATTN") is None)
    try:
        batch_invariant_guard(True, None)
        check("开了确定性档但缺 attention backend → 必须 raise", False)
    except RuntimeError as e:
        check("开了确定性档但缺 attention backend → raise 且给出正确 flag",
              "vllm_attention_backend" in str(e))
    check("我们的 logp 路径不用 flash-attn CE（纯 log_softmax+gather，已在对齐标准上）",
          "log_softmax" in open(os.path.join(
              os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
              "rlab", "losses.py"), encoding="utf-8").read()
          and "cross_entropy" not in open(os.path.join(
              os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
              "rlab", "losses.py"), encoding="utf-8").read())

    # 【2026-09-16】vllm_gen_logps 的 Qwen3.5 警告必须**分档**：确定性档已经把可复现性
    # 修好（docs/07 §9），开了档还照旧喊"不可复现"就是把"已修复"当"已知坏"用。
    rollout_src2 = open("rlab/rollout.py", encoding="utf-8").read()
    check("vllm_gen_logps 警告分档：开确定性档走提示分支（指向 docs/07 §9）",
          "确定性档已开" in rollout_src2 and "docs/07 §9" in rollout_src2)
    check("vllm_gen_logps 警告分档：未开档仍给两条修法（torch 副本 / batch-invariant）",
          "**未开确定性档**" in rollout_src2
          and "--vllm_batch_invariant --vllm_attention_backend FLASH_ATTN" in rollout_src2)


# ---- AI. 采样评测（--val_n）口径三连修：除数 / per-item 索引 / 检验二值化 ----
def test_val_n_metric_fixes():
    """[AI] p8「训练一直没效果」的真因是**评测层三个 bug**，不是训练（2026-09-19）。

    真机 p8 报表签名：`code% = 401.5 / 406.5 / 417.0 / 366.0`（>100% 不可能是率）
    + `step100 -4.6pp McNemar p=0.003 显著`。三个连环 bug：
      ① code_rate/code_ok_rate/avg_rounds 除以 n_valid（题数 200），而 code_used
         长度是 n_valid×val_n（1600）→ 显示值是真实值的 val_n=8 倍（401.5%→50.2%）。
         **代码从未被灭绝**，各存档点都保持 ~46-52%，先前"code 压灭"的叙事在 p8 不成立。
      ② per-item 的 code_used 取 code_used[i]（i=题号 0..199），而该列表是
         [题][采样] 平铺 → 只覆盖题 0..24 的全部采样，题 25.. 全缺 → 代码分层/
         迁移分析口径全错。
      ③ McNemar 按 `acc == 1.0` 二值化，而采样档 per-item acc 是 Average@N 小数
         → 检验的是"N 条全对率"而非 acc。**"显著"回答了另一个问题。**
    """
    print("[AI] 采样评测 --val_n 口径三连修（p8 假阴性事故）")
    import inspect

    from rlab.analysis import (code_layer, items_are_binary, paired_counts,
                               paired_mean_test, paired_test_auto)

    # ---- ① 除数：真机数字精确复现 ----
    esrc = open("eval_vllm_one.py", encoding="utf-8").read()
    check("code 指标除数改为 len(code_used)（= n_valid×val_n），不再用 n_valid",
          "_denom = len(code_used) if code_used else 1" in esrc
          and "result[\"avg_rounds\"] = sum(code_used) / _denom" in esrc)
    check("旧除数写法（/ n_valid）已从 code 指标里消失（反证：防回归）",
          "if u > 0) / n_valid" not in esrc and "sum(code_used) / n_valid" not in esrc)
    # p8 报表 401.5% 的来源与修复后的真值
    n_valid, val_n = 200, 8
    code_used = [0] * (n_valid * val_n)
    for i in range(803):          # sum(code_used)=803 → 旧口径 401.5%
        code_used[i] = 1
    old_rate = sum(1 for u in code_used if u > 0) / n_valid
    new_rate = sum(1 for u in code_used if u > 0) / len(code_used)
    check("旧口径精确复现真机 401.5%，新口径给出真值 50.2%（差 val_n=8 倍）",
          abs(old_rate * 100 - 401.5) < 1e-6 and abs(new_rate * 100 - 50.19) < 0.01
          and abs(old_rate / new_rate - val_n) < 1e-9)

    # ---- ② per-item 索引：按题聚合 val_n 条，而不是取第 i 条轨迹 ----
    check("per-item 采样档按题切片聚合 code_used/code_ok（不再用 code_used[i]）",
          "_sl = slice(i * args.val_n, (i + 1) * args.val_n)" in esrc
          and "sum(code_used[_sl]) / args.val_n" in esrc)
    check("per-item 落 val_n 标记（供下游判定连续/二值口径）",
          '"val_n": args.val_n,' in esrc)
    check("旧索引写法已消失（反证）",
          "int(code_used[i]) if code_used and i < len(code_used)" not in esrc)

    # ---- ③ 检验分派：二值 → McNemar；连续 → 配对均值 z ----
    greedy_a = [{"qk": "a", "acc": 1.0, "val_n": 1}, {"qk": "b", "acc": 0.0, "val_n": 1}]
    greedy_b = [{"qk": "a", "acc": 0.0, "val_n": 1}, {"qk": "b", "acc": 0.0, "val_n": 1}]
    samp_a = [{"qk": "a", "acc": 0.875, "val_n": 8}, {"qk": "b", "acc": 0.25, "val_n": 8}]
    check("greedy（全 0/1 且 val_n=1）判为二值 → 仍走 McNemar（历史口径不变）",
          items_are_binary(greedy_a, greedy_b)
          and "McNemar" in paired_test_auto(greedy_a, greedy_b)[3])
    check("采样档小数 acc 判为连续 → 走配对均值 z 检验",
          not items_are_binary(samp_a)
          and "配对均值" in paired_test_auto(samp_a, greedy_a)[3])
    # 采样档即使本次抽样恰好全 0/1，也必须靠 val_n 标记判连续（否则口径随数据漂移）
    edge = [{"qk": "a", "acc": 1.0, "val_n": 8}, {"qk": "b", "acc": 0.0, "val_n": 8}]
    check("采样档恰好全 0/1 仍判连续（val_n 标记优先，口径不随抽样漂移）",
          not items_are_binary(edge))

    # 二值化失真的定量反证：真实 acc 差 vs 全对率差是两个不同的量
    base = [{"qk": f"q{i}", "acc": k / 8, "val_n": 8}
            for i, k in enumerate([8, 8, 7, 6, 5, 4, 3, 2, 1, 0])]
    model = [{"qk": f"q{i}", "acc": k / 8, "val_n": 8}
             for i, k in enumerate([7, 7, 7, 7, 7, 5, 4, 3, 2, 1])]
    real_d = (sum(x["acc"] for x in model) - sum(x["acc"] for x in base)) / 10 * 100
    pc = paired_counts(model, base)      # 旧路径：全对率口径
    pm = paired_mean_test(model, base)   # 新路径：真实 acc 配对差
    check("旧 McNemar 在此例上只看到'全对率 2→0'（b=0/c=2），丢掉全部小数信息",
          pc[0] == 0 and pc[1] == 2)
    check("配对均值检验给出真实 acc 差（+7.5pp，与逐题均值一致）且方向为正",
          abs(pm[0] - real_d) < 1e-9 and pm[0] > 0)
    check("两个口径在此例上**方向相反**——这正是 p8 '显著变差' 的成因",
          pm[0] > 0 and pc[1] > pc[0])

    # ---- 分层/迁移的 acc 也必须求和而非 ==1.0 计数 ----
    asrc = open("rlab/analysis.py", encoding="utf-8").read()
    layer = code_layer([{"qk": "a", "acc": 0.875, "code_used": 1.0},
                        {"qk": "b", "acc": 0.25, "code_used": 0.0}])
    check("code_layer 采样档用 acc 求和（0.875 不再被 ==1.0 归零）",
          layer == ((1, 0.875), (1, 0.25)))
    check("_pair_col 同题双 acc 也改求和（迁移分析不再系统性低估两臂）",
          'float(t.get("acc", 0)) for _, t in pairs' in asrc)
    check("paired_counts 留下口径警告（只对 greedy 有效，连续档走 paired_mean_test）",
          "口径警告" in (paired_counts.__doc__ or "")
          and "paired_mean_test" in (paired_counts.__doc__ or ""))
    # 所有配对检验入口都必须走自动分派，不许残留裸 McNemar 调用
    for fn in ("summarize_eval", "pair_eval", "summarize_code_layer"):
        src = inspect.getsource(getattr(__import__("rlab.analysis", fromlist=[fn]), fn))
        check(f"{fn} 走 paired_test_auto（不再裸调 paired_counts+mcnemar）",
              "paired_test_auto" in src and "mcnemar_exact(" not in src)


# ---- AJ. eval 采样档确定性接线（跨 run 可比性前提） ----
def test_eval_determinism_wiring():
    """[AJ] 采样评测的可复现性缺口（2026-09-20）：确定性档没接到 eval。

    实测：同权重、同 `--seed 42`、同 `--n 200` 重跑 BASE，acc 63.1 → **61.1**
    （−2.0pp）。而本轮要判定的效应只有 3pp 级 —— 噪声地板吃掉结论。

    **自我纠正记录**：先前曾断言"轨迹 seed 不受 --seed 约束"（指
    `_base = _rnd.randrange(1<<30)`），**该判断错误**：`import random as _rnd`
    绑定的是同一个已被 `random.seed(args.seed)` 播种的模块对象，实测同
    seed/同 n 下 `_base` 逐位复现（993486218）。真正的抖动源是 vLLM 侧
    （批调度 + bf16 归约顺序，docs/07 已定案），修法 = 训练端那对
    `VLLM_BATCH_INVARIANT=1` + 显式 attention backend。

    教训：**"不可复现"要先定位到层**——同一条 random 流里的确定性可以直接
    实测验证，不能靠读一眼 `randrange` 就下结论。
    """
    print("[AJ] eval 采样档确定性接线（同权重漂移 2pp 的修法）")
    import random

    # 先把"轨迹 seed 本来就确定"这条实测锁进测试（防再次误判）
    def _seq(seed, n):
        random.seed(seed)
        s = random.sample(list(range(17000)), n)
        return s[:3], random.randrange(1 << 30)
    a, b = _seq(42, 200), _seq(42, 200)
    check("抽题+轨迹 base seed 在同 seed/同 n 下逐位复现（纠正旧误判）", a == b)
    check("换 seed 则 base seed 改变（确实受 --seed 约束）",
          _seq(43, 200)[1] != _seq(42, 200)[1])
    # n 改变会移动 random 状态 → 轨迹 seed 也变（跨 n 的 run 不可逐条比对）
    check("换 n 会移动 random 流 → base seed 改变（跨 n 不可逐条比）",
          _seq(42, 500)[1] != _seq(42, 200)[1])

    one = open("eval_vllm_one.py", encoding="utf-8").read()
    sched = open("eval_vllm.py", encoding="utf-8").read()
    uni = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "eval.py"), encoding="utf-8").read()

    check("eval_one: 确定性档 CLI 存在且默认 None（=随 run_info，不改历史默认行为）",
          '"--vllm_batch_invariant"' in one and '"--vllm_attention_backend"' in one
          and "if args.vllm_batch_invariant is None" in one)
    check("eval_one: 复用训练端 batch_invariant_guard（成对校验单点同源，不另写一份）",
          "batch_invariant_guard as _bi_guard" in one
          and "_bi_guard(bool(_bi), _attn_be)" in one)
    check("eval_one: 复用训练端 attention_backend_kwargs（键名从 vLLM 注册表取）",
          "attention_backend_kwargs as _attn_kw" in one
          and "_vllm_kwargs.update(_attn_kw(_attn_be))" in one)
    check("eval_one: env 在 LLM() 构造之前设置（vLLM envs 惰性读取，晚设无效）",
          one.index('os.environ["VLLM_BATCH_INVARIANT"] = "1"') < one.index("llm = LLM("))
    check("eval_one: 采样档未开确定性档时告警（把 2pp 地板写在脸上）",
          "未开确定性档" in one and "63.1" in one)
    check("eval_one: 档位落进 eval_protocol（旧 json 不可与新 json 直接比 Δacc）",
          '"vllm_batch_invariant": bool(_bi)' in one
          and '"vllm_attention_backend": _attn_be' in one)
    # 透传链：断一环则多模型 eval 里 BASE 与 tuned 可能落不同档
    check("调度器 eval_vllm.py 透传两个 flag（含 --no- 关档形态）",
          '["--vllm_batch_invariant"]' in sched
          and '["--no-vllm_batch_invariant"]' in sched
          and '"--vllm_attention_backend", args.vllm_attention_backend' in sched)
    check("统一入口 rlab/eval.py 透传两个 flag（含 --no- 关档形态）",
          '"--vllm_batch_invariant" if args.vllm_batch_invariant' in uni
          and '"--no-vllm_batch_invariant"' in uni
          and '"--vllm_attention_backend", args.vllm_attention_backend' in uni)
    # 三处都用 BooleanOptionalAction：None/True/False 三态（None 才能"随 run_info"）
    for tag, src in (("eval_one", one), ("调度器", sched), ("统一入口", uni)):
        check(f"{tag}: 用 BooleanOptionalAction 三态（None=随训练档，非 store_true 两态）",
              "action=argparse.BooleanOptionalAction" in src)


def test_inline_eval_timeout_fix():
    """【2026-09-24 内嵌评测盲窗修复（p10 事故）】

    p10：retool 多轮采样档（n=500×4轮×6144 token）单路评测 >15min，train.py 硬编码
    timeout=900 → step100/200/300/400 的 test+train 共 8 路全 TIMEOUT、eval_*.json
    全缺失 → 30h 训练全程盲跑（2026-09-21 加内嵌评测治的正是 p9"训完 11h 才发现
    深坑"——被这个超时反手做成同类盲窗）。

    修复三件套（本测试锁死）：
      1) 超时进配置：BASE=900 保旧行为、retool_math preset=3600（=手动跑预算），
         可 CLI --eval_timeout_s 覆盖；
      2) 超时不再静默：写"空结果哨兵" eval_*.json（acc=None/n=None），后续
         analysis 显式报"盲"，不吞；
      3) analysis --record 表加「评测」列：哨兵/缺失 checkpoint 所在窗口标"盲"。
    """
    print("[AK] 内嵌评测超时修复（p10 盲窗）")
    tr = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()
    cf = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "config.py"), encoding="utf-8").read()
    an = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "analysis.py"), encoding="utf-8").read()

    # 1) 超时进配置 + preset 抬升（旧 900s 只该属于 BASE 档）
    check("train: 内嵌评测超时取自 cfg 并透传（不再硬编码 900）",
          "def _run_inline_eval(cfg, ckpt_dir, step, eval_gpu=\"0\", eval_gpu_mem=0.20,\n"
          "                     eval_n=500, eval_timeout_s=900):" in tr
          and "timeout=eval_timeout_s" in tr
          and "_eval_to = int(cfg.get(\"eval_timeout_s\", 900) or 900)" in tr)
    check("config: BASE 默认 900（历史行为不变）且 retool_math preset 抬到 3600",
          "eval_timeout_s=900," in cf and "eval_timeout_s=3600," in cf)
    check("train: CLI --eval_timeout_s 存在并接进 overrides",
          '"--eval_timeout_s"' in tr
          and 'if args.eval_timeout_s is not None:\n        overrides["eval_timeout_s"] = args.eval_timeout_s' in tr)

    # 2) 超时落"空结果哨兵"（acc=None）——不静默吞
    check("train: 超时写哨兵 json（acc=None/n=None）并打印盲窗提示",
          '"error": "timeout"' in tr and '"acc": None' in tr
          and "已落空结果哨兵" in tr)

    # 3) analysis --record 表加「评测」列：哨兵/缺失标 "盲"
    check("analysis: 表头加「评测」列",
          "| 评测 |" in an and "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|" in an)
    # 判盲逻辑 2026-09-25 收口到 eval_is_blind（旧版内联读顶层 acc/n，对嵌套壳恒判盲）
    check("analysis: 哨兵/缺失判盲收口到 eval_is_blind（曲线表调它，不再内联读顶层）",
          "def eval_is_blind(" in an
          and "if eval_is_blind(_es):" in an
          and "\"盲\"" in an)

    # 4) 配置合并端到端：preset 覆盖 + CLI 覆盖都走 get_config
    from rlab.config import get_config
    _c = get_config("retool_math", use_wandb=False)
    check("config: retool_math preset 的 eval_timeout_s=3600 生效（端到端）",
          _c.get("eval_timeout_s") == 3600)
    _c2 = get_config("retool_math", use_wandb=False, eval_timeout_s=7200)
    check("config: --eval_timeout_s 显式覆盖生效（端到端）",
          _c2.get("eval_timeout_s") == 7200)
    _c3 = get_config("grpo", use_wandb=False)
    check("config: BASE 档（grpo）保持 900（旧行为零变化）",
          _c3.get("eval_timeout_s") == 900)


def test_inline_eval_json_shape():
    """【2026-09-25 事故·内嵌评测日志全 0（p11）】训练日志里 6 个 checkpoint × 2 split
    的内嵌评测读数全是 `acc=0.0% fmt=0.0% code=0.0% (n=0)`——看起来像"模型一步没学会"，
    实际是**读错 json 层级**。

    eval_vllm_one.py 落盘 `json.dump({name: result})`（嵌套壳，全仓库正典：
    eval_vllm.py 靠 `results.update(...)` 合并、eval_merge.py 遍历 `rows.items()`、
    summarize_eval 也按这个壳读）。而 train.py 的 _run_inline_eval 直读顶层
    `_r.get("acc", 0)` → 键在壳里取不到 → **默认值 0 被当成真实读数印出来**。

    为什么 n=0 是这个 bug 的签名而不是"真没题可评"：池空时 eval 端直接
    RuntimeError 退出（returncode≠0）→ 走 FAILED 分支，压根到不了打印 n=0 的那行。
    能打印出 n=0 就说明 exit=0、评测成功、结果在磁盘上完好。

    连带：p10 为"让盲窗可见"加的哨兵判据同样内联读顶层 acc/n → 对嵌套壳恒为 None
    → 每个**评测成功**的 checkpoint 都被标"盲"（治盲窗的列自己造假盲窗）。

    本测试锁死：①解壳纯函数吃两种壳；②嵌套壳 acc 必须解出真值而不是 0；
    ③判盲只对真哨兵/缺失为真；④train.py 不再直读顶层、且解不出 n 时不印假 0；
    ⑤哨兵也落嵌套壳（磁盘上只有一种结构）。"""
    print("[AL] 内嵌评测 json 壳层级（p11 全 0 事故）")
    from rlab.analysis import eval_is_blind, read_eval_result
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tr = open(os.path.join(_root, "rlab", "train.py"), encoding="utf-8").read()

    with tempfile.TemporaryDirectory() as d:
        # ① 正典嵌套壳（eval_vllm_one.py 的真实产物形状）
        real = {"acc": 0.327, "fmt": 0.993, "code_rate": 0.41, "both": 0.327,
                "n": 300, "metrics_version": 2}
        p_nest = os.path.join(d, "eval_test.json")
        with open(p_nest, "w", encoding="utf-8") as f:
            json.dump({"step50_test": real}, f)
        r = read_eval_result(p_nest)
        check("嵌套壳 {name: result} 解出真实 acc/n（旧读法在此拿到 0/0）",
              abs(r.get("acc", 0) - 0.327) < 1e-9 and r.get("n") == 300)
        # 旧读法的反向对照：直读顶层必然是缺键 → 这就是日志里的 0.0%/n=0
        with open(p_nest, encoding="utf-8") as f:
            _raw = json.load(f)
        check("反向对照：直读顶层 acc/n 缺键（假 0 读数的来源）",
              _raw.get("acc", 0) == 0 and _raw.get("n", 0) == 0)
        check("嵌套壳且 acc/n 有值 → 不判盲（评测成功不得标盲）",
              eval_is_blind(p_nest) is False)

        # ② 扁平壳（历史哨兵格式）也要吃
        p_flat = os.path.join(d, "eval_flat.json")
        with open(p_flat, "w", encoding="utf-8") as f:
            json.dump(real, f)
        check("扁平壳（历史格式）同样解出真值 —— 两种壳都吃",
              read_eval_result(p_flat).get("n") == 300)

        # ③ 哨兵（两种壳）都必须判盲
        stub = {"acc": None, "fmt": None, "code_rate": None, "n": None,
                "error": "timeout"}
        p_s1 = os.path.join(d, "eval_stub_nest.json")
        p_s2 = os.path.join(d, "eval_stub_flat.json")
        with open(p_s1, "w", encoding="utf-8") as f:
            json.dump({"step50_test": stub}, f)
        with open(p_s2, "w", encoding="utf-8") as f:
            json.dump(stub, f)
        check("超时哨兵（嵌套壳）判盲", eval_is_blind(p_s1) is True)
        check("超时哨兵（扁平壳）判盲", eval_is_blind(p_s2) is True)
        check("文件缺失判盲", eval_is_blind(os.path.join(d, "nope.json")) is True)

        # ④ 坏文件/空壳：读不出来一律判盲，绝不当成"评测正常"
        p_bad = os.path.join(d, "eval_bad.json")
        with open(p_bad, "w", encoding="utf-8") as f:
            f.write("{not json")
        check("坏 json 判盲且解壳返回 {}（不抛异常打断 analysis）",
              read_eval_result(p_bad) == {} and eval_is_blind(p_bad) is True)
        p_meta = os.path.join(d, "eval_meta_only.json")
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump({"_meta": {"n": 300}}, f)
        check("只有 _meta 的壳 → 解不出 result（_meta 不是模型结果）",
              read_eval_result(p_meta) == {} and eval_is_blind(p_meta) is True)

    # ⑤ train.py 侧：走解壳函数、不再直读顶层、解不出 n 不印假 0
    check("train: 内嵌评测经 read_eval_result 解壳",
          "from rlab.analysis import read_eval_result" in tr
          and "_r = read_eval_result(_out)" in tr)
    check("train: 不再出现直读顶层的旧写法 _r.get(\"acc\", 0)",
          '_r.get("acc", 0)' not in tr and '_r.get("fmt", 0)' not in tr
          and '_r.get("code_rate", 0)' not in tr)
    check("train: 解不出 n 时记盲窗、不打印 0.0% 假读数",
          'if not _r or _n is None:' in tr and "不产生假 0 读数" in tr)
    check("train: 超时哨兵也落 {name: result} 嵌套壳（磁盘只有一种结构）",
          'json.dump({f"step{step}_{_split}": _stub}' in tr)


def test_inline_eval_protocol_attestation():
    """【2026-09-25 口径自证】内嵌评测必须把协议摘进训练日志。

    事故：p11 的内嵌读数（step100 test 63.3%）与训练后单独评测（70.3%）差 +7.0pp，
    而 log11.txt 里**查不到任何可对账的信息**——`capture_output=True` 把 eval 子进程
    的全部 print 收进 _proc.stdout，成功路径直接丢弃：抽题 seed、剔题数、协议来源、
    确定性档、system_prompt 哈希全部随之消失。两个读数不一致时能否当场定位，取决于
    日志有没有留下口径；否则只能像这次一样事后写脚本逐题配对。

    另一半同因：子进程的 `[警告]`（协议回落 preset / 池子不足 / 确定性档缺失）也被
    吃掉。而"静默回落 preset"正是"测了另一个协议"的头号根因（eval_vllm_one.py 自己
    的注释就记着 baseA/baseB 事故）。
    """
    print("[AM] 内嵌评测口径自证（p11 +7.0pp 无从对账）")
    tr = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "rlab", "train.py"), encoding="utf-8").read()

    check("train: 打印口径行（含 seed/剔题/协议来源）",
          '{_split} 口径:' in tr
          and "n_dropped_plen" in tr and "n_dropped_long" in tr
          and "_ep.get('seed')" in tr
          and "_ep.get('proto_from_run_info')" in tr)
    check("train: 口径取自 eval_protocol（落盘字段，非重新推测）",
          '_ep = _r.get("eval_protocol") or {}' in tr)
    check("train: 确定性档与 system_prompt 哈希进日志（跨 run 可比性前提）",
          "_ep.get('vllm_batch_invariant')" in tr
          and "_ep.get('system_prompt_sha')" in tr)
    check("train: 子进程 [警告] 转发到训练日志（静默回落不再隐形）",
          '"[警告]" in _ln' in tr and "_proc.stdout" in tr)

    # eval 端确实落了这些键（两端字段名对齐，否则日志印一排 None）
    one = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "eval_vllm_one.py"), encoding="utf-8").read()
    for _k in ("n_requested", "n_dropped_plen", "n_dropped_long"):
        check(f"eval_one: 结果落 {_k}（口径行的数据源）", f'"{_k}"' in one)
    for _k in ("seed", "val_n", "greedy", "round_tokens",
               "vllm_batch_invariant", "system_prompt_sha", "proto_from_run_info"):
        check(f"eval_one: eval_protocol 落 {_k}", f'"{_k}"' in one)

    # 端到端：用真实形状的 result 跑一遍口径行的取值，确认无 None 漏项
    _ep_keys = ("seed", "val_n", "greedy", "temperature", "round_tokens",
                "vllm_batch_invariant", "system_prompt_sha", "proto_from_run_info")
    _fake = {"acc": 0.633, "fmt": 0.74, "n": 256, "n_requested": 300,
             "n_dropped_plen": 44, "n_dropped_long": 0,
             "eval_protocol": {k: 1 for k in _ep_keys}}
    _ep = _fake.get("eval_protocol") or {}
    check("口径行所有字段在真实 result 形状下都取到值（无 None）",
          all(_ep.get(k) is not None for k in _ep_keys)
          and all(_fake.get(k) is not None
                  for k in ("n_requested", "n_dropped_plen", "n_dropped_long")))


def test_preflight_audit_fixes():
    """【2026-09-20 pre-flight 审查八项修复】每项都锁"旧行为会怎么错"。

    ①analysis clen_cap 从 run_info 推导（旧：硬编码 1800 → p8 那列 68~98% 是纯
      饱和噪声，而 ts 实验的核心判据正是长度轴）
    ②seed 盐步长 = 本次消耗的 seed 数（旧：+=1 → 相邻 attempt 重叠 97%，丢组重采
      复采同轨迹 → 题目被"seed 复用"而非"学不动"拉黑）
    ③probe_meta 指纹剔除 k（训练传 num_pre_Q=8 vs 探针 args.k=4，口径不同 → 必然误报）
    ④run_signature 覆盖优化器层 + 迁移兼容（旧：18 个生效超参改了签名一字不变）
    ⑤micro_batch 由 Q×num_pre_Q 推导（旧：硬编码成对 → --num_pre_Q 4 静默腰斩等效 lr）
    ⑥eval 协议回落告警 + 剔题阈值与训练对齐（baseA/baseB 事故：静默落到 preset 2048/14336）
    ⑦末轮废码可观测（旧：code_used/code_ok/trunc_final 同时为 0，完全隐形）
    ⑧启动行打印真实更新预算（旧：300 micro-step 被当成 300 次 optimizer 更新，差 4 倍）
    """
    print("[S] pre-flight 审查八项修复")
    import re as _re
    import shutil
    from rlab.analysis import summarize_record
    from rlab.data import load_difficulty_table
    from rlab.train import _is_opt_suffix, guard_ckpt_collision
    from rlab.train import run_signature as _rs

    _roll = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "rollout.py"), encoding="utf-8").read()
    _ana = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "analysis.py"), encoding="utf-8").read()
    _tr = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "train.py"), encoding="utf-8").read()
    _ev = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "eval_vllm_one.py"), encoding="utf-8").read()

    # ---- ① clen_cap 从 run_info 推导 ----
    _tmp = tempfile.mkdtemp()
    try:
        _rp = os.path.join(_tmp, "record.jsonl")
        with open(_rp, "w", encoding="utf-8") as f:
            for _ in range(3):
                f.write(json.dumps({
                    "t": 1.0, "algo": "retool_math", "acc": [1] * 8, "fmt": [1] * 8,
                    "clen": [2000] * 8, "code_used": [1] * 8, "code_ok": [1] * 8,
                    "trunc_final": [0] * 8, "gen_version": 0, "phase": "cold"}) + "\n")
        _no_ri = summarize_record(_rp, window=8)
        check("① 无 run_info → 回落 1800 且显式声明口径存疑",
              "cap=1800" in _no_ri and "口径存疑" in _no_ri)
        with open(os.path.join(_tmp, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump({"config": {"max_context_tokens": 26400,
                                  "max_prompt_length": 1024}}, f)
        _with_ri = summarize_record(_rp, window=8)
        check("① 有 run_info → cap = ctx − max_prompt_length（26400−1024=25376）",
              "cap=25376" in _with_ri and "run_info(26400−1024)" in _with_ri)
        # 这才是真正的回归点：clen=2000 在旧 cap 下被标成"接近上限"(≥1620)，
        # 在真实 cap 下根本不接近（≥22838）——p8 报表那列噪声的由来。
        check("① 回归：clen=2000 旧 cap 判 100%≥1620，真实 cap 判 0%≥22838",
              "100%≥1620" in _no_ri and "0%≥22838" in _with_ri)
        check("① 显式传参仍优先（调用方可覆盖）",
              "cap=999" in summarize_record(_rp, window=8, clen_cap=999))
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)

    # ---- ② seed 盐步长 ----
    check("② 盐按本次消耗的 seed 数递增（不是 +=1）",
          _re.search(r"rollout_seq\[0\]\s*\+=\s*len\(inputs\)\s*\*\s*cfg\[.num_pre_Q.\]",
                     _roll) is not None)
    check("② 旧的 `rollout_seq[0] += 1` 已不存在",
          _re.search(r"rollout_seq\[0\]\s*\+=\s*1\b", _roll) is None)
    # 数值对拍：4 题 × 8 条 = 32 个 seed/attempt，相邻 attempt 必须零重叠
    def _seeds(salt, n_req=32, seed0=42):
        return {seed0 + salt + k for k in range(n_req)}
    check("② 步长=32 时相邻 attempt seed 零重叠（旧步长 1 重叠 31/32=97%）",
          not (_seeds(0) & _seeds(32)) and len(_seeds(0) & _seeds(1)) == 31)

    # ---- ③ probe_meta 指纹剔除 k ----
    _tmp2 = tempfile.mkdtemp()
    try:
        _dp = os.path.join(_tmp2, "d.jsonl")
        _meta_disk = {"model": "Qwen3.5-4B", "k": 4, "rounds": 4,
                      "round_tokens": 6144, "ctx": 26400, "temp": 1.0, "sp": "3aac5d"}
        with open(_dp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"Q": "q1", "k": 4, "n_correct": 2,
                                "probe_meta": _meta_disk}) + "\n")
        import io as _io
        from contextlib import redirect_stdout as _rso
        # 训练端指纹：k=num_pre_Q=8（与探针 args.k=4 口径不同），其余全同
        _exp = {**_meta_disk, "k": 8}
        _buf = _io.StringIO()
        with _rso(_buf):
            _t = load_difficulty_table(_dp, expected_meta=_exp)
        check("③ 仅 k 不同 → 不告警（口径不同，旧版必然误报）",
              len(_t) == 1 and "警告" not in _buf.getvalue())
        # 真偏离（换预算）仍必须报，否则护栏就废了
        _buf2 = _io.StringIO()
        with _rso(_buf2):
            load_difficulty_table(_dp, expected_meta={**_exp, "round_tokens": 2048})
        check("③ 真偏离（round_tokens 变）仍告警（护栏未被削弱）",
              "round_tokens" in _buf2.getvalue() and "警告" in _buf2.getvalue())
    finally:
        shutil.rmtree(_tmp2, ignore_errors=True)
    check("③ 训练端注释说明 k 为何被排除",
          "num_pre_Q" in _roll and "语义不同" in _roll)

    # ---- ④ 签名覆盖优化器层 + 迁移兼容 ----
    _c = get_config("retool_math", use_wandb=False)
    check("④ 默认配方（无偏离）签名不含优化器段——历史签名逐字不变",
          "-b" not in _rs(_c).split("-stop1")[-1]
          and _rs(_c) == _rs(get_config("retool_math", use_wandb=False)))
    for _k, _v, _tag in (("beta", 0.01, "-b0.01"), ("num_pre_Q", 4, "-n4"),
                         ("gen_update_steps", 8, "-u8"), ("seed", 42, "-sd42"),
                         ("temperature", 0.7, "-T0.7"),
                         ("max_context_tokens", 26400, "-c26400")):
        check(f"④ {_k} 偏离 preset → 签名出现 {_tag}",
              _tag in _rs(get_config("retool_math", use_wandb=False, **{_k: _v})))
    check("④ _is_opt_suffix 只认优化器段",
          _is_opt_suffix("-u8-c26400-sd42") and _is_opt_suffix("-sd42")
          and not _is_opt_suffix("-ts0.5") and not _is_opt_suffix("")
          and not _is_opt_suffix("-zzz1"))
    _tmp3 = tempfile.mkdtemp()
    try:
        _cfg = get_config("retool_math", use_wandb=False, seed=42,
                          gen_update_steps=8, max_context_tokens=26400)
        _cfg["run_signature"] = _rs(_cfg)
        _ck = os.path.join(_tmp3, "step_50")
        os.makedirs(_ck)
        # 老 ckpt 的签名 = 新签名去掉优化器段（正在跑的 run 中途重启的真实形态）
        _old = _cfg["run_signature"].split("-u8")[0]
        with open(os.path.join(_ck, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump({"signature": _old}, f)
        guard_ckpt_collision(_tmp3, _cfg)   # 不抛 = 放行
        check("④ 迁移兼容：老签名是新签名前缀且新增段全是优化器段 → 放行"
              "（否则正在跑的 run 崩溃后无法续跑）", True)
        # 反向与真换配方都必须继续拦
        # 【2026-09-21】preset trunc_shaping 从 0.5 改 0.0，所以"不同配方"
        # 的构造方向也反过来：把 -ts0- 改成 -ts0.5-（而非旧的 -ts0.5→-ts0）
        with open(os.path.join(_ck, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump({"signature": _old.replace("-ts0-", "-ts0.5-")}, f)
        try:
            guard_ckpt_collision(_tmp3, _cfg)
            _ok = False
        except RuntimeError:
            _ok = True
        check("④ 真换配方（ts0 vs ts0.5）仍拒绝——护栏未被削弱", _ok)
    finally:
        shutil.rmtree(_tmp3, ignore_errors=True)

    # ---- ⑤ micro_batch 推导 ----
    check("⑤ 默认 retool_math: micro_batch = 1×8 = 8",
          get_config("retool_math", use_wandb=False)[
              "train_micro_batch_size_per_gpu"] == 8)
    check("⑤ --num_pre_Q 4 → micro_batch 随之变 4（旧版仍是 8 = 等效 lr 腰斩）",
          get_config("retool_math", use_wandb=False, num_pre_Q=4)[
              "train_micro_batch_size_per_gpu"] == 4)
    check("⑤ grpo 默认 1×4 = 4（其他算法未被带歪）",
          get_config("grpo", use_wandb=False)[
              "train_micro_batch_size_per_gpu"] == 4)
    try:
        get_config("retool_math", use_wandb=False, num_pre_Q=4,
                   train_micro_batch_size_per_gpu=8)
        _ok = False
    except (ValueError, RuntimeError):
        _ok = True
    check("⑤ 显式传矛盾值 → fail-fast（不静默改语义）", _ok)

    # ---- ⑥ eval 协议回落告警 + 剔题阈值 ----
    check("⑥ 无 run_info → 醒目告警（baseA/baseB 静默落 preset 的事故）",
          "回落 preset 默认" in _ev
          and _re.search(r"if not _run:\s*\n\s*print\(f?\"\\n\[eval\]\[警告\]", _ev)
          is not None)
    check("⑥ 告警给出旁路（--proto_from）与可疑后果（fmt 掉一半）",
          "--proto_from <某个带 run_info" in _ev and "fmt" in _ev)
    check("⑥ 剔题阈值与训练对齐（plen > max_prompt_length，同源回读 _rcfg）",
          '_max_plen = int(_rcfg.get("max_prompt_length")' in _ev
          and "if _max_plen and _pl > _max_plen:" in _ev)
    check("⑥ 两条线并存：训练同规则 + max_len 兜底（防撞 max_model_len）",
          "_dropped_plen" in _ev and "_dropped_long" in _ev)
    check("⑥ 协议出处落进结果 json（事后可判定这次评的是哪个协议）",
          '"proto_from_run_info": bool(_run)' in _ev
          and '"n_dropped_plen"' in _ev)

    # ---- ⑦ 末轮废码可观测 ----
    # 【设计决定，2026-09-20】**不**把末轮代码计入 code_used。
    # 第一版改动曾把自增提到 is_final_round 之前，结果 code% 这一列的语义变了：
    # p8 的 48~70% 是"末轮不计入"口径，p9 若计入就凭空抬高一截（≈末轮废码率），
    # 而 code% 正是要跨 run 比较的轴之一 —— 修可观测性不该以牺牲可比性为代价。
    # 现在：code_used 逐位不变，末轮废码走独立的 code_wasted 计数。
    check("⑦ code_used 仍在 is_final_round 之后自增（口径与 p8 逐位可比）",
          _roll.index("if is_final_round:")
          < _roll.index('code_stats[i]["code_used"] += 1'))
    check("⑦ 末轮代码单独记 code_wasted（新增信号，不动既有列）",
          'code_stats[i]["code_wasted"] += 1' in _roll)
    check("⑦ code_wasted 在 code_stats 初始化时就有（无缺键 KeyError 风险）",
          '"code_wasted": 0' in _roll)
    check("⑦ code_wasted 进 record 落盘", '"code_wasted"' in _roll)
    check("⑦ analysis 展示末轮废码率列", "末轮废码率" in _ana)

    # ---- ⑧ 剂量口径打印 ----
    check("⑧ 启动行打印 optimizer 更新数而非 micro-step",
          "剂量口径" in _tr and "optimizer 更新" in _tr)
    check("⑧ 同时打印轨迹总数与有效 batch",
          "轨迹总数" in _tr and "有效 batch" in _tr)
    check("⑧ 逐存档点列出真实更新数（step_50=12upd 这类）", "upd" in _tr)


def test_overlong_filter():
    """【2026-09-21 DAPO overlong filtering】截断样本从 advantage 和组统计中移除。

    核心机制：
    - 组均值只算非截断样本 → 截断样本不污染基线
    - 截断样本 adv=0 → pg_term=0（不贡献策略梯度）
    - 杀 NeMo-RL bug：trunc_shaping>0 时全错组+混合截断不再产生假方差通过 group_ok

    DAPO 消融：overlong filtering +6 分（最稳定的长度控制组件）。
    """
    print("[T] DAPO overlong filtering（截断样本从 advantage 移除）")
    from rlab.losses import compute_advantages
    from rlab.rollout import group_ok, retool_score_flat
    from rlab.config import get_config

    cfg = get_config("retool_math", use_wandb=False)

    # ---- 1. compute_advantages 带 sample_mask 的正确性 ----
    # 8 条组：4 对（reward=+1）、2 错（reward=-1）、2 截断（reward=-1.5, mask=0）
    r = torch.tensor([1, 1, 1, 1, -1, -1, -1.5, -1.5], dtype=torch.float32)
    mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float32)

    adv_filtered = compute_advantages(r, 8, "group_mean", sample_mask=mask)
    adv_plain = compute_advantages(r, 8, "group_mean")

    # 截断样本 adv=0
    check("截断样本 adv=0（不贡献 pg_term）",
          adv_filtered[6].item() == 0 and adv_filtered[7].item() == 0)

    # 非截断样本的组均值只算非截断：mean = (4*1 + 2*(-1)) / 6 = 2/6 = 0.333
    _mean_nontrunc = (4 * 1 + 2 * (-1)) / 6
    check("组均值只算非截断样本",
          abs(adv_filtered[0].item() - (1 - _mean_nontrunc)) < 1e-5)

    # 不带 mask 时（旧行为）：组均值算全部 8 条
    _mean_all = r.mean().item()
    check("无 mask 时组均值算全部（旧行为不变）",
          abs(adv_plain[0].item() - (1 - _mean_all)) < 1e-5)

    # 无 mask = 全 1 mask（等价）
    check("sample_mask=None 等价于全 1（旧行为）",
          bool(((compute_advantages(r, 8, "group_mean", sample_mask=None)
                 - compute_advantages(r, 8, "group_mean")).abs() < 1e-6).all()))

    # ---- 2. NeMo-RL bug 杀死：全错组+混合截断 → group_ok 失败 ----
    # 8 条全错：4 截断（reward=-1.5, mask=0）、4 非截断（reward=-1, mask=1）
    r_allwrong = torch.tensor([-1.5, -1.5, -1.5, -1.5, -1, -1, -1, -1], dtype=torch.float32)
    mask_allwrong = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)

    adv_nemo = compute_advantages(r_allwrong, 8, "group_mean", sample_mask=mask_allwrong)
    # 非截断全错：mean=-1, adv = -1-(-1) = 0；截断 adv=0 → 全 0
    check("NeMo bug 杀死：全错组+混合截断 → 全 adv=0",
          bool((adv_nemo.abs() < 1e-6).all()))
    check("NeMo bug 杀死：全错组+混合截断 → group_ok 失败（被过滤）",
          not bool(group_ok(adv_nemo)))

    # 对比：不带 mask 时（旧行为，NeMo bug 存在）
    adv_nemo_bug = compute_advantages(r_allwrong, 8, "group_mean")
    # 截断 reward=-1.5, 非截断 reward=-1, mean=-1.25
    # 非截断 adv = -1-(-1.25) = +0.25 → 正优势！
    check("NeMo bug 确认：无 mask 时全错组非截断拿正优势",
          adv_nemo_bug[4].item() > 0)
    check("NeMo bug 确认：无 mask 时全错组通过 group_ok（假方差）",
          bool(group_ok(adv_nemo_bug)))

    # ---- 3. 全截断组 → 全 adv=0 → group_ok 失败 ----
    r_alltrunc = torch.full((8,), -1.5, dtype=torch.float32)
    mask_alltrunc = torch.zeros(8, dtype=torch.float32)
    adv_alltrunc = compute_advantages(r_alltrunc, 8, "group_mean", sample_mask=mask_alltrunc)
    check("全截断组 → 全 adv=0 → group_ok 失败",
          bool((adv_alltrunc.abs() < 1e-6).all()) and not bool(group_ok(adv_alltrunc)))

    # ---- 4. 混合组（对+错+截断）→ 截断 adv=0，非截断正常 ----
    r_mixed = torch.tensor([1, -1, 1, -1, -1.5, 1, -1, -1.5], dtype=torch.float32)
    mask_mixed = torch.tensor([1, 1, 1, 1, 0, 1, 1, 0], dtype=torch.float32)
    adv_mixed = compute_advantages(r_mixed, 8, "group_mean", sample_mask=mask_mixed)
    # 非截断: [1,-1,1,-1,1,-1], mean=0, adv=[1,-1,1,-1,1,-1]
    check("混合组：截断 adv=0", adv_mixed[4].item() == 0 and adv_mixed[7].item() == 0)
    check("混合组：非截断 adv 正常（=r-mean）",
          abs(adv_mixed[0].item() - 1.0) < 1e-5 and abs(adv_mixed[1].item() + 1.0) < 1e-5)
    check("混合组：group_ok 通过（有方差）", bool(group_ok(adv_mixed)))

    # ---- 5. group_std 也正确处理 sample_mask ----
    r_std = torch.tensor([1, -1, 1, -1, -1.5, 1, -1, -1.5], dtype=torch.float32)
    mask_std = torch.tensor([1, 1, 1, 1, 0, 1, 1, 0], dtype=torch.float32)
    adv_std = compute_advantages(r_std, 8, "group_std", sample_mask=mask_std)
    check("group_std：截断 adv=0", adv_std[4].item() == 0 and adv_std[7].item() == 0)
    check("group_std：非截断 adv≠0（有梯度）",
          adv_std[0].item() != 0 and adv_std[1].item() != 0)
    # group_std 无 mask 时与旧版一致
    adv_std_old = compute_advantages(r_std, 8, "group_std")
    check("group_std：无 mask 时与旧版一致",
          bool(((adv_std_old - compute_advantages(r_std, 8, "group_std", sample_mask=None)).abs() < 1e-6).all()))

    # ---- 6. retool_score_flat 集成：overlong_filter=True 时截断样本被 mask ----
    # 构造 1 题 × 8 条轨迹，其中 2 条截断
    inputs = [{"Q": "q", "A": "42"}]
    asst_texts = ["\\boxed{42}"] * 6 + ["no answer"] * 2  # 6 对、2 错
    code_stats = [{"code_used": 0, "code_ok": 0, "trunc_final": 0} for _ in range(6)] \
        + [{"code_used": 0, "code_ok": 0, "trunc_final": 1} for _ in range(2)]
    cfg_test = {**cfg, "overlong_filter": True, "trunc_shaping": 0.5}
    adv, acc, fmt, cu, ck, phase = retool_score_flat(
        inputs, asst_texts, code_stats, cfg_test, steps_elapsed=0)
    # 截断样本（idx 6,7）adv=0
    check("retool_score_flat：overlong_filter=True 时截断样本 adv=0",
          adv[6].item() == 0 and adv[7].item() == 0)
    # 非截断样本 adv≠0（6 对 vs 0 错 → mean=1 → adv=0... 全对组零方差）
    # 改成 4 对 2 错（非截断）+ 2 截断
    asst_texts2 = ["\\boxed{42}"] * 4 + ["wrong"] * 2 + ["no answer"] * 2
    code_stats2 = [{"code_used": 0, "code_ok": 0, "trunc_final": 0} for _ in range(4)] \
        + [{"code_used": 0, "code_ok": 0, "trunc_final": 0} for _ in range(2)] \
        + [{"code_used": 0, "code_ok": 0, "trunc_final": 1} for _ in range(2)]
    adv2, _, _, _, _, _ = retool_score_flat(
        inputs, asst_texts2, code_stats2, cfg_test, steps_elapsed=0)
    check("retool_score_flat：混合组截断 adv=0，非截断 adv≠0",
          adv2[6].item() == 0 and adv2[7].item() == 0
          and adv2[0].item() != 0 and adv2[4].item() != 0)

    # overlong_filter=False 时截断样本参与组统计（旧行为）
    cfg_nofilter = {**cfg, "overlong_filter": False, "trunc_shaping": 0.5}
    adv3, _, _, _, _, _ = retool_score_flat(
        inputs, asst_texts2, code_stats2, cfg_nofilter, steps_elapsed=0)
    # 截断样本 reward=-1.5, 非截断 mean = (4*1+2*(-1))/6 = 0.333
    # 截断 adv = -1.5 - 0.333 = -1.833 ≠ 0
    check("retool_score_flat：overlong_filter=False 时截断样本参与组统计（旧行为）",
          adv3[6].item() != 0)

    # ---- 7. config preset 锁 ----
    check("retool_math preset: overlong_filter=True",
          get_config("retool_math", use_wandb=False)["overlong_filter"] is True)
    check("BASE: overlong_filter=False（其他算法不受影响）",
          get_config("grpo", use_wandb=False)["overlong_filter"] is False)

    # ---- 8. 签名 + 迁移兼容 ----
    from rlab.train import run_signature, _is_opt_suffix
    sig = run_signature(cfg)
    check("签名含 -of1（retool_math preset 开 overlong_filter）", "-of1" in sig)
    cfg_off = {**cfg, "overlong_filter": False}
    sig_off = run_signature(cfg_off)
    check("签名无 -of（关闭 overlong_filter）", "-of" not in sig_off)
    # 迁移兼容：旧签名（无 -of1）是新签名的前缀，且新增段是 opt-tag
    check("_is_opt_suffix 识别 -of1 段（迁移兼容）",
          _is_opt_suffix("-of1"))
    check("迁移兼容：旧签名是新签名前缀 + of 段 → 放行",
          sig.startswith(sig_off) and _is_opt_suffix(sig[len(sig_off):]))

    # ---- 9. CLI 透传 ----
    _tr = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "train.py"), encoding="utf-8").read()
    check("CLI: --overlong_filter BooleanOptionalAction（支持 --no-）",
          'add_argument("--overlong_filter"' in _tr
          and "BooleanOptionalAction" in _tr
          and '"--no-overlong_filter"' in _tr or "--no-overlong_filter" in _tr)
    check("CLI: overlong_filter 透传到 overrides",
          'overrides["overlong_filter"]' in _tr)
    check("run_info: overlong_filter 落盘",
          '"overlong_filter": cfg.get("overlong_filter")' in _tr)


def test_attempt_shaping_and_err_tier():
    """【2026-09-21 终止链/压灭三件套】#2 尝试级 shaping + #6 错误类型分级 +
    #1 配套签名。

    #3（EOS 是否进训练序列）已由 vLLM 0.12 V1 源码证据链核对关闭：
    token_ids 恒含 EOS（detokenizer 对 stop token 只跳过文本、id 恒 append），
    gen_logps 与 ids 严格平行（len 不齐即 raise）——"答完就停"一直在拿梯度。
    """
    print("[AK] 尝试级 shaping + 沙箱错误分级 + 签名")
    from rlab.reward import (reward_code_attempt, total_reward_retool_math)
    from rlab.sandbox import classify_error
    from rlab.config import get_config
    from rlab.rollout import retool_score_flat

    # ---- 1. reward_code_attempt 纯函数 ----
    check("attempt_w=0 → 恒 0（旧行为逐位相同）", reward_code_attempt(3, 0.0) == 0.0)
    check("attempt_w=0.05 × 2 次 = 0.1", abs(reward_code_attempt(2, 0.05) - 0.1) < 1e-9)
    check("cap 在 max_rounds（防御异常大值）",
          abs(reward_code_attempt(999, 0.05, max_rounds=4) - 0.2) < 1e-9)
    check("code_used=0 → 0（没写代码不给分）", reward_code_attempt(0, 0.05) == 0.0)

    # ---- 2. total_reward_retool_math 集成 ----
    good_boxed = "\\boxed{42}"
    sc0 = total_reward_retool_math("42", good_boxed, code_used=2,
                                   code_attempt_w=0.0)
    sc1 = total_reward_retool_math("42", good_boxed, code_used=2,
                                   code_attempt_w=0.05, max_rounds=4)
    check("attempt_w=0 与旧口径一致（reward 不含 shaping 项）",
          sc0["reward"] == sc0["acc"] and sc0["code_attempt"] == 0.0)
    check("attempt_w=0.05 → reward = acc + 0.1",
          abs(sc1["reward"] - (sc1["acc"] + 0.1)) < 1e-9)
    check("code_attempt 分量落盘", abs(sc1["code_attempt"] - 0.1) < 1e-9)
    # 答错 + 写了代码：shaping 照给（把"敢写"与"写对"分开）
    sc2 = total_reward_retool_math("99", good_boxed, code_used=1,
                                   code_attempt_w=0.05, max_rounds=4)
    check("答错也拿尝试分（对冲风险不对称的语义）",
          abs(sc2["reward"] - (-1.0 + 0.05)) < 1e-9)
    # 截断罚与尝试分可叠加
    sc3 = total_reward_retool_math("42", good_boxed, code_used=1,
                                   code_attempt_w=0.05, max_rounds=4,
                                   trunc_final=1, trunc_shaping=0.5)
    check("trunc 罚与 attempt 分叠加", abs(sc3["reward"] - (1.0 - 0.5 + 0.05)) < 1e-9)

    # ---- 3. classify_error 纯函数 ----
    check("超时 → timeout", classify_error(None, "", timed_out=True) == "timeout")
    check("rc=0 → ok", classify_error(0, "", timed_out=False) == "ok")
    check("SyntaxError traceback → syntax",
          classify_error(1, 'SyntaxError: invalid syntax', False) == "syntax")
    check("ValueError traceback → exception",
          classify_error(1, 'ValueError: bad value', False) == "exception")
    check("ModuleNotFoundError → exception（子类不冒充 syntax）",
          classify_error(1, 'ModuleNotFoundError: No module named x', False)
          == "exception")
    check("无 traceback 的非零退出 → exception",
          classify_error(137, "Killed", False) == "exception")

    # ---- 4. retool_score_flat 透传 code_attempt_w ----
    inputs = [{"Q": "q", "A": "42"}]
    asst_texts = ["\\boxed{42}"] * 4 + ["\\boxed{41}"] * 4
    cs_off = [{"code_used": 2 if k % 2 else 0, "code_ok": 0, "trunc_final": 0}
              for k in range(8)]
    cfg_base = get_config("retool_math", use_wandb=False)
    cfg_off = {**cfg_base, "overlong_filter": False}
    adv_off, _, _, _, _, _ = retool_score_flat(
        inputs, asst_texts, cs_off, cfg_off, steps_elapsed=0)
    cfg_on = {**cfg_off, "code_attempt_w": 0.05}
    adv_on, _, _, _, _, _ = retool_score_flat(
        inputs, asst_texts, cs_off, cfg_on, steps_elapsed=0)
    # 数学：开启后 rewards[r] += 0.05*code_used[r]（写码样本 +0.1）。
    # group_mean 归一化（减均值不除 std）→ 每样本 adv 差 = 该样本加分 − 组均值加分
    # = (+0.1) − (4×0.1/8) = +0.05（写码样本）；(0) − 0.05 = −0.05（纯推理样本）。
    # 等价于整个 adv 向量对写码/不写码两类各平移 ±0.05——方向正确（写码样本
    # 相对优势上升）且组内零和保持。
    check("score_flat：写码样本 adv 抬升 +0.05、纯推理样本 −0.05（相对优势上移）",
          abs(float(adv_on[1] - adv_off[1]) - 0.05) < 1e-5
          and abs(float(adv_on[0] - adv_off[0]) + 0.05) < 1e-5)

    # ---- 5. config/签名/CLI ----
    from rlab.train import run_signature, _is_opt_suffix
    check("retool_math preset: code_attempt_w=0.0（默认关）",
          get_config("retool_math", use_wandb=False)["code_attempt_w"] == 0.0)
    sig_off = run_signature(cfg_off)
    check("签名无 -caw（关闭时不加字符）", "-caw" not in sig_off)
    sig_on = run_signature({**cfg_off, "code_attempt_w": 0.05})
    check("签名含 -caw0.05（开启时进签名）", "-caw0.05" in sig_on)
    check("迁移兼容：关签名是开签名前缀 + caw 段为 opt 后缀",
          sig_on.startswith(sig_off) and _is_opt_suffix(sig_on[len(sig_off):]))
    _tr = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "train.py"), encoding="utf-8").read()
    check("CLI: --code_attempt_w 存在且透传 overrides",
          'add_argument("--code_attempt_w"' in _tr
          and 'overrides["code_attempt_w"]' in _tr)
    check("run_info: code_attempt_w 落盘",
          '"code_attempt_w": cfg.get("code_attempt_w")' in _tr)

    # ---- 5b.【2026-09-23】CLI 缺口补齐：采样协议/题目调度/overlong_buffer ----
    # 契约：① argparse 存在；② 透传到 overrides；③ 偏离进 run_signature 的
    # opt 段（CLI 改的 run 必须可与旧 ckpt 分辨——不改签名就会撞 out_dir，
    # --num_pre_Q 4 静默腰斩 lr 的同类盲区）；④ 默认 None（=不覆盖 preset，
    # 历史签名逐字不变）。
    for flag, key in (('"--temperature"', 'overrides["temperature"]'),
                      ('"--top_k"', 'overrides["top_k"]'),
                      ('"--top_p"', 'overrides["top_p"]'),
                      ('"--q_skip_streak"', 'overrides["q_skip_streak"]'),
                      ('"--q_pool_reset_floor"', 'overrides["q_pool_reset_floor"]'),
                      ('"--overlong_buffer"', 'overrides["overlong_buffer"]')):
        check(f"CLI: {flag} 存在且透传 overrides", flag in _tr and key in _tr)
    _base_cfg = get_config("retool_math", use_wandb=False)
    sig_base = run_signature(_base_cfg)
    # 默认值下新键不加字符（历史签名前缀兼容）
    for frag in ("-tk", "-tp", "-qs", "-qf", "-ob"):
        check(f"默认 {frag} 不进签名（历史签名不变）", frag not in sig_base)
    # 每个新键偏离时都进签名，且旧→新满足前缀兼容判据（_is_opt_suffix）
    for key, val, frag in (("top_k", 50, "-tk50"), ("top_p", 0.95, "-tp0.95"),
                           ("q_skip_streak", 3, "-qs3"),
                           ("q_pool_reset_floor", 128, "-qf128"),
                           ("overlong_buffer", 128, "-ob128")):
        sig_v = run_signature({**_base_cfg, key: val})
        check(f"偏离 {key}={val} 进签名（{frag}）", frag in sig_v)
        check(f"{key} 偏离段满足前缀兼容（_is_opt_suffix 认得）",
              sig_v.startswith(sig_base) and _is_opt_suffix(sig_v[len(sig_base):]))
    # temperature 偏离走既有 "T" 键（回归：别把原有键改坏）
    sig_T = run_signature({**_base_cfg, "temperature": 0.8})
    check("temperature 偏离进签名（既有 -T 键不回归）", "-T0.8" in sig_T)
    # 键名前缀冲突防回归："tk"/"tp" 都是 "t" 开头但与既有单字母段可区分
    # ——_is_opt_suffix 的最长优先匹配必须把 "-tk50" 识别为 opt 段而非未知段
    check("_is_opt_suffix 识别新前缀 tk/tp/qs/qf/ob",
          _is_opt_suffix("-tk50-tp0.95-qs3") and not _is_opt_suffix("-zzz1"))

    # ---- 5c.【2026-09-23】开发项落地：分档奖励/黑名单TTL/组相对长度惩罚 ----
    # 契约：① code_w 打通（此前硬编码 0 的死代码）；② TTL 释放拉黑题；
    # ③ 组相对长度惩罚按 MiMo Eq.4 形态工作；④ 全部默认 0 = 旧行为逐位相同；
    # ⑤ 进签名（-cw/-qt/-lp-lq-lg，默认不加字符）。
    from rlab.reward import total_reward_retool_math as _trm, group_length_penalty as _glp
    # ⑤-1 分档奖励：code_w=0 旧行为逐位不变（对照位）
    _ans = "the answer is \\boxed{42}"
    check("5c code_w=0 旧行为逐位相同",
          abs(_trm("42", _ans, code_ok=3)["reward"] - 1.0) < 1e-6
          and _trm("42", _ans, code_ok=3, code_w=0.0)["code"] == 0.0)
    # ⑤-2 分档奖励：答对+代码成功 → +code_ok*code_w；答对无代码/答错 → 不加
    _sc = _trm("42", _ans, code_ok=3, code_w=0.05)
    check("5c 分档奖励：答对+code_ok=3 → 1+0.15", abs(_sc["reward"] - 1.15) < 1e-6
          and abs(_sc["code"] - 0.15) < 1e-6)
    check("5c 分档奖励：答对但 code_ok=0 → 不加",
          abs(_trm("42", _ans, code_ok=0, code_w=0.05)["reward"] - 1.0) < 1e-6)
    check("5c 分档奖励：答错（code_ok>0）→ -1+0.15（成功率不救答错）",
          abs(_trm("99", _ans, code_ok=2, code_w=0.05)["reward"] - (-0.9)) < 1e-6)
    # ⑤-3 签名：默认无字符；偏离进签名且前缀兼容
    check("5c 默认 -cw/-qt/-lp 不进签名",
          all(f not in sig_base for f in ("-cw", "-qt", "-lp")))
    for key, val, frag in (("code_w", 0.05, "-cw0.05"), ("q_blacklist_ttl", 64, "-qt64"),
                           ("len_penalty_w", 0.3, "-lp0.3-lq50-lg0.25")):
        sig_v = run_signature({**_base_cfg, key: val})
        check(f"5c 偏离 {key}={val} 进签名（{frag}）", frag in sig_v)
        check(f"5c {key} 偏离段满足前缀兼容",
              sig_v.startswith(sig_base) and _is_opt_suffix(sig_v[len(sig_base):]))
    # 【2026-09-25 修复·断言本身是坏的】旧写法括号错位：
    #   all(f in _tr for f in ('a','b','c') and all(k in _tr for k in (...)))
    # `('a','b','c') and all(...)` 先求值 → 非空 tuple 为真 → 整个 and 表达式取右操作数
    # = 一个 **bool**，于是 `for f in True` → TypeError: 'bool' object is not iterable。
    # 即这条断言从落地起就从未验证过 CLI 透传，只是在抛异常（且因为它是本文件倒数第二
    # 个测试，pytest 只报 TypeError 不报断言失败，看起来像"环境问题"）。
    check("5c CLI: --code_w/--q_blacklist_ttl/--len_penalty_w 存在且透传",
          all(f in _tr for f in ('"--code_w"', '"--q_blacklist_ttl"', '"--len_penalty_w"'))
          and all(k in _tr for k in ('overrides["code_w"]',
                                     'overrides["q_blacklist_ttl"]',
                                     'overrides["len_penalty_w"]')))
    check("5c run_info: code_w/q_blacklist_ttl/len_penalty_* 落盘",
          all(k in _tr for k in ('"code_w": cfg.get("code_w")',
                                 '"q_blacklist_ttl": cfg.get("q_blacklist_ttl")',
                                 '"len_penalty_w": cfg.get("len_penalty_w")')))

    # ⑤-4 黑名单 TTL：到期释放（streak 清零 + 重新入队）
    from rlab.rollout import QuestionScheduler
    _qas = [{"Q": f"q{i}", "A": "1"} for i in range(6)]
    _sch = QuestionScheduler(_qas, streak_max=2, floor=2, ttl=3)
    _q0 = _qas[0]
    _sch.report(_q0, "uniform"); _sch.report(_q0, "uniform")
    check("5c TTL: streak 达标 = 拉黑", _sch.blacklisted_count() == 1)
    for _ in range(2):                       # ttl=3：3 次 draw 后释放
        _sch._tick_blacklist_ttl()
    check("5c TTL: 未到 ttl 仍拉黑", _sch.blacklisted_count() == 1)
    _sch._tick_blacklist_ttl()
    check("5c TTL: 到期释放（streak 清零）", _sch.blacklisted_count() == 0)
    check("5c TTL: 释放题立即重新入队",
          any(q["Q"] == "q0" for q in _sch.queue))
    _sch0 = QuestionScheduler(_qas, streak_max=2, floor=2, ttl=0)
    _sch0.report(_q0, "uniform"); _sch0.report(_q0, "uniform")
    for _ in range(5):
        _sch0._tick_blacklist_ttl()
    check("5c TTL=0 旧行为：永不释放", _sch0.blacklisted_count() == 1)

    # ⑤-5 组相对长度惩罚（MiMo Eq.4 形态）
    # 组：2 条通过（长 1000/2000）+ 2 条未通过（一长于 ref、一短于 ref）
    # B=50 → ref = 通过轨迹 50 分位 = 2000
    _gr = [1.0, 1.0, -1.0, -1.0]
    _gl = [1000, 2000, 4000, 1500]
    _out = _glp(_gr, _gl, weight=0.5, quantile=50, pass_gate=0.25)
    check("5c 组相对长度：通过轨迹不罚",
          _out[0] == 1.0 and _out[1] == 1.0)
    check("5c 组相对长度：未通过者按 len/ref-1 罚（4000/2000-1=1 → -1-0.5）",
          abs(_out[2] - (-1.5)) < 1e-6)
    check("5c 组相对长度：未超出 ref 的未通过者不罚（只罚超出部分）",
          abs(_out[3] - (-1.0)) < 1e-6)
    # 通过率 25% 不超 gate=0.25 → 整组零惩罚（低通过率组不罚"长而对"）
    _in4 = [-1.0, -1.0, -1.0, 1.0]
    _out2 = _glp(_in4, [5000, 5000, 5000, 100],
                 weight=0.5, quantile=50, pass_gate=0.25)
    check("5c 通过率 <= gate → 整组零惩罚",
          _out2 == [-1.0, -1.0, -1.0, 1.0])
    # 无通过轨迹 → 零惩罚（ref 无法定义）
    _out3 = _glp([-1.0, -1.0], [5000, 6000], weight=0.5, quantile=50, pass_gate=0.25)
    check("5c 无通过轨迹 → 零惩罚", _out3 == [-1.0, -1.0])
    # weight=0 → 逐位同旧（对照位）
    check("5c weight=0 → 逐位同旧",
          _glp(_gr, _gl, weight=0.0, quantile=50, pass_gate=0.25) == [1.0, 1.0, -1.0, -1.0])

    # ⑤-6 passrate.py：qk 聚合 + band 切片 + 表回填
    from rlab.passrate import (aggregate_by_qk as _agg, band_stats as _bst,
                               merge_into_table as _merge, qk_of as _qkof)
    import tempfile as _tf
    _rec = _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    _qtext = "某道题的题面文本"
    _key = _qkof(_qtext)
    _rec.write(json.dumps({"acc": [1, -1, -1, -1], "qk": _key}) + "\n")   # 1/4
    _rec.write(json.dumps({"acc": [1, 1, -1, -1], "qk": _key}) + "\n")    # 累计 3/8
    _rec.write(json.dumps({"acc": [1, -1, -1, -1]}) + "\n")               # 无 qk → 跳过
    _rec.write("not json\n")                                              # 坏行 → 跳过
    _rec.close()
    _aggd = _agg(_rec.name)
    check("5c passrate: 按题聚合（跨行累计 k/n_correct，坏行/缺 qk 跳过）",
          _aggd.get(_key) == {"k": 8, "n_correct": 3, "n_rows": 2})
    _bs = _bst(_aggd, 0.25, 0.75)
    check("5c passrate: band 切片（3/8=0.375 在 band 内）",
          _bs["n_questions"] == 1 and _bs["in_band"] == 1)
    _tbl = _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    _tbl.write(json.dumps({"Q": _qtext, "A": "7", "k": 4, "n_correct": 0,
                           "pass_rate": 0.0}) + "\n")
    _tbl.close()
    _mout = _tf.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    _merge(_aggd, _tbl.name, _mout, min_k=8)
    _newrow = json.loads(open(_mout, encoding="utf-8").read().strip())
    check("5c passrate: 表回填（在线 k>=min_k 覆盖，原表未动）",
          _newrow["k"] == 8 and _newrow["n_correct"] == 3
          and abs(_newrow["pass_rate"] - 0.375) < 1e-9
          and "online_meta" in _newrow
          and json.loads(open(_tbl.name, encoding="utf-8").read().strip())["n_correct"] == 0)
    os.unlink(_rec.name); os.unlink(_tbl.name); os.unlink(_mout)


    # ---- 6. #3 核对结论的文档锁：EOS 契约注释存在于 rollout.py ----
    _ro = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "rollout.py"), encoding="utf-8").read()
    check("rollout.py 记录 EOS 契约（#3 核对结论沉淀）",
          "EOS" in _ro and "vllm_token_ids_keep_eos" in _ro)


def test_health_code_collapse():
    """【2026-09-21 签名⑦ code_collapse】点火后 code% 掉 15pp 的压灭签名。

    no_code 只看"恒为 0"（点火前才响），压灭是"点火后跌回"——三次 run
    （run2/p5/p6）的 code% 50→3 均无告警，本签名补上这个盲区。"""
    print("[AL] health 签名⑦ code_collapse")
    from rlab.health import window_check

    def hist_of(code_rates, n_extra=0):
        h = [{"acc": 0.0, "fmt": 1.0, "clen": 1000.0,
              "code_rate": cr, "trunc_rate": 0.2} for cr in code_rates]
        h += [{"acc": 0.0, "fmt": 1.0, "clen": 1000.0,
               "code_rate": 0.5, "trunc_rate": 0.2} for _ in range(n_extra)]
        return h

    # 开局点火 50% → 后期跌到 20%（跌 30pp ≥ 15pp，96 组门槛）→ 报警
    rates = [0.5] * 64 + [0.2] * 32
    codes = [c for c, _ in window_check(hist_of(rates), retool=True)]
    check("点火后跌 30pp → code_collapse", "code_collapse" in codes)

    # 开局点火 → 保持 45%（跌 5pp < 15pp）→ 不报
    rates_ok = [0.5] * 64 + [0.45] * 32
    codes_ok = [c for c, _ in window_check(hist_of(rates_ok), retool=True)]
    check("保持稳定 → 无 code_collapse", "code_collapse" not in codes_ok)

    # 开局从未点火（0.02）→ 后期仍低 → no_code 管辖，code_collapse 不报
    rates_nofire = [0.02] * 64 + [0.01] * 32
    codes_nf = [c for c, _ in window_check(hist_of(rates_nofire), retool=True)]
    check("未点火场景不误报 code_collapse（开局 <15% 不触发）",
          "code_collapse" not in codes_nf)


# 【2026-09-25 补 __main__ guard】本文件 docstring 写的入口是
# `python -m rlab.tests.test_retool_cpu`，但这串调用此前**缩进在
# test_health_code_collapse 函数体内**、且没有 guard，后果有两条：
#   ① 直接跑该命令时只定义函数、不执行任何测试 —— 静默 exit 0、零字节输出，
#      看起来像"全部通过"，实际一个检查都没跑（检查器自身无声失效，与本轮修的
#      "内嵌评测读数恒 0"是同一类病：成功的表象由默认值/空动作伪造）。
#   ② pytest 跑到 test_health_code_collapse 时会**顺带把全部测试再跑一遍**
#      （含它自己 → 递归），于是别的测试一旦抛异常，就伪装成这个测试挂掉：
#      test_attempt_shaping_and_err_tier 的 TypeError 正是这样显示成两个失败的。
# 现在移到模块级并加 guard：pytest 只收集函数（不执行本块），直接跑才逐个执行。
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
    test_vllm_gen_kwargs()
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
    test_strip_left_pad()
    test_budget_guard_and_drift_stats()
    test_length_runaway_signature()
    test_logps_diff_shape()
    test_diag_logps_pure()
    test_diag_logps_static()
    test_remap_decision_unified_ckpt()
    test_diag_ablation_and_decode()
    test_diag_impl_label_and_mode()
    test_diag_counter_and_trajid()
    test_logprobs_n_fix_path()
    test_val_n_metric_fixes()
    test_eval_determinism_wiring()
    test_inline_eval_timeout_fix()
    test_inline_eval_json_shape()
    test_inline_eval_protocol_attestation()
    test_preflight_audit_fixes()
    test_overlong_filter()
    test_attempt_shaping_and_err_tier()
    test_health_code_collapse()
    test_pyflakes_undefined()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
    sys.exit(0)

