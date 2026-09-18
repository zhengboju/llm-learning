# -*- coding: utf-8 -*-
"""rlab/tests/test_smoke_cpu.py — 阶段0 验收：无 GPU 环境的数值与链路冒烟测试。

覆盖：
  A. losses.compute_advantages 三种模式的数学性质
  B. 六种 loss 的解析值核对（构造 ratio=1 与 ratio>clip 的可手算场景）+ backward 通畅
  C. protocol 双布局编解码 roundtrip
  D. reward 正则/超长惩罚/退化比对路径
  E. data fixture 加载
  H. 抽取纯文本 checkpoint 的自检判据（替身模型，不依赖真权重）

运行：python -m rlab.tests.test_smoke_cpu
"""
import math
import sys

import torch

from rlab.config import ALGO_DEFAULTS, get_config
from rlab.data import load_qas
from rlab.losses import compute_advantages, compute_loss
from rlab.protocol import decode_batch, encode_batch
from rlab.reward import overlong_penalty, reward_correct, reward_format, total_reward

PASS = []


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


# ------------------------------------------------ A. advantages ----
def test_advantages():
    print("[A] compute_advantages")
    r = torch.tensor([3.0, 1.0, -2.0, 0.0, 5.0, 2.0])
    g = compute_advantages(r, group_size=2, mode="group_std")
    check("group_std 组内和为0", torch.allclose(g.view(3, 2).sum(1), torch.zeros(3), atol=1e-4))
    check("group_std 组内std≈1",
          torch.allclose(g.view(3, 2).std(dim=1, unbiased=True),
                         torch.ones(3), atol=1e-3))
    m = compute_advantages(r, group_size=2, mode="group_mean")
    check("group_mean = r - 组均值", torch.allclose(
        m, torch.tensor([1., -1., -1., 1., 1.5, -1.5]), atol=1e-4))
    gm = compute_advantages(r, group_size=2, mode="global_mean")
    check("global_mean = r - 全局均值", torch.allclose(
        gm, r - r.mean(), atol=1e-4))


# ------------------------------------------------------- B. loss ----
def _mk(B=4, T=8, seed=0):
    torch.manual_seed(seed)
    gen = torch.randn(B, T) - 1.0          # logps
    pol = gen.clone().requires_grad_(True)
    mask = torch.ones(B, T)
    return pol, gen.detach(), mask


def _ref(B=4, T=8, seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, T) - 1.0


def test_losses():
    print("[B] compute_loss 六算法")
    adv = torch.tensor([1.0, -0.5, 0.5, -1.0])
    ref = _ref()

    # --- grpo: ratio=1 -> loss = -mean(adv)（样本级 token-mean 后 batch 平均；β=0 去掉 KL 项）
    cfg = get_config("grpo", use_wandb=False, beta=0.0)
    pol, gen, mask = _mk()
    loss, st = compute_loss("grpo", pol, gen, adv, mask, cfg, ref_logps=ref)
    check("grpo 解析值", math.isclose(loss.item(), -adv.mean().item(), rel_tol=1e-5))
    loss.backward()
    check("grpo backward 有梯度", pol.grad is not None and pol.grad.abs().sum() > 0)

    # --- dapo: token_mean 与样本级在均匀 mask 下同值
    cfg = get_config("dapo", use_wandb=False, beta=0.0)
    pol, gen, mask = _mk()
    loss_d, _ = compute_loss("dapo", pol, gen, adv, mask, cfg, ref_logps=ref)
    check("dapo 均匀mask下 = grpo值", math.isclose(loss_d.item(), -adv.mean().item(), rel_tol=1e-5))
    # 非 uniform mask：token-level 与 sample-level 分离
    mask2 = torch.ones(4, 8); mask2[:2, 4:] = 0
    pol3, gen3, _ = _mk(seed=3)
    l_tok, _ = compute_loss("dapo", pol3, gen3, adv, mask2, cfg, ref_logps=ref)
    cfg_g = get_config("grpo", use_wandb=False)
    l_smp, _ = compute_loss("grpo", pol3, gen3, adv, mask2, cfg_g, ref_logps=ref)
    check("token_mean 与 sample_mean 在非均匀mask下分离", not math.isclose(l_tok.item(), l_smp.item(), rel_tol=1e-3))

    # --- dr_grpo: sum(-adv*mask)/(B*const)
    cfg = get_config("dr_grpo", use_wandb=False)
    pol, gen, mask = _mk()
    loss_c, _ = compute_loss("dr_grpo", pol, gen, adv, mask, cfg)
    expect = (-adv.unsqueeze(1).expand(4, 8).sum()) / (4 * cfg["max_gen_tokens"])
    check("dr_grpo 固定常数归一化", math.isclose(loss_c.item(), expect.item(), rel_tol=1e-5))

    # --- cispo: clip(ratio) 作 sg 权重，梯度经 logπ 流动（每个 token 有梯度）
    cfg = get_config("cispo", use_wandb=False)
    pol, gen, mask = _mk()          # ratio=1 -> w=1 -> loss = -A·logπ 的 token 均值
    loss_z, _ = compute_loss("cispo", pol, gen, adv, mask, cfg)
    expect_z = -(adv.unsqueeze(1) * pol.detach() * mask).sum() / mask.sum()
    check("cispo ratio=1 处 = -A·logπ token 均值", math.isclose(loss_z.item(), expect_z.item(), rel_tol=1e-5))
    loss_z.backward()
    check("cispo ratio=1 梯度 = -A/|e|（每 token 有梯度）",
          torch.allclose(pol.grad, -adv.unsqueeze(1).expand(4, 8) / 32, atol=1e-6))
    # 过界 token（ratio=e^0.5≈1.65>1.2）梯度仍非零——锁死 2026-09-07 梯度归零 bug
    pol_hi = (gen + 0.5).requires_grad_(True)
    loss_h, _ = compute_loss("cispo", pol_hi, gen, adv, mask, cfg)
    expect_h = -(adv.unsqueeze(1) * 1.2 * pol_hi.detach() * mask).sum() / mask.sum()
    check("cispo 过界 token 用 sg(clip(ratio)) 权重", math.isclose(loss_h.item(), expect_h.item(), rel_tol=1e-4))
    loss_h.backward()
    check("cispo 过界 token 梯度非零（修复锁死：旧实现 keep·clamp 梯度恒 0）",
          pol_hi.grad.abs().sum() > 0)

    # --- gspo: 序列级 ratio；ratio=1 时 = -mean(adv)
    cfg = get_config("gspo", use_wandb=False)
    pol, gen, mask = _mk()
    loss_g, _ = compute_loss("gspo", pol, gen, adv, mask, cfg)
    check("gspo ratio=1 处 = -mean(adv)", math.isclose(loss_g.item(), -adv.mean().item(), rel_tol=1e-5))
    # 序列级与 token 级的分离：同一序列内 token 扰动不影响 s（均值归一化后恒定）
    pol_v = (gen + torch.tensor([0.5, -0.5] * 4)).requires_grad_(True)  # 每行均值为0的扰动
    loss_v, _ = compute_loss("gspo", pol_v, gen, adv, mask, cfg)
    check("gspo 均值零扰动不改变序列ratio", math.isclose(loss_v.item(), loss_g.item(), rel_tol=1e-4))

    # --- rfpp: num_items 归一化 + per-token advantage
    cfg = get_config("rfpp", use_wandb=False)
    pol, gen, mask = _mk()
    adv_tok = torch.randn(4, 8)
    loss_r, _ = compute_loss("rfpp", pol, gen, adv_tok, mask, cfg, num_items_in_batch=100.0)
    expect = -(adv_tok).sum() / 100.0
    check("rfpp num_items 归一化", math.isclose(loss_r.item(), expect.item(), rel_tol=1e-5))

    # --- KL 项：beta>0 且 ref=policy 时 KL=0；ref≠policy 时 loss 变大（k3>=0）
    cfg = get_config("grpo", use_wandb=False)   # beta=0.04
    pol, gen, mask = _mk()
    l0, _ = compute_loss("grpo", pol, gen, adv, mask, cfg, ref_logps=pol.detach())
    check("KL: ref=policy 时等价 beta=0", math.isclose(l0.item(), -adv.mean().item(), rel_tol=1e-5))
    l1, _ = compute_loss("grpo", pol, gen, adv, mask, cfg, ref_logps=ref)
    check("KL: ref≠policy 时 loss 更大(k3>=0)", l1.item() > -adv.mean().item())

    # --- 统计量合法性
    check("stats 字段完整", all(k in st for k in ("clip_frac", "approx_kl", "mean_ratio")))


# -------------------------------------------------- C. protocol ----
def test_protocol():
    print("[C] protocol 编解码")
    meta = {"plen": 5, "algo": "grpo"}
    ids = torch.randint(0, 100, (2, 9))
    adv = torch.tensor([0.5, -0.5])
    gl = torch.randn(2, 4)
    acc = torch.tensor([1.0, -1.0]); fmt = torch.tensor([1.0, -1.0])
    raw = encode_batch(meta, ids, adv, gl, acc, fmt)
    # 模拟 passthrough ref_server：插 refs 到第3位
    from rlab.protocol import bytes_list_to_list, make_bytes_list, tensor_to_bytes
    dd = bytes_list_to_list(raw)
    refs = torch.randn(2, 4)
    out = make_bytes_list([dd[0], dd[1], dd[2], tensor_to_bytes(refs), dd[3], dd[4], dd[5]])
    d = decode_batch(out)
    check("passthrough roundtrip inputs", torch.equal(d["inputs"], ids))
    check("passthrough roundtrip adv", torch.equal(d["advantages"], adv))
    check("passthrough roundtrip refs", torch.equal(d["refs"], refs))
    check("passthrough roundtrip gen_logps", torch.equal(d["gen_logps"], gl))
    check("passthrough roundtrip acc/fmt",
          torch.equal(d["acc_scores"], acc) and torch.equal(d["format_scores"], fmt))
    # rfpp 布局
    meta_rf = {"plen": 5, "algo": "rfpp", "num_items_in_batch": 64}
    adv_tok = torch.randn(2, 4)
    raw_rf = encode_batch(meta_rf, ids, adv.view(2), gl, acc, fmt)  # part2=raw rewards
    dd = bytes_list_to_list(raw_rf)
    out_rf = make_bytes_list([dd[0], dd[1], dd[2], tensor_to_bytes(refs), dd[3],
                              tensor_to_bytes(adv_tok), dd[4], dd[5]])
    d2 = decode_batch(out_rf)
    check("rfpp 布局 advantages(B,T)", torch.equal(d2["advantages"], adv_tok))
    check("rfpp 布局 raw rewards 与 acc/fmt", torch.equal(d2["rewards"], adv.view(2))
          and torch.equal(d2["acc_scores"], acc) and torch.equal(d2["format_scores"], fmt))
    check("rfpp meta num_items 透传", d2["num_items_in_batch"] == 64)


# --------------------------------------------------- D/E. reward/data ----
def test_reward_and_data():
    print("[D] reward")
    good = "<think>abc</think><answer>\\boxed{42}</answer>"
    check("format 正例", reward_format(good) == 1.0)
    bad1 = "<think>abc<answer>42</answer>"          # 缺闭合
    bad2 = "<think>reasoning process here</think><answer>x</answer>"  # 抄模板
    bad3 = "<answer>42</answer>"                     # 缺 think
    check("format 缺闭合", reward_format(bad1) == -1.0)
    check("format 抄模板惩罚", reward_format(bad2) == -1.0)
    check("format 缺think", reward_format(bad3) == -1.0)
    check("format 双标签骗分拦截",
          reward_format("<think>a</think><answer><think>b</think>42</answer>") == -1.0)
    check("overlong 未触发", overlong_penalty(448, 512, 64) == 0.0)
    check("overlong 满扣", overlong_penalty(512, 512, 64) == 1.0)
    check("overlong 线性", math.isclose(overlong_penalty(480, 512, 64), 0.5))
    sc = total_reward("72", "<think>c</think><answer>72</answer>", w_acc=2.0)
    check("total_reward 全对 = 3.0", sc["reward"] == 3.0)
    sc2 = total_reward("10", "<think>c</think><answer>99</answer>", w_acc=2.0)
    check("total_reward 格式对答错 = -1.0 (2*(-1)+1)", sc2["reward"] == -1.0)
    # 数字提取退化路径（math_verify 不一定装在本机，两边都应可用）
    check("correct 纯文本", reward_correct("72", "answer is 72") == 1.0)

    print("[E] data fixture")
    qas = load_qas(fixture=True)
    check("fixture 数量", len(qas) == 32)
    check("fixture 字段", all(set(x) == {"Q", "A"} for x in qas[:2]))

    print("[F] config preset")
    check("dapo preset clip_higher", ALGO_DEFAULTS["dapo"]["clip_high"] == 0.28)
    check("rfpp preset beta=0", ALGO_DEFAULTS["rfpp"]["beta"] == 0.0)
    check("top_k=50 与 HF GenerationConfig 默认对齐（防 vLLM 全词表尾部，DAPO 缺口教训）",
          get_config("dapo", use_wandb=False)["top_k"] == 50)
    check("temperature=0.7 锁死（temp0.9 下 base 格式率仅 ~10%，格式信号被淹没会杀 "
          "dr_grpo/rfpp——2026-09-05 探针定案：0.9→10.4%/0.7→27.1%/0.6→37.5%）",
          get_config("dapo", use_wandb=False)["temperature"] == 0.7)
    cfg = get_config("dapo", use_wandb=False)
    check("preset 覆盖 + BASE 合并",
          cfg["clip_high"] == 0.28 and cfg["clip_low"] == 0.2 and cfg["lr"] == 1e-6)


# --------------------- H. 抽取纯文本 ckpt 的自检判据（替身模型） ----
def test_extract_selfcheck_judgement():
    """【2026-09-14】口径修正固化：旧判据把 0.1017 判成"抽取有误"。

    旧判据 `max|Δlogits| > 0.1` 的两处错：① 对拍两侧是两条**不同代码路径**——
    `full(...)` 走多模态复合 wrapper（自造 position_ids/attention_mask），
    `causal(...)` 是裸文本模型，同权重不同路径在 bf16 下逐层舍入，32 层混合线性注意力
    累积到 1e-1 属正常；② 阈值是绝对值、与 logits 量纲无关，实测只超线 1.7%，是阈值
    卡在噪声地板上的特征而非缺陷特征。

    新判据的主闸改成可判"对/错"的逐位相等（同类同权重同路径），本测试用替身模型钉死：
      ① 干净抽取必须放行（含反证控制生效+还原）；
      ② 错层复制必须被拦；
      ③ 只污染一个 bias 也必须被拦——**该场景 argmax 仍一致**，这正是不能拿 argmax/
         top-k 当主判据、必须用逐位相等的原因。
    """
    import contextlib
    import io
    from types import SimpleNamespace

    import torch.nn as nn

    from rlab.extract_text_model import _selfcheck

    print("[H] extract_text_model 自检判据（替身模型）")

    h, vocab, n_layer = 1024, 64, 3      # h 保证 block 权重 >1e6 元素，命中反证控制的挑选条件

    class _Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, h)
            self.layers = nn.ModuleList([nn.Linear(h, h) for _ in range(n_layer)])

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = torch.tanh(layer(x))
            return (x,)

    class _Full(nn.Module):
        """多模态复合模型替身：wrapper 路径故意与裸文本路径略有差异。"""

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = _Text()
            self.lm_head = nn.Linear(h, vocab, bias=False)

        def forward(self, input_ids):
            hidden = self.model.language_model(input_ids)[0]
            return SimpleNamespace(logits=self.lm_head(hidden) * 1.01)

    class _Causal(nn.Module):
        """抽取产物替身：同类、同权重、同路径。"""

        def __init__(self, full):
            super().__init__()
            self.model = _Text()
            self.model.load_state_dict(full.model.language_model.state_dict())
            self.lm_head = nn.Linear(h, vocab, bias=False)
            self.lm_head.load_state_dict(full.lm_head.state_dict())

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.lm_head(self.model(input_ids)[0]))

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    tok = SimpleNamespace(decode=lambda i: f"<{int(i)}>")

    def run(mutate=None):
        torch.manual_seed(0)                     # 两模型同源初始化，消除随机差
        full = _Full().to(torch.bfloat16)
        causal = _Causal(full).to(torch.bfloat16)
        if mutate is not None:
            mutate(causal)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                _selfcheck(full, full.model.language_model, causal, tok, ids)
                verdict = "pass"
            except RuntimeError as exc:
                verdict = "抽取有误" if "抽取有误" in str(exc) else str(exc)[:60]
        return verdict, buf.getvalue()

    v_clean, out_clean = run()
    v_wrong, _ = run(lambda c: c.model.layers[0].weight.data.zero_())
    v_bias, out_bias = run(lambda c: c.model.layers[1].bias.data.add_(0.5))

    check("抽取自检：同权重同路径 → 放行", v_clean == "pass")
    check("抽取自检：反证控制自身有效（扰动真生效且还原）",
          "扰动生效=True" in out_clean and "还原=True" in out_clean
          and "还原后 max|Δh|=0.000e+00" in out_clean)
    check("抽取自检：错层复制 → 判『抽取有误』", v_wrong == "抽取有误")
    check("抽取自检：只污染一个 bias（③ 的 argmax 仍一致）→ 主闸仍拦得住",
          v_bias == "抽取有误" and "argmax 一致=True" in out_bias)


# ------------------------------- G. eval 统计口径 + run 偏离签名 ----
def test_eval_stats_and_signature():
    """【2026-09-12】把 run2 暴露的两个测量缺口钉成回归测试：
    ①analysis 旧版用固定 ±2pp 地板判"真差异"，dapo_math N=500 下会把 m200−BASE 的
      +5.0pp（未配对 p≈0.11）误判成真差异；
    ②eval 只存聚合值 → 同题配对的 McNemar 永远算不出来（功效白丢，事后无法补救）。
    以及 run 偏离签名：让"这轮和参考差了哪几维"成为可 grep 的事实而非考古结论。"""
    import json as _json
    import os
    import tempfile

    from rlab.analysis import (ci95, diff_ci95, mcnemar_exact, paired_counts,
                               summarize_eval, pair_eval, _verdict)
    from rlab.train import run_signature, write_run_info

    print("[G] eval 统计口径（N-aware CI + McNemar）")
    check("ci95(0.5,500)≈±4.4pp（N=500 单臂）", abs(ci95(0.5, 500) - 4.38) < 0.05)
    check("ci95 n=0 → nan（不假装有精度）", ci95(0.5, 0) != ci95(0.5, 0))
    d, h = diff_ci95(0.516, 500, 0.466, 500)
    check("两臂差 = +5.0pp / ±6.2pp", abs(d - 5.0) < 0.05 and abs(h - 6.2) < 0.1)
    check("+5.0±6.2pp → 噪声内（旧 ±2pp 地板会误判真差异）", _verdict(d, h) == "噪声内")
    d2, h2 = diff_ci95(0.678, 500, 0.530, 500)
    check("fmt +14.8±6.0pp → 显著", _verdict(d2, h2) == "显著")
    check("有 McNemar p 时以 p 为准（配对功效更高）", _verdict(d, h, p=0.01) == "显著")
    check("mcnemar b=c=0 → p=1.0", mcnemar_exact(0, 0) == 1.0)
    check("mcnemar(20,5)≈0.004 <0.05 且两个方向对称",
          mcnemar_exact(20, 5) < 0.05 and mcnemar_exact(20, 5) == mcnemar_exact(5, 20))
    _m = [{"qk": "a", "acc": 1.0}, {"qk": "b", "acc": 0.0},
          {"qk": "c", "acc": 1.0}, {"qk": "d", "acc": 0.0}]
    _b = [{"qk": "a", "acc": 0.0}, {"qk": "b", "acc": 1.0},
          {"qk": "c", "acc": 1.0}, {"qk": "d", "acc": 0.0}]
    check("paired_counts: b=model对&base错 / c=model错&base对 / 只数分歧对",
          paired_counts(_m, _b) == (1, 1, 4))
    check("paired_counts 缺 items → None（回落两比例，不静默当成 0）",
          paired_counts(None, _b) is None and paired_counts(_m, []) is None)

    print("[G] run 偏离签名（配置自证）")
    cfg = get_config("retool_math", use_wandb=False, trunc_shaping=0.0,
                     all_steps=300, save_steps=50, lr=5e-6)
    sig = run_signature(cfg)
    check("签名含 algo/trunc/轮次/步数存盘/lr",
          sig.startswith("retool_math-ts0-ol1-r2x3072-s300x50-lr5e-06"))
    check("trunc_shaping 变化会改变签名（P1 消融可直接比对）",
          run_signature({**cfg, "trunc_shaping": 0.5}) != sig)
    check("lr=0.0 不被 falsy 吞掉（显式 0 仍进签名）", "-lr0-" in run_signature({**cfg, "lr": 0.0}))
    _dir = tempfile.mkdtemp()
    _p = os.path.join(_dir, "run_info.json")
    write_run_info(_p, {**cfg, "run_signature": sig})
    info = _json.load(open(_p, encoding="utf-8"))
    check("run_info 顶层带 signature + 消融维度（trunc/rounds/save_steps）",
          info["signature"] == sig and info["trunc_shaping"] == 0.0
          and info["max_rounds"] == 2 and info["save_steps"] == 50)
    check("run_info 仍保留完整 cfg（2026-09-11 provenance 契约不破）", len(info["config"]) > 50)

    # 【2026-09-17】系统提示偏离进签名：提示是协议的一半（难度表 = 模型×提示×预算），
    # 不进签名就会把"新提示下探的表"配"旧提示的 run"当同一配方对照。
    print("[G] 系统提示偏离进签名（-sp<hash6>）")
    from rlab.config import default_system_prompt
    check("default_system_prompt: retool_math/retool 各有 preset，其余取 BASE",
          "code_interpreter" in default_system_prompt("retool_math")
          and "[TOOL RESULT]" in default_system_prompt("retool")
          and default_system_prompt("grpo") == get_config("grpo", use_wandb=False)["system_prompt"])
    _sp_default = get_config("retool_math", use_wandb=False)
    check("默认提示的签名不含 -sp（历史签名串逐字不变 → P1b 对照口径与 ckpt 护栏不破）",
          "-sp" not in run_signature(_sp_default))
    _sp_new = default_system_prompt("retool_math") + "\nBe concise."
    _sig_sp = run_signature({**_sp_default, "system_prompt": _sp_new})
    check("提示偏离 → 追加 -sp<hash6>", "-sp" in _sig_sp and len(_sig_sp.split("-sp")[1]) == 6)
    check("提示指纹对内容敏感（差一个字符即变）",
          run_signature({**_sp_default, "system_prompt": _sp_new + " "}) != _sig_sp)
    # 【2026-09-17】难度表也要指纹：同 band 换表（半表→全表→换提示重探）会换训练池，
    # 只记 `d0-1` 会让两次不同的数据实验在不同 out_dir 里长得一模一样。
    _dt = os.path.join(_dir, "diff_table.jsonl")
    with open(_dt, "w", encoding="utf-8") as f:
        f.write('{"Q": "q1", "n_correct": 3, "k": 8}\n')
    _sig_t1 = run_signature({**_sp_default, "difficulty_path": _dt})
    _tag = _sig_t1.split("-d0-1-t")[-1][:6] if "-d0-1-t" in _sig_t1 else ""
    check("难度表指纹进签名（d<band>-t<hash6>），且仍含 band 供人眼辨别",
          len(_tag) == 6)
    with open(_dt, "a", encoding="utf-8") as f:      # 表内容变（如补齐到全表）
        f.write('{"Q": "q2", "n_correct": 0, "k": 8}\n')
    check("表内容变 → 签名变（半表 run 与全表 run 不再同签名）",
          run_signature({**_sp_default, "difficulty_path": _dt}) != _sig_t1)
    check("表路径不存在 → 指纹 NA（签名必须永远能打印，不抛）",
          "-tNA" in run_signature({**_sp_default, "difficulty_path": _dt + ".nope"}))
    check("无表（nodiff）不带表指纹（注意 ts0.5 里就含 -t，只能查 d<band>-t 形态）",
          "-nodiff-" in run_signature(_sp_default)
          and "d0-1-t" not in run_signature(_sp_default))
    check("两个 -sp 标签不影响其它字段（前缀仍逐字一致）",
          _sig_sp.startswith(run_signature(_sp_default)))
    # 候选提示文件：只在显式 --system_prompt_file 时生效，默认档零影响
    _spf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "prompts", "retool_math_concise.txt")
    check("候选提示文件存在且保留协议要件（围栏范例/工具标记/boxed 收尾）",
          os.path.exists(_spf) and all(
              s in open(_spf, encoding="utf-8").read()
              for s in ("```python", "[TOOL RESULT]", "\\boxed{<your final answer>}")))
    check("候选提示只增不删（行数变多且三处新增语句都在）",
          len(open(_spf, encoding="utf-8").read().strip().splitlines()) >
          len(default_system_prompt("retool_math").splitlines())
          and all(s in open(_spf, encoding="utf-8").read() for s in
                  ("Be concise", "no comments", "As soon as you have the answer")))

    print("[G] summarize_eval 表格口径")
    _j = os.path.join(_dir, "eval.json")
    _json.dump({"BASE": {"acc": 0.466, "fmt": 0.53, "both": 0.466, "n": 500},
                "m200": {"acc": 0.516, "fmt": 0.602, "both": 0.516, "n": 500},
                "_meta": {"base_path": "/root/Qwen3.5-4B"}},
               open(_j, "w", encoding="utf-8"), ensure_ascii=False)
    tbl = summarize_eval(_j)
    check("汇总表不再出现旧『真差异』固定地板判定", "真差异" not in tbl)
    check("汇总表给出 N-aware CI 与样本量", "±4.4pp" in tbl and "N=500" in tbl)
    check("_meta 不被当成模型行（审计字段与模型行分流）", "base_path" not in tbl)
    check("无 per-item 时明确标注是两比例检验（不冒充满配检验）",
          "两比例（无 per-item）" in tbl)

    print("[G] 代码分层分析（code_layer / summarize_code_layer）")
    # 【2026-09-18】p5 的 step300 增益被抹平 + record code_ok 56%→10% 崩——需区分
    # H1「理性压灭」（用码样本 acc 不高 ⇒ 工具路径无优势，无解）vs H2「激励不足」
    # （用码样本 acc 更高 ⇒ 可救）。数据源 = eval per-item 的 code_used，零训练成本。
    from rlab.analysis import code_layer, summarize_code_layer
    _cl_items = [
        {"qk": "q1", "acc": 1.0, "code_used": 1}, {"qk": "q2", "acc": 0.0, "code_used": 1},
        {"qk": "q3", "acc": 1.0, "code_used": 0}, {"qk": "q4", "acc": 0.0, "code_used": 0},
        {"qk": "q5", "acc": 1.0, "code_used": 0},
    ]
    _cl = code_layer(_cl_items)
    check("code_layer 分层 (n,acc) 正确（用码 2 题对 1 / 纯推理 3 题对 2）",
          _cl == ((2, 1), (3, 2)))
    check("code_layer 无 items → None（不把缺数据当 0）", code_layer([]) is None
          and code_layer(None) is None)
    _jcl = os.path.join(_dir, "eval_code_layer.json")
    _json.dump({
        "BASE": {"acc": 0.4, "n": 5, "items": _cl_items},
        "step200": {"acc": 0.6, "n": 5, "items": [
            {"qk": "q1", "acc": 1.0, "code_used": 1}, {"qk": "q2", "acc": 1.0, "code_used": 1},
            {"qk": "q3", "acc": 0.0, "code_used": 0}, {"qk": "q4", "acc": 0.0, "code_used": 0},
            {"qk": "q5", "acc": 1.0, "code_used": 0}]},
        "_meta": {"base_path": "/root/Qwen3.5-4B"}},
        open(_jcl, "w", encoding="utf-8"), ensure_ascii=False)
    _cltbl = summarize_code_layer(_jcl)
    check("分层表给出用码/纯推理两层的 n 与 acc", "用码题" in _cltbl
          and "纯推理题" in _cltbl and "50.0% (1/2)" in _cltbl)
    check("同层配对出现在表里（step200 vs BASE 的用码层）", "同题配对" in _cltbl
          and "p=" in _cltbl)

    print("[G] 分层迁移分析（code_migration）")
    # 【2026-09-18】--code-layer 只回答"每个存档点自己分层 acc"，回答不了
    # "BASE 用码的题到 step300 转纯推理后答得怎么样"——H1 判据的最后一块证据。
    from rlab.analysis import summarize_code_migration
    # BASE: q1/q2 用码（q1 对 q2 错），q3/q4/q5 纯推理（q3/q5 对 q4 错）
    # step200: q1/q2 仍用码（全对），q3/q4/q5 仍纯推理（q3/q5 对 q4 错），q6 新增
    _jcm = os.path.join(_dir, "eval_code_mig.json")
    _json.dump({
        "BASE": {"acc": 0.4, "n": 5, "items": [
            {"qk": "q1", "acc": 1.0, "code_used": 1}, {"qk": "q2", "acc": 0.0, "code_used": 1},
            {"qk": "q3", "acc": 1.0, "code_used": 0}, {"qk": "q4", "acc": 0.0, "code_used": 0},
            {"qk": "q5", "acc": 1.0, "code_used": 0}]},
        "step300": {"acc": 0.6, "n": 6, "items": [
            {"qk": "q1", "acc": 1.0, "code_used": 0},  # BASE 用码 → 转纯推理，对
            {"qk": "q2", "acc": 0.0, "code_used": 0},  # BASE 用码 → 转纯推理，错
            {"qk": "q3", "acc": 1.0, "code_used": 0},  # BASE 纯推理 → 仍纯推理
            {"qk": "q4", "acc": 0.0, "code_used": 0},
            {"qk": "q5", "acc": 1.0, "code_used": 0},
            {"qk": "q6", "acc": 1.0, "code_used": 0}]},
        "_meta": {}},
        open(_jcm, "w", encoding="utf-8"), ensure_ascii=False)
    _cm = summarize_code_migration(_jcm)
    check("迁移表以 BASE 分层为锚（两行层名 + 目标列）", "以 BASE 分层为锚" in _cm
          and "| 用码 |" in _cm and "| 纯推理 |" in _cm)
    check("BASE 用码题转纯推理格标 ◆（放弃代码的归宿）", "◆" in _cm)
    check("迁移计数正确：用码层 q1/q2 转纯推理(1对)；纯推理层 q3/q4/q5 仍同层(2对)",
          "50.0% (1/2) vs B 50.0% (1/2)" in _cm and "66.7% (2/3) vs B 66.7% (2/3)" in _cm)
    check("同题集上给出 target vs BASE 双 acc（放弃代码的代价可读）",
          "vs B" in _cm)

    print("[G] pair_eval 跨 json 配对（模型已灭失也可对照）")
    # 【2026-09-13】run2 的 m200 权重灭失（raw 被 P1 覆盖 + _mm 副本被删），
    # per-item json 是唯一遗物 —— 跨 json 同题配对让它仍能进对照表。
    _ja = os.path.join(_dir, "eval_new.json")
    _jb = os.path.join(_dir, "eval_old.json")
    _mk_items = lambda accs: [{"qk": f"q{i}", "acc": a} for i, a in enumerate(accs)]
    _json.dump({"p1s200": {"acc": 0.55, "n": 4, "items": _mk_items([1, 1, 0, 0])}},
               open(_ja, "w", encoding="utf-8"))
    _json.dump({"m200": {"acc": 0.50, "n": 4, "items": _mk_items([1, 0, 0, 1])}},
               open(_jb, "w", encoding="utf-8"))
    ptbl = pair_eval(_ja, _jb, "p1s200", "m200")
    check("跨 json 配对出 McNemar（b/c 来自两份 json 的 qk 交集）",
          "McNemar" in ptbl and "n=4" in ptbl)
    check("A/B 名字可以互换来源（主 json 找不到就去副 json 找）",
          "m200 ← eval_old.json" in ptbl and "p1s200 ← eval_new.json" in ptbl)
    missing = pair_eval(_ja, _jb, "p1s200", "nope")
    check("拼错模型名 → 列出可用名字（不静默空表）",
          "找不到" in missing and "m200" in missing)

    print("[G] summarize_record 多会话（record 追加写 = 常态）")
    # 【2026-09-17 真机】`"ABCDEFGH"[sess]` 在第 9 个会话 IndexError → `--record`
    # 整个不可用，而训练期曲线恰是"权重坏了 vs 存盘坏了"的唯一判别证据。
    from rlab.analysis import summarize_record, _sess_label
    check("会话标签：A..Z 之后不溢出", _sess_label(0) == "A" and _sess_label(7) == "H"
          and _sess_label(8) == "I" and _sess_label(30) == "S30")
    _rec = os.path.join(_dir, "record.jsonl")
    with open(_rec, "w", encoding="utf-8") as f:
        for s in range(12):                      # 12 个会话，跨过旧的 8 上限
            for _ in range(2):                   # 旧协议：无 gen_version → 时间判据
                f.write(_json.dumps({
                    "t": 1000.0 + s * 300.0, "algo": "retool_math",
                    "acc": [1.0] * 4 + [0.0] * 4, "fmt": [1.0] * 8,
                    "clen": [2000] * 8, "code_used": [1] * 8, "code_ok": [1] * 8,
                    "trunc_final": [0] * 8, "phase": "cold",
                }, ensure_ascii=False) + "\n")
    _rtbl = summarize_record(_rec)
    check("≥9 个会话不再 IndexError，且逐会话汇总齐全",
          "共 12 个会话" in _rtbl and "会话A(#0)" in _rtbl and "会话L(#11)" in _rtbl)
    check("多会话提示把『同签名重跑会覆盖 step_N』写进表头（防评到上一轮的 ckpt）",
          "run_info.json" in _rtbl)
    # 【2026-09-17 真机】纯时间判据把 bg1 的**一次** run 切成 31 个"会话"：真因是
    # `gen_questions_per_attempt=4` 让一次 attempt 的 4 条记录时间戳完全相同、attempt
    # 间隔 ~3min。新协议（有 gen_version）必须忽略这种 3 分钟级空档。
    check("曲线新增 trunc率/code_ok率 两列（崩坏形态：格式在、正确性死、长度掉）",
          "trunc率" in _rtbl and "code_ok率" in _rtbl)
    # 【2026-09-17】条件精度列 = "抽到 boxed 的轨迹里真做对的比例"——区分"学到数学"与
    # "学会收尾"的唯一干净指标（base 预算充足时 90%+，bg1 崩盘掉到 25~38%）。
    check("曲线新增条件精度列（acc 50% / fmt 100% → 50.0%）",
          "条件精度" in _rtbl and "| 50.0% | 100.0% | 50.0% |" in _rtbl)
    _rec_nofmt = os.path.join(_dir, "record_nofmt.jsonl")
    with open(_rec_nofmt, "w", encoding="utf-8") as f:
        for _ in range(4):                       # 格式死亡事件：fmt 恒 0 → 除法必须不崩
            f.write(_json.dumps({
                "t": 900.0, "acc": [0.0] * 8, "fmt": [0.0] * 8, "clen": [900] * 8,
                "code_used": [0] * 8, "code_ok": [0] * 8, "trunc_final": [0] * 8,
            }, ensure_ascii=False) + "\n")
    _rtbl_nf = summarize_record(_rec_nofmt)
    check("fmt 率 0 时条件精度列给 —（不 ZeroDivisionError）",
          "条件精度" in _rtbl_nf and "| 0.0% | 0.0% | — |" in _rtbl_nf)
    _rec_burst = os.path.join(_dir, "record_burst.jsonl")
    with open(_rec_burst, "w", encoding="utf-8") as f:
        for i in range(24):                      # 6 次 attempt × 4 条，间隔 180s>120s
            f.write(_json.dumps({
                "t": 2000.0 + (i // 4) * 180.0, "acc": [1.0] * 8, "fmt": [1.0] * 8,
                "clen": [1000] * 8, "code_used": [1] * 8, "code_ok": [1] * 8,
                "trunc_final": [0] * 8,
                "gen_version": 0 if i < 12 else 16, "phase": "cold",
            }, ensure_ascii=False) + "\n")
    _rtbl3 = summarize_record(_rec_burst)
    check("有 gen_version 时忽略 3min 级空档（burst 上传不再被切成假会话）",
          "共 1 个会话" in _rtbl3)
    _rec2 = os.path.join(_dir, "record_gv.jsonl")
    with open(_rec2, "w", encoding="utf-8") as f:
        for i, gv in enumerate([0, 16, 32, 0, 16, 32]):        # 中间一次回退 = 换 run
            f.write(_json.dumps({
                "t": 5000.0 + i * 5.0,                     # 间隔 5s：时间判据不切
                "acc": [1.0] * 8, "fmt": [1.0] * 8, "clen": [1000] * 8,
                "code_used": [1] * 8, "code_ok": [1] * 8, "trunc_final": [0] * 8,
                "gen_version": gv, "phase": "cold",
            }, ensure_ascii=False) + "\n")
    _rtbl2 = summarize_record(_rec2)
    check("gen_version 回退切出新会话（时间间隔不切时也能分开两次 run）",
          "共 2 个会话" in _rtbl2 and "gen_ver=0..32" in _rtbl2)
    # 【2026-09-18 staleness 列】off-policy 可观测：窗口内吃多旧的策略（opt-step）。
    # 单条 gv=0 的 record（样本 0..7，micro-step 0，floor(0/4)=0）→ staleness=0。
    _rec_st = os.path.join(_dir, "record_stale.jsonl")
    with open(_rec_st, "w", encoding="utf-8") as f:
        f.write(_json.dumps({
            "t": 9000.0, "acc": [1.0] * 8, "fmt": [1.0] * 8, "clen": [1000] * 8,
            "code_used": [1] * 8, "code_ok": [1] * 8, "trunc_final": [0] * 8,
            "gen_version": 0, "phase": "cold",
        }, ensure_ascii=False) + "\n")
        # gv=32：样本 8..15 = micro-step 1 → floor(1/4)=0 − floor(32/4)=8 = −8（超前）
        f.write(_json.dumps({
            "t": 9001.0, "acc": [0.0] * 8, "fmt": [1.0] * 8, "clen": [1000] * 8,
            "code_used": [1] * 8, "code_ok": [1] * 8, "trunc_final": [0] * 8,
            "gen_version": 32, "phase": "cold",
        }, ensure_ascii=False) + "\n")
    _rtbl_st = summarize_record(_rec_st, window=8)
    check("staleness 列存在且数值正确（gv=0 首条=0；gv=32 超前=-8）",
          "staleness" in _rtbl_st and "0.0（max 0）" in _rtbl_st
          and "-8.0（max -8）" in _rtbl_st)


if __name__ == "__main__":
    test_advantages()
    test_losses()
    test_protocol()
    test_reward_and_data()
    test_extract_selfcheck_judgement()
    test_eval_stats_and_signature()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
    sys.exit(0)
