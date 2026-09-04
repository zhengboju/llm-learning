# -*- coding: utf-8 -*-
"""rlab/tests/test_smoke_cpu.py — 阶段0 验收：无 GPU 环境的数值与链路冒烟测试。

覆盖：
  A. losses.compute_advantages 三种模式的数学性质
  B. 六种 loss 的解析值核对（构造 ratio=1 与 ratio>clip 的可手算场景）+ backward 通畅
  C. protocol 双布局编解码 roundtrip
  D. reward 正则/超长惩罚/退化比对路径
  E. data fixture 加载

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

    # --- cispo: ratio<=1+hi 时目标为0；ratio>1+hi 时 = -clamp(ratio)*adv
    cfg = get_config("cispo", use_wandb=False)
    pol, gen, mask = _mk()          # ratio=1 -> keep=0 -> loss=0
    loss_z, _ = compute_loss("cispo", pol, gen, adv, mask, cfg)
    check("cispo ratio=1 处目标为0", math.isclose(loss_z.item(), 0.0, abs_tol=1e-6))
    pol_hi = (gen + 0.5).requires_grad_(True)      # ratio = e^0.5 ≈ 1.649 > 1.2
    loss_h, _ = compute_loss("cispo", pol_hi, gen, adv, mask, cfg)
    expect = -(torch.clamp(torch.exp(torch.tensor(0.5)), max=1.2) * adv).sum() / (4 * 8)
    check("cispo 截断保留 min(ratio,1+eps) 梯度", math.isclose(loss_h.item(), expect.item(), rel_tol=1e-4))

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
    cfg = get_config("dapo", use_wandb=False)
    check("preset 覆盖 + BASE 合并",
          cfg["clip_high"] == 0.28 and cfg["clip_low"] == 0.2 and cfg["lr"] == 1e-6)


if __name__ == "__main__":
    test_advantages()
    test_losses()
    test_protocol()
    test_reward_and_data()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
    sys.exit(0)
