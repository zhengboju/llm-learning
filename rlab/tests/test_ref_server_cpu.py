# -*- coding: utf-8 -*-
"""rlab/tests/test_ref_server_cpu.py — ref_server 数学部分的 CPU 单测。

覆盖（纯函数，无需 GPU/HTTP）：
  A. get_eos_mask：最后一个有效 token 定位
  B. passthrough_repack：字节布局 roundtrip（refs 插第 3 位，extras 依序保留）
  C. rfpp_process_macro 不变量：
     - num_items_in_batch = 有效 token 总数 / macro_step
     - pad 位 advantage = 0；benefit 符号正确（高 reward 组 eos 位 adv 更大）
     - 标准化后 macro batch 有效位 mean≈0, std≈1
     - KL 项进入反向 cumsum：非 eos 的有效 token 也获得非零信用（信用回传接线验证）
     - 输出 parts 能被 protocol.decode_batch(rfpp 布局) 正确解回

运行：python -m rlab.tests.test_ref_server_cpu
"""
import torch

from rlab.protocol import bytes_list_to_list, bytes_to_tensor, decode_batch, make_bytes_list
from rlab.ref_server import get_eos_mask, masked_mean_std, passthrough_repack, rfpp_process_macro

PASS = []


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


def _mk_item(plen, comp_len, T, reward, gen_const=0.0, ref_const=0.0, bsz=1, seed=0,
             algo="rfpp"):
    """构造一个 upload item：inputs (B, plen+T)，completion 前 comp_len 个有效。"""
    torch.manual_seed(seed)
    ids = torch.randint(10, 100, (bsz, plen + T))
    ids[:, plen + comp_len:] = 7          # pad_id=7
    return {
        "base": {"plen": plen, "algo": algo},
        "inputs": ids,
        "rewards": torch.tensor([reward] * bsz, dtype=torch.float32),
        "refs": torch.full((bsz, T), float(ref_const)),
        "gen_logps": torch.full((bsz, T), float(gen_const)),
        "acc": torch.tensor([1.0] * bsz),
        "fmt": torch.tensor([1.0] * bsz),
        # upload 处理器的产物：dd[3:] 依序为 gen_logps/acc/fmt（passthrough 路径需要）
        "extras": [torch.full((bsz, T), float(gen_const)),
                   torch.tensor([1.0] * bsz), torch.tensor([1.0] * bsz)],
    }


def test_eos_mask():
    print("[A] get_eos_mask")
    cmask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.int)
    eos = get_eos_mask(cmask)
    check("eos 定位：末有效位为1",
          eos.tolist() == [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    check("eos 每行恰好一个", (eos.sum(1) == 1).all().item())


def test_passthrough():
    print("[B] passthrough_repack")
    d = _mk_item(plen=3, comp_len=2, T=5, reward=2.0, algo="grpo")
    refs = torch.randn(1, 5)
    parts = passthrough_repack(d, refs)
    raw = make_bytes_list(parts)
    dd = bytes_list_to_list(raw)
    out = {"base": dd[0]}
    out["inputs"] = bytes_to_tensor(dd[1]); out["rewards"] = bytes_to_tensor(dd[2])
    out["refs"] = bytes_to_tensor(dd[3]); out["gen_logps"] = bytes_to_tensor(dd[4])
    out["acc_scores"] = bytes_to_tensor(dd[5]); out["format_scores"] = bytes_to_tensor(dd[6])
    check("passthrough 7 段布局", len(dd) == 7)
    check("refs 在第3位且数值一致", torch.equal(out["refs"], refs))
    check("inputs/rewards 透传", torch.equal(out["inputs"], d["inputs"])
          and torch.equal(out["rewards"], d["rewards"]))
    check("decode_batch 兼容（GRPO 布局）", end2 := decode_batch(raw))
    check("decode keys 齐全", {"inputs", "advantages", "refs", "gen_logps",
                              "acc_scores", "format_scores"} <= set(end2.keys()))


def test_rfpp_math():
    print("[C] rfpp_process_macro")
    pad = 7
    # 两个 micro item：不同 comp 长度、不同 reward（低 reward / 高 reward 各一半）
    lo = _mk_item(plen=2, comp_len=3, T=6, reward=-2.0, bsz=2, seed=1)
    hi = _mk_item(plen=2, comp_len=5, T=6, reward=2.0, bsz=2, seed=2)
    items = [lo, hi]

    outs = rfpp_process_macro(items, beta=0.0, pad_id=pad)   # beta=0：纯 reward 路径
    check("输出条数 = macro_step", len(outs) == 2)
    base0, parts0 = outs[0]
    valid0 = int((lo["inputs"][:, 2:] != pad).sum())
    valid1 = int((hi["inputs"][:, 2:] != pad).sum())
    check("num_items_in_batch = 有效token/macro_step",
          abs(base0["num_items_in_batch"] - (valid0 + valid1) / 2) < 1e-6)

    adv0 = bytes_to_tensor(parts0[5])
    adv1 = bytes_to_tensor(outs[1][1][5])
    check("pad 位 advantage = 0", (adv0[:, 3:] == 0).all().item() and (adv1[:, 5:] == 0).all().item())

    # 标准化不变量：拼接有效位 mean≈0, std≈1
    masks = [(lo["inputs"][:, 2:] != pad).int(), (hi["inputs"][:, 2:] != pad).int()]
    m, s = masked_mean_std([adv0, adv1], masks)
    check("标准化 mean≈0", abs(m) < 1e-4)
    check("标准化 std≈1", abs(s - 1) < 1e-3)
    # 符号：高 reward 组 eos 位 advantage > 低 reward 组 eos 位 advantage
    check("reward 高低 -> adv 符号正确",
          adv1[0, 4].item() > adv0[0, 2].item())

    # KL 接线：refs != gen_logps，常数差 -> 每个有效 token 都有 KL 项进入 cumsum
    lo2 = _mk_item(plen=2, comp_len=3, T=6, reward=-2.0, gen_const=0.0,
                   ref_const=-1.0, bsz=1, seed=3)   # kl = gen-ref = +1
    hi2 = _mk_item(plen=2, comp_len=5, T=6, reward=2.0, gen_const=0.0,
                   ref_const=-1.0, bsz=1, seed=4)
    outs2 = rfpp_process_macro([lo2, hi2], beta=0.04, pad_id=pad)
    adv_lo2 = bytes_to_tensor(outs2[0][1][5])
    mid_valid = adv_lo2[0, 1]      # 非 eos 的有效 token（comp_len=3 -> 有效位 0,1,2）
    check("KL 信用回传：非 eos 有效位获得非零 advantage", mid_valid.abs().item() > 1e-6)

    # decode 兼容：rfpp 布局
    raw = make_bytes_list(outs2[0][1])
    d3 = decode_batch(raw)
    check("rfpp decode advantages 形状=(B,T)", d3["advantages"].shape == adv_lo2.shape)
    check("rfpp decode acc/fmt", d3["acc_scores"].shape == (1,)
          and d3["format_scores"].shape == (1,))
    check("rfpp meta num_items 透传", "num_items_in_batch" in d3)


if __name__ == "__main__":
    test_eos_mask()
    test_passthrough()
    test_rfpp_math()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
