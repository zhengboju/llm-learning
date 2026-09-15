# -*- coding: utf-8 -*-
"""rlab/tests/test_gdn_backend_cpu.py — GDN 后端探针的判据自检（CPU）。

背景（2026-09-15 pod 训练端第一个 backward 崩）：
    RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect
    results for gated chunk_bwd_dqkwg (see #640)
fla 的护栏拒绝了坏 triton 组合。修复走 tilelang 后端（详见
`rlab/probe_gdn_backend.py` 模块 docstring）。真实对拍必须在 GPU 上跑，
本文件只测**判据本身**——判据错了，GPU 上的绿灯就是假绿灯。

覆盖（A/B/C 为判据正体，D 为反证）：
  A. 形状解析：复合 config 钻 text_config / 平铺 config / GVA 展开口径 /
     缺字段与不整除的报错路径。
  B. 误差度量：官方口径 err_ratio 的解析值（逐位相等 → 0、整体缩放 → 缩放量）。
  C. 判定闸门：全过 / 单项超阈值 / 缺失项 / NaN / inf 的判定与报告行。
  D. 【反证】注入已知错误（梯度置零、梯度整体缩放、元素错位）→ **判据必须爆**。
     抓不住注入错误的判据等于没有判据。

运行：python -m rlab.tests.test_gdn_backend_cpu
"""
import sys

import torch

from rlab.probe_gdn_backend import (OFFICIAL_RATIO, err_ratio,
                                    gdn_shape_from_config, judge)

PASS = []


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


def check_raises(name, fn, exc=ValueError):
    try:
        fn()
    except exc:
        PASS.append(name)
        print(f"  ok - {name}")
        return
    raise AssertionError(f"[FAIL] {name}（没有抛 {exc.__name__}）")


def main():
    print("[A] 形状解析（GDN 真实形状 → tilelang 编译 key 的一半）")
    composite = {"text_config": {"linear_num_key_heads": 16,
                                 "linear_num_value_heads": 32,
                                 "linear_key_head_dim": 128,
                                 "linear_value_head_dim": 128}}
    s = gdn_shape_from_config(composite, batch=1, seq_len=256)
    check("复合 config 钻 text_config", s["HV"] == 32 and s["K"] == 128)
    # GVA：transformers 在 Python 侧 repeat_interleave → 进 kernel 时 H == HV
    check("GVA 展开口径 H == HV == num_v_heads", s["H"] == 32 and s["gva"] == 2 and s["B"] == 1)
    # 反证：若不展开（H=16）则 H != HV，与真实调用不符 —— 这条防止"照抄 config 就交差"
    check("反证：H 不能取 num_k_heads（未展开的错口径）", s["H"] != 16)

    flat = {"linear_num_key_heads": 8, "linear_num_value_heads": 8,
            "linear_key_head_dim": 64, "linear_value_head_dim": 64}
    s2 = gdn_shape_from_config(flat, batch=4, seq_len=512)
    check("平铺 config（纯文本抽取版）", s2["H"] == 8 and s2["HV"] == 8 and s2["gva"] == 1
          and s2["B"] == 4 and s2["T"] == 512)

    check_raises("缺字段 → ValueError（不静默用默认值）",
                 lambda: gdn_shape_from_config({"text_config": {"linear_num_key_heads": 16}}, 1, 256))
    check_raises("num_v 不能被 num_k 整除 → ValueError",
                 lambda: gdn_shape_from_config({"linear_num_key_heads": 3,
                                                "linear_num_value_heads": 8,
                                                "linear_key_head_dim": 128,
                                                "linear_value_head_dim": 128}, 1, 256))

    print("\n[B] 误差度量（官方口径：相对 RMS）")
    x = torch.randn(64, dtype=torch.float32)
    check("逐位相等 → 0", err_ratio(x, x.clone()) == 0.0)
    # 整体缩放 k 倍：err = |k-1|·base → ratio = |k-1|（解析可验，判据的刻度感）
    for k in (1.01, 1.1, 2.0):
        r = err_ratio(x, x * k)
        check(f"整体缩放 {k} → ratio≈{abs(k - 1)}", abs(r - abs(k - 1)) < 1e-6)
    z = torch.zeros_like(x)
    check("全零 → ratio≈1（分母也退化，但不得为 NaN）", abs(err_ratio(x, z) - 1.0) < 1e-6)
    check("两边全零 → 有限值（+1e-8 护栏生效）", err_ratio(z, z) == 0.0)

    print("\n[C] 判定闸门")
    ok, lines = judge({k: OFFICIAL_RATIO[k] * 0.1 for k in OFFICIAL_RATIO})
    check("全部远低于阈值 → 通过", ok)
    check("报告行带实测值与倍率", all("/ 阈值" in ln and "x)" in ln for ln in lines))

    r_edge = dict.fromkeys(OFFICIAL_RATIO, 0.001)
    r_edge["dq"] = OFFICIAL_RATIO["dq"] * 1.017      # 刚好超线 1.7%：极易误判成"缺陷"
    ok_edge, lines_edge = judge(r_edge)
    check("单项刚好超线 1.7% → 判失败（不因差距小而放行）", not ok_edge)
    check("报告行标出该项 FAIL", any("FAIL" in ln and "dq" in ln for ln in lines_edge))

    ok_miss, _ = judge({k: 0.001 for k in OFFICIAL_RATIO if k != "dg"})
    check("缺失项（没算出来）→ 判失败", not ok_miss)

    ok_nan, lines_nan = judge({**{k: 0.001 for k in OFFICIAL_RATIO}, "dg": float("nan")})
    check("NaN → 判失败（不靠比较放行）", not ok_nan)
    check("NaN 被标出", any("FAIL" in ln and "dg" in ln for ln in lines_nan))
    ok_inf, _ = judge({**{k: 0.001 for k in OFFICIAL_RATIO}, "dk": float("inf")})
    check("inf → 判失败", not ok_inf)

    print("\n[D] 【反证】注入已知错误 → 判据必须爆")
    ref = torch.randn(128, dtype=torch.float32)
    base = {k: err_ratio(ref, ref.clone()) for k in OFFICIAL_RATIO}
    ok_clean, _ = judge(base)
    check("干净输入先确认判据是绿的（否则下面的'爆'不算数）", ok_clean)

    # ① 梯度整体置零 —— fla#640 类"该有梯度却算没了"的最强签名
    r_zero = dict(base, dq=err_ratio(ref, torch.zeros_like(ref)))
    ok_z, _ = judge(r_zero)
    check("① 梯度置零 → 判据爆", not ok_z)

    # ② 梯度整体缩放 5%（数值路径轻微错误，最像真实 miscompile 的形态）
    r_scale = dict(base, dg=err_ratio(ref, ref * 1.05))
    ok_s, _ = judge(r_scale)
    check("② 梯度缩放 5% → 判据爆（1.05 的 ratio=0.05 > 阈值 0.02）", not ok_s)

    # ③ 元素错位 —— 值域没变、能量没变，只有"位置"错了
    r_roll = dict(base, dk=err_ratio(ref, ref.roll(1, dims=0)))
    ok_r, _ = judge(r_roll)
    check("③ 元素错位 → 判据爆（错位不改变范数，只靠相对 RMS 抓）", not ok_r)

    # ④ 反证的边界：缩放 0.1%（远小于所有阈值）**不应**爆——
    #    否则说明判据过敏，训练里会把 bf16 噪声当故障报
    r_tiny = dict(base, dv=err_ratio(ref, ref * 1.001))
    ok_t, _ = judge(r_tiny)
    check("④ 反证边界：0.1% 缩放不该爆（判据不过敏）", ok_t)

    print("\n[E] 起跑前置判据（preflight_gdn.guard_risk）")
    from rlab.preflight_gdn import guard_risk
    check("Hopper + 坏 triton + 无 tilelang → 拦截（就是 09-15 那次）",
          guard_risk(is_hopper=True, triton_bad=True, tilelang_ok=False) is not None)
    check("Hopper + 坏 triton + tilelang 可用 → 放行（装包即修）",
          guard_risk(is_hopper=True, triton_bad=True, tilelang_ok=True) is None)
    # 反证三连：三个条件各拿掉一个，判定都必须翻转 —— 证明判据不是恒真/恒假
    check("反证①: triton 不在坏区间（≥3.7.1）→ 放行",
          guard_risk(is_hopper=True, triton_bad=False, tilelang_ok=False) is None)
    check("反证②: 非 Hopper（如 A100）→ 放行",
          guard_risk(is_hopper=False, triton_bad=True, tilelang_ok=False) is None)
    check("反证③: tilelang 可用但 triton 不在坏区间 → 仍放行（不因装了包就拦）",
          guard_risk(is_hopper=False, triton_bad=False, tilelang_ok=True) is None)
    msg = guard_risk(is_hopper=True, triton_bad=True, tilelang_ok=False)
    check("拦截消息点名了 chunk_bwd_dqkwg 与 tilelang（可行动）",
          "chunk_bwd_dqkwg" in msg and "tilelang" in msg)

    print(f"\n{len(PASS)} 项全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
