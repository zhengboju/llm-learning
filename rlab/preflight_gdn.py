# -*- coding: utf-8 -*-
"""rlab/preflight_gdn.py — 起跑前置：GDN 反向护栏风险检查（Hopper × Triton 坏区间）。

【为什么需要】这个故障是**确定性**的：缺 tilelang 时，训练端必然在**第一个
backward** 崩（护栏 raise，见 docs/06-gdn-backend-tilelang.md）。而走到那一步已经
花掉十几分钟——ref_server 加载模型、vLLM 起引擎、首轮生成全部完成。本检查把它提前
到启动前几秒，并在消息里直接给出修法。

【判定用的是"能力"不是"形状"】不放"tilelang 装了没"这种代理指标，而是问 fla 自己
的 `TileLangBackend.is_available() and is_enabled()`——它才是 dispatch 真正会用的
判据（含 nvcc 可用性）。装了包但 nvcc 不可用、或 backend 被 `FLA_TILELANG=0` 关掉，
都会如实地判成"仍会撞护栏"。

【失败模式：判不了就不拦】环境不支持（无 CUDA / fla 没装 / 导入报错）时一律
exit 0 —— 那是别的故障域的事（例如 Qwen2.5-3B 的纯注意力模型根本不走 fla）。
**不静默**：判定依据会打印出来，`--explain` 可看完整推理。

用法（run_gsm8k.sh 的 Pre-flight 3 会自动调；也可手工跑）：
    CUDA_VISIBLE_DEVICES=1 python -m rlab.preflight_gdn
      exit 3 = 会撞护栏（缺 tilelang / nvcc 不可用 / backend 被关）；0 = 无风险或判不了

【逃生阀】`ALLOW_GDN_GUARD_RISK=1` 跳过拦截（例如明知护栏条件成立、本次只想跑
生成端做别的验证时）。默认不设 = 拦截。
"""
import argparse
import os
import sys

EXIT_RISK = 3           # 与 run_gsm8k.sh 约定的退出码


def guard_risk(*, is_hopper: bool, triton_bad: bool, tilelang_ok: bool) -> str | None:
    """纯函数（CPU 可测）：判定是否"会撞护栏"，返回风险消息或 None。

    Args:
        is_hopper: 当前设备是 Hopper（H20 = sm_90a）。
        triton_bad: triton ∈ [3.4.0, 3.7.1)（含下界，不含上界）。
        tilelang_ok: TileLangBackend 真能用（装了包 **且** nvcc 可用 **且** 未被关闭）
                     —— 这是 dispatch 的判据，不是"包存在"这种代理指标。
    """
    if not (is_hopper and triton_bad):
        return None                      # 坏组合不成立，本来就不会撞
    if tilelang_ok:
        return None                      # tilelang 会接管，护栏不会被触到
    return ("Hopper + Triton∈[3.4.0,3.7.1)：triton 对 gated chunk_bwd_dqkwg 会算错梯度，"
            "fla 的护栏会在训练第一个 backward raise。当前没有可用的 tilelang 后端")


def collect() -> tuple[dict, str | None]:
    """收集事实并判定。返回 (facts, risk)。任何"判不了"的情形都返回 risk=None。"""
    facts: dict = {}
    try:
        import importlib.util

        from fla.utils import (IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0,
                               TRITON_ABOVE_3_7_1)
        import torch
        facts["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "<无CUDA>"
        facts["is_hopper"] = IS_NVIDIA_HOPPER
        facts["triton_bad"] = bool(TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1)
    except Exception as exc:
        facts["skip"] = f"无法判定（{type(exc).__name__}: {exc}）"
        return facts, None

    tl_ok = False
    if facts["is_hopper"] and facts["triton_bad"]:
        facts["tilelang_installed"] = importlib.util.find_spec("tilelang") is not None
        # 判定只看 backend 自己的能力方法（它内部就含 nvcc 可用性检查）；
        # 误拦合法起跑是前置检查最坏的失败模式，所以任何异常都当作"判不了"而非"有风险"
        try:
            from fla.ops.common.backends.tilelang import TileLangBackend
            facts["tl_available"] = TileLangBackend.is_available()
            facts["tl_enabled"] = TileLangBackend.is_enabled()
            tl_ok = bool(facts["tl_available"] and facts["tl_enabled"])
        except Exception as exc:
            facts["tl_error"] = f"{type(exc).__name__}: {exc}"
        try:                                   # 仅用于展示，失败不影响判定
            from fla.utils import has_usable_nvcc
            facts["nvcc"] = has_usable_nvcc()
        except Exception:
            facts["nvcc"] = "<未知>"
    facts["tilelang_ok"] = tl_ok
    return facts, guard_risk(is_hopper=facts["is_hopper"], triton_bad=facts["triton_bad"],
                             tilelang_ok=tl_ok)


def main() -> int:
    ap = argparse.ArgumentParser(description="GDN 反向护栏起跑前置检查")
    ap.add_argument("--explain", action="store_true", help="打印判定依据（无论有无风险）")
    args = ap.parse_args()

    facts, risk = collect()
    if args.explain or risk or "skip" in facts:
        print(f"[preflight] GDN 护栏检查: {facts}")
    if risk:
        if os.environ.get("ALLOW_GDN_GUARD_RISK") == "1":
            print(f"[preflight] 风险存在但 ALLOW_GDN_GUARD_RISK=1，放行：{risk}")
            return 0
        print(f"[preflight] 拦截：{risk}")
        print("[preflight]   修法：pip install tilelang --no-deps（装完重启进程），"
              "详见 docs/06-gdn-backend-tilelang.md")
        return EXIT_RISK
    return 0


if __name__ == "__main__":
    sys.exit(main())
