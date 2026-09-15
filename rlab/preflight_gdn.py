# -*- coding: utf-8 -*-
"""rlab/preflight_gdn.py — 起跑前置：GDN 反向护栏风险检查（Hopper × Triton 坏区间）。

【为什么需要】这个故障是**确定性**的：缺 tilelang 时，训练端必然在**第一个
backward** 崩（护栏 raise，见 docs/06-gdn-backend-tilelang.md）。而走到那一步已经
花掉十几分钟——ref_server 加载模型、vLLM 起引擎、首轮生成全部完成。本检查把它提前
到启动前几秒，并在消息里直接给出修法。

【判定用的是"能力"不是"形状"】分两层，缺一层就是假绿灯：

  ① `TileLangBackend.is_available() and is_enabled()`——dispatch 真正会看的开关
     （含 nvcc 可用性、`FLA_TILELANG=0`）。装了包但 nvcc 不可用、或 backend 被关掉，
     都会如实判成"仍会撞护栏"。
  ② **真 `import tilelang` 一次**（2026-09-15 实机补的）。①是代理判据：它内部是
     `find_spec("tilelang")`，只证明"包在 sys.path 上"，**证不了 dlopen 得起来**。
     实机就撞了这个空子——三个旗标全绿，`import tilelang` 却 OSError
     （缺 libz3.so.4.15，TileLang 的编译期 SMT 依赖）；前向不 import tilelang 所以
     没事，**第一次反向 dispatch 才炸**，正是本检查想防的那个失败模式。见 docs/06 §3.2。

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


def tilelang_verdict(*, installed: bool, import_error: str | None,
                     backend_available: bool, backend_enabled: bool,
                     backend_error: str | None = None) -> tuple[bool, str]:
    """纯函数（CPU 可测）：判定 tilelang 后端是否**真能用**，返回 (可用, 理由)。

    【为什么不能只看 fla 的 is_available()】它内部是 `find_spec("tilelang")` + nvcc +
    设备能力——"包里有没有这个模块"是**代理指标**，管不了"模块能不能 dlopen"。
    2026-09-15 实机：装了 tilelang（find_spec=True、is_available=True、is_enabled=True），
    但 `import tilelang` 在 dlopen 阶段就 OSError（缺 libz3.so.4.15）→ 三绿齐亮、
    训练照旧死在第一个 backward。**判据要按能力（真跑一次同源的加载路径），不按形状。**

    Args:
        installed: `find_spec("tilelang") is not None`（代理指标，只作兜底报错用）。
        import_error: **真 import** 的报错串；None = import 成功。
        backend_available / backend_enabled: fla 的两个开关方法。
        backend_error: `import TileLangBackend` 本身的异常（用于分辨"没装"与"fla 坏了"）。
    """
    if not installed:
        return False, "未装 tilelang（pip install tilelang --no-deps，装完须重启进程）"
    if import_error is not None:
        return False, f"tilelang 装了但 import 失败：{import_error}"
    if backend_error is not None:
        return False, f"fla 的 tilelang 后端模块 import 失败：{backend_error}"
    if not backend_available:
        return False, "TileLangBackend.is_available()=False（nvcc 不可用 / 设备非 Hopper）"
    if not backend_enabled:
        return False, "TileLangBackend.is_enabled()=False（FLA_TILELANG=0？）"
    return True, "真 import 通过且 fla 会启用该后端"


def probe_tilelang_import() -> str | None:
    """真 import 一次 tilelang；返回 None = 成功，否则错误串。

    **与训练时那条加载路径同源**：fla 的 `chunk_bwd` 里就是 `import tilelang`。
    代价是几百 MB RSS / 数秒（TVM 加载），只在"Hopper × 坏 triton"这一支才付。
    """
    try:
        import tilelang  # noqa: F401
        return None
    except Exception as exc:                         # pragma: no cover - 环境相关
        return f"{type(exc).__name__}: {exc}"


def guard_risk(*, is_hopper: bool, triton_bad: bool, tilelang_ok: bool,
               note: str = "") -> str | None:
    """纯函数（CPU 可测）：判定是否"会撞护栏"，返回风险消息或 None。

    Args:
        is_hopper: 当前设备是 Hopper（H20 = sm_90a）。
        triton_bad: triton ∈ [3.4.0, 3.7.1)（含下界，不含上界）。
        tilelang_ok: tilelang 后端**真能用**（见 `tilelang_verdict`：装了包 **且** 真 import
                     通过 **且** nvcc 可用 **且** 未被关闭）——不是"包存在"这种代理指标。
        note: 具体原因（`tilelang_verdict` 的第二项）。失败信息里必须带一个能**分辨
              两种假设**的字段，否则"没装"与"装了但加载不起来"会混成同一句废话。
    """
    if not (is_hopper and triton_bad):
        return None                      # 坏组合不成立，本来就不会撞
    if tilelang_ok:
        return None                      # tilelang 会接管，护栏不会被触到
    msg = ("Hopper + Triton∈[3.4.0,3.7.1)：triton 对 gated chunk_bwd_dqkwg 会算错梯度，"
           "fla 的护栏会在训练第一个 backward raise。当前没有可用的 tilelang 后端")
    return f"{msg}（{note}）" if note else msg


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
        # 代理层：fla 自己的能力方法（含 nvcc 检查）。任何异常都当作"判不了"而非"有风险"，
        # 但它本身的结论也不能直接信——见下面真 import 那条。
        try:
            from fla.ops.common.backends.tilelang import TileLangBackend
            facts["tl_available"] = TileLangBackend.is_available()
            facts["tl_enabled"] = TileLangBackend.is_enabled()
        except Exception as exc:
            facts["tl_error"] = f"{type(exc).__name__}: {exc}"
            facts["tl_available"] = facts["tl_enabled"] = False
        # 能力层：真 import 一次（与 fla 内部那条加载路径同源）。代理层管不了 dlopen。
        facts["tl_import_error"] = (probe_tilelang_import()
                                    if facts["tilelang_installed"] else None)
        tl_ok, why = tilelang_verdict(installed=facts["tilelang_installed"],
                                      import_error=facts["tl_import_error"],
                                      backend_available=facts["tl_available"],
                                      backend_enabled=facts["tl_enabled"],
                                      backend_error=facts.get("tl_error"))
        facts["tilelang_why"] = why
        try:                                   # 仅用于展示，失败不影响判定
            from fla.utils import has_usable_nvcc
            facts["nvcc"] = has_usable_nvcc()
        except Exception:
            facts["nvcc"] = "<未知>"
    facts["tilelang_ok"] = tl_ok
    return facts, guard_risk(is_hopper=facts["is_hopper"], triton_bad=facts["triton_bad"],
                             tilelang_ok=tl_ok, note=facts.get("tilelang_why", ""))


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
        if facts.get("tl_import_error"):
            # 代理判据全绿但加载不起来（缺动态库），修法与"没装"完全不同，不能混着给
            print("[preflight]   `pip install tilelang` 已经做过了 —— 问题在**依赖**不在包：")
            print("[preflight]   先复现：python -c \"import tilelang\"；"
                  "缺 .so（如 libz3.so.4.15）的修法见 docs/06 §3.2")
        else:
            print("[preflight]   修法：pip install tilelang --no-deps（装完重启进程），"
                  "详见 docs/06-gdn-backend-tilelang.md")
        return EXIT_RISK
    return 0


if __name__ == "__main__":
    sys.exit(main())
