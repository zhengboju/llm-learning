# -*- coding: utf-8 -*-
"""rlab/tests/test_extract_kernel_cpu.py — extract_text_model 的 kernel 分派自检（CPU）。

背景（2026-09-14 pod 实跑崩）：
    ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    at fla/ops/gated_delta_rule/chunk.py → l2norm_fwd
纯 CPU 的自检前向被塞进了 fla 的 triton kernel。根因是 transformers 的
`use_kernel_func_from_hub_with_fallback` **只在 import 期判断"原包装装没装"，
完全不看张量设备**——pod 装了 fla（vLLM GDN 的依赖）就必走 triton。
修复见 rlab/extract_text_model._force_torch_reference_kernels。

覆盖：
  A. 【反证】装了包（用临时目录伪造一个 fla）→ 装饰器必取外部包，且**与设备无关**
     （喂进去的就是 CPU 标量）。这一条证明"设备无关"这个根因判断成立，
     同时给 B 提供对照：B 若也走外部包，说明屏蔽失效。
  B. 【正】屏蔽后同一结构的装饰器取回纯 torch 参考实现，且 `import (fla)` 直接抛 ImportError。
  C. helper 返回值认得出"装了"——判据要分得清"本机没装"（没改变分派）与"装了被屏蔽"。
  D. 【静态守卫】helper 的调用必须排在 `import transformers.models.qwen3_5` 之前：
     分派发生在装饰器求值（模块 import 期），运行时再调已经晚了。

不在本测试覆盖（本地无 transformers 5.x）：真实 hub_kernels 装饰器本身——A/B 复刻的是
它 main 分支（hub_kernels.py:883-925）的分派结构，真实验证在 pod 实跑时完成。

运行：python -m rlab.tests.test_extract_kernel_cpu
"""
import importlib
import importlib.util
import os
import pathlib
import sys
import tempfile

from rlab.extract_text_model import _force_torch_reference_kernels

PASS = []


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


def _install_fake_fla() -> str:
    """造一个真的能被 import 的假 `fla` 包（假装 pod 上装了 fla）。"""
    root = tempfile.mkdtemp(prefix="fake_fla_")
    pkg = os.path.join(root, "fla")
    os.makedirs(pkg)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("def chunk_gated_delta_rule(*a, **k):\n    return 'FLA_TRITON'\n")
    sys.path.insert(0, root)
    importlib.invalidate_caches()      # 新目录要刷新，否则 find_spec 看不见
    return root


def _mirror_fallback_decorator(func_name: str, package: str):
    """复刻 transformers hub_kernels.use_kernel_func_from_hub_with_fallback 的分派结构。

    原结构（main 分支 883-925）：
        try:    implementation = importlib.import_module(package) 内的 func
        except: implementation = torch_function      # 参考实现
        wrapped = 调用 implementation
    这里额外在 wrapped 上挂 _is_new 供探针断言"到底走了哪条路"——原实现是闭包变量，
    外部看不见（这正是当初很难判断"是不是被 fla 接管了"的原因之一）。
    """
    def decorator(torch_function):
        implementation = None
        try:
            module = importlib.import_module(package)
            implementation = getattr(module, func_name, None)
        except Exception:
            implementation = torch_function
        finally:
            implementation = torch_function if implementation is None else implementation
        is_new = implementation is not torch_function

        def wrapped(*args, **kwargs):
            return implementation(*args, **kwargs)

        wrapped._is_new = is_new
        return wrapped

    return decorator


def _torch_reference(x):
    """被装饰的"参考实现"——真实现见 modeling_qwen3_5.py:301 torch_chunk_gated_delta_rule。"""
    return "TORCH_REFERENCE"


def test_dispatch_picks_installed_package():
    """A. 反证：装了包就必走外部包（设备无关）——补不出这个现象，后面的结论都不成立。"""
    print("[A] 反证：装了 fla → 装饰器取 fla（与设备无关）")
    _install_fake_fla()
    fn = _mirror_fallback_decorator("chunk_gated_delta_rule", "fla")(_torch_reference)
    out = fn(0)                       # 参数就是个 CPU 标量：模拟"纯 CPU 前向"
    check("装了 fla 时走 fla 实现", out == "FLA_TRITON" and fn._is_new)
    check("分派不看设备（CPU 参数照样被塞进外部实现）", out == "FLA_TRITON")


def test_blocked_dispatch_falls_back_to_torch():
    """B. 正：屏蔽后取回纯 torch 参考实现，且 import fla 直接失败。"""
    print("[B] 正：_force_torch_reference_kernels 后回落参考实现")
    r = _force_torch_reference_kernels()
    check("返回值认得出装了 fla", "fla" in r)
    try:
        importlib.import_module("fla")
        raised = False
    except ImportError:
        raised = True
    check("import fla 抛 ImportError（装饰器 except 分支才会命中）", raised)
    fn = _mirror_fallback_decorator("chunk_gated_delta_rule", "fla")(_torch_reference)
    check("屏蔽后走 torch 参考实现", fn(0) == "TORCH_REFERENCE" and not fn._is_new)


def test_criterion_distinguishes_installed_from_absent():
    """C. 判据分辨力：find_spec 必须在置 None **之前** 探测。

    置 None 之后再探测，装有/没装都返回 None → 日志会把"屏蔽了一个真包"和"本机啥也没装"
    说成同一句话，排查时看不出到底改没改变分派（"观测点要能分辨两种假设"）。
    """
    print("[C] 判据分辨力：探测时序")
    check("被屏蔽的包 find_spec 返回 None", importlib.util.find_spec("fla") is None)
    check("压根不存在的包 find_spec 也返回 None（⇒ 事后探测分不出这两者）",
          importlib.util.find_spec("no_such_pkg_xyz") is None)
    placeholder = sys.modules["fla"]           # 临时摘掉占位（假包仍在 sys.path 上）
    del sys.modules["fla"]
    try:
        check("摘掉占位后又能找到 → 与'不存在'可分",
              importlib.util.find_spec("fla") is not None)
    finally:
        sys.modules["fla"] = placeholder


def test_call_site_precedes_transformers_import():
    """D. 静态守卫：调用排在建模模块 import 之前（import 期语义决定的先决条件）。"""
    print("[D] 静态守卫：调用时序")
    src = pathlib.Path(__file__).resolve().parents[1] / "extract_text_model.py"
    lines = src.read_text(encoding="utf-8").splitlines()
    call_line = next(i for i, l in enumerate(lines)
                     if l == "    _force_torch_reference_kernels()")
    imp_line = next(i for i, l in enumerate(lines)
                    if l.startswith("    from transformers.models.qwen3_5 import"))
    main_line = next(i for i, l in enumerate(lines) if l.startswith("def main("))
    check("调用在 main() 内", main_line < call_line)
    check("屏蔽调用排在 import modeling 之前", call_line < imp_line)


if __name__ == "__main__":
    test_dispatch_picks_installed_package()       # 必须在屏蔽前跑（A 是 B 的对照）
    test_blocked_dispatch_falls_back_to_torch()
    test_criterion_distinguishes_installed_from_absent()
    test_call_site_precedes_transformers_import()
    print(f"\n全部通过：{len(PASS)} 项检查 ✅")
    sys.exit(0)
