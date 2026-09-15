# -*- coding: utf-8 -*-
"""rlab/probe_gdn_backend.py — GDN 反向 kernel 后端探针（Hopper × Triton 坏区间护栏）。

【背景：2026-09-15 pod 训练端在第一个 backward 崩】

    RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect
    results for gated chunk_bwd_dqkwg (see #640). Please upgrade Triton to
    >= 3.7.1 or install tilelang: `pip install tilelang`
      @ fla/ops/common/chunk_o.py::chunk_bwd_dqkwg

这不是崩溃，是 **fla 的主动护栏**：Triton ∈ [3.4.0, 3.7.1) 在 Hopper（H20 = sm_90a）
上对 gated `chunk_bwd_dqkwg` 会算出**错误的梯度**（fla issue #640），fla 宁可 raise
也不给错梯度。Qwen3.5 的 GDN 层必然 gated（`g is not None`），所以训练端第一个
backward 必撞。

【三条出路】

  ① 升 Triton ≥ 3.7.1 —— 要动 torch 2.10 pin 住的 triton 版本（整条 CUDA 栈连带
     vLLM 的 triton kernel），风险最高。
  ② **装 tilelang**（本探针验证这条）—— fla 为此准备的备用后端，在 Hopper +
     Triton ≥ 3.4.0 下**默认启用**（`TileLangBackend.is_enabled`），dispatch 自动
     接管 `chunk_bwd_dqkwg`，护栏不再触发。**只影响这一个算子**：该后端只实现了
     chunk_bwd_dqkwg + parallel_attn_*（后者挂在 'attn' registry，我们的路径不走），
     其余 common 算子仍走 triton —— 前向与生成端的数值路径完全不变。
  ③ 什么都不做 —— 训练跑不起来。

【坑中坑：装了 tilelang ≠ 能用（2026-09-15 实机）】

`pip install tilelang --no-deps` 之后环境行**三绿**（已装=True、backend可用=True、
backend启用=True），Q1 却直接 OSError：

    File "fla/ops/common/backends/tilelang/chunk_bwd.py", line 8, in <module>
        import tilelang
    OSError: libz3.so.4.15: cannot open shared object file: No such file or directory

tilelang 的 libtvm 在 **dlopen 阶段**就要 libz3.so.4.15（TileLang 把 Z3 SMT 求解器
集成进整数分析 pass，用于 layout inference / bound analysis —— 是**编译期**依赖）。
而 fla 的 `is_available()` 是 `find_spec("tilelang")` 之类**代理判据**：它证明"包在
sys.path 上"，证不了"加载得起来"。前向不 import tilelang 所以一路正常，**第一次反向
dispatch 才炸** —— 代理判据给绿灯，训练照旧死在第一个 backward。

本探针的处置：`tilelang_verdict()`（复用 `rlab/preflight_gdn.py`，CPU 有测试）在代理
判据之外加一条**真 import**，与训练时那条加载路径同源；不通过就早退，不再把
"没撞护栏"误读成"tilelang 生效了"。修法见 docs/06 §3.2。

【本探针回答三个问题（每个都带反证对照）】

  Q1 tilelang 到底接管了没有、数值对不对？
     主进程跑 `chunk_gated_delta_rule` 前向+反向（逐参数复刻 transformers 的
     Qwen3.5 GDN 调用），与 fla 官方 naive 参考实现对拍。判据 = 官方口径
     `get_err_ratio`（相对 RMS 误差）+ `tests/ops/test_gdn.py` 的 ratio 阈值
     （o 0.005，dq/dk/dv 0.008，db/dg 0.02）。**同时打印实测值与倍率**——
     判据卡线时看数量级比看 pass/fail 有用。

  Q2 【反证】不装/不用 tilelang 时，同一调用是否必 raise？
     子进程里 FLA_TILELANG=0 复跑同一形状。**必须 raise 同一护栏**，否则说明
     当前环境根本没踩坑（比如 triton 已 ≥3.7.1），主进程的"通过"就不能记在
     tilelang 头上——这正是"判据必须能分辨两种假设"。

  Q3 9/13–9/14 那次 run2 的梯度是否被静默污染？
     fla 的护栏 commit 是 2026-09-11 才进的 main。若 run2 当时跑在"坏 triton +
     尚未带护栏的旧 fla"上，那次长跑的反向梯度就是错的。`--check-bad-triton`
     会**手工把护栏条件置否**（有意的错误注入）跑一次 triton 实现，量出它与参考
     的偏离幅度：若与 tilelang 同量级 → run2 可信；若显著偏离 → run2（乃至 P1）
     的结论都要重新审视。

【用法（pod，单卡空载；跑完顺带完成 tilelang JIT 预热）】

    # 先确认目标卡空闲（ref_server / 生成端会占卡）
    CUDA_VISIBLE_DEVICES=0 python rlab/probe_gdn_backend.py \
        --model_config /root/Qwen3.5-4B-text/config.json

    # 追查 run2 是否被污染（可选，多花约 1 分钟）
    CUDA_VISIBLE_DEVICES=0 python rlab/probe_gdn_backend.py \
        --model_config /root/Qwen3.5-4B-text/config.json --check-bad-triton

【为什么形状/参数要逐项复刻真实调用】

tilelang 的编译缓存 key 含 (B, H, HV, K, V, BT, BK, BV, NK, hD1, hD2, dtype,
USE_G, USE_DW, STATE_V_FIRST, IS_VARLEN)——T 是 dynamic 不进 key。形状或开关
复刻错了，预热就是白做：训练时仍要现场编译，而那个窗口正好叠在三方搬权重的
起跑期（撞宿主 RAM 上限的前科见 docs/04 与 [[rlab-startup-jit-ram-oomkill]]）。

逐项对照 transformers `modeling_qwen3_5.py` 的 GDN 层（本探针据此复刻）：
  · GVA 在 **Python 侧**已展开（`repeat_interleave`）→ 进 kernel 时 H == HV == 32
  · `g` 在 Python 侧算好（`-A_log.exp() * softplus(a + dt_bias)`，**fp32**），
    所以**不传** use_gate_in_kernel（传了反而偏离真实路径）
  · `beta` 是已 sigmoid 的值（不传 use_beta_sigmoid_in_kernel）
  · 传 use_qk_l2norm_in_kernel=True
  · 不传 cu_seqlens（训练侧逐行前向，IS_VARLEN=False）
"""
import argparse
import json
import os
import subprocess
import sys

# 允许 `python rlab/probe_gdn_backend.py` 直接跑（sys.path[0] 会变成 rlab/ 目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 官方判据阈值：fla tests/ops/test_gdn.py::test_chunk 的 assert_close ratio
OFFICIAL_RATIO = {"o": 0.005, "dq": 0.008, "dk": 0.008,
                  "dv": 0.008, "db": 0.02, "dg": 0.02}

# 护栏文案的特征串（判别"是不是这个坑"不依赖行号/版本）
GUARD_SIGNATURE = "produces incorrect results for"
GUARD_HINT = "see #640"

# 动态库加载失败的特征串（ld.so 的固定文案）。与护栏是**两种**假设：前者说"不让跑"，
# 后者说"根本跑不起来"。混成一句"跑不通"就等于没判据。
DLOPEN_SIGNATURE = "cannot open shared object file"


# ---------------------------------------------------------------- 纯函数（CPU 可测）

def gdn_shape_from_config(cfg: dict, batch: int, seq_len: int) -> dict:
    """从模型 config 取 GDN 层的真实形状。纯函数，本地 CPU 可测。

    Qwen3.5 是多模态复合 config：GDN 参数在 text_config 里；抽取出的纯文本
    checkpoint 则可能平铺在顶层。两种都接受。

    【H 为什么按 GVA 展开后算】transformers 的 GDN 层在 **Python 侧**就做了展开
    （`query.repeat_interleave(num_v_heads // num_k_heads)`），所以进 fla kernel 时
    H == HV == num_v_heads。复刻错这一点，预热 key 与真实调用就对不上。
    """
    node = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    n_k = int(node.get("linear_num_key_heads") or 0)
    n_v = int(node.get("linear_num_value_heads") or 0)
    d_k = int(node.get("linear_key_head_dim") or 0)
    d_v = int(node.get("linear_value_head_dim") or 0)
    if not all((n_k, n_v, d_k, d_v)):
        raise ValueError(
            "config 里没找到 GDN 形状字段（linear_num_key_heads / "
            "linear_num_value_heads / linear_key_head_dim / linear_value_head_dim）；"
            f"实际取到 n_k={n_k} n_v={n_v} d_k={d_k} d_v={d_v}。"
            "若模型不是 Qwen3.5 系，请用 --gdn_shape 手工指定")
    if n_v % n_k != 0:
        raise ValueError(f"linear_num_value_heads({n_v}) 不能被 linear_num_key_heads"
                         f"({n_k}) 整除 —— 与 Qwen3.5 的 GVA 约定不符，先核对 config")
    gva = n_v // n_k
    return {"B": batch, "T": seq_len, "H": n_k * gva, "HV": n_v,
            "K": d_k, "V": d_v, "gva": gva}


def err_ratio(ref, tri) -> float:
    """官方口径的相对 RMS 误差（fla/utils/_testing.py::get_err_ratio）。

    纯张量算术，CPU 可测。**不做 atol 短路**——探针要如实报出"差异真实存在、
    只是绝对值小"的情况，不能靠短路放行。
    """
    err = (ref.detach() - tri.detach()).flatten().square().mean().sqrt().item()
    base = ref.detach().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def judge(ratios: dict, thresholds: dict | None = None) -> tuple[bool, list[str]]:
    """按官方阈值判定，返回 (是否全过, 逐项报告行)。纯函数，CPU 可测。

    报告行带 **实测值 / 阈值 / 倍率** —— 卡线时这三个数才是可判读的
    （「刚好超线 1.7%」是阈值太紧的特征，不是缺陷的特征）。
    """
    thresholds = thresholds or OFFICIAL_RATIO
    lines, ok = [], True
    for name, thr in thresholds.items():
        if name not in ratios:
            lines.append(f"    FAIL {name:>4}: <缺失>（该项没算出来）")
            ok = False
            continue
        val = ratios[name]
        # 非有限值一律判失败：NaN/inf 是数值坏掉的最强签名，不能靠比较放行
        finite = val == val and abs(val) != float("inf")
        passed = finite and val < thr
        ok = ok and passed
        lines.append(f"    {'ok  ' if passed else 'FAIL'} {name:>4}: "
                     f"{val:.6f} / 阈值 {thr}  ({val / thr:.2f}x)")
    return ok, lines


# ---------------------------------------------------------------- 环境事实

def env_facts() -> dict:
    """收集环境事实。每项都设计成能分辨两种假设，避免含糊结论。"""
    import importlib.util

    import torch
    facts: dict = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import triton
        facts["triton"] = triton.__version__
    except Exception as exc:                        # pragma: no cover - 环境相关
        facts["triton"] = f"<import 失败: {exc}>"

    if torch.cuda.is_available():
        facts["device"] = torch.cuda.get_device_name(0)
        facts["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    else:
        facts["device"], facts["capability"] = "<无 CUDA>", "-"

    # 护栏旗标直接问 fla 要（比我们自己 parse 版本可靠；也不会随版本漂移）
    try:
        from fla.utils import (IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0,
                               TRITON_ABOVE_3_7_1, has_usable_nvcc)
        facts.update(is_hopper=IS_NVIDIA_HOPPER, triton_34=TRITON_ABOVE_3_4_0,
                     triton_371=TRITON_ABOVE_3_7_1, nvcc=has_usable_nvcc())
        # 护栏条件（省略 g is not None —— GDN 必然 gated）
        facts["guard_would_fire"] = bool(IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0
                                         and not TRITON_ABOVE_3_7_1)
    except Exception as exc:                        # pragma: no cover
        facts["fla_utils_error"] = f"{type(exc).__name__}: {exc}"

    # "装了 tilelang" 与 "backend 真能用" 是两回事，分开报。
    # 【代理层】find_spec + fla 的 is_available/is_enabled —— 只证明"包在"，证不了能 dlopen。
    facts["tilelang_installed"] = importlib.util.find_spec("tilelang") is not None
    try:
        from fla.ops.common.backends.tilelang import TileLangBackend
        facts["tl_available"] = TileLangBackend.is_available()
        facts["tl_enabled"] = TileLangBackend.is_enabled()
    except Exception as exc:                        # pragma: no cover
        facts["tl_available"] = facts["tl_enabled"] = False
        facts["tl_error"] = f"{type(exc).__name__}: {exc}"
    # 【能力层】真 import 一次 —— 2026-09-15 实机就是在这里翻的车：上面三个旗标全绿，
    # 但 tilelang 的 libtvm 在 dlopen 阶段要 libz3.so.4.15（TileLang 的编译期 SMT 依赖），
    # OSError 一路飘到训练第一个 backward 才炸。判据必须与真实加载路径同源。
    from rlab.preflight_gdn import probe_tilelang_import, tilelang_verdict
    facts["tl_import_error"] = (probe_tilelang_import()
                                if facts["tilelang_installed"] else None)
    facts["tl_ok"], facts["tl_why"] = tilelang_verdict(
        installed=facts["tilelang_installed"], import_error=facts["tl_import_error"],
        backend_available=facts["tl_available"], backend_enabled=facts["tl_enabled"],
        backend_error=facts.get("tl_error"))
    return facts


def print_env(f: dict) -> None:
    print("[probe] ===== 环境事实 =====")
    print(f"[probe] torch={f.get('torch')} cuda={f.get('cuda')} triton={f.get('triton')}")
    print(f"[probe] device={f.get('device')} capability={f.get('capability')}")
    if "fla_utils_error" in f:
        print(f"[probe] !! fla.utils import 失败：{f['fla_utils_error']}")
        return
    print(f"[probe] fla 护栏条件: hopper={f['is_hopper']} triton>=3.4.0={f['triton_34']} "
          f"triton>=3.7.1={f['triton_371']} → 会触发护栏={f['guard_would_fire']}")
    imp = f.get("tl_import_error")
    print(f"[probe] tilelang: 已装={f['tilelang_installed']} "
          f"真import={'通过' if (not imp and f['tilelang_installed']) else ('未跑' if not f['tilelang_installed'] else '失败')} "
          f"nvcc可用={f.get('nvcc')} backend可用={f['tl_available']} "
          f"backend启用={f['tl_enabled']}")
    if not f["guard_would_fire"]:
        print("[probe] → 护栏条件不成立：当前 triton 不在坏区间（或非 Hopper），"
              "本来就不会撞这个坑")
    elif imp:
        # 代理判据与真实能力打架：三绿齐亮但加载不起来。这是 2026-09-15 实机的形态，
        # 单列一段，避免被上面那行"backend可用=True"盖过去。
        print("[probe] !! 代理判据假阳性：find_spec / backend 旗标都说可用，"
              "但真 import 失败 ——")
        print(f"[probe]    {imp}")
        print("[probe]    装了包 ≠ 能用：包在 sys.path 上，证不了 dlopen 得起来。"
              "缺 .so 的修法见 docs/06 §3.2")
    elif not f["tilelang_installed"]:
        print("[probe] → 未装 tilelang：护栏会照常触发。先 `pip install tilelang`")
    elif not f.get("nvcc"):
        print("[probe] → 装了 tilelang 但 nvcc 不可用：backend is_available()=False，"
              "仍走 triton 撞护栏。检查 CUDA_HOME / nvcc 是否在 PATH")
    elif not f["tl_enabled"]:
        print("[probe] → backend 被显式关掉了（FLA_TILELANG=0？），不会接管")


# ---------------------------------------------------------------- 输入与两条路径

def build_inputs(shape: dict, dtype, seed: int, device):
    """构造与 transformers Qwen3.5 GDN 层**同形态**的输入（见模块 docstring）。

    g 用 fp32（transformers 在 Python 侧算的就是 fp32，fla kernel 也按 fp32 读）；
    β 是 post-sigmoid 的值。两者都带门控语义，不是可随便取的随机数。
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    B, T, H, HV, K, V = (shape[k] for k in ("B", "T", "H", "HV", "K", "V"))
    kw = {"device": device, "dtype": dtype}
    q = torch.randn(B, T, H, K, **kw)
    k = torch.randn(B, T, H, K, **kw)
    v = torch.randn(B, T, HV, V, **kw)
    beta = torch.rand(B, T, HV, device=device, dtype=dtype).sigmoid()
    # g = -exp(A_log) * softplus(a + dt_bias)：恒负，即对数域衰减（真实 GDN 门控形态）
    a = torch.randn(B, T, HV, device=device, dtype=torch.float32)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32)
    g = -A_log.exp() * F.softplus(a + dt_bias)
    return q, k, v, beta, g


def upstream_grad(shape: dict, dtype, seed: int, device):
    """固定的上游梯度 do。

    **必须两条路径共用同一张量**：早先版本用 `randn_like(o)` 分别生成，而 chunk
    出 bf16、naive 出 fp32，dtype 不同 ⇒ 随机流不同 ⇒ 两边比的是**不同目标函数的
    梯度**，判据就废了。这里显式以 fp32 生成，两侧统一 `.float()` 参与。
    """
    import torch
    torch.manual_seed(seed + 9973)
    return torch.randn(shape["B"], shape["T"], shape["HV"], shape["V"],
                       device=device, dtype=torch.float32)


def run_chunk(shape, dtype, seed, device, do):
    """训练真实路径：fla chunk 实现（tilelang 应在此接管）。"""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    q, k, v, beta, g = build_inputs(shape, dtype, seed, device)
    for t in (q, k, v, beta, g):
        t.requires_grad_(True)
    o, _ = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,        # transformers 真实传参
    )
    (o.float() * do).sum().backward()
    return {"o": o, "dq": q.grad, "dk": k.grad, "dv": v.grad,
            "db": beta.grad, "dg": g.grad}


def run_naive(shape, dtype, seed, device, do):
    """官方参考路径：纯 torch 逐步递归（fp32 计算）。

    【l2norm 的梯度必须穿过】chunk 路径的归一化在 kernel 内、梯度经它回传到原始
    q/k；参考路径若把 `F.normalize(q)` 后的张量当成叶子，拿到的 dq 是"对归一化后
    的 q"的梯度，与 chunk 的 dq **不可比**。这里 q/k 保持叶子身份，normalize 留在
    计算图上，autograd 自动把梯度带回 —— 与官方 test_gdn.py 的做法一致。
    """
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule
    q, k, v, beta, g = build_inputs(shape, dtype, seed, device)
    for t in (q, k, v, beta, g):
        t.requires_grad_(True)
    o, _ = naive_recurrent_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1),
        v=v, beta=beta, g=g, scale=None,
        initial_state=None, output_final_state=False)
    (o.float() * do).sum().backward()
    return {"o": o, "dq": q.grad, "dk": k.grad, "dv": v.grad,
            "db": beta.grad, "dg": g.grad}


def compare(shape, dtype, seed, device) -> dict:
    """跑两条路径并按官方口径比对。返回 {name: ratio}。"""
    do = upstream_grad(shape, dtype, seed, device)
    tri = run_chunk(shape, dtype, seed, device, do)
    ref = run_naive(shape, dtype, seed, device, do)
    return {n: err_ratio(ref[n], tri[n]) for n in OFFICIAL_RATIO}


# ---------------------------------------------------------------- 子进程反证

def _child_main(args) -> None:
    """子进程：父进程已用 FLA_TILELANG=0 关掉 tilelang；按 --child 决定是否绕过护栏。

    输出一行 @@JSON@@ 供父进程解析；面向人的信息走 stderr，不污染协议行。
    """
    out: dict = {"mode": args.child}
    try:
        import fla.ops.common.chunk_o as co          # 护栏宿主模块
        if args.child == "bad-triton":
            # 【有意的错误注入】把"triton 在坏区间"这一哨兵置否 → 护栏放行，
            # 从而量出坏实现本身的偏离幅度（Q3）。
            co.TRITON_ABOVE_3_4_0 = False
        shape = _resolve_shape(args)
        out["ratios"] = compare(shape, _torch_dtype(args.dtype), args.seed, "cuda")
        out["ok"] = True
    except RuntimeError as exc:
        out["ok"] = False
        out["guard_fired"] = GUARD_SIGNATURE in str(exc)
        out["error"] = str(exc).strip().splitlines()[0][:300]
    except Exception as exc:                         # pragma: no cover - 环境相关
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    print("@@JSON@@" + json.dumps(out, ensure_ascii=False))
    if not out.get("ok"):
        print(f"[child:{args.child}] {out.get('error')}", file=sys.stderr)


def _spawn_child(mode: str, args) -> dict:
    """跑反证子进程（tilelang 一律关掉：FLA_TILELANG=0）。"""
    env = dict(os.environ, FLA_TILELANG="0")
    cmd = [sys.executable, os.path.abspath(__file__), "--child", mode,
           "--model_config", args.model_config, "--batch", str(args.batch),
           "--seq_len", str(args.seq_len), "--dtype", args.dtype,
           "--seed", str(args.seed)]
    if args.gdn_shape:
        cmd += ["--gdn_shape", args.gdn_shape]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    return {"ok": False,
            "error": f"子进程无结果（exit={proc.returncode}）："
                     f"{(proc.stderr or proc.stdout or '')[-400:]}"}


# ---------------------------------------------------------------- 主流程

def _torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _resolve_shape(args) -> dict:
    if args.gdn_shape:
        h, hv, k, v = (int(x) for x in args.gdn_shape.split(","))
        if hv % h:
            raise ValueError(f"--gdn_shape 里 HV({hv}) 必须能被 H({h}) 整除")
        return {"B": args.batch, "T": args.seq_len, "H": h, "HV": hv,
                "K": k, "V": v, "gva": hv // h}
    with open(args.model_config, encoding="utf-8") as f:
        return gdn_shape_from_config(json.load(f), args.batch, args.seq_len)


def main():
    ap = argparse.ArgumentParser(
        description="GDN 反向 kernel 后端探针（Hopper × Triton 坏区间护栏）")
    ap.add_argument("--model_config", default="/root/Qwen3.5-4B-text/config.json",
                    help="模型 config.json（复合 ckpt 会自动钻 text_config 取 GDN 形状）")
    ap.add_argument("--gdn_shape", default=None,
                    help="手工指定 H,HV,K,V（绕开 config 读取；H 是 GVA 展开后的值）")
    ap.add_argument("--batch", type=int, default=1,
                    help="batch=1 对应 4B 训练的 micro_rows=1。"
                         "tilelang 编译 key 含 B，必须与训练一致才起预热作用")
    ap.add_argument("--seq_len", type=int, default=256,
                    help="T 是 dynamic、不进编译 key，任意值即可")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="默认 bfloat16 = 训练真实 dtype")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-bad-triton", action="store_true",
                    help="额外绕过护栏跑坏 triton 实现，量偏离幅度（Q3：run2 是否被污染）")
    ap.add_argument("--child", default=None, choices=["guard-check", "bad-triton"],
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        _child_main(args)
        return 0

    import torch
    facts = env_facts()
    print_env(facts)
    if not torch.cuda.is_available():
        print("\n[probe] 无 CUDA —— 本探针必须在 GPU 机器上跑。退出。")
        return 2
    # 先闸 fla 再闸 tilelang：两者缺失的处置完全不同，混在一起会把排查带偏
    if "fla_utils_error" in facts:
        print(f"\n[probe] 环境缺 fla（或 fla 版本不匹配）：{facts['fla_utils_error']}")
        print("[probe] 本坑的前提是「fla 在跑 GDN」——先确认 fla 装好且可 import")
        return 1
    # 闸门按**能力**不按形状：装了包但 import 不通过时，Q1 根本没有意义（chunk 路径
    # 必挂在同一个 dlopen 上），而且那会让"护栏没触发"被误读成"tilelang 生效了"。
    if not facts.get("tl_ok"):
        print(f"\n[probe] tilelang 后端不可用：{facts.get('tl_why')}")
        if facts.get("tl_import_error"):
            print("[probe]   装包这步已经做过了，问题在**依赖**：先 `python -c "
                  "\"import tilelang\"` 复现，缺 .so（libz3.so.4.15 一类）的修法见 "
                  "docs/06 §3.2")
        else:
            print("[probe]   先装好再跑本探针（装完必须重启进程：backend 可用性在 "
                  "import 期就定死了）")
        return 1

    shape = _resolve_shape(args)
    print(f"\n[probe] GDN 形状: B={shape['B']} T={shape['T']} H={shape['H']} "
          f"HV={shape['HV']} K={shape['K']} V={shape['V']} (GVA={shape['gva']})")

    # ---- Q1：tilelang 路径 vs 参考 ----
    print("\n[probe] ===== Q1：chunk 路径（应为 tilelang）vs naive 参考 =====")
    try:
        ratios = compare(shape, _torch_dtype(args.dtype), args.seed, "cuda")
    except Exception as exc:
        msg = str(exc)
        if GUARD_HINT in msg or GUARD_SIGNATURE in msg:
            print("[probe] chunk 路径撞上护栏 —— tilelang 没接管。")
            print("[probe] 对照上面 3 行 tilelang 状态定位原因；装/改完要"
                  "**重启进程**再跑（backend 可用性在 import 期定死）。")
            return 1
        # 加载期炸 vs 数值炸：两种假设必须分开报，否则"跑不通"会被笼统归到护栏上
        if isinstance(exc, OSError) or DLOPEN_SIGNATURE in msg:
            print("[probe] chunk 路径挂在**动态库加载**上（不是护栏、也不是数值问题）：")
            print(f"[probe]   {msg.strip().splitlines()[0]}")
            print("[probe]   装了 tilelang ≠ 能用；缺 .so 的修法见 docs/06 §3.2。")
            return 1
        raise
    ok, lines = judge(ratios)
    print("\n".join(lines))
    print(f"[probe] 数值判据: {'通过' if ok else '**不通过**'}（官方 test_gdn.py 阈值）")

    # ---- Q2：反证（关掉 tilelang 必须 raise）----
    print("\n[probe] ===== Q2 反证：FLA_TILELANG=0（禁用 tilelang）=====")
    guard = _spawn_child("guard-check", args)
    if guard.get("ok"):
        print("[probe] !! 反证失败：禁用 tilelang 后竟然也跑通了。")
        print("[probe]    说明当前环境根本没在坏区间（护栏条件不成立），"
              "上面的'通过'**不能**归功于 tilelang。看环境行的护栏条件。")
    elif guard.get("guard_fired"):
        print("[probe] 反证成立：禁用 tilelang 必 raise 同一护栏"
              " → 修复确由 tilelang 提供。")
    else:
        print(f"[probe] 反证异常（不是护栏，是别的错，需单独查）：{guard.get('error')}")

    # ---- Q3：坏 triton 的偏离幅度 ----
    if args.check_bad_triton:
        print("\n[probe] ===== Q3：绕过护栏跑坏 triton，量偏离幅度 =====")
        bad = _spawn_child("bad-triton", args)
        if not bad.get("ok"):
            print(f"[probe] 未能跑通（{bad.get('error')}）——不追。")
        else:
            br = bad["ratios"]
            print("[probe]  项   坏triton    tilelang       阈值")
            for n in OFFICIAL_RATIO:
                print(f"    {n:>4}: {br[n]:11.6f}  {ratios[n]:11.6f}  "
                      f"{OFFICIAL_RATIO[n]:>8}")
            worst = max(br.values())
            if worst > OFFICIAL_RATIO["dq"] * 10:
                print(f"[probe] 结论：坏 triton 的最大偏离 {worst:.4f} 远超噪声 —— "
                      "若 run2 跑在**无护栏的旧 fla + 同一 triton** 上，"
                      "那次梯度被静默污染，docs/05 的结论需重新审视。")
            else:
                print(f"[probe] 结论：坏 triton 最大偏离 {worst:.4f} 与噪声同量级 —— "
                      "run2 的梯度可信度不受此坑影响。")

    print("\n[probe] 完成。tilelang JIT 缓存已预热（训练首个 backward 免现场编译）。")
    return 0 if (ok and not guard.get("ok")) else 1


if __name__ == "__main__":
    sys.exit(main())
