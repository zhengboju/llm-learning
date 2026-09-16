# GDN 反向 kernel 后端：Hopper × Triton 坏区间护栏（2026-09-15）

> 训练端在**第一个 backward** 崩：
>
> ```
> RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect
> results for gated chunk_bwd_dqkwg (see #640). Please upgrade Triton to
> >= 3.7.1 or install tilelang: `pip install tilelang`
>   @ fla/ops/common/chunk_o.py::chunk_bwd_dqkwg
> ```
>
> 配套：显存/资源类炸点见 `04-qwen35-4b-gpu-memory-oom.md`；本坑阻塞的 P1b
> 见 `05-retool-math-diagnosis-plan.md` §6。探针 `rlab/probe_gdn_backend.py`，
> CPU 判据 `rlab/tests/test_gdn_backend_cpu.py`。
> **这是本项目 GDN kernel 生态的第三个坑**（前两个：vLLM 侧 FlashInfer JIT 抢
> 宿主 RAM、CPU 前向被 fla 的 triton 实现接管），三者共同的一句话教训见 §7。

---

## 1. 这不是崩溃，是护栏

fla 在 `chunk_bwd_dqkwg` 里主动查条件并 raise：

```python
if g is not None and IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1:
    raise RuntimeError("... produces incorrect results ... see #640 ... or install tilelang")
```

即 **Triton ∈ [3.4.0, 3.7.1) 在 Hopper 上算 gated chunk_bwd_dqkwg 结果是错的**。
fla 的姿态是"宁可崩也不给错梯度"——这个坑的危险形式是被绕过（见 §5），不是崩溃。

四个条件对本项目全中：

| 条件 | 实际值 | 依据 |
|---|---|---|
| `g is not None` | True | Qwen3.5 是 GDN（门控 delta 规则），必然带 g |
| `IS_NVIDIA_HOPPER` | True | H20 = sm_90a |
| `TRITON_ABOVE_3_4_0` | True | **护栏触发本身即证据**：这四个条件与判据同源，它 raise 了就说明都成立 |
| `not TRITON_ABOVE_3_7_1` | True | 同上（确切版本号不必猜，`probe_gdn_backend.py` 会打印） |

前向不受影响（护栏只在反向）：训练端 loss 能算出来，死在第 1 步的 `engine.backward`。

## 2. 为什么 9/13–9/14 的长跑没崩（**待核实的疑点**）

fla 的护栏 commit `516143e` 是 **2026-09-11** 才进 main 的。而 run2（09-13）与 P1
（09-13）都完整跑完（719/1000 步、300/300 步）——**说明当时的环境里这个护栏没有
生效**。只有两种可能：

| 分支 | 含义 | run2 梯度可信度 |
|---|---|---|
| (a) 当时 triton 不在坏区间 | 例如 triton < 3.4.0 | **可信**（坏区间外，结果正确） |
| (b) 当时 fla 是 09-11 之前的版本（无护栏）+ triton 在坏区间 | 护栏不存在 → triton 错实现静默跑完全程 | **存疑**，run2/P1 的结论需重新审视 |

区分方法：查当时的 fla/triton 实际版本（pod 上的 pip 记录 / 历史 `train_log2.txt`
里若有版本行）。**本地无法判定**，必须在 pod 上核。

`probe_gdn_backend.py --check-bad-triton` 提供另一条独立证据：它**有意绕过护栏**
跑一次坏 triton 实现，量出它与参考实现的偏离幅度。若偏离与噪声同量级 → 分支 (a)；
若显著偏离 → 说明这个 miscompile 真的会毁掉梯度，分支 (b) 下 run2 需要重新解读。

## 3. 修复：装 tilelang（**不动 triton 版本**）

三条路，只有第二条不牵动整条 CUDA 栈：

| 方案 | 动作 | 风险 |
|---|---|---|
| ① 升 Triton ≥ 3.7.1 | 换 triton 版本 | torch 的 wheel pin 着一个具体 triton 版本，强升会脱pin；vLLM 的 triton kernel（含 `gdn_prefill_backend=triton`）同栈受影响。改版本 = 全栈重新标定 |
| ② **装 tilelang** ✅ | 加一个包 | fla 官方的备用后端；`TileLangBackend` 在 **Hopper + Triton ≥ 3.4.0 下默认启用**（`is_enabled` 无需环境变量），自动接管 `chunk_bwd_dqkwg` |
| ③ 不动 | — | 训练跑不起来 |

**为什么装 tilelang 是安全的（安全边界已核对）**——该后端只实现三个算子：

```
TileLangBackend 实现:  chunk_bwd_dqkwg / parallel_attn_fwd / parallel_attn_bwd
@dispatch('common') 的算子: chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu,
    chunk_fwd_h, chunk_bwd_dh, chunk_fwd_o, chunk_bwd_dv_local, chunk_bwd_dqkwg,
    chunk_scaled_dot_kkt_fwd, fused_beta_sigmoid_{fwd,bwd}
```

其余算子 `getattr(be, func_name)` 取不到 → 继续走 triton。而 `parallel_attn_*` 挂在
`'attn'` registry（Qwen3.5 的全注意力层走 transformers 自己的 sdpa，不经 fla）。
**结论：装 tilelang 在训练路径上只换掉 `chunk_bwd_dqkwg` 一个 kernel，
前向与生成端（vLLM / ref / torch 副本）的数值路径完全不变。**

### 3.1 逐参数复刻真实调用（否则预热无效）

tilelang 编译缓存的 key 含 `(B, H, HV, K, V, BT, BK, BV, NK, hD1, hD2, dtype,
USE_G, USE_DW, STATE_V_FIRST, IS_VARLEN)`，**T 是 dynamic 不进 key**。对照
transformers `modeling_qwen3_5.py` 的真实调用：

| 事实 | 值 | 为什么关键 |
|---|---|---|
| GVA 在 **Python 侧**已展开 | 进 kernel 时 H == HV == 32 | 不是 config 里的 `num_key_heads=16`；照抄 config 会预热错 key |
| `g` 在 Python 侧算好 | `-A_log.exp()*softplus(a+dt_bias)`，**fp32** | 所以**不传** `use_gate_in_kernel`；传了反而偏离真实路径 |
| `beta` 已 sigmoid | 不传 `use_beta_sigmoid_in_kernel` | 同上 |
| `use_qk_l2norm_in_kernel=True` | 真实传参 | 影响数值口径 |
| 训练侧逐行前向（micro_rows=1） | B=1，`IS_VARLEN=False` | B 进编译 key，必须与训练一致 |

### 3.2 装了包 ≠ 能用：要真 import 过（**2026-09-15 实机第二次炸**）

`pip install tilelang --no-deps` 装完，探针的环境行**三绿齐亮**——已装=True、
backend可用=True、backend启用=True——Q1 却直接崩：

```
File "fla/ops/common/backends/tilelang/chunk_bwd.py", line 8, in <module>
    import tilelang
  ...
  File "tilelang/__init__.py", line 179, in <module>
    import tvm
OSError: libz3.so.4.15: cannot open shared object file: No such file or directory
```

**根因两层**：

1. tilelang 的 libtvm 在 **dlopen 阶段**就要 `libz3.so.4.15`。TileLang 把 **Z3 SMT
   求解器**集成进整数分析 pass（layout inference / memory hazard / bound analysis /
   向量化判定），Z3 是**编译期**依赖 —— 所以 `import tilelang` 过了也还不算完。
2. fla 的 `is_available()` 是**代理判据**（`find_spec("tilelang")` + nvcc + 设备能力）：
   它证明"包在 sys.path 上"，**证不了 dlopen 得起来**。前向不 import tilelang（走
   triton）所以一路正常，**第一次反向 dispatch 才炸** —— 代理判据给绿灯，训练照旧
   死在第一个 backward。**这正是 preflight 存在的理由被绕过去了。**

**判据修正（已落进代码）**：`tilelang_verdict()`（`rlab/preflight_gdn.py`，纯函数 +
CPU 测试 F 组）在代理判据之外加一条**真 `import tilelang`**——与 fla 内部那条加载
路径同源。三绿而 import 挂 → 判**不可用**，并打印真实错误串。
`probe_gdn_backend.py` 复用它做前置闸门（不再把"没撞护栏"误读成"tilelang 生效"）；
`preflight_gdn.py` 复用它做拦截，且拦截消息**区分**"没装"与"装了但加载不起来"——
两者修法完全不同，混成一句就等于没判据。

**修法（让 ld.so 按 soname 找得到）**：

```bash
# 1) 先看事实：是不是就是这条路
python -c "import tilelang" 2>&1 | tail -2
find / -name "libz3.so*" 2>/dev/null        # 有没有现成的？版本号对得上吗？

# 2) 补库（版本必须对上 DT_NEEDED 里那条 soname）
pip install "z3-solver==4.15.*" --no-deps
python -c "import z3; print(z3.get_version_string(), z3.__file__)"

# 3) 让 loader 找得到 —— 放系统路径比 LD_LIBRARY_PATH 稳：训练端、子进程、
#    别人起的进程都不会漏掉这个环境变量
Z3LIB=$(python -c "import z3,os;print(os.path.join(os.path.dirname(z3.__file__),'lib'))")
ln -sf $Z3LIB/libz3.so.4.15 /usr/lib/x86_64-linux-gnu/libz3.so.4.15
ldconfig
python -c "import tilelang; print('ok', tilelang.__version__)"   # 判据：这一行必须过
```

**两个坑**：① `ln -sf` 的**文件名必须是 `libz3.so.4.15`**（ld.so 按 `DT_NEEDED` 的
名字找文件，**不校验**文件内的 SONAME）；但目标必须是**真 4.15 ABI**，拿 4.13 改名
顶上会在符号解析时炸。② `import tilelang` 过了**还不算完**：Z3 是**编译期**用的，
必须靠 §4 的 Q1 真跑一次 kernel 才能确认端到端可用。

### 3.3 根因：`--no-deps` 把 tilelang 的真依赖一起跳过了（**§3.2 与 3.4 的共同来源**）

`pip install tilelang --no-deps` 是照抄全局规则"装编译型 CUDA 扩展一律 `--no-deps`，
防 pip 动 torch"。规则的**意图**（别让 pip 顺手换 torch）是对的，**手段**用错了：
`--no-deps` 是把**所有依赖**一起砍掉，而不是"只保护 torch"。

tilelang 0.1.14 在 PyPI 上声明的运行期依赖（`importlib.metadata.requires`）：

```
apache-tvm-ffi<0.1.13,>=0.1.11
z3-solver<4.15.5,>=4.13.0
torch-c-dlpack-ext ; python_version < "3.14"
cloudpickle / ml-dtypes / numpy>=1.23.5 / psutil / tqdm / typing-extensions
torch                              ← 就是这条让 --no-deps 显得"有必要"
```

两条被跳过的依赖各自引爆一次：

| 缺的 | 后果 | 伪装成什么 |
|---|---|---|
| `z3-solver` | tilelang 的 libtvm 在 dlopen 阶段找不到 `libz3.so.4.15` | 三绿齐亮的"代理判据假阳性"（§3.2） |
| `apache-tvm-ffi` | 退回去用**环境里已有的那份**（本机是 FlashInfer 装的），版本对不上 | `import` 期报 `ffi.Tensor already has a registered class`，看着像"tilelang 自己坏了"（§3.4） |

**正确姿势**：用**约束文件**护住不能动的包，让 pip 去装目标包的真依赖。

```bash
python -c "import torch, triton; print(f'torch=={torch.__version__}'); print(f'triton=={triton.__version__}')" > /tmp/pins.txt
pip install --dry-run --report /tmp/tl-report.json tilelang -c /tmp/pins.txt   # 先看要动谁，不动环境
pip install tilelang -c /tmp/pins.txt
```

`-c`（constraints）与 `--no-deps` 的区别是本质的：前者说"**这几个包不许换版本**，
其余依赖正常解析"，后者说"**什么依赖都别装**"。要护 torch，用前者。
（`--dry-run --report` 先出解析结果，符合"改环境前先看判据、别先动手"。）

### 3.4 修完 libz3 后的下一个炸点：`tvm_ffi` 类注册冲突（**同一根因**）

补上 libz3 后 `import tilelang` 换了个错（说明 §3.2 的修法是对的，只是没修完）：

```
File "tilelang/3rdparty/tvm/python/tvm/runtime/_tensor.py", line 65, in <module>
    @tvm_ffi.register_object("ffi.Tensor")
File "site-packages/tvm_ffi/registry.py", line 86, in _register
ValueError: Type 'ffi.Tensor' already has a registered class
  (<class 'tvm_ffi.core.Tensor'>); re-registering it with the larger wrapper class
  (<class 'tvm.runtime._tensor.Tensor'>) is not supported.
  Register the class before any object of this type is materialized
  (which auto-creates a fallback).
```

**读法**：`tvm_ffi` 维护一张「C++ 类型索引 → Python 包装类」的全局注册表。某个索引
第一次被**实例化**时会自动造一个兜底类并占位；之后"正牌"包装类再注册就晚了。
所以这条报错的含义是：**在 tilelang vendored 的 `tvm/runtime/_tensor.py` 第 65 行之前，
已经有人把 `ffi.Tensor` 实例化过** —— 而 `tvm_ffi` 是从 `site-packages/` 加载的，
**不是 tilelang 自带的那份**。

**同一个根因**（§3.3）：`--no-deps` 跳过了 `apache-tvm-ffi<0.1.13,>=0.1.11`，
tilelang 只好退回用环境里已有的那份——本机是 **FlashInfer**（docs/04：GDN prefill 走
FlashInfer JIT）装进去的，版本与 tilelang vendored tvm 期望的对不上。

**关键约束**：`apache-tvm-ffi` 不是"再装一个就完事"——**FlashInfer 也在用它**。
两边的版本区间必须**有交集**，否则是真冲突，就得升级到"换方案"层面重新权衡
（见 §3 的三条路）。所以动手前先查清楚：

```bash
python -c "import importlib.metadata as m; print('tilelang:', m.version('tilelang')); \
print('tvm-ffi:', m.version('apache-tvm-ffi')); print('z3:', m.version('z3-solver'))"
python -c "import importlib.metadata as m; print([r for r in (m.requires('flashinfer-python') or []) if 'tvm' in r.lower()])"
python -c "import importlib.metadata as m; print(m.requires('tilelang'))"   # ★ 与 §3.3 的表对照，看还缺谁
pip install --dry-run tilelang -c /tmp/pins.txt        # 看 pip 的解析结果，先不动环境
```

## 4. 验证判据（带反证）

`python rlab/probe_gdn_backend.py --model_config /root/Qwen3.5-4B-text/config.json`

它对拍 **chunk 路径 vs fla 官方 naive 参考实现**，判据用官方口径
（`get_err_ratio` 相对 RMS + `tests/ops/test_gdn.py` 的 ratio 阈值：
o 0.005 / dq·dk·dv 0.008 / db·dg 0.02），并打印**实测值与倍率**。

三条判据 + 两条反证：

| 项 | 内容 | 反证 |
|---|---|---|
| Q0 | **前置闸门**：tilelang 真 import 通过（见 §3.2） | 三绿而 import 挂 → 早退，不把"没撞护栏"当成"tilelang 生效" |
| Q1 | chunk 路径不撞护栏且数值对拍通过 | 报告里同时给实测 ratio；判据卡线时看倍率而非 pass/fail |
| Q2 | `FLA_TILELANG=0`（子进程）**必须** raise 同一护栏 | 若禁用后也能跑通 → 说明环境根本没踩坑，Q1 的"通过"不能归功于 tilelang |
| Q3 | `--check-bad-triton`：绕过护栏量坏实现偏离 | 与噪声同量级 → run2 可信；显著偏离 → run2 存疑（§2） |

判据本身的正确性由 `rlab/tests/test_gdn_backend_cpu.py`（42 项，CPU 可跑）保证，
其中 D 组是**注入反证**：梯度置零 / 缩放 5% / 元素错位 → 判据必须爆；
0.1% 缩放 → 判据**不该**爆（防止把 bf16 噪声当故障报）。F 组是**可用性判据的反证**：
三绿（find_spec/is_available/is_enabled 全真）但 import 失败 → 必须判不可用，且把
`import_error` 单独置 None 后判定要翻转——证明这条差别真的在起作用，而不是恒真/恒假。

## 5. 最危险的形态：护栏被绕过

**绝不要**用以下方式"修"：

- 改 fla 源码 / monkeypatch 掉那个 `raise` → 直接跑坏 kernel，梯度静默错误；
- 强制 `FLA_DISABLE_BACKEND_DISPATCH=1` → 绕过 dispatch 直取默认实现，等于上一条；
- 只看到"不崩了"就当修好 —— 崩溃是**保护**，不是故障。

同理，`probe_gdn_backend.py --check-bad-triton` 里的绕过是**一次性诊断**（跑在
子进程、结果只用于量偏离），不得进入训练路径。

## 6. pod 操作顺序

```bash
# 0) 先确认卡的占用情况（探针要跑在空闲卡上）
nvidia-smi

# 1) 装 tilelang —— ★ 不要用 --no-deps！见 §3.3：它的运行期依赖（apache-tvm-ffi /
#    z3-solver / torch-c-dlpack-ext）被跳过后，会分别退化成"缺 .so"和"用错版本的
#    邻居包"，而查包在不在的代理判据看不出来。正确姿势是**用约束文件护住不能动的包**：
python -c "import torch, triton; print(f'torch=={torch.__version__}'); print(f'triton=={triton.__version__}')" > /tmp/pins.txt
pip install --dry-run --report /tmp/tilelang-report.json tilelang -c /tmp/pins.txt
#    先看 pip 打算动谁（--dry-run 不动环境）；确认只补不换之后再真装：
pip install tilelang -c /tmp/pins.txt
#    装完必须验依赖没被碰：
pip show torch | head -3          # 版本应与装之前一致
#    ★ 这一步**不是**形式主义：09-15 就是这一步没过（缺 libz3.so.4.15），
#      而 fla 的 find_spec 类代理判据照样报"可用"。过了才继续，否则看 §3.2/§3.3。
python -c "import torch, triton, fla, tilelang; print('import ok')"

# 2) 跑探针（同时完成 tilelang JIT 预热 —— 避免编译窗口叠在三方搬权重的起跑期）
CUDA_VISIBLE_DEVICES=0 python rlab/probe_gdn_backend.py \
    --model_config /root/Qwen3.5-4B-text/config.json
#    期望：Q1 判据通过 + Q2 反证成立（禁用 tilelang 必 raise）

# 3) 追查 run2 是否被污染（可选，多花约 1 分钟）
CUDA_VISIBLE_DEVICES=0 python rlab/probe_gdn_backend.py \
    --model_config /root/Qwen3.5-4B-text/config.json --check-bad-triton

# 4) 起跑（P1b 原命令；本坑不改变任何 cfg，run 签名不变）
bash rlab/run_gsm8k.sh retool_math ...   # 见 docs/05 §6 P1b
```

**注意**：`tilelang` 的 backend 可用性在 **import 期**定死（`find_spec` 缓存），
装完必须**重启进程**；不要在已经在跑的训练进程里指望它生效。

## 7. 可推广的一条

本项目的四个 GDN kernel 坑，形态各异但同源——**"换了模型架构，等于换了整个
运行时依赖图"**：

| # | 坑 | 出现在 | 表现形式 |
|---|---|---|---|
| 1 | vLLM GDN prefill 走 FlashInfer JIT，编译窗口撞宿主 RAM 上限 | 生成端起跑期 | 进程被 SIGKILL，**无 traceback** |
| 2 | transformers 5.x 的 kernel 分派只在 import 期看"包装没装"，与设备无关 | CPU 自检前向 | triton 拿 CPU 张量，ValueError |
| 3 | triton 在坏区间算错 GDN 反向，fla 主动 raise | 训练端 backward | 护栏 raise（**危险的是它不 raise 的版本**） |
| 4 | tilelang 的编译期依赖 libz3 缺失，而 check 用的是 `find_spec` 代理判据 | 训练端 backward 首次 dispatch | **判据三绿齐亮**，`import` 阶段 OSError 缺 .so |
| 5 | `--no-deps` 跳过 tilelang 的真依赖，它退回用邻居包（FlashInfer）的 `tvm_ffi` | `import tilelang` | `ffi.Tensor already has a registered class` |

**第 4 条把前三条的话补完整了**：前三句话说"换了基座 = 换了整张运行时的图"，
第 4 条说明这张图上**"点存在"不等于"边通"**——包在 `sys.path` 上（点），
不代表它加载得起来（边）。凡是"能不能用"的判据，落到最后都必须是**真跑一次那条
最短路径**（真 import / 真构造 / 真前向），而不是查"东西在不在"。

**第 5 条是第 4 条的根因，也是本项目最贵的一条教训**：`--no-deps` 这个从
"装 flash-attn 砸了 torch"里总结出来的防御动作，**自己在另一处造了两次故障**。
它的错不在"防 pip 动 torch"，而在**手段比目的粗**——把"不许换这几个包"做成了
"什么依赖都别装"。第三、四条坑的修法都指向同一件事：**依赖图上的"边"也是要装的**，
而这次踩的正是"我们以为自己只需要一个点，其实需要一整条边"。护 torch 用
`-c constraints.txt`，不要用 `--no-deps`。

四者的共同动作：**换基座后，把"运行时依赖图"重新审一遍**（哪些包会被 import、
各自在什么条件下换实现、编译发生在什么时机、失败时是崩还是静默错）。
显存账本（docs/04 §2）只覆盖显存一项，覆盖不了这张图。
