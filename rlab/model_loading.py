# -*- coding: utf-8 -*-
"""rlab/model_loading.py — torch 侧 causal LM 的统一加载口径 + 防"静默缺键"护栏。

【为什么需要】Qwen3.5-4B 官方 ckpt 是**多模态复合体**（config 顶层没有 vocab_size，
它在 text_config 里），而 rlab 全链路按纯文本 causal LM 设计。复合 config 不能直接喂
文本模型类，必须先把 text_config 解出来——三种做法的可靠性差别极大：

  · 靠 transformers **自动解包**（`AutoModelForCausalLM.from_pretrained(目录)` 不带 config）：
    **随进程环境翻转**。2026-09-15 实机（同一份 ckpt、同一台机、同一次 run）：train 进程
    （裸 torch）解包成功，gen 进程（进程内 import 过 vLLM）不解包 → `AttributeError:
    'Qwen3_5Config' object has no attribute 'vocab_size'`（docs/04 A1）。差别只在进程。
  · 靠预检"复刻真实加载路径"：复刻不全 = 假绿灯。同一次 run 里预检用 `from_config(顶层
    cfg)` 探、真实加载走 `from_pretrained(目录)` 的自动解包——前者自己会解包、后者在 gen
    进程里不会，于是预检在随后崩溃之前**打印了"通过"**。
  · **显式喂 text_config**（本模块做法）：不依赖任何自动解包语义，两个进程同一条路。
    本地 5.17.0 + 微型复合 ckpt 逐位核对：`from_pretrained(目录, config=text_config)` →
    missing=[]、主干权重与多模态主干逐位相等、前向 hidden_states max|Δ|=0。
    键名的字面量映射（`model.language_model.X` -> `model.X`）仍由 transformers 内置的
    "qwen3_5_text" 转换映射负责（5.16 起）。

【护栏：为什么必须显式炸】若 transformers 回退到没有该映射的版本，from_pretrained 对
缺键**只打 warning 就放行**——模型带着一堆**随机初始化**权重继续训练。实测（2026-09-15，
摘掉内置映射后同一份 ckpt）：missing 56/55 个张量，进程一声不响。这类失败完全静默，
判据只能自己立：`missing_keys` 必须为空。unexpected 不拦——多模态目录里的 vision 键
被文本模型忽略属预期。
"""


def resolve_load_config(model_path: str):
    """解析出**加载时真正要喂给模型类**的 config。返回 (config, is_composite)。

    这是"预检探的输入"与"真实加载跑的输入"的**唯一来源**——两者必须调同一个函数，
    否则会分叉成假绿灯（见模块 docstring 第二条）。
    复合多模态 ckpt → 返回其 text_config；纯文本 ckpt → 返回顶层 config。
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    text_cfg = getattr(cfg, "text_config", None)
    return (text_cfg, True) if text_cfg is not None else (cfg, False)


def build_load_kwargs(cfg, composite: bool, dtype, attn_implementation: str | None = None) -> dict:
    """构造 from_pretrained 的实参。**纯函数**——本地 CPU 就能测，不必真加载模型。

    【attn 为什么必须用公开名】私有名 `_attn_implementation` 只在"**不传** config"时才
    被 `AutoConfig.from_pretrained(**kwargs)` 顺手 setattr 到 config 上；一旦显式传 config，
    AutoConfig 不参与解析，私有名就原样漏进 `cls(config, **model_kwargs)` →
    `TypeError: Qwen3_5ForCausalLM.__init__() got an unexpected keyword argument
    '_attn_implementation'`（pod 实机 2026-09-15，ref_server 最先炸——三处加载点全中）。
    公开名由 from_pretrained 自己消费：`config._attn_implementation =
    kwargs.pop("attn_implementation")`（modeling_utils.py:1424），与传不传 config 无关；
    FA2 可用性校验照旧发生在模型 `__init__`（同一处），行为不变。
    """
    kwargs = {"torch_dtype": dtype}
    if composite:
        # 显式喂 text_config：绕开"自动解包随进程环境翻转"。纯文本 ckpt 保持原样加载。
        kwargs["config"] = cfg
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def load_causal_lm(model_path: str, *, dtype, attn_implementation: str | None = None):
    """加载 causal LM：口径 = resolve_load_config + 防静默缺键。返回模型（不搬设备）。"""
    from transformers import AutoModelForCausalLM

    cfg, composite = resolve_load_config(model_path)
    model, info = AutoModelForCausalLM.from_pretrained(
        model_path, output_loading_info=True,
        **build_load_kwargs(cfg, composite, dtype, attn_implementation))
    missing = sorted(info.get("missing_keys") or [])
    if missing:
        raise RuntimeError(
            f"[model_loading] {model_path} 有 {len(missing)} 个张量**没被加载**"
            f"（模型是随机初始化的，会静默跑出错误结果）：{missing[:5]}\n"
            "  最常见原因：transformers 版本缺 qwen3_5_text 的前缀转换映射"
            "（`model.language_model.X` 没被剥成 `model.X`）——用 config.py 记的"
            "版本跑，或先用 rlab/extract_text_model.py 抽一份纯文本 ckpt。\n"
            "  排查：AutoConfig.from_pretrained 后看 model_type 是否 qwen3_5_text、"
            "以及 conversion_mapping 里有无该键。")
    return model
