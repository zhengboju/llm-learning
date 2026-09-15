# -*- coding: utf-8 -*-
"""rlab/model_loading.py — torch 侧 causal LM 的统一加载口径 + 防"静默缺键"护栏。

【为什么需要】Qwen3.5-4B 官方 ckpt 是**多模态复合体**（config 顶层没有 vocab_size，
它在 text_config 里），而 rlab 全链路按纯文本 causal LM 设计：
  · transformers ≥5.16 自带 `"qwen3_5_text": PrefixChange(prefix_to_remove="language_model")`
    映射（conversion_mapping.py），≥5.17 的 from_pretrained 还会**自己解包** text_config
    → 裸 torch 进程直连多模态目录即可，主干权重与多模态主干**逐位相等**（2026-09-15 本地
    5.17.0 + 微型复合 ckpt 实测：missing=[]、前向 hidden_states max|Δ|=0）；仓库里 09-14
    的现场记录也一致（rollout.py:699：train 侧两次 run 都加载成功）。
  · 但**进程内 import 过 vLLM 之后**构造会抛 A1（`'Qwen3_5Config' object has no attribute
    'vocab_size'`）——同 ckpt 同版本，差别只在**进程环境**（vLLM 可能注册了自己的 config
    类，令 isinstance 式的解包判断失效）。此时用 `explicit_text_config=True` 显式喂
    text_config，绕开 auto 解析与解包判断。

【护栏：为什么必须显式炸】若 transformers 回退到没有该映射的版本，from_pretrained 对
缺键**只打 warning 就放行**——模型带着一堆**随机初始化**权重继续训练。实测（2026-09-15，
摘掉内置映射后同一份 ckpt）：missing 56/55 个张量，进程不报错。这类失败完全静默，
判据只能自己立：`missing_keys` 必须为空。unexpected 不拦——多模态目录里的 vision 键
被文本模型忽略属预期。
"""


def load_causal_lm(model_path: str, *, dtype, attn_implementation: str | None = None,
                   explicit_text_config: bool = False):
    """加载 causal LM，并**防静默缺键**（missing_keys 非空即抛）。返回模型（不搬设备）。

    explicit_text_config=True：复合 ckpt 显式喂 text_config（vLLM 进程内的 A1 兜底）。
    纯文本 ckpt 传 True 会直接报错——别把"没这个口径"静默当成"不需要"。
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = {"torch_dtype": dtype}
    if attn_implementation:
        kwargs["_attn_implementation"] = attn_implementation
    if explicit_text_config:
        cfg = AutoConfig.from_pretrained(model_path)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is None:
            raise RuntimeError(
                f"{model_path} 不是多模态复合 config（无 text_config），"
                "不需要 explicit_text_config——请检查预检口径是否用错")
        kwargs["config"] = text_cfg

    model, info = AutoModelForCausalLM.from_pretrained(
        model_path, output_loading_info=True, **kwargs)
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
