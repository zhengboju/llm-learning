# -*- coding: utf-8 -*-
"""rlab/sync.py — 训练端 → vLLM 生成端 的权重同步。

【工程教训内置】
1. vllm_load_weights 必须是本模块的顶层函数：vLLM 0.12 EngineCore 是独立 spawn
   进程，pickle 按 "模块名.函数名" 反序列化，__main__ 里的函数无法跨进程。
2. 必须传 list：odict_items 视图不可 pickle。
3. 优先走官方 apply_model（V1 引擎唯一可靠路径，需环境变量
   VLLM_ALLOW_INSECURE_SERIALIZATION=1）；V0 旧路径作兜底。
4. fail-fast：同步失败宁可崩掉生成进程让训练端 "waiting for batch" 卡死暴露问题，
   也绝不用旧权重静默续训（生成器冻结在 base 的废跑教训）。
"""


def vllm_load_weights(model, sd_items):
    """由 apply_model RPC 到 EngineCore 进程内就地执行。"""
    model.load_weights(sd_items)
    return "loaded"


def remap_text_to_multimodal(sd_items, lm_prefix="model.language_model."):
    """纯文本 torch 键名 -> vLLM 多模态 Qwen3.5 实现的键名（同名张量搬运）。

    【2026-09-11 多模态 Qwen3.5 分裂加载】vLLM 只认多模态 Qwen3.5 checkpoint
    （纯文本 qwen3_5_text 被它路由到多模态实现、processor 崩），torch 侧只能
    加载抽取的纯文本模型（AutoModelForCausalLM 对复合 config 崩）→ 两端用
    不同目录，同步时做键名映射。HF ForConditionalGeneration 布局：
      "model.X"       -> "model.language_model.X"
      "lm_head.*"     -> 保持顶层（两布局同名）
    未知键名 fail-fast：映射表必须与真实布局核对过，静默漏同步 = 生成端用旧权重。
    """
    out = []
    for name, tensor in sd_items:
        if name.startswith("lm_head."):
            out.append((name, tensor))
        elif name.startswith("model."):
            out.append((lm_prefix + name[len("model."):], tensor))
        else:
            raise KeyError(
                f"remap_text_to_multimodal: 未知键名 {name!r}——映射表与模型布局"
                "不匹配，先核对 vLLM qwen3_5.py load_weights 的期望键名再扩展")
    return out


def sync_weights_into_vllm(vllm_gen, state_dict, name_remap=None) -> str:
    """把训练端 state_dict 推进 vLLM。返回实际使用的同步路径，失败抛异常。

    name_remap: 可选 (sd_items)->sd_items 键名映射（多模态 vLLM + 纯文本 torch
    分裂加载时传 remap_text_to_multimodal）。"""
    sd_items = list(state_dict.items())
    if name_remap is not None:
        sd_items = name_remap(sd_items)
    if hasattr(vllm_gen, "apply_model"):
        import functools
        vllm_gen.apply_model(functools.partial(vllm_load_weights, sd_items=sd_items))
        return "apply_model"
    if hasattr(vllm_gen.llm_engine, "model_executor"):  # V0 引擎兜底
        vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model \
            .load_weights(sd_items)
        return "v0 model_executor"
    raise AttributeError("当前 vLLM 版本无可用权重同步路径（V1 引擎且无 apply_model）")
