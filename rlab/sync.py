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


def sync_weights_into_vllm(vllm_gen, state_dict) -> str:
    """把训练端 state_dict 推进 vLLM。返回实际使用的同步路径，失败抛异常。"""
    sd_items = list(state_dict.items())
    if hasattr(vllm_gen, "apply_model"):
        import functools
        vllm_gen.apply_model(functools.partial(vllm_load_weights, sd_items=sd_items))
        return "apply_model"
    if hasattr(vllm_gen.llm_engine, "model_executor"):  # V0 引擎兜底
        vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model \
            .load_weights(sd_items)
        return "v0 model_executor"
    raise AttributeError("当前 vLLM 版本无可用权重同步路径（V1 引擎且无 apply_model）")
