# 权重同步辅助函数：必须放在独立模块（不能在 __main__ 脚本里），
# 因为 vLLM 0.12 的 EngineCore 是独立 spawn 进程，pickle 反序列化时
# 按 "模块名.函数名" 寻址，__main__ 在子进程里不是本脚本。
def vllm_load_weights(model, sd_items):
    """把 (name, tensor) 列表就地加载进 vLLM worker 内的模型，由 apply_model RPC 执行。"""
    model.load_weights(sd_items)
    return 'loaded'
