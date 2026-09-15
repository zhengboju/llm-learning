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
    """由 apply_model RPC 到 EngineCore 进程内就地执行。

    返回 loaded/sent 计数（stacked 融合如 q/k/v->qkv_proj 会让 loaded<sent，
    属正常；此计数只作观测，防静默漏同步靠 AutoWeightsLoader 对未知键报错）。

    【2026-09-15 全零 fail-fast】`loaded == 0 且 sent > 0` 只有一个解释：**键名体系
    对不上**（例如 torch 侧发 `model.layers.X`、vLLM 多模态侧叫
    `model.language_model.X`，而键名映射被关掉了）。这不是"同步了一部分"，是
    "一个张量都没认领"——历史事故（RF++：同步静默失败 → 生成器冻结在 base、
    训练照跑）就是这么来的，必须当场炸。
    loaded=None（老版本 load_weights 不返回清单）时判不了，只告警不拦——
    "宁可不拦，不可错拦"。"""
    loaded = model.load_weights(sd_items)
    n = len(loaded or [])
    if loaded is None:
        print("[sync][警告] 本 vLLM 的 load_weights 不返回已加载清单——"
              "无法用全零判据核验同步是否真的生效，请以生成端权重指纹为准")
    elif n == 0 and len(sd_items) > 0:
        raise RuntimeError(
            f"[sync] 送进 vLLM 的 {len(sd_items)} 个张量**一个都没被认领**——键名体系不匹配。\n"
            f"  首键：{sd_items[0][0]!r}\n"
            "  最常见原因：torch 侧是纯文本布局（model.X / lm_head.*），而 vLLM 侧是"
            "多模态 Qwen3.5（model.language_model.X），键名映射没开。\n"
            "  判据见 rlab.sync.need_text_to_mm_remap（键名形态决定，与两份 checkpoint "
            "是否同一个目录无关）；生成端启动行会打印本次判定的开/关。")
    return f"loaded {n}/{len(sd_items)} tensors"


def need_text_to_mm_remap(*, vllm_checkpoint_composite: bool,
                          vllm_model_path_set: bool = False) -> bool:
    """是否需要"纯文本 torch 键名 → 多模态 vLLM 键名"映射。纯函数（CPU 可测）。

    【2026-09-15 澄清：判据是键名形态，不是目录是否相同】
    c991f84 之后 torch 侧可直连复合 ckpt（load_causal_lm 显式喂 text_config），
    于是"统一用 /root/Qwen3.5-4B 一份目录"完全可行——但**加载能读**不等于
    **同步能对上**：torch 侧一律按纯文本类加载，参数名是 `model.X` / `lm_head.*`；
    vLLM 侧只要吃的是多模态复合体（config 里有 text_config），参数名就是
    `model.language_model.X`。所以：
      · 统一目录（model_path 与 vLLM 同一份复合 ckpt）→ **仍需映射 True**；
        旧判据 `bool(vllm_model_path)` 在这里会给 False → 同步全落空；
      · Qwen2.5 系（vLLM 侧 config 无 text_config）→ False，映射反而不该做；
      · 显式给了 vllm_model_path → 恒 True（保留旧行为：配置文件读不出来时，
        用户显式给的那个 flag 就是唯一可靠信号）。"""
    return bool(vllm_checkpoint_composite) or bool(vllm_model_path_set)


def remap_text_to_multimodal(sd_items, lm_prefix="model.language_model.",
                             drop_tied_lm_head=True):
    """纯文本 torch 键名 -> vLLM 多模态 Qwen3.5 实现的键名（同名张量搬运）。

    【2026-09-11 立；2026-09-15 澄清适用范围】vLLM 只认多模态 Qwen3.5 checkpoint
    （纯文本 qwen3_5_text 被它路由到多模态实现、processor 崩），torch 侧一律按纯文本
    类加载。**是否用同一份目录与本映射无关**：c991f84 后 torch 可直连复合 ckpt
    （显式喂 text_config），"统一用 /root/Qwen3.5-4B 一份目录"是允许的写法——但
    torch 内存里的参数名仍是 `model.X`，vLLM 多模态实现要的是
    `model.language_model.X`，所以映射照做。判据见 need_text_to_mm_remap。
    已按 vLLM qwen3_5.py 源码核实（2026-09-11）：
    wrapper 的 load_weights = AutoWeightsLoader(+hf_to_vllm_mapper)，原始
    checkpoint 的 "model.language_model.X" 键名验证可被加载——映射产出同形态：
      "model.X"       -> "model.language_model.X"
      "lm_head.*"     -> 丢弃（Qwen3.5-4B tie_word_embeddings=True，原
                         checkpoint 无此键；torch state_dict 的 lm_head 是共享
                         张量重复键，发给 AutoWeightsLoader 会报未知参数）
    未知键名 fail-fast：映射表必须与真实布局核对过，静默漏同步 = 生成端用旧权重。
    """
    out = []
    for name, tensor in sd_items:
        if name.startswith("lm_head."):
            if drop_tied_lm_head:
                continue
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

    name_remap: 可选 (sd_items)->sd_items 键名映射（torch 纯文本布局 → 多模态 vLLM
    时传 remap_text_to_multimodal；判定见 need_text_to_mm_remap）。"""
    sd_items = list(state_dict.items())
    if name_remap is not None:
        sd_items = name_remap(sd_items)
    if hasattr(vllm_gen, "apply_model"):
        import functools
        res = vllm_gen.apply_model(
            functools.partial(vllm_load_weights, sd_items=sd_items))
        return f"apply_model [{res}]"   # 计数进日志：loaded<sent 常态（stacked 融合）
    if hasattr(vllm_gen.llm_engine, "model_executor"):  # V0 引擎兜底
        vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model \
            .load_weights(sd_items)
        return "v0 model_executor"
    raise AttributeError("当前 vLLM 版本无可用权重同步路径（V1 引擎且无 apply_model）")
