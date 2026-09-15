# -*- coding: utf-8 -*-
"""rlab/extract_text_model.py — 从多模态 Qwen3.5 checkpoint 抽取纯文本模型。

背景（2026-09-11）：Qwen3.5-4B 官方 checkpoint 是原生多模态
（architectures=Qwen3_5ForConditionalGeneration，config 顶层为 text_config+
vision_config 复合体，vocab_size 在 text_config 里）。rlab 全链路按纯文本
causal LM 设计（与 Qwen2.5 同款）：AutoModelForCausalLM 对复合 config 会拿到
ForCausalLM 类但喂进复合 config → Qwen3_5Config.vocab_size MISSING 直接崩。

不选"让 rlab 适配多模态包装"的原因：权重名多一层 language_model 前缀会打崩
sync.py 的 vLLM 权重同步；vision tower 权重白占 DeepSpeed 优化器显存；ref 模型/
gen_logps 副本/训练端三处都要改。一次性抽取纯文本 checkpoint 是总成本最低解。

用法（pod，CPU 即可，内存 ~20G，约 10 分钟）：
    python -m rlab.extract_text_model --src /root/Qwen3.5-4B --dst /root/Qwen3.5-4B-text

产出 /dst（纯文本 Qwen3_5ForCausalLM + tokenizer + chat template）。之后探针/
训练/评测的 model_path 一律用 --dst；脚本内置分层自检（见 _selfcheck：同权重同路径
逐位对拍为主判据，wrapper 路径偏差单列），不通过会显式报错而不是静默产出坏 ckpt。

注意：本脚本全程 CPU，但 pod 上装了 fla（vLLM 跑 Qwen3.5 GDN 的依赖），必须在 import
modeling 之前显式屏蔽掉它，否则 CPU 自检前向会被塞进 fla 的 triton kernel 崩掉——
根因见 _force_torch_reference_kernels。
"""

import argparse
import importlib.util
import os
import sys

import torch


def _force_torch_reference_kernels() -> list[str]:
    """屏蔽 fla / causal_conv1d，让线性注意力分派回落到 modeling 内的纯 torch 参考实现。

    症状（2026-09-14 pod 实跑）：纯 CPU 自检前向崩在
    `fla/ops/gated_delta_rule/chunk.py` → `l2norm_fwd` 的 triton kernel，报
    `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`。

    根因：transformers 的 `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule",
    "fla")` 是**纯 import 期**决策——`importlib.import_module("fla")` 一成功就换成 fla 的
    triton 实现，**完全不看张量在哪个设备上**。pod 装了 fla（vLLM GDN 的依赖），于是 CPU
    前向也被塞进 triton kernel。旁证：日志里 `causal_conv1d_fn` 那句 "falling back to its
    reference PyTorch implementation" 是同一个装饰器，只因 pod 没装 causal_conv1d 而正常回落。

    规避：把包名在 sys.modules 里置 None（CPython 语义：再次 import 抛 ModuleNotFoundError），
    装饰器的 except 分支即取回参考实现——纯 torch、设备无关，且对拍两侧同路径，① 的逐位
    判据不受影响。自检只有几个 7-token 前向，参考实现的慢无所谓。
    注：USE_HUB_KERNELS=0 挡不住这条路径（它只关掉 kernels 包的 hub 实现，不管原包装回退）。
    若将来要改跑 GPU 自检（省时间但有显存占用），删掉这个调用即可。

    返回真正被屏蔽掉的包名（"装了"才改变分派，"本机没装"不算）——供日志与回归测试断言。
    """
    blocked = []
    for pkg in ("fla", "causal_conv1d"):   # 两者都是 CUDA-only 的 triton/cuda 扩展
        # 先探测"装没装"（find_spec 不执行模块），再看 sys.modules——判据要能分辨
        # "本机没装"（本就没走加速路径）与"装了但被我们屏蔽"（才是真正改变了分派）
        if importlib.util.find_spec(pkg) is not None:
            blocked.append(pkg)
        sys.modules[pkg] = None            # type: ignore[assignment]
    print("[extract] 线性注意力走纯 torch 参考实现（屏蔽了加速 kernel: "
          f"{', '.join(blocked) if blocked else '无，本机未装'}）")
    # 真探针：上面的 print 只是"声明"，这里确认屏蔽**真的**生效（否则自检会以 triton
    # 那句误导性的报错崩掉，看不出是分派没挡住）
    for pkg in blocked:
        try:
            importlib.import_module(pkg)
        except ImportError:
            continue
        raise RuntimeError(
            f"屏蔽 {pkg} 失败（import 仍成功）——装饰器会再次取到加速实现，"
            "自检会在 triton kernel 里以 'cpu tensor?' 崩掉")
    return blocked


def _selfcheck(full, backbone, causal, tok, ids) -> None:
    """抽取自检：分层判据 + 反证控制（2026-09-14 重写）。

    旧判据 `max|Δlogits| > 0.1` 一票否决，实测 0.1017 被判"抽取有误"，是口径问题、
    不是缺陷：
      ① 对拍两侧是两条**不同代码路径**——`full(...)` 走多模态复合 wrapper（自己构造
         position_ids/attention_mask），`causal(...)` 是裸文本模型。同权重不同路径在
         bf16 下逐层舍入，32 层混合线性注意力（本脚本已强制走纯 torch 参考实现，见
         _force_torch_reference_kernels）累积到 1e-1 量级属正常，与"权重抄错"无关。
      ② 阈值是**绝对值**，与 logits 量纲无关，且卡在噪声地板上（0.1017 只超线 1.7%）。

    新判据分三层，①②＋反证是硬闸，③ 只报数不拦：
      ① 同权重同路径必须**逐位相等**：直接调 backbone + full.lm_head 与 causal 对拍，
         max|Δ| 必须恰为 0.0——这才是"抽取正确"的充分证据，不需要任何阈值。
      ② determinism：同一模型两次前向逐位相等。新建的 Qwen3_5ForCausalLM 默认 train
         模式，配置里 dropout 一旦非 0，对拍比的就不是权重而是随机数。
      ③ wrapper 路径偏差单列，用 argmax/top-5 一致性 + softmax 总变差（TV，无量纲）报。
         数学等价的两条路径 TV 应在 1e-3 量级，真错（错层/漏层/lm_head 没绑）TV ~ 1。
         不设硬闸：① 已证明抽取逐位精确，残余偏差属 Qwen3.5 复合前向自身的数值路径
         差异，不该由抽取脚本判死；超 1e-2 打 WARN，值得单独记录。

    反证控制：把某个 block 权重 ×1.02（先确认扰动真落在 bf16 网格上，bf16 相对精度
    ~4e-3）后 ① 必须爆——判据抓不住故意注入的错误，就等于没有判据。还原后复测 ①，
    确认产出不会被污染。
    """
    full.eval()
    causal.eval()          # 新建模块默认 train 模式：dropout 非 0 会污染对拍

    if type(backbone) is not type(causal.model):
        raise RuntimeError(
            f"主干类不同（{type(backbone).__name__} vs {type(causal.model).__name__}）"
            "——① 的逐位前提不成立")

    def h_of(module) -> torch.Tensor:
        with torch.no_grad():
            # ModelOutput 支持按位置取值，第 0 项即 last_hidden_state
            return module(input_ids=ids)[0]

    # ---- ① 同权重同路径：逐位相等（硬闸）----
    h_ref, h_cau = h_of(backbone), h_of(causal.model)
    d_backbone = (h_ref - h_cau).abs().max().item()
    eq_backbone = torch.equal(h_ref, h_cau)
    with torch.no_grad():
        out_ref = full.lm_head(h_ref)            # 直连 lm_head，绕开复合 wrapper
        out_cau = causal(input_ids=ids).logits
    d_head = (out_ref[0, -1] - out_cau[0, -1]).abs().max().item()
    eq_head = torch.equal(out_ref[0, -1], out_cau[0, -1])
    print(f"[extract] ① 主干/lm_head 直连对拍（同权重同路径，要求逐位相等）: "
          f"max|Δh|={d_backbone:.3e} bitwise={eq_backbone} | "
          f"max|Δlogits|={d_head:.3e} bitwise={eq_head}")

    # ---- ② determinism：排除 dropout / RNG（硬闸）----
    det = torch.equal(h_of(causal.model), h_of(causal.model))
    print(f"[extract] ② determinism（同模型两次前向逐位相等）: {det}")

    # ---- ③ wrapper 路径偏差：只报数（信息项，非抽取缺陷）----
    with torch.no_grad():
        lg_cau = out_cau[0, -1].float()
        lg_wrap = full(input_ids=ids).logits[0, -1].float()
    d_wrap = (lg_wrap - lg_cau).abs().max().item()
    tv = 0.5 * (lg_cau.softmax(-1) - lg_wrap.softmax(-1)).abs().sum().item()
    same_top = int(lg_cau.argmax()) == int(lg_wrap.argmax())
    top5 = (set(lg_cau.topk(5).indices.tolist())
            == set(lg_wrap.topk(5).indices.tolist()))
    print(f"[extract] ③ 多模态 wrapper 路径 vs 裸文本路径: max|Δlogits|={d_wrap:.4f} "
          f"softmax TV={tv:.2e} argmax 一致={same_top} top5 一致={top5} "
          f"(top={tok.decode(int(lg_cau.argmax()))!r})")
    if tv > 1e-2:
        print(f"[extract] WARN TV={tv:.2e} 偏大（数学等价的两条路径经验值 ~1e-3）——"
              "残余属复合前向自身差异；可加 --dtype float32 复核是否纯 bf16 舍入")

    # ---- 反证控制：判据必须抓得住"权重错"这类真错误 ----
    target = next(((n, p) for n, p in causal.model.named_parameters()
                   if n.startswith("layers.") and p.dim() >= 2
                   and p.numel() >= 1_000_000), None)
    if target is None:
        raise RuntimeError("找不到可扰动的 block 权重，反证控制无法执行")
    name, w = target
    with torch.no_grad():      # 叶子参数带 requires_grad，原地改写必须在 no_grad 下
        backup = w.detach().clone()
        w.mul_(1.02)
        moved = not torch.equal(w, backup)  # bf16 相对精度 ~4e-3，2% 必须真的改到值
    d_ctrl = (h_of(backbone) - h_of(causal.model)).abs().max().item()
    with torch.no_grad():
        w.copy_(backup)
    restored = torch.equal(w, backup)          # 直接验张量还原，不依赖对拍基数
    d_after = (h_of(backbone) - h_of(causal.model)).abs().max().item()
    print(f"[extract] 反证控制（{name} ×1.02，扰动生效={moved}）: max|Δh|={d_ctrl:.3e} "
          f"| 还原={restored} 还原后 max|Δh|={d_after:.3e}（基线 {d_backbone:.3e}）")

    # ---- 硬闸汇总：按"主判据优先"排序，真缺陷的报错不能被控制项的先决条件掩盖 ----
    # （原地改写都在上面已还原，这里 raise 不影响权重状态；分项数值上面已全部打出）
    if not det:
        raise RuntimeError("同一模型两次前向不逐位相等——存在随机性（dropout/未 eval），"
                           "自检结果不可用")
    if not eq_backbone or not eq_head:
        raise RuntimeError(f"抽取有误：同权重同路径不逐位相等（max|Δh|={d_backbone:.3e} "
                           f"bitwise={eq_backbone}, lm_head bitwise={eq_head}）"
                           "——这才是真错误，不产出 checkpoint")
    if not restored:
        raise RuntimeError(f"{name} 未能还原——产出 checkpoint 会被污染，已中止")
    if not moved or d_ctrl < 1e-3:
        raise RuntimeError(f"判据无区分力：扰动权重后主干差值仅 {d_ctrl:.3e}，自检形同虚设")


def _find_text_backbone(full):
    """定位多模态包装内的纯文本主干（transformers 5.x: model.language_model）。"""
    m = full.model
    for name in ("language_model", "text_model", "model"):
        sub = getattr(m, name, None)
        if sub is not None and hasattr(sub, "layers"):
            print(f"[extract] 文本主干: full.model.{name} ({type(sub).__name__})")
            return sub
    raise RuntimeError(
        f"未找到文本主干（model 下属性: {list(dict(full.model.named_children()))}）")


def main():
    ap = argparse.ArgumentParser(description="多模态 Qwen3.5 -> 纯文本 checkpoint")
    ap.add_argument("--src", required=True, help="多模态 checkpoint 路径")
    ap.add_argument("--dst", required=True, help="输出纯文本 checkpoint 路径")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # 必须在 import modeling 之前：kernel 分派发生在装饰器求值（模块 import 期）
    _force_torch_reference_kernels()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5 import (Qwen3_5ForCausalLM,
                                             Qwen3_5ForConditionalGeneration)

    cfg = AutoConfig.from_pretrained(args.src)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        raise RuntimeError("config 无 text_config——不是多模态复合体，无需抽取")
    print(f"[extract] text_config: vocab_size={text_cfg.vocab_size} "
          f"layers={text_cfg.num_hidden_layers} tie={text_cfg.tie_word_embeddings}")

    print("[extract] 加载多模态全模型（内存峰值 ~2x 模型大小）...")
    full = Qwen3_5ForConditionalGeneration.from_pretrained(args.src, dtype=dtype)
    causal = Qwen3_5ForCausalLM(text_cfg).to(dtype)

    backbone = _find_text_backbone(full)
    missing, unexpected = causal.model.load_state_dict(
        backbone.state_dict(), strict=True)
    print(f"[extract] 主干权重复制完成 missing={missing} unexpected={unexpected}")
    if not text_cfg.tie_word_embeddings:
        causal.lm_head.load_state_dict(full.lm_head.state_dict(), strict=True)
    else:
        causal.tie_weights()   # lm_head 与 embedding 共享，绑一下即可

    # ---- 自检：分层判据（详见 _selfcheck docstring）----
    print("[extract] 分层自检...")
    tok = AutoTokenizer.from_pretrained(args.src)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    _selfcheck(full, backbone, causal, tok, ids)

    # ---- 落盘：模型 + tokenizer（含 chat template，enable_thinking 靠它）----
    print(f"[extract] 保存 -> {args.dst}")
    causal.generation_config = full.generation_config   # eos 列表等停止语义要带走
    causal.save_pretrained(args.dst)
    AutoTokenizer.from_pretrained(args.src).save_pretrained(args.dst)
    # README 等杂项不拷贝；确认产物
    print("[extract] 产物:", sorted(os.listdir(args.dst)))
    print("[extract] 完成。后续 model_path 一律用", args.dst)


if __name__ == "__main__":
    main()
