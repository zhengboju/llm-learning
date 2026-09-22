# -*- coding: utf-8 -*-
"""【终止链核对 #3】EOS 是否进了训练序列（纯 CPU，零标签字面量）。

问题：multi_turn_rollout_group 的 assistant 段 ids 直接取
vllm.outputs[0].token_ids。vLLM 对"自然结束(EOS)"的生成，token_ids
是否包含 EOS 本身？gen_logps 是否覆盖它？若不包含：
  - 模型对"答完就停"这个决策从未拿过梯度（EOS 不在序列里无从学）；
  - merged 训练序列以最后一个内容 token 结尾，训练前向永远看不到终止位。
核对方式（本地可完成，不占卡）：
  1) vLLM 源码级：读本地 vllm 包 sampler/stop 检查路径，确认 EOS 是否
     append 进 token_ids（版本相关的实现事实）。
  2) 序列级：用 record.jsonl 里已完成 run 的 merged 上传批不可得（未落盘），
     但 eval_vllm_one.py 的输出 text 与同一 vLLM 版本行为一致——退而求其次，
     直接构造 FakeGen 走 multi_turn_rollout_group，检查 segs 末段 ids：
     FakeGen 回放的 ids 我们自己控制，无法暴露 vLLM 真行为 —— 所以真正的
     核对必须落在 vLLM 源码 + 官方文档两条证据上，本脚本输出证据链。
运行：python -m rlab.check_eos_in_train_seq
"""
import inspect
import importlib
import sys

def main():
    findings = []

    # ---- 证据 1：vLLM StopChecker / sampler 对 EOS 的处理 ----
    try:
        mod = importlib.import_module("vllm.sampling_params")
        sp = mod.SamplingParams
        doc = sp.__doc__ or ""
        findings.append(("vllm 版本", getattr(importlib.import_module("vllm"), "__version__", "?")))
    except Exception as e:
        findings.append(("vllm 导入失败", repr(e)))
        _print(findings)
        return

    # stop_token_ids / ignore_eos 语义（官方字段即 EOS 行为的合同）
    try:
        import vllm.sampling_params as spm
        src_fields = [f for f in spm.SamplingParams.__dataclass_fields__
                      if "eos" in f or "stop" in f]
        findings.append(("SamplingParams EOS 相关字段", src_fields))
        findings.append(("ignore_eos 默认值",
                         spm.SamplingParams.__dataclass_fields__["ignore_eos"].default))
    except Exception as e:
        findings.append(("SamplingParams 反射失败", repr(e)))

    # ---- 证据 2：输出构造路径里 EOS 是否 append ----
    # vLLM >=0.4：OutputProcessor/StopChecker._process_one_prompt
    candidates = []
    for name in ("vllm.v1.engine.detokenizer",
                 "vllm.v1.engine.output_processor",
                 "vllm.engine.output_processor"):
        try:
            candidates.append(importlib.import_module(name))
        except Exception:
            pass
    found = False
    for m in candidates:
        try:
            src = inspect.getsource(m)
        except Exception:
            continue
        for kw in ("append_token_ids", "new_token_ids.append"):
            if kw in src:
                # 抓上下文行
                lines = src.splitlines()
                hits = [f"{m.__name__}:{i+1}: {ln.strip()}"
                        for i, ln in enumerate(lines) if kw in ln]
                for a, b in (("eos", "append"), ("stop_reason", "append")):
                    pass
                findings.append((f"{m.__name__} 里的 {kw}", hits[:6]))
                found = True
        # EOS append 语义：找"遇到 EOS 且不在 ignore_eos 时是否进输出"
        if "request.eos_token_id" in src:
            lines = src.splitlines()
            hits = [f"{m.__name__}:{i+1}: {ln.strip()}"
                    for i, ln in enumerate(lines) if "eos_token_id" in ln]
            findings.append(("eos_token_id 引用", hits[:10]))
            found = True
    if not found:
        findings.append(("输出构造路径", "未定位到（版本差异），需真机复跑确认"))

    _print(findings)

def _print(findings):
    print("=" * 72)
    print("【终止链核对 #3】EOS 是否进 vLLM 输出 token_ids（进而不进训练序列）")
    print("=" * 72)
    for k, v in findings:
        print(f"\n## {k}")
        if isinstance(v, list) and v:
            for ln in v:
                print("   ", ln)
        else:
            print("   ", v)
    print("\n判读指引：")
    print("  - vLLM 标准语义：EOS 自然停止时 token_ids **包含** EOS"
          "（ignore_eos=False 下 EOS 也 append，再触发 stopped）；")
    print("  - 若证据链支持 EOS 进 token_ids：merged 序列含 EOS、gen_logps 覆盖它")
    print("    → #3 关闭，'答完就停'有梯度，只需 #1 的提示/health 配套；")
    print("  - 若证据链显示 EOS 被 drop（如 vLLM 某些 detokenize 路径）或")
    print("    sampling_params 显式 ignore_eos / 特殊 stop 处理吞掉：")
    print("    → 必须在 gen_worker 采样参数显式核对，并给 #1 换修法（显式")
    print("      append eos + logps 补位）。")
    print("=" * 72)

if __name__ == "__main__":
    main()
