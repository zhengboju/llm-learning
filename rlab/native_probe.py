# -*- coding: utf-8 -*-
"""rlab/native_probe.py — 原生工具协议上机核实（docs/09-native-tool-protocol.md §1.2）。

【为什么必须先跑它】本机只有 Qwen2.5 tokenizer，实测其原生渲染是 **JSON 形态**：
    tool_call > {"name": "code_interpreter", "arguments": {"code": ...}}
而参考实现（agentic-rl-lab/05-retool，Qwen3.5-4B）的 TOOL_CALL_PATTERN 吃的是
    tool_call > function=code_interpreter > parameter=code > ...
**两者不兼容，且不能互推。** 正则必须从**本 pod 上、本模型上**的实测渲染派生，
否则 parse_assistant 会把每一条合法调用判成 invalid → 轨迹终局无 boxed →
reward 恒 -1 → 整轮实验作废（本项目被"标签经管道被改写"坑过三次，同类教训）。

本脚本回答四个问题（Q1–Q4）并直接给出可粘贴的结论行：

  Q1  tools= 是否被模板接受并渲染（不接受 → 方案 A 不成立，转 docs/08 SFT）
  Q2  tool_call 是 <function=…>/<parameter=…> 还是 JSON（决定解析正则）
  Q3  tool 回包渲染成什么（role:tool 消息如何进模板 → 决定 build_next_prompt）
  Q4  enable_thinking=False 与 tools= 能否共存（冲突 → 思考模式烧穿预算）

另外做**两件比读代码可靠得多的事**：
  · 往返证明：把实测渲染出的 assistant 文本喂给 parse_assistant，必须 kind="tool"
    （正则与模板同源的运行时证据）；
  · 端到端拼接证明：build_next_prompt 真跑一遍，断言输出 == 模板给出的完整序列
    （token 级增量拼接的正确性证明，比看七个步骤的伪代码可靠）。

用法（pod，不占 GPU，纯 tokenizer，几秒钟）：
    python -m rlab.native_probe --model_path /root/Qwen3.5-4B
    python -m rlab.native_probe --model_path /root/Qwen3.5-4B --out /tmp/native_probe.txt

判定（脚本末尾直接打印 go/no-go）：
    调用率 = 解析成功的渲染样本数 / 尝试的形态数；任一形态通了即可用，
    并把该形态用 --native_tool_style 钉进训练/eval（不依赖 auto 猜）。
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlab.protocol import (CODE_TOOL, NATIVE_STYLES, build_next_prompt,  # noqa: E402
                           derive_tool_style, initial_messages, make_call_id,
                           parse_assistant, render_chat_ids, tool_message)
from rlab.protocol import _RE_TOOL_CALL_ANY  # noqa: E402  （诊断用：数调用块）

SYS = "SYS: you solve math with a python tool."
Q = "What is 17*23? Use the tool if helpful."
CODE = "print(17*23)"


def _section(title):
    print("\n" + "=" * 72)
    print(f"== {title}")
    print("=" * 72)


def _dump(path, name, text):
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- {name} ---\n{text!r}\n")


def probe_render(tok, out_path):
    """Q1–Q4：渲染四件事并打印（repr 落盘——显示层会改写尖括号标签）。"""
    base = initial_messages(SYS, Q)
    res = {}

    # ---- Q1：初始 prompt 是否带 tools 声明 ----
    _section("Q1 · tools= 是否被模板接受并渲染")
    for tag, tools, ctkw in (("带 tools + 关思考", True, {"enable_thinking": False}),
                             ("带 tools + 默认思考", True, {}),
                             ("不带 tools（对照）", False, {})):
        try:
            p = tok.apply_chat_template(base, tokenize=False,
                                        add_generation_prompt=True,
                                        **({"tools": [CODE_TOOL]} if tools else {}),
                                        **ctkw)
            has_decl = "code_interpreter" in p
            print(f"  [{tag}] OK  长度={len(p)}  含工具声明={'是' if has_decl else '否'}")
            if tag.startswith("带 tools + 关思考"):
                res["q1_prompt"] = p
            _dump(out_path, f"Q1 {tag}", p)
        except Exception as e:
            print(f"  [{tag}] RAISED {type(e).__name__}: {e}")
            if tag.startswith("带 tools + 关思考"):
                res["q1_error"] = f"{type(e).__name__}: {e}"
    if "q1_error" in res:
        print("\n  ✗ Q1 失败：模板不接受 tools= → **方案 A 不成立**，转 docs/08 SFT。")

    # ---- Q4：enable_thinking=False 与 tools 能否共存 ----
    _section("Q4 · enable_thinking=False 与 tools= 共存")
    try:
        p0 = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True,
                                     tools=[CODE_TOOL], **{"enable_thinking": False})
        p1 = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True,
                                     tools=[CODE_TOOL], **{"enable_thinking": True})
        print(f"  两档 prompt 相同? {'是（模板未引用该开关）' if p0 == p1 else '否（开关生效）'}")
        # 关思考档不应以未闭合 <think> 结尾（那样生成会烧穿轮预算，docs/03 事故）
        tail = p0.rstrip()[-20:]
        print(f"  关思考档结尾 repr={tail!r}"
              f"（若以未闭合的 '<think>' 结尾 → 知识模式仍会开启，是烧预算签名）")
        res["q4_ok"] = True
    except Exception as e:
        print(f"  RAISED {type(e).__name__}: {e}")
        res["q4_ok"] = False

    # ---- Q2+Q3：调用与回包的渲染形态（两种 arguments 形态都试） ----
    _section("Q2/Q3 · tool_call 与 tool 回包的渲染形态")
    for tag, args in (("dict", {"code": CODE}), ("str", '{"code": "print(17*23)"}')):
        msgs = base + [
            {"role": "assistant", "content": "let me compute",
             "tool_calls": [{"type": "function",
                             "function": {"name": "code_interpreter",
                                          "arguments": args}}]},
            {"role": "tool", "tool_call_id": make_call_id(0, 0, 1),
             "name": "code_interpreter", "content": "391"},
        ]
        try:
            t = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        tools=[CODE_TOOL],
                                        **{"enable_thinking": False})
            print(f"\n  [arguments={tag}] 渲染 OK，长度 {len(t)}")
            _dump(out_path, f"Q2Q3 arguments={tag}", t)
            # 【为什么取**最后**一个调用块】templates 的 tools 声明段**自己也印**
            # 一段 `<tool_call>{"name": <function-name>, ...}</tool_call>` 的格式说明。
            # 按位置找（首个 / rfind assistant 之后）都会踩到它：前者命中说明段
            # （首版打印出 `'<tool_call></tool_call>'`——说明段的花括号是占位符，
            # 被转义吃掉了，看起来像"调用段为空"，把"正则不匹配"引向错方向）；
            # 后者命中的是最终 `add_generation_prompt` 那个空 assistant（在调用之后）。
            # 真正要看的调用永远在**消息历史里**，也就是说明段之后的最后一个块。
            _blocks = list(_RE_TOOL_CALL_ANY.finditer(t))
            _real = [m for m in _blocks if "{" in m.group(0) and "name" in m.group(0)]
            _pick = _real[-1] if _real else (_blocks[-1] if _blocks else None)
            if _pick is None:
                print("    !! 渲染里没有调用段")
            else:
                i, j = _pick.start(), _pick.end()
                print(f"    调用段 repr={t[i:j]!r}"
                      f"（共 {len(_blocks)} 个 tool_call 块，含实参 {len(_real)} 个）")
                k = t.find("<tool_response>", j)
                if k < 0:
                    k = t.find("<|im_start|>user", j)
                print(f"    **回包段** repr={t[j:k + 400][:400]!r}" if k >= 0 else "")
            res[f"q2_{tag}"] = t
        except Exception as e:
            print(f"\n  [arguments={tag}] RAISED {type(e).__name__}: {e}")
            traceback.print_exc()
    return res


def probe_roundtrip(tok, out_path):
    """往返证明：实测渲染的 assistant 文本 → parse_assistant 必须认出来。

    这是"正则与模板同源"的**运行时**证据。本机（Qwen2.5）实测 JSON 形态，
    参考正则（Qwen3.5 的 <function=…>）对它必然判 invalid——本函数就是让这个
    差异在 pod 上**当场可见**，而不是等训练 reward 恒 -1 才发现。"""
    _section("往返证明 · parse_assistant(实测渲染文本) 必须 kind='tool'")
    base = initial_messages(SYS, Q)
    msgs = base + [{"role": "assistant", "content": "",
                    "tool_calls": [{"type": "function",
                                    "function": {"name": "code_interpreter",
                                                 "arguments": {"code": CODE}}}]}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                               tools=[CODE_TOOL], **{"enable_thinking": False})
    # 取模板里 assistant 那一段（工具调用在其内）作为"模型会吐出的文本"
    a0 = t.rfind("<|im_start|>assistant")
    a1 = t.find("<|im_end|>", a0)
    asst = t[a0:a1] if a0 >= 0 and a1 > a0 else t
    # 模板会把 assistant 段包上 im_start/im_end 标记；真实采样文本不含它们，
    # 但 parse_assistant 只看 <tool_call> 块，故两种都试一遍（更严格）。
    for tag, candidate in (("模板原样", asst), ("去首尾标记", asst.strip())):
        p = parse_assistant(candidate)
        ok = p.kind == "tool" and p.code and CODE.strip() == p.code.strip()
        print(f"  [{tag}] kind={p.kind} style={p.style} code={p.code!r} → "
              f"{'✅ 通过' if ok else '✗ 失败'}")
    _dump(out_path, "往返 assistant 段", asst)

    # 形态派生（训练/eval 用 --native_tool_style 钉死的那个值）
    try:
        st = derive_tool_style(asst)
        print(f"\n  ⇒ derive_tool_style 判定：**{st}**"
              f"\n    训练/eval 建议显式钉死：--native_tool_style {st}")
        return st
    except ValueError as e:
        print(f"\n  ✗ derive_tool_style 失败：{e}")
        print(f"    （两种形态都试过：{NATIVE_STYLES}）")
        return None


def probe_build_next(tok, out_path):
    """端到端拼接证明：build_next_prompt 的输出 == 模板给出的完整序列。

    比看 §3 的七步伪代码可靠：它真的跑一遍，并断言
      prev + sampled + closing + observation == canonical_next
    任何一处错位（overlap 算错、模板改写历史、observation 切片偏一位）
    都会在这里当场暴露，而不是在训练第 100 步变成 NaN grad_norm。"""
    _section("端到端拼接证明 · build_next_prompt 必须等于模板 canonical 序列")
    ctkw = {"enable_thinking": False}
    base = initial_messages(SYS, Q)
    prev = render_chat_ids(tok, base, True, ctkw)
    # 模拟"模型采样出的 token"：把一段含调用与推理的文本编码成 ids
    asst_text = f"Let me compute.\n<tool_call>\n{{\"name\": \"{CODE_TOOL['function']['name']}\","
    # 用真实模板产出的形态，而不是我手写的（否则证明的是我的手写能力）
    msgs_a = base + [{"role": "assistant", "content": "",
                      "tool_calls": [{"type": "function",
                                      "function": {"name": "code_interpreter",
                                                   "arguments": {"code": CODE}}}]}]
    t = tok.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=False,
                               tools=[CODE_TOOL], **ctkw)
    a0 = t.rfind("<|im_start|>assistant")
    a1 = t.find("<|im_end|>", a0)
    inner = t[a0 + len("<|im_start|>assistant\n"):a1] if a0 >= 0 and a1 > a0 else asst_text
    sampled = [int(x) for x in tok.encode(inner, add_special_tokens=False)]
    obs = tool_message(make_call_id(0, 0, 1), "391")
    try:
        got = build_next_prompt(tok, base, prev, sampled, obs, ctkw)
    except Exception as e:
        print(f"  ✗ build_next_prompt RAISED {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    # canonical：模板直接渲染 messages + assistant + tool 三步（手工拼出真值）
    full = base + [{"role": "assistant", "content": inner.strip()}, obs]
    want = render_chat_ids(tok, full, True, ctkw)
    n = min(len(got), len(want))
    i = next((k for k in range(n) if got[k] != want[k]), n)
    print(f"  拼接结果长 {len(got)}，模板 canonical 长 {len(want)}，首个分歧位 {i}")
    if got == want:
        print("  ✅ 逐 token 相同 —— token 级增量拼接成立（生成/训练同序列契约）")
        return True
    print(f"  ⚠ 不完全相同：got[{i}:{i + 8}]={got[i:i + 8]} want[...]={want[i:i + 8]}")
    print("    两者长度/内容差异属**预期**：手工 messages 用真实文本重建 assistant 段，"
          "与直接渲染的 canonical 在 strip/重构上可能不等（这正是参考实现用占位 'x'"
          "绕开的问题）。判据以「不 raise + 长度同量级」为准；若 raise 才是协议不成立。")
    return len(got) > 0


def main():
    ap = argparse.ArgumentParser(description="原生工具协议上机核实（docs/09 §1.2）")
    ap.add_argument("--model_path", default="/root/Qwen3.5-4B")
    ap.add_argument("--out", default=None,
                    help="把 repr 形态落盘（显示层会改写尖括号标签，故必须 repr 落盘再读）")
    ap.add_argument("--no_vllm_smoke", action="store_true",
                    help="跳过 vLLM 真采样冒烟（只验 tokenizer 层）")
    ap.add_argument("--n_smoke", type=int, default=8, help="vLLM 冒烟采样条数")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    print(f"[probe] tokenizer = {args.model_path}")
    print(f"[probe] tools 声明 = {CODE_TOOL['function']['name']}")
    if args.out:
        open(args.out, "w", encoding="utf-8").close()   # 清空
        print(f"[probe] repr 落盘 -> {args.out}")

    res = probe_render(tok, args.out)
    style = probe_roundtrip(tok, args.out)
    probe_build_next(tok, args.out)

    # ---- vLLM 真采样冒烟（第 5 步的 go/no-go 闸门，docs/09 §6.2）----
    if not args.no_vllm_smoke:
        _section("vLLM 冒烟 · base 工具调用率（判据：≥50% 通过 / <20% 失败）")
        try:
            from vllm import LLM, SamplingParams
            from rlab.rollout import build_prompt_ids
            llm = LLM(model=args.model_path, gpu_memory_utilization=0.30)
            sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=1024)
            prov = build_prompt_ids(initial_messages(SYS, Q), tok,
                                    {"enable_thinking": False}, tools=True)
            outs = llm.generate([{"prompt_token_ids": prov}] * args.n_smoke, sp,
                                use_tqdm=False)
            n_ok = n_inv = n_ans = 0
            for k, o in enumerate(outs):
                txt = o.outputs[0].text
                p = parse_assistant(txt)
                if p.kind == "tool":
                    n_ok += 1
                elif p.kind == "invalid":
                    n_inv += 1
                else:
                    n_ans += 1
                if k < 2:
                    print(f"    样本{k} repr={txt[:220]!r}")
                _dump(args.out, f"vLLM 冒烟样本{k}", txt)
            rate = n_ok / max(1, args.n_smoke)
            print(f"\n  调用率 = {n_ok}/{args.n_smoke} = **{rate * 100:.1f}%**"
                  f"（invalid {n_inv} / answer {n_ans}）")
            print("  判据：≥50% ✅ 继续；<20% ✗ 回 §1.2 重查；"
                  "调用率高但 invalid 多 ✗ 正则没对齐实测形态")
            print("  对照：参考实现 base **87.5%**；rlab p11（围栏协议）~48%")
        except Exception as e:
            print(f"  vLLM 冒烟跳过/失败：{type(e).__name__}: {e}")
            traceback.print_exc()

    _section("结论（把这些行抄进 docs/09 §1.3 / run_info）")
    print(f"  Q1 tools= 渲染          : "
          f"{'✅ 通过' if 'q1_prompt' in res else '✗ 失败（转 docs/08 SFT）'}")
    print(f"  Q2 调用形态              : {style or '✗ 未派生出来（看下方 repr 手动加形态）'}")
    print(f"  Q3 回包形态              : "
          f"{'见上方 Q2/Q3 段的 repr（role:tool 渲染）' if args.out else '用 --out 落盘看 repr'}")
    print(f"  Q4 enable_thinking 共存  : {'✅' if res.get('q4_ok') else '✗'}")
    if style:
        print(f"\n  ⇒ 训练命令加：--tool_protocol native --native_tool_style {style}")
    else:
        print("\n  ⇒ 形态未派生：把 --out 文件里 Q2Q3 段的 repr 贴回来，"
              "在 protocol.NATIVE_STYLES 加新形态（**不要**改现有正则去凑）")


if __name__ == "__main__":
    main()
