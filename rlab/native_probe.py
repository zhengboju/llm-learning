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
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlab.protocol import (CODE_TOOL, NATIVE_STYLES, TOOL_NAME,  # noqa: E402
                           build_next_prompt, derive_tool_style,
                           encoded_text_tokens, initial_messages, make_call_id,
                           parse_assistant, render_chat_ids, tool_message)
from rlab.protocol import (_RE_TOOL_CALL_ANY,  # noqa: E402  （诊断用：数调用块）
                           _TOOL_CLOSE, _TOOL_OPEN)
from rlab.config import TOOL_PROTOCOLS  # noqa: E402

DEFAULT_ENGINE_KWARGS = {"gdn_prefill_backend": "triton"}


def engine_kwargs(args) -> dict:
    """vLLM 引擎构造参数（默认档 = triton GDN prefill）。

    【2026-09-25 真机首跑·本文件自己的 bug】Qwen3.5-4B 带 GDN 层，vLLM 默认走
    FlashInfer 的 GDN prefill kernel——那是一次 **ninja 现场 JIT 编译**，本 pod
    实锤打爆宿主内存被 SIGKILL（只打印 `Killed`，**无 traceback**，看起来像
    "冒烟没输出"）。仓库里其它 GPU 入口（rollout.gen_worker / probe_difficulty /
    diag_logps / eval）都传 `gdn_prefill_backend=triton`，唯独本文件漏了——于是
    "go/no-go 判据"那一关根本没跑出数字。探针与训练同档是铁律，这里补默认档。"""
    if args.vllm_gen_kwargs:
        return json.loads(args.vllm_gen_kwargs)     # 整体替换语义（与 train.py 一致）
    return dict(DEFAULT_ENGINE_KWARGS)


def engine_env_warn(model_path, kwargs):
    """起引擎前的两条环境告警（都是本 pod 实锤过的、无 traceback 的坑）。"""
    try:
        from rlab.rollout import gdn_backend_missing
        if gdn_backend_missing(model_path, kwargs):
            print("[probe][警告] 引擎参数里没有 gdn_prefill_backend → Qwen3.5 的 GDN "
                  "prefill 会落到 FlashInfer **现场 JIT**（本 pod 实锤：ninja 打爆宿主 "
                  "RAM → SIGKILL、无 traceback）。", flush=True)
    except Exception:
        pass
    _bi = str(os.environ.get("VLLM_BATCH_INVARIANT", "")).strip().lower()
    if _bi not in ("", "0", "false"):
        print(f"[probe][警告] 环境里继承的 VLLM_BATCH_INVARIANT="
              f"{os.environ.get('VLLM_BATCH_INVARIANT')!r}（多半是训练 run 留下的 "
              f"export）——vLLM 的 batch-invariant 检查跑在 attention backend 解析"
              f"**之前**，没同时给 attention backend 会启动即 RuntimeError。冒烟不需要"
              f"确定性档，建议先 `unset VLLM_BATCH_INVARIANT` 再跑。", flush=True)

SYS = "SYS: you solve math with a python tool."
Q = "What is 17*23? Use the tool if helpful."
CODE = "print(17*23)"

# 冒烟用题：**不能用 17*23 这种口算题**——base 直接心算就把答案说了，调用率被
# 系统性低估（判据 ≥50% 是"会不会主动用工具"，必须给它一个值得用工具的问题）。
# 取几道竞赛风格题，n_smoke 条请求轮着喂，避免单题方差。
SMOKE_QUESTIONS = [
    "Find the sum of all positive integers n such that n^2 + 1 is divisible by n + 1.",
    "How many ordered pairs of positive integers (m, n) satisfy 1/m + 1/n = 1/6?",
    "Let f(x) = x^3 - 3x + 1. Find the sum of the squares of the roots of f(x) = 0.",
    "Compute the remainder when 7^2024 is divided by 1000.",
    "A sequence satisfies a_1 = 1, a_{n+1} = a_n + 2n. Find a_100.",
]


def smoke_config(algo: str, tool_protocol: str):
    """冒烟/渲染用的**训练同口径**配置（系统提示 + 采样参数）。

    【2026-09-25 真机首跑后的自我纠正】首版冒烟用的是本文件里的玩具提示
    `SYS = "SYS: you solve math with a python tool."`——它把"用工具"写在提示里，
    等于**手把手教模型调用**，测出来的调用率是"被提示后的"而不是"base 自己的"。
    于是那句 `100% vs 参考 87.5%` 是**苹果比橘子**：参考的 87.5% 是在它自己的
    正式提示下测的。本项目对探针有一条铁律（probe_difficulty 2026-09-17）：
    **探针与训练同口径**——提示是协议的一半（难度表就是"模型×提示×预算"的联合
    产物），提示不同就是另一个分布。故这里改成从 preset 取真提示与真采样参数，
    保证"调用率"这个 go/no-go 数字说的是**训练将遇到的那个协议**。

    Q1–Q4 的渲染核实也用它：模板行为可能依赖 system 段内容（tools 声明的位置、
    是否触发工具相关的分支），用玩具提示验出来的形态不能外推到训练提示。
    """
    from rlab.config import default_system_prompt, get_config
    cfg = get_config(algo, tool_protocol=tool_protocol)
    return cfg, default_system_prompt(algo, tool_protocol)


def _section(title):
    print("\n" + "=" * 72)
    print(f"== {title}")
    print("=" * 72)


def _dump(path, name, text):
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- {name} ---\n{text!r}\n")


def probe_render(tok, out_path, sys_prompt, question, ctkw):
    """Q1–Q4：渲染四件事并打印（repr 落盘——显示层会改写尖括号标签）。

    一律用**训练同口径**的 system 提示与 chat_template_kwargs（见 smoke_config）：
    模板行为可能依赖 system 段内容，玩具提示验出来的形态不能外推到训练提示。"""
    base = initial_messages(sys_prompt, question)
    res = {}
    # 【Q1 判据不能拿工具名当标志物】训练用的系统提示**自己**就写满了
    # `code_interpreter`（提示的职责就是教模型用这个工具）——于是"不带 tools"
    # 的对照档会打印"含工具声明=是"，把对照读成"模板没渲染也照样有声明"。
    # 标志物必须**只由声明段贡献**，且**跨模板可移植**。实测（本机 Qwen2.5 + 正式提示）：
    #   · `code_interpreter`      ❌ 提示自己就写了
    #   · `<tools>`              ✅ 但那是 Qwen2.5 模板的包装标签，Qwen3.5 未必同名
    #   · CODE_TOOL 描述里的短语  ✅✅ 模板把 description **原样**插进声明段，
    #                                故该短语跨模板可移植（本轮真机已用 Qwen3.5 验证）
    # 取 description 的一个独特子串，随 CODE_TOOL 自动一致（改描述不会让判据失效）。
    _decl_mark = "Execute code in an isolated environment"

    # ---- Q1：初始 prompt 是否带 tools 声明 ----
    _section("Q1 · tools= 是否被模板接受并渲染")
    for tag, tools, _kw in (("带 tools + 关思考", True, ctkw),
                            ("带 tools + 默认思考", True, {}),
                            ("不带 tools（对照）", False, {})):
        try:
            p = tok.apply_chat_template(base, tokenize=False,
                                        add_generation_prompt=True,
                                        **({"tools": [CODE_TOOL]} if tools else {}),
                                        **_kw)
            has_decl = _decl_mark in p
            print(f"  [{tag}] OK  长度={len(p)}  含工具声明段={'是' if has_decl else '否'}"
                  f"（标志物 {_decl_mark!r}）")
            if tag.startswith("带 tools + 关思考"):
                res["q1_prompt"] = p
            _dump(out_path, f"Q1 {tag}", p)
        except Exception as e:
            print(f"  [{tag}] RAISED {type(e).__name__}: {e}")
            if tag.startswith("带 tools + 关思考"):
                res["q1_error"] = f"{type(e).__name__}: {e}"
    if "q1_error" in res:
        print("\n  ✗ Q1 失败：模板不接受 tools= → **方案 A 不成立**，转 docs/08 SFT。")
    elif not (_decl_mark in (res.get("q1_prompt") or "")):
        print(f"\n  ✗ Q1 失败：模板接受了 tools= 但**没渲染出声明段**"
              f"（找不到 {_decl_mark!r}）→ 模型看不到工具签名 → 方案 A 不成立。")

    # ---- Q4：配置档的 thinking 开关与 tools 能否共存 ----
    _section("Q4 · thinking 开关与 tools= 共存（档位取训练同口径）")
    try:
        p0 = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True,
                                     tools=[CODE_TOOL], **ctkw)
        p1 = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True,
                                     tools=[CODE_TOOL],
                                     **{**ctkw, "enable_thinking": True})
        same = p0 == p1
        print(f"  训练档 ctkw={ctkw}")
        print(f"  与 enable_thinking=True 相同? {'是（模板未引用该开关）' if same else '否（开关生效）'}")
        # 【判据】配置档必须让 think 段**闭合**。Qwen3.5 默认 enable_thinking=True，
        # docs/03 已实锤 4B 开着 thinking 会烧穿单轮预算（探针 v1：截断 98.9%、
        # 无 boxed 99.2%）。这里直接断言"训练档渲染出的结尾不是未闭合 <think>"。
        tail = p0.rstrip()[-24:]
        _open_think = tail.rstrip().endswith("<think>")
        print(f"  训练档结尾 repr={tail!r}")
        if _open_think:
            print("  ✗ 训练档以**未闭合** <think> 结尾 → 知识模式仍开启，会烧穿轮预算！"
                  "（docs/03 事故签名：末段截断↑、无 boxed↑）")
            print("    处置：训练/eval 都必须带 --chat_template_kwargs "
                  "'{\"enable_thinking\": false}'")
        else:
            print("  ✅ 训练档结尾不含未闭合 <think>（不烧预算）")
        res["q4_ok"] = not _open_think
        res["q4_same"] = same
    except Exception as e:
        print(f"  RAISED {type(e).__name__}: {e}")
        res["q4_ok"] = False

    # ---- Q2+Q3：调用与回包的渲染形态（两种 arguments 形态都试） ----
    _section("Q2/Q3 · tool_call 与 tool 回包的渲染形态")
    for tag, args in (("dict", {"code": CODE}), ("str", '{"code": "print(17*23)"}')):
        msgs = base + [
            {"role": "assistant", "content": "let me compute",
             "tool_calls": [{"type": "function",
                             "function": {"name": TOOL_NAME,
                                          "arguments": args}}]},
            {"role": "tool", "tool_call_id": make_call_id(0, 0, 1),
             "name": TOOL_NAME, "content": "391"},
        ]
        try:
            t = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        tools=[CODE_TOOL], **ctkw)
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
            # 【判据用**生产解析器**，不用"像不像 JSON"的启发式】旧版按 `"{" in 块
            # and "name" in 块` 找真调用——那只对 Qwen2.5 的 JSON 形态成立；在
            # Qwen3.5 的 `<function=…>` 形态下它会数出 **0 个**，于是打印
            # "含实参 0 个"，把一次正常的渲染读成"调用段是空的"（真机首跑就撞上了）。
            # 现在直接问 parse_assistant：它认的才是"真调用"，且天然跨形态。
            _real = [m for m in _blocks if parse_assistant(m.group(0)).kind == "tool"]
            _pick = _real[-1] if _real else (_blocks[-1] if _blocks else None)
            if _pick is None:
                print("    !! 渲染里没有调用段")
            else:
                i, j = _pick.start(), _pick.end()
                print(f"    调用段 repr={t[i:j]!r}")
                print(f"    （共 {len(_blocks)} 个 tool_call 块：{len(_real)} 个真调用 + "
                      f"{len(_blocks) - len(_real)} 个模板自己的格式说明块——"
                      f"说明块在 system 段，是占位符，不是模型输出）")
                for _n, _m in enumerate(_blocks):
                    _inner = _m.group(0)[len(_TOOL_OPEN):-len(_TOOL_CLOSE)]
                    print(f"      块{_n}: {'真调用' if _m in _real else '格式说明'} "
                          f"载荷={_inner[:120]!r}")
                k = t.find("<tool_response>", j)
                if k < 0:
                    k = t.find("<|im_start|>user", j)
                print(f"    **回包段** repr={t[j:k + 400][:400]!r}" if k >= 0 else "")
            res[f"q2_{tag}"] = t
        except TypeError as e:
            # 【预期差异，不是故障】Qwen3.5 的模板用 `arguments.items()` 展开实参，
            # 故 arguments 必须是 mapping；Qwen2.5 的模板能吃 JSON 字符串。本项目
            # 从不给模板喂 `tool_calls`（token-in token-out 走拼接，见 build_next_prompt），
            # 所以这一档只影响本探针的渲染测试——不要为此改协议。
            print(f"\n  [arguments={tag}] 模板不接受（{e}）——该版模板要求 arguments "
                  f"是 mapping。**对本项目无影响**：我们从不把 tool_calls 喂给模板"
                  f"（续写走 token 拼接），此项仅用于对照各代模板的严格程度。")
        except Exception as e:
            print(f"\n  [arguments={tag}] RAISED {type(e).__name__}: {e}")
            traceback.print_exc()
    return res


def probe_roundtrip(tok, out_path, sys_prompt, question, ctkw):
    """往返证明：实测渲染的 assistant 文本 → parse_assistant 必须认出来。

    这是"正则与模板同源"的**运行时**证据。本机（Qwen2.5）实测 JSON 形态，
    参考正则（Qwen3.5 的 <function=…>）对它必然判 invalid——本函数就是让这个
    差异在 pod 上**当场可见**，而不是等训练 reward 恒 -1 才发现。"""
    _section("往返证明 · parse_assistant(实测渲染文本) 必须 kind='tool'")
    base = initial_messages(sys_prompt, question)
    msgs = base + [{"role": "assistant", "content": "",
                    "tool_calls": [{"type": "function",
                                    "function": {"name": TOOL_NAME,
                                                 "arguments": {"code": CODE}}}]}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                tools=[CODE_TOOL], **ctkw)
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


def _decode(tok, ids):
    try:
        return tok.decode([int(t) for t in ids], skip_special_tokens=False)
    except Exception:
        return repr(list(ids))


def probe_build_next(tok, out_path, sys_prompt, question, ctkw):
    """端到端拼接证明：断言 build_next_prompt 的**硬契约**，并解码任何偏差。

    比看 §3 的七步伪代码可靠：它真的跑一遍。分两层判据——

      【硬契约·必须成立】拼接结果 = prev + sampled + (结束符差额) + observation：
        ① got 以 prev 开头（前缀校验不炸隐含了这条，这里显式断言）
        ② got[len(prev):] 以 sampled 开头（模型采样的 token 逐位保留，没被重编码）
        ③ got 比 prev+sampled 长（observation 真的拼进去了）
      这三条才是"生成/训练同一条序列"的实质。任何一条不成立 = 协议不可用。

      【参考对照·可解释】与模板 canonical 比对，**逐 token 解码**出差异区间。
        canonical 用 `tool_calls` 字段（模板认可的形态）构造；探针若用
        `content=` 塞调用标记，模板会把它当**普通文本**渲染，多出/少掉
        `<tool_call>` 之类的结构标记——那种差异是**探针构造方式**造成的，
        不是 build_next_prompt 的 bug。故差异一律解码打印，由人判读归属，
        不再打印"属预期"这类无信息量的判词（2026-09-25 真机首跑教训：
        该判词恰好盖住了一次未查明的 4-token 偏差）。"""
    _section("端到端拼接证明 · build_next_prompt 的硬契约 + 与模板 canonical 的逐 token 对照")
    base = initial_messages(sys_prompt, question)
    prev = render_chat_ids(tok, base, True, ctkw)
    msgs_a = base + [{"role": "assistant", "content": "",
                      "tool_calls": [{"type": "function",
                                      "function": {"name": TOOL_NAME,
                                                   "arguments": {"code": CODE}}}]}]
    t = tok.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=False,
                                tools=[CODE_TOOL], **ctkw)
    a0 = t.rfind("<|im_start|>assistant")
    a1 = t.find("<|im_end|>", a0)
    inner = t[a0 + len("<|im_start|>assistant\n"):a1] if a0 >= 0 and a1 > a0 else ""
    # 模拟"模型采样出的 token"：模板渲染出的 assistant 段原文逐字编码（含 im_end，
    # 与 vLLM 实际采样一致——EOS 在 token_ids 里，docs/05 §12.3）。
    sampled = encoded_text_tokens(tok, inner)
    obs = tool_message(make_call_id(0, 0, 1), "391")
    try:
        got = build_next_prompt(tok, base, prev, sampled, obs, ctkw)
    except Exception as e:
        print(f"  ✗ build_next_prompt RAISED {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # ---- 硬契约三条 ----
    ok_pref = got[:len(prev)] == prev
    ok_samp = got[len(prev):len(prev) + len(sampled)] == sampled
    ok_long = len(got) > len(prev) + len(sampled)
    print(f"  ① got 以 prev 开头                : {'✅' if ok_pref else '✗'} "
          f"(prev {len(prev)} tok)")
    print(f"  ② got[len(prev):] 以 sampled 开头 : {'✅' if ok_samp else '✗'} "
          f"(sampled {len(sampled)} tok)")
    print(f"  ③ observation 真的拼进去了        : {'✅' if ok_long else '✗'} "
          f"(got {len(got)} vs prev+sampled {len(prev) + len(sampled)}，"
          f"增量 {len(got) - len(prev) - len(sampled)} tok)")
    if not ok_long:
        print("  ✗ 硬契约失败：observation 没拼进去 → 该版模板的 tool 回包渲染"
              "不可用（看 Q3 的 repr），方案 A 不成立。")
    if not (ok_pref and ok_samp):
        print("  ✗ 硬契约失败：拼接改动了模型已采样的 token → 生成/训练序列分叉。")

    # ---- 参考对照：模板认可的 canonical（tool_calls 形态）----
    asst_tc = {"role": "assistant", "content": "",
               "tool_calls": [{"type": "function",
                               "function": {"name": TOOL_NAME,
                                            "arguments": {"code": CODE}}}]}
    try:
        want = render_chat_ids(tok, base + [asst_tc, obs], True, ctkw)
    except Exception as e:
        print(f"  （canonical 渲染失败，跳过对照：{type(e).__name__}: {e}）")
        want = None
    if want is not None:
        n = min(len(got), len(want))
        i = next((k for k in range(n) if got[k] != want[k]), n)
        print(f"\n  对照模板 canonical（tool_calls 形态）：got {len(got)} tok / "
              f"want {len(want)} tok，首个分歧位 {i}")
        if got == want:
            print("  ✅ 与模板 canonical 逐 token 相同")
        else:
            lo, hi = max(0, i - 2), min(max(len(got), len(want)), i + 10)
            print("    差异区间解码（判读归属：拼接 bug / 探针构造差异）：")
            print(f"      got [{lo}:{hi}] ids={list(got[lo:hi])}")
            print(f"          文本={_decode(tok, got[lo:hi])!r}")
            print(f"      want[{lo}:{hi}] ids={list(want[lo:hi])}")
            print(f"          文本={_decode(tok, want[lo:hi])!r}")
            # 多出来的那段单独指认（真机首跑就是这样一段 4-token 结构标记）
            if len(got) > len(want) and got[:i] == want[:i] and got[i + (len(got) - len(want)):] == want[i:]:
                _extra = got[i:i + (len(got) - len(want))]
                print(f"    ⇒ got 比 want 多出 {len(_extra)} tok：ids={list(_extra)} "
                      f"文本={_decode(tok, _extra)!r}")
                print("      若该段是**空的调用块**（结构标记包着空内容），说明"
                      "canonical 的 tool_calls 被模板展开成了真调用段，而拼接路径"
                      "把它当作已在 prev/sampled 里——两者对**同一状态的表示**不同。")
            print(f"    ⇒ 硬契约{'成立' if (ok_pref and ok_samp and ok_long) else '不成立'}"
                  f"；此处差异{'不影响' if (ok_pref and ok_samp and ok_long) else '影响'}"
                  f"「生成/训练同序列」这一实质判据。")
    _dump(out_path, "拼接 got", _decode(tok, got))
    return bool(ok_pref and ok_samp and ok_long)


def main():
    ap = argparse.ArgumentParser(description="原生工具协议上机核实（docs/09 §1.2）")
    ap.add_argument("--model_path", default="/root/Qwen3.5-4B")
    ap.add_argument("--out", default=None,
                    help="把 repr 形态落盘（显示层会改写尖括号标签，故必须 repr 落盘再读）")
    ap.add_argument("--no_vllm_smoke", action="store_true",
                    help="跳过 vLLM 真采样冒烟（只验 tokenizer 层）")
    ap.add_argument("--n_smoke", type=int, default=8, help="vLLM 冒烟采样条数")
    ap.add_argument("--gpu_mem", type=float, default=0.30, help="vLLM 显存占比")
    ap.add_argument("--smoke_max_tokens", type=int, default=1024,
                    help="冒烟单轮生成上限（要够写下一个完整调用块）")
    ap.add_argument("--native_tool_style", default=None,
                    help="钉死解析形态（把 --no_vllm_smoke 那轮的 derive 结论传进来，"
                         "冒烟的 invalid 计数才可归因）")
    ap.add_argument("--vllm_gen_kwargs", default=None,
                    help='JSON dict 覆盖引擎参数（默认 {"gdn_prefill_backend": "triton"}——'
                         "不传会落到 FlashInfer GDN 的现场 JIT，本 pod 实锤 SIGKILL 无 traceback）")
    ap.add_argument("--vllm_attention_backend", default=None,
                    help="显式 attention backend（如 FLASH_ATTN）；确定性档必需")
    ap.add_argument("--algo", default="retool_math",
                    help="取哪个 preset 的系统提示/采样参数（**必须与即将训练的算法一致**："
                         "提示是协议的一半，用玩具提示测出的调用率会显著偏高）")
    ap.add_argument("--tool_protocol", default="native", choices=list(TOOL_PROTOCOLS))
    ap.add_argument("--question", default=SMOKE_QUESTIONS[0],
                    help="Q1–Q4 渲染核实用的题面（默认取一道竞赛题）")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg, sys_prompt = smoke_config(args.algo, args.tool_protocol)
    _ctkw = cfg.get("chat_template_kwargs")
    print(f"[probe] tokenizer = {args.model_path}")
    print(f"[probe] tools 声明 = {TOOL_NAME}")
    print(f"[probe] 口径 = 训练同口径：algo={args.algo} tool_protocol={args.tool_protocol}")
    print(f"[probe] system 提示 = {sys_prompt[:90]!r}...（preset 原文，共 "
          f"{len(sys_prompt)} 字符）")
    print(f"[probe] chat_template_kwargs = {_ctkw}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if args.out:
        open(args.out, "w", encoding="utf-8").close()   # 清空
        print(f"[probe] repr 落盘 -> {args.out}")

    res = probe_render(tok, args.out, sys_prompt, args.question, _ctkw)
    style = probe_roundtrip(tok, args.out, sys_prompt, args.question, _ctkw)
    res["build_next_ok"] = probe_build_next(tok, args.out, sys_prompt,
                                            args.question, _ctkw)

    # ---- vLLM 真采样冒烟（第 5 步的 go/no-go 闸门，docs/09 §6.2）----
    if not args.no_vllm_smoke:
        _section("vLLM 冒烟 · base 工具调用率（判据：≥50% 通过 / <20% 失败）")
        _kw = engine_kwargs(args)
        if args.vllm_attention_backend:
            try:
                from rlab.rollout import attention_backend_kwargs
                _kw.update(attention_backend_kwargs(args.vllm_attention_backend))
            except Exception as e:
                print(f"  （attention backend 参数未能接线：{type(e).__name__}: {e}）")
        engine_env_warn(args.model_path, _kw)
        print(f"  引擎参数: {_kw}")
        try:
            from vllm import LLM, SamplingParams
            from rlab.rollout import build_prompt_ids
            llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_mem, **_kw)
            # 【采样参数也取训练同口径】首版硬编码 temperature=1.0/top_p=1.0——
            # 恰好与 retool_math preset 相同，但那是巧合；从 cfg 取才是结构性保证
            # （改了 preset 的采样参数，冒烟的"调用率"会跟着变，不该各写一份）。
            sp = SamplingParams(n=1, temperature=cfg["temperature"],
                                top_p=cfg["top_p"],
                                top_k=int(cfg.get("top_k", -1)),
                                max_tokens=args.smoke_max_tokens)
            print(f"  采样: temperature={sp.temperature} top_p={sp.top_p} "
                  f"top_k={sp.top_k} max_tokens={sp.max_tokens}（preset 同口径）")
            provs = [build_prompt_ids(
                        initial_messages(sys_prompt,
                                         SMOKE_QUESTIONS[k % len(SMOKE_QUESTIONS)]),
                        tok, _ctkw, tools=True)
                     for k in range(args.n_smoke)]
            outs = llm.generate([{"prompt_token_ids": p} for p in provs], sp,
                                use_tqdm=False)
            n_ok = n_inv = n_ans = 0
            n_ok_auto = 0
            _pin = args.native_tool_style or "auto"
            for k, o in enumerate(outs):
                txt = o.outputs[0].text
                p = parse_assistant(txt, style=_pin)
                if p.kind == "tool":
                    n_ok += 1
                elif p.kind == "invalid":
                    n_inv += 1
                else:
                    n_ans += 1
                if parse_assistant(txt).kind == "tool":
                    n_ok_auto += 1
                if k < 2:
                    print(f"    样本{k} repr={txt[:220]!r}")
                _dump(args.out, f"vLLM 冒烟样本{k}", txt)
            rate = n_ok / max(1, args.n_smoke)
            print(f"\n  调用率 = {n_ok}/{args.n_smoke} = **{rate * 100:.1f}%**"
                  f"（invalid {n_inv} / answer {n_ans}）")
            if _pin != "auto" and n_ok_auto != n_ok:
                # 【防"钉错形态被读成 base 不会调工具"】钉死档与 auto 不一致时两者
                # 都打印：auto 高说明 base 会调用、只是形态与钉死档不同（改钉
                # derive 的结论），而不是能力问题。
                print(f"  ⚠ 钉死档({_pin}) 与 auto 计数不一致：auto = {n_ok_auto}/"
                      f"{args.n_smoke}。以 **auto 高者**判断「base 会不会调用」，"
                      f"以钉死档判断「训练将用哪套解析」——两者不等就要重跑 "
                      f"derive 或核对 --native_tool_style。")
            print("  判据：≥50% ✅ 继续；<20% ✗ 回 §1.2 重查；"
                  "调用率高但 invalid 多 ✗ 正则没对齐实测形态")
            print("  对照：参考实现 base 87.5%（**在它自己的正式提示下**测的——"
                  "本探针现在也用 preset 正式提示，故两者可比；"
                  "若你看到本行打印的是玩具提示，说明代码退回了旧版）")
            print("  另注：rlab p11 围栏协议 ~48% 是**自造格式**的调用率，"
                  "与原生协议的先验不可直接比（前者需 SFT，后者 base 自带）")
            if n_ok + n_inv > 0:
                print(f"  形态分布：命中调用 {n_ok} / 有调用标记但解析失败 {n_inv}"
                      f"（后者 >0 说明 {args.native_tool_style or 'auto'} 与实际形态有偏差）")
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
    _bn = res.get("build_next_ok")
    print(f"  Q5 拼接硬契约(前缀/采样/回包): "
          f"{'✅' if _bn else ('✗ 不成立' if _bn is False else '未跑')}")
    if style:
        print(f"\n  ⇒ 训练命令加：--tool_protocol native --native_tool_style {style}")
    else:
        print("\n  ⇒ 形态未派生：把 --out 文件里 Q2Q3 段的 repr 贴回来，"
              "在 protocol.NATIVE_STYLES 加新形态（**不要**改现有正则去凑）")


if __name__ == "__main__":
    main()
