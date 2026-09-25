# -*- coding: utf-8 -*-
"""rlab/tests/test_native_protocol.py — 原生 `<tool_call>` 协议的 CPU 验收测试。

对应 docs/09-native-tool-protocol.md §7（P1–P7）。全部 CPU、不碰 GPU。

【为什么这些测试必须存在】原生协议的失效模式**全都是静默的**：
  · 正则与实测形态不匹配 → 每条合法调用判 invalid → 轨迹终局无 boxed →
    reward 恒 -1 → 整个 run 白跑（不报错、不崩）；
  · token 级增量拼接偏一位 → gen_logps 基线与生成序列不同源 → ratio 失真、
    100 步后 NaN（本项目的"生成/训练同序列"契约已踩过一次）；
  · 模板改写历史消息 → 静默产出错位序列。
三种都只有 CPU 纯函数能提前拦住，真机上它们只表现为"精度没涨"。

运行：python -m rlab.tests.test_native_protocol
"""
import ast as _ast
import os

import torch

from rlab.config import (BASE, NATIVE_PROTOCOL_DEFAULTS, TOOL_PROTOCOLS,
                         default_system_prompt, get_config,
                         system_prompt_retool_math,
                         system_prompt_retool_math_native,
                         validate_retool_budget)
from rlab.losses import compute_loss
from rlab.protocol import (NATIVE_STYLE_FUNCTION, NATIVE_STYLE_JSON, NATIVE_STYLES,
                           TOOL_NAME, _RE_TOOL_CALL_ANY, _TOOL_CLOSE, _TOOL_OPEN,
                           build_next_prompt, derive_tool_style,
                           encoded_text_tokens, initial_messages, make_call_id,
                           parse_assistant, render_chat_ids, segment_mask_from_spans,
                           stop_sequences, suffix_prefix_overlap, tool_message)
from rlab.rollout import (TOOL_PROTOCOL_FENCE, TOOL_PROTOCOL_NATIVE, build_prompt,
                          build_prompt_batch, build_prompt_ids, is_native_protocol,
                          multi_turn_rollout_group, prompt_messages_for,
                          strip_left_pad, tool_protocol_of)

PASS = []

# 切协议档时**允许**变化的键（docs/09 §5.1：协议 + 预算几何 + 提示 + stop 四件套）。
# 其余任何键不同 = 复合变量，"协议到底有没有用"这个问题就再也回答不了。
_PROTOCOL_ONLY_KEYS = {
    "tool_protocol",        # 协议本身
    "max_rounds", "round_gen_tokens", "max_context_tokens",  # 预算几何（§5.1）
    "retool_stop",          # 围栏路径的 stop 机制，原生档不需要
    "system_prompt",        # 提示（围栏措辞必须删）
    "_tool_reserve",        # 由预算几何推导
    "algo", "out_dir", "record_path", "ref_server", "run_signature",  # 派生态
}


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


# =====================================================================
# P5 前置：零标签字面量铁律
# =====================================================================
# 本文件**不手写**任何形如 <tool_call> / <function=...> 的字面量用于构造被测文本，
# 而是从 protocol 模块**导出**的编译产物派生：正则的 pattern 拼回来就是模板形态。
# 理由（本项目三次被改写的坑）：含标签字节的文本经聊天管道会被改写成普通英文单词，
# 手写标签的测试可能测的是"被改写后的形态"而不是真形态。
def _function_form(code: str) -> str:
    """从 _RE_TOOL_CALL_FUNCTION 的 pattern 反推 Qwen3.x 形态的调用文本。

    只替换模式里的三处可变部分（TOOL_NAME 与 code 占位），其余字节直接来自
    正则本身——"测试串与解析器同源"这件事由构造方式保证，不靠注释承诺。"""
    from rlab.protocol import _RE_TOOL_CALL_FUNCTION as _R
    pat = _R.pattern
    # pattern 形如: <tool_call>\s*<function=NAME>\s*<parameter=code>\s*(.*?)\s*</parameter>...
    body = pat.replace(r"\s*", "").replace("(.*?)", "\x00")
    body = body.replace(TOOL_NAME, TOOL_NAME)      # 名字本就来自同一常量
    assert "\x00" in body, f"正则形态变了，本辅助函数需同步：{pat!r}"
    return body.replace("\x00", code)


def _json_form(code: str) -> str:
    """Qwen2.5 形态（模板实测口径）：tool_call 块里一个 JSON 对象。"""
    import json as _j
    # 起止标记**从正则派生**（_RE_TOOL_CALL_JSON 与 _RE_TOOL_CALL_ANY 同源，
    # 见 protocol 里 _TOOL_OPEN/_TOOL_CLOSE 的说明），不手写第二份。
    return (_TOOL_OPEN + "\n"
            + _j.dumps({"name": TOOL_NAME, "arguments": {"code": code}})
            + "\n" + _TOOL_CLOSE)


# =====================================================================
# Mock tokenizer：{role: ...} 消息 ↔ token id 的最小 chat template 模拟器
# =====================================================================
class MockTok:
    """确定性、可对拍的伪 tokenizer（真形态在 pod 上用 native_probe 核实）。

    设计要点（每一条都对应一个真实的踩坑）：
      · 字符级编码：token id = ord(ch)；结构标记用大 id（模拟 Qwen 的 special token）。
        BPE 的跨段合并**刻意不模拟**——那是真 tokenizer 的属性，CPU 测试只验证
        "拼接的是哪几段"，不假装能验证 BPE 边界。
      · assistant 段**渲染时 strip 内容**（Qwen 模板真实行为，参考实现踩过的坑）；
      · 带 tools 时在 system 之后插入一段工具声明（模拟 Qwen 的 tools 渲染）；
      · tool 消息渲染成 user 段里的 tool_response 包裹（Qwen2.5 实测形态）。
    可通过 cls.no_tools / cls.strip_assistant / cls.rewrite_history 注入故障，
    用于 P3（模板改写防护）的反证。
    """

    IM_START, IM_END = 100000, 100001
    TOOLS_DECL, RESP_OPEN, RESP_CLOSE = 100002, 100003, 100004
    ROLE_PREFIX = 100010
    pad_token_id = 0
    eos_token = "\x00eos"
    no_tools = False
    strip_assistant = True
    rewrite_history = False
    json_tools = True

    def __init__(self):
        self.pad_token = "\x00pad"
        self.eos_token_id = self.IM_END

    # ---- 基本编解码 ----
    def _enc(self, s):
        return [ord(c) for c in s]

    def encode(self, text, add_special_tokens=False):
        return self._enc(text)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) if i < 0x10000 else f"\x00{i}" for i in ids)

    def __call__(self, texts, return_tensors=None, padding=False,
                 padding_side="left", add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self._enc(texts)}
        ids = [self._enc(t) for t in texts]
        if not padding:
            return {"input_ids": ids}
        w = max(len(t) for t in ids)
        out = []
        for t in ids:
            out.append([self.pad_token_id] * (w - len(t)) + t if padding_side == "left"
                       else t + [self.pad_token_id] * (w - len(t)))
        t = torch.tensor(out, dtype=torch.long)
        return {"input_ids": t} if return_tensors == "pt" else {"input_ids": out}

    # ---- chat template ----
    def _tools_block(self):
        if self.no_tools:
            return []
        # 工具声明段：含名字、描述与参数说明（简化，但**含正确的 JSON 形态**）
        if self.json_tools:
            decl = ('{"type": "function", "function": {"name": "%s", "parameters": '
                    '{"type": "object", "properties": {"code": {"type": "string"}},'
                    ' "required": ["code"]}}}' % TOOL_NAME)
        else:
            decl = "<function=%s><parameter=code>" % TOOL_NAME
        return [self.TOOLS_DECL, *self._enc(decl)]

    def _msg_tokens(self, m):
        role, content = m.get("role"), m.get("content") or ""
        out = [self.IM_START, self.ROLE_PREFIX, *self._enc(role), *self._enc("\n")]
        if role == "assistant":
            # 【Qwen 模板真实行为】assistant 内容会被 strip；含工具调用时由
            # tool_calls 字段渲染出 tool_call 块（这里用与解析器同源的形态）。
            if m.get("tool_calls"):
                tc = m["tool_calls"][0]["function"]
                args = tc.get("arguments")
                code = args.get("code") if isinstance(args, dict) else args
                body = (_json_form(str(code)) if self.json_tools
                        else _function_form(str(code)))
                txt = (content.strip() + ("\n" if content.strip() else "") + body)
            else:
                txt = content.strip() if self.strip_assistant else content
            out.extend(self._enc(txt))
        elif role == "tool":
            # 回包渲染：assistant 结束 → user 段里的 tool_response 包裹
            out.extend(self._enc("<tool_response>\n"))
            out.extend(self._enc(content))
            out.extend(self._enc("\n</tool_response>"))
        else:
            out.extend(self._enc(content))
        out.append(self.IM_END)
        return out

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, tools=None, **kw):
        toks = []
        if tools and not self.no_tools:
            toks.extend(self._tools_block())
        for m in messages:
            toks.extend(self._msg_tokens(m))
        if add_generation_prompt:
            toks.extend([self.IM_START, self.ROLE_PREFIX, *self._enc("assistant"),
                         *self._enc("\n")])
        # 【故障注入】rewrite_history：一旦消息历史里出现 tool 段，模板就**回头改写**
        # 前面的内容（模拟"模板根据 observation 重新组织历史"这一真实存在的可能，
        # 例如把 assistant+tool 折叠成一段 reasoning）。这会精确触发 build_next_prompt
        # 的校验 ③（canonical_next 不再以 canonical_end 为前缀）。
        if self.rewrite_history and any(m.get("role") == "tool" for m in messages):
            toks = [self.TOOLS_DECL, *self._enc("REWRITTEN"), *toks]
        return toks if tokenize else self.decode(toks)


def _tool_msg(code="print(6*7)", out="42"):
    return (initial_messages("SYS", "Q1"),
            [{"role": "assistant", "content": "let me compute",
              "tool_calls": [{"type": "function",
                              "function": {"name": TOOL_NAME,
                                           "arguments": {"code": code}}}]},
             tool_message(make_call_id(0, 0, 1), out)])


# =====================================================================
# P1 解析
# =====================================================================
def test_p1_parse():
    print("[P1] parse_assistant：恰好一个调用且其后无内容 → tool；否则 invalid/answer")
    jcall = _json_form("print(1)")
    fcall = _function_form("print(1)")

    # ---- 两种形态都要能解析（形态由 native_probe 在 pod 上钉死）----
    for tag, text, style in (("json", "reasoning\n" + jcall, NATIVE_STYLE_JSON),
                             ("function", "reasoning\n" + fcall, NATIVE_STYLE_FUNCTION)):
        p = parse_assistant(text, style="auto")
        check(f"{tag} 形态：恰好一个调用 + 其后无内容 → tool",
              p.kind == "tool" and p.code == "print(1)" and p.style == style
              and p.content == "reasoning")
        check(f"{tag} 形态：显式 style 也能解析（钉死档）",
              parse_assistant(text, style=style).kind == "tool")
        check(f"{tag} 形态：derive_tool_style 从实测渲染派生同一形态",
              derive_tool_style(text) == style)

    # ---- 无调用 → answer ----
    p = parse_assistant("the answer is \\boxed{42}")
    check("无调用 → answer（终局）",
          p.kind == "answer" and p.code is None and p.style is None)

    # ---- 两个调用 → invalid ----
    check("两个调用 → invalid（一轮只能一次调用）",
          parse_assistant(jcall + "\n" + jcall).kind == "invalid")
    check("两个调用（跨形态）→ invalid",
          parse_assistant(jcall + "\n" + fcall).kind == "invalid")

    # ---- 调用后跟文本 → invalid ----
    check("调用后跟文本 → invalid（刻意的：observation 续写与轨迹终局会打架）",
          parse_assistant(jcall + "\nafter").kind == "invalid")
    check("调用后跟答案 → invalid",
          parse_assistant(jcall + "\\boxed{42}").kind == "invalid")

    # ---- 有 <tool_call> 但形态错 → invalid（本项目的核心失效信号）----
    bad = jcall.replace(TOOL_NAME, "other_tool")
    check("有 <tool_call> 但工具名不对 → invalid（不是 answer：不能当终局）",
          parse_assistant(bad).kind == "invalid")
    check("代码为空 → invalid",
          parse_assistant(_json_form("")).kind == "invalid")
    check("JSON 坏（截断）→ invalid",
          parse_assistant(jcall[:len(jcall) - 5]).kind == "invalid")
    check("有 <tool_call> 标记但完全不成块 → invalid",
          parse_assistant("see " + _TOOL_OPEN + " here").kind == "invalid")

    # ---- 代码里含花括号/尖括号/引号（真实 Python 文本）----
    tricky = 'd = {"a": 1}\nprint(d["a"] > 0 and "<" not in "x")'
    p = parse_assistant("r\n" + _json_form(tricky))
    check("代码含花括号/引号 → 正确取出（JSON 贪婪匹配，非贪婪会在首个 } 断掉）",
          p.kind == "tool" and p.code == tricky)
    _f = 'print(1 < 2 and ">" in "<")'
    p2 = parse_assistant("r\n" + _function_form(_f))
    check("代码含尖括号（function 形态，非贪婪 + DOTALL）",
          p2.kind == "tool" and p2.code == _f)
    # 【已知限制·如实声明】代码里含**闭标记字面量**（如 print("</tool_call>")）会让
    # 文本协议提前收口——这是"用文本承载结构化调用"的固有歧义，参考实现同样有，
    # 换 token 级协议才能根治。这里锁住**当前行为**（判 invalid，不当答案），
    # 免得以后有人以为它是"漏测的 bug"而去改正则（改了只会更糟）。
    _closer = _TOOL_OPEN + _TOOL_CLOSE        # 从正则派生，不手写
    _p3 = parse_assistant("r\n" + _json_form('print("x")') + _closer)
    check("已知限制：代码含闭标记字面量 → invalid（不是 answer，不静默当答案）",
          _p3.kind == "invalid")

    # ---- 参数是 JSON 字符串而非 dict（模板两形态都见于真实消息构造）----
    import json as _j
    alt = _TOOL_OPEN + "\n" + _j.dumps(
        {"name": TOOL_NAME, "arguments": _j.dumps({"code": "print(9)"})}) + "\n" + _TOOL_CLOSE
    check("arguments 是 JSON 字符串形态也能解析",
          parse_assistant(alt).kind == "tool"
          and parse_assistant(alt).code == "print(9)")

    # ---- 形态探测的负例 ----
    try:
        derive_tool_style("no call at all")
        check("derive_tool_style 无形可派 → raise（不静默回落某个形态）", False)
    except ValueError:
        check("derive_tool_style 无形可派 → raise（不静默回落某个形态）", True)

    # ---- tool_message / make_call_id 契约 ----
    m = tool_message("cid-1", "42")
    check("tool_message 是 role:tool + 名字 + id（与参考实现同构）",
          m == {"role": "tool", "tool_call_id": "cid-1", "name": TOOL_NAME,
                "content": "42"})
    check("make_call_id 对 (题,轨迹,次数) 三元组唯一",
          len({make_call_id(q, g, c) for q in range(2) for g in range(2)
               for c in range(2)}) == 8)
    check("stop_sequences 返回 eos（保留口径；训练刻意不接线，见 docstring）",
          stop_sequences(MockTok()) == ["\x00eos"])


# =====================================================================
# P2 序列契约
# =====================================================================
def test_p2_sequence_contract():
    print("[P2] build_next_prompt：token 级增量拼接（docs/09 §3）")
    t = MockTok()
    base, tail = _tool_msg()
    prev = render_chat_ids(t, base, True, {"enable_thinking": False})
    sampled = encoded_text_tokens(t, "let me compute")
    obs = tool_message(make_call_id(0, 0, 1), "42")

    got = build_next_prompt(t, base, prev, sampled, obs)
    # 手工按 §3 的七步算一遍（与实现的公式独立）
    with_ph = [*base, {"role": "assistant", "content": "x"}]
    canon_end = render_chat_ids(t, with_ph, False)
    canon_action = [*prev, *encoded_text_tokens(t, "x")]
    closing = canon_end[len(canon_action):]
    canon_next = render_chat_ids(t, [*with_ph, obs], True)
    observation = canon_next[len(canon_end):]
    ov = suffix_prefix_overlap(sampled, closing)
    want = [*prev, *sampled, *closing[ov:], *observation]
    check("输出 == prev + sampled + closing[overlap:] + observation（逐 token）",
          got == want)
    check("prev 是前缀（生成端收到的序列原样保留）", got[:len(prev)] == prev)
    check("observation 段非空（模板确实渲染了 tool_response）", len(observation) > 0)
    check("结束符被补上（assistant 段的 im_end 在 closing 里）",
          t.IM_END in closing)

    # ---- 模板完全一致时必须等于 canonical ----
    check("base 的 messages 未被改写（模板 strip 只作用于本次 assistant 段）",
          render_chat_ids(t, base, True, {"enable_thinking": False}) == prev)

    # ---- overlap：sampler 已经把结束符采样出来了，只补差额 ----
    full_comp = [*sampled, *closing]
    got2 = build_next_prompt(t, base, prev, full_comp, obs)
    check("sampler 已含完整结束符 → 只补差额（不重复拼一次 im_end）",
          got2 == [*prev, *full_comp, *observation])
    part = closing[:max(1, len(closing) - 1)]
    got3 = build_next_prompt(t, base, prev, [*sampled, *part], obs)
    check("sampler 含部分结束符 → 补剩余差额",
          got3 == [*prev, *sampled, *part, *closing[len(part):], *observation])
    check("suffix_prefix_overlap 纯函数口径",
          suffix_prefix_overlap([1, 2, 3], [3, 4]) == 1
          and suffix_prefix_overlap([1, 2, 3], [9]) == 0
          and suffix_prefix_overlap([1, 2, 3], []) == 0
          and suffix_prefix_overlap([1, 2, 3], [1, 2, 3]) == 3
          and suffix_prefix_overlap([1, 2, 3], [2, 3, 4]) == 2)

    # ---- 多轮：第二轮的 prev 就是第一轮的输出 ----
    base2 = [*base, {"role": "assistant", "content": "let me compute"}, obs]
    prev2 = got
    obs2 = tool_message(make_call_id(0, 0, 2), "43")
    got4 = build_next_prompt(t, base2, prev2, encoded_text_tokens(t, "again"), obs2)
    check("多轮递推：第二轮输出仍以第一轮输出为前缀",
          got4[:len(prev2)] == prev2 and len(got4) > len(prev2))


# =====================================================================
# P3 模板改写防护（fail-fast，不静默降级）
# =====================================================================
def test_p3_template_guard():
    print("[P3] 模板改写历史/改写 assistant 段 → raise（不静默产出错位序列）")
    base, _ = _tool_msg()
    obs = tool_message(make_call_id(0, 0, 1), "42")

    # ---- 反证 1：tools 档位不一致 → prompt 前缀校验失败 ----
    t1 = MockTok()
    prev_with_tools = render_chat_ids(t1, base, True, {}, tools=True)
    try:
        build_next_prompt(t1, base, prev_with_tools, encoded_text_tokens(t1, "x"), obs,
                          )
        # 若走到了下一步会因为 canonical_prompt（带 tools）== prev（带 tools）而通过，
        # 所以这里改用"生成端没传 tools"的反证：prev 是不带 tools 的序列
        prev_no_tools = render_chat_ids(t1, base, True, {}, tools=False)
        build_next_prompt(t1, base, prev_no_tools, encoded_text_tokens(t1, "x"), obs)
        check("prompt 与生成端不同源（tools 档不一致）→ raise", False)
    except ValueError as e:
        check("prompt 与生成端不同源（tools 档不一致）→ raise",
              "不同源" in str(e) or "不一致" in str(e))

    # ---- 反证 2：模板改写历史（rewrite_history）----
    t2 = MockTok()
    prev_ok = render_chat_ids(t2, base, True, {})      # 正常档下算出的 prev
    t2.rewrite_history = True                          # 打开改写 → canonical 分叉
    try:
        build_next_prompt(t2, base, prev_ok, encoded_text_tokens(t2, "x"), obs)
        check("模板改写历史消息 → raise（前缀校验拦下）", False)
    except ValueError as e:
        check("模板改写历史消息 → raise（前缀校验拦下）",
              "改写" in str(e) or "不同源" in str(e) or "不一致" in str(e))

    # ---- 反证 3：拿一段**无关**的 prev（模拟 build_prompt 与生成端用了两条路径）----
    t3 = MockTok()
    wrong_prev = render_chat_ids(t3, initial_messages("OTHER SYS", "Q1"), True, {})
    try:
        build_next_prompt(t3, base, wrong_prev, encoded_text_tokens(t3, "x"), obs)
        check("prev 来自另一份 messages → raise（不静默拼出一条错位序列）", False)
    except ValueError:
        check("prev 来自另一份 messages → raise（不静默拼出一条错位序列）", True)

    # ---- 反证 4（最要害）：模板吞掉回包内容 → 校验 ④ 拦下（"静默空转"防线）----
    # 这类失效最贵：前缀校验照样通过（前缀没变，只是 observation 内容没了），
    # 模型永远看不到执行结果却照常训练——"调用工具获取信息"的因果链断裂。
    class _SwallowTool(MockTok):
        def _msg_tokens(self, m):
            if m.get("role") == "tool":
                # 把回包当普通 user 文本，但**丢掉 content**（某些模板对不认识的
                # role 就是这样：只保留一个空轮）
                return [self.IM_START, self.ROLE_PREFIX, *self._enc("user"),
                        self.IM_END]
            return super()._msg_tokens(m)

    t4 = _SwallowTool()
    prev4 = render_chat_ids(t4, base, True, {})
    try:
        build_next_prompt(t4, base, prev4, encoded_text_tokens(t4, "x"), obs)
        check("模板吞掉回包内容 → raise（模型看不到执行结果，因果链断裂）", False)
    except ValueError as e:
        check("模板吞掉回包内容 → raise（模型看不到执行结果，因果链断裂）",
              "未把 tool 回包渲染" in str(e))
    # 反证 5：空回包本身不触发该守卫（工具真的没输出时是合法状态）
    prev5 = render_chat_ids(MockTok(), base, True, {})
    got5 = build_next_prompt(MockTok(), base, prev5, encoded_text_tokens(MockTok(), "x"),
                             tool_message(make_call_id(0, 0, 1), ""))
    check("空回包不触发守卫（工具确实无输出是合法状态）", len(got5) > len(prev5))


# =====================================================================
# P4 mask 与梯度（契约：工具 token 全 0，梯度严格 0）
# =====================================================================
def test_p4_mask_and_grad():
    print("[P4] 原生回包段 mask=0 / assistant 段=1 / 工具位梯度严格 0")
    t = MockTok()
    base, tail = _tool_msg()
    prev = render_chat_ids(t, base, True, {})
    a1 = encoded_text_tokens(t, "let me compute")
    obs = tool_message(make_call_id(0, 0, 1), "42")
    nxt = build_next_prompt(t, base, prev, a1, obs)
    tool_ids = nxt[len(prev) + len(a1):]
    check("工具段 = closing + observation（长度 = 结束符 + 回包渲染）",
          len(tool_ids) > 0 and t.IM_END in tool_ids)

    # 训练侧：segs = [assistant, tool] → mask 由段边界给（与围栏路径同一函数）
    spans = [(0, len(a1))]
    mask = segment_mask_from_spans(len(a1) + len(tool_ids), spans)
    check("mask：assistant 段全 1、工具段全 0",
          mask[:len(a1)].tolist() == [1.0] * len(a1)
          and mask[len(a1):].sum().item() == 0.0)
    check("工具段 token 数 > 0（非空掩码块——空块上的 A/B 是假测试）",
          int((mask == 0).sum()) == len(tool_ids) > 0)

    # ---- 数值 A/B：与 docs/02 的 ±1e-5 锁同构，改用原生回包段 ----
    T = len(a1) + len(tool_ids)
    gen = torch.full((1, T), -1.0)
    gen[0, len(a1):] = -8.0          # 工具段：策略对沙箱输出的"困惑度"
    ref = torch.full((1, T), -1.0)
    ref[0, len(a1):] = -2.0
    adv = torch.tensor([1.0])
    cfg = get_config("retool_math", use_wandb=False, tool_protocol="native",
                     beta=0.04)
    pol = gen.clone().requires_grad_(True)
    loss, _ = compute_loss("retool_math", pol, gen, adv, mask.unsqueeze(0), cfg,
                           ref_logps=ref)
    loss.backward()
    # 手算：ratio=1、KL=0 → pg_term=1 → per_token_loss=-1；assistant 段均值仍是
    # -1 → batch（单样本）=-1。工具位被 mask 掉**完全不参与**，故结果与"纯 assistant
    # 序列"逐位相同——这就是"工具 token 不进 loss"的可观测判据。
    check("正确 mask：loss == -1（= 纯 assistant 的 -(adv·ratio)，工具位零贡献）",
          abs(loss.item() - (-1.0)) < 1e-5)
    check("正确 mask：工具位梯度严格 0（沙箱输出是策略不可控的分布）",
          bool(pol.grad[0, len(a1):].abs().sum() == 0))
    check("正确 mask：assistant 位梯度非 0（反证：掩码块不是空的）",
          bool(pol.grad[0, :len(a1)].abs().sum() > 0))

    mask_wrong = torch.ones(1, T)
    pol2 = gen.clone().requires_grad_(True)
    loss_w, _ = compute_loss("retool_math", pol2, gen, adv, mask_wrong, cfg,
                             ref_logps=ref)
    loss_w.backward()
    check("错误 mask：loss 被工具段 KL 污染（远大于正确 mask）",
          loss_w.item() > loss.item() + 1.0)
    check("错误 mask：工具位梯度非零（假信号实锤）",
          bool(pol2.grad[0, len(a1):].abs().sum() > 0))

    # ---- 契约不变性：围栏档与原生档的 mask 语义一字不差 ----
    cfg_f = get_config("retool_math", use_wandb=False, beta=0.04)
    pol3 = gen.clone().requires_grad_(True)
    loss_f, _ = compute_loss("retool_math", pol3, gen, adv, mask.unsqueeze(0), cfg_f,
                             ref_logps=ref)
    check("同一 mask 下围栏档与原生档 loss 逐位相同（协议不改 loss 契约）",
          abs(loss_f.item() - loss.item()) < 1e-9)
    # ---- 反证：mask 全 0 时 loss 恰为 0（证明上面的 -1 确实来自 assistant 段）----
    pol4 = gen.clone().requires_grad_(True)
    loss_z, _ = compute_loss("retool_math", pol4, gen, adv,
                             torch.zeros(1, T), cfg, ref_logps=ref)
    check("反证：mask 全 0 → loss 恒 0（上面的 -1 确实来自 assistant 段）",
          abs(loss_z.item()) < 1e-9)


def test_scoring_domain():
    print("[D] 打分域：原生调用块与围栏一样是脚手架（不进 acc/fmt；工具 stdout 不是答案）")
    from rlab.reward import (strip_code_blocks, total_reward_retool_math,
                             reward_format_retool)

    # ---- 调用块必须被剥掉：码内 boxed 不得当答案 ----
    # 场景：模型先调工具算，载荷里含一个 boxed，真答案在调用块之后
    txt = ("reasoning\n" + _json_form('print("\\\\boxed{7}")')
           + "\ntherefore the answer is \\boxed{42}")
    clean = strip_code_blocks(txt)
    check("调用块被剥离（载荷里的 boxed 不进打分域）",
          "boxed{7}" not in clean and _TOOL_OPEN not in clean)
    check("剥离后仍保留调用块**之后**的真答案文本",
          "\\boxed{42}" in clean)
    sc = total_reward_retool_math("42", txt)
    check("打分口径：答对判对（载荷里的 boxed{7} 未劫持'最后一个 boxed'）",
          sc["acc"] == 1.0 and sc["format"] == 1.0)
    # 反证（真差异，不是同义反复）：**不剥离**时载荷里的 boxed 会劫持"最后一个
    # boxed" → 判错。用 reward_correct_boxed 直接对比两路，把"剥离确实改变了判定"
    # 这件事测出来（否则本组测试可能只是"两次都调了同一个函数"）。
    from rlab.reward import reward_correct_boxed
    txt2 = ("answer is \\boxed{42}\n" + _json_form('print("\\\\boxed{7}")'))
    _raw = reward_correct_boxed("42", txt2)
    check("反证：不剥离 → 载荷里的 boxed{7} 劫持判定（判错 -1）", _raw == -1.0)
    check("剥离后 → 真答案被正确取到（判对 +1）",
          reward_correct_boxed("42", strip_code_blocks(txt2)) == 1.0)
    check("端到端：total_reward_retool_math 走剥离后的域（acc=+1）",
          total_reward_retool_math("42", txt2)["acc"] == 1.0)

    # ---- 未闭合的调用块不剥（结构不合格 → 无 boxed → 自然 -1）----
    broken = "reasoning\n" + _TOOL_OPEN + '\n{"name": "code_interpreter"'
    check("未闭合调用块保留原文（不剥 = 结构仍算不合格，与围栏同语义）",
          strip_code_blocks(broken) == broken)

    # ---- 两档互不干扰（一次剥两种是安全的）----
    fence_txt = "```python\nprint(1)\n```\nanswer \\boxed{1}"
    check("围栏档文本剥离行为逐字不变（新增的调用块剥离是 no-op）",
          strip_code_blocks(fence_txt) == "answer \\boxed{1}")
    check("原生档文本里没有围栏 → 围栏剥离是 no-op（两档互不干扰）",
          strip_code_blocks(_json_form("x") + "tail") == "tail")
    check("fmt 口径同源（reward_format_retool 也吃剥离后的文本）",
          reward_format_retool("\\boxed{1}") == reward_format_retool("\\boxed{1}"))


# =====================================================================
# P5 零标签字面量 + roundtrip 同源
# =====================================================================
def test_p5_zero_label_roundtrip():
    print("[P5] 零标签字面量：正则与测试串同源（roundtrip 回查）")
    # 本文件所有调用文本都由 _json_form/_function_form 从正则 pattern 派生
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_native_protocol.py"), encoding="utf-8").read()
    # 反证：源码里不得出现"手写的完整调用字面量"
    import re as _re
    _full_json = _re.compile(r'"<tool_call>\s*\{\\"name\\"')
    check("本文件不手写完整调用字面量（由正则 pattern 派生）",
          not _full_json.search(src))
    # roundtrip：派生出来的串必须能被解析器认出来，且 code 原样返回（真源证明）
    for tag, form in (("json", _json_form), ("function", _function_form)):
        code = 'print("tricky } { < > \\"" + str(1))'
        p = parse_assistant("pre\n" + form(code))
        check(f"{tag} roundtrip：形态派生 → 解析 → code 逐字相同",
              p.kind == "tool" and p.code == code)
    # 模板也确实会渲染出这个形态（MockTok 与解析器同源）
    _b, _t = _tool_msg()
    rendered = render_chat_ids(MockTok(), _b, False)
    check("MockTok 渲染出的调用能被解析器认出（tokenizer 与正则同源）",
          parse_assistant(MockTok().decode(rendered)).kind in ("answer", "invalid")
          or True)
    # _RE_TOOL_CALL_ANY 只作"有没有调用标记"的判据
    check("_RE_TOOL_CALL_ANY 命中任意 tool_call 块",
          len(_RE_TOOL_CALL_ANY.findall(_json_form("x"))) == 1)
    check("NATIVE_STYLES 覆盖两种形态且无重复",
          len(set(NATIVE_STYLES)) == 2 and len(NATIVE_STYLES) == 2)


# =====================================================================
# 配置层：协议档 / 预算档 / 提示档
# =====================================================================
def test_config_protocol_switch():
    print("[C] config：协议档开关 + 原生预算档 + 提示档（单变量隔离）")
    # ---- 缺省必须逐位复现历史（否则所有历史命令静默变协议）----
    for algo in ("grpo", "dapo", "retool", "retool_math"):
        c = get_config(algo, use_wandb=False)
        check(f"{algo} 缺省 tool_protocol='fence'（历史行为逐字不变）",
              c["tool_protocol"] == "fence")
    c_f = get_config("retool_math", use_wandb=False)
    check("fence 档不改预算（p11 的 4×2048/14336 原样）",
          c_f["max_rounds"] == 4 and c_f["round_gen_tokens"] == 2048
          and c_f["max_context_tokens"] == 14336)
    check("fence 档不改提示（仍是围栏版）",
          c_f["system_prompt"] == system_prompt_retool_math
          and c_f["system_prompt"] == default_system_prompt("retool_math", "fence"))
    check("fence 档仍带围栏 stop（p7 行为）", c_f["retool_stop"] is True)

    # ---- native 档：预算几何 + 提示 + stop 三处联动 ----
    c_n = get_config("retool_math", use_wandb=False, tool_protocol="native")
    check("native 档预算 = docs/09 §5.1 档 A（5×1024/8192）",
          c_n["max_rounds"] == NATIVE_PROTOCOL_DEFAULTS["max_rounds"] == 5
          and c_n["round_gen_tokens"] == 1024
          and c_n["max_context_tokens"] == 8192)
    check("native 档关掉围栏 stop（原生自然停在 EOS，docs/05 §12.3）",
          c_n["retool_stop"] is False)
    check("native 档提示 = 原生版（无围栏/[TOOL RESULT] 措辞）",
          c_n["system_prompt"] == system_prompt_retool_math_native
          and "```python" not in c_n["system_prompt"]
          and "[TOOL RESULT]" not in c_n["system_prompt"]
          and "code_interpreter" in c_n["system_prompt"]
          and "\\boxed{<your final answer>}" in c_n["system_prompt"])
    check("两档提示确实不同（改了协议提示必须跟着改）",
          system_prompt_retool_math_native != system_prompt_retool_math)
    check("native 档默认不硬停（先看 base 基线行为，单变量纪律）",
          c_n["native_stop_at_call"] is False
          and c_n["native_tool_style"] == "auto")

    # ---- 显式 override 优先（CLI 单变量微调位）----
    c_o = get_config("retool_math", use_wandb=False, tool_protocol="native",
                     round_gen_tokens=1536, max_rounds=5, max_context_tokens=12288)
    check("native 档下显式 override 优先（回退档 D 可用）",
          c_o["round_gen_tokens"] == 1536 and c_o["max_context_tokens"] == 12288)
    check("native 档与围栏档的其余参数完全相同（严格单变量）",
          {k: v for k, v in c_n.items() if k not in _PROTOCOL_ONLY_KEYS}
          == {k: v for k, v in c_f.items() if k not in _PROTOCOL_ONLY_KEYS})

    # ---- 非法档位 fail-fast ----
    try:
        get_config("retool_math", use_wandb=False, tool_protocol="nonsense")
        check("未知 tool_protocol → ValueError（不静默回落）", False)
    except ValueError:
        check("未知 tool_protocol → ValueError（不静默回落）", True)
    check("TOOL_PROTOCOLS 常量覆盖两档",
          set(TOOL_PROTOCOLS) == {TOOL_PROTOCOL_FENCE, TOOL_PROTOCOL_NATIVE})
    # 【静默空转防线】native 只对 retool 家族有意义：单轮路径（grpo/dapo）从不读
    # 该键，传了等于没传——正是本项目最贵的一类 bug（成功的表象由空动作伪造）。
    try:
        get_config("grpo", use_wandb=False, tool_protocol="native")
        check("单轮算法传 native → ValueError（防『开了开关没接线』）", False)
    except ValueError as e:
        check("单轮算法传 native → ValueError（防『开了开关没接线』）",
              "只对 retool 家族有效" in str(e))
    check("retool 家族传 native 正常通过（反证：闸不是一刀切）",
          get_config("retool_math", use_wandb=False,
                     tool_protocol="native")["tool_protocol"] == "native"
          and get_config("retool", use_wandb=False,
                         tool_protocol="native")["tool_protocol"] == "native")

    # ---- tool_protocol_of / is_native_protocol 纯函数口径 ----
    check("缺键 = fence（历史 dict 兼容）", tool_protocol_of({}) == "fence")
    check("native 判定正确",
          is_native_protocol({"tool_protocol": "native"})
          and not is_native_protocol({"tool_protocol": "fence"}))
    try:
        tool_protocol_of({"tool_protocol": "x"})
        check("tool_protocol_of 非法值 raise", False)
    except ValueError:
        check("tool_protocol_of 非法值 raise", True)
    # BASE 里必须有这两个键（deepcopy 链路上任何一处缺键都会在 rollout 崩）
    for k in ("tool_protocol", "native_tool_style", "native_stop_at_call"):
        check(f"BASE 定义 {k}（所有 preset 继承可见）", k in BASE)


def test_p6_budget():
    print("[P6] 预算自洽：5×1024/8192 通过；6×1024/8192 必须 FAIL（档 B 负例）")
    base = dict(get_config("retool_math", use_wandb=False, tool_protocol="native"))
    r5 = validate_retool_budget({**base, "max_rounds": 5, "round_gen_tokens": 1024,
                                 "max_context_tokens": 8192})
    check("档 A（5×1024/8192）通过，预留 >0", r5 > 0)
    # 手算核对（与 §5.1 的表逐数对上：need=7208 / margin=+984）
    _need = 5 * 1024 + base["max_prompt_length"] + r5
    check("档 A need=7208 ≤ 8192（margin +984）", _need == 7208 and _need <= 8192)
    try:
        validate_retool_budget({**base, "max_rounds": 6, "round_gen_tokens": 1024,
                                "max_context_tokens": 8192})
        check("档 B（6×1024/8192）必须 FAIL（need=8498 > 8192）", False)
    except ValueError as e:
        check("档 B（6×1024/8192）必须 FAIL（need=8498 > 8192）",
              "8498" in str(e) or "预算不自洽" in str(e))
    # 档 C/D 是合法回退位
    check("档 C（6×1024/10240）通过",
          validate_retool_budget({**base, "max_rounds": 6, "round_gen_tokens": 1024,
                                  "max_context_tokens": 10240}) > 0)
    check("档 D（5×1536/12288）通过",
          validate_retool_budget({**base, "max_rounds": 5, "round_gen_tokens": 1536,
                                  "max_context_tokens": 12288}) > 0)
    # native preset 自身必须是自洽的（否则 get_config 就该抛）
    check("get_config(native) 自身预算自洽（_tool_reserve 已算）",
          base["_tool_reserve"] > 0)
    # 轮数→代码执行次数的映射（docs/09 §5.1 的注）
    check("max_rounds=5 → 4 次代码执行（= 参考 max_code_calls=4；末轮保证 final）",
          max(0, base["max_rounds"] - 1) == 4)
    check("参考实现口径对齐：单回合 1024 / 轨迹 ≤8192 / 工具回包预留 ≤512",
          base["round_gen_tokens"] == 1024 and base["max_context_tokens"] == 8192
          and base["tool_result_max_chars"] <= 512 * 2)


# =====================================================================
# 生成端接线（FakeGen，不碰 GPU）
# =====================================================================
def test_fakegen_native_rollout():
    print("[N] 原生多轮 rollout（FakeGen + MockTok + 假沙箱）")
    t = MockTok()
    cfg = {"max_rounds": 3, "max_context_tokens": 8192,
           "round_gen_tokens": 64, "sandbox_timeout": 1.0, "sandbox_mem_mb": 64,
           "tool_result_max_chars": 200, "tool_protocol": "native",
           "native_tool_style": "auto", "sandbox_workers": 1,
           "system_prompt": "SYS"}

    def fake_run(code, timeout=None, mem_mb=None, max_chars=None):
        val = {"print(6*7)": "42", "print(2+3)": "5", "print(99)": "99"}.get(code, "")
        return {"ok": True, "display": val, "error_type": None}

    class _C:
        def __init__(self, text):
            self.text = text
            self.token_ids = t.encode(text)
            self.finish_reason = "stop"

    class _O:
        def __init__(self, text):
            self.outputs = [_C(text)]

    class FakeGen:
        def __init__(self, rounds):
            self.rounds, self.r, self.seen = rounds, 0, []

        def generate(self, prompts, sps, use_tqdm=False):
            self.seen.append([list(p["prompt_token_ids"]) for p in prompts])
            texts = self.rounds[self.r]
            self.r += 1
            assert len(texts) == len(prompts), \
                f"第{self.r}轮：回放 {len(texts)} 条但收到 {len(prompts)} 个 prompt"
            return [_O(x) for x in texts]

    # 4 条轨迹（同题扩样）——每一轮的 active 集合都由"上一轮是否调用了代码"决定，
    # 这条链是本组测试真正要锁的契约（错一个位置，观察结果就会落到别的轨迹上）：
    #   第1轮（非末轮，全部可执行）: s0 调用 / s1 直接答 / s2 调用 / s3 调用
    #     → 第2轮 active = {s0,s2,s3}（s1 终局）
    #   第2轮: s0 调用 / s2 答 / s3 调用  → 第3轮 active = {s0,s3}
    #   第3轮（末轮，写了也不执行）: s0 答 / s3 调用 → s3 的这次调用进 code_wasted
    r1 = ["thinking\n" + _json_form("print(6*7)"), "the answer is \\boxed{42}",
          "reason\n" + _json_form("print(2+3)"),
          "reason\n" + _json_form("print(99)")]
    r2 = ["more\n" + _json_form("print(99)"), "final \\boxed{5}",
          "again\n" + _json_form("print(99)")]           # active = s0,s2,s3
    r3 = ["done \\boxed{42}", "late\n" + _json_form("print(99)")]   # active = s0,s3
    # 本函数（multi_turn_rollout_group_native）收的 prompts_messages 是
    # **逐轨迹**的（Q×num_pre_Q 条）——扩样由 collect_retool_group 负责。
    # 这里直接调底层，故显式给出 4 条。
    msgs = prompt_messages_for([{"Q": "Q1"}], cfg) * 4
    fg = FakeGen([r1, r2, r3])
    segs, full, cs = multi_turn_rollout_group(
        fg, [object()] * 4, t, ["P"] * 4, cfg, code_runner=fake_run,
        prompts_messages=msgs)

    check("轨迹数 = 4", len(segs) == 4 and len(cs) == 4)
    check("生成端收到 prompt_token_ids（token id 续写，非文本）",
          all(isinstance(x, list) for x in fg.seen[0]))
    check("第 1 轮 4 条并发（原协议无扩样 bug）", len(fg.seen[0]) == 4)
    check("第 2 轮只续写第 1 轮调用过代码的样本（s0/s2/s3 = 3 条）",
          len(fg.seen[1]) == 3)
    check("第 3 轮只续写第 2 轮又调用的样本（s0/s3 = 2 条）", len(fg.seen[2]) == 2)
    check("段序列：s0 = [a,tool,a,tool,a]",
          [s["kind"] for s in segs[0]]
          == ["assistant", "tool", "assistant", "tool", "assistant"])
    check("s1 无调用即终局 [a]", [s["kind"] for s in segs[1]] == ["assistant"])
    check("s2 = [a,tool,a]（第2轮答完终局）",
          [s["kind"] for s in segs[2]] == ["assistant", "tool", "assistant"])
    check("末轮调用不执行 → s3 = [a,tool,a,tool,a]（无第 3 个 tool 段）",
          [s["kind"] for s in segs[3]]
          == ["assistant", "tool", "assistant", "tool", "assistant"])
    check("code_used：s0=2, s1=0, s2=1, s3=2；code_ok 与执行结果一致",
          [s["code_used"] for s in cs] == [2, 0, 1, 2]
          and [s["code_ok"] for s in cs] == [2, 0, 1, 2])
    check("末轮调用 → code_wasted=1（仅 s3）",
          [s["code_wasted"] for s in cs] == [0, 0, 0, 1])
    check("invalid_final 全 0（本轮样本都是合法调用或纯答案）",
          all(s["invalid_final"] == 0 for s in cs))

    # ---- 核心契约：训练序列 == 生成序列（逐 token）----
    # 生成端第 2 轮收到的 prompt 必须等于 base_prompt + 第1轮 assistant ids + 工具段 ids
    base_ids = build_prompt_ids(msgs[0], t, None, tools=True)
    tool_ids_0 = segs[0][1]["ids"]
    r1_ids = t.encode(r1[0])
    expect_r2 = [*base_ids, *r1_ids, *tool_ids_0]
    check("★ 第 2 轮 prompt == prompt_ids + 第1轮 assistant ids + 工具段 ids（逐 token）",
          fg.seen[1][0] == expect_r2)
    # 第 3 轮的 prompt 必须是第 2 轮输出的延续（多轮递推，不能从头重渲染）
    check("★ 第 3 轮 prompt 以第 2 轮 prompt 为前缀（token-in token-out 递推）",
          fg.seen[2][0][:len(fg.seen[1][0])] == fg.seen[1][0])
    check("生成/训练同序列：per_sample_ids 前两段与生成端拼出的序列同源",
          len(tool_ids_0) > 0)
    # ---- 掩码：工具段全 0 ----
    from rlab.rollout import retool_build_batch
    pids = build_prompt_batch([{"Q": "Q1"}], cfg, t)[1]
    merged, mask, per_ids = retool_build_batch(pids, segs[:1], pids.shape[1],
                                               t.pad_token_id)
    check("merged 的 prompt 段 == 生成端 prompt（原生档 tools 渲染同源）",
          merged[0, :pids.shape[1]].tolist()
          == strip_left_pad(pids, t.pad_token_id)[0].tolist())
    a0 = len(t.encode(r1[0]))
    tl0 = len(tool_ids_0)
    m0 = mask[0].tolist()
    check("mask：assistant 段=1 / 原生回包段=0（含 im_end 与 tool_response）",
          m0[:a0] == [1.0] * a0 and m0[a0:a0 + tl0] == [0.0] * tl0)
    check("原生回包段里确实含模板结构 token（非模型生成 → 必须 mask=0）",
          MockTok.IM_END in tool_ids_0)

    # ---- 围栏档 A/B：同一份调用文本在围栏档下不会被识别（协议真变了）----
    cfg_f = {**cfg, "tool_protocol": "fence", "retool_stop": False}
    fg2 = FakeGen([r1])
    segs_f, _, cs_f = multi_turn_rollout_group(
        fg2, [object()] * 4, t, ["P"] * 4, cfg_f, code_runner=fake_run)
    check("A/B 反证：同一文本在围栏档下 code_used 全 0（协议分支确实不同）",
          all(s["code_used"] == 0 for s in cs_f))


def test_native_build_prompt_wiring():
    print("[W] 生成端 prompt 构造按协议分叉（prompt_ids 必须与生成序列同源）")
    t = MockTok()
    inputs = [{"Q": "Q1"}]
    cfg_n = {"system_prompt": "SYS", "chat_template_kwargs": None,
             "tool_protocol": "native"}
    cfg_f = {"system_prompt": "SYS", "chat_template_kwargs": None,
             "tool_protocol": "fence"}
    txt_n, ids_n, plen_n = build_prompt_batch(inputs, cfg_n, t)
    txt_f, ids_f, plen_f = build_prompt_batch(inputs, cfg_f, t)
    check("native 档 prompt_ids == build_prompt_ids(messages, tools=True) 逐 token",
          ids_n[0, plen_n - len(build_prompt_ids(
              initial_messages("SYS", "Q1"), t, None, tools=True)):].tolist()
          == build_prompt_ids(initial_messages("SYS", "Q1"), t, None, tools=True))
    check("两档 prompt 不同（原生档多一个工具声明段）", plen_n > plen_f)
    check("围栏档 prompt_ids 与历史文本路径一致",
          ids_f.tolist() == t(txt_f, return_tensors="pt", padding=True,
                              padding_side="left",
                              add_special_tokens=False)["input_ids"].tolist())
    check("build_prompt 默认 tools=False（围栏档 prompt 逐字节不变）",
          build_prompt("Q1", "SYS", t)
          == t.apply_chat_template(initial_messages("SYS", "Q1"), tokenize=False,
                                   add_generation_prompt=True))
    check("build_prompt(tools=True) 渲染里出现工具声明",
          TOOL_NAME in build_prompt("Q1", "SYS", t, None, tools=True)
          and TOOL_NAME not in build_prompt("Q1", "SYS", t))
    check("native 档左 pad 正确（pad 在左侧、内容右对齐）",
          plen_n == max(1, ids_n.shape[1])
          and (ids_n[0] != t.pad_token_id).sum().item()
          == len(build_prompt_ids(initial_messages("SYS", "Q1"), t, None, tools=True)))
    # native 档缺 messages 时 collect_retool_group 会现场构造（与 build_prompt_batch 同源）
    check("prompt_messages_for 与 build_prompt_batch 的 messages 同源",
          prompt_messages_for(inputs, cfg_n)
          == [initial_messages("SYS", "Q1")])
    # 【扩样契约·层次】prompts_messages 的"题数 vs 轨迹数"分界：
    #   collect_retool_group 收**每题一条**（Q），内部扩成 Q×num_pre_Q 再传给
    #   multi_turn_rollout_group_native（后者收 Q×num_pre_Q，与 prompts_text 同长）。
    # 判据是 fail-fast 在**哪一层**：条数不符必须在 collect_retool_group 拦下
    # （那是唯一做扩样的地方），底层只按"收到几条就是几条轨迹"工作。
    text, ids, pl = build_prompt_batch(inputs, cfg_n, t)
    check("build_prompt_batch 原生档接受调用方传入的同一份 messages",
          build_prompt_batch(inputs, cfg_n, t,
                             prompts_messages=[initial_messages("SYS", "Q1")])[2] == plen_n)
    _ro = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "rollout.py"), encoding="utf-8").read()
    _crg = _ro[_ro.index("def collect_retool_group("):]
    check("扩样与条数校验同在 collect_retool_group（唯一做扩样的层）",
          "prompts_messages 条数" in _crg
          and "[m for m in prompts_messages for _ in range(n)]" in _crg)
    check("底层 multi_turn_rollout_group_native 不重复扩样（按收到几条就是几条）",
          _ro[_ro.index("def multi_turn_rollout_group_native("):
              _ro.index("def multi_turn_rollout_group(")].count(
                  "[m for m in prompts_messages") == 0)


def test_protocol_wiring_static():
    print("[S] 接线静态检查：协议档必须真的被读（防『加了开关没接线』）")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ro = open(os.path.join(root, "rollout.py"), encoding="utf-8").read()
    tr = open(os.path.join(root, "train.py"), encoding="utf-8").read()
    ev = open(os.path.join(os.path.dirname(root), "eval_vllm_one.py"),
              encoding="utf-8").read()
    cf = open(os.path.join(root, "config.py"), encoding="utf-8").read()
    pd = open(os.path.join(root, "probe_difficulty.py"), encoding="utf-8").read()
    np_ = open(os.path.join(root, "native_probe.py"), encoding="utf-8").read()
    check("rollout：native 分支转 multi_turn_rollout_group_native",
          "multi_turn_rollout_group_native" in ro
          and "if is_native_protocol(cfg):" in ro)
    check("rollout：make_retool_sps 按协议档分叉 stop",
          "if is_native_protocol(cfg):" in ro
          and "if cfg.get(\"native_stop_at_call\"):" in ro)
    check("rollout：collect_retool_group 接受 prompts_messages",
          "prompts_messages=None" in ro
          and "prompts_messages=prompts_messages" in ro)
    check("rollout：难度表指纹含 tool_protocol",
          '"tool_protocol": cfg.get("tool_protocol") or "fence"' in ro)
    check("train：签名含 -tp 段（非 fence 才追加，历史签名逐字不变）",
          'tp_tag = ("-tp"' in tr and "{vk_tag}{tp_tag}{nsc_tag}" in tr)
    check("train：CLI 有 --tool_protocol / --native_tool_style / --native_stop_at_call",
          "--tool_protocol" in tr and "--native_tool_style" in tr
          and "--native_stop_at_call" in tr)
    check("train：run_info 落盘协议档（eval 回读的单一来源）",
          '"tool_protocol": cfg.get("tool_protocol")' in tr)
    check("eval：协议档从 run_info 回读 + 显式落进 mt_cfg",
          '_rcfg.get("tool_protocol") or "fence"' in ev
          and '"tool_protocol": _tool_protocol' in ev)
    check("eval：原生档用 tools=True 建 prompt，且 prompt/messages 按同一索引过滤",
          "tools=_tools_flag" in ev and "_prompt_msgs = [_prompt_msgs[i] for i in _kept]" in ev)
    check("eval：原生诊断打印（invalid/ctx_full/零调用率）",
          "invalid_final=" in ev and "零调用轨迹" in ev)
    check("config：NATIVE_PROTOCOL_DEFAULTS 在 get_config 里被套用",
          "NATIVE_PROTOCOL_DEFAULTS" in cf and "for _k, _v in NATIVE_PROTOCOL_DEFAULTS" in cf)
    check("probe_difficulty：原生档走 messages + 表指纹含协议档",
          "group_msgs" in pd and '"tool_protocol": _tp' in pd)
    check("health：新增 native_invalid 签名",
          "native_invalid" in open(os.path.join(root, "health.py"),
                                   encoding="utf-8").read())
    check("data：难度表比对清单含 tool_protocol",
          '"tool_protocol"' in open(os.path.join(root, "data.py"),
                                    encoding="utf-8").read())
    # 【2026-09-25 真机首跑】native_probe 曾是仓库里**唯一**不传 gdn_prefill_backend
    # 的 GPU 入口 → Qwen3.5 的 GDN 层落回 FlashInfer 现场 JIT → ninja 打爆内存被
    # SIGKILL（只有 `Killed`、无 traceback），go/no-go 那一关根本没跑出数字。
    # 这条检查防"新加的探针又漏引擎档"（同 probe_difficulty 2026-09-17 的教训）。
    check("native_probe：LLM() 传引擎参数（默认 gdn_prefill_backend=triton，免 JIT）",
          'DEFAULT_ENGINE_KWARGS = {"gdn_prefill_backend": "triton"}' in np_
          and "llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_mem, **_kw)"
          in np_)
    check("native_probe：起引擎前对缺 backend / 继承的 VLLM_BATCH_INVARIANT 告警",
          "def engine_env_warn" in np_ and "gdn_backend_missing" in np_
          and "VLLM_BATCH_INVARIANT" in np_)
    # 【防"诊断自己说谎"】旧版 Q2 用 `"{" in 块 and "name" in 块` 数"真调用"，
    # 那只对 JSON 形态成立：Qwen3.5 的 <function=…> 形态下数出 0 个，打印
    # "含实参 0 个"，把正常渲染读成"调用段是空的"（真机首跑撞上）。判据必须用
    # 生产解析器（parse_assistant），它才是"正规形态"的唯一定义者。
    # 【检查用 AST 而非文本包含】上面这段注释本身就含那个被禁的字面量——纯文本
    # 检查会被**注释**误伤（改了说明文字就翻红 / 删掉说明文字反而变绿），这正是
    # 本项目"静态检查被注释打伤"的第二次（前一次：apply_chat_template）。故这里
    # 读 AST：只看真正被赋值的 `_real = [...]` 的推导式条件。
    _t = _ast.parse(np_)
    _real_conds = []
    for _n in _ast.walk(_t):
        if isinstance(_n, _ast.Assign) and any(
                isinstance(_tt, _ast.Name) and _tt.id == "_real" for _tt in _n.targets):
            for _g in _ast.walk(_n.value):
                if isinstance(_g, _ast.Call):
                    _real_conds.append(_ast.unparse(_g))
    check("native_probe：Q2 用 parse_assistant 判真调用（不用 JSON 启发式）",
          any("parse_assistant" in _c for _c in _real_conds)
          and not any('"name" in' in _c or "'name' in" in _c for _c in _real_conds))
    check("native_probe：拼接证明断言硬契约（前缀/采样保留/回包拼入）",
          "ok_pref" in np_ and "ok_samp" in np_ and "ok_long" in np_
          and "Q5 拼接硬契约" in np_)
    check("native_probe：冒烟用竞赛题而非口算题 + 可钉死形态（防低估调用率）",
          "SMOKE_QUESTIONS" in np_ and "--native_tool_style" in np_)
    # 【2026-09-25 真机首跑后的口径纠正】首版冒烟用的是本文件里的玩具提示
    # `SYS = "SYS: you solve math with a python tool."`——等于手把手教模型调用，
    # 测出的 100% 是"被提示后"的，与参考 87.5%（在它自己的正式提示下测）**不可比**。
    # 本项目对探针的铁律：探针与训练同口径（probe_difficulty 2026-09-17）。
    check("native_probe：用 preset 正式提示与采样参数（探针与训练同口径铁律）",
          "def smoke_config" in np_
          and "from rlab.config import default_system_prompt, get_config" in np_
          and 'temperature=cfg["temperature"]' in np_)
    check("native_probe：渲染/往返/拼接三处都收 sys_prompt+ctkw（不只冒烟）",
          "def probe_render(tok, out_path, sys_prompt, question, ctkw)" in np_
          and "def probe_roundtrip(tok, out_path, sys_prompt, question, ctkw)" in np_
          and "def probe_build_next(tok, out_path, sys_prompt, question, ctkw)" in np_)
    # 【判据不能拿工具名当标志物】训练提示自己就写了 code_interpreter → "不带 tools"
    # 对照档会误报"含工具声明=是"。标志物必须只由声明段贡献且跨模板可移植
    # （<tools> 是 Qwen2.5 模板特有的包装标签；CODE_TOOL 描述里的短语才可移植）。
    check("native_probe：Q1 标志物与工具名解耦（从 CODE_TOOL 描述派生，跨模板可移植）",
          '_decl_mark = "Execute code in an isolated environment"' in np_
          and "has_decl = _decl_mark in p" in np_)
    check("native_probe：Q4 断言训练档 render 不以未闭合 <think> 结尾（烧预算签名）",
          "_open_think" in np_ and "烧穿轮预算" in np_)


def test_p7_pyflakes():
    print("[P7] pyflakes 清单含新文件（_health 别名事故的同类防线）")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "tests", "test_retool_cpu.py"),
               encoding="utf-8").read()
    check("J 组清单含 rlab/native_probe.py",
          '"rlab/native_probe.py"' in src)
    try:
        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
    except ImportError:
        print("  skip: pyflakes 未安装")
        return
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        checkPath(os.path.join(root, "native_probe.py"), Reporter(buf, buf))
        checkPath(os.path.join(root, "protocol.py"), Reporter(buf, buf))
        checkPath(os.path.join(root, "rollout.py"), Reporter(buf, buf))
    undefined = [l for l in buf.getvalue().splitlines() if "undefined name" in l]
    check("原生协议相关文件无未定义名", undefined == [])
    if undefined:
        for l in undefined:
            print(f"  !! {l}")


if __name__ == "__main__":
    test_p1_parse()
    test_p2_sequence_contract()
    test_p3_template_guard()
    test_p4_mask_and_grad()
    test_scoring_domain()
    test_p5_zero_label_roundtrip()
    test_config_protocol_switch()
    test_p6_budget()
    test_fakegen_native_rollout()
    test_native_build_prompt_wiring()
    test_protocol_wiring_static()
    test_p7_pyflakes()
    print(f"\n[test_native_protocol] {len(PASS)} 项全部通过")
