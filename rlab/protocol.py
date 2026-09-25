# -*- coding: utf-8 -*-
"""rlab/protocol.py — 训练端↔生成端 的 batch 契约与字节编解码。

字节流格式（沿用本项目 ref_server 协议，bytes_list）：
  [0] json meta   : {"plen": int, "num_items_in_batch": int(可选), ...}
  [1] merged_ids  : (B, plen+T) prompt + 右 pad completion
                    （阶段0/1 单轮批的 prompt 段含左 pad；阶段2 retool 逐题用
                    rollout.strip_left_pad 剥掉左 pad 后建批，plen=本题真实 prompt
                    长——左 pad 会同时污染注意力键与位置编码，见 docs/02）
  [2] advantages  : (B,) 已在生成端按 adv_mode 归一化（或 (B,T) per-token，RF++）
  [3] refs        : (B, T) ref 模型 per-token logps（ref_server 补充）
  [4] gen_logps   : (B, T) 生成时 policy 的 per-token logps（torch 副本算）
  [5] acc_scores  : (B,) 正确性原始分（仅记录/监控用，不进 loss）
  [6] format_scores: (B,) 格式原始分（同上）

mask 约定：
- 阶段0/1（单轮）：completion 区 pad 位由训练端 inputs!=pad 重算，协议不传 mask。
- 阶段2（retool 多段工具轨迹）：meta 带 "has_mask":1，extras 首槽为 (B,T) 0/1 完成掩码
  （assistant token=1 / 工具返回段=0 / pad=0）。训练端直接采用，不再自行重算——
  工具返回 token 不进 loss 是 TIR 的核心契约，必须由生成端按段边界精确给出。
"""

import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------- 阶段2 常量 ----
# 工具段起止标记：用纯文本方括号，不用尖括号/think/response 字节
# （零标签字面量铁律——含标签字节的文本经聊天管道会被改写成普通英文单词）。
# 多段轨迹里 assistant 生成的 token 进 loss，工具返回段（TOOL_START..TOOL_END）
# 只作上下文、mask 置 0 不进 loss——这是 TIR 训练最易错的点（见 tests/test_retool_cpu.py）。
TOOL_START = "\n[TOOL RESULT]\n"
TOOL_END = "\n[/TOOL RESULT]"

# python 围栏代码块提取（```python ... ```，DOTALL 跨行）
_PY_FENCE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

# 特殊 token 字面量（<|im_end|> / <|endoftext|> 等，Qwen 系通用形态）
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")

# 【2026-09-18 stop 机制·工具调用节奏的核心修复】模型写到**代码块闭合围栏**立即
# 停止生成，沙箱结果紧跟代码回填——修复"代码→自己瞎猜→[真结果]"的错位。
# 无 stop 时（p5/p6 三轮 run）：一段生成写满 max_tokens 才结束，代码块被事后正则
# 提取、TOOL_RESULT 拼在**整段末尾**——模型先猜了结果才看到真结果，"调用工具获取
# 信息"的因果链断裂；同时每段烧满预算 → 末段 finish_reason=length → trunc 42~55%
# → 无 boxed → reward -1，代码路径结构性负 advantage 被 RL 压灭（code% 50→3）。
# 参考实现（Auto_Program/hjy_grpo_program.py:151）同款机制：stop 句 + include_stop。
# stop 串选 "```\n"（闭围栏+换行）而非 "```"：经 chr 验证 "```python\n" **不含**
# "```\n" 子串（"```" 后面跟的是 "python"），开围栏不会误停；闭围栏 "```\n" 命中后
# include_stop_str_in_output=True 保留围栏字节，extract_python_blocks 仍能拿到完整
# ```python...``` 块。附带收益：每段最多一个代码块（blocks[-1] 白写问题自然消解）。
RETOOL_STOP_KWARGS = {"stop": ["```\n"], "include_stop_str_in_output": True}


# =====================================================================
# 原生工具协议（docs/09-native-tool-protocol.md · 方案 A）
# =====================================================================
# 与上面的"自造围栏协议"并列存在，由 `cfg["tool_protocol"]` 二选一
# （"fence" = p1–p11 逐位可复现；"native" = 本方案）。
#
# 【为什么不复用围栏】围栏协议是**从零教**的格式（base 无先验、需 SFT），
# 原生 `<tool_call>` 是 Qwen 系 chat template 自带的（参考实现 base 调用率
# 87.5%、跳过 cold-start SFT 直接 RL 拿到 +23.89pp）。docs/09 §0 三方对照。

CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        # 描述参照参考实现（官方复现指南里实测"能涨点"的写法）：
        # 强调 print 输出、每次执行独立无状态。
        "description": (
            "A Python code execution environment that allows you to:\n"
            "- Run Python code for calculations, data analysis, and other computational tasks\n"
            "- Get results through the `print()` function output\n"
            "- Execute code in an isolated environment (each execution starts fresh)\n\n"
            "Important notes:\n"
            "- Results are captured from `print()` statements\n"
            "- Returns empty string if no output is printed\n"
            "- Each execution is independent (no state persistence between runs)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to be executed.",
                }
            },
            "required": ["code"],
        },
    },
}

TOOL_NAME = "code_interpreter"

# 【关键·不要照抄参考的正则】Qwen 世代的 tools 渲染形态**不同，且不能互推**：
# 本机实测（D:\tmp\qwen_tok_audit 的 Qwen2.5-3B tokenizer，transformers 4.41.2）
# Qwen2.5 渲染 JSON 形态；参考实现的 TOOL_CALL_PATTERN 吃的是 Qwen3.5 的
# `<function=…>/<parameter=…>` 形态。故这里**两种都实现**，运行期由
# `derive_tool_style()` 从**实测渲染结果**派生该用哪种，而不是猜。
NATIVE_STYLE_FUNCTION = "function"
NATIVE_STYLE_JSON = "json"
NATIVE_STYLES = (NATIVE_STYLE_FUNCTION, NATIVE_STYLE_JSON)

# Qwen3.x 形态（参考实现口径）
_RE_TOOL_CALL_FUNCTION = re.compile(
    r"<tool_call>\s*<function=" + TOOL_NAME + r">\s*<parameter=code>\s*(.*?)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)
# Qwen2.5 形态：<tool_call>{"name": ..., "arguments": {...}}</tool_call>
# 【为什么用贪婪 .* 而不是 .*?】非贪婪会在 code 字符串里第一个 `}` 处收口
# （如 "print({})"），json.loads 随即失败 → 合法调用被误判 invalid。贪婪匹配
# 回溯到 `</tool_call>` 前的**最后**一个 `}`，才是真正的对象边界；块内不合法
# 的情况由 json.loads 兜住（宁判 invalid，不猜）。
_RE_TOOL_CALL_JSON = re.compile(r"<tool_call>\s*(\{.*\})\s*</tool_call>", re.DOTALL)
# 形态探测用：任意 <tool_call> 块（用于区分"有调用但形态不认识"vs"根本没有调用"）
_RE_TOOL_CALL_ANY = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
# 调用块的起止标记**从正则本身派生**，全模块只此一处字面量（零标签字面量铁律：
# 测试串与解析器必须同源；手写第二份就等于埋一个"改了一处忘另一处"的坑）。
_TOOL_OPEN, _TOOL_CLOSE = _RE_TOOL_CALL_ANY.pattern.split(".*?")

# 工具调用块的收尾标记。仅当 cfg["native_stop_at_call"]=True 时作为 vLLM stop 串
# 使用（默认 False = 参考实现的原生节奏：靠模型自己吐 im_end 收尾）。
# 【为什么留这个开关】若 base 冒烟看到 text[调用之后还有内容] → parse 判 invalid
# → 整条轨迹终局无 boxed（reward -1），这是原生协议唯一的"结构性负奖励"入口；
# 打开它 = 用 stop 串在调用边界硬停（等价于围栏路径 p7 的 stop 机制）。
# 字面量**从正则派生**（零标签字面量铁律：全模块只此一处真源）。
NATIVE_CALL_STOP = _TOOL_CLOSE


@dataclass(frozen=True)
class ParsedAssistant:
    """一次 assistant 输出的协议解析结果（与参考实现同构）。

    kind: "tool"（要执行代码）/ "answer"（终局）/ "invalid"（有调用但形态非法）。
    content: 调用前的推理文本，或完整答案/非法输出。
    code: kind=="tool" 时的待执行 Python 代码。
    style: 命中的形态名（供日志/诊断；未命中为 None）。
    """

    kind: str
    content: str
    code: str | None = None
    style: str | None = None


def _code_from_json_call(blob: str):
    """Qwen2.5 JSON 形态：{"name": ..., "arguments": {"code": ...}} → code。

    arguments 在模板里既可能是 dict、也可能是 JSON 字符串（自造消息进模板时
    常见），两种都吃；name 不匹配 / 没有 code / JSON 坏 → None（=调用非法）。"""
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, Mapping):
        return None
    if str(obj.get("name") or obj.get("function") or "") != TOOL_NAME:
        return None
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return None
    if not isinstance(args, Mapping):
        return None
    code = args.get("code")
    return code if isinstance(code, str) else None


def _find_call(text: str, style: str):
    """按指定形态找**唯一**一个完整调用，返回 (start, end, code)；不合法 → None。

    【合法性判据是**形态无关**的，分两层】
      ① 结构层（本函数）：整段文本里恰好一个完整调用块、其后不再有其他内容、
         代码非空。跨形态的"两个调用"（一个 JSON 一个 function）也必须被拒——
         否则 "auto" 逐形态试时，JSON 形态只会看到 1 个 JSON 块而放行，同一段
         文本在两种 style 下得到不同结论（首版正是这么错的：一条样本在 auto 下
         被当合法调用、在 function 钉死档被判 invalid，训练/评测结论分叉）。
      ② 形态层：块内必须是本形态的合法载荷（JSON 能解析出 code / function 能匹配）。
    """
    blocks = list(_RE_TOOL_CALL_ANY.finditer(text))
    if len(blocks) != 1 or text[blocks[0].end():].strip():
        return None                      # 多层结构非法：与风格无关
    b0, b1 = blocks[0].start(), blocks[0].end()
    if style == NATIVE_STYLE_FUNCTION:
        ms = list(_RE_TOOL_CALL_FUNCTION.finditer(text))
        if len(ms) != 1:
            return None
        code = ms[0].group(1).strip()
        return (b0, b1, code) if code else None
    if style == NATIVE_STYLE_JSON:
        ms = list(_RE_TOOL_CALL_JSON.finditer(text))
        if len(ms) != 1:
            return None
        code = _code_from_json_call(ms[0].group(1))
        code = code.strip() if isinstance(code, str) else ""
        return (b0, b1, code) if code else None
    raise KeyError(f"未知 tools 形态 {style!r}，可选 {NATIVE_STYLES}")


def parse_assistant(text: str, style: str = "auto") -> ParsedAssistant:
    """把 assistant 文本识别成代码调用 / 最终回答 / 非法输出。

    style: "auto"（逐形态试，先 function 后 json）或 NATIVE_STYLES 之一。
    解析失败**不猜**，按"有没有动过调用标记"二分：
      · 文本里出现**开标记**（含未闭合的）→ "invalid"。这是刻意的：模型开始写
        调用但没写完整（被轮长切断 / 形态错 / 参数坏）绝不能被当成终局答案——
        那会让"残缺调用"拿到"无 boxed → -1"之外的另一条路径（当答案去判分），
        并掩盖协议失效信号（docs/09 §6.2 的判据就靠 invalid 率）。
        与围栏协议的教训同型：未闭合围栏 `extract_python_blocks` 返回空 →
        结构不合格，而不是"没有代码"。
      · 完全没有调用标记 → "answer"（终局，去打分）。"""
    text = text or ""
    styles = NATIVE_STYLES if style == "auto" else (style,)
    for st in styles:
        hit = _find_call(text, st)
        if hit is not None:
            return ParsedAssistant(kind="tool", content=text[:hit[0]].strip(),
                                   code=hit[2], style=st)
    return ParsedAssistant(
        kind="invalid" if _TOOL_OPEN in text else "answer",
        content=text.strip(), style=None)


def derive_tool_style(rendered: str) -> str:
    """从**实测渲染结果**派生该用哪种解析形态（docs/09 §1.2 的 Q2）。

    输入 = 模板渲染出的、含一次 code_interpreter 调用的文本（见
    rlab/native_probe.py）。出不来就 raise——静默回落到某个形态会让
    "解析器与模板不同源"这一整类 bug 重新长回来（本项目被标签管道改写坑过三次）。
    """
    for st in NATIVE_STYLES:
        if _find_call(rendered, st) is not None:
            return st
    raise ValueError(
        "[protocol] 实测渲染里找不到可解析的 code_interpreter 调用——"
        f"两种形态都试过（{NATIVE_STYLES}）。\n"
        f"  渲染片段（前 800 字符）：{rendered[:800]!r}\n"
        f"  处置：看 Q2 的实际形态，把新形态加进 NATIVE_STYLES/_find_call"
        f"（不要改现有正则去凑）。")


def tool_message(call_id: str, content: str) -> dict:
    """构造一条结构化工具结果消息（role:"tool"，与参考实现同构）。

    content 必须已经 sanitize_tool_text 消毒（沙箱 stdout 是模型经 print
    间接可控的通道——注入面不因协议从围栏换成原生而消失，消毒点不变）。"""
    return {"role": "tool", "tool_call_id": str(call_id),
            "name": TOOL_NAME, "content": content}


def make_call_id(question_index: int, group_index: int, call_no: int) -> str:
    """调用 id（参考实现同构）。唯一性由 (题, 组内轨迹, 第几次) 三元组保证——
    回包按 id 归属，错位会让 observation 落到别的轨迹上下文里。"""
    return f"code-{int(question_index)}-{int(group_index)}-{int(call_no)}"


def initial_messages(system_prompt: str, question: str) -> list:
    """原生协议下的初始消息（system + user）。"""
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": question}]


def render_chat_ids(tokenizer, messages: list, add_generation_prompt: bool,
                    chat_template_kwargs: dict | None = None,
                    tools: bool = True) -> list:
    """apply_chat_template(tokenize=True) → 一维 int 列表（跨 tokenizer 返回形态归一）。

    不同 transformers 版本返回 list / tensor / {"input_ids": ...} / 嵌套，
    统一收口在这里（参考实现的 _render_chat 同款）。"""
    kw = dict(chat_template_kwargs or {})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
        **({"tools": [CODE_TOOL]} if tools else {}), **kw)
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(t) for t in rendered]


def encoded_text_tokens(tokenizer, text: str) -> list:
    """纯文本编码 → 一维 int 列表（占位内容 token 用）。"""
    enc = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(enc, "tolist"):
        enc = enc.tolist()
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return [int(t) for t in enc]


def suffix_prefix_overlap(tokens: list, suffix: list) -> int:
    """tokens 末尾与 suffix 开头的最长重叠长度（纯函数，CPU 可测）。

    vLLM 有时会把 `im_end` 一起采样出来（EOS 在 token_ids 里，docs/05 §12.3），
    此时结束符已经在了——只补差额，否则重复拼一次 im_end 会污染序列。"""
    if not suffix:
        return 0
    for n in range(min(len(tokens), len(suffix)), 0, -1):
        if tokens[-n:] == suffix[:n]:
            return n
    return 0


# 纯空白 token 的判别缓存（token id → bool）。整段 decode 在热路径上太贵
# （每段都可能命中：8 样本 × 4 轮 × 每步），单 token decode 便宜且可缓存。
_WS_TOKEN_CACHE: dict = {}


def _drop_ws_tokens(tokenizer, ids):
    """去掉**纯空白 token**；解码失败 → 返回 None（调用方退回严格比较，宁严不宽）。

    【为什么判据是"单 token 解码后 strip 为空"】Qwen 是 byte-level BPE，ASCII 空白
    （空格/换行）都是单 token 且单独解码干净；多字节字符被切开时单 token 解码成
    替换符（非空白）→ **保守保留**。故该判据只会"漏容忍"、不会"错容忍"——正是
    要的方向（宁可多 raise 一次，不可静默拼出错位序列）。

    【本函数只服务校验，不参与构造】序列永远由原始 token 拼出（token-in token-out），
    绝不拿这里的结果去构造任何东西。见 build_next_prompt 校验① 的事故说明。"""
    out = []
    for t in ids:
        t = int(t)
        hit = _WS_TOKEN_CACHE.get(t)
        if hit is None:
            try:
                s = tokenizer.decode([t], skip_special_tokens=False)
            except Exception:
                return None
            if not isinstance(s, str):
                return None
            hit = (s.strip() == "")
            _WS_TOKEN_CACHE[t] = hit
        if not hit:
            out.append(t)
    return out


def _decode_span(tokenizer, ids, start, width):
    """把分歧位附近解码成文本，供报错信息直接显示（数字看不出"错在哪"）。"""
    lo = max(0, int(start) - 4)
    hi = min(len(ids), int(start) + int(width))
    try:
        return tokenizer.decode([int(t) for t in ids[lo:hi]], skip_special_tokens=False)
    except Exception:
        return repr([int(t) for t in ids[lo:hi]])


# 容忍只告警一次（否则每个样本每轮都刷一行，训练日志被淹没）
_WS_TOLERANCE_WARNED = False


def build_next_prompt(tokenizer, messages_before_assistant: list,
                      previous_prompt_tokens: list, completion_tokens: list,
                      next_tool_message: dict,
                      chat_template_kwargs: dict | None = None) -> list:
    """token-in token-out 续写序列（docs/09 §3，本方案最易错处）。

    返回 = previous_prompt_tokens + completion_tokens
           + assistant 结束符(只补 overlap 之外的差额) + tool observation 增量。

    【为什么不能重渲染整段历史】官方实测 text↔token 重编码不可逆 → 约 100 步后
    性能崩塌、grad_norm NaN；且生成/训练必须是**逐 token 同一条序列**（本项目
    围栏路径已为此改成 token id 续写，见 multi_turn_rollout_group docstring）。

    【为什么用占位内容】Qwen 的 chat template 会 strip assistant 内容，采样文本
    含 `</think>` 时还会被重构成 reasoning/content 两段——用真实文本定位结束边界
    必然失败。增量片段只与 tool 消息有关、与 assistant 内容无关，故 canonical
    计算一律用占位 "x"。
    **"增量与 assistant 内容无关"已实测**（9 种内容变体——空/首部换行/首尾空格/
    普通叙述——算出的 observation 逐 token 完全相同），这正是占位法成立的根据。

    三处 fail-fast（异常当场抛，不静默产出错位序列）：
      ① canonical_prompt 与 prev 的**可见内容**不同（模板 tokenize 路径与生成端
         喂给 vLLM 的 ids 不同源；空白差异容忍——见 _visible_text 的事故说明）；
      ② 占位 assistant 拼接后前缀不等（模板改写了历史）；
      ③ 加入 tool 消息后前缀不等（observation 渲染污染了历史）。"""
    global _WS_TOLERANCE_WARNED
    prev = [int(t) for t in previous_prompt_tokens]
    canonical_prompt = render_chat_ids(tokenizer, messages_before_assistant, True,
                                       chat_template_kwargs)
    if canonical_prompt != prev:
        # 【两级判据】先看是否**只是空白归一化差异**（模板 strip vs 原样采样）；
        # 是 → 容忍（否则"某段以换行开头"就会炸掉整轮训练）；否 → 真不同源，抛。
        _cv = _drop_ws_tokens(tokenizer, canonical_prompt)
        _pv = _drop_ws_tokens(tokenizer, prev)
        if _cv is not None and _cv == _pv:
            if not _WS_TOLERANCE_WARNED:
                _WS_TOLERANCE_WARNED = True
                print(f"[protocol] 模板渲染与拼接上下文差 {abs(len(canonical_prompt) - len(prev))} "
                      f"个 token，但**去掉空白 token 后完全相同**（模板对 assistant 内容做 "
                      f"strip，拼接保留原样采样 token）→ 判为归一化差异、继续。"
                      f"只报这一次。", flush=True)
        else:
            _n = min(len(canonical_prompt), len(prev))
            _i = next((k for k in range(_n) if canonical_prompt[k] != prev[k]), _n)
            _ctx = 60
            raise ValueError(
                "[protocol] 模板渲染的 prompt 与生成端实际喂给 vLLM 的 token 不一致"
                f"（长度 {len(canonical_prompt)} vs {len(prev)}，首个分歧位 {_i}）；"
                f"且差异**不只是空白**（可见内容不同）→ 真的不同源。\n"
                f"  模板侧 [{_i}:{_i + 8}] = {[int(t) for t in canonical_prompt[_i:_i + 8]]}\n"
                f"  拼接侧 [{_i}:{_i + 8}] = {[int(t) for t in prev[_i:_i + 8]]}\n"
                f"  模板侧文本 …{_decode_span(tokenizer, canonical_prompt, _i, _ctx)!r}\n"
                f"  拼接侧文本 …{_decode_span(tokenizer, prev, _i, _ctx)!r}\n"
                "  后果：续写序列与采样序列不同源 → gen_logps 基线失真、训练/生成分布分叉。\n"
                "  处置：检查 build_prompt 的 tools/chat_template_kwargs 是否与 "
                "apply_chat_template 同参（tools 必须两边都传或都不传）。")
    placeholder = {"role": "assistant", "content": "x"}
    with_placeholder = [*messages_before_assistant, placeholder]
    canonical_end = render_chat_ids(tokenizer, with_placeholder, False,
                                    chat_template_kwargs)
    canonical_action = [*canonical_prompt, *encoded_text_tokens(tokenizer, "x")]
    if canonical_end[:len(canonical_action)] != canonical_action:
        raise ValueError(
            "[protocol] chat template 无法定位 assistant 结束边界（占位内容也不匹配）"
            "——模板在这一版 transformers 上改写了 assistant 段，token 级增量拼接"
            "失效。docs/09 §8：改用 docs/08 的 SFT 路线或修 render 参数。")
    closing = canonical_end[len(canonical_action):]

    canonical_next = render_chat_ids(tokenizer, [*with_placeholder, next_tool_message],
                                     True, chat_template_kwargs)
    if canonical_next[:len(canonical_end)] != canonical_end:
        raise ValueError(
            "[protocol] 加入 tool observation 后 chat template 改写了历史消息"
            "——observation 增量片段不可信，拒绝产出错位序列。")
    observation = canonical_next[len(canonical_end):]
    # 校验 ④：模板必须**真的把回包渲染进去**。若该版模板不认识 role:"tool"（吞掉
    # 整条消息），observation 只会剩一个空的 user 轮 + 生成提示——模型永远看不到
    # 执行结果，"调用工具获取信息"的因果链断裂，而前缀校验（校验 ③）**照样通过**
    # （前缀没变，只是内容没了）。这类"静默空转"正是本项目最贵的一类 bug，必须在
    # 这里 fail-fast：用同一条消息 + 空 content 再渲染一次，有内容的回包若没有
    # 产出更多 token，就说明模板把 content 丢了。
    _content = str(next_tool_message.get("content") or "")
    if _content.strip():
        _probe = dict(next_tool_message)
        _probe["content"] = ""
        _empty_len = len(render_chat_ids(tokenizer, [*with_placeholder, _probe], True,
                                         chat_template_kwargs))
        if len(canonical_next) <= _empty_len:
            raise ValueError(
                "[protocol] chat template 未把 tool 回包渲染进 observation"
                f"（有内容 {len(_content)} 字符的回包渲染长度 {len(canonical_next)} "
                f"≤ 空回包 {_empty_len}）——模型看不到执行结果，调用工具的因果链断裂。\n"
                "  处置：该版模板可能只认特定形态的 tool 消息（先跑 "
                "python -m rlab.native_probe 看 Q3 的回包渲染），或改用 docs/08 SFT 路线。")

    comp = [int(t) for t in completion_tokens]
    overlap = suffix_prefix_overlap(comp, closing)
    return [*prev, *comp, *closing[overlap:], *observation]


def stop_sequences(tokenizer) -> list:
    """一轮 assistant 输出的停止串（与参考实现同构）。

    【刻意不接线】参考实现把它塞进 SamplingParams.stop；本项目**不用**
    （docs/05 §12.3 已核对 EOS 契约：vLLM 0.12 V1 的 token_ids 恒含 EOS、
    logprobs 与 ids 严格平行 → 原生协议自然停在 im_end 并拿到梯度）。
    传 tokenizer.eos_token 反而有踩到"eos(151643) ≠ im_end(151645)"旧坑的风险。
    保留本函数只为对齐口径与测试引用。"""
    eos = getattr(tokenizer, "eos_token", None)
    return [eos] if eos else []


def strip_tool_call_blocks(text: str) -> str:
    """剥掉 assistant 文本里的工具调用块——打分域的"脚手架剥离"（原生档）。

    【与 strip_code_blocks 同一条原则】围栏档把 ```python``` 块剥掉再判 acc/fmt
    （代码是脚手架：码内的数字/boxed 不当模型答案，docs/02 第四轮教训）；原生档
    的等价脚手架就是调用块本身——`{"code": "print(\\\\boxed{7})"}` 这类载荷若留在
    打分文本里，"最后一个 boxed/数字"就会被判成模型的答案。

    与 `_RE_TOOL_CALL_ANY` 同一正则：**完整**块才剥（模型被轮长切断、调用块没闭合
    时结构仍算不合格，保留原文让 invalid/无 boxed 的语义自然成立）。
    只作用于**打分域**，不动 mask/训练序列（那里调用块是模型真实生成的内容，
    必须进 loss——两者不可混淆）。"""
    return _RE_TOOL_CALL_ANY.sub("", text)


def sanitize_tool_text(text: str) -> str:
    """沙箱输出拼回模型上下文前的无害化消毒（对齐 agentic-rl-lab/05-retool 教训：
    "tool 返回不消毒会污染 observation 结构"）。

    沙箱 stdout 是模型经 print() 间接可控的通道——若模型打印 <|im_end|> 类
    特殊 token 字面量，分段 tokenize 时会被 Qwen tokenizer 还原成真的 special
    token id，往训练序列注入非模型生成的 EOS/边界 token（gen_logps/mask/训练
    全被污染）；打印 [TOOL RESULT] 字面量则可伪造嵌套工具边界混淆上下文。
    两类字节一律剥除——与 mask 契约同一条原则：凡"非模型生成但要拼进模型
    上下文"的字节流，都是训练序列的信任边界。

    【2026-09-25 原生协议】注入面清单随协议变化：围栏档要剥 `[TOOL RESULT]`，
    原生档要剥**工具的调用块标记**（打印一个假的 `<tool_call>…</tool_call>` 会
    让模型以为"已经调用过了"，扭曲它对当前轮次状态的判断）。两档的标记都剥：
    消毒函数是**唯一**的拼接口，按协议各自的最小子集消毒只会让"换了协议忘换
    消毒清单"成为一个静默的安全洞。"""
    out = _SPECIAL_TOKEN_RE.sub("", text)
    out = out.replace(TOOL_START, "").replace(TOOL_END, "")
    return out.replace(_TOOL_OPEN, "").replace(_TOOL_CLOSE, "")


def extract_python_blocks(text: str):
    """返回文本里所有完整 ```python``` 代码块（去围栏与首尾空白）。"""
    return [m.group(1).strip() for m in _PY_FENCE_RE.finditer(text)]


def segment_mask_from_spans(total: int, assistant_spans) -> torch.Tensor:
    """纯函数：由 completion 总长 T 与 assistant 生成区间 [(s,e),...]（左闭右开）
    构造 (T,) 0/1 mask——assistant token=1，工具段/其余=0。
    阶段2 最易错点：工具返回 token 若置 1，其不可信的 logps（策略对沙箱输出
    的困惑度/KL）会污染 loss，甚至产生假梯度信号。"""
    m = torch.zeros(total)
    for s, e in assistant_spans:
        m[s:e] = 1.0
    return m


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def bytes_to_tensor(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)


def make_bytes_list(blist) -> bytes:
    buf = io.BytesIO()
    buf.write(len(blist).to_bytes(4, "big"))
    for b in blist:
        buf.write(len(b).to_bytes(4, "big"))
        buf.write(b)
    return buf.getvalue()


def bytes_list_to_list(b: bytes):
    buf = io.BytesIO(b)
    num = int.from_bytes(buf.read(4), "big")
    out = []
    for _ in range(num):
        l = int.from_bytes(buf.read(4), "big")
        out.append(buf.read(l))
    return out


def encode_batch(meta: dict, merged_ids: torch.Tensor, advantages: torch.Tensor,
                 *extra_tensors: torch.Tensor) -> bytes:
    """生成端打包。extra_tensors 依序：单轮为 gen_logps / acc_scores / format_scores；
    retool（meta["has_mask"]=1）为 gen_logps / mask / acc_scores / format_scores。"""
    parts = [json.dumps(meta).encode(), tensor_to_bytes(merged_ids),
             tensor_to_bytes(advantages)]
    parts.extend(tensor_to_bytes(t) for t in extra_tensors)
    return make_bytes_list(parts)


def decode_batch(raw: bytes) -> dict:
    """训练端解包（ref_server /get 返回）。布局由 meta['algo'] 决定：

    passthrough 输出（GRPO 家族）:
      [meta, inputs, advantages, refs, gen_logps, acc_scores, format_scores]
      retool（meta['has_mask']=1）:
      [meta, inputs, advantages, refs, gen_logps, mask, acc_scores, format_scores]
      retool + overlong_filter（meta['has_mask']=1 且 meta['has_sw']=1）:
      [meta, inputs, advantages, refs, gen_logps, mask, acc_scores, format_scores, sample_weight]
    rfpp 输出（多一个服务端算好的 per-token advantages 段）:
      [meta, inputs, raw_rewards, refs, gen_logps, advantages(B,T), acc_scores, format_scores]
    """
    dd = bytes_list_to_list(raw)
    data = json.loads(dd[0])
    data["inputs"] = bytes_to_tensor(dd[1])
    data["rewards"] = bytes_to_tensor(dd[2])
    data["refs"] = bytes_to_tensor(dd[3])
    data["gen_logps"] = bytes_to_tensor(dd[4])
    if data.get("algo") == "rfpp":
        data["advantages"] = bytes_to_tensor(dd[5])
        if len(dd) >= 7:
            data["acc_scores"] = bytes_to_tensor(dd[6])
        if len(dd) >= 8:
            data["format_scores"] = bytes_to_tensor(dd[7])
    else:
        data["advantages"] = data["rewards"]      # GRPO 家族：上传的就是归一化 advantage
        if data.get("has_mask"):
            data["mask"] = bytes_to_tensor(dd[5])
            if len(dd) >= 7:
                data["acc_scores"] = bytes_to_tensor(dd[6])
            if len(dd) >= 8:
                data["format_scores"] = bytes_to_tensor(dd[7])
            # 【2026-09-21 overlong_filter·loss 排除截断样本】sample_weight (B,)
            # 标记截断样本（weight=0）：它们 adv=0（不贡献 pg_term）但 KL 仍活跃，
            # sample_mean 的 .mean() 会把它们的 KL 算进分母稀释 pg 梯度。
            # sample_weight 让归一化只算有效样本，彻底排除截断样本的所有贡献。
            if len(dd) >= 9:
                data["sample_weight"] = bytes_to_tensor(dd[8])
        else:
            if len(dd) >= 6:
                data["acc_scores"] = bytes_to_tensor(dd[5])
            if len(dd) >= 7:
                data["format_scores"] = bytes_to_tensor(dd[6])
    return data
