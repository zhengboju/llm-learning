# -*- coding: utf-8 -*-
"""rlab/swe_l0_synth.py — L0 合成 bug 生成器（docs/17 §2.5 要求的那个"生成器"）。

输入一个**已通过全部测试**的正确实现 + 它的测试，输出一批候选 buggy 实现
（每个都是一处语义变异）。**候选是否可用由执行判定，不由定义判定**——
这正是 docs/17 §1 说的"环境即工程"：合成 bug 的价值全在"它真的让测试变红"。

为什么用 AST 定位 + **文本拼接**改字节（而不是 `ast.unparse` 重写整个文件）
------------------------------------------------------------------------
`ast.unparse` 会丢掉注释、空行与原有引号风格，于是 buggy 版本与正确版本在
**格式上也不同**——模型交上来的 patch 会同时包含"语义修复"与"格式还原"两类
改动，diff 噪声直接掩掉 L0 想测的那个信号（单函数语义修复）。文本拼接只改
被变异的那个语法片段，其余字节逐位不变，diff 里就只剩目标改动。

变异算子（都是"人类工程师真会写错"的形态，不是随机乱码）
------------------------------------------------------
  · 比较符翻转 `<`↔`<=`、`>`↔`>=`、`==`↔`!=`（边界/取等号，最常见的 off-by-one）
  · 整型常量 ±1（off-by-one）
  · `and`↔`or`（条件短路语义）
  · `+`↔`-`（算符方向）
  · `max`↔`min`（极值方向）
  · `not X` → `X`（取反丢失）
  · `return X` → `return None`（提前返回/漏返回）

刻意**不做**的变异（会制造不可学的 instance，如实记录）
----------------------------------------------------
  · 改字符串常量：L0 种子里字符串多为异常消息/文档，改了往往无测试覆盖 →
    大批"变异不可观测"的废候选，白烧测试执行（本项目最贵的一步）。
  · 删语句/换行：极易产生 NameError 级联，把"单点语义 bug"变成"整个模块不可
    导入"，F2P 退化成"全部测试"，P2P 恒空——回归信号消失（见 prepare 的统计）。
  · 随机插 token：那不是 bug，是乱码；模型学到的是"修复语法"而非"修复语义"。

零标签字面量铁律：本模块只处理 python 源码文本，不含任何格式标签字节。
"""

import ast

# 变异族名（进 instance id 与落盘统计，便于按族分析"模型修得好哪类 bug"）
OP_CMP_FLIP = "cmp_flip"
OP_INT_OFF_BY_ONE = "int_offby1"
OP_BOOL_FLIP = "bool_flip"
OP_ARITH_FLIP = "arith_flip"
OP_EXTREME_FLIP = "extreme_flip"
OP_NOT_DROP = "not_drop"
OP_RETURN_NONE = "return_none"

_CMP_SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
_CMP_TOKENS = ("<=", ">=", "==", "!=", "<", ">")
_EXTREME = {"max": "min", "min": "max"}


class MutationSite:
    """一处可变异的位置：节点 + 片段范围 + 变异族 + 替换文本。

    `start`/`end` 是 **原文** 的字符偏移（`lineno/col_offset` + `end_lineno/
    end_col_offset` 换算）。`col_offset` 在 CPython 里是 **UTF-8 字节**偏移，
    不是字符偏移——含中文的源码（本项目种子含中文注释/文档串）若不换算，拼接
    位置会整体错位，产出**看起来正常但改错行**的 buggy 版本（最坏的一类静默
    错误：测试仍然变红，只是红的不是我们要测的那一处）。换算见 `_offset`。
    """

    __slots__ = ("family", "start", "end", "replacement", "label")

    def __init__(self, family, start, end, replacement, label):
        self.family = family
        self.start = start
        self.end = end
        self.replacement = replacement
        self.label = label

    def __repr__(self):
        return f"<Site {self.family} {self.start}:{self.end} -> {self.replacement!r}>"


def _offset(source: str, lineno: int, col: int) -> int:
    """(行号, UTF-8 字节列) → 全文字符偏移。

    CPython 报的 col_offset 是 **UTF-8 字节**列；源码里出现非 ASCII 字符时
    1 个中文 = 3 字节，直接当字符列用会偏移。这里按行累计 + 解码前缀换算。
    """
    lines = source.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        return -1
    base = sum(len(x) for x in lines[:lineno - 1])
    line = lines[lineno - 1]
    if col <= 0:
        return base
    # 把该行的 UTF-8 前缀截到 col 字节，再数出字符数
    raw = line.encode("utf-8")
    if col > len(raw):
        col = len(raw)
    try:
        prefix = raw[:col].decode("utf-8")
    except UnicodeDecodeError:
        return base + col
    return base + len(prefix)


def _node_span(source: str, node) -> tuple:
    start = _offset(source, node.lineno, node.col_offset)
    end = _offset(source, node.end_lineno, node.end_col_offset)
    return start, end


def _gap_between(source: str, left, right) -> tuple:
    """两个相邻节点之间的原文区段（算符/关键字都落在这里）。"""
    _, lend = _node_span(source, left)
    rstart, _ = _node_span(source, right)
    return lend, rstart


def _find_op_in_gap(source: str, left, right, tokens) -> tuple:
    """在 gap 里按**最长优先**找算符 token，返回 (start, end, token)。"""
    gs, ge = _gap_between(source, left, right)
    if gs < 0 or ge < 0 or ge <= gs:
        return None
    gap = source[gs:ge]
    for tok in tokens:
        idx = gap.find(tok)
        if idx >= 0:
            return gs + idx, gs + idx + len(tok), tok
    return None


def enumerate_sites(source: str) -> list:
    """列出所有可变异点（纯函数，CPU 可测）。返回按原文偏移排序的 Site 列表。

    只变异**函数体内**的片段：模块级常量/import/文档串不动（改它们通常
    直接 ImportError，把语义实例退化成语法实例）。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    sites = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            # ---- 比较符 ----
            if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                    and len(node.comparators) == 1:
                sym = _cmp_symbol(node.ops[0])
                if sym and sym in _CMP_SWAP:
                    hit = _find_op_in_gap(source, node.left, node.comparators[0],
                                          _CMP_TOKENS)
                    if hit:
                        s, e, tok = hit
                        if tok == sym:      # gap 里找到的必须就是该算符
                            sites.append(MutationSite(
                                OP_CMP_FLIP, s, e, _CMP_SWAP[sym],
                                f"{sym} -> {_CMP_SWAP[sym]}"))
            # ---- 整型常量 ±1 ----
            elif isinstance(node, ast.Constant) and isinstance(node.value, int) \
                    and not isinstance(node.value, bool):
                s, e = _node_span(source, node)
                if s >= 0:
                    sites.append(MutationSite(
                        OP_INT_OFF_BY_ONE, s, e, str(node.value + 1),
                        f"{node.value} -> {node.value + 1}"))
            # ---- 布尔算符 ----
            elif isinstance(node, ast.BoolOp) and len(node.values) >= 2:
                is_and = isinstance(node.op, ast.And)
                hit = _find_op_in_gap(source, node.values[0], node.values[1],
                                      ("and", "or"))
                if hit:
                    s, e, tok = hit
                    want = "and" if is_and else "or"
                    if tok == want:
                        new = "or" if want == "and" else "and"
                        sites.append(MutationSite(OP_BOOL_FLIP, s, e, new,
                                                  f"{want} -> {new}"))
            # ---- 加减算符 ----
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add,
                                                                     ast.Sub)):
                want = "+" if isinstance(node.op, ast.Add) else "-"
                hit = _find_op_in_gap(source, node.left, node.right, ("+", "-"))
                if hit:
                    s, e, tok = hit
                    if tok == want:
                        new = "-" if want == "+" else "+"
                        sites.append(MutationSite(OP_ARITH_FLIP, s, e, new,
                                                  f"{want} -> {new}"))
            # ---- max/min ----
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in _EXTREME and node.args:
                s, e = _node_span(source, node.func)
                if s >= 0:
                    sites.append(MutationSite(
                        OP_EXTREME_FLIP, s, e, _EXTREME[node.func.id],
                        f"{node.func.id} -> {_EXTREME[node.func.id]}"))
            # ---- not 丢失 ----
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                s, e = _node_span(source, node)
                if s >= 0 and source[s:e].lstrip().startswith("not "):
                    operand = source[s:e].lstrip()[len("not "):]
                    sites.append(MutationSite(OP_NOT_DROP, s, e, f"({operand})",
                                              "drop not"))
            # ---- return X -> return None ----
            elif isinstance(node, ast.Return) and node.value is not None:
                s, e = _node_span(source, node)
                if s >= 0:
                    sites.append(MutationSite(OP_RETURN_NONE, s, e, "return None",
                                              "return -> None"))
    # 去重（同范围同替换）并按位置排序：保证同一份源码的候选集**确定**
    uniq, seen = [], set()
    for site in sorted(sites, key=lambda x: (x.start, x.end, x.replacement)):
        key = (site.start, site.end, site.replacement)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(site)
    return uniq


def _cmp_symbol(op) -> str:
    for sym, cls in (("<", ast.Lt), ("<=", ast.LtE), (">", ast.Gt),
                     (">=", ast.GtE), ("==", ast.Eq), ("!=", ast.NotEq)):
        if isinstance(op, cls):
            return sym
    return ""


def apply_site(source: str, site: MutationSite) -> str:
    """在原文上做**单点**文本替换，返回变异后的源码（纯函数）。"""
    return source[:site.start] + site.replacement + source[site.end:]


def mutate(source: str, site: MutationSite) -> str:
    """同 apply_site（别名，语义更直白）；变异后**必须**能重新解析，否则抛。"""
    out = apply_site(source, site)
    ast.parse(out)          # 变异产物必须仍是合法 python（不可解析的变异无意义）
    return out


def candidate_mutations(source: str) -> list:
    """该源码的全部候选变异：[(site, mutated_source), ...]（不可解析的已被剔除）。"""
    out = []
    for site in enumerate_sites(source):
        try:
            out.append((site, mutate(source, site)))
        except SyntaxError:
            continue
    return out


def family_counts(sites) -> dict:
    """按变异族计数（落盘统计用）。"""
    out = {}
    for site in sites:
        out[site.family] = out.get(site.family, 0) + 1
    return out
