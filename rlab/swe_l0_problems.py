# -*- coding: utf-8 -*-
"""rlab/swe_l0_problems.py — SWE-RL L0 层的**种子问题语料**（docs/17 §2）。

L0 = HumanEvalFix 式合成 bug：单函数 + 单测现成 + 定位已给出（docs/17 §2.5：
"L0=单函数+单测"，不放整库）。本模块只提供**种子**（`(源码, 单测, 注入 bug 的
方式)`），真正的 instance（含 F2P/P2P 名单）由 `prepare_swe_l0.py` **实测**生成
——F2P 永不手写：手写的失败名单会在"注入的 bug 其实没被覆盖"时静默变成假 F2P，
而那正是"精确 0% / 恒 1"这把尺子失去意义的方式（AGENTS.md 系统性 bug 签名）。

【零标签字面量铁律】本模块只含 python 源码与测试源码，不含任何格式标签字节。

【设计约束（都是"让 L0 的 RL 信号可测"的必要条件）】
  1. **注入的 bug 必须被至少一条测试抓到**：prepare 会先跑「原始码 + 全部测试」
     断言全绿（种子自证），再跑「buggy 码 + 全部测试」要求 ≥1 失败；不满足的种子
     直接剔除并计数（落盘），绝不静默入池。
  2. **原始码 + 测试必须全过**：这是 P2P 的来源（"修好 bug 不能弄坏已有行为"），
     也是"任务本身可解"的唯一证据。
  3. **每个种子必须自带可跑的单测**：L0 不依赖任何外部仓库/依赖环境（docs/17 §2
     的核心降级理由——docker 构建成本是 L2 的门槛，L0 把它整个绕开）。
  4. 种子描述（`issue`）用**自然语言**说"哪里不对、期望什么"，不说"改哪一行"：
     定位已给出（给的是整个单函数 + 测试），但"怎么改"必须由模型产出 patch。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Seed:
    """一个 L0 种子。

    name    : 唯一 id（也是 instance 文件名前缀）
    source  : 正确实现（修复后应当等于它——不等于也能通过测试，见 prepare 的说明）
    buggy   : 注入 bug 后的实现（模型的起点）
    tests   : 单测源码（pytest 风格；与源码同目录、模块名可 import）
    target  : 被测模块文件名（不含 .py），如 "calc"
    testname: 测试文件名（不含 .py），如 "test_calc"
    issue   : 给模型看的问题描述（自然语言）
    tag     : 难度/主题分类（只作分析用，不参与训练）
    """

    name: str
    source: str
    buggy: str
    tests: str
    target: str
    testname: str
    issue: str
    tag: str


# =====================================================================
# 种子 1：边界条件（off-by-one / 闭区间 vs 半开区间）
# =====================================================================
_S1_SRC = '''def clamp(x, lo, hi):
    """把 x 限制在闭区间 [lo, hi] 内。lo > hi 时抛 ValueError。"""
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''

_S1_BUG = '''def clamp(x, lo, hi):
    """把 x 限制在闭区间 [lo, hi] 内。lo > hi 时抛 ValueError。"""
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    if x < lo:
        return lo
    if x > hi:
        return hi - 1
    return x
'''

_S1_TST = '''import pytest

from calc import clamp


def test_below_range():
    assert clamp(-5, 0, 10) == 0


def test_above_range():
    assert clamp(50, 0, 10) == 10


def test_inside_range():
    assert clamp(5, 0, 10) == 5


def test_exact_bounds():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10


def test_negative_range():
    assert clamp(-20, -10, -5) == -10
    assert clamp(-2, -10, -5) == -5


def test_bad_range_raises():
    with pytest.raises(ValueError):
        clamp(1, 10, 0)
'''

# =====================================================================
# 种子 2：整除/取模语义（负数与零）
# =====================================================================
_S2_SRC = '''def safe_div(a, b):
    """整数除法，向下取整（-7 / 2 得 -4）；b == 0 时抛 ZeroDivisionError。"""
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a // b
'''

# 注入 bug：把负数除法写成"向零截断"（abs 后再补符号）——这是从 C/Java 迁过来的
# 最典型语义错位。**它必须让测试真的红**，否则这个种子会被 prepare 剔除（见模块
# docstring 的第 1 条约束）。
_S2_BUG = '''def safe_div(a, b):
    """整数除法，向下取整（-7 / 2 得 -4）；b == 0 时抛 ZeroDivisionError。"""
    if b == 0:
        raise ZeroDivisionError("division by zero")
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        return -q
    return q
'''

_S2_TST = '''import pytest

from calc import safe_div


def test_positive():
    assert safe_div(7, 2) == 3


def test_negative_dividend():
    assert safe_div(-7, 2) == -4


def test_negative_divisor():
    assert safe_div(7, -2) == -4


def test_both_negative():
    assert safe_div(-7, -2) == 3


def test_exact():
    assert safe_div(8, 2) == 4
    assert safe_div(-8, 2) == -4


def test_zero_divisor():
    with pytest.raises(ZeroDivisionError):
        safe_div(1, 0)
'''

# =====================================================================
# 种子 3：可变默认参数 / 就地修改（最常见 python 语义 bug）
# =====================================================================
_S3_SRC = '''def append_once(items, value):
    """返回一个新列表 = items + [value]；不修改 items。"""
    out = list(items)
    out.append(value)
    return out


def merge_counts(base, extra):
    """合并两个计数字典，返回新字典；同名键相加。不修改入参。"""
    out = dict(base)
    for key, val in extra.items():
        out[key] = out.get(key, 0) + val
    return out
'''

_S3_BUG = '''def append_once(items, value):
    """返回一个新列表 = items + [value]；不修改 items。"""
    items.append(value)
    return items


def merge_counts(base, extra):
    """合并两个计数字典，返回新字典；同名键相加。不修改入参。"""
    for key, val in extra.items():
        base[key] = base.get(key, 0) + val
    return base
'''

_S3_TST = '''from calc import append_once, merge_counts


def test_append_returns_new_list():
    src = [1, 2]
    out = append_once(src, 3)
    assert out == [1, 2, 3]


def test_append_does_not_mutate():
    src = [1, 2]
    append_once(src, 3)
    assert src == [1, 2]


def test_append_empty():
    assert append_once([], 1) == [1]


def test_merge_values():
    assert merge_counts({"a": 1}, {"a": 2, "b": 1}) == {"a": 3, "b": 1}


def test_merge_does_not_mutate():
    base = {"a": 1}
    extra = {"b": 2}
    merge_counts(base, extra)
    assert base == {"a": 1}
    assert extra == {"b": 2}


def test_merge_empty():
    assert merge_counts({}, {}) == {}
'''

# =====================================================================
# 种子 4：字符串/解析（strip、split、空串边界）
# =====================================================================
_S4_SRC = '''def parse_kv(line):
    """把 "k=v" 解析成 (k, v)；两侧空白去掉。

    空行、无 '='、键为空 → 返回 None（调用方据此跳过脏行）。
    值可以为空串（"k=" 合法）。值里出现多个 '=' 时只按**第一个**切分。
    """
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text or "=" not in text:
        return None
    key, _, val = text.partition("=")
    key = key.strip()
    if not key:
        return None
    return key, val.strip()
'''

_S4_BUG = '''def parse_kv(line):
    """把 "k=v" 解析成 (k, v)；两侧空白去掉。

    空行、无 '='、键为空 → 返回 None（调用方据此跳过脏行）。
    值可以为空串（"k=" 合法）。值里出现多个 '=' 时只按**第一个**切分。
    """
    if not isinstance(line, str):
        return None
    text = line.strip()
    key, val = text.split("=")
    return key, val
'''

_S4_TST = '''from calc import parse_kv


def test_simple():
    assert parse_kv("a=1") == ("a", "1")


def test_spaces_stripped():
    assert parse_kv("  a = 1  ") == ("a", "1")


def test_empty_value_is_legal():
    assert parse_kv("a=") == ("a", "")


def test_multiple_equals_keeps_rest():
    assert parse_kv("a=b=c") == ("a", "b=c")


def test_empty_line_is_none():
    assert parse_kv("") is None
    assert parse_kv("   ") is None


def test_no_equals_is_none():
    assert parse_kv("abc") is None


def test_empty_key_is_none():
    assert parse_kv("=1") is None


def test_non_string_is_none():
    assert parse_kv(None) is None
    assert parse_kv(42) is None
'''

# =====================================================================
# 种子 5：列表算法（去重保序 + 边界）
# =====================================================================
_S5_SRC = '''def unique_stable(items):
    """去重并保持首次出现顺序。不可哈希元素按 == 比较。"""
    out = []
    for item in items:
        found = False
        for seen in out:
            if seen == item:
                found = True
                break
        if not found:
            out.append(item)
    return out
'''

_S5_BUG = '''def unique_stable(items):
    """去重并保持首次出现顺序。不可哈希元素按 == 比较。"""
    return list(set(items))
'''

_S5_TST = '''from calc import unique_stable


def test_keeps_first_order():
    assert unique_stable([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_no_duplicates():
    assert unique_stable([1, 2, 3]) == [1, 2, 3]


def test_all_same():
    assert unique_stable([7, 7, 7]) == [7]


def test_empty():
    assert unique_stable([]) == []


def test_unhashable_elements():
    assert unique_stable([[1], [1], [2]]) == [[1], [2]]
'''

# =====================================================================
# 种子 6：区间合并（排序 + 相邻边界）
# =====================================================================
_S6_SRC = '''def merge_intervals(intervals):
    """合并重叠或相邻的闭区间；返回按起点升序的新列表。

    输入不保证有序，可为空；相邻（[1,2] 与 [3,4]）也合并。
    """
    if not intervals:
        return []
    ordered = sorted((int(a), int(b)) for a, b in intervals)
    out = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= out[-1][1] + 1:
            if hi > out[-1][1]:
                out[-1][1] = hi
        else:
            out.append([lo, hi])
    return out
'''

_S6_BUG = '''def merge_intervals(intervals):
    """合并重叠或相邻的闭区间；返回按起点升序的新列表。

    输入不保证有序，可为空；相邻（[1,2] 与 [3,4]）也合并。
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo < out[-1][1]:
            out[-1][1] = hi
        else:
            out.append([lo, hi])
    return out
'''

_S6_TST = '''from calc import merge_intervals


def test_empty():
    assert merge_intervals([]) == []


def test_single():
    assert merge_intervals([[1, 3]]) == [[1, 3]]


def test_overlap():
    assert merge_intervals([[1, 3], [2, 6]]) == [[1, 6]]


def test_adjacent_merged():
    assert merge_intervals([[1, 2], [3, 4]]) == [[1, 4]]


def test_unsorted_input():
    assert merge_intervals([[8, 10], [1, 3], [2, 4]]) == [[1, 4], [8, 10]]


def test_disjoint_kept():
    assert merge_intervals([[1, 2], [10, 12]]) == [[1, 2], [10, 12]]


def test_nested_not_shrinking():
    assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]


def test_duplicates():
    assert merge_intervals([[1, 2], [1, 2]]) == [[1, 2]]
'''

# =====================================================================
# 种子 7：异常传播与校验（"静默吞错"是最常见的坏修复起点）
# =====================================================================
_S7_SRC = '''def require_positive(values):
    """全部为正整数时返回原列表（新列表）；任一非正或非整数 → ValueError。"""
    if not isinstance(values, (list, tuple)):
        raise TypeError("values must be a list or tuple")
    out = []
    for val in values:
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError("values must be integers")
        if val <= 0:
            raise ValueError("values must be positive")
        out.append(val)
    return out
'''

_S7_BUG = '''def require_positive(values):
    """全部为正整数时返回原列表（新列表）；任一非正或非整数 → ValueError。"""
    out = []
    for val in values:
        if val <= 0:
            continue
        out.append(val)
    return out
'''

_S7_TST = '''import pytest

from calc import require_positive


def test_all_positive():
    assert require_positive([1, 2, 3]) == [1, 2, 3]


def test_returns_new_list():
    src = [1, 2]
    out = require_positive(src)
    assert out == src
    assert out is not src


def test_zero_raises():
    with pytest.raises(ValueError):
        require_positive([1, 0, 2])


def test_negative_raises():
    with pytest.raises(ValueError):
        require_positive([-1])


def test_non_int_raises():
    with pytest.raises(ValueError):
        require_positive([1, "2"])


def test_bool_raises():
    with pytest.raises(ValueError):
        require_positive([True])


def test_bad_container_raises():
    with pytest.raises(TypeError):
        require_positive("123")
'''

# =====================================================================
# 种子 8：递归/记忆化（返回类型与空输入）
# =====================================================================
_S8_SRC = '''def fib(n):
    """第 n 个斐波那契数（fib(0)=0, fib(1)=1）；n < 0 → ValueError。"""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
'''

_S8_BUG = '''def fib(n):
    """第 n 个斐波那契数（fib(0)=0, fib(1)=1）；n < 0 → ValueError。"""
    if not isinstance(n, int) or isinstance(n, bool):
        return n
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return max(n, 1)
    return fib(n - 1) + fib(n - 2)
'''

_S8_TST = '''import pytest

from calc import fib


def test_base():
    assert fib(0) == 0
    assert fib(1) == 1


def test_small():
    assert fib(2) == 1
    assert fib(5) == 5
    assert fib(10) == 55


def test_large():
    assert fib(25) == 75025


def test_negative_raises():
    with pytest.raises(ValueError):
        fib(-1)


def test_non_int_raises():
    with pytest.raises(TypeError):
        fib(3.5)
'''
SEEDS = (
    Seed(name="l0_clamp_bounds", source=_S1_SRC, buggy=_S1_BUG, tests=_S1_TST,
         target="calc", testname="test_calc", tag="boundary",
         issue=(
             "`clamp(x, lo, hi)` 应该把 x 限制在闭区间 [lo, hi] 内。"
             "现在当 x 大于上界时，返回的值比上界小 1，"
             "所以 clamp(50, 0, 10) 得到的是 9 而不是 10。"
             "请修复它，使上界被正确返回，且其余行为（下界、区间内、"
             "lo > hi 抛 ValueError）保持不变。")),
    Seed(name="l0_safe_div_negative", source=_S2_SRC, buggy=_S2_BUG, tests=_S2_TST,
         target="calc", testname="test_calc", tag="arithmetic",
         issue=(
             "`safe_div(a, b)` 应做整数除法并向下取整（-7 // 2 是 -4 而不是 -3），"
             "b == 0 时抛 ZeroDivisionError。现在的实现在负数上返回了截断结果。"
             "请修复符号与取整语义，保持除以 0 的行为不变。")),
    Seed(name="l0_append_mutates", source=_S3_SRC, buggy=_S3_BUG, tests=_S3_TST,
         target="calc", testname="test_calc", tag="mutation",
         issue=(
             "`append_once(items, value)` 和 `merge_counts(base, extra)` 的文档承诺"
             "返回新对象、不修改入参，但现在的实现会就地修改传入的 list/dict，"
             "调用方的数据被意外改掉了。请修复成不修改入参、返回新对象，"
             "合并/追加的结果保持不变。")),
    Seed(name="l0_parse_kv_robust", source=_S4_SRC, buggy=_S4_BUG, tests=_S4_TST,
         target="calc", testname="test_calc", tag="parsing",
         issue=(
             "`parse_kv(line)` 解析 \"k=v\" 形式的行，文档要求：空行/无 '='/空键"
             "返回 None，值可以为空串，值里出现多个 '=' 时只按第一个切分。"
             "现在的实现遇到这些边界会抛异常或切分错误。请按文档修复。")),
    Seed(name="l0_unique_stable", source=_S5_SRC, buggy=_S5_BUG, tests=_S5_TST,
         target="calc", testname="test_calc", tag="list",
         issue=(
             "`unique_stable(items)` 要去重并保持**首次出现顺序**，且要能处理"
             "不可哈希的元素（比如 list）。现在的实现用 set 去重，"
             "既打乱了顺序，也无法处理不可哈希元素。请修复。")),
    Seed(name="l0_merge_intervals", source=_S6_SRC, buggy=_S6_BUG, tests=_S6_TST,
         target="calc", testname="test_calc", tag="algorithm",
         issue=(
             "`merge_intervals(intervals)` 应合并**重叠或相邻**的闭区间，"
             "输入不保证有序。现在的实现有三处偏差：相邻区间没合并、"
             "被完全包含的区间会把结果缩短、输入未做数值归一。请按文档修复。")),
    Seed(name="l0_require_positive_raises", source=_S7_SRC, buggy=_S7_BUG, tests=_S7_TST,
         target="calc", testname="test_calc", tag="validation",
         issue=(
             "`require_positive(values)` 必须在遇到非正整数时抛异常"
             "（非 list/tuple 抛 TypeError；非 int、bool、非正数抛 ValueError），"
             "而不是跳过它们。现在的实现静默过滤了坏值，把错误数据变成了正常结果。"
             "请恢复校验与异常，合法输入的返回值保持不变。")),
    Seed(name="l0_fib_int_guard", source=_S8_SRC, buggy=_S8_BUG, tests=_S8_TST,
         target="calc", testname="test_calc", tag="recursion",
         issue=(
             "`fib(n)` 返回第 n 个斐波那契数（fib(0)=0, fib(1)=1），"
             "n < 0 抛 ValueError，非 int 抛 TypeError。现在的实现一是缺类型校验，"
             "二是用朴素递归，n 稍大就算不动（fib(120) 跑不完）。"
             "请改成迭代实现并补上类型校验，取值保持不变。")),
)
