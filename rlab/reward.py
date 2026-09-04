# -*- coding: utf-8 -*-
"""rlab/reward.py — 奖励函数（与 simple_grpo_v1 完全同口径，保证跨实验可比）。

三个组件，可组合：
  reward_correct : GSM8K 口径——取答案串最后一个数字与标准答案 math_verify 比对（±w_acc）
  reward_format  : <think>..</think><answer>..</answer> 结构正则（±1），惩罚抄模板占位符
  overlong_penalty: DAPO 软悬崖——接近生成长度上限线性扣分 [0,1]

【工程教训内置】
- math_verify 在线程环境必须传 parsing_timeout=None / timeout_seconds=None，
  signal.alarm 只能在主线程；本模块统一在调用处关闭其内部超时。
- format 正则与 grpo_ref_split.py 严格一致（紧连式），不与 rf++ 旧正则混用。
"""

import re

# math_verify 导入失败时（纯 CPU 冒烟环境）退化为纯文本比对
try:
    from math_verify import parse, verify, ExprExtractionConfig
    HAS_MATH_VERIFY = True
except ImportError:  # pragma: no cover - 冒烟环境
    HAS_MATH_VERIFY = False

_NUM_RE = r"\d+\.\d+|\d+/\d+|\d+"
_FORMAT_RE = r"^<think>.*?</think><answer>.*?</answer>$"


def _mv_parse(text: str):
    """math_verify 解析，线程安全（禁用其内部 signal 超时）。"""
    return parse(text, extraction_config=[ExprExtractionConfig()],
                 parsing_timeout=None)


def reward_correct(ground_truth: str, answer: str) -> float:
    """正确性得分：答对 +1.0，未提取到数字/答错 -1.0。"""
    nums = re.findall(_NUM_RE, answer)
    if not nums:
        return -1.0
    if not HAS_MATH_VERIFY:
        # 冒烟环境退化：纯文本相等判对
        return 1.0 if nums[-1] == ground_truth.strip() else -1.0
    try:
        ans = _mv_parse(nums[-1])
        gt = _mv_parse(ground_truth)
        return 1.0 if verify(ans, gt, timeout_seconds=None) else -1.0
    except Exception:
        return -1.0


def reward_format(answer: str) -> float:
    """格式得分：结构完整 +1.0，否则 -1.0；惩罚抄系统提示占位符的奖励黑客。"""
    if "reasoning process here" in answer.lower():
        return -1.0
    ok = bool(re.match(_FORMAT_RE, answer, re.DOTALL | re.VERBOSE))
    if ok:
        # 标签各出现恰好一次（防 <answer> 里再嵌一套标签骗分）
        ok = (answer.count("<think>") + answer.count("</think>") == 2
              and answer.count("<answer>") + answer.count("</answer>") == 2)
    return 1.0 if ok else -1.0


def overlong_penalty(completion_len: int, max_gen_tokens: int, buffer: int = 64) -> float:
    """DAPO overlong shaping：超过 (max-buffer) 后线性扣分，封顶 1.0。"""
    trigger = max_gen_tokens - buffer
    if completion_len <= trigger:
        return 0.0
    return min((completion_len - trigger) / buffer, 1.0)


def total_reward(ground_truth: str, answer: str, *, w_acc: float = 2.0,
                 completion_len: int = 0, max_gen_tokens: int = 512,
                 overlong_buffer: int = 64, overlong_shaping: bool = False) -> dict:
    """组合口径（与 grpo_dapo 一致）：2.0*correct + 1.0*format - overlong。
    返回分量 dict，供 record 记录与监控。"""
    acc = reward_correct(ground_truth, answer)
    fmt = reward_format(answer)
    r = w_acc * acc + fmt
    pen = 0.0
    if overlong_shaping:
        pen = overlong_penalty(completion_len, max_gen_tokens, overlong_buffer)
        r -= pen
    return {"reward": r, "acc": acc, "format": fmt, "overlong": pen}
