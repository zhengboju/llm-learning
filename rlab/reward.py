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

# 围栏正则复用协议层（与 extract_python_blocks 同一条），用于打分前剥离代码块
from rlab.protocol import _PY_FENCE_RE

# 剥离专用正则：与 _PY_FENCE_RE 同结构，但额外吞掉围栏后的空白——
# 代码块被移除后若残留换行，^ 锚定的格式正则依旧必败（2026-09-08 实测）。
_STRIP_FENCE_RE = re.compile(r"```python\s*(.*?)```\s*", re.DOTALL)

# math_verify 导入失败时（纯 CPU 冒烟环境）退化为纯文本比对
try:
    from math_verify import parse, verify, ExprExtractionConfig
    HAS_MATH_VERIFY = True
except ImportError:  # pragma: no cover - 冒烟环境
    HAS_MATH_VERIFY = False

_NUM_RE = r"\d+\.\d+|\d+/\d+|\d+"
_FORMAT_RE = r"^<think>.*?</think><answer>.*?</answer>$"

# ---- 方案1：DAPO-Math / AIME 口径（对齐 agentic-rl-lab/05-retool） ----
_MATH_BOXED_WINDOW = 300  # 官方只看末尾 300字符


def extract_last_boxed(text: str) -> str | None:
    """提取最后一个 \\boxed{...} 的内容，花括号配平（对齐 agentic/reward.py）。"""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos]
    return None


def reward_correct_boxed(ground_truth: str, answer: str) -> float:
    """DAPO-Math/AIME 口径：取回答末 300字符里最后一个 \\boxed{} 与 gt 数学等价判定（±1）。"""
    boxed = extract_last_boxed(answer[-_MATH_BOXED_WINDOW:])
    if boxed is None:
        return -1.0
    if not HAS_MATH_VERIFY:
        return 1.0 if boxed.strip() == ground_truth.strip() else -1.0
    try:
        # agentic 用 parse(f"${x}$") 包一层；我们复用 _mv_parse 但带 $ 前缀以对齐
        # _mv_parse 内部已用 ExprExtractionConfig，$ 前缀不影响数字/表达式提取
        ans = _mv_parse(f"${boxed.strip()}$")
        gt = _mv_parse(f"${ground_truth.strip()}$")
        return 1.0 if verify(ans, gt, timeout_seconds=None) else -1.0
    except Exception:
        return -1.0


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


def strip_code_blocks(text: str) -> str:
    """去掉 ```python``` 围栏代码块——打分域只认模型自己的"回答文本"。

    代码是脚手架：其内容可含中间计算/沙箱相关数字，不参与 acc/fmt 打分
    （与"工具段只作上下文、不进 loss 不进打分"同一条原则在文本域的投影）。
    只剥离**完整**围栏块（与 extract_python_blocks 同一正则），并连同其后的
    空白一起移除（否则残留换行会让 ^ 锚定的格式正则依旧失败）；
    未闭合围栏 = 模型没写完代码，结构仍算不合格，保留原文。"""
    return _STRIP_FENCE_RE.sub("", text)


def reward_format_retool(answer: str) -> float:
    """retool 格式口径：先剥离代码块，再按阶段0/1 的 reward_format 判结构。

    【2026-09-08 第三轮真机教训】MUST 提示让模型"先写代码"（围栏开局），
    而 _FORMAT_RE 以 ^ 锚定要求回答文本以 thinking 开头——两者冲突导致所有
    代码先行样本格式结构性失败：训练期 fmt 恒 -1 信号死亡（代码再次被灭绝），
    eval 端 BASE fmt 48.7→4.7、retool300 fmt 0.0%（精确 0/300 系统性签名）。
    剥离代码后格式检查回到"回答文本本身结构"的语义：代码在哪都不影响格式。"""
    return reward_format(strip_code_blocks(answer))


def overlong_penalty(completion_len: int, max_gen_tokens: int, buffer: int = 64) -> float:
    """DAPO overlong shaping：超过 (max-buffer) 后线性扣分，封顶 1.0。"""
    trigger = max_gen_tokens - buffer
    if completion_len <= trigger:
        return 0.0
    return min((completion_len - trigger) / buffer, 1.0)


def overlong_ref_tokens(cfg: dict) -> int:
    """overlong shaping 的长度参考系（单点真相，retool/单轮两路共用）。

    单轮路径参考 = max_gen_tokens；retool 多轮路径参考 = **可写满的总预算**，
    即 min(max_rounds × round_gen_tokens, max_context_tokens − max_prompt_length)。
    取 min 是双保险：预算自洽时前者更小（也是真正的"用完轮数预算"信号），
    万一配置被绕过，也绝不会把 trigger 放到 overlong 丢弃线之外（那样又会变成
    永远够不着的死开关）。

    【2026-09-11 修复】旧版两路都拿 cfg["max_gen_tokens"] 当参考——retool_math
    preset 的 8192 是单轮时代遗留值，而 3×3072(CLI 探针口径)=9216 > 8192：
    合法用满预算的轨迹越过 trigger 即被整额扣 1.0，而本算法 reward 域是 ±1，
    等于把做对的 +1 抹成 0、把做错的 -1 加倍成 -2——是误伤不是 shaping。

    【2026-09-12 二次修复】上一版直接返回 rounds × per_round，但在
    `3×3072=9216 > max_context_tokens=8192` 的配置下它 **大于丢弃线** →
    retool_context_overlong 先把样本丢了，trigger 永远够不着 = 死开关。
    根因（预算不自洽）已由 config.validate_retool_budget() fail-fast 拦死，
    这里再加 min() 兜底，让"shaping 永远可达"成为结构性保证。
    """
    rounds = cfg.get("max_rounds", 1)
    per_round = cfg.get("round_gen_tokens")
    if cfg.get("algo", "").startswith("retool") and per_round:
        cap = rounds * per_round
        ctx = cfg.get("max_context_tokens")
        if ctx:
            usable = ctx - int(cfg.get("max_prompt_length", 0) or 0)
            if usable > 0:
                return min(cap, usable)
        return cap
    return cfg["max_gen_tokens"]


def trunc_penalty(trunc_final: int, weight: float = 0.0) -> float:
    """末段被轮长上限切断的靶向惩罚（纯函数）。

    【2026-09-12 为什么需要它】completion 总长惩罚（overlong_penalty）够不到
    "单轮就结束"的 prose 轨迹：clen ≤ round_gen_tokens < overlong trigger，
    而这类样本正是长度膨胀的受害主体。实测（200 步 run，1040 条样本）：
      trunc_final=1 的 458 条 acc+ = 5.0%；trunc_final=0 的 582 条 acc+ = 64.3%；
      组内更长的轨迹答对率 75.4%（corr(clen,acc)=+0.25~+0.43）。
    纯 ±1 outcome 奖励会把"长度"当作正确性的代理来强化（长→更可能答对→
    正 advantage→再变长），必须有反向项把"撞上限"这件事本身标成负信号。
    权重为 0 时本项完全不存在（其他算法协议零变化）。
    """
    if weight <= 0.0:
        return 0.0
    return weight if int(trunc_final) else 0.0


# ------------------------------------------------- 阶段2 ReTool 奖励 ----
# 代码可用率小权重：执行成功的代码块数 * code_w（失败/超时不加分）。
# Auto_Program 原口径 call_python = (python_cnt - error_cnt) * 0.1，
# 在我们的统计里 = code_ok * 0.1（成功执行次数），cap(max_rounds) 内。
def reward_code(code_ok: int, code_w: float = 0.1) -> float:
    return code_ok * code_w


def reward_phase(steps_elapsed: int, switch_step: int) -> str:
    """冷启动/后期权重切换：optimizer step < switch_step 为 "cold"。
    Auto_Program 用 16 次权重推送(=16*16=256 步)作为阈值——冷启动期代码/格式权重
    更大，引导模型先学会"写代码+套格式"，正确性权重小；后期翻转（2*acc 主导）。
    本函数是纯逻辑，训练端/生成端共用（生成端用推送次数*gen_update_steps 近似）。"""
    return "cold" if steps_elapsed < switch_step else "hot"


def total_reward_retool(ground_truth: str, answer: str, *, code_ok: int,
                        phase: str, code_w: float = 0.1,
                        cold_w: tuple = (1.0, 2.0, 2.0), hot_w: tuple = (2.0, 1.0, 1.0),
                        completion_len: int = 0, max_gen_tokens: int = 512,
                        overlong_buffer: int = 64, overlong_shaping: bool = False) -> dict:
    """阶段2 组合口径：w_acc*acc + w_fmt*fmt + w_code*code_ok*code_w。

    cold（冷启动，默认 (1,2,2)）：代码/格式权重大，先学会工具与格式；
    hot（后期，默认 (2,1,1)）：正确性主导（与阶段1 的 2.0*acc+fmt 对齐）。
    Auto_Program 的 cold 权重其实等价于 acc + 2*fmt + 2*call_python。
    返回分量 dict 供 record 记录与监控。

    【打分域】acc/fmt 一律在剥离代码块后的回答文本上判（strip_code_blocks）：
    ①MUST 提示下模型"代码先行"，^ 锚定的格式正则若直接判会结构性失败——
      fmt 恒 -1 → 训练信号死亡、eval 恒 0%（第三轮真机实锤，见 reward_format_retool）；
    ②代码里的数字（中间计算/打印）不能当"模型答案"（与"工具 stdout 不是答案"
      同一原则）；③代码放哪段（thinking 前/中间/答案后）都不影响格式与取数。"""
    clean = strip_code_blocks(answer)
    acc = reward_correct(ground_truth, clean)
    fmt = reward_format(clean)
    w_acc, w_fmt, w_code = cold_w if phase == "cold" else hot_w
    code = reward_code(code_ok, code_w)
    r = w_acc * acc + w_fmt * fmt + w_code * code
    pen = 0.0
    if overlong_shaping:
        pen = overlong_penalty(completion_len, max_gen_tokens, overlong_buffer)
        r -= pen
    return {"reward": r, "acc": acc, "format": fmt, "code": code,
            "code_ok": code_ok, "overlong": pen}


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


# ---- 方案1：DAPO-Math / AIME outcome-only（对齐 agentic-rl-lab/05-retool） ----
def total_reward_math(ground_truth: str, answer: str, *,
                      completion_len: int = 0, max_gen_tokens: int = 8192,
                      overlong_buffer: int = 64, overlong_shaping: bool = False) -> dict:
    """数学 outcome-only：只看末尾 \\boxed{} 是否与 gt 数学等价（±1），无格式分。"""
    acc = reward_correct_boxed(ground_truth, answer)
    # fmt 沿用 boxed 是否存在作监控（1=抽到 boxed，-1=无），但不进 reward
    fmt = 1.0 if extract_last_boxed(answer[-_MATH_BOXED_WINDOW:]) is not None else -1.0
    r = acc
    pen = 0.0
    if overlong_shaping:
        pen = overlong_penalty(completion_len, max_gen_tokens, overlong_buffer)
        r -= pen
    return {"reward": r, "acc": acc, "format": fmt, "overlong": pen}


def total_reward_retool_math(ground_truth: str, answer: str, *, code_ok: int = 0,
                             completion_len: int = 0, max_gen_tokens: int = 8192,
                             overlong_buffer: int = 64, overlong_shaping: bool = False,
                             trunc_final: int = 0, trunc_shaping: float = 0.0) -> dict:
    """retool-math outcome-only：与 total_reward_math 同 reward（±1），
    工具使用完全靠结果涌现，不额外奖励 code_ok。code 仅作监控记录。

    【2026-09-09 审查修复·打分域与 eval 统一】先剥离代码块再提取 boxed：
    旧版直接在 assistant 拼接上取 boxed——模型在最终 boxed 之后补一段验证代码
    （或码内含 boxed）时，训练端 rfind 会取到码内 boxed、eval 端（先剥离）取不到，
    同一轨迹两边对错判定漂移。剥离后代码是脚手架（与 retool 口径一致）：
    码内 boxed 不当答案，末 300 字符窗口也作用于剥离后的真实回答文本。

    【2026-09-12 长度控制】trunc_final（末段被轮长上限切断）叠加 trunc_shaping
    扣分——总长惩罚够不到单轮 prose 轨迹，这是唯一能作用于它的反向信号，
    见 trunc_penalty 的实测依据。trunc_shaping=0 时行为与旧版逐位相同。"""
    base = total_reward_math(ground_truth, strip_code_blocks(answer),
                             completion_len=completion_len,
                             max_gen_tokens=max_gen_tokens, overlong_buffer=overlong_buffer,
                             overlong_shaping=overlong_shaping)
    tp = trunc_penalty(trunc_final, trunc_shaping)
    if tp:
        base["reward"] = base["reward"] - tp
    base["trunc_penalty"] = tp
    # 保留 code 字段供 record 监控，但 reward 不含它
    base["code"] = reward_code(code_ok, 0.0)
    base["code_ok"] = code_ok
    return base
