# -*- coding: utf-8 -*-
"""rlab/health.py — 训练期健康检查：窗口签名 → 异常报警。

【设计依据】本项目四次"训完评测才发现"的 bug 全部存在训练期可观测签名：
  - cispo 策略梯度归零 → 训练 acc 百步平坦（0→300 步纹丝不动）；
  - rfpp 表面解题模式退化 → 训练 acc 见顶后单调下滑；
  - retool 打分域 bug → fmt 信号窗口内恒为常数（组内归一化后梯度恒零）；
  - 权重同步静默失败 → 权重指纹推送间不变 / "model updated" 迟迟不出现。
原则："信号恒为常数"是系统性 bug 的签名；"长期平坦/单调下滑"是浪费 GPU
的签名——两者都该当场停下排查，而不是跑完全程再评测。

用法：生成端每上传一组就 HealthMonitor.observe(...) 聚合一条组级摘要，
距上次检查新攒 check_every 组时 maybe_check() 打印一次新告警（同一告警只报一次）。
规则全部在纯函数 window_check 里，CPU 可测。
"""


def _wmean(vals):
    return sum(vals) / len(vals)


def _wsd(vals):
    m = _wmean(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def window_check(hist, *, retool=False, max_clen=None):
    """对组级摘要序列做一次窗口检查。hist: list[dict(acc, fmt, clen, code_rate)]，
    一条 = 一个组（训练 acc/fmt 为 ±1 口径）。返回 [(code, msg)]。"""
    alerts = []
    n = len(hist)
    if n < 32:
        return alerts
    k = 32
    fmt_m, fmt_sd = _wmean([h["fmt"] for h in hist[-k:]]), _wsd([h["fmt"] for h in hist[-k:]])
    acc_m, acc_sd = _wmean([h["acc"] for h in hist[-k:]]), _wsd([h["acc"] for h in hist[-k:]])

    # --- 签名①：信号死亡（恒定且低位）——retool 打分域 bug 的 fmt 恒 -1。
    # 注意：fmt 恒 +1.0 是格式学满（收敛，健康）；只有"恒定且低位"才是死亡。
    if fmt_sd < 1e-6 and fmt_m < -0.5:
        alerts.append(("fmt_const",
                       f"fmt 信号最近 32 组恒为 {fmt_m:.1f}（低位常数）→ 格式信号死亡"
                       "（打分域/温度 bug 签名，组内归一化后无梯度），建议停止排查打分逻辑"))
    if acc_sd < 1e-6:
        alerts.append(("acc_const",
                       "acc 信号最近 32 组恒为常数 → 正确性信号死亡（组内对比失效？），建议停止排查"))

    # --- 签名②：格式学不动（起步低位且 100 组不涨）——温度混杂事件
    if n >= 100 and fmt_m < -0.5:
        alerts.append(("fmt_low",
                       f"100 组后格式率窗口均值仍 <25%（{fmt_m:.2f} ±1口径）→ 格式学不动，"
                       "查温度/打分域（temp0.9 事件签名）"))

    # --- 签名③：长期平坦 / 单调下滑 —— 零梯度 bug / rfpp 退化
    if n >= 256:
        first_m = _wmean([h["acc"] for h in hist[:k]])
        if acc_m < first_m - 0.10:
            alerts.append(("decline",
                           f"训练 acc 较开局下滑 >5pp（{first_m:.2f}→{acc_m:.2f}）"
                           "→ 退化签名（rfpp 表面解题模式），建议早停"))
        elif acc_m - first_m < 0.02:
            alerts.append(("flat",
                           f"{n} 组后训练 acc 净提升 <1pp（{first_m:.2f}→{acc_m:.2f} ±1口径）"
                           "→ 没有学习签名（梯度死亡/信号死亡），建议停止排查"))

    # --- 签名④：截断坍缩
    if max_clen and _wmean([h["clen"] for h in hist[-k:]]) > 0.95 * max_clen:
        alerts.append(("trunc",
                       f"completion 长度窗口均值顶满上限（>{0.95 * max_clen:.0f}）"
                       "→ 截断坍缩（答案被切、奖励学不到），查生成长度预算"))

    # --- 签名④b：retool 末段截断（轮长上限切断 final 答案——retool 家族真正的
    # 截断失败模式；clen 顶满上限检查探不到它，因为每轮各自 cap 在 round_gen_tokens）
    if retool and _wmean([h.get("trunc_rate", 0.0) for h in hist[-k:]]) > 0.2:
        alerts.append(("retool_trunc",
                       "最近 32 组 >20% 样本的末段被轮长上限切断 → final 答案被截、"
                       "acc 结构性受损，建议调大 round_gen_tokens 或 max_rounds"))

    # --- 签名④c：长度膨胀（2026-09-12 新增，本轮静默跑废的直接签名）
    # 事故形态：outcome-only ±1 奖励下"更长的轨迹答对率 75.4%"（corr(clen,acc)=
    # +0.25~+0.43）→ 组内优势把长度当正确性代理来强化 → clen 中位数从 3072 顶满、
    # 均值 2950→3832 单调爬升 → 尾部撞 max_context_tokens 被 overlong 全丢。
    # 全程没有任何告警：既有四种签名（恒定/平坦/退化/截断）都看不见"均值在爬"。
    # 这条把它们补上，并且只在窗口均值显著高于开局时触发（不误报自然波动）。
    _clen_w = _wmean([h["clen"] for h in hist[-k:]])
    _clen_0 = _wmean([h["clen"] for h in hist[:k]])
    if n >= 96 and _clen_0 > 0 and _clen_w > 1.35 * _clen_0:
        alerts.append(("length_runaway",
                       f"completion 长度窗口均值较开局涨 {( _clen_w / _clen_0 - 1) * 100:.0f}%"
                       f"（{_clen_0:.0f}→{_clen_w:.0f}）→ 长度膨胀签名。outcome-only 奖励下"
                       "长度常与正确性相关，梯度会把它当代理强化。查：①是否有长度反向项"
                       "（trunc_shaping/overlong_shaping 是否真可达）；②预算是否已自洽"
                       "（否则尾部会被整组丢弃，丢弃率随之攀升）"))

    # --- 签名⑤：retool 代码信号未出现（提示性，非致命）
    if retool and n >= 128 and _wmean([h["code_rate"] for h in hist[-k:]]) == 0.0:
        alerts.append(("no_code",
                       "128 组后代码调用率仍为 0 → 代码信号未出现（冷启动权重过稀疏？"
                       "模型从未被奖励写代码），记录在案，验收时 code_rate 指标必然为 0"))

    return alerts


def _fp64_sum(t):
    """float64 分块 |w| 求和（避免 3e8 元素整块 .double() 的内存尖峰）。"""
    flat = t.detach().reshape(-1)
    s = 0.0
    for i in range(0, flat.numel(), 1 << 20):
        s += float(flat[i:i + (1 << 20)].double().abs().sum())
    return s


def weight_fingerprint(state_dict):
    """推送权重的位级敏感指纹（首/中/尾三个张量的 float64 |w| 和）。

    【2026-09-08 真机假阳性教训】指纹必须用 float64：float32 求和在 3e8 元素
    （embed_tokens）上分辨率约 0.5，而 lr=1e-6 下 16 步的 bf16 单权重翻转只有
    ~1e-4，完全淹没在分辨率以下——float32 指纹会对"权重在正常更新"误报
    "权重未变化"。float64 分辨率 ~1e-9，任何一个 bf16 位翻转都能确证。"""
    keys = list(state_dict.keys())
    picks = [keys[0], keys[len(keys) // 2], keys[-1]]
    return tuple(_fp64_sum(state_dict[k]) for k in picks)


class HealthMonitor:
    """滚动收集组级摘要 → 周期性窗口检查 → 同一告警只报一次。"""

    def __init__(self, check_every: int = 16):
        self.hist = []
        self.check_every = check_every
        self.fired = set()
        self._last_check = 0

    def observe(self, acc_list, fmt_list, clen_list, code_used_list=None,
                trunc_list=None):
        """聚合一个组的标量摘要（acc/fmt 为 ±1 口径列表）。trunc_list：每条轨迹
        末段是否被轮长上限切断（0/1，retool 家族；缺省按 0 记）。"""
        e = {"acc": _wmean(list(acc_list)), "fmt": _wmean(list(fmt_list)),
             "clen": _wmean(list(clen_list)),
             "code_rate": (sum(1 for u in code_used_list if u > 0) / len(code_used_list))
             if code_used_list else 0.0,
             "trunc_rate": (_wmean(list(trunc_list)) if trunc_list else 0.0)}
        self.hist.append(e)

    def maybe_check(self, retool: bool = False, max_clen=None):
        """距上次检查新攒了 check_every 组（且 ≥32 组）就检查一次；新告警打印
        （带 [健康检查] 前缀，同一告警只报一次）。
        【2026-09-13 P1 教训】旧版门是 `len(hist) % check_every == 0`，而本方法
        每个外层轮末才被调用一次、外层轮产组数不定 → 多数 16 倍数永远落不上
        检查点（P1 实测：组 112→240 约 1 小时零检查，期间 fmt_low/length_runaway
        的触发条件真的成立过，却从未被报告）。改滚动门后，稀疏调用也保证
        每 check_every 组至少检查一次。"""
        n = len(self.hist)
        if n < 32 or n - self._last_check < self.check_every:
            return
        self._last_check = n
        for code, msg in window_check(self.hist, retool=retool, max_clen=max_clen):
            if code not in self.fired:
                self.fired.add(code)
                print(f"\n[健康检查] {msg}\n", flush=True)
