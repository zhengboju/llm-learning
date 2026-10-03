# -*- coding: utf-8 -*-
"""rlab/rollout.py — 生成端 worker（vLLM 采样 + torch gen_logps 副本 + 上传）。

职责（阶段0 范围 = 单轮 GSM8K 式任务；阶段2 多轮工具调用在此文件扩展）：
  1. vLLM 批量采样每题 num_pre_Q 条；
  2. torch 模型副本前向算 gen_logps（vLLM prompt_logprobs 路径在本环境会 hang——实测教训）；
  3. rlab.reward 打分 -> rlab.losses.compute_advantages 归一化；
  4. protocol 打包上传 ref_server；
  5. mp.Queue 收训练端权重，rlab.sync 同步进 vLLM（fail-fast）。

上传契约：
  非 rfpp : [meta, merged_ids, advantages(B,), gen_logps, acc, fmt]
  rfpp    : [meta, merged_ids, raw_rewards(B,), gen_logps, acc, fmt]
            —— advantage 由 ref_server 的 macro-batch 路径计算（eos-mask + 反向
               cumsum 信用分配 + 全局标准化 + num_items），是 RF++ 算法定义的一部分

【工程教训内置】
- 清除 DeepSpeed 分布式环境变量，避免 vLLM 子进程冲突；
- gen_logps 副本与 vLLM 必须同权重（同步顺序：先 vLLM 后 torch 副本）；
- 同步失败 raise，绝不静默吞掉（生成器冻结在 base 的废跑教训）；
- VLLM_ENABLE_V1_MULTIPROCESSING=0 等 env 在本进程内、import vllm 前主动 setdefault。

独立运行（分进程模式）：
    python -m rlab.rollout --algo dapo --gen_device 0
"""

import json
import os
import queue as _queue
import random
import time
from collections import deque

# vLLM 相关环境变量必须在实际 import vllm 之前设置（教训：RPC 模式权重同步会 stall）
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # EngineCore 进程内运行，无 RPC 序列化
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from torch.nn.utils.rnn import pad_sequence

from rlab.config import get_config
from rlab.data import filter_qas_by_difficulty, load_difficulty_table, load_qas
from rlab.health import HealthMonitor as _HealthMonitor
from rlab.health import weight_fingerprint as _weight_fingerprint
from rlab.losses import compute_advantages, forward_per_token_logps
from rlab.model_loading import load_causal_lm, resolve_load_config
from rlab.protocol import (CODE_TOOL, NATIVE_BAD_WORDS, NATIVE_CALL_STOP,
                           NATIVE_STYLE_FUNCTION, NATIVE_STYLE_JSON,
                           RETOOL_STOP_KWARGS as _RETOOL_STOP_KWARGS,
                           TOOL_END, TOOL_START, assert_native_sampling_ban,
                           build_next_prompt, encode_batch,
                           extract_python_blocks, initial_messages, make_call_id,
                           make_bytes_list, native_special_ban_words,
                           parse_assistant, render_chat_ids,
                           sanitize_tool_text, segment_mask_from_spans,
                           tensor_to_bytes, tool_message,
                           # 【2026-10-02 trunc_in_call】调用块的开/闭标记**从协议
                           # 正则派生**（`_RE_TOOL_CALL_ANY.pattern.split(".*?")`），
                           # 绝不手写标签字面量——本项目三次被会话管道改写字节。
                           _TOOL_OPEN as _CALL_OPEN, _TOOL_CLOSE as _CALL_CLOSE)
from rlab.reward import (overlong_ref_tokens, reward_phase, total_reward,
                         group_eff_bonus,
                         total_reward_math, total_reward_retool,
                         total_reward_retool_math, group_length_penalty)
from rlab.sandbox import run_code
from rlab.sync import need_text_to_mm_remap, remap_text_to_multimodal, sync_weights_into_vllm

# 工具协议档位（docs/09-native-tool-protocol.md）：
#   "fence"  = p1–p11 的 python 围栏 + [TOOL RESULT] 文本回填（逐位可复现，回退位）
#   "native" = Qwen 原生 <tool_call> 工具协议（方案 A 主线）
TOOL_PROTOCOL_FENCE = "fence"
TOOL_PROTOCOL_NATIVE = "native"

# 效率项"开了开关却没接线"的告警去重标志（与 protocol._RENDER_TOLERANCE_WARNED
# 同一约定：这类提示只报一次，否则每组合刷一行、真信号被淹掉）。
_EFF_WARNED = [False]

# 清除分布式环境变量（gen worker 进程内 vLLM 不允许看到 DeepSpeed 的 WORLD_SIZE 等）
_DEEPSPEED_ENV_KEYS = [
    "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK",
    "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
    "GROUP_WORLD_SIZE", "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_COMM_ID", "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
]


def build_prompt(question: str, system_prompt: str, tokenizer,
                 chat_template_kwargs: dict | None = None,
                 tools: bool = False) -> str:
    """单轮 prompt 模板。

    chat_template_kwargs 透传 apply_chat_template（Qwen3.5 系需
    {"enable_thinking": false}，见 config.chat_template_kwargs 注释）。
    tools=True 时把 CODE_TOOL 声明进模板（原生协议用，docs/09 §2.2）。
    默认 False —— 围栏家族（p1–p11）的 prompt 逐字节不变，回退位真的能回退。"""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
        **({"tools": [CODE_TOOL]} if tools else {}),
        **(chat_template_kwargs or {}))


def build_prompt_ids(messages: list, tokenizer, chat_template_kwargs: dict | None = None,
                     tools: bool = False) -> list:
    """多轮消息 → prompt token ids（**生成端唯一入口**，与 build_next_prompt 同源）。

    【为什么必须单独一个函数】原生协议下 build_next_prompt 内部会用
    apply_chat_template(tokenize=True) 反推 canonical 序列，若生成首轮的 prompt
    是另一条路径（如先 tokenize=False 再 tokenizer()），两条路径的 token 序列
    可能不同源（BPE 边界/特殊 token 处理），build_next_prompt 的前缀校验会直接
    raise。所有 prompt 一律走这里，契约才成立。"""
    return render_chat_ids(tokenizer, messages, True, chat_template_kwargs, tools=tools)


def prompt_messages_for(inputs, cfg: dict) -> list:
    """每题一条初始消息（原生协议的唯一来源，纯函数）。

    `build_prompt_batch`（生成端 ids）与 `collect_retool_group`（原生多轮
    messages）都从这里取——两处若各自拼一遍，一旦分叉就是"喂给 vLLM 的序列"
    与"续写用的 messages"不同源，`build_next_prompt` 的前缀校验会炸在几十步后
    的运行期而不是启动期。"""
    return [initial_messages(cfg["system_prompt"], x["Q"]) for x in inputs]


def build_prompt_batch(inputs, cfg: dict, tokenizer, prompts_messages=None):
    """按当前协议档位构造一批 prompt（生成端**唯一**入口，纯调度无 GPU）。

    返回 (prompts_text, prompt_ids, plen)：
      · 围栏档：prompts_text = 模板文本；prompt_ids = 文本批量左 pad tokenize
        （历史路径逐字不变，p1–p11 可复现）；prompts_messages 被忽略。
      · 原生档：prompts_text 仅作日志兜底；prompt_ids 由 `build_prompt_ids`
        （= apply_chat_template(tokenize=True, tools=...)）产出并手工左 pad。

    【为什么必须在这里分叉】`prompt_ids` 不只是"喂给 vLLM 的输入"——它同时是
    `strip_left_pad` → `retool_build_batch` 里 merged 序列的 prompt 段。两条协议
    的 prompt 渲染不同（原生档多一个 tools 声明段），若 prompt_ids 用错档位，
    merged 序列与生成序列**逐 token 不同源** → gen_logps 基线失真（这正是
    docs/09 §3 与 2026-09-09 同序列契约要防的同一类错，只是换了个入口）。

    【prompts_messages 参数的作用】调用方（gen_worker 主循环）先算一次
    `prompt_messages_for`，把**同一份**消息同时喂给本函数（出 ids）与
    `collect_retool_group`（原生多轮用）——两处各自拼一遍也能对，但那靠的是
    "两处代码恰好一致"这种约定；显式传同一对象才是结构性保证（一旦分叉，
    build_next_prompt 的前缀校验会炸在运行期几十步后，而不是启动期）。"""
    native = is_native_protocol(cfg)
    ctkw = cfg.get("chat_template_kwargs")
    if native:
        msgs = prompts_messages or prompt_messages_for(inputs, cfg)
        ids = [build_prompt_ids(m, tokenizer, ctkw, tools=True) for m in msgs]
        # 文本仅供日志/记录（原生档实际喂 vLLM 的是 token ids）
        texts = [tokenizer.decode(t, skip_special_tokens=False) for t in ids]
        plen = max(len(t) for t in ids)
        pad_id = tokenizer.pad_token_id
        padded = torch.full((len(ids), plen), pad_id, dtype=torch.long)
        for r, t in enumerate(ids):
            padded[r, plen - len(t):] = torch.tensor(t, dtype=torch.long)
        return texts, padded, plen
    texts = [build_prompt(x["Q"], cfg["system_prompt"], tokenizer, ctkw) for x in inputs]
    prompt_ids = tokenizer(texts, return_tensors="pt", padding=True,
                           padding_side="left", add_special_tokens=False)["input_ids"]
    return texts, prompt_ids, prompt_ids.shape[1]


def group_ok(scores: torch.Tensor) -> bool:
    """序列级要求组间有方差；逐tokenprocess reward只要存在非零梯度即上传。"""
    if scores.dim() == 1:
        return (scores.max() - scores.min()).item() >= 1e-4
    if scores.dim() == 2:
        # (B,T) 含绝对的工具动作成本，不应再被组均值规则抵消：即使整组都做了
        # 同一种浪费调用，也有“降低该动作概率”的有效process-reward梯度。
        return scores.abs().max().item() >= 1e-4
    raise ValueError(f"组分数必须是(B,)或(B,T)，收到 {tuple(scores.shape)}")


def sampling_discard_counts(attempts: int, uploaded: int, uniform: int,
                            overlong: int, prompt_overlong: int):
    """返回（真实丢弃数，未归因丢弃数），统一采样统计与熔断口径。

    真实丢弃必须按 attempts-uploaded 算；只加已知原因会漏掉新分支。2026-09-30
    native_p4 实测 464 attempts / 368 uploaded / 76 uniform / 0 overlong，旧日志
    报 16%，实际还有 20 次 prompt 超限，真实丢弃率是 96/464=20.7%。
    """
    vals = tuple(int(x) for x in
                 (attempts, uploaded, uniform, overlong, prompt_overlong))
    if any(x < 0 for x in vals):
        raise ValueError(f"sampling stats 不能为负数: {vals}")
    attempts, uploaded, uniform, overlong, prompt_overlong = vals
    discarded = attempts - uploaded
    attributed = uniform + overlong + prompt_overlong
    if discarded < 0 or attributed > discarded:
        raise ValueError(
            "sampling stats 账目不自洽: "
            f"attempts={attempts} uploaded={uploaded} discarded={discarded} "
            f"uniform={uniform} trajectory_overlong={overlong} "
            f"prompt_overlong={prompt_overlong}")
    return discarded, discarded - attributed


def filter_question_pool(QAs, q_stat: dict, streak_max: int, floor: int):
    """题目级动态采样（2026-09-09 丢弃率 81% 根因修复，纯函数 CPU 可测）。

    背景：retool_math outcome-only ±1 下，3B base 通过率 p≈5%，组存活率
    1-(1-p)^n-p^n ≈ 19%。关键数学：动态采样下每组有效产出的期望轨迹数 ≈ 1/p，
    **与 num_pre_Q 无关**（n 翻倍存活率翻倍但每次成本也翻倍）——调大 n 省不了。
    唯一有效的杠杆是把算力集中到"当前学得动"的题上：连续 streak_max 次产出
    零方差组（当前全错/全对，永远没有梯度）的题跳过；池子低于 floor 时全部
    重置（模型变强后难题重新入场，也防止池子枯竭）。
    返回 (候选池, 是否触发了重置)。"""
    cand = [q for q in QAs if q_stat.get(q["Q"], 0) < streak_max]
    if len(cand) < floor:
        return list(QAs), True
    return cand, False


class QuestionScheduler:
    """题目级调度器（2026-09-10 训练变慢根因修复）。

    【旧实现为何是死代码】主循环 random.sample 从全池抽题——16.5k 题的池子
    同题被再次抽中的概率 1/16500，q_skip_streak=2 的"两次零方差"条件几乎
    永远无法对同一题累计 → 黑名单永远建不起来 → 丢弃率维持裸 P(uniform)≈64%，
    "81%→20-30%" 的预期落空（真机"训练更慢"的第一根因）。

    【新语义】shuffle 队列顺序走池：
    - 每题一次 attempt；uniform 组 → streak+1 且插回队首（尽快重试），
      streak 达到 streak_max 即拉黑（不再入队，filter_question_pool 在补给时排除）；
    - 有效组 → streak 清零；超长组 → 与题目难度无关，不计 streak 随队列轮转；
    - 队列耗尽时按 filter_question_pool 补充；候选低于 floor 全量重置
      （难题随模型变强重新入场）。
    "连续"按"自上次成功以来累计 uniform 数"计——两次 attempt 之间可能隔一次
    补给重排，语义近似但不影响黑名单的收敛速度（队首插入保证难题两轮内出局）。
    rng 默认用全局 random（gen_worker 已按 seed 播种 → 抽题顺序可复现）。"""

    def __init__(self, QAs, streak_max, floor, rng=None, ttl: int = 0):
        self.QAs = QAs
        self.streak_max = max(1, int(streak_max))
        self.floor = max(1, int(floor))
        self.rng = rng if rng is not None else random
        # 【2026-09-23 黑名单 TTL】拉黑不是永久的：累计被报 uniform/overlong 达
        # ttl 次后 streak 清零重新入场（难题随训练推进重新可学）。0 = 关闭
        # （旧行为：streak 永久累计直到 floor 全量重置）。
        self.ttl = max(0, int(ttl))
        self._age = {}   # Q -> 自拉黑以来的 report 次数
        self._ttl_released = 0   # TTL 释放累计（观测）
        self.q_stat = {}
        self.queue = deque()

    def _tick_blacklist_ttl(self):
        """拉黑题 TTL 计龄（在 draw 入口调用）：每次 draw 对所有已拉黑题 +1，
        累计 ttl 次 draw 后释放（streak 清零 + 立即入队）。语义 = "拉黑后再采
        ttl 轮题就给它重新入场的机会"，难题随训练推进重新可学。
        【为什么挂在 draw 而不是 report】拉黑题被 draw 跳过、filter_question_pool
        也不会把它排进队列——report 对拉黑题永远不会被调用，挂在那里是死代码。
        draw 调用粒度 = 每次 attempt 一轮，与 attempt 数同量纲、可预期。"""
        if not self.ttl:
            return
        for key in [k for k, v in self.q_stat.items() if v >= self.streak_max]:
            n = self._age.get(key, 0) + 1
            if n >= self.ttl:
                self.q_stat[key] = 0
                self._age.pop(key, None)
                self._ttl_released += 1
                for _q in self.QAs:
                    if _q["Q"] == key:
                        self.queue.append(_q)   # 立即入队，不等下次 refill
                        break
                if self._ttl_released % 10 == 1:
                    print(f"[rollout] 黑名单到期释放（累计 {self._ttl_released} 题）："
                          "难题随训练推进重新入场", flush=True)
            else:
                self._age[key] = n

    def _refill(self):
        cand, reset = filter_question_pool(
            self.QAs, self.q_stat, self.streak_max, self.floor)
        if reset:
            self.q_stat.clear()
            print("[rollout] 题目级过滤池低于下限，全部重置（难题重新入场）", flush=True)
        pool = cand[:]
        self.rng.shuffle(pool)
        self.queue = deque(pool)

    def draw(self, k):
        """取 k 道题；跳过已达拉黑阈值却仍留在队列里的残留（首 uniform 插队
        后第二次也 uniform 的题，队列里已无它——此分支只防极端交错）。"""
        out = []
        for _ in range(3):        # 补给上限：重置后为全池，两轮必够
            self._tick_blacklist_ttl()   # 拉黑题 TTL 计龄（每次 draw 一次）
            while len(out) < k and self.queue:
                q = self.queue.popleft()
                if self.q_stat.get(q["Q"], 0) >= self.streak_max:
                    continue
                out.append(q)
            if len(out) >= k:
                break
            self._refill()
        return out

    def report(self, q, status, count_overlong: bool = True):
        """attempt 结果回填：status ∈ {"uniform", "ok", "overlong"}。

        【2026-09-12 反压修复】旧版对 "overlong" 什么都不做：题目从队列 pop 出来后
        既不拉黑也不插回，`_refill()` 又把整池放回 → 当模型整体变长、几乎每题都
        判超长时，采样主循环就对着同一批题无限空转（真机事故：丢弃率 90%、日志里
        18274 行 "waiting for batch..."、末尾 5 小时零产出，且没有任何告警）。
        现在超长也计 streak（**不插队首**——超长不是"值得立刻重试"，而是"这题当前
        预算装不下"，攒够 streak_max 就暂时移出池子），让题池停止原地打转。
        count_overlong=False 可退回旧语义（对照实验用）。"""
        key = q["Q"]
        if status == "uniform":
            self.q_stat[key] = self.q_stat.get(key, 0) + 1
            if self.q_stat[key] < self.streak_max:
                self.queue.appendleft(q)   # 未达标：插队首，下一 attempt 尽快重试
            # 达标：不入队 = 拉黑（blacklisted_count / 下次补给排除）
        elif status == "overlong":
            if count_overlong:
                self.q_stat[key] = self.q_stat.get(key, 0) + 1
        elif status == "ok":
            self.q_stat[key] = 0

    def blacklisted_count(self):
        return sum(1 for v in self.q_stat.values() if v >= self.streak_max)


def sampled_logps_from_output(out, ids) -> list:
    """从 vLLM 单条 completion 提取"被采样 token"的逐 token logprob。

    【减法① 2026-09-11】SamplingParams(logprobs=N) 时 vLLM 为每个生成位置返回
    {token_id: Logprob}；取其中实际采样到的那个 token 的 logprob，数学上就是
    log π(tok | 完整前文)——多轮场景下每轮请求的上下文已包含之前所有轮的工具结果，
    所以逐轮收集再拼接 == compute_gen_logps 在拼接序列上重算，严格同义，且完全
    绕开 prompt_logprobs 路径（那条路在本环境会 hang，是 torch 副本存在的起因）。
    任何一步对不齐都 raise：静默错位会让 gen_logps 全错而训练照跑。

    【2026-09-15 真机实锤：N=0 这条形态本身不可信】同一 prompt、同一位置、同一
    token 上，N=0 报 -0.602，而 N=20（raw）报 -7.40、torch 独立重算 -7.400。
    N=0 是唯一离群者 → vLLM 的"只报被采样 token"路径报错了数，与分布无关
    （同批探针 top-1 36/36 相同、top-20 交集 0.968）。改用 N≥1（top-K 里挑被采样
    token）后与 torch 的差回到 mean 0.02/max 0.23 量级。cf. cfg `vllm_logprobs_n`。"""
    lps = getattr(out, "logprobs", None)
    if lps is None:
        raise RuntimeError(
            "[rollout] vLLM 未返回 logprobs（SamplingParams 缺 logprobs=N）——"
            "vllm_gen_logps 路径要求采样时就带上该参数")
    if len(lps) != len(ids):
        raise ValueError(
            f"[rollout] logprobs 条数 {len(lps)} != 采样 token 数 {len(ids)}"
            "（位置对不齐，gen_logps 会全错）")
    vals = []
    for pos, tid in enumerate(ids):
        entry = lps[pos]
        if not entry:
            raise ValueError(f"[rollout] 第 {pos} 个位置没有 logprobs（entry={entry!r}）")
        lp = entry.get(tid)
        if lp is None:
            raise ValueError(
                f"[rollout] 第 {pos} 个位置缺被采样 token {tid} 的 logprob"
                f"（可用键：{list(entry)[:4]}）——logprobs=N≥1 时 vLLM 应始终把被采样"
                "token 放进返回字典；若这里报缺，说明该版本行为不同，"
                "别退回 N=0（那条路径报的数已被实锤不可信，见函数 docstring）")
        vals.append(float(getattr(lp, "logprob", lp)))
    return vals


def logps_diff_shape(gv, gt, mask, k: int = 3) -> dict:
    """对拍差异的**形态学**指标（纯张量函数，CPU 可测）。

    【为什么不能只看 mean/p50/p99/max】2026-09-15 实机对拍出现 max=12.8，而
    "12.8 是尾部低概率 token 的舍入"这个解释被同一次数据推翻：该点 vLLM 的
    logp=-0.946（p≈0.39）、torch=-13.75（p≈1e-6）——一个被采样 token 两侧分布
    实质不同，bf16 舍入（logit 级 ~0.1）物理上给不出 12.8 nat。要分辨"状态漂移"
    与"局部 kernel 尖峰"，必须再要三样东西：
      · frac>1 / frac>0.1：越线点的**占比**（单点 max 无法区分"1 个怪点"和"1% 全歪"）；
      · 前后半段 mean|d|：**逐行**按该行有效位中点切（不按全局列切，防长行主导与
        右 pad 混入）。后半段显著更大 = 递归状态累积漂移；两半相当 = 局部尖峰；
      · 最差 k 点（行/列/两路原值）：尖峰是否紧贴工具段边界、是否落在列 0，
        一眼可读——这正是需要回看的东西，事后不该只剩一个 max。

    两路原值都取 float（gv 已是 float32，gt 上转），避免"比较时的 dtype 又把
    两侧一起量化"这个旧坑（见 gen_logps_from_segs docstring）。"""
    m = mask.bool()
    gv_f, gt_f = gv.float(), gt.to(gv.dtype).float()
    d = (gv_f[m] - gt_f[m]).abs()
    out = {"n": int(d.numel())}
    if d.numel() == 0:
        return out
    out["mean"] = float(d.mean())
    out["p50"] = float(d.median())
    out["p99"] = float(d.kthvalue(max(1, int(d.numel() * 0.99))).values)
    out["max"] = float(d.max())
    out["frac_gt_01"] = float((d > 0.1).float().mean())
    out["frac_gt_1"] = float((d > 1.0).float().mean())
    first, second = [], []
    for r in range(int(m.shape[0])):
        cols = m[r].nonzero().flatten()
        if cols.numel() < 2:
            continue
        mid = cols.numel() // 2
        row_d = (gv_f[r] - gt_f[r]).abs()
        first.append(float(row_d[cols[:mid]].mean()))
        second.append(float(row_d[cols[mid:]].mean()))
    out["half_mean_first"] = (sum(first) / len(first)) if first else None
    out["half_mean_second"] = (sum(second) / len(second)) if second else None
    flat = m.nonzero()
    vals, idx = d.topk(min(int(k), int(d.numel())))
    out["worst"] = [{"row": int(flat[i][0]), "col": int(flat[i][1]),
                     "vllm": float(gv_f[flat[i][0], flat[i][1]]),
                     "torch": float(gt_f[flat[i][0], flat[i][1]]),
                     "d": float(v)} for v, i in zip(vals.tolist(), idx.tolist())]
    return out


class LogpsVerifier:
    """vLLM 路 vs torch 路 gen_logps 对拍器（减法① 的口径验证闸门）。

    wants() 是**调用前**的闸门：预算用尽后返回 False，调用方据此不再触发 torch
    重算——否则 verify 结束释放 torch 副本后，钩子仍会被调用而 raise。on_finish
    在最后一组比对完成后执行（gen_worker 用它把副本还给 GPU0）。
    """

    def __init__(self, budget: int, on_finish=None):
        self.budget = int(budget)
        self.on_finish = on_finish
        self.n = 0
        self.max_diff = 0.0
        self.stats = []          # 每组的形态学指标（报告/事后分析用，见 logps_diff_shape）

    def wants(self) -> bool:
        return self.n < self.budget

    def __call__(self, gv, gt, mask):
        if not self.wants():
            return
        self.n += 1
        m = mask.bool()
        # 只报 max 会掩盖病因（2026-09-11 首轮对拍 max=11.3 但不知道是"整体平移"
        # 还是"个别离群"）——补分布统计与最大差位置，一次对拍就能定性
        d = (gv[m].float() - gt.to(gv.dtype)[m].float()).abs()
        if d.numel() == 0:
            print(f"[rollout][verify] 第 {self.n}/{self.budget} 组：无有效位，跳过", flush=True)
            return
        mx = d.max().item()
        mean = d.mean().item()
        p50 = d.median().item()
        p99 = d.kthvalue(max(1, int(d.numel() * 0.99))).values.item()
        pos = m.nonzero()[d.argmax()].tolist() if d.numel() else [-1, -1]
        self.max_diff = max(self.max_diff, mx)
        print(f"[rollout][verify] 第 {self.n}/{self.budget} 组：vLLM vs torch gen_logps "
              f"| diff mean={mean:.3e} p50={p50:.3e} p99={p99:.3e} max={mx:.3e}"
              f"（有效位 {int(m.sum())}，最大差在 [行{pos[0]}, 列{pos[1]}]，"
              f"vLLM={float(gv[m][d.argmax()]):.3f} torch={float(gt.to(gv.dtype)[m][d.argmax()]):.3f}）",
              flush=True)
        # 形态学第二行（2026-09-15）：单点 max 定不了责，占比/前后半段/最差 k 点才能
        st = logps_diff_shape(gv, gt, m)
        self.stats.append(st)
        _half = ("None" if st.get("half_mean_first") is None
                 else f"{st['half_mean_first']:.2e}→{st['half_mean_second']:.2e}")
        _worst = "  ".join(
            f"[行{w['row']},列{w['col']} v={w['vllm']:.3f} t={w['torch']:.3f} Δ={w['d']:.2f}]"
            for w in st.get("worst", []))
        print(f"[rollout][verify]   形态: >0.1={st['frac_gt_01']:.2%} "
              f">1={st['frac_gt_1']:.2%} 前后半段 mean|d| {_half} 最差{len(st.get('worst', []))}点 {_worst}",
              flush=True)
        if not self.wants() and self.on_finish is not None:
            self.on_finish()


def gen_logps_from_segs(segs, pad_value: float = 0.0):
    """由每段记录的被采样 logprob 拼出 (B, T) 的 gen_logps（与 mask 同布局）。

    工具段 token 是环境插入的、vLLM 从未采样过（无从给出 logprob），置 0——它们在
    mask 里恒为 0 不进 loss，ratio 的有效位统计也被 mask 过滤，唯一要求是取值有限
    （policy_logps ≤ 0 → exp(·) ≤ 1，不溢出）。

    【2026-09-12 dtype 修复：bf16 → float32】vLLM 返回的 logprob 是 float32 精度，
    旧版在这里降到 bf16，引入 `|δ| ≈ |logp|·2⁻⁹` 的逐 token 量化误差（|logp|=5 时
    ~0.01、|logp|=30 时 ~0.06）。后果有两层：
      ① ratio 上凭空多了 ~1% 的噪声，低频 token 尤甚（正是对拍里 p99=0.109 /
         max=0.77 那批），给策略梯度注入与策略无关的扰动；
      ② **approx_kl 出现一个 ~1.5e-4 的硬地板**（step 1 实测 5.11e-4 里绝大部分
         来自它），让人无法判断后面 0.01~0.10 的读数里有多少是真实 off-policy。
    存储代价可忽略（B=8、T=8000 的 (B,T) float32 仅 256KB），而精度收益直接决定
    诊断能否用。对拍器注意：LogpsVerifier 里 `gt.to(gv.dtype)` 会把 torch 路也降到
    bf16——改 dtype 后两边都是 float32，报出的 diff 才是真实实现口径差（旧版的
    对拍结果被 bf16 量化同时污染了两侧）。"""
    rows = []
    for segs_i in segs:
        vals = []
        for seg in segs_i:
            if seg["kind"] == "assistant":
                lp = seg.get("logps")
                if lp is None or len(lp) != len(seg["ids"]):
                    raise ValueError(
                        "[rollout] assistant 段缺 logps 或与 ids 长度不齐——"
                        "采样时未开 collect_logps/logprobs？")
                vals.extend(lp)
            else:
                vals.extend([pad_value] * len(seg["ids"]))
        rows.append(torch.tensor(vals, dtype=torch.float32))
    return pad_sequence(rows, batch_first=True, padding_value=pad_value)


def tool_protocol_of(cfg: dict) -> str:
    """当前 run 的工具协议档位（纯函数，CPU 可测）。缺键 = "fence"（历史行为）。

    【为什么做成运行时开关而不是重写】docs/09 §8 的回退设计：`"fence"` 必须
    逐位复现 p1–p11，协议本身才是**可 A/B 的单变量**；一次性重写会让"原生协议
    到底有没有用"这个问题永远无法在同一个代码库里回答。"""
    v = cfg.get("tool_protocol") or TOOL_PROTOCOL_FENCE
    if v not in (TOOL_PROTOCOL_FENCE, TOOL_PROTOCOL_NATIVE):
        raise ValueError(f"[rollout] 未知 tool_protocol={v!r}，可选 "
                         f"{TOOL_PROTOCOL_FENCE!r} / {TOOL_PROTOCOL_NATIVE!r}")
    return v


def is_native_protocol(cfg: dict) -> bool:
    return tool_protocol_of(cfg) == TOOL_PROTOCOL_NATIVE


def multi_turn_rollout_group_native(vllm_gen, sampling_params, tokenizer,
                                    prompts_messages, cfg, code_runner=run_code,
                                    collect_logps: bool = False):
    """原生 `<tool_call>` 协议的多轮工具调用（docs/09 方案 A 主线）。

    与围栏版的**结构差异**（其余全部同构，便于 A/B 对照）：
      · 检测：`parse_assistant`（正则从实测渲染派生）替代围栏正则；
      · 续写：`build_next_prompt` 的 token 级增量取代"拼 [TOOL RESULT] 文本"；
      · 停止：**不需要 stop 串**——模型自然停在 im_end（EOS 已在 token_ids 里，
        docs/05 §12.3 已核对）；`cfg["native_stop_at_call"]` 只在 base 冒烟发现
        "调用后又继续瞎写"时才打开（那是原生协议唯一的结构性负奖励入口）；
      · 预算：`max_rounds` 轮，前 `max_rounds-1` 轮可执行代码（末轮保证是 final
        轮，docs/02 发现3 的语义在原生协议下同样成立）。

    参数同围栏版，只是 `prompts_messages` 是 `[[{role,...},...], ...]`——原生协议
    必须从 messages 渲染（tools 声明在模板里），不能像围栏版那样只吃文本。

    返回 (segs, full_text, code_stats)：结构与围栏版**完全一致**
    （`retool_build_batch`/打分/上传三处零改动，这正是 docs/09 §4 的契约）。"""
    n = len(prompts_messages)
    ctkw = cfg.get("chat_template_kwargs")
    style = cfg.get("native_tool_style") or "auto"
    if style not in ("auto", NATIVE_STYLE_FUNCTION, NATIVE_STYLE_JSON):
        raise ValueError(f"[rollout] 未知 native_tool_style={style!r}（可选 auto/"
                         f"{NATIVE_STYLE_FUNCTION}/{NATIVE_STYLE_JSON}）")
    # 假 </think> + special token 禁言 fail-fast（事故 C/D，docs/09 §10.6.3/§10.6.4）：
    # 缺禁言时"会不会炸校验①"是随机事件（哪一条采样在第几步吐出 </think> 或
    # special token），真机形态是运行几十/上百步后 abort——入口断言把失败提前到
    # 第 1 组之前。传 tokenizer = 连 special 禁言一起查（事故 D）。
    assert_native_sampling_ban(sampling_params, tokenizer)
    budget = int(cfg.get("max_context_tokens", 8192))
    max_rounds = int(cfg.get("max_rounds", 5))
    max_code_calls = max(0, max_rounds - 1)      # 末轮不许执行代码（发现3）
    # 【2026-09-29 token 预算档】max_traj_tokens>0 时三条语义同时切换（见 config.BASE
    # 同名注释）：预算判据取代末轮判据、单轮切断改为续写、回包注入剩余额度。
    # 缺省 0 → tok_mode=False，本函数对历史档逐位不变（A/B 对照位）。
    traj_budget = int(cfg.get("max_traj_tokens", 0) or 0)
    tok_mode = traj_budget > 0
    _hint = bool(cfg.get("budget_hint", False)) and tok_mode
    # 每样本状态：messages（协议用）/ prompt_ids（生成端喂 vLLM 的真实序列）/ 调用数
    msgs = [list(m) for m in prompts_messages]
    ctx_ids = [build_prompt_ids(m, tokenizer, ctkw, tools=True) for m in msgs]
    segs = [[] for _ in range(n)]
    code_stats = [{"code_used": 0, "code_ok": 0, "code_wasted": 0,
                   "invalid_final": 0, "ctx_full": 0, "trunc_final": 0,
                   # 【2026-10-02】"预算耗尽时正处在未写完的调用块里"（见
                   # trunc_in_call_flag）：A 桶（trunc∩invalid）在 token 档恒 0，
                   # 没有这个字段就无法把"调用写到一半被墙切"从 B 桶里分出来。
                   "trunc_in_call": 0, "err_types": []}
                  for _ in range(n)]
    # 【2026-09-29 token 预算档·续写缓冲】"一次生成"与"一个 assistant 轮"在这一档
    # 下解耦：被单轮上限切断（finish_reason=length）时该轮**没有结束**，累积进
    # open_ids 继续写，直到自然收尾或预算耗尽才落成一个 assistant 段。
    #
    # 三个上下文的职责必须分清（搞混会产生"序列重复计入"或"校验① 判不同源"）：
    #   ctx_ids  = 当前**完整**上下文（含本轮已生成部分），下一轮生成喂它；
    #   turn_ctx = 本轮**起点**上下文（轮内不变）——build_next_prompt 的 prev 必须是
    #              它，传 ctx_ids 会把本轮 comp_ids 重复计入序列；
    #   open_ids = 本轮至今累积的生成 token（跨续写块）。
    # 非 tok_mode 档下 open_ids 每轮清空、turn_ctx 恒等于轮结束时的 ctx_ids →
    # 每轮恰好落一段，与旧版逐位相同（历史档可复现的保证）。
    open_ids = [[] for _ in range(n)]
    open_raw = ["" for _ in range(n)]
    open_lps = [[] for _ in range(n)]
    turn_ctx = [list(c) for c in ctx_ids]
    p0 = [len(c) for c in ctx_ids]      # 各样本真实 prompt 长（逐条构造，无 pad）

    def _used_tokens(i):
        """本样本已消耗的轨迹预算（prompt 之外的全部：assistant + 工具段）。"""
        return len(ctx_ids[i]) - p0[i]

    # 【循环上限只是安全阀】真正的约束是 max_traj_tokens。轮数不该再当预算用——
    # 5 轮→6 轮实测无收益的真因是 answer/invalid 为**终局**、剩余轮数被整段作废
    # （不是轮数本身不够）。上限只防"每轮只花 1 token"的病态循环。
    _iter_cap = max(8, max_rounds * 4) if tok_mode else max_rounds
    active = list(range(n))
    for _rnd in range(_iter_cap):
        if not active:
            break
        if tok_mode:
            # 预算耗尽的样本先出局，并打上 trunc_final——这正是 token 预算档下
            # trunc_final 的正确定义："预算用尽而未能收尾"（单轮上限被切断已改为
            # 续写，不再是终局，故旧的 finish_reason 口径在这一档不再适用）。
            _still = []
            for i in active:
                if _used_tokens(i) < traj_budget:
                    _still.append(i)
                    continue
                code_stats[i]["trunc_final"] = 1
                # 【必须先把续写缓冲落段】预算耗尽的典型路径正是"最后一次生成被
                # 单轮上限切断"——此时 open_ids 里的 token 是模型真实采样过的输出，
                # 不落段就会凭空丢失（打分看不到可能已经写出的 boxed → 恒 -1，
                # 且与生成端实际序列不符）。
                if open_ids[i]:
                    _s = {"kind": "assistant", "text": open_raw[i],
                          "ids": list(open_ids[i]), "finish_reason": "length"}
                    if collect_logps:
                        _s["logps"] = list(open_lps[i])
                    segs[i].append(_s)
                    # 【2026-10-02】切点是不是落在**未闭合的调用块**里——必须在这里
                    # 判：本分支只置 trunc_final，A 桶（trunc∩invalid）在 token 档
                    # 恒 0，不单独记就永久分不出"调用被墙切"与"纯散文被墙切"。
                    code_stats[i]["trunc_in_call"] = trunc_in_call_flag(
                        open_raw[i], style)
                    open_ids[i], open_raw[i], open_lps[i] = [], "", []
            active = _still
            if not active:
                break
        is_final_round = (_rnd == max_rounds - 1) and not tok_mode
        if isinstance(sampling_params, list):
            sps = [sampling_params[i] for i in active]
            if tok_mode:
                # 把本轮生成长度夹到剩余预算：不夹的话模型能越过 max_traj_tokens，
                # 让整题撞 retool_context_overlong 被丢弃（白跑一整题）。
                for _sp, _i in zip(sps, active):
                    _sp.max_tokens = max(1, min(
                        int(cfg.get("round_gen_tokens", 400) or 400),
                        traj_budget - _used_tokens(_i)))
        else:
            sps = sampling_params
            if tok_mode:
                # 单对象档（eval 贪心）：无法逐样本夹，用"整体预算"作静态上界。
                # 训练档恒为列表（逐样本 seed），走上面的精确夹取。
                sps.max_tokens = max(1, min(
                    int(cfg.get("round_gen_tokens", 400) or 400), traj_budget))
        outs = vllm_gen.generate([{"prompt_token_ids": ctx_ids[i]} for i in active],
                                 sps, use_tqdm=False)
        exec_jobs = []
        # 本轮"仍在续写中"的样本（被单轮上限切断 → 同一轮继续写）。它们既不执行
        # 工具也不终局，必须与 exec_jobs 一起构成下一轮的 active——否则会被下面
        # 的 round-end 记账当成"已终局"而提前淘汰（token 档实测：续写样本在第 2 轮
        # 凭空消失）。
        _continuing = []
        for i, o in zip(active, outs):
            new_ids = list(o.outputs[0].token_ids)
            new_text = o.outputs[0].text
            fin = getattr(o.outputs[0], "finish_reason", None)
            open_ids[i].extend(new_ids)
            if tok_mode:
                open_raw[i] += (new_text or "")
                if collect_logps:
                    open_lps[i].extend(sampled_logps_from_output(o.outputs[0], new_ids))
                # 序列推进：ctx_ids = 轮起点 + 本轮累积（下一块从这个位置续生成）
                ctx_ids[i] = [*turn_ctx[i], *open_ids[i]]
                if fin == "length" and parse_assistant(
                        open_raw[i].strip(), style=style).kind != "tool":
                    # 被单轮上限切断 → 同一轮继续写（不落段、不解析、不终局）。
                    # 这是本档与旧版最本质的差别：旧版这里 trunc_final=1 且剩余
                    # 轮数整段作废（A/B 桶高发的结构性原因）。
                    #
                    # 【为什么必须排除"已构成完整调用"】若模型写完了调用块才撞上限，
                    # 继续生成会把后续文字接在调用块后面 → parse_assistant 见到
                    # "开标记之后还有别的文本" 判 **invalid**，一个本来合法的调用被
                    # 续写毁掉。故续写的条件是"累积文本**尚不构成**完整调用"。
                    _continuing.append(i)
                    continue
            elif collect_logps:
                open_lps[i] = sampled_logps_from_output(o.outputs[0], new_ids)
            ids_full = list(open_ids[i])
            raw_text = open_raw[i] if tok_mode else new_text
            lps_full = list(open_lps[i]) if collect_logps else None
            open_ids[i], open_raw[i], open_lps[i] = [], "", []
            # 与参考实现一致：chat template 会 strip assistant 内容，解析与结束
            # 边界计算都用 strip 后的文本，否则 canonical 位置对不上。
            asst_text = raw_text.strip()
            parsed = parse_assistant(asst_text, style=style)
            # 整轮落成**一个** assistant 段（续写的多块合成一段）：mask 只认
            # assistant 区间，一段或几块等价；合成一段让"生成序列 == 训练序列"
            # 的逐 token 断言仍成立（工具段由 build_next_prompt 从轮起点追加）。
            seg = {"kind": "assistant", "text": raw_text, "ids": ids_full,
                   "finish_reason": fin}
            if collect_logps:
                seg["logps"] = lps_full
            segs[i].append(seg)
            # 注意：turn_ctx[i] 在此**保持为本轮起点**（不含本轮 assistant 内容），
            # 因为 build_next_prompt 的 prev 参数要的正是"本轮 assistant 之前"的
            # 序列（它内部会追加 comp_ids 与 observation）。轮起点的推进发生在
            # 工具回填成功之后（见下方 turn_ctx[i] = nxt）。
            # 【token 预算档·预算判据】能否执行调用改由**剩余 token 预算**决定，
            # 不再由"是不是末轮"决定 —— 人为的"末轮截止"消失，code_wasted 从此
            # 只在**预算真耗尽**时发生（那是"确实没空间了"的真实信号）。
            # 执行一次调用要花掉两笔：工具回包（obs）+ 之后作答的空间
            # （answer_reserve），两者都装得下才执行。
            if tok_mode:
                _obs = int(cfg.get("tool_result_max_chars", 500) or 500) // 2 + 16
                _res = max(1, int(cfg.get("answer_reserve", 0) or 0))
                _can = (traj_budget - _used_tokens(i) - _obs) >= _res
            else:
                _can = (not is_final_round
                        and code_stats[i]["code_used"] < max_code_calls)
            if parsed.kind == "tool" and _can:
                code_stats[i]["code_used"] += 1
                seg["tool_action"] = "executed"
                exec_jobs.append((i, parsed.code, list(msgs[i]), ids_full, asst_text))
            else:
                # 终局：answer / invalid / 末轮写了调用 / 超出 max_code_calls。
                # invalid 单列计数——它是原生协议**唯一**的结构性负奖励入口
                # （无 boxed → reward -1），不单列就会与"啰嗦跑飞"在数据里同形。
                if parsed.kind == "invalid":
                    code_stats[i]["invalid_final"] += 1
                elif parsed.kind == "tool":
                    code_stats[i]["code_wasted"] += 1
                    seg["tool_action"] = "wasted"
                msgs[i].append({"role": "assistant", "content": asst_text})

        # 沙箱并行执行（与围栏版同一提速策略）
        if exec_jobs:
            workers = max(1, int(cfg.get("sandbox_workers", 4)))
            def _run(job):
                i, code, _m, _ids, _t = job
                return i, code_runner(code, timeout=cfg.get("sandbox_timeout", 5.0),
                                      mem_mb=cfg.get("sandbox_mem_mb", 256),
                                      max_chars=cfg.get("tool_result_max_chars", 500))
            if len(exec_jobs) > 1 and workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(workers, len(exec_jobs))) as ex:
                    done = list(ex.map(_run, exec_jobs))
            else:
                done = [_run(j) for j in exec_jobs]
            done_map = {i: r for i, r in done}
            next_active = []
            for i, _code, msgs_before, comp_ids, asst_text in exec_jobs:
                res = done_map[i]
                code_stats[i]["code_ok"] += int(res["ok"])
                etype = res.get("error_type") or ("ok" if res["ok"] else "exception")
                code_stats[i].setdefault("err_types", []).append(etype)
                # 消毒后再拼回（消毒点不因协议变化而移动：沙箱 stdout 仍是注入面）
                body = sanitize_tool_text(res["display"])
                if not res["ok"] and etype != "ok":
                    body = f"[{etype}] " + body
                if _hint:
                    # 【2026-09-29 可观测性】把剩余额度写进工具回包。C 桶（额度耗尽
                    # 还在调用）占 ok 族无 boxed 的 70%、dropped 族 40%，本质是**信息
                    # 不对称**——模型看得到全部历史，却看不到"还剩多少额度"。
                    # 回包是环境插入的、不进 loss、不参与打分（见 retool_score_flat），
                    # 加一行零成本；它把 C 从"不可学的悬崖"变成"可学的判断"。
                    # 剩余额度按**当前 ctx_ids** 算（本样本此刻的真实占用）。
                    _left = max(0, traj_budget - (len(ctx_ids[i]) - p0[i]) - len(body) // 2)
                    body += (f"\n[budget] {_left} tokens left in this trajectory. "
                             f"Finish with your final answer when you have enough.")
                call_id = make_call_id(0, i, code_stats[i]["code_used"])
                obs_msg = tool_message(call_id, body)
                try:
                    nxt = build_next_prompt(tokenizer, msgs_before, turn_ctx[i],
                                            comp_ids, obs_msg, ctkw)
                except ValueError as e:
                    # 【不静默降级】模板行为异常是协议级问题，继续跑会产出成批错位
                    # 序列并把整轮实验作废——当场抛，留下可排查的错（docs/09 §3）。
                    raise RuntimeError(
                        f"[rollout] 原生协议 token 增量拼接失败（样本 {i}，第 "
                        f"{len(segs[i])} 段）：{e}") from e
                if len(nxt) > budget:
                    # 放不下 = 这条轨迹到此为止（与参考实现的 fit_tool_content 同
                    # 语义：observation 进不去就终局，不硬塞）。工具段 ids 也不入
                    # segs —— merged 序列必须与实际喂给模型的序列逐 token 相同，
                    # 否则 gen_logps 的基线就是一串没发生过的位置。
                    # 【为什么单列 ctx_full 而不并进 code_wasted】code_wasted 的既有
                    # 语义是"写了代码但不会被执行的调用"（末轮/超调用上限），是
                    # **协议结构性**的；ctx_full 是**预算**性的，两者混在一起
                    # 会让"预算够不够"这个单变量失去可读数。
                    code_stats[i]["ctx_full"] += 1
                    msgs[i].append({"role": "assistant", "content": asst_text})
                    if tok_mode:
                        # token 预算档下这是**预算**终止（obs 装不下）→ trunc_final
                        code_stats[i]["trunc_final"] = 1
                    continue
                tool_ids = nxt[len(turn_ctx[i]) + len(comp_ids):]
                # 只有 assistant 段进 loss；工具段 ids 是"结束符+observation"增量
                # 【2026-10-02】`text` 是**模板渲染**（含 im_end/im_start/think 等
                # special token 与回答骨架），`body` 才是消毒后的纯回包（含 [etype]
                # 前缀与 [budget] 额度行）。轨迹落盘必须带上 body：真机 dump 里
                # 4898 条工具段的 `text` 全部以同一个 `‹|im_end|›‹|im_start|›user…`
                # 开头，剥包装才能读到观察本身。
                segs[i].append({"kind": "tool", "ids": tool_ids,
                                "body": body,
                                "text": tokenizer.decode(tool_ids,
                                                         skip_special_tokens=False)})
                msgs[i] = [*msgs_before, {"role": "assistant", "content": asst_text},
                           obs_msg]
                turn_ctx[i] = nxt
                ctx_ids[i] = nxt
                next_active.append(i)
            # 下一轮 active = 执行了工具并成功回填的样本 ∪ 仍在同一轮续写的样本。
            # 【2026-09-29 修复·预存在 bug（HEAD 即在，非本次改动引入）】旧版把
            # `active = next_active` 写在 `if exec_jobs:` **块内**：某一轮全员都没有
            # 工具调用时 exec_jobs 为空 → active 永不被更新 → 本应终局（answer/
            # invalid）的样本被**重新生成**到 max_rounds。后果：segs[i] 累计多段
            # assistant（max_rounds=3 时 3 段）、打分文本成了答案的重复拼接、
            # merged 序列是"prompt+答+答+答"——纯粹是循环记账错误，与模型行为无关。
            # 触发面：**整组**某轮无人调用工具（原生档 base 直接作答时很常见）。
            # token 预算档下这个 bug 是致命的：_iter_cap 是安全阀而非 max_rounds，
            # 样本会被反复重采直到撞上它。围栏路径没有这个 bug（那里的
            # `active = next_active` 在条件块之外，见 multi_turn_rollout_group）。
            active = sorted(set(next_active) | set(_continuing))
        else:
            active = list(_continuing)

    full_text = ["".join(s["text"] for s in segs_i) for segs_i in segs]
    if tok_mode:
        # 【循环安全阀耗尽时的兜底】_iter_cap 只防"每轮花 1 token"的病态循环；
        # 真撞上它时缓冲里还压着模型真实采样过的 token，必须落段并记 trunc_final
        # （理由同循环内的 flush：不落段 = 凭证丢失 = 打分与生成序列不符）。
        for i in range(n):
            if open_ids[i]:
                _txt = open_raw[i]
                _s = {"kind": "assistant", "text": _txt,
                      "ids": list(open_ids[i]), "finish_reason": "length"}
                if collect_logps:
                    _s["logps"] = list(open_lps[i])
                segs[i].append(_s)
                open_ids[i], open_raw[i], open_lps[i] = [], "", []
                code_stats[i]["trunc_final"] = 1
                # 与循环内"预算耗尽"分支同口径：安全阀耗尽也是"没写完就停"，
                # 切点落在未闭合调用块里同样要单列（否则又混进 B 桶）。
                code_stats[i]["trunc_in_call"] = trunc_in_call_flag(_txt, style)
        full_text = ["".join(s["text"] for s in segs_i) for segs_i in segs]
    else:
        # 旧口径：末段被单轮上限切断即 trunc_final。token 预算档下该口径由循环内的
        # "预算耗尽"判定接管——续写之后 finish_reason=length 不再等于失败。
        for i in range(n):
            last_a = next((s for s in reversed(segs[i])
                           if s["kind"] == "assistant"), None)
            code_stats[i]["trunc_final"] = int(
                bool(last_a) and last_a.get("finish_reason") == "length")
    return segs, full_text, code_stats


def multi_turn_rollout_group(vllm_gen, sampling_params, tokenizer, prompts_text, cfg,
                             code_runner=run_code, collect_logps: bool = False,
                             prompts_messages=None):
    """阶段2 ReTool：代码交织多轮生成（一组样本并行走）——token id 续写版。

    两条协议分支（`cfg["tool_protocol"]`，docs/09 §8）：
      · "fence"（缺省，p1–p11 逐位可复现）→ 本函数的下半部分（围栏正则 +
        `[TOOL RESULT]` 文本回填）；
      · "native"（方案 A 主线）→ 转 `multi_turn_rollout_group_native`
        （原生 `<tool_call>` + token 级增量拼接），需提供 `prompts_messages`。

    【2026-09-09 修复·生成/训练同序列契约】续写一律走 token id（vLLM
    prompt_token_ids），每段直接采用 vLLM 采样返回的 token_ids：旧版给 vLLM
    整串**文本**续写（内部整串 tokenize），而训练端 retool_build_batch 是分段
    tokenize 拼接——实测 Qwen2.5 tokenizer 下 assistant 段尾 "```" 接工具段头
    "\n" 时整串合并为单 token 13874、分段则是两个 token（典型轨迹 whole=67 vs
    split=68），所有"以代码围栏结尾"的样本（恰是触发工具调用的样本）边界必
    错位 → gen_logps 基线失真、采样分布≠训练序列。token id 续写后生成/训练/
    mask 三方共用同一序列，text 只用于围栏提取/沙箱/打分。

    【2026-09-21 核对·EOS 契约（#3 终止链核对，vllm_token_ids_keep_eos）】
    vLLM 0.12 V1 源码证据链（output_processor/detokenizer）：EngineCoreOutput
    的 new_token_ids 原样透传到 CompletionOutput.token_ids，EOS 自然停止时
    token_ids **恒含 EOS**（detokenizer 的 stop-token 排除只作用于文本，
    token id 走 token_ids.append(skipped_stop_token_id) 恒保留）；且
    sampled_logps_from_output 对 len(logprobs)!=len(ids) 直接 raise，logprobs
    与 ids 严格平行 → EOS 位有 logprob。结论：merged 训练序列含 EOS、
    gen_logps 覆盖它——"答完就停"这个决策一直在拿梯度，无需补 EOS。
    附带推论：Qwen 的 tokenizer.eos ≠ generation_config stop 词的坑在 vLLM
    采样路径天然规避。已知妥协：stop-string 停止也报 finish_reason="stop"，
    record 的 finish_reason 无法区分 EOS 停与围栏停（code_wasted 单列补救）。

    对每组样本：生成一段 → 检测 python 围栏代码块 → 有则沙箱执行 → 结果按
    TOOL_START/TOOL_END 回填 → 续生成下一轮；本轮无代码块则该样本结束（后续
    应给出最终答案）。最多 cfg['max_rounds'] 轮，其中**只有前 max_rounds-1 轮
    执行代码**：最后一轮即使写了代码也不执行不回填——循环已结束，执行结果
    永远无人消费，只会白烧沙箱并给 code_ok 记无效分（2026-09-09 审查发现3）。

    sampling_params: 单个 SamplingParams（所有请求共用，eval 贪心用）或与
      prompts_text 等长的列表——**训练时必须是列表且每样本 seed 不同**：
      同题 num_pre_Q 条是独立请求，共用 seed 会让 vLLM 生成 n 条完全相同的
      轨迹（组内零方差 → group_ok 永假 → 无限重采）。

    返回 (segs, full_text, code_stats)：
      segs:      list[list[dict]] —— 每个样本一段段的
                 {"kind": "assistant"|"tool", "text": str, "ids": list[int]}
                 （ids = 该段真实 token 序列，assistant 段即 vLLM 采样 token）
      full_text: list[str] —— 每个样本的完整轨迹文本（prompt+全部段，日志用）
      code_stats: list[{"code_used": int, "code_ok": int, "trunc_final": int,
                        "code_wasted": int}]
                  （trunc_final=1 表示末个 assistant 段被轮长上限切断；
                   code_wasted = 末轮写了代码但不会被执行的次数——该轨迹
                   结构性无 boxed，且 retool_stop 下 trunc_final 也记不到它）
    """
    if int(cfg.get("max_traj_tokens", 0) or 0) > 0 and not is_native_protocol(cfg):
        # 【为什么不给围栏档也开】围栏档的轮结构里"代码块闭合即停"（retool_stop）
        # 与 [TOOL RESULT] 文本回填是一套独立机制，改它要重新核对生成/训练同序列
        # 契约与 stop 语义——而本档要解决的三件事（末轮废码、切断即终局、额度不可
        # 见）在原生档才是实测出来的痛点（native_p3）。fail-fast 而不是静默降级：
        # 静默忽略会让"我开了 token 预算档"变成一句空话，正是本项目最贵的 bug 类型。
        raise ValueError(
            "[rollout] max_traj_tokens>0（token 预算档）目前只在原生协议下实现，"
            f"当前 tool_protocol={cfg.get('tool_protocol')!r}。\n"
            "  处置：加 --tool_protocol native 走原生档，或去掉 --max_traj_tokens "
            "回到轮数预算档（max_rounds × round_gen_tokens）。")
    if is_native_protocol(cfg):
        if prompts_messages is None:
            raise ValueError(
                "[rollout] tool_protocol='native' 必须传 prompts_messages（原生协议从 "
                "messages 渲染——tools 声明在模板里，只给纯文本无法渲染工具段）。\n"
                "  调用点：collect_retool_group / probe_difficulty / eval_vllm_one。")
        return multi_turn_rollout_group_native(
            vllm_gen, sampling_params, tokenizer, prompts_messages, cfg,
            code_runner=code_runner, collect_logps=collect_logps)
    n = len(prompts_text)
    # 每条请求的无 pad prompt token（与批量左 pad prompt_ids 同源：去 pad 即得）
    ctx_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts_text]
    segs = [[] for _ in range(n)]
    code_stats = [{"code_used": 0, "code_ok": 0, "code_wasted": 0,
                   "err_types": []} for _ in range(n)]
    active = list(range(n))            # 还在"代码-执行-续写"循环里的样本
    n_rounds = int(cfg.get("max_rounds", 3))
    for _rnd in range(n_rounds):
        if not active:
            break
        is_final_round = (_rnd == n_rounds - 1)
        if isinstance(sampling_params, list):
            sps = [sampling_params[i] for i in active]
        else:
            sps = sampling_params
        outs = vllm_gen.generate([{"prompt_token_ids": ctx_ids[i]} for i in active],
                                 sps, use_tqdm=False)
        new_ids_map, results, exec_jobs = {}, {}, []
        for i, o in zip(active, outs):
            new_ids = list(o.outputs[0].token_ids)
            new_text = o.outputs[0].text
            # finish_reason（"length"=被轮长上限切断，"stop"=自然停；FakeGen 等测试
            # 替身无此属性时按 None≈stop 处理——末段截断是 retool 答案被切的直接签名，
            # 训练期不记录就永远看不见（2026-09-09 审查发现6）
            fin = getattr(o.outputs[0], "finish_reason", None)
            seg = {"kind": "assistant", "text": new_text, "ids": new_ids,
                   "finish_reason": fin}
            if collect_logps:
                # 本轮被采样 token 的 logprob（= log π(tok|完整前文)），逐轮收集
                seg["logps"] = sampled_logps_from_output(o.outputs[0], new_ids)
            segs[i].append(seg)
            new_ids_map[i] = new_ids
            blocks = extract_python_blocks(new_text)
            if not blocks:
                continue              # 本轮无代码块 → 样本结束，等待最终答案
            if is_final_round:
                # 【2026-09-20 可观测性修复】末轮代码不执行（结果无人消费，见
                # docstring）→ 这次调用是纯浪费，且该轨迹结构性地不会产出 boxed。
                # 旧版在此直接 continue，于是这类轨迹在 code_used/code_ok/
                # trunc_final 三个统计量里**同时为 0**（retool_stop 让末段
                # finish_reason="stop" 而非 "length"）——reward=-1（无 boxed）却
                # 显示"既没写代码也没被截断"，与"啰嗦跑飞"在数据里完全同形。
                # 【为什么不计入 code_used】code_used 的既有语义是"真正执行过的
                # 代码调用次数"，它是 analysis 的 code% 列与跨 run 对照（p8 的
                # 48~70%）的口径。把末轮废码并进去会静默抬高该列、破坏可比性，
                # 而 code% 正是长度/代码轴的判据之一。故**单列新计数**：既有口径
                # 逐位不变，末轮废码从此可见。
                code_stats[i]["code_wasted"] += 1
                seg["tool_action"] = "wasted"
                continue
            code_stats[i]["code_used"] += 1
            seg["tool_action"] = "executed"
            code = blocks[-1]          # 执行最后一个完整代码块（最新计算意图；
                                        # Auto_Program 原版取第一个——并非一致，是有意改进）
            exec_jobs.append((i, code))

        # 沙箱并行执行（2026-09-09 提速）：run_code 是 subprocess，线程池并发安全。
        # 旧版逐个串行：每个 subprocess 启动 ~0.1-0.3s，一条 5s 超时的死循环代码
        # 会让同轮其余样本全部干等——4 条并行最多省 ~4x 的沙箱墙钟时间。
        if exec_jobs:
            workers = max(1, int(cfg.get("sandbox_workers", 4)))
            def _run(pair):
                i, code = pair
                return i, code_runner(code, timeout=cfg.get("sandbox_timeout", 5.0),
                                      mem_mb=cfg.get("sandbox_mem_mb", 256),
                                      max_chars=cfg.get("tool_result_max_chars", 500))
            if len(exec_jobs) > 1 and workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(workers, len(exec_jobs))) as ex:
                    done = list(ex.map(_run, exec_jobs))
            else:
                done = [_run(p) for p in exec_jobs]
            for i, res in done:
                code_stats[i]["code_ok"] += int(res["ok"])
                # 错误类型分级（2026-09-21，#6）：模型反馈里带上结构化错误类型，
                # 学差异化修复（timeout≠syntax≠算错）；code_stats 记录供监控。
                etype = res.get("error_type") or ("ok" if res["ok"] else "exception")
                code_stats[i].setdefault("err_types", []).append(etype)
                # 消毒后再拼回（2026-09-10）：沙箱 stdout 模型间接可控，
                # 特殊 token/工具标记字面量必须剥除——见 protocol.sanitize_tool_text
                body = sanitize_tool_text(res["display"])
                if not res["ok"] and etype not in ("ok",):
                    body = f"[{etype}] " + body
                tool_text = TOOL_START + body + TOOL_END
                tool_ids = tokenizer(tool_text, add_special_tokens=False)["input_ids"]
                segs[i].append({"kind": "tool", "text": tool_text, "ids": tool_ids,
                                "body": body})
                results[i] = tool_ids
        # 下一轮：只有执行过代码的样本续写（扩展上下文）；其余样本就此定格
        next_active = [i for i in active if i in results]
        for i in next_active:
            ctx_ids[i] = ctx_ids[i] + new_ids_map[i] + results[i]
        active = next_active

    full_text = [p + "".join(s["text"] for s in segs_i)
                 for p, segs_i in zip(prompts_text, segs)]
    # 末段截断标记：最后一个 assistant 段 finish_reason=="length" → final 答案
    # 很可能被 round_gen_tokens 切断（acc 必 -1 的直接原因，训练期可见）
    for i in range(n):
        last_a = next((s for s in reversed(segs[i]) if s["kind"] == "assistant"), None)
        code_stats[i]["trunc_final"] = int(bool(last_a) and last_a.get("finish_reason") == "length")
    return segs, full_text, code_stats


def retool_build_batch(prompt_ids, segs, plen, pad_token_id):
    """由分段轨迹构造 merged_ids + 工具段 mask（assistant=1/tool=0/pad=0）。

    【2026-09-09 修复】直接采用各段生成时记录的 token ids（assistant 段 =
    vLLM 采样 token_ids、工具段 = 分段 tokenize），**不再对段文本重新
    tokenize**——生成/训练必须逐 token 同一条序列：文本往返与段边界 BPE
    合并都会破坏该契约（见 multi_turn_rollout_group docstring）。mask 由
    assistant 区间纯函数给出——工具返回 token 不进 loss 的契约。"""
    per_sample_ids, masks = [], []
    for segs_i in segs:
        ids, spans = [], []
        for seg in segs_i:
            t = list(seg["ids"])
            start = len(ids); ids.extend(t); end = len(ids)
            if seg["kind"] == "assistant":
                spans.append((start, end))
        per_sample_ids.append(ids)
        masks.append(segment_mask_from_spans(len(ids), spans))
    output_ids = pad_sequence([torch.tensor(t) for t in per_sample_ids],
                              batch_first=True, padding_value=pad_token_id)
    mask = pad_sequence(masks, batch_first=True, padding_value=0.0)
    n = output_ids.shape[0]
    Qrep = prompt_ids.repeat(1, n).view(-1, plen)
    merged_ids = torch.cat([Qrep, output_ids], dim=1)
    return merged_ids, mask, per_sample_ids


def strip_left_pad(prompt_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """剥掉 prompt 的左 pad 前缀，返回 (B, 真实长)（纯函数，CPU 可测）。

    【2026-09-11 生成/打分口径分叉根治】多题并采时 tokenizer 按批内最长 prompt
    左 pad，而 vLLM 生成侧是逐条**无 pad** 的序列。带 pad 前向有**两处**与生成侧
    不等价：
      ① pad token 作为 attention 的 key，被后面所有真实位置读到；
      ② 位置编码整体后移——主流 HF 实现（Qwen/GPT 系）的 position_ids 取的是下标
         （cache_position = arange），**不按 attention_mask 做 cumsum 修正**，短题
         的真实 token 拿到的是"下标位置"而非"真实位置"，embedding 处就错了。
    实测（tiny 模型逐位对拍）：只补 ①（传 attention_mask 把 pad 键权重压到 0，
    已验证生效）真实位仍与无 pad 前向差 1.0 量级——②没修掉，缺口就还在。
    剥掉 pad 则 ①②同时消失，且**不依赖模型是否支持 2D 掩码或位置修正**——merged
    序列逐 token、逐位置与生成时完全一致。配合 retool_build_batch 的分段 ids
    拼接，"生成序列 == 训练序列"从此在数据构造层一次性成立。

    pad 只在左侧（tokenizer padding_side="left"）：取最后一个非 pad 位置之前的
    全部。整行都是 pad（空 prompt）fail-fast——静默返回空序列会让 plen=0，
    下游 `logps[:, plen-1:]` 变成 -1 切片，切出整条错位序列。"""
    row = prompt_ids[0]
    keep = int((row != pad_token_id).sum())
    if keep == 0:
        raise ValueError("[rollout] prompt 整行都是 pad（空 prompt），无法剥出真实长度")
    return prompt_ids[:, prompt_ids.shape[1] - keep:]


def tool_credit_advantages(base_adv: torch.Tensor, segs, total_len: int,
                           call_cost: float = 0.0,
                           waste_penalty: float = 0.0,
                           trunc_flags=None,
                           trunc_tail_penalty: float = 0.0) -> torch.Tensor:
    """序列级任务优势 + turn-level 工具动作成本 -> (B,T) per-token advantage。

    普通 assistant 轮继承组内任务优势；已执行工具调用轮减 call_cost；预算不足仍
    调用的浪费轮覆盖为 -waste_penalty；截断轨迹（trunc_flags[i] 为真）的**最后一个
    assistant 段**覆盖为 -trunc_tail_penalty。工具回包/pad 保持 0；成本在构造时按
    该样本有效 assistant token 数归一化，抵达 sample_mean 后是固定的每次动作成本。

    【2026-10-01 截断末段信用】trunc_final 轨迹的任务结果不可信（预算耗尽时可能
    还没作答），故其 base_adv 恒为 0（retool_score_flat 的 sample_mask 已把它排除
    出组均值），此前有效推理/工具轮不连坐；但"把预算烧光"这个**末段动作本身**
    获得直接负信用 —— 这正是旧口径（sw=0 整行过滤）唯一缺失的反向信号。
    """
    if base_adv.dim() != 1 or len(base_adv) != len(segs):
        raise ValueError(
            f"tool credit 需要 base_adv(B,) 与 segs(B) 对齐，收到 "
            f"{tuple(base_adv.shape)} / {len(segs)}")
    if call_cost < 0.0 or waste_penalty < 0.0 or trunc_tail_penalty < 0.0:
        raise ValueError(
            "tool_call_cost/tool_waste_penalty/trunc_tail_penalty 必须 >= 0")
    if trunc_flags is not None and len(trunc_flags) != len(segs):
        raise ValueError(
            f"trunc_flags 必须与 segs 等长，收到 {len(trunc_flags)} / {len(segs)}")
    out = torch.zeros((len(segs), int(total_len)), dtype=base_adv.dtype,
                      device=base_adv.device)
    for i, segs_i in enumerate(segs):
        pos = 0
        assistant_len = sum(len(seg["ids"]) for seg in segs_i
                            if seg["kind"] == "assistant")
        if assistant_len <= 0:
            continue
        # 末段下标：只对截断轨迹找（预算耗尽 ⇒ 末尾必是被切断的 assistant 段）。
        # penalty=0 时不计算，保持关闭档与历史行为逐位相同。
        tail_idx = -1
        if (trunc_flags is not None and int(trunc_flags[i])
                and trunc_tail_penalty > 0.0):
            for _k in range(len(segs_i) - 1, -1, -1):
                if segs_i[_k]["kind"] == "assistant":
                    tail_idx = _k
                    break
        for _k, seg in enumerate(segs_i):
            end = pos + len(seg["ids"])
            if end > total_len:
                raise ValueError(
                    f"tool credit 段长越界: sample={i} end={end} total={total_len}")
            if seg["kind"] == "assistant":
                action = seg.get("tool_action")
                m = len(seg["ids"])
                if m <= 0:
                    pos = end
                    continue
                if action == "wasted":
                    # 预算已不足却继续调用是当前动作本身的坏结果：覆盖任务优势，
                    # 只给固定的局部负信用；此前正确推理不被连坐。
                    val = -float(waste_penalty) * assistant_len / m
                elif _k == tail_idx:
                    # 预算耗尽（trunc_final）：同样只罚末段动作，整条轨迹的未知
                    # 任务结果不参与（base_adv 已被 sample_mask 置 0）。
                    val = -float(trunc_tail_penalty) * assistant_len / m
                elif action == "executed":
                    val = float(base_adv[i]) - float(call_cost) * assistant_len / m
                else:
                    val = float(base_adv[i])
                out[i, pos:end] = val
            pos = end
    return out


def retool_context_overlong(per_sample_ids, plen, max_context_tokens):
    """逐样本全长 token 预算检查（模块级纯函数，CPU 可测）。任一样本超限即超。

    【2026-09-09 修复·两级】
    ①全长口径（assistant+工具段一起计）——旧版用 mask.sum() 只数 assistant
      token，工具段（每轮 ≤500 字符 ≈150-200 token，3 轮 ≈600）被漏算：防 OOM
      防线可被超出 ~25%，且与 eval max_len（max_prompt_length+max_context_tokens）
      联动错位——训练放行的最长轨迹会撞 eval 的 vLLM max_model_len。
    ②逐样本口径——旧版按组均值（total > max*b）判定，单条超长样本可被组内短
      样本均摊掩盖而放行；但 padded batch 按【最长样本】计激活显存，均值检查
      挡不住单条尖峰。改为任一样本 len(ids)+plen > max_context_tokens 即整组丢弃。"""
    return any(len(t) + plen > max_context_tokens for t in per_sample_ids)


def retool_score_flat(inputs, asst_texts, code_stats, cfg, steps_elapsed,
                      completion_lens=None):
    """阶段2 打分（模块级纯函数，CPU 可测）。

    索引契约（2026-09-08 真机 IndexError 教训）：asst_texts/code_stats 必须是
    Q_batch_size × num_pre_Q 条——即每道题先扩成 num_pre_Q 条独立轨迹再进
    multi_turn_rollout_group，idx = i*num_pre_Q + j 一一对应。

    打分文本 = 模型自己的 assistant 段拼接（**不含工具段**，2026-09-08 真机
    fmt=0.0% 教训）：全文含 prompt → 格式正则 ^ 锚定必败 → fmt 恒为常数 →
    组内归一化后 fmt 梯度信号彻底死亡；全文还含沙箱输出 → "最后一个数字"
    变成工具 stdout，把工具结果当模型答案发信用。工具段只作上下文，也绝不
    参与打分——与"工具 token 不进 loss"是同一条原则在奖励端的投影。

    completion_lens（可选）：每条轨迹的 token 全长（collect_retool_group 传入），
    供 math 分支的 overlong shaping 使用——不传则 shaping 静默无效
    （2026-09-09 审查发现12：旧版调用点从不传 completion_len，开了
    overlong_shaping 也不生效）。

    返回 (adv, acc_s, fmt_s, code_used, code_ok, phase)。"""
    phase = reward_phase(steps_elapsed, cfg["reward_switch_step"])
    rewards, acc_s, fmt_s, cu, ck = [], [], [], [], []
    trunc_finals = []  # overlong filtering: 1=末段被截断 → 从 advantage/组统计中移除
    # code_wasted 的失败 reward 不进入任务基线：否则无 boxed 的 -1 会系统性抬高
    # 同组其他样本优势。开启 tool_waste_penalty 后，它仍在这里得到 base_adv=0，随后
    # collect_retool_group 只给浪费调用轮注入局部负优势；此前有效推理不会被连坐。
    wasted_flags = []
    n = cfg["num_pre_Q"]
    assert len(asst_texts) == len(inputs) * n, \
        f"轨迹数 {len(asst_texts)} != 题数{len(inputs)}×num_pre_Q{n}（检查是否漏了扩样）"
    is_math = cfg.get("data_task") in ("dapo_math", "dapo-math-17k", "math_dapo")
    # overlong 参考系 = 可写满的多轮总预算（min(max_rounds×round_gen_tokens,
    # max_context_tokens−max_prompt_length)），非单轮 max_gen_tokens——否则要么
    # 用满预算的轨迹被整额扣分，要么 trigger 落在丢弃线之外变成死开关
    # （两代 bug 都记录在 reward.overlong_ref_tokens 的 docstring 里）
    _ol_ref = overlong_ref_tokens(cfg)
    _trunc_w = float(cfg.get("trunc_shaping", 0.0) or 0.0)
    _do_filter = bool(cfg.get("overlong_filter", False))
    # 尝试级 shaping（#2 风险不对称修正）：默认 0 = 关闭，行为与旧版逐位相同
    _att_w = float(cfg.get("code_attempt_w", 0.0) or 0.0)
    _max_rounds = int(cfg.get("max_rounds", 8))
    # 【2026-09-23 分档奖励】code_w>0：答对且代码执行成功叠加 code_ok×code_w
    # （ReTool 官方 per-success 口径，打通此前硬编码 0 的死代码字段）
    _code_w = float(cfg.get("code_w", 0.0) or 0.0)
    # 【2026-09-29 反激励修正】True 时 code_attempt/code_w 由"每次调用累加"改
    # "一次性"——去掉"多调用多拿分"（答对时 0 次 +1.00 < 4 次 +1.40，与
    # "最少 token"目标反向）。保留"敢写代码"的对冲作用。
    _once = bool(cfg.get("code_shaping_once", False))
    for i, inp in enumerate(inputs):
        for j in range(n):
            idx = i * n + j
            _tf = code_stats[idx].get("trunc_final", 0)
            if is_math:
                sc = total_reward_retool_math(
                    inp["A"], asst_texts[idx], code_ok=code_stats[idx]["code_ok"],
                    completion_len=(completion_lens[idx] if completion_lens is not None else 0),
                    max_gen_tokens=_ol_ref,
                    overlong_buffer=cfg["overlong_buffer"],
                    overlong_shaping=cfg.get("overlong_shaping", False),
                    # 末段被轮长上限切断 → 额外扣分（prose 路径唯一够得到的长度反向信号）
                    trunc_final=_tf,
                    trunc_shaping=_trunc_w,
                    code_used=code_stats[idx]["code_used"],
                    code_attempt_w=_att_w,
                    code_w=_code_w,
                    max_rounds=_max_rounds,
                    code_shaping_once=_once)
            else:
                sc = total_reward_retool(
                    inp["A"], asst_texts[idx], code_ok=code_stats[idx]["code_ok"],
                    phase=phase, code_w=cfg["code_w"],
                    completion_len=(completion_lens[idx] if completion_lens is not None else 0),
                    max_gen_tokens=_ol_ref,
                    overlong_buffer=cfg["overlong_buffer"],
                    overlong_shaping=cfg.get("overlong_shaping", False),
                    cold_w=cfg["reward_cold_w"], hot_w=cfg["reward_hot_w"])
            rewards.append(sc["reward"]); acc_s.append(sc["acc"])
            fmt_s.append(sc["format"]); cu.append(code_stats[idx]["code_used"])
            ck.append(code_stats[idx]["code_ok"])
            trunc_finals.append(_tf)
            wasted_flags.append(int(code_stats[idx].get("code_wasted", 0)))
    rewards = torch.tensor(rewards, dtype=torch.float32)
    # 【2026-09-23 组相对长度惩罚（MiMo Eq.4 形态）】在 advantage 之前按组应用：
    # 对通过轨迹的长度分位数起坡、只在通过率超阈值的组生效、只罚未通过轨迹。
    # 与 trunc_shaping（绝对惩罚，run2 表面收尾事故根源）本质不同：相对化后
    # "组内都变短"不改变任何 advantage，无捷径可钻。weight=0 完全跳过（逐位同旧）。
    _lp_w = float(cfg.get("len_penalty_w", 0.0) or 0.0)
    _eff_w0 = float(cfg.get("len_eff_w", 0.0) or 0.0)
    # 两项都吃 completion_lens：任一开启就取（否则效率项静默无效）
    _lp_clens = ([int(x) for x in (completion_lens if completion_lens is not None else [])]
                 if (_lp_w > 0.0 or _eff_w0 > 0.0) else [])
    if _lp_w > 0.0 and len(_lp_clens) == len(rewards):
        _pen = group_length_penalty(rewards.tolist(), _lp_clens,
                                    weight=_lp_w,
                                    quantile=int(cfg.get("len_penalty_quantile", 50)),
                                    pass_gate=float(cfg.get("len_penalty_gate", 0.25)))
        rewards = torch.tensor(_pen, dtype=torch.float32)
    # 【2026-09-29 效率激励】通过轨迹内部的组内相对长度奖励（与上面的惩罚互补：
    # 惩罚管未通过、奖励管通过；两项都不改 ±1 的正负号）。必须在 advantage
    # 之前应用——组均值会减掉平移，绝对长度项无效，只有相对项能进 advantage。
    # 前置条件与 len_penalty 相同（要 completion_lens）；不满足则跳过并告警一次，
    # 因为"开了开关却没接线"正是本项目最贵的一类静默失败。
    _eff_w = _eff_w0
    if _eff_w > 0.0:
        if len(_lp_clens) == len(rewards):
            rewards = torch.tensor(
                group_eff_bonus(rewards.tolist(), _lp_clens, weight=_eff_w),
                dtype=torch.float32)
        elif not _EFF_WARNED[0]:
            _EFF_WARNED[0] = True
            print("[rollout][提示] len_eff_w>0 但本组没拿到 completion_lens → "
                  "效率项静默无效（检查调用点是否传了 completion_lens）。", flush=True)
    # 【2026-09-21 DAPO overlong filtering】截断样本从 advantage 和组统计中移除：
    # 组均值只算非截断 → 截断样本 adv=0 → 不贡献 pg_term。
    # 杀 NeMo-RL bug：全错组+混合截断不再因 trunc_shaping 产生假方差通过 group_ok。
    # 【2026-09-28 F1】code_wasted（末轮废码）并入同一排除口径：它与截断一样是
    # 结构性无 boxed 的 -1，且 loss 侧 sample_weight 早已把它整行清零
    # （collect_retool_group 的 sw 构造）——旧版只把 trunc 排出统计，废码样本以 -1
    # 进组均值（基线被压低 → 其他样本 adv 被系统性抬高，实测扭曲 +0.07/组），自身
    # 又因 sw=0 零梯度（"末轮写调用"永远拿不到直接惩罚），与 sw 注释声称的
    # "它们 adv=0" 矛盾。修复后：组统计与 loss 归一化排除**同一人群**
    # （trunc_final OR code_wasted），旧行为仅在 overlong_filter 关闭时保留。
    excl = [1 if (tf or wf) else 0 for tf, wf in zip(trunc_finals, wasted_flags)]
    if _do_filter and any(excl):
        sample_mask = torch.tensor([0 if e else 1 for e in excl],
                                   dtype=torch.float32)
        adv = compute_advantages(rewards, n, cfg["adv_mode"], sample_mask=sample_mask)
    else:
        adv = compute_advantages(rewards, n, cfg["adv_mode"])
    return (adv, torch.tensor(acc_s), torch.tensor(fmt_s), cu, ck, phase)


def trunc_in_call_flag(text: str, style: str = "auto") -> int:
    """预算/轮长用尽时，模型是否正处在**未写完的调用块**里（0/1）。纯函数，CPU 可测。

    【为什么必须有这个字段（2026-10-02）】token 预算档下"预算耗尽"只置
    `trunc_final`，**不置** `invalid_final`（invalid 只在轮结束分支判）→ 而 A 桶的
    判据正是 `trunc ∩ invalid` ⇒ **A 恒 0**，于是所有"调用写到一半被墙切断"的样本
    都被 B 桶吞掉，而 B 的定义是"没写出任何调用"——两类处置相反（要治"别把调用
    拖到最后" vs 治"啰嗦"）却在数据里同形。真机 `eval_vllm_s200.traj.jsonl` 实测：
    无 boxed 822 条里 **27%（219 条）**是这种（开标记多于闭标记）。

    判据 = **开标记多于闭标记**（调用确实没闭合）**且** `parse_assistant` 判 invalid。
    只写了"调用 + 尾巴"（闭标记齐）**不**置位——那种更接近末轮废码，混进"纯散文"
    会污染读数，真机里只占 22/227。

    调用标记从 `protocol` 正则派生导入（AGENTS.md 标签字节铁律：绝不手写标签字面量）。
    旧 record 无该键 → analysis 按"不可用"降级，历史桶口径逐位不变。"""
    if not text or not text.strip():
        return 0
    t = text.strip()
    if t.count(_CALL_OPEN) <= t.count(_CALL_CLOSE):
        return 0
    return int(parse_assistant(t, style=style).kind == "invalid")


def traj_dump_row(segs_i, *, Q, A, qk, status, stats=None, acc=None, fmt=None,
                  clen=None, gen_version=None, t=None):
    """把一条轨迹打成可落盘的一行（纯函数，CPU 可测；2026-10-02）。

    **段级**而不是拼接文本：B/C 桶判读的核心问题正是"末段写了什么、有多长"，
    拼接后 assistant/工具边界丢失就答不了。

    ⚠ **工具段的两个文本字段不是一回事**（2026-10-02 真机实锤）：`text` 是**模板
    渲染**（`‹|im_end|›‹|im_start|›user …‹tool_response›…` + special token + 回答
    骨架），4898 条工具段的 `text` 全部以同一个包装开头；读观察必须用 `body`
    ——消毒后的纯回包（含 `[etype]` 错误前缀与 `[budget]` 额度行，若开了 hint）。
    两者都由 `len` 给出 token 数（`text` 的长度含包装，别拿它当回包大小）。

    `status` 三态：ok（已上传）/ uniform（零方差丢弃）/ overlong（轨迹超长丢弃）。
    后两类整组不进 record.jsonl——dump 含它们才治得了记录口径的幸存者偏差
    （record 只看得见"活下来的组"）。计数类与 record 同源（同一个 code_stats），
    acc/fmt 是 ±1 奖励原值；overlong 分支没打分 → None（不写 0 冒充"全错"）。

    ⚠ 阅读路径：dump 里含调用标记，用 read/控制台看会被渲染成无括号普通词——
    判字节真伪要逐字符 ord 直出（AGENTS.md 标签铁律）。"""
    _st = stats or {}
    # seg 的长度优先用调用方**已经记录**的 token 数（`len`）：单轮档没有段结构，
    # 只有生成时记下的 token 数——重新 tokenize 会因为 BPE 跨段边界合并而变长
    # （本项目"生成/训练同序列"铁律的另一面）。
    _segs = []
    for s in segs_i:
        _d = {"kind": s.get("kind"),
              "len": int(s["len"]) if s.get("len") is not None
                     else len(s.get("ids") or ()),
              "text": s.get("text") or ""}
        if s.get("body") is not None:
            _d["body"] = s["body"]      # 纯回包（模板包装之外的那部分）
        _segs.append(_d)
    return {"t": t if t is not None else time.time(), "Q": Q, "A": A, "qk": qk,
            "status": status, "gen_version": gen_version,
            "acc": acc, "fmt": fmt,
            "clen": int(clen) if clen is not None else sum(s["len"] for s in _segs),
            "code_used": int(_st.get("code_used", 0)),
            "code_ok": int(_st.get("code_ok", 0)),
            "trunc_final": int(_st.get("trunc_final", 0)),
            "code_wasted": int(_st.get("code_wasted", 0)),
            "invalid_final": int(_st.get("invalid_final", 0)),
            "ctx_full": int(_st.get("ctx_full", 0)),
            "n_segs": len(_segs), "segs": _segs}


def collect_retool_group(vllm_gen, tokenizer, cfg, compute_gen_logps,
                         inputs, prompts_text, prompt_ids, plen,
                         sampling_params, steps_elapsed=0, verify_logps=None,
                         prompts_messages=None, traj_sink=None):
    """多轮 rollout → 打分 → 按题拆分的上传就绪结果（模块级，FakeGen CPU 可测）。

    【2026-09-10 结构修改·采样并发与按题拆分】一次调用处理 len(inputs) 道题
    （每题 num_pre_Q 条），vLLM 每轮并发 = 题数×num_pre_Q——retool_math 为
    4×8=32（旧版 Q_batch_size=1 恒为 8，H20 上 vLLM 严重欠利用，是训练变慢
    的第二根因）。打分后**按题拆分**：每题独立过超长/零方差检查、独立构造
    num_pre_Q 行批上传——训练端 micro-batch 契约（=num_pre_Q 行）不变、
    DeepSpeed/grad_accum/有效 batch 全部零改动，且单题 uniform/超长只丢弃
    该题不再连坐整批（整批连坐在多题并采下会放大丢弃损失）。

    sampling_params: 与 Q*num_pre_Q 条轨迹等长的列表（gen_worker 构造，每条
      独立 seed；拆出来是为了让本函数不依赖 vllm import，FakeGen 可直接测）。

    返回 per-question list，每项 {"status": "ok"|"uniform"|"overlong"}；
    ok 项另含 merged/mask/gen_logps/adv/acc/fmt/cu/ck/phase/clen/trunc/plen
    与段长画像 segl（assistant 段长表）/tsegl（工具段长表）——两者 per-sample、
    与 acc/fmt 同下标对齐（2026-10-02）。

    `traj_sink`（可选 list）：非 None 时把**每条轨迹**（含未上传的 uniform /
    overlong）按 `traj_dump_row` 追加进去，由调用方落盘。默认 None = 零成本
    （连字符串都不拼），故不影响任何既有调用点。
    adv 通常为 (B,) 序列级；启用 tool_call_cost/tool_waste_penalty 后为 (B,T)
    逐token任务优势+工具动作成本（losses._adv_broadcast 原生支持）。
    （plen = **本题** prompt 的真实长度——每道题先剥掉左 pad 再建批，见
    strip_left_pad：带 pad 会让打分序列与 vLLM 生成序列在位置编码与注意力键上
    双重分叉。上传 meta 与 gen_logps/训练前向共用同一基准）。"""
    n = int(cfg["num_pre_Q"])
    nq = len(inputs)
    # 【2026-09-23 在线通过率监控】题目指纹（与 eval_vllm_one 的 items.qk 同一
    # 算法：sha1(Q)[:12]）——record.jsonl 从此带题标识，passrate.py 才能按题聚合
    # 训练期通过率分布（观测 band 漂移 / 回填难度表）。
    import hashlib as _hashlib
    _qks = [_hashlib.sha1(str(x["Q"]).encode("utf-8")).hexdigest()[:12] for x in inputs]
    assert prompt_ids.shape[1] == plen, \
        f"prompt_ids 宽 {prompt_ids.shape[1]} != plen {plen}（调用约定：未剥 pad 的整批）"
    group_prompts = [p for p in prompts_text for _ in range(n)]   # Q*n 条
    # 原生协议：从 messages 渲染（tools 声明在模板里）。每题先扩成 num_pre_Q 条
    # 独立轨迹，与围栏版同构（2026-09-08 扩样 bug 的契约不变）。
    # 【参数口径】prompts_messages 收的是**每题一条**（Q 条，与 inputs 等长）——
    # 扩样（×num_pre_Q）在本函数内做，与 group_prompts 的扩样同构。调用方
    # （gen_worker / FakeGen 测试）只负责给"与 inputs 对齐"的那一份，不必知道
    # 内部要扩成几条（否则扩样口径会散落到每个调用点，正是 2026-09-08
    # IndexError 的形态）。
    if is_native_protocol(cfg):
        if prompts_messages is None:
            prompts_messages = prompt_messages_for(inputs, cfg)
        if len(prompts_messages) != nq:
            raise ValueError(
                f"[rollout] prompts_messages 条数 {len(prompts_messages)} != 题数 {nq}"
                "（约定：收每题一条，扩样在 collect_retool_group 内做）")
        prompts_messages = [m for m in prompts_messages for _ in range(n)]
    # gen_logps 来源：vLLM 逐轮 logprobs（需采样时就带上 logprobs=0）或 torch 副本重算
    use_vllm_logps = bool(cfg.get("vllm_gen_logps"))
    segs, _full_texts, code_stats = multi_turn_rollout_group(
        vllm_gen, sampling_params, tokenizer, group_prompts, cfg,
        collect_logps=use_vllm_logps, prompts_messages=prompts_messages)
    asst_texts = ["".join(s["text"] for s in segs_i if s["kind"] == "assistant")
                  for segs_i in segs]
    results = []
    for i in range(nq):
        segs_i = segs[i * n:(i + 1) * n]
        # 【2026-09-11】逐题剥掉左 pad（批内最长 prompt 补出来的）——本题 plen_i 即
        # 真实 prompt 长；merged 因此逐 token/逐位置等于 vLLM 生成时看到的序列
        # （详见 strip_left_pad：pad 会同时污染注意力键与位置编码两处）。
        prompt_i = strip_left_pad(prompt_ids[i:i + 1], tokenizer.pad_token_id)
        plen_i = prompt_i.shape[1]
        merged_i, mask_i, per_ids_i = retool_build_batch(
            prompt_i, segs_i, plen_i, tokenizer.pad_token_id)
        clen_i = [len(t) for t in per_ids_i]
        # 【2026-10-02 B 桶长度画像·为什么必须落盘】native_p4_trunc 的核心悬案：
        # `B_cut_mid_prose`（预算耗尽、在散文里被切）到底该抬 `answer_reserve`，
        # 还是该治"写散文不收尾"？判别读数是**每段 assistant 的长度**，而 record
        # 此前只有整条 clen，段长在落盘时被丢掉 → 只能靠猜。逐样本对齐落盘：
        #   segl[i]  = 第 i 条样本的 assistant 段 token 数（按时间序，末位=末段）
        #   tsegl[i] = 该样本的工具回包段 token 数
        # 工具段长是"回包到底吃掉多少预算"的唯一实测标尺——`tool_result_max_chars`
        # 是**字符**口径的估算（`//2+16`），只有它能给出真实 token 占用。
        _segl = [[len(sg["ids"]) for sg in si if sg.get("kind") == "assistant"]
                 for si in segs_i]
        _tsegl = [[len(sg["ids"]) for sg in si if sg.get("kind") == "tool"]
                  for si in segs_i]

        def _sink(status, acc_list=None, fmt_list=None):
            """轨迹落盘钩子（traj_sink=None → 零成本，见 collect_retool_group docstring）。

            三个终局分支（overlong / uniform / ok）共用它——**overlong 也必须落**：
            超长整组不进 record.jsonl，只在生成端日志留一个计数，正是记录口径
            幸存者偏差的主体；要回答"长度失控时模型在写什么"只能靠这批轨迹。"""
            if traj_sink is None:
                return
            for j in range(n):
                traj_sink.append(traj_dump_row(
                    segs_i[j], Q=inputs[i]["Q"], A=inputs[i]["A"], qk=_qks[i],
                    status=status, stats=code_stats[i * n + j], clen=clen_i[j],
                    acc=None if acc_list is None else float(acc_list[j]),
                    fmt=None if fmt_list is None else float(fmt_list[j])))

        # 逐样本全长预算检查（按题：单题超长不再连坐其他题，2026-09-10）。
        # plen_i 为真实 prompt 长——旧版用批内最长（含 pad）会把 pad 宽度算进
        # 每个样本的 token 预算，单题超长判定偏严。
        if mask_i.shape[1] == 0 or retool_context_overlong(
                per_ids_i, plen_i, cfg["max_context_tokens"]):
            _sink("overlong")
            results.append({"status": "overlong"})
            continue
        adv_i, acc_i, fmt_i, cu_i, ck_i, phase = retool_score_flat(
            [inputs[i]], asst_texts[i * n:(i + 1) * n],
            code_stats[i * n:(i + 1) * n], cfg, steps_elapsed=steps_elapsed,
            completion_lens=clen_i)
        _stats_i = code_stats[i * n:(i + 1) * n]
        _trunc_i = [int(s["trunc_final"]) for s in _stats_i]
        _call_cost = float(cfg.get("tool_call_cost", 0.0) or 0.0)
        _waste_pen = float(cfg.get("tool_waste_penalty", 0.0) or 0.0)
        _trunc_tail = float(cfg.get("trunc_tail_penalty", 0.0) or 0.0)
        if _call_cost > 0.0 or _waste_pen > 0.0 or _trunc_tail > 0.0:
            adv_i = tool_credit_advantages(
                adv_i, segs_i, mask_i.shape[1],
                call_cost=_call_cost, waste_penalty=_waste_pen,
                trunc_flags=_trunc_i, trunc_tail_penalty=_trunc_tail)
        # 零方差组（全对/全错，adv 恒 0 无梯度）：按题判定（2026-09-09 起
        # 与超长分流；2026-09-10 起不再连坐同批其他题）
        if not group_ok(adv_i):
            # 【2026-10-02】丢弃组也进轨迹 dump——它们不进 record.jsonl，
            # 却是"硬题是怎么死的"的唯一证据。
            _sink("uniform", acc_i, fmt_i)
            # 【2026-09-21 健康检查选择偏差修复】丢弃组也带诊断数据（acc/fmt/clen/
            # trunc），让 health.observe 能观测到被过滤组的截断率——否则高截断组
            # 被 overlong_filter 判为 uniform 后健康检查只看存活组 → trunc_rate 被低估。
            results.append({"status": "uniform", "acc": acc_i, "fmt": fmt_i,
                            "clen": clen_i, "cu": cu_i, "ck": ck_i,
                            "trunc": _trunc_i,
                            # 【2026-10-02】丢弃组同样带段长画像：B 桶在丢弃族也高发
                            # （硬题两头顶死），缺它会让"段长画像"只在 ok 族可见
                            # ——与 2026-09-21 clen/trunc 上送是同一类选择偏差修复。
                            "segl": _segl, "tsegl": _tsegl,
                            "inv": [int(s.get("invalid_final", 0))
                                    for s in code_stats[i * n:(i + 1) * n]],
                            # 【2026-09-25】uniform 组的末轮废码也如实上送：旧版这里
                            # 硬编码 [0]*n，把"丢弃组的 code_wasted"永久记成 0——
                            # 而原生协议的末轮调用正是丢弃组的高发形态（答案轮之前
                            # 才想起来调用），恒 0 会让该列在丢弃组上系统性偏低。
                            "cw": [int(s.get("code_wasted", 0))
                                   for s in code_stats[i * n:(i + 1) * n]],
                            # 【2026-10-02】"调用写到一半被预算切"也如实上送
                            "tric": [int(s.get("trunc_in_call", 0))
                                     for s in code_stats[i * n:(i + 1) * n]],
                            "qk": _qks[i], "Q": inputs[i]["Q"]})
            continue
        if use_vllm_logps:
            gen_logps_i = gen_logps_from_segs(segs_i)
            # 对拍（口径变更的验证钩子）只在 mask 有效位上比——工具段两路语义不同
            # （vLLM 路置 0，torch 路是真实重算值），比了没有意义。wants() 闸门
            # 在 torch 重算之前：预算用尽后不再触发，副本才能安全释放。
            if verify_logps is not None and (getattr(verify_logps, "wants", None) is None
                                             or verify_logps.wants()):
                verify_logps(gen_logps_i, compute_gen_logps(merged_i, plen_i), mask_i)
        else:
            gen_logps_i = compute_gen_logps(merged_i, plen_i)
        results.append({"status": "ok", "merged": merged_i, "mask": mask_i,
                        "gen_logps": gen_logps_i, "adv": adv_i, "acc": acc_i,
                        "fmt": fmt_i, "cu": cu_i, "ck": ck_i, "phase": phase,
                        "qk": _qks[i], "Q": inputs[i]["Q"],
                        "clen": clen_i,
                        # 【2026-10-02】段长画像（见上方 _segl 注释）：
                        # segl = assistant 段长表，tsegl = 工具段长表
                        "segl": _segl, "tsegl": _tsegl,
                        "trunc": [int(s["trunc_final"])
                                  for s in code_stats[i * n:(i + 1) * n]],
                        # 末轮写了代码却不会被执行的次数（该轨迹结构性无 boxed，
                        # 且 retool_stop 下 trunc_final 记不到它）
                        "cw": [int(s.get("code_wasted", 0))
                               for s in code_stats[i * n:(i + 1) * n]],
                        # 【2026-10-02 trunc_in_call】"预算耗尽时正卡在未闭合调用块里"：
                        # token 档下 A 桶（trunc∩invalid）恒 0，这一个字段是 B 桶做
                        # "纯散文 / 调用被切"二分的唯一凭据。围栏档恒 0（键仍在）。
                        "tric": [int(s.get("trunc_in_call", 0))
                                 for s in code_stats[i * n:(i + 1) * n]],
                        # 【2026-09-25 原生协议诊断列】两列在围栏档恒为 0（键仍在
                        # record 里，方便同一张表逐列读两档）：
                        #   inv = 有 <tool_call> 但形态不认识 / 调用后还跟内容
                        #         （原生协议**唯一**的结构性负奖励入口）
                        #   ctxf = observation 放不下预算而终局
                        # 不落盘就看不见"模型在调用后又继续写"这种档位特有失效——
                        # 它与"啰嗦跑飞"在 acc/code/trunc 三列里完全同形。
                        "inv": [int(s.get("invalid_final", 0))
                                for s in code_stats[i * n:(i + 1) * n]],
                        "ctxf": [int(s.get("ctx_full", 0))
                                 for s in code_stats[i * n:(i + 1) * n]],
                        # sample_weight 只做整条轨迹过滤。两类结构性失败在拿到
                        # 逐token局部负优势后必须重新纳入 loss，否则又回到
                        # “判失败但零梯度”：废调用（tool_waste_penalty>0）与
                        # 截断末段（trunc_tail_penalty>0）。关闭新机制时保持历史
                        # 口径：trunc OR code_wasted 都整行清零。
                        "sw": torch.tensor(
                            [0.0 if ((s["trunc_final"] and _trunc_tail <= 0.0) or
                                     (s.get("code_wasted", 0) and _waste_pen <= 0.0))
                             else 1.0 for s in _stats_i],
                            dtype=torch.float32),
                        "plen": plen_i})
        # 【2026-10-02】上传组的轨迹 dump（与上面 overlong/uniform 同一落盘口径）
        _sink("ok", acc_i, fmt_i)
    return results


def _vllm_config_readback(llm, key: str, max_depth: int = 4, max_objs: int = 400):
    """best-effort 在**对象图**里回读 vLLM 构造参数，返回 (值, 相近键名提示)。

    【为什么不是写死路径】首版硬编码 LLM.llm_engine[.vllm_config].model_config 四条
    路径，2026-09-14 实机打回 None——该版引擎的对象布局与它们不符。于是"回读 None"
    既可能=键被静默忽略、也可能=我找错了地方，**判据失去分辨力**（那次只能靠"日志里
    没有 ninja"间接推断修复生效）。改为有界 BFS：**图遍历只走 `__dict__`**（大容器不
    展开）、**取值用 getattr**（覆盖类属性/property，只对已访问对象、逐键调用）、
    **提示用 dir**（只取名字，不触发 descriptor）。命中即返回；未命中把图里见过的
    "相近名字 + 宿主类型名"一并带回，让这个 None 自解释。定位是纯可观测，读不到不抛
    （版本间布局会继续漂）。
    """
    import types
    seen, visited, hints, seen_hints = set(), 0, [], set()
    queue = deque([(llm, 0)])
    stem = key.split("_")[0]        # 如 gdn：捞"名字相近但位置/拼法不同"的键，用于分辨 None
    while queue and visited < max_objs:
        obj, depth = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        visited += 1
        try:
            # 取值走 getattr（覆盖类属性/property，不只实例 __dict__）；属性可能惰性
            # 初始化并抛非 AttributeError → 整段兜住，只跳过这一个对象
            if hasattr(obj, key):
                return getattr(obj, key), hints[:12]
        except Exception:
            pass
        for k in dir(obj):          # 只取名字：dir 不触发 descriptor，安全
            if stem and stem in k and k not in seen_hints:
                seen_hints.add(k)
                hints.append(f"{type(obj).__name__}.{k}")   # 带宿主类型名，miss 时能直接定位布局
        if depth >= max_depth:
            continue
        for v in getattr(obj, "__dict__", {}).values():
            if isinstance(v, (types.ModuleType, torch.Tensor, str, bytes, type)) or v is None:
                continue
            if isinstance(v, (int, float, bool)):
                continue
            if isinstance(v, (list, tuple, dict, set)) and len(v) > 32:
                continue    # 大容器多半是权重/数据，不进图（防 BFS 爆开）
            queue.append((v, depth + 1))
    return None, hints[:12]


def _probe_torch_construct(model_path: str) -> dict:
    """在 **meta 设备**上真构造一次 AutoModelForCausalLM，**喂与真实加载同一个 config**。

    返回 {"error": 错误串|None, "composite": bool, "model_type": str}。

    【为什么必须同源】2026-09-15 实机教训：首版用 `from_config(顶层 cfg)` 探，而真实加载
    走 `from_pretrained(目录)` 的**自动解包**——前者自己会解包（本地实测两条都通），后者在
    gen 进程（import 过 vLLM）里不解包。于是预检在随后崩溃之前打印了"通过"：**预检通过的
    那条路，恰好不是会坏的那条路**。现在预检与加载共用
    `rlab.model_loading.resolve_load_config`，喂的输入相同，结论才可迁移。

    与真实加载同一条**类解析 + `__init__`** 路径（A1 就发生在这一步），但 meta 张量只记
    形状、不分配内存/显存，毫秒~秒级、零资源代价。不打印、不抛——由调用方处置。
    """
    from rlab.model_loading import resolve_load_config
    from transformers import AutoModelForCausalLM

    out = {"error": None, "composite": False, "model_type": "?"}
    try:
        cfg, composite = resolve_load_config(model_path)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["composite"], out["model_type"] = composite, getattr(cfg, "model_type", "?")
    try:
        with torch.device("meta"):
            _m = AutoModelForCausalLM.from_config(cfg)
        del _m
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _assert_torch_replica_loadable(model_path: str) -> None:
    """fail-fast：torch 侧副本能不能加载？**用与真实加载同一个 config** 真构造一次。

    【为什么不能按形状判】2026-09-14 实测的不对称：同一份复合 ckpt，train 侧（裸 torch
    进程）两次 run 都加载成功，gen 侧（进程内 import 过 vLLM）必抛 A1——差别在
    **环境/进程**，不在 ckpt 形状。按形状判会在环境修好（或换 transformers/vLLM 版本）
    之后**误杀合法启动**，而这正是"宁可不拦，不可错拦"要避免的。
    【判据】resolve_load_config 解析出的 config（复合 ckpt 即其 text_config——与加载
    同源）在 meta 上构造得出来 → 放行；报错含 vocab_size（A1 特征）→ fail-fast 给两条
    改法；其他失败 → 只告警，交给真实加载定夺。
    """
    r = _probe_torch_construct(model_path)
    if r["error"] is None:
        print(f"[rollout] torch 副本预检通过：本进程可构造 {model_path}"
              f"（口径={'显式 text_config' if r['composite'] else '顶层 config'}"
              f"/{r['model_type']}，与真实加载同一个 config）", flush=True)
        return
    err = r["error"]
    if "vocab_size" in err:
        raise RuntimeError(
            f"[rollout] torch 侧副本无法加载（{model_path}）：{err}\n"
            f"  这是 docs/04 A1（复合多模态 config 喂文本类），gen 侧副本是它的第三个"
            f"入口（用 --verify_gen_logps 做对拍时踩到）。"
            f"{'该 ckpt 顶层确实没有 vocab_size（在 text_config 里）。' if r['composite'] else ''}"
            f"注意本预检已与加载同源（复合 ckpt 显式喂 text_config），仍失败说明该口径"
            f"在本进程里也绕不过去。\n"
            f"  改法一（先查 transformers 版本）：要能加载复合 ckpt 必须有 qwen3_5_text "
            f"的前缀转换映射（`model.language_model.X` -> `model.X`，5.16 起内置）；"
            f"版本对就直接用统一目录 /root/Qwen3.5-4B（键名映射会自动开）。\n"
            f"  改法二（回退分裂加载）：model_path=<extract_text_model 抽出的纯文本目录> "
            f"+ --vllm_model_path <原多模态目录>；\n"
            f"  改法二（放弃对拍窗口）：去掉 --verify_gen_logps（副本只在 vllm_gen_logps "
            f"档位做对拍时才加载）。")
    print(f"[rollout][警告] torch 副本预检失败但**不像 A1**，继续让真实加载定夺：{err}",
          flush=True)


def map_attention_backend(backend: str, known_keys) -> dict:
    """把"显式指定 attention backend"映射成本版 vLLM 认识的键名。纯函数（CPU 可测）。

    【2026-09-15 真机】`VLLM_BATCH_INVARIANT=1` 在 v0.19.1 里存在（envs.py:78），但直接
    开会启动即失败：
      RuntimeError: VLLM batch_invariant mode requires an attention backend in
      ['FLASH_ATTN','TRITON_ATTN','FLASH_ATTN_MLA','TRITON_MLA'], but got 'None'
    因为该检查跑在 backend 解析**之前**，必须显式给。键名跨版本有
    `attention_config={"backend": ...}` 与 `attention_backend=...` 两形态——**问注册表**而
    不硬编码（同 _check_vllm_gen_kwargs）；两个都没有就 raise：静默忽略会让"开了确定性档"
    变成"其实没开"。"""
    if "attention_config" in known_keys:
        return {"attention_config": {"backend": str(backend)}}
    if "attention_backend" in known_keys:
        return {"attention_backend": str(backend)}
    raise RuntimeError(
        f"[rollout] 本版 vLLM 的引擎参数里既没有 attention_config 也没有 attention_backend"
        f"（已知键样例：{sorted(k for k in known_keys if 'attn' in k)[:8]}）——"
        "无法显式指定 attention backend，VLLM_BATCH_INVARIANT=1 会启动即失败")


def attention_backend_kwargs(backend: str) -> dict:
    """从 vLLM 自己的 CLI 注册表取键名（探测不到就 raise，不静默）。"""
    import argparse
    from vllm.engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser)
    return map_attention_backend(backend, {a.dest for a in parser._actions})


def batch_invariant_guard(enabled, attention_backend) -> None:
    """确定性档的前置检查（fail-fast 在引擎构造之前）。纯逻辑，CPU 可测。

    真机代价：`VLLM_BATCH_INVARIANT=1` 而没给 attention backend → 引擎初始化时抛
    RuntimeError（白等一次 ~17s 的引擎启动）；反过来给了 backend 但没开确定性档，
    则"确定性"这个前提其实不成立——所以两件事必须同时声明。"""
    if enabled and not attention_backend:
        raise RuntimeError(
            "[rollout] vllm_batch_invariant=True 但没有 vllm_attention_backend："
            "vLLM 的 batch-invariant 检查跑在 attention backend 解析之前，会直接抛 "
            "RuntimeError（requires an attention backend in ['FLASH_ATTN', ...], got 'None'）。"
            "请同时给 --vllm_attention_backend FLASH_ATTN（或 TRITON_ATTN）")


def gdn_backend_missing(vllm_path, gen_kwargs) -> bool:
    """Qwen3.5 系 + 生效引擎参数里没有 `gdn_prefill_backend` ⇒ 会落到 FlashInfer GDN prefill。

    【2026-09-16 为什么单独一条】自本日起 `BASE.vllm_gen_kwargs` 默认就含
    `{"gdn_prefill_backend": "triton"}`，但 **默认 ≠ 兜底**：`--vllm_gen_kwargs` 是
    **整体替换**（train.py 写 overrides → config.get_config 的 `cfg.update`），
    所以"只想再加一个键"（例如 `enable_prefix_caching`）时若不把 triton 一并写回，
    就会静默掉回 FlashInfer——正是 2026-09-14 那个**无 traceback** 的 SIGKILL 档。
    对照：`--vllm_attention_backend` 没有这个陷阱（rollout 里是 merge 进副本）。

    纯逻辑，CPU 可测；调用点只告警不擅自改档（静默替换被测配置是禁止操作）。
    """
    if "Qwen3.5" not in str(vllm_path or ""):
        return False                       # 非 GDN 模型：该键本就不被使用，缺了也无害
    return not (gen_kwargs or {}).get("gdn_prefill_backend")


def _check_vllm_gen_kwargs(kwargs: dict) -> None:
    """fail-fast：在**构造 LLM 之前**把 vLLM 不认识的引擎参数键名拦下。

    【为什么不能只靠回读】LLM(...) 的构造过程本身就是那次 FlashInfer JIT（GDN warmup
    跑在引擎初始化里）——等构造完再回读，进程可能已经被 OOM-killer 带走了。判据必须
    在构造之前。键名表取自 vLLM 自己的 CLI 注册表（EngineArgs.add_cli_args），不硬编码
    键名；探测不到该入口（版本漂）时**跳过检查**并告警，宁可不拦、不可错拦。
    """
    if not kwargs:
        return
    try:
        import argparse
        from vllm.engine.arg_utils import EngineArgs
        parser = argparse.ArgumentParser(add_help=False)
        EngineArgs.add_cli_args(parser)
        known = {a.dest for a in parser._actions}   # dest 即下划线键名（--gdn-prefill-backend -> gdn_prefill_backend）
    except Exception as e:
        print(f"[rollout][警告] 无法核对 vLLM 引擎参数键名（{type(e).__name__}: {e}）"
              f"——键名写错会被 vLLM 静默忽略，请自查", flush=True)
        return
    unknown = sorted(set(kwargs) - known)
    if not unknown:
        return
    near = sorted(k for k in known if any(t in k for t in ("gdn", "attn", "eager", "logits")))
    raise RuntimeError(
        f"[rollout] vllm_gen_kwargs 里有 vLLM 不认识的键：{unknown}\n"
        f"  vLLM 对未知构造参数是静默忽略的——键名写错的代价是「修复看起来做了、"
        f"实际没做」（照旧走 FlashInfer JIT -> 起跑期被 OOM-kill，且无 traceback）。\n"
        f"  键名用下划线形态：CLI 的 --gdn-prefill-backend 对应 gdn_prefill_backend。\n"
        f"  相近的已注册键样例：{near[:12]}")


def gen_worker(Q, cfg: dict):
    """生成端主入口（由 train.py rank0 spawn，或分进程模式独立运行）。

    Q: mp.Queue，训练端每 gen_update_steps 步 put 一次 state_dict；独立模式传 None。
    """
    for key in _DEEPSPEED_ENV_KEYS:
        os.environ.pop(key, None)
    # 【2026-09-11 IPC 坑】train.py 顶层把 allocator 环境强制成 False（CUDA IPC
    # 过 mp.Queue 需要普通段，见 train.py 注释），spawn 子进程会原样继承——生成端
    # 入口改回 True：GPU0 三方共居（vLLM 0.30 池 + ref + torch 副本）的碎片治理
    # 仍依赖 expandable_segments。必须在首个 CUDA 分配前改（此处只 set_device，
    # 尚无 caching-allocator 分配，env 语义来得及生效）。
    for _alloc_k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        os.environ[_alloc_k] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gen_device"])
    torch.cuda.set_device(0)
    print(f"[rollout] generation worker on GPU {cfg['gen_device']}")

    import requests
    from transformers import AutoTokenizer   # 副本加载收口到 rlab.model_loading
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    # 显存占比 0.35→0.45（2026-09-09 提速）：GPU0 = ref(~7G) + vLLM + torch副本(~7G)
    # + gen_logps logits 瞬时峰(~6-12G)，0.45×96=43G 总计 ~70G < 96G，安全。
    # 3B+GQA 的 KV 极小（每条 5k token 才 ~370MB），旧 0.35 的 KV 池大量闲置——
    # 多给 vLLM 显存主要扩大 continuous batching 的调度余量。
    # 【2026-09-11】vllm_gen_logps 档位不再加载 torch 副本（见下），腾出的 ~8G
    # 应回灌给 vLLM（gen_gpu_mem 0.30 → 0.38 量级）——生成是每步耗时的大头。
    # 【2026-09-14 起跑期 OOM-kill】引擎参数透传（config.vllm_gen_kwargs，None=零变化）。
    # 首例 gdn_prefill_backend=triton：Qwen3.5 的 GDN 层默认走 FlashInfer JIT 现场
    # 编译，nvcc 的宿主 RAM 峰值与三方搬权重的启动期重合 → 生成端被 SIGKILL 带走、
    # 无 traceback（详见 config.vllm_gen_kwargs 的症状→根因记录）。
    # 【副本档位判定前置】这两行只读 cfg、与引擎无关——提前到 LLM() 之前，好让 A1
    # 护栏在任何 GPU 分配**之前**拦下（否则要先花 ~1 分钟把 vLLM 起起来才崩，2026-09-14
    # 实机就是这么白烧了 2.5 分钟才看到一条 transformers 内部 traceback）。
    _use_vllm_logps = bool(cfg.get("vllm_gen_logps")) and cfg["algo"] in ("retool", "retool_math")
    _verify_budget = int(cfg.get("verify_gen_logps", 0) or 0)
    # 【2026-09-15 实锤】Qwen3.5-4B + vLLM v0.19.1 + triton GDN prefill 下，vLLM 采样
    # logprobs 不可复现（同一命令两次：token 一致率 2.91%，|Δlogp| p99=5.79、max=14.5）。
    # 【2026-09-16 修好】`VLLM_BATCH_INVARIANT=1` + 显式 attention backend 后跨进程
    # 逐位可复现（100.00%、Δ=0），跨引擎残差 max 0.78nat、step-1 clip_frac 0.0008
    # ≈ torch 副本地板 0.0007（docs/07 §9）。所以这条警告必须**分档**说——否则开了
    # 确定性档还照旧吓人，等于把"已修复"当"已知坏"用。
    if _use_vllm_logps and "Qwen3.5-4B" in cfg.get("model_path", ""):
        if cfg.get("vllm_batch_invariant"):
            print("[rollout] vllm_gen_logps=True 且确定性档已开（VLLM_BATCH_INVARIANT=1）："
                  "本环境实测跨进程逐位可复现（token 一致率 100%、|Δlogp|=0），"
                  "跨引擎残差 max≈0.78nat、step-1 clip_frac 0.0008 ≈ torch 副本地板 0.0007。"
                  "注意：开档会改变采样数值本身，**跨档的数字不可比**。详见 docs/07 §9",
                  flush=True)
        else:
            print("[rollout][警告] vllm_gen_logps=True 且 model_path 含 Qwen3.5-4B，"
                  "**未开确定性档**：本环境 vLLM 采样 logprobs 已实测不可复现（同一命令"
                  "两次 token 一致率 2.91%、max 14.5nat），会把随机量注入 importance ratio；"
                  "两条修法——① vllm_gen_logps=False（torch 副本重算，与训练前向同源）；"
                  "② 加 --vllm_batch_invariant --vllm_attention_backend FLASH_ATTN"
                  "（实测可复现，见 docs/07 §9）。", flush=True)
    if (not _use_vllm_logps) or _verify_budget > 0:
        _assert_torch_replica_loadable(cfg["model_path"])   # A1：复合 ckpt 进不了 torch
    _gen_kwargs = cfg.get("vllm_gen_kwargs") or {}
    # 【2026-09-15】确定性档：真机实测不开时同进程背靠背同请求的 top-K 字典 3/3 不同、
    # top-1 logp 抖动 0.19nat；开 VLLM_BATCH_INVARIANT=1 + 显式 attention backend 后
    # 3/3 全同（spread=0）。env 必须在 vLLM 读取之前设好（envs 是惰性 lambda，本进程内
    # 设置即可），故放在这里一次性设置 + 前置检查。
    _attn_be = cfg.get("vllm_attention_backend")
    batch_invariant_guard(bool(cfg.get("vllm_batch_invariant")), _attn_be)
    if cfg.get("vllm_batch_invariant"):
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        print("[rollout] 已开 VLLM_BATCH_INVARIANT=1（确定性档：会关 custom all-reduce、"
              "改用确定性 kernel，吞吐有代价）", flush=True)
    if _attn_be:
        _gen_kwargs = dict(_gen_kwargs)
        _gen_kwargs.update(attention_backend_kwargs(_attn_be))
        print(f"[rollout] attention backend={_attn_be} → {_gen_kwargs}", flush=True)
    if gdn_backend_missing(cfg.get("vllm_model_path") or cfg.get("model_path"),
                           _gen_kwargs):
        print("[rollout][警告] 本档 vllm_gen_kwargs 里没有 gdn_prefill_backend，"
              "Qwen3.5 的 GDN prefill 会落到 **FlashInfer JIT**——本 pod 已两次实锤："
              "ninja 调 nvcc 打爆宿主 RAM → 生成端被 SIGKILL(9)、无 traceback、"
              "训练端只看到「生成端进程已退出」。\n"
              "        改法（注意 `--vllm_gen_kwargs` 是整体替换，默认的 triton 要写回）："
              " --vllm_gen_kwargs '{\"gdn_prefill_backend\": \"triton\"}'"
              "（要与别的键合并就一并写在同一个 JSON 里）。", flush=True)
    _check_vllm_gen_kwargs(_gen_kwargs)   # 键名错 = 静默忽略 = 修复白做，必须构造前拦
    vllm_gen = LLM(model=cfg.get("vllm_model_path") or cfg["model_path"],
                   gpu_memory_utilization=float(cfg.get("gen_gpu_mem", 0.45)),
                   **_gen_kwargs)
    if _gen_kwargs:
        # 「开了 X」与「X 在生效」是两回事（docs/04 §5.3）：回读是唯一能当场证明 kernel
        # 换成功了的证据。**None 有歧义**（键被忽略 / 我找错了地方），故把"相近键名"
        # 一并打出来自解释：有相近名字=布局变了（值大概率生效），一个都没有=真被忽略。
        print(f"[rollout] vLLM 引擎参数透传: {_gen_kwargs}", flush=True)
        for _k in sorted(_gen_kwargs):
            _v, _hints = _vllm_config_readback(vllm_gen, _k)
            print(f"[rollout]   回读 {_k} = {_v!r}"
                  f"{'' if _v is not None else f'｜相似键名 {_hints}（空=图里没这个名字，多半被静默忽略；非空=布局变了，值仍可能生效）'}",
                  flush=True)
    # torch 副本：只用它前向算 gen_logps（vLLM prompt_logprobs 路径 hang 的教训）。
    # 【减法① 2026-09-11】vllm_gen_logps 档位改用逐轮采样 logprobs，不再需要副本
    # ——GPU0 省 ~8G（可抬高 gen_gpu_mem 扩大 KV 池提速生成）；仅当要对拍
    # （verify_gen_logps>0）时才临时加载，验完立即释放。（档位判定已前置到 LLM() 之前）
    # SamplingParams 的可用字段（msgspec/dataclass 两代实现）——logprobs_mode 是
    # 较新版本才有的字段，老版本硬传会 TypeError，用字段表探测而不是 try/except
    _sp_fields = set(getattr(SamplingParams, "__struct_fields__", ()) or ()) | \
        set(getattr(SamplingParams, "__dataclass_fields__", {}) or {})
    _torch_holder = [None]
    if (not _use_vllm_logps) or _verify_budget > 0:
        # A1 护栏已在 LLM() 之前拦过。加载与预检**同源**（rlab.model_loading.resolve_load_config）：
        # 探的输入就是跑的输入，不存在"探了 A、跑了 B"。
        _torch_holder[0] = load_causal_lm(
            cfg["model_path"], dtype=torch.bfloat16,
            attn_implementation=cfg.get("attn_implementation", "sdpa")).cuda().eval()
        print(f"[rollout] torch gen_logps 副本已加载"
              f"{'（仅用于前 %d 组对拍，验完释放）' % _verify_budget if _use_vllm_logps else ''}")
    else:
        print("[rollout] vllm_gen_logps 档位：不加载 torch 副本（GPU0 省 ~8G）")
    if _use_vllm_logps:
        _t = float(cfg.get("temperature", 1.0))
        _tk = cfg.get("top_k", -1)
        _tp = float(cfg.get("top_p", 1.0))
        if _t != 1.0 or _tk not in (-1, None) or _tp != 1.0:
            print(f"[rollout][警告] vllm_gen_logps 下采样有后处理（temperature={_t} "
                  f"top_k={_tk} top_p={_tp}）——若 vLLM 版本不支持 logprobs_mode="
                  "raw_logprobs，返回的是后处理 logprob，与 torch 重算口径不一致；"
                  "务必用 --verify_gen_logps 对拍后再长跑", flush=True)

    sampling_params = SamplingParams(n=cfg["num_pre_Q"], temperature=cfg["temperature"],
                                     max_tokens=cfg["max_gen_tokens"], top_p=cfg["top_p"],
                                     top_k=cfg.get("top_k", 50),
                                     seed=cfg.get("seed"))
    # 阶段2 retool：每题先扩成 num_pre_Q 条独立轨迹再进 multi_turn_rollout_group，
    # 每条一个独立请求（n=1）且 seed 各不相同——共用 seed 会让同题各条生成完全
    # 相同的轨迹（组内零方差 → group_ok 永假 → 无限重采）。单段长度上限
    # round_gen_tokens，总上下文由 max_rounds × 单段约束。

    # 可复现种子：抽题顺序(random) + 生成采样(vLLM SamplingParams.seed)。
    # 对比实验固定 seed 后可按"更新数配对"做单变量比较（同 seed 下 vLLM 采样可复现）。
    seed = cfg.get("seed")
    if seed is not None:
        import random as _random
        _random.seed(seed)
        torch.manual_seed(seed)
        print(f"[rollout] 已固定训练种子 seed={seed}（抽题顺序 + vLLM 采样）")

    QAs = load_qas(cfg["data_task"])
    print(f"[rollout] 数据集 {cfg['data_task']} 共 {len(QAs)} 题")
    # 离线难度预探测过滤（2026-09-10）：必须在 QuestionScheduler 构造之前——
    # scheduler 的黑名单只覆盖"在线观察到连续零方差"的题，base 从未做对过的题
    # （p≈0，丢弃率 81% 的主体）由静态过滤在训练开始前一次性出清。
    if cfg.get("difficulty_path"):
        # 【2026-09-18 M4】训练端当前协议指纹 → load_difficulty_table 校验表是否
        # 同一『模型×提示×预算』下探的（换预算/提示/k 续跑同一 --out 会静默混表）。
        import hashlib as _hl
        _expected_meta = {
            "model": os.path.basename(str(cfg["model_path"]).rstrip("/")),
            # 【2026-09-20】不再传 k：探针的 k（每题探几条）与训练的 num_pre_Q
            # （每题采几条）语义不同、本就不相等，传了会让 load_difficulty_table
            # 每次启动必误报一次（真混表信号被"忽略习惯"淹掉）。见 data.py docstring。
            "rounds": cfg.get("max_rounds"), "round_tokens": cfg.get("round_gen_tokens"),
            "ctx": cfg.get("max_context_tokens"), "temp": cfg.get("temperature"),
            "sp": _hl.sha1(str(cfg.get("system_prompt", "")).encode("utf-8")).hexdigest()[:6],
            # 【2026-09-25】协议档也进比对：表是"模型×提示×预算×**协议**"的联合
            # 产物（两档的 base 通过率是两个分布），不比对就会把围栏表当原生档用。
            "tool_protocol": cfg.get("tool_protocol") or "fence",
            # 【2026-09-29】预算档也进比对：轮数档与 token 档的终止结构不同（末轮
            # 截止 vs 预算判据），同模型同提示下的通过率是两个分布——不比对就会把
            # 轮数档探的表当 token 档的难度表用（与协议档同一类静默混表）。
            "max_traj_tokens": int(cfg.get("max_traj_tokens", 0) or 0),
            "answer_reserve": int(cfg.get("answer_reserve", 0) or 0),
            # 【2026-09-29 budget 第四件套】prompt 上限也进比对：它决定哪些题被
            # 跳组（plen 超限不采）且进 overlong 参考系（ctx − plen），换它探的
            # 表与当前训练分布不符（与预算档/协议档同一类静默混表）。
            "max_prompt_length": int(cfg.get("max_prompt_length", 0) or 0),
        }
        _table = load_difficulty_table(cfg["difficulty_path"], expected_meta=_expected_meta)
        _lo, _hi = cfg.get("difficulty_band", (0.0, 1.0))
        QAs, _dstat = filter_qas_by_difficulty(QAs, _table, lo=_lo, hi=_hi)
        print(f"[rollout] 难度过滤 band=({_lo},{_hi}): {_dstat['total']} -> {_dstat['kept']} 题"
              f"（全错 {_dstat['p_zero']} / 全对 {_dstat['p_one']} / 区间外 {_dstat['band_out']}"
              f" / 表中缺失 {_dstat['missing']}）")
        if not QAs:
            raise RuntimeError(
                "[rollout] 难度过滤后训练池为空——重跑 rlab.probe_difficulty 刷新探针表，"
                "放宽 difficulty_band，或去掉 --difficulty_path")
        if len(QAs) < cfg["q_pool_reset_floor"]:
            print(f"[rollout] 警告: 过滤后池子 {len(QAs)} 题 < q_pool_reset_floor="
                  f"{cfg['q_pool_reset_floor']}（QuestionScheduler 的重置机制会很活跃）")
    ref_server = cfg["ref_server"]
    pushes = [0]   # 权重推送次数（每 gen_update_steps 优化步一次；近似 optimizer step）
    last_fp = [None]   # 上次推送的权重指纹（两次相同 = 训练端权重没在变）
    # 当前 vLLM/副本 权重对应的 train micro-step（staleness 标签）。
    # 【初值必须是 0 而不是 None】第 1~15 步还没发生第一次 state_dict 推送，但
    # rollout 的权重就是训练端的初始 checkpoint（= version 0），真实 staleness
    # 是 `step − 0`。初值 None 会让这段最"新鲜"的数据报成 `staleness=-1`
    # （2026-09-12 实测日志：`gen_version=None staleness=-1 micro-step(-0.2)`），
    # 把整段标签作废。None 只应出现在"训练端用旧协议裸传 state_dict"的兼容路径。
    policy_version = [0]
    health = _HealthMonitor()
    # 键名映射判定：【2026-09-15 澄清】判据是**两端模型的键名形态**，不是"两份
    # checkpoint 是不是同一个目录"。torch 侧一律按纯文本类加载（model.X/lm_head.*），
    # vLLM 侧吃多模态复合体时参数叫 model.language_model.X → 统一目录也必须映射。
    # 旧判据 `bool(vllm_model_path)` 会在"统一用一份复合 ckpt"时给出 False（映射被
    # 静默关掉 → 同步全落空），见 sync.need_text_to_mm_remap 的 docstring。
    _vllm_path = cfg.get("vllm_model_path") or cfg["model_path"]
    try:
        _vk_composite = bool(resolve_load_config(_vllm_path)[1])
    except Exception as _e:
        _vk_composite = False
        print(f"[rollout][警告] 解析 vLLM 侧 config 失败（{type(_e).__name__}: {_e}）"
              "——键名映射退回旧判据（只看 vllm_model_path 是否显式给出）", flush=True)
    _need_remap = need_text_to_mm_remap(vllm_checkpoint_composite=_vk_composite,
                                        vllm_model_path_set=bool(cfg.get("vllm_model_path")))
    print(f"[rollout] 权重同步键名映射: {'开' if _need_remap else '关'}"
          f"（torch={cfg['model_path']} 纯文本布局 → vLLM={_vllm_path} "
          f"{'多模态复合体' if _vk_composite else '同布局'}；判据=键名形态而非目录异同）",
          flush=True)

    def try_update_model():
        nonlocal pushes
        if Q is None:
            return
        try:
            item = Q.get_nowait()
        except _queue.Empty:
            return
        # 【2026-09-12 staleness 可观测】训练端推的是 (version, state_dict)，
        # version = 推送时的 train micro-step。旧协议只有 state_dict（version=None）。
        # 没有这个标签，训练端只能**假设**"这批数据是 16×k 步前的权重生成的"——
        # 而真机 run 的 approx_kl 忽 5e-4 忽 9.9e-2（200× 地板）、完全对不上 16 步
        # 推送周期，说明真实新鲜度根本不可控也不可测。标签让 PPO ratio 修正的
        # baseline 从"猜"变成"测"。
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
            gen_version, state_dict = item
        else:
            gen_version, state_dict = None, item
        print(f"[rollout] recving new model ... (version={gen_version})")
        try:
            # 顺序强制：先 vLLM 后 torch 副本，两者必须保持同一份权重
            # 键名映射开着时（判据见上面 need_text_to_mm_remap：torch 纯文本布局 →
            # vLLM 多模态布局，统一目录也算）同步走 remap_text_to_multimodal
            path = sync_weights_into_vllm(
                vllm_gen, state_dict,
                name_remap=remap_text_to_multimodal if _need_remap else None)
            if _torch_holder[0] is not None:
                # 副本必须与 vLLM 同步（顺序强制：先 vLLM 后副本），否则 gen_logps
                # 会用旧策略算——vllm_gen_logps 档位下无副本，跳过即可（不改语义）
                _torch_holder[0].load_state_dict(
                    {k: v.to(torch.bfloat16) for k, v in state_dict.items()})
            else:
                print("[rollout] vllm_gen_logps 档位：无 torch 副本，跳过其权重同步")
            print(f"[rollout] model updated via {path}, {len(state_dict)} tensors"
                  f" (version={gen_version})")
            policy_version[0] = gen_version
            pushes[0] += 1            # 权重推送计数（用于冷启动/后期奖励切换）
            # 权重指纹（float64，位级敏感）：两次推送指纹完全相同 = 训练端权重
            # 位级未变（优化器未步进/更新全被 bf16 舍入吞掉）。float32 求和会在
            # 3e8 元素上分辨率 ~0.5，淹没 bf16 单权重翻转 ~1e-4 → 假阳性（教训）。
            fp = _weight_fingerprint(state_dict)
            if fp == last_fp[0]:
                print("[健康检查] 本次推送权重指纹（float64）与上次完全相同 → 训练端"
                      "权重位级未变；若连续 2+ 次推送均如此，判定权重冻结，停止排查"
                      "训练端优化器路径", flush=True)
            last_fp[0] = fp
            del state_dict
        except Exception:
            import traceback
            traceback.print_exc()
            raise RuntimeError("[rollout] weight sync failed -> gen worker abort (fail-fast)")

    def compute_gen_logps(merged_ids: torch.Tensor, plen: int) -> torch.Tensor:
        # 已知妥协（阶段0 遗留，如实记录）：前向不传 attention_mask，左 pad 区
        # token 参与 attention——但训练端 policy 前向与 ref_server 前向同样不传，
        # 三方一致的偏差在 ratio（policy/gen）中抵消；教学规模实测可用。
        # 【2026-09-11】改走分块 logps：全长 logits (8, ~5.4k, 248320) ~22G 实测 OOM。
        if _torch_holder[0] is None:
            raise RuntimeError(
                "[rollout] vllm_gen_logps 档位不该调用 torch 重算路径（副本未加载）"
                "——检查 verify_gen_logps 与调用点是否一致")
        with torch.inference_mode():
            logps = forward_per_token_logps(
                _torch_holder[0], merged_ids.to(_torch_holder[0].device),
                seq_chunk=512,
                batch_chunk=max(1, int(cfg.get("fwd_batch_chunk", 1) or 1)))
            return logps[:, plen - 1:].cpu()

    def _free_torch_copy():
        _torch_holder[0] = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("[rollout][verify] 对拍结束，torch gen_logps 副本已释放（GPU0 归还 ~8G）",
              flush=True)

    _verifier = (LogpsVerifier(_verify_budget, on_finish=_free_torch_copy)
                 if (_use_vllm_logps and _verify_budget > 0) else None)

    def score_group(inputs, answers, completion_lens):
        """打分。返回 (scores, acc_s, fmt_s)。

        scores = advantage（非 rfpp，已组内归一化）或原始 reward（rfpp）。
        """
        rewards, acc_s, fmt_s = [], [], []
        n = cfg["num_pre_Q"]
        for i, inp in enumerate(inputs):
            for j, a in enumerate(answers[i * n:(i + 1) * n]):
                if cfg.get("data_task") in ("dapo_math", "dapo-math-17k", "math_dapo"):
                    sc = total_reward_math(inp["A"], a,
                                           completion_len=completion_lens[i * n + j],
                                           max_gen_tokens=cfg["max_gen_tokens"],
                                           overlong_buffer=cfg["overlong_buffer"],
                                           overlong_shaping=cfg["overlong_shaping"])
                else:
                    sc = total_reward(inp["A"], a, w_acc=2.0,
                                      completion_len=completion_lens[i * n + j],
                                      max_gen_tokens=cfg["max_gen_tokens"],
                                      overlong_buffer=cfg["overlong_buffer"],
                                      overlong_shaping=cfg["overlong_shaping"])
                rewards.append(sc["reward"]); acc_s.append(sc["acc"]); fmt_s.append(sc["format"])
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if cfg["algo"] == "rfpp":
            return rewards, torch.tensor(acc_s), torch.tensor(fmt_s)  # 原始分直传，服务端算 advantage
        adv = compute_advantages(rewards, cfg["num_pre_Q"], cfg["adv_mode"])
        return adv, torch.tensor(acc_s), torch.tensor(fmt_s)

    def make_retool_sps(n_req, seed_salt):
        """retool 家族每轨迹独立 SamplingParams（每条一个请求且 seed 各不相同
        ——共用 seed 会让同题各条生成完全相同的轨迹，组内零方差 → group_ok
        永假无限重采）。seed_salt：随重采/轮次递增的盐，防同 seed 复采同轨迹。

        【2026-09-18 stop 机制】cfg.retool_stop=True 时带上 protocol.
        RETOOL_STOP_KWARGS：模型写到代码块闭合围栏立即停（include_stop 保留围栏
        字节，extract_python_blocks 拿到完整块），沙箱结果紧跟代码回填——修复
        "代码→瞎猜→[结果]"错位与高截断（p5/p6 三轮 run 代码压灭的根因）。"""
        seed0 = cfg.get("seed")
        kw = dict(n=1, temperature=cfg["temperature"],
                  max_tokens=cfg.get("round_gen_tokens", 400),
                  top_p=cfg["top_p"], top_k=cfg.get("top_k", 50))
        if is_native_protocol(cfg):
            # 【原生协议默认不带 stop】模型自然停在 im_end（EOS 已在 token_ids 里，
            # docs/05 §12.3 已核对），参考实现也只传 tokenizer.eos_token。
            # `native_stop_at_call=True` 是**只在 base 冒烟看到"调用后还继续瞎写"**
            # 时才打开的兜底：那种样本 parse_assistant 判 invalid → 终局无 boxed
            # → reward -1，是原生协议唯一的结构性负奖励入口（docs/09 §0.1 症状③
            # 在原生协议下的对应形态）。默认关 = 单变量纪律（先看基线行为）。
            if cfg.get("native_stop_at_call"):
                kw.update({"stop": [NATIVE_CALL_STOP],
                           "include_stop_str_in_output": True})
            # 假 </think> 禁言（事故 C 根因修复，docs/09 §10.6.3）：et=False 档
            # think 段已由生成提示预填闭合，内容里再出现 </think> 永远是协议垃圾；
            # 不禁言时它进入 history 会被 Qwen3.5 模板重构成 reasoning/content 两段，
            # build_next_prompt 校验① 必炸。gen_logps 取 raw_logprobs（禁言前的
            # 真实策略 logp），训练 ratio 不失真——与 top_k=50 截断同一性质。
            if "bad_words" not in _sp_fields:
                raise ValueError(
                    "[rollout] 原生协议需要 SamplingParams.bad_words（禁言假 "
                    f"{NATIVE_BAD_WORDS[0]}，docs/09 §10.6.3），当前 vLLM 版本的 "
                    "SamplingParams 没有该字段——升级 vLLM 或回退 fence 档。")
            # 【事故 D（docs/09 §10.6.4）】special=True 的 added token（im_start/
            # endoftext/vision 系，除 eos）同样必须禁言：模型偶发在内容里采样出
            # 它们，vLLM .text 会静默丢弃（token_ids 保留）→ 下一轮校验① 必炸。
            kw["bad_words"] = (list(NATIVE_BAD_WORDS)
                               + native_special_ban_words(tokenizer))
        elif cfg.get("retool_stop"):
            kw.update(_RETOOL_STOP_KWARGS)
        if _use_vllm_logps:
            # 被采样 token 的 logprob（= log π(tok|完整前文)），逐轮收集即 gen_logps。
            # logprobs_mode=raw_logprobs：显式要"后处理前"的 logprob，防某些 vLLM
            # 版本默认返回经 temperature/top-k 处理后的值（本配置 temperature=1、
            # 无截断时两者相同，但显式声明不留歧义）。
            # 【2026-09-15 真机实锤】logprobs=0 这条"只报被采样 token"的形态在 vLLM
            # v0.19.1 + Qwen3.5 GDN 上**报的数与分布不符**：同一 prompt、同一位置、
            # 同一 token，logprobs=0 报 -0.602，而 logprobs=20（raw）报 -7.40、torch
            # 独立重算也是 -7.400 —— 两条独立计算一致、只有 logprobs=0 是离群者，
            # prefill 位（每轮首 token）最狠，单点差 6.8 nat（训练对拍里的 12.8 同源）。
            # 故取"top-K + 从中挑被采样 token"的形态；K 由 cfg 给（默认 0 = 旧行为，
            # 便于 A/B；修复档建议 1 起步——够用且返回体最小）。
            kw["logprobs"] = int(cfg.get("vllm_logprobs_n", 0) or 0)
            if "logprobs_mode" in _sp_fields:
                kw["logprobs_mode"] = "raw_logprobs"
        return [SamplingParams(**kw,
                               seed=(seed0 + seed_salt + k
                                     if seed0 is not None else None))
                for k in range(n_req)]

    # ------------------------- 采样主循环 -------------------------
    os.makedirs(os.path.dirname(os.path.abspath(cfg["record_path"])), exist_ok=True)
    fout = open(cfg["record_path"], "a", encoding="utf-8")
    uploaded_total = 0
    is_retool = cfg["algo"] in ("retool", "retool_math")
    # 【2026-10-02 轨迹全量落盘】record.jsonl 只有统计，看不到"模型写了什么"。
    # 开启后写 <out_dir>/traj.jsonl：**含被丢弃的 attempt**（uniform/overlong）
    # ——那两类整组不进 record，是记录口径幸存者偏差的主体。逐 attempt flush：
    # 生成端被信号杀死时（本项目已遇两次）最后一批轨迹不随缓冲区蒸发。
    # 只实现于 retool 家族（段结构 = 「末段写了什么」的载体）；单轮档若开，
    # 显式告警而不是留一个空文件让人以为"dump 了但都是空"。
    ftraj = None
    if cfg.get("traj_dump") and not is_retool:
        print("[rollout][警告] traj_dump 目前只实现于 retool 家族（单轮档无段结构）"
              "→ 本 run 不会落轨迹", flush=True)
    if cfg.get("traj_dump") and is_retool:
        _traj_path = os.path.join(
            os.path.dirname(os.path.abspath(cfg["record_path"])), "traj.jsonl")
        ftraj = open(_traj_path, "a", encoding="utf-8")
        print(f"[rollout] 轨迹全量落盘 -> {_traj_path}"
              f"（段级文本；含被丢弃的 uniform/overlong attempt）", flush=True)
    n_traj_rows = 0
    n_traj_bad = 0     # 无法序列化而跳过的行（见下；绝不因此中断训练）
    uploaded_total = 0
    rollout_seq = [0]   # 全局递增的 rollout 计数（丢组重采的 seed 盐，防同 seed 复采）
    samp_stats = {"attempts": 0, "uniform": 0, "overlong": 0,
                  "prompt_overlong": 0}
    _pad_logged = [False]   # 剥左 pad 的可见性只打一次（见 retool 分支）
    # 题目级调度两条路径（2026-09-10 重构）：
    # - 队列路径（gen_questions_per_attempt>1，当前仅 retool_math）：QuestionScheduler
    #   顺序走池 + 同题重试 + 拉黑——旧 random.sample 全池抽题下同题重抽概率
    #   1/16500，streak 永不累计，过滤器是死代码（丢弃率维持 ~64% 的根因）。
    # - 旧路径（gen_questions_per_attempt=1，GSM8K retool 家族）：random.sample
    #   + q_stat 原样保留——阶段2 GSM8K 协议可比性不破坏。
    multi_q = max(1, int(cfg.get("gen_questions_per_attempt", 1) or 1))
    use_qqueue = is_retool and multi_q > 1 and bool(cfg.get("q_skip_streak"))
    sched = (QuestionScheduler(QAs, cfg["q_skip_streak"], cfg["q_pool_reset_floor"],
                               ttl=int(cfg.get("q_blacklist_ttl", 0) or 0))
             if use_qqueue else None)
    q_stat = {}   # 旧路径专用：Q 文本 -> 连续零方差组次数
    # 【2026-09-12 反压状态机】外层轮次零产出计数 + 窗口丢弃率游标
    _zy_limit = int(cfg.get("sampler_max_zero_yield", 0) or 0)
    _disc_alert = float(cfg.get("discard_alert", 0.0) or 0.0)
    _disc_abort = float(cfg.get("discard_abort", 0.0) or 0.0)
    _count_ov = bool(cfg.get("overlong_counts_toward_skip", True))
    zero_yield = 0
    _mark = {"a": 0, "d": 0}     # 上次打点时的 attempts / 丢弃总数（算窗口率，不是累计率）
    _disc_fired = set()
    while True:
        try_update_model()
        # dynamic sampling（DAPO 机制2）：全同组不占配额，继续采直到攒够 Q_batch_size 组
        need = cfg["Q_batch_size"]
        groups = []   # 单轮: (inputs, prompt_text, prompt_ids, ans_ids, adv, acc, fmt, plen)
                      # retool: ready_dict（含 plen，见 collect_retool_group）
        attempts = 0   # 按题计数（丢弃率口径 = 白跑的题次数）
        max_attempts = need * cfg["dynamic_max_attempts_mult"]
        if use_qqueue:
            max_attempts *= multi_q   # 队列路径一次并采 multi_q 题，上限按题数等比放大
        while len(groups) < need and attempts < max_attempts:
            if use_qqueue:
                inputs = sched.draw(multi_q)
                if not inputs:
                    break
                attempt_units = len(inputs)
                attempts += attempt_units
            else:
                attempt_units = 1
                attempts += attempt_units
                if is_retool and cfg.get("q_skip_streak"):
                    cand, _reset = filter_question_pool(
                        QAs, q_stat, cfg["q_skip_streak"], cfg["q_pool_reset_floor"])
                    if _reset:
                        print("[rollout] 题目级过滤池低于下限，全部重置（难题重新入场）")
                    inputs = random.sample(cand, need)
                else:
                    inputs = random.sample(QAs, need)
            qkey = inputs[0]["Q"] if need == 1 else None
            # 按协议档位构造 prompt。原生档**同一份 messages** 同时喂这里（出 ids）
            # 与 collect_retool_group（多轮用）——结构性保证两处同源。
            _pmsgs = prompt_messages_for(inputs, cfg) if is_native_protocol(cfg) else None
            prompts_text, prompt_ids, plen = build_prompt_batch(
                inputs, cfg, tokenizer, prompts_messages=_pmsgs)
            if plen > cfg["max_prompt_length"]:
                # attempts 按题计数；整批 prompt 超限时每道题都已消费一次 attempt，
                # 但尚未进入 uniform/trajectory-overlong 分支。旧版直接 continue，
                # 导致累计/窗口丢弃率少算这批题（native_p4 实测低报约 4pp）。
                samp_stats["prompt_overlong"] += attempt_units
                continue
            if is_retool:
                # 阶段2：多轮代码交织（并采 multi_q 题，vLLM 并发 = 题数×num_pre_Q）
                # → 按题打分/拆分 → 每题独立上传批（mask 已按段边界算好）
                sps = make_retool_sps(len(inputs) * cfg["num_pre_Q"], rollout_seq[0])
                _sink = [] if ftraj is not None else None
                results = collect_retool_group(
                    vllm_gen, tokenizer, cfg, compute_gen_logps,
                    inputs, prompts_text, prompt_ids, plen, sps,
                    steps_elapsed=pushes[0] * cfg["gen_update_steps"],
                    verify_logps=_verifier, prompts_messages=_pmsgs,
                    traj_sink=_sink)
                if _sink:
                    # gen_version 由调用方补（rollout 循环看不到权重推送计数）：
                    # 轨迹与训练 step 的对应关系全靠它，缺了就无法按"哪版权重"分层。
                    for _row in _sink:
                        _row["gen_version"] = policy_version[0]
                        try:
                            ftraj.write(json.dumps(_row, ensure_ascii=False) + "\n")
                        except (TypeError, ValueError, UnicodeEncodeError):
                            # 【诊断产物绝不拖死训练】文本里万一出现无法序列化的
                            # 东西（孤立代理项/意外类型），只跳过这一行并计数；
                            # 让观测面把 16h 的训练 run 崩掉是本末倒置。
                            n_traj_bad += 1
                            continue
                    ftraj.flush()
                    n_traj_rows += len(_sink)
                # 剥 pad 可见性（一次性）：本批最长 prompt token 数 vs 各题真实长度。
                # 静默改变 token 预算是这类"口径修正"最难排查的形态，打一行自证。
                if not _pad_logged[0]:
                    _trues = [r["plen"] for r in results if r["status"] == "ok"]
                    if _trues and min(_trues) < plen:
                        _pad_logged[0] = True   # 只在真发生剥除时消费这一次机会
                        print(f"[rollout] 已剥左 pad：批内最长 prompt {plen} token，"
                              f"各题真实 {min(_trues)}~{max(_trues)}"
                              "（打分序列与 vLLM 生成序列逐 token/逐位置同源）",
                              flush=True)
                # 【2026-09-20 修复·盐步长必须等于本次消耗的 seed 数】
                # make_retool_sps 占用 [salt, salt+n_req)，n_req = 题数×num_pre_Q
                # （retool_math = 4×8 = 32）。旧版每次 attempt 只 +=1 → 相邻 attempt
                # 的 seed 区间**重叠 31/32 = 97%**，要 32 次 attempt 才走出重叠区。
                # 后果（与 vllm_batch_invariant=True 叠加最狠）：同题 uniform 被插回
                # 队首重采时，8 条里 7-8 条 seed 上次已用过，而相邻 attempt 之间多数
                # 没有权重推送（gen_update_steps=8）→ 同 seed+同 prompt+同权重 =
                # 同轨迹 → "重采"复现上次结果 → 再次 uniform → q_skip_streak=2 达标
                # 拉黑。**题目被拉黑的真实原因是 seed 复用，不是"当前学不动"**，
                # QuestionScheduler 的判据因此失真（p8 丢弃率 25~31% 零方差主导且
                # 不随时间下降）。组内 8 条 seed 一直是互不相同的（零方差与此无关）。
                rollout_seq[0] += len(inputs) * cfg["num_pre_Q"]
                for q, res in zip(inputs, results):
                    if res["status"] == "uniform":
                        samp_stats["uniform"] += 1
                        # 【2026-09-21 健康检查选择偏差修复】丢弃组也观测：
                        # overlong_filter 下高截断组被判 uniform，不观测则
                        # trunc_rate 只看存活组 → 被低估 → 不触发告警。
                        if is_retool and "acc" in res:
                            health.observe(res["acc"].tolist(), res["fmt"].tolist(),
                                           res["clen"], res["cu"], res["trunc"],
                                           res.get("inv"))
                        # 题目级过滤：零方差组（全错/全对）当前无梯度，累计达标拉黑
                        if sched is not None:
                            sched.report(q, "uniform")
                        elif qkey is not None:
                            q_stat[qkey] = q_stat.get(qkey, 0) + 1
                        # 【2026-09-23 在线通过率监控】uniform 组也落盘（带题指纹与
                        # acc 数组——p≈0 题全错与 p≈1 题全对在 record 里可分）
                        if "acc" in res:
                            fout.write(json.dumps({
                                "t": time.time(), "algo": cfg["algo"],
                                "acc": res["acc"].tolist(), "fmt": res["fmt"].tolist(),
                                "clen": res["clen"], "code_used": res["cu"],
                                "code_ok": res["ck"],
                                "trunc_final": res["trunc"],
                                "code_wasted": res.get("cw", [0] * len(res["cu"])),
                                "invalid_final": res.get("inv", [0] * len(res["cu"])),
                                # 【2026-10-02】段长画像（旧键缺失由 analysis 补 None）
                                "segl": res.get("segl", []),
                                "tsegl": res.get("tsegl", []),
                                # 【2026-10-02】trunc_in_call（B 桶二分凭据）
                                "trunc_in_call": res.get("tric", []),
                                "qk": res.get("qk"), "q_status": "uniform",
                                "gen_version": policy_version[0],
                                "phase": "dropped"}, ensure_ascii=False) + "\n")
                    elif res["status"] == "overlong":
                        samp_stats["overlong"] += 1
                        # 【2026-09-12】超长也回填调度器（不再"与难度无关所以不管"）：
                        # 旧语义下超长既不拉黑也不插回，池子 refill 后又回到同一批题
                        # → 模型整体变长时采样主循环无限空转。见 QuestionScheduler.report。
                        if sched is not None:
                            sched.report(q, "overlong", count_overlong=_count_ov)
                    else:
                        if sched is not None:
                            sched.report(q, "ok")
                        elif qkey is not None:
                            q_stat[qkey] = 0   # 有梯度组：重置该题连败计数
                        groups.append(res)
                continue
            voutputs = vllm_gen.generate(prompts_text, sampling_params, use_tqdm=False)
            answers, ans_token_ids = [], []
            for v in voutputs:
                for z in v.outputs:
                    answers.append(z.text); ans_token_ids.append(list(z.token_ids))
            completion_lens = [len(t) for t in ans_token_ids]

            adv, acc_s, fmt_s = score_group(inputs, answers, completion_lens)
            if not group_ok(adv):
                continue  # 全同组：重采（对 rfpp 即原始 reward 全同；语义一致）

            groups.append((inputs, prompts_text, prompt_ids, ans_token_ids, adv, acc_s, fmt_s, plen))

        # 丢弃重采可见性：uniform=零方差组（全错/全对），overlong=轨迹超长
        # （retool 家族 uniform 丢弃已由题目级过滤大幅削减，剩余部分是模型学不动的
        #  "边缘题"——正常训练中随通过率上升自然回落）
        samp_stats["attempts"] += attempts
        for g in groups:
            if is_retool:
                r = g
                plen = r["plen"]
                meta = {"plen": plen, "algo": cfg["algo"], "has_mask": 1,
                        # 【2026-09-12】本组轨迹是哪一版权重生成的（train micro-step）。
                        # 训练端据此算出**真实** staleness，不再靠"16 步推送周期"猜。
                        "gen_version": policy_version[0]}
                # 【2026-09-21 overlong_filter】有截断样本时传 sample_weight
                # 让 compute_loss 排除它们（adv=0 但 KL 仍活跃 → 稀释 pg 梯度）
                _sw = r.get("sw")
                if _sw is not None:
                    meta["has_sw"] = 1
                    xdata = encode_batch(meta, r["merged"], r["adv"], r["gen_logps"],
                                         r["mask"], r["acc"], r["fmt"], _sw)
                else:
                    xdata = encode_batch(meta, r["merged"], r["adv"], r["gen_logps"],
                                         r["mask"], r["acc"], r["fmt"])
                requests.post(f"{ref_server}/upload", data=xdata)
                uploaded_total += 1
                fout.write(json.dumps({
                    "t": time.time(), "algo": cfg["algo"],
                    "acc": r["acc"].tolist(), "fmt": r["fmt"].tolist(),
                    "clen": r["clen"], "code_used": r["cu"], "code_ok": r["ck"],
                    "trunc_final": r["trunc"],
                    "code_wasted": r["cw"],
                    # 【2026-09-25 原生协议诊断】围栏档恒 0（键仍在，同表可逐列读）
                    "invalid_final": r["inv"], "ctx_full": r["ctxf"],
                    # 【2026-10-02】B 桶二分凭据：预算耗尽时是否卡在未闭合调用块
                    "trunc_in_call": r["tric"],
                    # 【2026-10-02】段长画像：segl=assistant 段长表 / tsegl=工具段长表
                    # （per-sample，与 clen/trunc 同下标；B 桶长度画像的输入）
                    "segl": r["segl"], "tsegl": r["tsegl"],
                    # 【2026-09-23 在线通过率监控】题目指纹 + 状态（uniform 组也落盘，
                    # 否则 p≈0 题在 record 里不可见，band 漂移观测有偏）
                    "qk": r.get("qk"), "q_status": "ok",
                    "gen_version": policy_version[0],
                    "phase": r["phase"]}, ensure_ascii=False) + "\n")
                health.observe(r["acc"].tolist(), r["fmt"].tolist(), r["clen"], r["cu"],
                               r["trunc"], r.get("inv"))
                continue
            inputs, prompts_text, prompt_ids, ans_token_ids, adv, acc_s, fmt_s, plen = g
            tensor_list = [torch.tensor(t) for t in ans_token_ids]
            output_ids = pad_sequence(tensor_list, batch_first=True,
                                      padding_value=tokenizer.pad_token_id)
            n = output_ids.shape[0]
            Qrep = prompt_ids.repeat(1, n).view(-1, plen)
            merged_ids = torch.cat([Qrep, output_ids], dim=1)
            gen_logps = compute_gen_logps(merged_ids, plen)

            meta = {"plen": plen, "algo": cfg["algo"]}
            xdata = encode_batch(meta, merged_ids, adv, gen_logps, acc_s, fmt_s)
            requests.post(f"{ref_server}/upload", data=xdata)
            uploaded_total += 1
            fout.write(json.dumps({
                "t": time.time(), "algo": cfg["algo"],
                "acc": acc_s.tolist(), "fmt": fmt_s.tolist(),
                "clen": [len(t) for t in ans_token_ids]}, ensure_ascii=False) + "\n")
            health.observe(acc_s.tolist(), fmt_s.tolist(),
                           [len(t) for t in ans_token_ids])
        # 【2026-09-12 反压①：外层轮次零产出熔断】内层循环到 max_attempts 仍没攒到
        # 一个可用组 = 这一整轮白跑。旧版外层是 `while True` 无任何计数，模型一旦
        # 整体变长就无限重来（真机：日志里 18274 行 "waiting for batch..."，训练端
        # 最后 5 小时零产出且零告警）。连续 N 轮零产出直接 fail-fast，把"无限空转"
        # 变成"当场崩掉并留下可排查的错"。
        if not groups:
            zero_yield += 1
            print(f"[rollout] 本轮零产出（{zero_yield}/{_zy_limit or '∞'}）"
                  f"：attempts={attempts} 全部被丢弃"
                  f"（uniform={samp_stats['uniform']} "
                  f"trajectory_overlong={samp_stats['overlong']} "
                  f"prompt_overlong={samp_stats['prompt_overlong']}）",
                  flush=True)
            if _zy_limit and zero_yield >= _zy_limit:
                raise RuntimeError(
                    f"[rollout] 连续 {zero_yield} 轮零产出 → 采样已死锁，fail-fast。\n"
                    f"  累计 attempts={samp_stats['attempts']} "
                    f"uniform={samp_stats['uniform']} "
                    f"trajectory_overlong={samp_stats['overlong']} "
                    f"prompt_overlong={samp_stats['prompt_overlong']}"
                    f" uploaded={uploaded_total}\n"
                    f"  最常见根因：预算不自洽/模型长度膨胀 → overlong 全丢"
                    f"（查 config.validate_retool_budget、record 的 clen/trunc_final 分布）；"
                    f"或题目池被拉黑到空（查 q_skip_streak / q_pool_reset_floor）。")
        else:
            zero_yield = 0
        if uploaded_total % 10 == 0:
            fout.flush()
        if uploaded_total and uploaded_total % 16 == 0:
            _a = samp_stats["attempts"]
            _u, _o = samp_stats["uniform"], samp_stats["overlong"]
            _po = samp_stats["prompt_overlong"]
            # 总丢弃以 attempts-uploaded 为唯一真值；原因项只负责归因。旧版用
            # uniform+overlong 当总数，native_p4 因 prompt 超限 20 次把 20.7% 低报成
            # 16.4%。other 始终打印，未来增加新丢弃分支时不会再静默少算。
            _discarded, _other = sampling_discard_counts(
                _a, uploaded_total, _u, _o, _po)
            # "被跳过"口径：队列路径 = 已拉黑题（streak 达标，未来不会再采）；
            # 旧路径 = 出现过 uniform 的题（旧语义保留）
            _skipped = (sched.blacklisted_count() if sched is not None
                        else sum(1 for v in q_stat.values() if v > 0))
            print(f"[rollout] 采样统计: 累计尝试 {_a} 次 / 有效上传 {uploaded_total} 组"
                  f"（真实丢弃率 {_discarded / max(1, _a) * 100:.0f}% = "
                  f"零方差 {_u} + 轨迹超长 {_o} + prompt超限 {_po} + 其他 {_other}；"
                  f"题目过滤中 {_skipped}/{len(QAs)} 题被跳过）"
                  # 【2026-10-02】轨迹落盘量只在本档开启时附加（关闭时该行逐字不变，
                  # 既有日志断言不受影响）；跳过行数非 0 = 落盘有损，必须可见。
                  + (f"；轨迹落盘 {n_traj_rows} 条"
                     + (f"（跳过 {n_traj_bad}）" if n_traj_bad else "")
                     if ftraj is not None else ""),
                  flush=True)
            # 【2026-09-12 反压②：窗口丢弃率告警/熔断】累计率会被开局的正常波动
            # 永久污染，所以按"上次打点以来的增量"算窗口率。总数同样必须取
            # attempts-uploaded，不能把已知原因相加冒充总数。
            _da = _a - _mark["a"]; _dd = _discarded - _mark["d"]
            _mark["a"], _mark["d"] = _a, _discarded
            if _da > 0:
                _rate = _dd / _da
                if _disc_alert and _rate >= _disc_alert and "discard" not in _disc_fired:
                    _disc_fired.add("discard")
                    print(f"\n[健康检查] 窗口真实丢弃率 {_rate * 100:.0f}% ≥ "
                          f"{_disc_alert * 100:.0f}%（{_dd}/{_da}）→ 采集端在大量白跑。"
                          f"判别：轨迹超长占多 = 预算/长度失控；prompt超限占多 = "
                          f"max_prompt_length/题面长度不匹配；零方差占多 = "
                          f"难度带过窄或温度过低。\n", flush=True)
                if _disc_abort and _rate >= _disc_abort:
                    raise RuntimeError(
                        f"[rollout] 窗口真实丢弃率 {_rate * 100:.0f}% ≥ 熔断线 "
                        f"{_disc_abort * 100:.0f}%（{_dd}/{_da}）→ 采集端已无有效产能，"
                        f"fail-fast 而非继续烧 GPU。累计：uniform={_u} "
                        f"trajectory_overlong={_o} prompt_overlong={_po} "
                        f"other={_other} uploaded={uploaded_total}。")
        # 训练期健康检查：窗口签名告警（fmt 恒定/没有学习/退化/截断/长度膨胀/代码缺失）
        # retool 家族的 clen 上限按"轮数×每轮预算"计——旧版用
        # max_context_tokens-max_prompt_length（8192-1024=7168），而轨迹实际上限
        # ≈3×1024+工具段，永远摸不到 0.95×上限，trunc 签名形同虚设
        # （2026-09-09 审查发现6；真正的截断签名另见 trunc_final/retool_trunc）
        # 【2026-09-29 token 预算档】上限换成 max_traj_tokens（那一档的真实轨迹
        # 预算）；继续用轮数乘积会把阈值放到一个该档下摸不到的位置 → 又变死开关。
        _health_cap = (int(cfg.get("max_traj_tokens", 0) or 0)
                       or (cfg["max_rounds"] * cfg.get("round_gen_tokens", 400)))
        health.maybe_check(
            retool=is_retool,
            max_clen=_health_cap if is_retool else cfg["max_gen_tokens"])


def main():
    import argparse
    ap = argparse.ArgumentParser(description="rlab 生成端独立运行（分进程模式）")
    ap.add_argument("--algo", required=True,
                    choices=("grpo", "dapo", "dr_grpo", "cispo", "gspo", "rfpp", "retool", "retool_math"))
    ap.add_argument("--gen_device", type=int, default=0)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--port", type=int, default=59875)
    ap.add_argument("--difficulty_path", default=None,
                    help="probe_difficulty.py 产出的通过率表（离线难度预过滤）")
    args = ap.parse_args()
    overrides = {}
    if args.model_path:
        overrides["model_path"] = args.model_path
    if args.difficulty_path:
        overrides["difficulty_path"] = args.difficulty_path
    cfg = get_config(args.algo, gen_device=args.gen_device, ref_server_port=args.port,
                     **overrides)
    gen_worker(None, cfg)


if __name__ == "__main__":
    main()
