# -*- coding: utf-8 -*-
"""rlab/search_qa.py — Search-GRPO 的 QA 打分与数据加载（docs/16）。

两块内容，都是 CPU 纯函数/文件 IO，不依赖 torch/vLLM：

1. **打分**（SQuAD/NQ 口径）：normalize + EM/token-F1（多别名取最大），
   最终答案从 `\\boxed{}` 取；reward 是 **outcome-only ±1**（与 math 同域），
   F1 只作监控（record 列 / 报告用），不进 reward。
2. **数据**：`rlab/datasets/search_qa/{train,dev}.jsonl` 的加载，行 schema
   `{"Q","A","aliases":[...],"source","id"}`。

【为什么打分域要先剥工具调用块】与 retool_math 的 `strip_code_blocks` 同一条原则：
调用块是脚手架——`{"query": "...\\boxed{7}"}` 这类载荷若留在打分文本里，"最后一个
boxed"就会被判成模型的答案（docs/09 §10.2 第 9 条的同族事故）。标签一律从
`rlab.protocol` 的正则派生（AGENTS.md 标签字节铁律：绝不手写标签字面量）。
"""
import json
import os
import re
import unicodedata

# ---------------------------------------------------------------- 打分 ----

# SQuAD/NQ 标准归一化里的"无信息词"（冠词）与标点
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """SQuAD/NQ 口径归一化：NFKC → 小写 → 去标点 → 去冠词 → 折叠空白。

    NFKC 是必要的：语料里混着全角/半角与连字（`Ⅳ`/`IV`、`ﬁ`/`fi`），不归一化
    会让"字面相同的答案"判成不等——那是评测噪声，不是能力差异。
    """
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = "".join(ch for ch in t if ch not in _PUNCT)
    t = _ARTICLES_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _gold_list(golds) -> list:
    """golds 归一：str / [str] / QA row dict（A + aliases）都吃。"""
    if isinstance(golds, dict):
        vals = []
        a = golds.get("A") or golds.get("answer")
        if isinstance(a, (str, list)):
            vals.extend(a if isinstance(a, list) else [a])
        al = golds.get("aliases")
        if isinstance(al, list):
            vals.extend(al)
        return [str(v) for v in vals if isinstance(v, (str, int, float))]
    if isinstance(golds, (list, tuple, set)):
        return [str(v) for v in golds]
    if golds is None:
        return []
    return [str(golds)]


def _tokens(text: str) -> list:
    n = normalize_answer(text)
    return n.split() if n else []


def qa_metrics(prediction: str, golds) -> tuple:
    """返回 (em, f1)，多别名取最大（SQuAD 官方口径）。纯函数。

    【为什么金标准全空要 raise】空 gold 无法定义"答对"：放过它会让
    `normalize("") == normalize("")` 把"模型什么都没答"判成满分（假分数），
    而静默返回 0 又会让"数据坏了"伪装成"模型不行"。数据侧（loader）已强制
    校验，这里是第二道闸——两道都在，任何一道被绕过都还会炸。
    """
    norm_golds = [g for g in (normalize_answer(g) for g in _gold_list(golds)) if g]
    if not norm_golds:
        raise ValueError(
            "[search_qa] 金标准为空（规范化后）：无法定义答对——"
            "数据侧 loader 应已拦下，请检查 prepare_search_qa 的产物")
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0, 0.0
    best_em, best_f1 = 0.0, 0.0
    pt = pred.split()
    for gold in norm_golds:
        if pred == gold:
            best_em = 1.0
        gt = gold.split()
        common = 0
        if pt and gt:
            # 多集合交集计数（词频敏感，SQuAD 同款）
            from collections import Counter
            common = sum((Counter(pt) & Counter(gt)).values())
        if common:
            precision = common / len(pt)
            recall = common / len(gt)
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return best_em, best_f1


def _final_answer_text(assistant_text: str) -> str:
    """打分域 = 剥掉工具调用脚手架后的模型回答（调用载荷里的 boxed 不得当答案）。

    · **完整**调用块整块剥掉（与 strip_code_blocks 同一正则，同源）；
    · **未闭合**的调用块（模型被轮长/预算切断）从开标记起截去——残缺调用里的
      boxed 同样是脚手架，不是答案；这与 `parse_assistant` 判 invalid 同向。
    """
    from rlab.protocol import _RE_TOOL_CALL_ANY, _TOOL_OPEN
    text = assistant_text or ""
    text = _RE_TOOL_CALL_ANY.sub("", text)
    cut = text.find(_TOOL_OPEN)
    if cut >= 0:
        text = text[:cut]
    return text


def extract_final_answer(assistant_text: str) -> str:
    """取最终答案：脚手架剥离后最后一个 `\\boxed{}`（花括号配平）。无则空串。

    与 `rlab.reward.extract_last_boxed` 同一算法，但**自持一份**：reward.py 会连带
    引入 math_verify 等重依赖，而本模块要求纯轻量（CPU 测试与数据侧都能直接用）。"""
    text = _final_answer_text(assistant_text)
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    start = idx + len(marker)
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos].strip()
    return ""


def evaluate_search_answer(golds, assistant_text: str) -> dict:
    """一条轨迹的 QA 评测：返回 {"em","f1","format","prediction"}（0/1 与 0..1 原值）。

    format = 是否给出了最终 `\\boxed{}`（1/0）——**分母保留**：没交卷也算 0 分，
    与项目"空答案计 0 分入分母"的既有契约一致（否则交卷率会伪装成精度提升）。"""
    prediction = extract_final_answer(assistant_text)
    em, f1 = qa_metrics(prediction, golds)
    return {"em": em, "f1": f1, "format": 1.0 if prediction else 0.0,
            "prediction": prediction}


def total_reward_search_qa(golds, assistant_text: str, *,
                           completion_len: int = 0, max_gen_tokens: int = 8192,
                           overlong_buffer: int = 64, overlong_shaping: bool = False,
                           trunc_final: int = 0, trunc_shaping: float = 0.0) -> dict:
    """Search-GRPO 的 outcome-only 奖励（±1 域，与 math 同口径）。

    reward = 2·EM − 1：答对 +1 / 答错 −1——**正确性是唯一奖励**（R0 档，docs/16 §3）。
    检索次数/成功率不进 reward：先看"会不会检索"能否靠结果涌现（ReTool 的教训：
    一上来加 shaping 就把 outcome 主信号淹了，且答案对错才是我们要的东西）。

    f1 只作监控（0..1 原值，落 record 供分析），不参与 reward。
    长度项与 retool_math 同档：overlong 起坡扣分 + 末段截断扣分（权重默认 0）。
    """
    m = evaluate_search_answer(golds, assistant_text)
    reward = 2.0 * m["em"] - 1.0
    ov = 0.0
    if overlong_shaping:
        from rlab.reward import overlong_penalty
        ov = overlong_penalty(completion_len, max_gen_tokens, overlong_buffer)
        reward -= ov
    tp = 0.0
    if trunc_shaping:
        from rlab.reward import trunc_penalty
        tp = trunc_penalty(trunc_final, trunc_shaping)
        reward -= tp
    return {"reward": reward, "acc": 2.0 * m["em"] - 1.0,
            "format": 2.0 * m["format"] - 1.0, "f1": m["f1"],
            "em": m["em"], "prediction": m["prediction"],
            "overlong": ov, "trunc_penalty": tp}


# ---------------------------------------------------------------- 数据 ----

_SEARCH_QA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "datasets", "search_qa")

_PREPARE_HINT = ("  先运行: python -m rlab.prepare_search_qa --dataset nq_open\n"
                 "  或离线导入: python -m rlab.prepare_search_qa "
                 "--train_input <train.jsonl> --dev_input <dev.jsonl>")


def search_qa_dir() -> str:
    return _SEARCH_QA_DIR


def _read_search_rows(path: str) -> list:
    """读 search_qa 产物（保留 aliases；键名容错 question/answer）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise RuntimeError(f"[search_qa] {path} 第 {n} 行不是合法 JSON：{e}")
            q = r.get("question") or r.get("Q")
            a = r.get("answer") or r.get("A")
            al = r.get("aliases")
            if isinstance(a, list):        # 兼容"答案直接是列表"的源格式
                al = list(a) + (al if isinstance(al, list) else [])
                a = a[0] if a else None
            row = {"Q": str(q or ""), "A": str(a or ""),
                   "aliases": [str(x) for x in al] if isinstance(al, list) else []}
            if r.get("source") is not None:
                row["source"] = str(r["source"])
            if r.get("id") is not None:
                row["id"] = str(r["id"])
            rows.append(row)
    return rows


def validate_search_rows(rows: list, where: str = "") -> None:
    """fail-fast：Q 非空、A 或 aliases 至少一个**规范化后非空**。

    【为什么规范化后判】`A="the"`/`A="-"` 这类在 SQuAD 归一化下会变成空串——
    它们在打分时无法定义对错（qa_metrics 会 raise）。在入口（数据侧）就拦下，
    错误才留在能排查的地方，而不是让生成端每条轨迹都炸在打分里。
    """
    for i, r in enumerate(rows):
        if not isinstance(r.get("Q"), str) or not r["Q"].strip():
            raise RuntimeError(f"[search_qa] {where} 第 {i} 行 Q 缺失/非字符串：{r!r}")
        cands = [r.get("A")] + list(r.get("aliases") or [])
        if not any(normalize_answer(c) for c in cands if isinstance(c, str)):
            raise RuntimeError(
                f"[search_qa] {where} 第 {i} 行答案在归一化后为空（A={r.get('A')!r}，"
                f"aliases={r.get('aliases')!r}）——无法定义答对，拒绝入池。\n"
                f"  规范：A 是非空字符串，aliases 是别名列表（可为空表）。")


def aliases_of(row: dict) -> list:
    """一条题的所有金标准（A + aliases，去重保序）——训练端打分与离线评测共用。"""
    out, seen = [], set()
    for c in [row.get("A")] + list(row.get("aliases") or []):
        if isinstance(c, str) and c.strip() and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _load_one(path: str, where: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[search_qa] {where} 缺失: {path}\n{_PREPARE_HINT}")
    rows = _read_search_rows(path)
    if not rows:
        raise RuntimeError(f"[search_qa] {where} 存在但为空: {path}\n{_PREPARE_HINT}")
    validate_search_rows(rows, where)
    for r in rows:
        r["aliases"] = aliases_of(r)
    return rows


def load_search_qa_dev(path: str | None = None) -> list:
    """held-out 评测池 = dev.jsonl（prepare_search_qa 产物，绝不回落训练池）。"""
    p = path or os.path.join(_SEARCH_QA_DIR, "dev.jsonl")
    rows = _load_one(p, "search_qa dev.jsonl (held-out)")
    print(f"[search_qa] dev.jsonl (held-out): {len(rows)} 题")
    return rows


def load_search_qa_train(path: str | None = None, dev_path: str | None = None) -> list:
    """训练池 = train.jsonl（prepare 产物，已剔除 dev）。

    【held-out 契约必须落盘复读】"已剔除 dev"是**声明**，不是证据（2026-09-09 同池
    污染 + 2026-09-16 prepare 事故的教训）。这里读**文件**核对：
      · dev.jsonl 缺失 → 拒跑（没有 held-out 就无法回答"有没有学到"）；
      · 题面与 dev 相交 → 当场剔除并大声告警（不静默缩水，也不放行）。
    训练端与 prepare 端两道互不依赖的检查，任何一道被绕过都还会炸。
    """
    p = path or os.path.join(_SEARCH_QA_DIR, "train.jsonl")
    dp = dev_path or os.path.join(_SEARCH_QA_DIR, "dev.jsonl")
    rows = _load_one(p, "search_qa train.jsonl")
    if not os.path.exists(dp):
        raise FileNotFoundError(
            f"[search_qa] held-out dev.jsonl 缺失: {dp}\n"
            f"  没有 held-out 就无法判断训练是否真的提升了泛化（docs/16 §4）。\n"
            f"{_PREPARE_HINT}")
    dev_rows = _load_one(dp, "search_qa dev.jsonl")
    dev_qs = {r["Q"].strip() for r in dev_rows}
    kept = [r for r in rows if r["Q"].strip() not in dev_qs]
    if len(kept) != len(rows):
        print(f"[search_qa][警告] train.jsonl 含 {len(rows) - len(kept)} 条 held-out 题"
              f"——该文件不是 prepare_search_qa 的干净产物（或被别的流程覆盖）。"
              f"已当场剔除 {len(rows)} -> {len(kept)}；建议重跑 prepare。", flush=True)
    # (Q,A) 去重：同一题面带多个答案时**整体保留**（多解是监督信号，按 Q 去重会丢）
    seen, out = set(), []
    for r in kept:
        key = (r["Q"], tuple(r["aliases"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if len(out) != len(kept):
        print(f"[search_qa] 训练池按 (Q,aliases) 去重: {len(kept)} -> {len(out)} 条")
    print(f"[search_qa] 训练池: {len(out)} 题（已核 dev 不相交；held-out {len(dev_rows)} 题）",
          flush=True)
    return out
