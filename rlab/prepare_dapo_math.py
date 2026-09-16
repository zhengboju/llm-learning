# -*- coding: utf-8 -*-
"""下载并整理 DAPO-Math-17k（对齐 agentic-rl-lab/05-retool/prepare_data.py）。

【2026-09-16 事故与修复】数据源（HF 侧）**每题被复制 100~400 份**：1,791,200 行 /
~16.7k 唯一题面（HF 与 modelscope 同名 id 的行数并不一致）。旧版 `random.shuffle`
后按**位置**切 `dev=records[:500] / train=records[500:]`——重复副本让"前 500 条"的
题面几乎必然也躺在后面，实测 train.jsonl 里含 **52,700 条 dev 题**（500 题 × ~105 份），
即 **held-out 100% 污染**：dev 每一题都被完整训过，所有 dev 评测数字虚高。这是
2026-09-09「同池污染」以更隐蔽的形式复发，而文件里那句"已剔除 dev"当时是假的。

现在：①按 (Q, A) 去重；②以**题面**为单位切（同题多解整体归一侧）；③**落盘后复读**
断言两侧题面不相交（契约落在文件上，不靠注释）。训练端另有独立核实
（`data.verify_train_pool_clean`），两道互不依赖。
"""
import argparse
import json
import random
from pathlib import Path

DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def strip_dapo_template(q: str) -> str:
    if q.startswith(PROMPT_PREFIX):
        q = q[len(PROMPT_PREFIX):]
    if q.endswith(PROMPT_SUFFIX):
        q = q[:-len(PROMPT_SUFFIX)]
    return q.strip()


def normalize_row(idx, row):
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return None
    q = strip_dapo_template(str(prompt[0].get("content") or ""))
    rm = row.get("reward_model") or {}
    gt = rm.get("ground_truth") if isinstance(rm, dict) else None
    if isinstance(gt, list):
        gt = gt[0] if gt else None
    a = str(gt or "").strip()
    if not q or not a:
        return None
    return {"id": str(row.get("extra_info", {}).get("index") or idx), "question": q,
            "answer": a, "data_source": str(row.get("data_source") or "dapo_math")}


def split_train_dev(records: list, dev_size: int, seed: int) -> tuple:
    """纯函数：按**题面**切 held-out，返回 `(train, dev)`。

    【为什么不能按位置切】源数据每题有 100~400 份重复副本时，"前 500 条"里的题面
    几乎必然也出现在第 500 条之后 → 按位置切必然泄漏（真机实测 52,700 条）。
    切分单位必须是**题面**，不是行。

    同题多解（n_conflict_q>0）整体归一侧，避免"同一个问题的答案一半在 dev 一半在
    train"这种更隐蔽的污染。dev 抽满 dev_size 即停（多解题面可能让 dev 略多于目标，
    真实行数由调用方打印）。
    """
    groups = {}
    for r in records:
        groups.setdefault(r["Q"], []).append(r)
    qs = list(groups)
    random.Random(seed).shuffle(qs)
    dev, train = [], []
    for q in qs:
        (dev if len(dev) < dev_size else train).extend(groups[q])
    return train, dev


def _load_records() -> list:
    """HF 优先、modelscope 兜底（与 data.py 的默认顺序相反：本脚本是一次性产物生成，
    两边的唯一题面集合相同——行数差异由调用方的去重报告暴露）。"""
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_ID, split="train")
        src = f"HF {DATASET_ID}"
    except Exception as e:
        from modelscope.msdatasets import MsDataset
        try:
            from rlab.data import _patch_verification_mode
            _patch_verification_mode()
        except Exception:
            pass
        ds = list(MsDataset.load(DATASET_ID, split="train", trust_remote_code=True))
        src = f"modelscope {DATASET_ID}（HF 失败: {type(e).__name__}）"
    records = [r for idx, row in enumerate(ds)
               if (r := normalize_row(idx, row)) is not None]
    print(f"[prepare] 源={src}｜原始行 {len(ds)} -> 可用记录 {len(records)}", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "datasets" / "dapo_math")
    # dev=500（2026-09-09 审查修复）：原 50 题的二项噪声在 acc≈0.7 处约 ±6.5pp，
    # 比项目 ±2pp 评测噪声地板大 3 倍，支撑不了任何结论；500 题约 ±2pp。
    ap.add_argument("--dev-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from rlab.data import _read_qa_jsonl, dedup_questions, pool_report_line

    records = _load_records()
    if not records:
        raise RuntimeError("[prepare] 源数据清洗后为空")
    records, st = dedup_questions(records)          # ① (Q,A) 去重（无损）
    print(pool_report_line(st, "prepare 源数据"), flush=True)
    train, dev = split_train_dev(records, args.dev_size, args.seed)   # ② 按题面切

    dev_qs, train_qs = {r["Q"] for r in dev}, {r["Q"] for r in train}
    if dev_qs & train_qs:
        raise RuntimeError(f"[prepare] 切分后 train/dev 题面仍相交 {len(dev_qs & train_qs)} 题"
                           "——拒绝产出污染文件")

    for p, lst in ((args.output_dir / "train.jsonl", train),
                   (args.output_dir / "dev.jsonl", dev)):
        with p.open("w", encoding="utf-8") as f:
            for r in lst:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ③ 落盘自证：训练端读的是**文件**，契约必须落在文件上（旧版正是这里缺一道检查）
    back_tr = {r["Q"] for r in _read_qa_jsonl(args.output_dir / "train.jsonl")}
    back_dv = {r["Q"] for r in _read_qa_jsonl(args.output_dir / "dev.jsonl")}
    if back_tr & back_dv:
        raise RuntimeError(f"[prepare] 落盘复读发现相交 {len(back_tr & back_dv)} 题"
                           "——文件与内存不一致，拒绝放行")
    print(f"[prepare] train {len(train)} / dev {len(dev)}"
          f"｜唯一题面 {len(train_qs | dev_qs)}（落盘复读不相交 ✅）", flush=True)
    print(f"[prepare] 下一步：bash rlab/run_gsm8k.sh retool_math <model> ..."
          f"（训练端启动行会再核一次 dev 契约并打印池子去重形态）", flush=True)


if __name__ == "__main__":
    main()
