# -*- coding: utf-8 -*-
"""下载并整理 DAPO-Math-17k（对齐 agentic-rl-lab/05-retool/prepare_data.py）。"""
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
    return {"id": str(row.get("extra_info", {}).get("index") or idx), "question": q, "answer": a, "data_source": str(row.get("data_source") or "dapo_math")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "datasets" / "dapo_math")
    # dev=500（2026-09-09 审查修复）：原 50 题的二项噪声在 acc≈0.7 处约 ±6.5pp，
    # 比项目 ±2pp 评测噪声地板大 3 倍，支撑不了任何结论；500 题约 ±2pp。
    ap.add_argument("--dev-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_ID, split="train")
    except Exception as e:
        # fallback modelscope
        from modelscope.msdatasets import MsDataset
        # 尝试 patch
        try:
            from rlab.data import _patch_verification_mode
            _patch_verification_mode()
        except Exception:
            pass
        ds = MsDataset.load("BytedTsinghua-SIA/DAPO-Math-17k", split="train", trust_remote_code=True)
        # MsDataset -> list-like
        ds = list(ds)
        # 统一为 datasets 风格迭代
        class _Wrap:
            def __iter__(self_inner): return iter(ds)
            def __len__(self_inner): return len(ds)
        ds = _Wrap()
        # 重写循环逻辑
        records = []
        for idx, row in enumerate(ds):
            # MsDataset row 可能是 dict 且 prompt 为 list
            rec = normalize_row(idx, row)
            if rec: records.append(rec)
        random.Random(args.seed).shuffle(records)
        train = records[args.dev_size:]; dev = records[:args.dev_size]
        for p, lst in [(args.output_dir/"train.jsonl", train), (args.output_dir/"dev.jsonl", dev)]:
            with p.open("w", encoding="utf-8") as f:
                for r in lst: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        print(f"train {len(train)} dev {len(dev)}")
        return
    records = [r for idx, row in enumerate(ds) if (r:=normalize_row(idx, row)) is not None]
    random.Random(args.seed).shuffle(records)
    train = records[args.dev_size:]; dev = records[:args.dev_size]
    for p, lst in [(args.output_dir/"train.jsonl", train), (args.output_dir/"dev.jsonl", dev)]:
        with p.open("w", encoding="utf-8") as f:
            for r in lst: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"train {len(train)} dev {len(dev)}")

if __name__ == "__main__":
    main()
