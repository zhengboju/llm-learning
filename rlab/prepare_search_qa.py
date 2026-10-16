# -*- coding: utf-8 -*-
"""rlab/prepare_search_qa.py — 生成 Search-GRPO 的训练/评测池（docs/16 §4）。

产物目录 `rlab/datasets/search_qa/`：
  train.jsonl / dev.jsonl / manifest.json

两种来源：
  1. **离线导入**（pod 上 HF 不通时的主路径）：`--train_input x.jsonl --dev_input y.jsonl`
     （官方 split 各自一份文件，字段 question/answer 或 Q/A 都吃）；
  2. **在线拉取**：`--dataset nq_open|hotpotqa`（HF 优先、modelscope 兜底，复用
     rlab/data.py 的 verification_mode 垫片）。

【契约（落盘复读，不靠注释）】
  · 题面去重（按 Q；同题多答案时 aliases 合并，不丢监督信号）；
  · train ∩ dev = ∅（按题面），落盘后**复读文件**再核一次——2026-09-09 同池污染与
    2026-09-16 prepare 事故都是"声明与文件不符"，故 manifest 里记 SHA 与计数；
  · 已存在的输出目录**拒绝覆盖**（防误删旧产物）；
  · `--max_train` 固定 seed 裁剪，池子不足时告警并落 actual N。

【为什么 dev 不裁】held-out 越大越准（N≥300 是本项目硬协议）；裁剪只作用于训练池。
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from rlab.search_qa import normalize_answer, search_qa_dir, validate_search_rows

# 源字段映射（官方 split 的真实字段名；不把 evidence/context 注进 Q —— 那是泄漏）
_LOADERS = ("nq_open", "hotpotqa")


def _norm_source_row(src: str, row: dict):
    """把源数据的一条映射成规范行（Q/A/aliases），失败返回 None。

    · NQ (`nq_open`)：question / answer（answer 是 **list**，NQ 的公认多解）；
    · HotpotQA：question / answer（str，含 yes/no；distractor 档带 context，
      但**不注入 Q**——那等于把证据直接给模型，检索就被架空了）。
    """
    if src == "nq_open":
        q = row.get("question")
        a = row.get("answer")
        if isinstance(q, str) and q.strip() and isinstance(a, list) and a:
            al = [str(x) for x in a if str(x).strip()]
            if not al:
                return None
            # 首个答案进 A，其余进 aliases（两处都是金标准，打分取最大）
            return {"Q": q.strip(), "A": al[0], "aliases": al[1:], "source": src}
        return None
    if src == "hotpotqa":
        q = row.get("question")
        a = row.get("answer")
        if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
            return {"Q": q.strip(), "A": a.strip(), "aliases": [], "source": src}
        return None
    raise KeyError(f"未知数据源 {src!r}（可选 {_LOADERS}）")


def _load_online(src: str) -> tuple:
    """在线拉取 (train_rows, dev_rows)。HF 优先、modelscope 兜底。"""
    hf_ids = {"nq_open": "google-research-datasets/nq_open",
              "hotpotqa": "hotpot_qa"}
    ms_ids = {"nq_open": "modelscope/nq_open"}
    errs = []
    for backend in ("hf", "ms"):
        try:
            if backend == "hf":
                from datasets import load_dataset
                if src == "nq_open":
                    tr = load_dataset(hf_ids[src], split="train")
                    dv = load_dataset(hf_ids[src], split="validation")
                else:
                    tr = load_dataset(hf_ids[src], "distractor", split="train")
                    dv = load_dataset(hf_ids[src], "distractor", split="validation")
            else:
                from modelscope.msdatasets import MsDataset
                try:
                    from rlab.data import _patch_verification_mode
                    _patch_verification_mode()
                except Exception:
                    pass
                mid = ms_ids.get(src)
                if not mid:
                    raise RuntimeError(f"modelscope 未登记 {src} 的镜像 id")
                split_map = {"train": "train", "validation": "validation"}
                tr = MsDataset.load(mid, split=split_map["train"],
                                    trust_remote_code=True)
                dv = MsDataset.load(mid, split=split_map["validation"],
                                    trust_remote_code=True)
            tr_rows = [r for row in tr if (r := _norm_source_row(src, row))]
            dv_rows = [r for row in dv if (r := _norm_source_row(src, row))]
            if tr_rows and dv_rows:
                print(f"[prepare] 源={backend} {src}：train {len(tr_rows)} / "
                      f"dev {len(dv_rows)}", flush=True)
                return tr_rows, dv_rows
            errs.append(f"{backend}: 清洗后为空")
        except Exception as e:
            errs.append(f"{backend}: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"[prepare] {src} 在线拉取失败（HF 与 modelscope 都不通）：\n  "
        + "\n  ".join(errs)
        + "\n  离线方案：先在有网的机器上用官方脚本导出 train/dev 两份 JSONL，"
          "再 --train_input/--dev_input 导入（pod 的推荐路径）。")


def _dedup_by_question(rows: list) -> tuple:
    """按题面去重：同题多答案**合并进 aliases**（不丢监督信号）。返回 (rows, stats)。"""
    merged = {}
    for r in rows:
        q = r["Q"].strip()
        cur = merged.get(q)
        if cur is None:
            merged[q] = {"Q": q, "A": r["A"], "aliases": list(r.get("aliases") or []),
                         "source": r.get("source")}
            continue
        # 同题再出现：答案并入 aliases（A 保持首个）
        for cand in [r["A"]] + list(r.get("aliases") or []):
            if normalize_answer(cand) and cand not in cur["aliases"] \
                    and normalize_answer(cand) != normalize_answer(cur["A"]):
                cur["aliases"].append(cand)
    out = list(merged.values())
    return out, {"n_in": len(rows), "n_out": len(out),
                 "n_multi": sum(1 for r in out if r["aliases"])}


def _drop_unscoreable(rows: list) -> tuple:
    """丢弃"所有金标准在 SQuAD 归一化下都为空"的行，返回 (保留行, 丢弃数)。"""
    kept, dropped = [], 0
    for r in rows:
        cands = [r.get("A")] + list(r.get("aliases") or [])
        if any(normalize_answer(c) for c in cands if isinstance(c, str)):
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=_LOADERS, default=None,
                    help="在线数据源（HF 优先/modelscope 兜底）")
    ap.add_argument("--train_input", default=None, help="离线导入：训练池 JSONL")
    ap.add_argument("--dev_input", default=None, help="离线导入：held-out JSONL")
    ap.add_argument("--output_dir", default=None,
                    help=f"产物目录（默认 {search_qa_dir()}）")
    ap.add_argument("--max_train", type=int, default=0,
                    help="训练池上限（0=全用）；固定 seed 裁剪，池子不足会告警")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已存在的产物（默认拒绝，防误删旧池）")
    args = ap.parse_args()

    out_dir = Path(args.output_dir or search_qa_dir())
    train_p, dev_p = out_dir / "train.jsonl", out_dir / "dev.jsonl"
    if not args.force and (train_p.exists() or dev_p.exists()):
        raise SystemExit(
            f"[prepare] 输出已存在，拒绝覆盖（防误删旧池）：\n"
            f"  {train_p}\n  {dev_p}\n"
            f"  改用 --output_dir 换目录，或确认要重建时加 --force。")

    # ---- 取源 ----
    if args.train_input or args.dev_input:
        if not (args.train_input and args.dev_input):
            raise SystemExit("[prepare] 离线导入需要同时给 --train_input 与 --dev_input")
        from rlab.search_qa import _read_search_rows
        src_tag = f"offline {args.train_input} / {args.dev_input}"
        tr_rows = _read_search_rows(args.train_input)
        dv_rows = _read_search_rows(args.dev_input)
        for r in tr_rows + dv_rows:
            r.setdefault("source", "offline")
        print(f"[prepare] 源=offline：train {len(tr_rows)} / dev {len(dv_rows)}", flush=True)
    elif args.dataset:
        src_tag = f"online {args.dataset}"
        tr_rows, dv_rows = _load_online(args.dataset)
    else:
        raise SystemExit("[prepare] 需要 --dataset 或 --train_input/--dev_input")

    # ---- 清洗：先各自去重，再按题面剔除 dev 交叠 ----
    tr_rows, st = _dedup_by_question(tr_rows)
    dv_rows, dst = _dedup_by_question(dv_rows)
    print(f"[prepare] 去重：train {st['n_in']}->{st['n_out']}"
          f"（多解 {st['n_multi']}）｜dev {dst['n_in']}->{dst['n_out']}", flush=True)

    dev_qs = {r["Q"] for r in dv_rows}
    before = len(tr_rows)
    tr_rows = [r for r in tr_rows if r["Q"] not in dev_qs]
    if len(tr_rows) != before:
        print(f"[prepare] 训练池剔除与 dev 同题面 {before - len(tr_rows)} 条", flush=True)

    if args.max_train and len(tr_rows) > args.max_train:
        tr_rows = random.Random(args.seed).sample(tr_rows, args.max_train)
    elif args.max_train and len(tr_rows) < args.max_train:
        print(f"[prepare][警告] 训练池 {len(tr_rows)} 条 < --max_train {args.max_train}："
              f"按实际数量落盘（不重复补足——重复会改变题目分布）", flush=True)

    # 【丢弃"无对错可定义"的行】规范化后为空的答案（`a`/`the`/`-`/标点）在 SQuAD 口径
    # 下无法定义答对：留着会让打分 raise，或更糟——若判据放宽会变成"空==空"的假满分。
    # 清洗脚本在这里**丢弃并计数**（数据集的个别脏行不该让整次 prepare 失败）；
    # loader 侧仍硬拦（读到的文件应当已经干净，任何漏网都该炸）。
    tr_rows, n_drop_tr = _drop_unscoreable(tr_rows)
    dv_rows, n_drop_dv = _drop_unscoreable(dv_rows)
    if n_drop_tr or n_drop_dv:
        print(f"[prepare] 丢弃不可打分（答案规范化后为空）：train {n_drop_tr} / "
              f"dev {n_drop_dv}", flush=True)

    # ---- fail-fast 校验（含"答案规范化后为空"）----
    validate_search_rows(tr_rows, "prepare train")
    validate_search_rows(dv_rows, "prepare dev")
    if not tr_rows or not dv_rows:
        raise SystemExit(f"[prepare] 清洗后池子为空：train={len(tr_rows)}, dev={len(dv_rows)}")
    if len(dv_rows) < 300:
        print(f"[prepare][警告] held-out 只有 {len(dv_rows)} 题 < 300（本项目评测协议下限）"
              f"：±1~2pp 的效应读不出来", flush=True)

    # ---- 落盘 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_p, tr_rows)
    _write_jsonl(dev_p, dv_rows)

    # ---- 落盘复读自证（契约落在文件上，不靠注释）----
    from rlab.search_qa import _read_search_rows
    back_tr = _read_search_rows(str(train_p))
    back_dv = _read_search_rows(str(dev_p))
    back_tr_q = {r["Q"] for r in back_tr}
    back_dv_q = {r["Q"] for r in back_dv}
    inter = back_tr_q & back_dv_q
    if inter:
        raise RuntimeError(
            f"[prepare] 落盘复读发现 train/dev 题面仍相交 {len(inter)} 题"
            f"（例：{sorted(inter)[:2]}）——文件与内存不一致，拒绝放行")
    if len(back_tr) != len(tr_rows) or len(back_dv) != len(dv_rows):
        raise RuntimeError(
            f"[prepare] 落盘复读行数不符：train {len(back_tr)} vs {len(tr_rows)}，"
            f"dev {len(back_dv)} vs {len(dv_rows)}")

    manifest = {
        "source": src_tag, "seed": args.seed, "max_train": args.max_train,
        "train_n": len(tr_rows), "dev_n": len(dv_rows),
        "train_unique_q": len(back_tr_q), "dev_unique_q": len(back_dv_q),
        "train_sha256": _sha256(train_p), "dev_sha256": _sha256(dev_p),
        "holdout_ok": True, "dedup_train": st, "dedup_dev": dst,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] train {len(tr_rows)} / dev {len(dv_rows)} → {out_dir}"
          f"（落盘复读：题面不相交 ✅，行数一致 ✅）", flush=True)
    print("[prepare] 下一步：先建索引与起检索服务（python -m rlab.search_server "
          "build-index corpus.jsonl index.sqlite；serve index.sqlite --port 8000），"
          "再 python -m rlab.train --algo search_qa --search_endpoint "
          "http://<host>:8000 ...", flush=True)


if __name__ == "__main__":
    main()
