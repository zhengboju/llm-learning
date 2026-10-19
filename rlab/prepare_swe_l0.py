# -*- coding: utf-8 -*-
"""rlab/prepare_swe_l0.py — 生成 SWE-RL L0 的 instance 池（docs/17 §2/§3）。

产物目录 `rlab/datasets/swe_l0/`：
  train.jsonl / dev.jsonl / manifest.json

两个来源
--------
1. **种子语料**（`--seeds`，默认）：`rlab/swe_l0_problems.py` 里手写的
   `(正确实现, 注入 bug, 单测)` 三元组。注入的 bug 是"人类真会犯的错"，
   `issue` 描述也是自然语言写的——它们的价值是**语义干净**。
2. **合成变异**（`--synth`）：`rlab/swe_l0_synth.py` 对每个种子的正确实现
   自动做单点语义变异，再由**执行**筛选。收益是池子可放大（L0 只有 8 个种子，
   而 RL 需要题目分布足够宽）；代价是变异可读性不如手写。

**两条路径共用同一道验收**（这是本脚本存在的理由）：
  · 原始码 + 全部测试 必须 **全绿**（种子自证：任务可解、P2P 有来源）
  · buggy 码 + 全部测试 必须 **至少一条红**（否则这个 instance 无法定义"修好"）
  · F2P = 在 buggy 上红、在原始码上绿的那些用例（**实测**，绝不手写）
  · P2P = 两边都绿的那些用例
  · F2P 为空 / P2P 为空 / 测试跑不起来 → 剔除并计数落盘

【为什么"P2P 为空"也算不合格】P2P 是"修 bug 不能拆台"的唯一约束。若某个
instance 的所有测试都在 F2P 里（bug 让整个模块都跑不动），那它测的其实是
"能不能让代码可导入"，与 L0 想教的东西（定位+语义修复）不是一回事，且会让
条件修复率的分子分母同时退化。这类实例剔除后如实记录在 manifest 里。

【held-out 按题分，不按行】dev 切分以 **instance id** 为单位（种子名/变异 id），
不从同一种子派生的多个 instance 里各取一部分——那等于把同一处代码的多个
变体同时放进 train 与 dev，held-out 就失效了（同池污染的经典形态）。

【落盘复读自证】写完立刻**重新读文件**核对：行数一致、train∩dev 的 id 与 Q
都不相交、f2p 仍然非空。声明不算证据（2026-09-09 / 09-16 两次事故的教训）。

零标签字面量铁律：本模块只拼 python 源码与纯文本提示，不含格式标签字节。
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

from rlab import swe_l0 as S
from rlab.swe_l0_problems import SEEDS
from rlab.swe_l0_synth import candidate_mutations, family_counts

_MIN_BASELINE_TESTS = 2       # P2P 至少 1 条 → 全部测试至少 2 条，否则剔除


# =====================================================================
# 提示构造（L0 = issue + **已定位的局部代码** + 测试，docs/17 §2.5）
# =====================================================================
_PROMPT_HEAD = (
    "You are fixing a bug in a small Python project.\n"
    "You are given the bug report, the current content of the file that must be "
    "fixed, and the test file that defines correct behaviour. The bug is inside "
    "the file shown below; you do not need to look anywhere else.\n\n"
    "Bug report:\n{issue}\n\n"
    "File `{target}` (current, contains the bug):\n"
    "```python\n{code}\n```\n\n"
    "Test file `{testname}` (do not modify it):\n"
    "```python\n{tests}\n```\n\n"
    "Fix the file so that every test passes, without changing behaviour that "
    "already works and without editing the tests. Output only a unified diff "
    "(`git apply -p1` compatible) inside a single `diff` fenced code block.\n"
    "The diff must modify `{target}` only. Do not output any other text."
)


def build_prompt(issue: str, target: str, code: str, testname: str,
                 tests: str) -> str:
    return _PROMPT_HEAD.format(issue=issue.strip(), target=target,
                               code=code.rstrip(), testname=testname,
                               tests=tests.rstrip())


# =====================================================================
# 验收：把 (源码, 测试) 拆成 F2P/P2P
# =====================================================================
def derive_spec(target: str, testname: str, tests: str, source: str,
                buggy: str, *, timeout: float = 60.0,
                mem_mb: int = 2048) -> dict:
    """实测 F2P/P2P。返回 {"ok", "reason", "f2p", "p2p", "results_*"}。

    【为什么测两遍（source 与 buggy）】F2P 的定义是"buggy 上红、source 上绿"，
    只看 buggy 的红色名单会把"本来就坏的测试"也算成 F2P——那种 instance 无论
    模型怎么改都过不了，reward 恒 -1，是纯粹的噪声来源。
    """
    out = {"ok": False, "reason": "", "f2p": [], "p2p": [],
           "results_source": {}, "results_buggy": {}}
    tfile = f"{testname}.py"

    def _run(code: str) -> dict:
        row = {"files": {f"{target}.py": code}, "tests": {tfile: tests}}
        wd = S.make_workdir_factory(row)()
        try:
            return S.run_tests(wd, [tfile], timeout=timeout, mem_mb=mem_mb)
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    src = _run(source)
    bug = _run(buggy)
    out["results_source"] = src["results"]
    out["results_buggy"] = bug["results"]
    if not src["results"]:
        out["reason"] = "原始码测试未产出用例结果（收集期失败/超时）"
        return out
    if bug["timed_out"]:
        # buggy 侧超时（如朴素递归爆掉）：**不剔除**，但记下来。
        # 它是合法的"性能退化"类 bug，只是无法得到 buggy 的完整用例表——
        # F2P 退化为"所有在 source 上绿的用例"（不能在 buggy 上判红的，
        # 只能假设"本来红"）。这种口径必须在 manifest 里可见，不能静默。
        out["reason"] = "buggy 侧超时（F2P 按 source 全绿口径给出）"
    bad_src = [k for k, v in src["results"].items() if v != "passed"]
    if bad_src:
        out["reason"] = (f"原始码上有 {len(bad_src)} 条测试不绿"
                         f"（例：{bad_src[:2]}）——种子本身不自洽")
        return out
    if bug["timed_out"]:
        f2p = [k for k, v in src["results"].items() if v == "passed"]
        p2p = []
    else:
        f2p = [k for k, v in bug["results"].items() if v != "passed"
               and src["results"].get(k) == "passed"]
        p2p = [k for k, v in bug["results"].items() if v == "passed"
               and src["results"].get(k) == "passed"]
    if not f2p:
        out["reason"] = "buggy 码没有让任何测试变红——该 bug 未被测试覆盖"
        return out
    if not p2p and not bug["timed_out"]:
        out["reason"] = ("P2P 为空：bug 让整个模块不可用，"
                         "该 instance 测的其实是'让代码可导入'")
        return out
    if len(src["results"]) < _MIN_BASELINE_TESTS:
        out["reason"] = f"测试数 {len(src['results'])} < {_MIN_BASELINE_TESTS}"
        return out
    out.update(ok=True, f2p=sorted(f2p), p2p=sorted(p2p))
    return out


def make_instance(*, iid: str, tag: str, origin: str, target: str,
                  testname: str, source: str, buggy: str, tests: str,
                  issue: str) -> tuple:
    """验收并组装一行数据。返回 (row_or_None, reject_reason)。"""
    spec = derive_spec(target, testname, tests, source, buggy)
    if not spec["ok"]:
        return None, spec["reason"]
    tfile = f"{testname}.py"
    row = {
        "id": iid,
        "Q": build_prompt(issue, f"{target}.py", buggy, tfile, tests),
        "A": "every fail-to-pass test passes and no pass-to-pass test regresses",
        # 仓库快照（**buggy** 起始态）+ 测试；路径一律相对 POSIX
        "files": {f"{target}.py": buggy},
        "tests": {tfile: tests},
        "test_files": [tfile],
        "f2p": spec["f2p"],
        "p2p": spec["p2p"],
        # 参考修复（只作诊断/上限基线，**不进**训练信号：模型的 patch 不等于它
        # 也能拿满分——测试是唯一判据，这是 L0 的口径）
        "source": {f"{target}.py": source},
        "origin": origin,
        "tag": tag,
        "issue": issue,
        "n_tests": len(spec["results_source"]),
    }
    return row, ""


# =====================================================================
# 池子构造
# =====================================================================
def build_seed_rows(timeout: float = 60.0) -> tuple:
    rows, rejects = [], []
    for seed in SEEDS:
        row, why = make_instance(
            iid=seed.name, tag=seed.tag, origin="seed", target=seed.target,
            testname=seed.testname, source=seed.source, buggy=seed.buggy,
            tests=seed.tests, issue=seed.issue)
        if row is None:
            rejects.append({"id": seed.name, "origin": "seed", "reason": why})
        else:
            rows.append(row)
    return rows, rejects


def build_synth_rows(timeout: float = 60.0, per_seed: int = 0) -> tuple:
    """对每个种子的**正确实现**做单点变异，逐条执行验收。

    `per_seed` > 0 时每个种子最多保留这么多条（按变异族轮转取样，保证族多样性
    ——全取 `int_offby1` 会让池子退化成一类 bug）。
    """
    rows, rejects = [], []
    for seed in SEEDS:
        cands = candidate_mutations(seed.source)
        # 按族轮转排序：族多样性优先于"原文顺序"
        by_fam = {}
        for site, mutated in cands:
            by_fam.setdefault(site.family, []).append((site, mutated))
        ordered = []
        while any(by_fam.values()):
            for fam in sorted(by_fam):
                if by_fam[fam]:
                    ordered.append(by_fam[fam].pop(0))
        kept_this = 0
        for site, mutated in ordered:
            if per_seed and kept_this >= per_seed:
                break
            iid = f"{seed.name}__{site.family}_{site.start}_{site.end}"
            issue = (
                f"The function in `{seed.target}.py` behaves incorrectly for some "
                f"inputs. A previously working version was changed and the tests "
                f"now fail. Restore the correct behaviour.\n"
                f"Hint about the kind of defect: a {site.family.replace('_', ' ')} "
                f"in the implementation.\n"
                f"(Original requirement: {seed.issue.strip().splitlines()[0]})"
            )
            row, why = make_instance(
                iid=iid, tag=f"{seed.tag}/{site.family}", origin="synth",
                target=seed.target, testname=seed.testname, source=seed.source,
                buggy=mutated, tests=seed.tests, issue=issue)
            if row is None:
                rejects.append({"id": iid, "origin": "synth",
                                "family": site.family, "reason": why})
                continue
            row["mutation"] = {"family": site.family, "label": site.label,
                               "start": site.start, "end": site.end}
            rows.append(row)
            kept_this += 1
    return rows, rejects


def split_train_dev(rows: list, dev_ratio: float, seed: int) -> tuple:
    """按 **instance id** 切分（同种子派生的变体不跨集）。

    【为什么按"源种子"而不是按 instance id 字面量分组】synth 行的 id 形如
    `l0_clamp_bounds__cmp_flip_12_13`——它的源种子是 `l0_clamp_bounds`。
    若按 id 字面量随机切，同一个种子的 `cmp_flip` 变体可能在 train、`int_offby1`
    变体在 dev：两边的"正确实现"是**同一份代码**，dev 就不再是 held-out
    （同池污染的变体形态）。
    """
    groups = {}
    for r in rows:
        key = str(r["id"]).split("__")[0]
        groups.setdefault(key, []).append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_dev = max(1, int(round(len(keys) * max(0.0, min(1.0, dev_ratio)))))
    dev_keys = set(keys[:n_dev])
    train, dev = [], []
    for key in keys:
        (dev if key in dev_keys else train).extend(groups[key])
    return train, dev, {"n_groups": len(keys), "dev_groups": sorted(dev_keys)}


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default=None,
                    help=f"产物目录（默认 {S.swe_l0_dir()}）")
    ap.add_argument("--seeds", action=argparse.BooleanOptionalAction, default=True,
                    help="使用 rlab/swe_l0_problems.py 的手写种子（默认开）")
    ap.add_argument("--synth", action=argparse.BooleanOptionalAction, default=True,
                    help="用 swe_l0_synth 对种子做单点变异并执行验收（默认开）")
    ap.add_argument("--synth_per_seed", type=int, default=0,
                    help="每个种子最多保留多少条变异（0=不限，按变异族轮转取样）")
    ap.add_argument("--dev_ratio", type=float, default=0.25,
                    help="dev 比例（按源种子分组切，默认 0.25）")
    ap.add_argument("--test_timeout", type=float, default=60.0,
                    help="单次测试执行超时秒数（默认 60）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已存在的产物（默认拒绝，防误删旧池）")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir or S.swe_l0_dir())
    train_p, dev_p = out_dir / "train.jsonl", out_dir / "dev.jsonl"
    if not args.force and (train_p.exists() or dev_p.exists()):
        raise SystemExit(
            f"[prepare] 输出已存在，拒绝覆盖（防误删旧池）：\n"
            f"  {train_p}\n  {dev_p}\n"
            f"  改用 --output_dir 换目录，或确认要重建时加 --force。")

    rows, rejects = [], []
    if args.seeds:
        r_seed, j_seed = build_seed_rows(timeout=args.test_timeout)
        rows.extend(r_seed)
        rejects.extend(j_seed)
        print(f"[prepare] 手写种子: 通过 {len(r_seed)}｜剔除 {len(j_seed)}", flush=True)
    if args.synth:
        r_syn, j_syn = build_synth_rows(timeout=args.test_timeout,
                                        per_seed=args.synth_per_seed)
        rows.extend(r_syn)
        rejects.extend(j_syn)
        fam = family_counts(_sites_of(r_syn))
        print(f"[prepare] 合成变异: 通过 {len(r_syn)}｜剔除 {len(j_syn)}"
              f"｜保留族分布 {fam}", flush=True)
    if not rows:
        raise SystemExit(
            "[prepare] 没有任何 instance 通过验收——检查 swe_l0_problems.py 的种子"
            "是否自洽（原始码应当全绿、buggy 码应当至少一条红）。\n"
            f"剔除原因前 5 条：{rejects[:5]}")

    # 同 (Q) 去重（不同种子恰好产出同样提示时只留一条）
    seen_q, uniq = set(), []
    for r in rows:
        if r["Q"] in seen_q:
            continue
        seen_q.add(r["Q"])
        uniq.append(r)
    if len(uniq) != len(rows):
        print(f"[prepare] 按提示去重: {len(rows)} -> {len(uniq)}")
    rows = uniq

    train, dev, sp = split_train_dev(rows, args.dev_ratio, args.seed)
    if not train or not dev:
        raise SystemExit(
            f"[prepare] 切分后 train={len(train)} dev={len(dev)}：池子太小，"
            f"无法同时给出训练池与 held-out（提高 --synth_per_seed 或加种子）")

    # ---- fail-fast 校验（与加载侧同一套判据）----
    S.validate_swe_rows(train, "prepare train")
    S.validate_swe_rows(dev, "prepare dev")
    if len(dev) < 30:
        print(f"[prepare][警告] held-out 只有 {len(dev)} 个 instance——"
              f"L0 的 ±1~2pp 效应读不出来（docs/17 §4 的 N≥300 是 L2 级协议；"
              f"L0 起步规模小是事实，报告里必须写明）", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_p, train)
    _write_jsonl(dev_p, dev)

    # ---- 落盘复读自证（契约落在文件上）----
    back_tr = S._read_swe_rows(str(train_p))
    back_dv = S._read_swe_rows(str(dev_p))
    inter_id = {r["id"] for r in back_tr} & {r["id"] for r in back_dv}
    inter_q = {r["Q"] for r in back_tr} & {r["Q"] for r in back_dv}
    if inter_id or inter_q:
        raise RuntimeError(
            f"[prepare] 落盘复读发现 train/dev 相交：id {len(inter_id)} 条、"
            f"Q {len(inter_q)} 条（例：{sorted(inter_id)[:2]}）——拒绝放行")
    if len(back_tr) != len(train) or len(back_dv) != len(dev):
        raise RuntimeError(
            f"[prepare] 落盘复读行数不符：train {len(back_tr)} vs {len(train)}，"
            f"dev {len(back_dv)} vs {len(dev)}")
    bad_f2p = [r["id"] for r in back_tr + back_dv if not r.get("f2p")]
    if bad_f2p:
        raise RuntimeError(f"[prepare] 落盘复读发现 f2p 为空的行：{bad_f2p[:5]}")

    manifest = {
        "source": "swe_l0_problems + swe_l0_synth",
        "seed": args.seed, "dev_ratio": args.dev_ratio,
        "train_n": len(train), "dev_n": len(dev),
        "train_sha256": _sha256(train_p), "dev_sha256": _sha256(dev_p),
        "holdout_ok": True, "split": sp,
        "n_rejected": len(rejects),
        "reject_reasons": _reason_counts(rejects),
        "origin_counts": _origin_counts(rows),
        "f2p_total": sum(len(r["f2p"]) for r in rows),
        "p2p_total": sum(len(r["p2p"]) for r in rows),
        "rejected": rejects[:200],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] train {len(train)} / dev {len(dev)} → {out_dir}"
          f"（落盘复读：id/Q 不相交 ✅，行数一致 ✅，f2p 非空 ✅）", flush=True)
    print(f"[prepare] 剔除 {len(rejects)} 条，原因分布 {manifest['reject_reasons']}",
          flush=True)
    print("[prepare] 下一步：python -m rlab.train --algo swe_l0 --model_path "
          "<4B> --out_dir rlab_out/swe_l0_p1 ...", flush=True)


def _sites_of(rows):
    """把行里的 mutation 还原成可计数的轻量对象（family_counts 的入参）。"""
    class _S:
        __slots__ = ("family",)

        def __init__(self, fam):
            self.family = fam
    return [_S((r.get("mutation") or {}).get("family", "?"))
            for r in rows if r.get("mutation")]


def _reason_counts(rejects) -> dict:
    out = {}
    for r in rejects:
        why = (r.get("reason") or "?").split("（")[0].split("：")[0][:60]
        out[why] = out.get(why, 0) + 1
    return out


def _origin_counts(rows) -> dict:
    out = {}
    for r in rows:
        key = r.get("origin") or "?"
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
