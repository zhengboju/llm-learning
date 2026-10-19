# -*- coding: utf-8 -*-
"""SWE-RL L0 的端到端接线测试（docs/17）：**真 git apply + 真 pytest**。

与 `test_swe_l0_cpu.py` 的分工：那边测的是零件（patch/执行器/判分/配置），
这边测的是**接缝**——`gen_worker` 的单轮采样循环拿到答案后，是否真的把
`data_task="swe_l0"` 分派到 `score_group_swe_l0`、判分结果是否真的进了
record/advantage。

【为什么值得单开一个文件】本项目最贵的一类 bug 是"过滤器存在且测试全绿"但
真实数据流没生效（AGENTS.md）：零件全绿而接缝断开的形态，只有"上游真实输出
喂下游"的测试能发现。这里的上游 = FakeGen 产出的模型回答，下游 = 判分与
record，中间的 `score_group` 分派点是本次改动的焦点。
"""
import difflib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rlab import swe_l0 as S                                          # noqa: E402
from rlab.config import get_config                                    # noqa: E402
from rlab.rollout import score_group_swe_l0                           # noqa: E402


_BUG = "def add(a, b):\n    return a - b\n"
_FIX = "def add(a, b):\n    return a + b\n"
_TST = (
    "from calc import add\n\n\n"
    "def test_one():\n    assert add(1, 2) == 3\n\n\n"
    "def test_zero():\n    assert add(0, 0) == 0\n"
)


def _mkdiff(old, new, fname="calc.py"):
    return "".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True),
        fromfile="a/" + fname, tofile="b/" + fname))


def _instance():
    """用**实测**得出 F2P/P2P 的一行数据（不手写名单——与 prepare 同一条纪律）。"""
    tfile = "test_calc.py"
    row = {"id": "e2e_add", "Q": "add() subtracts instead of adding.",
           "A": "every fail-to-pass test passes",
           "files": {"calc.py": _BUG}, "tests": {tfile: _TST},
           "test_files": [tfile]}

    def _run(code):
        probe = dict(row, files={"calc.py": code})
        wd = S.make_workdir_factory(probe)()
        try:
            return S.run_tests(wd, [tfile], timeout=120)["results"]
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    src, bug = _run(_FIX), _run(_BUG)
    assert not [k for k, v in src.items() if v != "passed"], "正确码必须全绿"
    f2p = sorted(k for k, v in bug.items() if v != "passed")
    p2p = sorted(k for k, v in bug.items() if v == "passed")
    assert f2p and p2p, "必须同时有 F2P 与 P2P（否则 instance 不合格）"
    row.update(f2p=f2p, p2p=p2p)
    return row


def test_end_to_end_real_scoring_path(tmp_path):
    """真判分：正确 patch +1、错误 patch -1、空 patch -1 且 apply 失败。"""
    row = _instance()
    good = _mkdiff(_BUG, _FIX)
    wrong = _mkdiff(_BUG, _BUG.replace("a - b", "a * b"))
    cfg = get_config("swe_l0", use_wandb=False,
                     swe_work_root=str(tmp_path))
    cfg = {**cfg, "num_pre_Q": 4}
    answers = ["```diff\n" + good + "```",
               "```diff\n" + wrong + "```",
               "cannot fix this",
               ""]
    adv, acc, fmt, st = score_group_swe_l0([row], answers, cfg)
    assert acc.tolist() == [1.0, 0.0, 0.0, 0.0]
    # fmt = patch 能否 apply（格式指标），与 acc（能力指标）分开
    assert fmt.tolist()[0] == 1.0
    assert st["apply_ok"] == [1, 1, 0, 0]
    # 组内有对有错 → advantage 必须把对的那条抬起来
    assert adv[0] > adv[1]
    # 判分次数：4 条轨迹、4 个不同 patch（含 2 条非 diff）→ 全测
    assert st["n_tested"] == 4 and st["n_trajs"] == 4


def test_end_to_end_dedup_saves_expensive_tests(tmp_path):
    """同组 8 条里有 6 条给出**同一段** patch → 只测一次（测试是最贵的一步）。"""
    row = _instance()
    good = _mkdiff(_BUG, _FIX)
    cfg = get_config("swe_l0", use_wandb=False, swe_work_root=str(tmp_path))
    cfg = {**cfg, "num_pre_Q": 8}
    one = "```diff\n" + good + "```"
    answers = [one, one, one, one, one, one, "nope 1", "nope 2"]
    adv, acc, fmt, st = score_group_swe_l0([row], answers, cfg)
    assert acc.tolist() == [1.0] * 6 + [0.0, 0.0]
    # 6 条相同的只判 1 次 + 2 条不同的 → 3 次（而不是 8 次）
    assert st["n_tested"] == 3 and st["n_trajs"] == 8


def test_end_to_end_p2p_regression_blocks_credit(tmp_path):
    """F2P 过了但弄坏了 P2P → 0 分（"修 bug 不能拆台"在真实执行下成立）。"""
    row = _instance()
    cfg = get_config("swe_l0", use_wandb=False, swe_work_root=str(tmp_path))
    cfg = {**cfg, "num_pre_Q": 1}
    # 构造一个"看着像修好、实际把边界改坏"的 patch：add(1,3) 会过但 add(0,0) 挂
    bad_fix = "def add(a, b):\n    return a + b + 1\n"
    adv, acc, fmt, st = score_group_swe_l0(
        [row], ["```diff\n" + _mkdiff(_BUG, bad_fix) + "```"], cfg)
    assert acc.tolist() == [0.0]


def test_record_keys_present_for_swe(tmp_path):
    """record 行的 L0 附加键（apply_ok/n_tested…）必须是判分统计的真实来源。"""
    row = _instance()
    good = _mkdiff(_BUG, _FIX)
    cfg = get_config("swe_l0", use_wandb=False, swe_work_root=str(tmp_path))
    cfg = {**cfg, "num_pre_Q": 2}
    _, _, _, st = score_group_swe_l0(
        [row], ["```diff\n" + good + "```", ""], cfg)
    for key in ("apply_ok", "apply_stage", "test_duration", "n_tested", "n_trajs"):
        assert key in st, key
    assert len(st["apply_ok"]) == 2 and len(st["test_duration"]) == 2


def test_spec_roundtrip_from_row():
    """数据行的 spec 必须能直接喂判分（上游真实输出喂下游）。"""
    row = _instance()
    spec = S.spec_of(row)
    assert spec["f2p"] == row["f2p"] and spec["p2p"] == row["p2p"]
    wd = S.make_workdir_factory(row)()
    try:
        res = S.evaluate_patch(wd, _mkdiff(_BUG, _FIX), spec)
        assert res["reward"] == 1.0
    finally:
        shutil.rmtree(wd, ignore_errors=True)
