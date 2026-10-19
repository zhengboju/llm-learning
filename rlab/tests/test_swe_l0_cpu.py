# -*- coding: utf-8 -*-
"""SWE-RL L0 的 CPU 回归测试（docs/17）。

全部不碰 GPU/网络；判分路径用的是**真 pytest 子进程 + 真 git apply**
（L0 的全部价值都在"真实执行"，用 mock 测出的绿字证明不了任何东西——
这与 search_qa 用真 HTTP 服务端到端测的原则一致）。

覆盖五层：
  P  patch 文本处理：围栏提取三种形态 / 路径消毒 / 危险路径拒绝
  G  git apply：check+apply 真跑 / 上下文不符 / 空 patch / 原子性（失败不留半成品）
  T  测试执行器：JUnit XML 解析 / 用例 id 归一 / 超时 / 收集期失败（空 results）
  E  判分与奖励：F2P 全过才 +1 / P2P 回归即 -1 / apply 失败单列 / 组内去重
  C  配置与接线：preset / fail-fast 矩阵 / 签名段 / data 分派 / 数据契约

【为什么必须测"收集期失败 ≠ 通过"】若测试文件语法错，pytest 的 results 是空的，
按"没有失败即通过"判就会把语法错误算成满分——这是最贵的一种假分数。
"""
import difflib
import json
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rlab import swe_l0 as S                                          # noqa: E402
from rlab.config import get_config                                   # noqa: E402
from rlab.swe_l0_problems import SEEDS                               # noqa: E402
from rlab.swe_l0_synth import (candidate_mutations, enumerate_sites,  # noqa: E402
                               family_counts)

_CLAMP_SRC = textwrap.dedent('''\
    def clamp(x, lo, hi):
        if lo > hi:
            raise ValueError("bad range")
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x
''')

_CLAMP_BUG = _CLAMP_SRC.replace("return hi", "return hi - 1")

_CLAMP_TST = textwrap.dedent('''\
    import pytest

    from calc import clamp


    def test_below():
        assert clamp(-5, 0, 10) == 0


    def test_above():
        assert clamp(50, 0, 10) == 10


    def test_inside():
        assert clamp(5, 0, 10) == 5


    def test_bad_range():
        with pytest.raises(ValueError):
            clamp(1, 10, 0)
''')


def _row(files=None, tests=None, f2p=None, p2p=None, rid="t1", q=None):
    return {"id": rid, "Q": q or f"fix the bug in {rid}", "A": "fixed",
            "files": files or {"calc.py": _CLAMP_BUG},
            "tests": tests or {"test_calc.py": _CLAMP_TST},
            "test_files": ["test_calc.py"],
            "f2p": f2p if f2p is not None else ["test_calc::test_above"],
            "p2p": p2p if p2p is not None else ["test_calc::test_below"]}


def _diff(old, new, fname="calc.py"):
    return "".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True),
        fromfile="a/" + fname, tofile="b/" + fname))


# =====================================================================
# P 组：patch 文本处理
# =====================================================================

def test_p1_extract_fenced_diff():
    d = _diff(_CLAMP_BUG, _CLAMP_SRC)
    from rlab.rollout import extract_patch_text
    assert "return hi\n" in extract_patch_text("ok:\n```diff\n" + d + "```\ndone")


def test_p2_extract_fenced_patch():
    from rlab.rollout import extract_patch_text
    d = _diff(_CLAMP_BUG, _CLAMP_SRC)
    assert "return hi\n" in extract_patch_text("```patch\n" + d + "```")


def test_p3_extract_bare_diff():
    from rlab.rollout import extract_patch_text
    d = _diff(_CLAMP_BUG, _CLAMP_SRC)
    out = extract_patch_text("here is the fix.\n" + d + "\nhope it helps")
    assert out.lstrip().startswith("--- ") and "return hi" in out


def test_p4_extract_takes_last_fence():
    """模型"先草稿后最终版"时取最后一个——取第一个会拿被自己否定的草稿判分。"""
    from rlab.rollout import extract_patch_text
    d1 = _diff(_CLAMP_BUG, _CLAMP_SRC)
    d2 = _diff("a\n", "b\n", "other.py")
    out = extract_patch_text("```diff\n" + d1 + "```\nActually:\n```diff\n" + d2 + "```")
    assert "other.py" in out and "calc.py" not in out


def test_p5_extract_no_diff_passes_through():
    """没有 diff 时**原样返回**（让 apply 去失败并如实归因，不静默丢成空串）。"""
    from rlab.rollout import extract_patch_text
    assert extract_patch_text("I cannot fix this.") == "I cannot fix this."


def test_p6_patch_paths_and_sanitize():
    txt = "--- a/x.py\n+++ b/x.py\n--- /dev/null\n+++ b/pkg/mod.py\n"
    assert S.patch_paths(txt) == ["x.py", "pkg/mod.py"]
    assert S.sanitize_patch_path("a/b/c.py") == "b/c.py"
    # 危险形态一律拒绝
    for bad in ("/etc/passwd", "../evil.py", "a/../../x.py", "c:\\win.py", ""):
        assert S.sanitize_patch_path(bad) is None


def test_p7_patch_paths_empty_for_non_diff():
    assert S.patch_paths("just prose, no headers") == []
    assert S.patch_paths("") == []


# =====================================================================
# G 组：git apply（真跑）
# =====================================================================

def test_g1_good_patch_applies(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    res = S.check_patch(_diff(_CLAMP_BUG, _CLAMP_SRC), wd)
    assert res["ok"] and res["stage"] == "applied" and res["files"] == ["calc.py"]
    assert "return hi\n" in open(os.path.join(wd, "calc.py"), encoding="utf-8").read()


def test_g2_context_mismatch_fails_with_stage_check(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    bad = _diff(_CLAMP_BUG.replace("return hi", "return hh"), _CLAMP_SRC)
    res = S.check_patch(bad, wd)
    assert not res["ok"]
    # --check 挡在写盘之前 → stage="check"（不是"apply"）
    assert res["stage"] == "check"


def test_g3_empty_patch_is_stage_empty(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    before = open(os.path.join(wd, "calc.py"), encoding="utf-8").read()
    res = S.check_patch("no diff here", wd)
    assert res["stage"] == "empty"
    # 文件必须原封不动（empty 档也不许碰盘）
    assert open(os.path.join(wd, "calc.py"), encoding="utf-8").read() == before


def test_g4_no_partial_write_on_check_failure(tmp_path):
    """--check 失败时**文件必须没被改**（原子性：不留半成品让后续判分受污染）。"""
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    before = open(os.path.join(wd, "calc.py"), encoding="utf-8").read()
    S.check_patch(_diff("wrong\n", _CLAMP_SRC), wd)
    assert open(os.path.join(wd, "calc.py"), encoding="utf-8").read() == before


def test_g5_too_many_files_rejected(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    txt = "".join("--- a/f%d.py\n+++ b/f%d.py\n@@ -1 +1 @@\n-a\n+b\n" % (i, i)
                  for i in range(S.MAX_PATCH_FILES + 1))
    res = S.check_patch(txt, wd)
    assert res["stage"] == "too_many_files"


def test_g6_oversized_patch_rejected(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    res = S.check_patch("--- a/x\n+++ b/x\n" + "x" * (S.MAX_PATCH_CHARS + 10), wd)
    assert res["stage"] == "too_long"


# =====================================================================
# T 组：测试执行器
# =====================================================================

def test_t1_junit_parse_states():
    xml = textwrap.dedent('''\
        <testsuites><testsuite>
        <testcase classname="test_m" name="a" time="0.1" />
        <testcase classname="test_m" name="b" time="0.1"><failure message="x">boom</failure></testcase>
        <testcase classname="test_m" name="c" time="0.1"><skipped /></testcase>
        <testcase classname="test_m" name="d" time="0.1"><error message="e" /></testcase>
        </testsuite></testsuites>
    ''')
    got = S.parse_junit_xml(xml)
    assert got == {"test_m::a": "passed", "test_m::b": "failed",
                   "test_m::c": "skipped", "test_m::d": "failed"}


def test_t2_case_id_module_and_class():
    assert S.test_case_id("test_calc", "test_x") == "test_calc::test_x"
    # 类内用例：保留类名（与 pytest 自己的节点 id 一致）
    assert S.test_case_id("test_calc.TestC", "test_x") == "test_calc::TestC::test_x"


def test_t3_real_pytest_run(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(files={"calc.py": _CLAMP_BUG}), wd)
    res = S.run_tests(wd, ["test_calc.py"], timeout=120)
    assert res["results"]["test_calc::test_above"] == "failed"
    assert res["results"]["test_calc::test_below"] == "passed"
    assert not res["ok"]


def test_t4_collection_error_never_scores(tmp_path):
    """收集期失败（语法错）→ 绝不能拿到满分。

    【实测形态】pytest 遇到 import/语法错时会在 JUnit 里写一条
    `classname="" name="<文件名>"` 的伪用例（不是真用例 id）——所以 `results`
    非空，但它与 f2p 的 id 对不上 → f2p_failed 覆盖全部 → reward -1。
    这个断言钉的就是"语法错误必须表现为不通过"，而不是"results 必须为空"
    （后者是初版测试的错假设，真跑才发现）。
    """
    wd = str(tmp_path)
    row = _row(tests={"test_calc.py": "def broken(:\n    pass\n"})
    S.materialize_repo(row, wd)
    res = S.run_tests(wd, ["test_calc.py"], timeout=120)
    assert not res["ok"]
    # 伪用例的 id 不含 `::`（真用例才有），故它永远不会被误认成 f2p 命中
    assert all("::" not in k for k in res["results"]) or not res["results"]


def test_t4b_collection_error_scores_minus_one(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(tests={"test_calc.py": "def broken(:\n    pass\n"}), wd)
    res = S.evaluate_patch(wd, _diff(_CLAMP_BUG, _CLAMP_SRC), _spec())
    assert res["apply_ok"] and res["reward"] == -1.0 and res["acc"] == 0.0


def test_t5_timeout_is_flagged(tmp_path):
    wd = str(tmp_path)
    row = _row(tests={"test_calc.py": "import time\n\ndef test_slow():\n"
                                     "    time.sleep(30)\n"})
    S.materialize_repo(row, wd)
    res = S.run_tests(wd, ["test_calc.py"], timeout=3)
    assert res["timed_out"] and not res["ok"]


def test_t6_missing_test_file_is_not_ok(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    res = S.run_tests(wd, [], timeout=30)
    assert not res["ok"] and res["detail"]


# =====================================================================
# E 组：判分与奖励
# =====================================================================

def _spec(f2p=None, p2p=None):
    return {"f2p": f2p if f2p is not None else ["test_calc::test_above"],
            "p2p": p2p if p2p is not None else ["test_calc::test_below"],
            "test_files": ["test_calc.py"]}


def test_e1_reference_fix_scores_plus_one(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    res = S.evaluate_patch(wd, _diff(_CLAMP_BUG, _CLAMP_SRC), _spec())
    assert res["apply_ok"] and res["reward"] == 1.0 and res["acc"] == 1.0
    assert res["f2p_pass"] == 1 and not res["f2p_failed"] and not res["p2p_broken"]


def test_e2_wrong_fix_scores_minus_one(tmp_path):
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    wrong = _diff(_CLAMP_BUG, _CLAMP_BUG.replace("return hi - 1", "return hi - 2"))
    res = S.evaluate_patch(wd, wrong, _spec())
    assert res["apply_ok"] and res["reward"] == -1.0 and res["acc"] == 0.0


def test_e3_p2p_regression_scores_minus_one(tmp_path):
    """F2P 过了但弄坏了已通过的测试 → 0 分（"修 bug 不能拆台"）。"""
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    both = _diff(_CLAMP_BUG, _CLAMP_SRC.replace("return lo\n", "return lo + 100\n"))
    res = S.evaluate_patch(wd, both, _spec())
    assert res["apply_ok"] and res["reward"] == -1.0
    assert res["p2p_broken"] == ["test_calc::test_below"]


def test_e4_apply_failure_is_format_not_capability(tmp_path):
    """apply 失败：reward -1、format 0，但 **apply_stage 单列**（条件修复率的分母来源）。"""
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    res = S.evaluate_patch(wd, "not a patch", _spec())
    assert res["reward"] == -1.0 and res["format"] == 0.0
    assert res["apply_ok"] is False and res["apply_stage"] in ("empty", "check")
    # 关键：没跑测试也要说清"为什么没跑"
    assert res["detail"]


def test_e5_empty_f2p_refused(tmp_path):
    """F2P 为空 → 无法定义"修好"，必须拒判（而不是恒 +1）。"""
    wd = str(tmp_path)
    S.materialize_repo(_row(), wd)
    res = S.evaluate_patch(wd, _diff(_CLAMP_BUG, _CLAMP_SRC), _spec(f2p=[]))
    assert res["reward"] == -1.0 and "f2p" in res["detail"].lower()
    # spec 里 f2p 为空时一次测试都不该跑（省最贵的那一步）
    assert res["format"] == 0.0


def test_e6_total_reward_uses_fresh_copy():
    """每次判分必须是**全新副本**：复用会让第二次看到上一次的 patch。"""
    row = _row()
    factory = S.make_workdir_factory(row)
    good = _diff(_CLAMP_BUG, _CLAMP_SRC)
    r1 = S.total_reward_swe_l0(_spec(), good, factory)
    r2 = S.total_reward_swe_l0(_spec(), good, factory)
    assert r1["reward"] == 1.0 and r2["reward"] == 1.0
    # 补丁若被重复应用，第二次会 fail（"return hi" 已不在）→ 这里断言它没有发生
    assert r2["apply_ok"] is True


def test_e7_dedup_group_indices():
    assert S.dedup_group_indices(["a\nb", "a\nb ", "", "c"]) == {
        0: [0, 1], 2: [2], 3: [3]}
    # 空 patch 不合并（各自都是 -1，合并只会让"几条空轨迹"失真）
    assert S.dedup_group_indices(["", ""]) == {0: [0], 1: [1]}


def test_e8_group_scoring_dedups_and_reports(tmp_path):
    from rlab.rollout import score_group_swe_l0
    cfg = get_config("swe_l0", use_wandb=False)
    cfg = {**cfg, "num_pre_Q": 4}
    row = _row()
    calls = []

    def fake_exec(r, patch):
        calls.append(patch)
        good = "return hi\n" in patch
        return {"reward": 1.0 if good else -1.0, "acc": 1.0 if good else 0.0,
                "format": 1.0 if patch.strip().startswith("---") else 0.0,
                "apply_ok": patch.strip().startswith("---"),
                "apply_stage": "applied" if patch.strip().startswith("---") else "empty",
                "f2p_pass": 1 if good else 0, "f2p_total": 1, "p2p_broken": 0,
                "p2p_total": 1, "detail": "", "duration": 0.01}

    good = _diff(_CLAMP_BUG, _CLAMP_SRC)
    answers = ["```diff\n" + good + "```", "```diff\n" + good + "```",
               "no idea", ""]
    adv, acc, fmt, st = score_group_swe_l0([row], answers, cfg,
                                           test_executor=fake_exec)
    # 两条相同的 patch 只测一次 + 两条不同的 → 3 次
    assert len(calls) == 3
    assert st["n_tested"] == 3 and st["n_trajs"] == 4
    assert acc.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert st["apply_ok"] == [1, 1, 0, 0]


def test_e9_synth_mutations_are_parseable():
    """合成变异产物必须仍是合法 python（不可解析的变异无意义）。"""
    sites = enumerate_sites(_CLAMP_SRC)
    assert sites, "clamp 源码上应当能找到变异点"
    for site, mutated in candidate_mutations(_CLAMP_SRC):
        compile(mutated, "<mut>", "exec")
        assert mutated != _CLAMP_SRC


def test_e10_synth_site_offsets_are_char_not_byte():
    """含中文的源码里，变异位置必须按**字符**偏移（CPython 报的是 UTF-8 字节列）。"""
    src = textwrap.dedent('''\
        def f(x):
            """中文文档串，让 col_offset 与字符偏移不同。"""
            if x < 0:
                return 0
            return x
    ''')
    for site, mutated in candidate_mutations(src):
        assert mutated != src
        compile(mutated, "<mut>", "exec")


def test_e11_synth_family_counts():
    fam = family_counts(enumerate_sites(_CLAMP_SRC))
    assert isinstance(fam, dict) and sum(fam.values()) >= 4


# =====================================================================
# C 组：配置与接线
# =====================================================================

def test_c1_preset_ok():
    cfg = get_config("swe_l0", use_wandb=False)
    assert cfg["data_task"] == "swe_l0"
    assert cfg["max_rounds"] == 1                 # 单轮
    assert cfg["tool_protocol"] == "fence"        # 无工具
    assert cfg["max_prompt_length"] + cfg["max_gen_tokens"] \
        <= cfg["max_context_tokens"]
    assert cfg["_swe_check"]["swe_test_timeout"] > 0


def test_c2_preset_has_no_shaping():
    """L0 的 reward 是 F2P 硬判据：长度/工具 shaping 一律为 0（防捷径）。"""
    cfg = get_config("swe_l0", use_wandb=False)
    for key in ("code_w", "code_attempt_w", "len_penalty_w", "tool_call_cost",
                "tool_waste_penalty", "trunc_tail_penalty", "len_eff_w",
                "len_eff_penalty_w", "trunc_shaping"):
        assert float(cfg.get(key, 0.0) or 0.0) == 0.0, key
    assert cfg["overlong_shaping"] is False and cfg["overlong_filter"] is False


@pytest.mark.parametrize("kw", [
    {"tool_protocol": "native"},
    {"max_rounds": 3},
    {"max_gen_tokens": 256},
    {"max_prompt_length": 4000},
    {"code_w": 0.1},
    {"tool_call_cost": 0.02},
    {"swe_test_timeout": 0},
    {"swe_test_mem_mb": 64},
    {"swe_test_workers": 0},
    # 【静默空转防线】单轮路径不读它（use_qqueue 要求 is_retool）→ 必须拒
    {"gen_questions_per_attempt": 4},
])
def test_c3_fail_fast_matrix(kw):
    with pytest.raises(ValueError):
        get_config("swe_l0", use_wandb=False, **kw)


def test_c4_other_presets_unaffected():
    """泛化不得改动既有 preset 的任何预算项（历史对照位）。"""
    c = get_config("retool_math", use_wandb=False)
    assert (c["max_rounds"], c["round_gen_tokens"], c["max_context_tokens"]) \
        == (4, 2048, 14336)
    s = get_config("search_qa", use_wandb=False, search_endpoint="http://x:1")
    assert s["native_tool_name"] == "search"
    g = get_config("grpo", use_wandb=False)
    assert g["max_gen_tokens"] == 512 and g["data_task"] == "gsm8k"


def test_c5_signature_has_swe_segments():
    from rlab.train import run_signature
    a = get_config("swe_l0", use_wandb=False)
    b = get_config("swe_l0", use_wandb=False, swe_test_timeout=30.0)
    for c in (a, b):
        c["all_steps"], c["save_steps"] = 20, 5
    assert "-swt60" not in run_signature(a)        # 默认值不占签名
    assert "-swt30" in run_signature(b)            # 偏离必须可分辨


def test_c6_swe_suffixes_are_opt_segments():
    """-swt/-swp 必须登记为优化器段，否则 L0 的 run 崩溃后重启会被自己拦死。"""
    from rlab.train import _is_opt_suffix
    assert _is_opt_suffix("-swt30") and _is_opt_suffix("-swpdeadbe")
    assert _is_opt_suffix("-swt30-swpdeadbe")


def test_c7_system_prompt_no_tags():
    cfg = get_config("swe_l0", use_wandb=False)
    sp = cfg["system_prompt"]
    from rlab.protocol import _TOOL_OPEN
    assert _TOOL_OPEN not in sp and "tool_call" not in sp
    # 必须说清"只输出一个 diff 块"（格式失败率是 L0 的主要瓶颈）
    assert "diff" in sp.lower() and "git apply" in sp.lower()


def test_c8_load_qas_dispatches_swe_l0(tmp_path, monkeypatch):
    import rlab.data as D
    tr = tmp_path / "train.jsonl"
    dv = tmp_path / "dev.jsonl"
    tr.write_text(json.dumps(_row(rid="a", q="question a?")) + "\n", encoding="utf-8")
    dv.write_text(json.dumps(_row(rid="b", q="question b?")) + "\n", encoding="utf-8")
    monkeypatch.setattr(S, "_SWE_L0_DIR", str(tmp_path))
    rows = D.load_qas("swe_l0")
    assert len(rows) == 1 and rows[0]["id"] == "a"


def test_c9_loader_rejects_empty_f2p(tmp_path):
    """数据侧硬闸：f2p 为空的行必须炸（"恒 100%" 的数据侧入口）。"""
    bad = _row()
    bad["f2p"] = []
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        S.validate_swe_rows([bad], "test")
    with pytest.raises(RuntimeError):
        S._load_one(str(p), "test")


def test_c10_loader_rejects_unsafe_paths(tmp_path):
    bad = _row()
    bad["files"] = {"../evil.py": "x"}
    with pytest.raises(RuntimeError):
        S.validate_swe_rows([bad], "test")


def test_c11_heldout_contract_is_checked(tmp_path):
    """train 里混入 dev 的 id → 当场剔除并告警（契约落盘，不靠声明）。

    注意 Q 也要互不相同：held-out 契约按 **id 或 Q** 双判（本项目两次事故
    都是"同题面不同行"），本测试想验的是 id 那一半。
    """
    tr = tmp_path / "train.jsonl"
    dv = tmp_path / "dev.jsonl"
    tr.write_text(json.dumps(_row(rid="same", q="shared question?")) + "\n"
                  + json.dumps(_row(rid="only_train", q="unique question?")) + "\n",
                  encoding="utf-8")
    dv.write_text(json.dumps(_row(rid="same", q="shared question?")) + "\n",
                  encoding="utf-8")
    rows = S.load_swe_l0_train(str(tr), str(dv))
    assert [r["id"] for r in rows] == ["only_train"]


def test_c12_missing_dev_refuses(tmp_path):
    tr = tmp_path / "train.jsonl"
    tr.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        S.load_swe_l0_train(str(tr), str(tmp_path / "nope.jsonl"))


def test_c13_all_seeds_are_self_consistent():
    """每个种子：正确码全绿、buggy 码至少一条红（种子自证的门槛）。"""
    from rlab.prepare_swe_l0 import derive_spec
    failures = []
    for seed in SEEDS:
        got = derive_spec(seed.target, seed.testname, seed.tests,
                          seed.source, seed.buggy, timeout=120)
        if not got["ok"]:
            failures.append((seed.name, got["reason"]))
    assert not failures, f"种子不自洽: {failures}"


def test_c14_materialize_nested_paths(tmp_path):
    row = _row(files={"pkg/mod.py": "X = 1\n"})
    dest = str(tmp_path / "repo")
    S.materialize_repo(row, dest)
    assert os.path.exists(os.path.join(dest, "pkg", "mod.py"))
    assert os.path.exists(os.path.join(dest, "test_calc.py"))
