# -*- coding: utf-8 -*-
"""rlab/swe_l0.py — SWE-RL L0 层的环境：patch 校验、测试执行、奖励、数据加载。

docs/17 的 L0 层 = HumanEvalFix 式合成 bug：**单函数 + 单测现成 + 定位已给出**，
动作空间是**单轮 patch-only**（模型读 issue + 已定位的局部代码，直接输出 diff，
没有工具调用——docs/17 §2.5 的定版）。本模块因此不碰 torch/vLLM，全是
"文件 IO + 子进程 + 纯函数"，CPU 可测，与 `search_qa.py` 的定位同构。

四块内容
--------
1. **patch 校验**（`check_patch`）：`git apply --check` + 实际 apply，输出
   `{"ok", "stage", "detail", "files"}`。stage 区分**格式指标**与**能力指标**
   （docs/17 §3 的硬要求：apply 失败单独计数，报条件修复率 = 修复率/apply 成功率）。
2. **pytest 执行器**（`run_tests`）：在**独立子进程**里跑指定测试节点，解析
   JUnit XML 得逐条 pass/fail，带超时与资源上限（复用 sandbox.py 的
   RLIMIT 思路）。**不用 `-q` 文本解析**：文本格式随 pytest 版本漂移，JUnit XML
   是有 schema 的稳定契约（`<testcase classname= name=>` 里有无 `<failure>`）。
3. **奖励**（`total_reward_swe_l0`）：F2P 全过 + P2P 不回归 → +1，否则 -1；
   apply 失败**不等于**做错——它是格式指标，单独上报（`apply_ok`），
   与 acc/fmt 拆分同一原则。
4. **数据**：`rlab/datasets/swe_l0/{train,dev}.jsonl` 的加载与 held-out 契约
   （与 search_qa 同一套"落盘复读、真去读 dev"的做法）。

【为什么 F2P/P2P 名单绝不手写】
F2P（fail-to-pass）= 在 buggy 码上失败、在正确码上通过的测试；P2P（pass-to-pass）
= 两边都通过的测试。它们是**实测产物**（`prepare_swe_l0.py` 跑的），手写会在
"注入的 bug 其实没被覆盖"时静默变成假 F2P —— 而那时 reward 恒 1、"精确 100%"
正是本项目认定的系统性 bug 签名（AGENTS.md）。加载侧另有一道闸：
F2P 为空直接拒入池（没有失败测试的 instance 无法定义"修好"）。

【零标签字面量铁律】本模块不含任何格式标签字节；patch 文本原样透传。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# =====================================================================
# 常量与上限（防御"模型写了病态 patch/测试"把训练机拖死）
# =====================================================================
MAX_PATCH_CHARS = 64 * 1024        # patch 文本上限（L0 是单函数级，64K 已极宽）
MAX_PATCH_FILES = 8                # 允许改动的文件数上限（L0 只该动 1~2 个）
PATCH_TIMEOUT = 60.0               # git apply 超时（秒）
TEST_TIMEOUT_DEFAULT = 120.0       # 一次测试执行的超时（秒）
TEST_MEM_MB_DEFAULT = 2048         # 子进程内存上限（Linux RLIMIT_AS；Windows 跳过）
TEST_OUTPUT_MAX_CHARS = 4000       # 失败详情截断（防输出炸弹）

# 允许的改动路径：相对仓库根的 POSIX 路径，拒绝绝对路径/`..`/盘符
_SAFE_REL_RE = re.compile(r"^[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*$")


class PatchError(ValueError):
    """patch 文本本身不合法（超长/含非法路径/结构坏）——与"apply 失败"分开。"""


# =====================================================================
# 1. patch 校验
# =====================================================================
def sanitize_patch_path(path: str):
    """把 diff 头部里的路径归一成安全相对路径；不可用返回 None。

    `git apply` 本身会拒绝绝对路径与 `..`，但**它只在 apply 阶段拒**——那时
    我们已经把文件写进临时目录了。这里先做一遍纯文本检查，让"病态路径"在
    任何文件系统操作之前就被拦下（与 sandbox 的"消毒在拼接口做"同一条原则）。

    【绝对路径必须先判再剥】**不能**用 `p.lstrip("./")` 去前缀：lstrip 接的是
    **字符集**不是前缀，`/etc/passwd` 会被剥成 `etc/passwd`——一个绝对路径
    悄悄变成合法相对路径，正是这个函数存在的意义被它自己绕过（CPU 回归 P6
    锁死了这一点）。
    """
    if not isinstance(path, str):
        return None
    p = path.strip().strip('"')
    # /dev/null 是 diff 的合法占位（新增/删除文件），不是被改动的路径
    if p == "/dev/null":
        return None
    # ① 绝对路径与盘符：在任何剥离之前判
    if p.startswith(("/", "\\")) or (len(p) >= 2 and p[1] == ":"):
        return None
    # ② a/ b/ 与 ./a/ ./b/ 前缀（-p1 的常见形态），按**前缀**精确剥
    for pfx in ("./a/", "./b/", "a/", "b/"):
        if p.startswith(pfx):
            p = p[len(pfx):]
            break
    while p.startswith("./"):
        p = p[2:]
    if not p or "\\" in p or ".." in p.split("/"):
        return None
    if not _SAFE_REL_RE.match(p):
        return None
    return p


def patch_paths(patch_text: str) -> list:
    """抽 diff 里所有**被改动**的文件路径（去重保序）。纯函数，CPU 可测。

    只认 unified diff 的三种行：`--- `、`+++ `、`diff --git `。模型常写成
    `+++ b/x.py`（正常）或 `+++ x.py`（免 a/ b/ 前缀），两种都吃——但**路径一律
    过 sanitize**，不放行绝对路径/上跳。返回 [] 表示"这段文本看起来不像 diff"。
    """
    out, seen = [], set()
    for raw in (patch_text or "").splitlines():
        cand = None
        if raw.startswith("--- ") or raw.startswith("+++ "):
            cand = raw[4:].split("\t")[0].strip()
        elif raw.startswith("diff --git "):
            parts = raw[len("diff --git "):].split()
            if len(parts) >= 2:
                cand = parts[1]
        if not cand:
            continue
        p = sanitize_patch_path(cand)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _git_env() -> dict:
    """git 子进程环境：隔离用户/系统配置，让 apply 行为与开发机无关。

    `core.autocrlf` 必须显式关掉：仓库里若开了它，`git apply` 会按 CRLF 改写
    被 apply 的文件，于是"测试命中与否"取决于机器的 git 配置，而不是 patch
    本身——这正是本项目最忌讳的"口径随环境漂移"。
    """
    env = dict(os.environ)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return env


def _git_apply(workdir: str, patch_file: str, *, check: bool, extra: list = None):
    """跑一次 git apply。返回 (returncode, stderr)。

    `--recount` 让"@@ 行数写错但上下文对"的 patch 仍然能落（模型最常见的
    格式瑕疵之一，docs/17 §1 把格式错误率列为 L0 的主要瓶颈之一）；不接受
    `--ignore-whitespace`（那会让"缩进写错"的 patch 也通过，而缩进在 python
    里就是语义——放行它等于把语法错误算成"格式没问题"）。
    """
    cmd = ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
           "apply", "-p1", "--recount"]
    if check:
        cmd.append("--check")
    cmd.extend(extra or [])
    cmd.append(patch_file)
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                              timeout=PATCH_TIMEOUT, env=_git_env())
    except subprocess.TimeoutExpired:
        return 124, "git apply timed out"
    except OSError as exc:                       # git 不存在/不可执行
        return 127, f"git unavailable: {exc}"
    return proc.returncode, (proc.stderr or "").strip()


def check_patch(patch_text: str, workdir: str) -> dict:
    """校验并实际 apply 一段 patch 到 `workdir`（原地修改，调用方负责副本）。

    返回 `{"ok", "stage", "detail", "files", "duration"}`：
      · `ok=True`  → stage="applied"，patch 已落到 workdir；
      · `ok=False` → stage ∈ {"empty", "too_long", "no_paths", "too_many_files",
        "check", "apply"}，detail 是给人看的短原因。

    【stage 为什么要分细】docs/17 §3 要求把 **apply 失败单独计数**（它是格式
    指标，不等同能力指标）：全失败率 90% 与"能 apply 但修不对"是两种完全不同的
    病，处置相反（前者治输出格式/长度预算，后者治推理）。只报一个"修复率 2%"
    会让两者混成一个数字。
    """
    t0 = time.monotonic()
    res = {"ok": False, "stage": "empty", "detail": "", "files": [],
           "duration": 0.0}
    text = patch_text or ""
    if len(text) > MAX_PATCH_CHARS:
        res.update(stage="too_long",
                   detail=f"patch 超过 {MAX_PATCH_CHARS} 字符上限（{len(text)}）")
        return res
    files = patch_paths(text)
    res["files"] = files
    if not files:
        res.update(stage="empty", detail="没有可识别的 diff 文件头（--/++/diff --git）")
        return res
    if len(files) > MAX_PATCH_FILES:
        res.update(stage="too_many_files",
                   detail=f"改动了 {len(files)} 个文件 > 上限 {MAX_PATCH_FILES}")
        return res
    # 缺结尾换行是 unified diff 的经典杀手（git 会当成"最后一行无换行"的语义）
    if not text.endswith("\n"):
        text = text + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False,
                                     encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        patch_file = fh.name
    try:
        rc, err = _git_apply(workdir, patch_file, check=True)
        if rc != 0:
            res.update(stage="check", detail=(err or "git apply --check 失败")[:600])
            return res
        rc, err = _git_apply(workdir, patch_file, check=False)
        if rc != 0:
            # --check 过了但真 apply 失败：磁盘/权限类问题，单独一档（它不是
            # 模型的错，却会静默变成"修不对"）。
            res.update(stage="apply", detail=(err or "git apply 失败")[:600])
            return res
        res.update(ok=True, stage="applied", detail="")
        return res
    finally:
        try:
            os.unlink(patch_file)
        except OSError:
            pass
        res["duration"] = time.monotonic() - t0


# =====================================================================
# 2. pytest 执行器
# =====================================================================
_JUNIT_CASE_RE = re.compile(r"<testcase\b([^>]*?)(/>|>(.*?)</testcase>)", re.DOTALL)
_ATTR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*"([^"]*)"')


def _xml_unescape(text: str) -> str:
    return (text.replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&"))


def parse_junit_xml(xml_text: str) -> dict:
    """JUnit XML → {"case_id": "failed"|"passed"|"skipped"}（纯函数，CPU 可测）。

    case_id 的形态与 pytest 的节点 id 对齐成 `文件::用例`（见 `test_case_id`）。
    用正则而不是 xml.etree：JUnit 里嵌的 `<failure>` 正文常含未转义的内容
    （pytest 会把 assertion 的 repr 原样塞进去），XML 解析器会整棵树炸掉——
    而那时我们需要的是"哪条失败"，不是"XML 为什么不合法"。
    """
    out = {}
    for m in _JUNIT_CASE_RE.finditer(xml_text or ""):
        attrs = dict(_ATTR_RE.findall(m.group(1) or ""))
        name, cls = attrs.get("name"), attrs.get("classname")
        if not name:
            continue
        cid = test_case_id(cls or "", name)
        body = m.group(3) or ""
        if "<failure" in body or "<error" in body:
            out[cid] = "failed"
        elif "<skipped" in body:
            out[cid] = "skipped"
        else:
            out[cid] = "passed"
    return out


def test_case_id(classname: str, name: str) -> str:
    """JUnit 的 (classname, name) → 本模块的节点 id。

    pytest 的 classname 形如 `test_calc`（模块）或 `tests.test_calc.TestX`
    （类内用例），name 是用例函数名。L0 的 instance 只有**模块级**用例，故
    id = `<文件>::<函数>`；类内用例保留 `文件::类::函数` 形态（更具体，且与
    pytest 自己的节点 id 一致，方便人肉复核）。
    """
    cls = (classname or "").strip()
    fn = (name or "").strip()
    if not cls:
        return fn
    # classname 用点分隔：最后一段是模块（或类），前面是包路径
    parts = [p for p in cls.split(".") if p]
    if not parts:
        return fn
    # pytest 对模块级用例给 classname=<模块名>；对类内用例给 <模块名>.<类名>
    if len(parts) >= 2:
        return f"{parts[-2]}::{parts[-1]}::{fn}"
    return f"{parts[0]}::{fn}"


def _limit_resources(mem_mb: int):
    """子进程 exec 前设内存/文件描述符上限（仅 Linux；与 sandbox 同思路）。"""
    import resource
    cap = max(256, int(mem_mb)) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))


def run_tests(workdir: str, test_files, *, timeout: float = TEST_TIMEOUT_DEFAULT,
              mem_mb: int = TEST_MEM_MB_DEFAULT, env_extra: dict = None) -> dict:
    """在独立子进程里跑指定测试文件，返回逐用例结果。

    参数
        workdir   : 仓库副本根目录（`calc.py` 与 `test_calc.py` 所在处）
        test_files: 要跑的测试文件名列表（如 ["test_calc.py"]）
    返回
        {"ok": bool, "returncode", "timed_out", "duration",
         "results": {case_id: "passed"|"failed"|"skipped"}, "detail": str}

    【为什么在子进程里跑】与 sandbox.run_code 同一条理由：模型写的代码可能死循环/
    吃内存/`sys.exit()`——在生成 worker 进程内跑会把整个训练带走。这里是
    `python -m pytest` 的独立进程 + 超时 SIGKILL。
    【为什么用 JUnit XML 而不是解析 stdout】`-q` 的文本格式跨 pytest 版本会变
    （"3 passed" vs "3 passed, 1 warning"）；XML 的 `<testcase>` 是有 schema 的
    契约。这一条已经在 CPU 回归里用真实 pytest 输出锁死。
    """
    t0 = time.monotonic()
    res = {"ok": False, "returncode": None, "timed_out": False, "duration": 0.0,
           "results": {}, "detail": ""}
    files = [str(f) for f in (test_files or []) if str(f).strip()]
    if not files:
        res["detail"] = "没有测试文件"
        res["duration"] = time.monotonic() - t0
        return res
    xml_path = os.path.join(workdir, ".rlab_junit.xml")
    try:
        if os.path.exists(xml_path):
            os.unlink(xml_path)
    except OSError:
        pass
    cmd = [sys.executable, "-m", "pytest", *files,
           "-p", "no:cacheprovider", "-p", "no:randomly",
           "--no-header", "-q", f"--junit-xml={xml_path}"]
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["PYTHONPATH"] = workdir + (os.pathsep + env["PYTHONPATH"]
                                   if env.get("PYTHONPATH") else "")
    # 禁掉外部插件对行为的随机化（pytest-randomly 会改用例顺序→同一 patch 两次
    # 判分可能不同；L0 的 reward 必须对同输入确定）。
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.update(env_extra or {})
    preexec = (lambda: _limit_resources(mem_mb)) if sys.platform.startswith("linux") \
        else None
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                              timeout=timeout, env=env, preexec_fn=preexec)
        res["returncode"] = proc.returncode
        out = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
        res["detail"] = out[-TEST_OUTPUT_MAX_CHARS:]
    except subprocess.TimeoutExpired:
        res["timed_out"] = True
        res["detail"] = f"测试执行超时（{timeout:.0f}s）"
    except OSError as exc:
        res["detail"] = f"测试进程启动失败: {exc}"
    # XML 优先：即使超时/非零退出，只要 XML 已经写出来就按它判（超时前跑完的
    # 用例结论仍然有效——"一半跑过一半挂住"不该整组作废）。
    try:
        with open(xml_path, encoding="utf-8") as fh:
            res["results"] = parse_junit_xml(fh.read())
    except OSError:
        res["results"] = {}
    finally:
        try:
            os.unlink(xml_path)
        except OSError:
            pass
    res["ok"] = bool(res["results"]) and not res["timed_out"] and all(
        v == "passed" for v in res["results"].values())
    res["duration"] = time.monotonic() - t0
    return res


# =====================================================================
# 3. 判分（F2P / P2P）
# =====================================================================
def evaluate_patch(workdir: str, patch_text: str, spec: dict, *,
                   timeout: float = TEST_TIMEOUT_DEFAULT,
                   mem_mb: int = TEST_MEM_MB_DEFAULT) -> dict:
    """把一段 patch 应用到 workdir 副本并判分。纯 IO/子进程，无 torch。

    spec 需要 `f2p`（必需，非空）与 `p2p`（可为空）两个 case_id 列表。
    返回：
        {"apply_ok", "apply_stage", "f2p_pass", "f2p_failed", "p2p_broken",
         "reward", "acc", "format", "duration", "detail"}

    【口径三条（与 docs/17 §3/§4 对齐）】
      · `apply_ok` = **格式指标**：patch 落不下去 → acc 记 0 但单独上报；
      · `f2p` 必须**全过**（任一失败即 0 分）——L0 的"修好"是硬判据；
      · `p2p` 一个都不许坏（回归即 0 分）：这是"修 bug 不能拆台"的唯一约束，
        也是 L1 之后条件修复率的雏形。p2p 为空时视为不参与判分。
    """
    t0 = time.monotonic()
    out = {"apply_ok": False, "apply_stage": "empty", "f2p_pass": 0,
           "f2p_failed": [], "p2p_broken": [], "reward": -1.0, "acc": 0.0,
           "format": 0.0, "duration": 0.0, "detail": ""}
    f2p = [str(x) for x in (spec.get("f2p") or [])]
    p2p = [str(x) for x in (spec.get("p2p") or [])]
    if not f2p:
        out["detail"] = "spec.f2p 为空——没有失败测试就无法定义'修好'"
        out["duration"] = time.monotonic() - t0
        return out
    applied = check_patch(patch_text, workdir)
    out["apply_stage"] = applied["stage"]
    out["apply_ok"] = bool(applied["ok"])
    out["format"] = 1.0 if applied["ok"] else 0.0
    if not applied["ok"]:
        out["detail"] = f"apply 失败（{applied['stage']}）：{applied['detail']}"
        out["duration"] = time.monotonic() - t0
        return out
    res = run_tests(workdir, spec.get("test_files") or [], timeout=timeout,
                    mem_mb=mem_mb)
    results = res["results"]
    # 测试**根本跑不起来**（收集期报错/语法错）与"某条用例失败"必须分开：
    # 前者的 results 是空的，若按"没有失败即通过"判就会把语法错误算成满分——
    # 这是最贵的一种假分数。
    if not results:
        out["detail"] = ("测试未能产出任何用例结果（收集期失败/语法错误/超时）："
                         + (res["detail"] or "")[:800])
        out["duration"] = time.monotonic() - t0
        return out
    out["f2p_pass"] = sum(1 for c in f2p if results.get(c) == "passed")
    out["f2p_failed"] = [c for c in f2p if results.get(c) != "passed"]
    out["p2p_broken"] = [c for c in p2p if results.get(c) not in (None, "passed")]
    ok = not out["f2p_failed"] and not out["p2p_broken"]
    out["acc"] = 1.0 if ok else 0.0
    out["reward"] = 1.0 if ok else -1.0
    if not ok:
        bits = []
        if out["f2p_failed"]:
            bits.append(f"F2P 未过 {len(out['f2p_failed'])}/{len(f2p)}")
        if out["p2p_broken"]:
            bits.append(f"P2P 回归 {len(out['p2p_broken'])}/{len(p2p)}")
        out["detail"] = "；".join(bits) + "：" + (res["detail"] or "")[:800]
    out["duration"] = time.monotonic() - t0
    return out


def total_reward_swe_l0(spec: dict, patch_text: str, workdir_factory, *,
                        timeout: float = TEST_TIMEOUT_DEFAULT,
                        mem_mb: int = TEST_MEM_MB_DEFAULT, **unused) -> dict:
    """一条轨迹的奖励：把 patch 应用到**干净副本**后跑测试（±1 域）。

    `workdir_factory()` 必须每次返回一个**全新的**仓库副本路径（调用方负责
    清理）。同组内相同 patch 的去重在 `group_reward` 里做（docs/17 §3
    "同 instance 多采样（group）内去重相同 patch 只测一次"——测试执行是本项目
    最贵的一步，去重直接乘在吞吐上）。

    `**unused` 吸收 retool/math 那套 `completion_len/max_gen_tokens/
    overlong_shaping` 关键字——调用点（rollout/eval）是同一份循环，传参形态
    对齐才不必在每个调用点写分支；这些项在 L0 的 reward **不参与**（L0 是
    F2P 硬判据，长度项没有着力点：patch 长度与"修对没修对"无关）。
    """
    workdir = workdir_factory()
    try:
        res = evaluate_patch(workdir, patch_text, spec, timeout=timeout,
                             mem_mb=mem_mb)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"reward": res["reward"], "acc": res["acc"], "format": res["format"],
            "apply_ok": res["apply_ok"], "apply_stage": res["apply_stage"],
            "f2p_pass": res["f2p_pass"], "f2p_total": len(spec.get("f2p") or []),
            "p2p_broken": len(res["p2p_broken"]),
            "p2p_total": len(spec.get("p2p") or []),
            "detail": res["detail"], "duration": res["duration"]}


def patch_fingerprint(patch_text: str) -> str:
    """patch 的比对指纹（组内去重用）。忽略行尾空白与首尾空行——
    "少一个换行"不该让同一段修复被测两次（测试执行是最贵的一步）。"""
    norm = "\n".join(line.rstrip() for line in (patch_text or "").splitlines())
    return norm.strip()


def dedup_group_indices(patch_texts) -> dict:
    """组内去重：返回 {首次出现的下标: [该 patch 的所有下标]}（纯函数，CPU 可测）。

    docs/17 §3 的"group 内去重相同 patch 只测一次"的最小实现。空 patch
    （模型什么都没产出）**不合并**：它们各自都是 -1，合并省不到钱，却会让
    记录里"几条空轨迹"失真。
    """
    out, first_by_fp = {}, {}
    for idx, text in enumerate(patch_texts):
        if not (text or "").strip():
            out[idx] = [idx]
            continue
        fp = patch_fingerprint(text)
        if fp in first_by_fp:
            out[first_by_fp[fp]].append(idx)
        else:
            first_by_fp[fp] = idx
            out[idx] = [idx]
    return out


# =====================================================================
# 4. 数据加载（held-out 契约照旧：落盘复读，真去读 dev）
# =====================================================================
_SWE_L0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "datasets", "swe_l0")

_PREPARE_HINT = ("  先运行: python -m rlab.prepare_swe_l0\n"
                 "  （从 rlab/swe_l0_problems.py 的种子实测生成 F2P/P2P）")


def swe_l0_dir() -> str:
    return _SWE_L0_DIR


def _read_swe_rows(path: str) -> list:
    """读 L0 产物。行 schema（prepare_swe_l0 落盘）：

        {"Q": issue+定位代码的完整提示, "A": "F2P 全过 + P2P 不回归",
         "id", "files": {路径: 内容}, "tests": {路径: 内容},
         "f2p": [...], "p2p": [...], "test_files": [...], "source": ...}

    【为什么 Q 里已经含代码】L0 的定位是**给定**的（docs/17 §2.5）：
    提示 = issue 描述 + 目标文件的当前（buggy）内容 + 测试文件内容。
    整库进上下文是 L2 才有的问题，L0 天然放得下。
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError as exc:
                raise RuntimeError(f"[swe_l0] {path} 第 {n} 行不是合法 JSON：{exc}")
            if not isinstance(r, dict):
                raise RuntimeError(f"[swe_l0] {path} 第 {n} 行不是 JSON 对象")
            rows.append(r)
    return rows


def validate_swe_rows(rows: list, where: str = "") -> None:
    """fail-fast：Q/A 非空，files/tests 非空且路径安全，F2P 非空。

    【F2P 非空为什么是硬闸】没有失败测试的 instance，其"修复率"恒为 100%
    （任何 patch 都"通过了全部 F2P"）——这正是"精确 100%"那个系统性 bug 签名
    的数据侧入口。prepare 已经剔除过一遍，这里再拦一次（两道互不依赖的闸，
    任何一道被绕过都还会炸；与 search_qa 的 validate 同构）。
    """
    for i, r in enumerate(rows):
        tag = f"{where} 第 {i} 行"
        if not isinstance(r.get("Q"), str) or not r["Q"].strip():
            raise RuntimeError(f"[swe_l0] {tag} Q 缺失/非字符串")
        if not isinstance(r.get("A"), str) or not r["A"].strip():
            raise RuntimeError(f"[swe_l0] {tag} A 缺失/非字符串")
        for key in ("files", "tests"):
            val = r.get(key)
            if not isinstance(val, dict) or not val:
                raise RuntimeError(f"[swe_l0] {tag} {key} 必须是非空 dict（路径→内容）")
            for pth, content in val.items():
                if not sanitize_patch_path(pth) or sanitize_patch_path(pth) != pth:
                    raise RuntimeError(
                        f"[swe_l0] {tag} {key} 里的路径不安全/未归一: {pth!r}\n"
                        f"  规范：相对 POSIX 路径（如 'calc.py'），不含 a/ b/ 前缀、"
                        f"不含 .. 或绝对路径。")
                if not isinstance(content, str):
                    raise RuntimeError(f"[swe_l0] {tag} {key}[{pth!r}] 不是字符串")
        f2p = r.get("f2p")
        if not isinstance(f2p, list) or not f2p:
            raise RuntimeError(
                f"[swe_l0] {tag} f2p 为空——没有失败测试就无法定义'修好'"
                f"（这种行会让修复率恒为 100%）。重跑 prepare_swe_l0。")
        for key in ("f2p", "p2p", "test_files"):
            val = r.get(key)
            if val is not None and not isinstance(val, list):
                raise RuntimeError(f"[swe_l0] {tag} {key} 必须是列表")


def _load_one(path: str, where: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[swe_l0] {where} 缺失: {path}\n{_PREPARE_HINT}")
    rows = _read_swe_rows(path)
    if not rows:
        raise RuntimeError(f"[swe_l0] {where} 存在但为空: {path}\n{_PREPARE_HINT}")
    validate_swe_rows(rows, where)
    return rows


def load_swe_l0_dev(path: str = None) -> list:
    """held-out 评测池 = dev.jsonl（prepare 产物，绝不回落训练池）。"""
    p = path or os.path.join(_SWE_L0_DIR, "dev.jsonl")
    rows = _load_one(p, "swe_l0 dev.jsonl (held-out)")
    print(f"[swe_l0] dev.jsonl (held-out): {len(rows)} 个 instance")
    return rows


def load_swe_l0_train(path: str = None, dev_path: str = None) -> list:
    """训练池 = train.jsonl，**真去读 dev.jsonl 核对 id 不相交**。

    【为什么按 id 而不按题面】L0 的 instance id 是种子名（`l0_clamp_bounds`），
    题面（Q）里含代码、会比 id 更易因表述微调而变化；id 是 prepare 落盘的
    稳定契约。两条都不相交才放行（id 为主，Q 为辅）。
    """
    p = path or os.path.join(_SWE_L0_DIR, "train.jsonl")
    dp = dev_path or os.path.join(_SWE_L0_DIR, "dev.jsonl")
    rows = _load_one(p, "swe_l0 train.jsonl")
    if not os.path.exists(dp):
        raise FileNotFoundError(
            f"[swe_l0] held-out dev.jsonl 缺失: {dp}\n"
            f"  没有 held-out 就无法判断训练是否真的提升了泛化（docs/17 §4）。\n"
            f"{_PREPARE_HINT}")
    dev_rows = _load_one(dp, "swe_l0 dev.jsonl")
    dev_ids = {str(r.get("id") or "") for r in dev_rows}
    dev_qs = {r["Q"].strip() for r in dev_rows}
    kept = [r for r in rows
            if str(r.get("id") or "") not in dev_ids and r["Q"].strip() not in dev_qs]
    if len(kept) != len(rows):
        print(f"[swe_l0][警告] train.jsonl 含 {len(rows) - len(kept)} 条 held-out "
              f"instance——该文件不是 prepare_swe_l0 的干净产物（或被别的流程覆盖）。"
              f"已当场剔除 {len(rows)} -> {len(kept)}；建议重跑 prepare。", flush=True)
    seen, out = set(), []
    for r in kept:
        key = str(r.get("id") or r["Q"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if len(out) != len(kept):
        print(f"[swe_l0] 训练池按 id 去重: {len(kept)} -> {len(out)} 条")
    print(f"[swe_l0] 训练池: {len(out)} 个 instance（已核 dev 不相交；"
          f"held-out {len(dev_rows)} 个）", flush=True)
    return out


def spec_of(row: dict) -> dict:
    """从一行数据取出判分 spec（evaluate_patch / total_reward_swe_l0 的入参）。"""
    return {"f2p": list(row.get("f2p") or []),
            "p2p": list(row.get("p2p") or []),
            "test_files": list(row.get("test_files") or [])}


def materialize_repo(row: dict, dest: str) -> None:
    """把一行数据的仓库快照写到 `dest`（files + tests，全部按相对路径落盘）。

    目录创建是必需的：L0 种子目前都是平铺文件，但 schema 允许多级路径
    （`files: {"pkg/mod.py": ...}`），少了 makedirs 就会在 L1 迁移时炸。
    """
    os.makedirs(dest, exist_ok=True)
    for bucket in ("files", "tests"):
        for rel, content in (row.get(bucket) or {}).items():
            path = os.path.join(dest, *rel.split("/"))
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)


def make_workdir_factory(row: dict, root: str = None):
    """返回一个"每次调用给一个干净仓库副本"的工厂（判分用）。

    每次都是**全新副本**：apply 会原地改文件，复用同一目录会让第二次判分
    看到上一次的 patch（"同组 8 条只测一条"那类错误的最隐蔽形态——结果看起来
    正常，只是都等于第一条）。
    """
    import uuid
    base = root or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)

    def _make() -> str:
        path = os.path.join(base, f"rlab_swe_{uuid.uuid4().hex[:16]}")
        materialize_repo(row, path)
        return path

    return _make
