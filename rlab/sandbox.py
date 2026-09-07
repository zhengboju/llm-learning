# -*- coding: utf-8 -*-
"""rlab/sandbox.py — 阶段2：Python 代码沙箱（subprocess 隔离执行）。

替代 Auto_Program 玩具实现里 `exec()` + signal.alarm 的方案：
  - exec() 在生成 worker 进程内直接执行不可信代码：崩溃/死循环/内存爆炸会带走
    整个 vLLM 训练进程；signal.alarm 只能在主线程用（线程环境必炸，本项目教训）。
  - 这里改为独立子进程 `python -I user_code.py`：隔离崩溃 + 超时 SIGKILL +
    内存上限（Linux RLIMIT_AS，best-effort）+ stdout/stderr 截断。

【局限说明（教学规模可接受，如实记录）】
  - 网络并未真正禁止：子进程仍可开 socket。真禁网需要容器/seccomp/命名空间；
    教学规模下用 `-I -E` 隔离模式（不读用户 site、不带代理等环境变量）+ 不给
    任何凭证，已足够挡掉绝大多数误用。真机训练若需硬禁网，应把沙箱换成
    `bubblewrap`/`firejail` 或每轮起临时容器（本项目不引入该复杂度）。
  - 内存上限仅 Linux 生效（resource 模块），Windows 冒烟环境跳过（best-effort）。

用法（生成端 worker 每轮代码调用一次，串行安全）：
    from rlab.sandbox import run_code
    res = run_code(code, timeout=cfg["sandbox_timeout"], mem_mb=cfg["sandbox_mem_mb"])
    # res = {ok, returncode, timed_out, duration, display, stderr}
    # display = 成功时的 stdout（截断）或 "Error! <类型>: <摘要>"
"""

import os
import subprocess
import sys
import tempfile
import time

_IS_LINUX = sys.platform.startswith("linux")


def _limit_resources(mem_mb: int):
    """Linux 上在子进程 exec 前调用：限制虚拟内存（RLIMIT_AS 含堆/栈/映射）。
    用 RLIMIT_AS 而非 RLIMIT_DATA：Python 的 import 与 JIT 也会吃地址空间。"""
    import resource
    cap = max(64, mem_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    # 同时限制文件描述符数量，挡"疯狂开文件/连接"的常见死循环
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def run_code(code: str, *, timeout: float = 5.0, mem_mb: int = 256,
             max_chars: int = 500) -> dict:
    """在独立子进程执行一段 python 代码，返回结构化结果。

    Args:
        code: 代码文本（生成 worker 已从围栏提取并 strip）。
        timeout: 秒；超时即 SIGKILL 整个子进程组。
        mem_mb: 内存上限（仅 Linux 生效）。
        max_chars: stdout/stderr 截断长度（防输出炸弹）。
    """
    t0 = time.time()
    res = {"ok": False, "returncode": None, "timed_out": False,
           "duration": 0.0, "display": "Error! Empty code block", "stderr": ""}
    if not code or not code.strip():
        return res

    with tempfile.TemporaryDirectory(prefix="rlab_sandbox_") as td:
        path = os.path.join(td, "user_code.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        preexec = (lambda: _limit_resources(mem_mb)) if _IS_LINUX else None
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-E", path],
                capture_output=True, text=True, timeout=timeout, preexec_fn=preexec)
            res["returncode"] = proc.returncode
            res["ok"] = proc.returncode == 0
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            res["stderr"] = err[-max_chars:]
            if proc.returncode == 0:
                res["display"] = out[-max_chars:] if out else "Error! No output"
            else:
                # 崩溃/异常：只取最后几行错误摘要（traceback 太长）
                tail = err.splitlines()[-1] if err.splitlines() else f"exit {proc.returncode}"
                res["display"] = f"Error! {tail[:300]}"
        except subprocess.TimeoutExpired:
            res["timed_out"] = True
            res["display"] = "Error! The Code Execution timeout!"
        except OSError as e:
            res["display"] = f"Error! Sandbox launch failed: {e}"
        finally:
            res["duration"] = time.time() - t0
    return res
