# -*- coding: utf-8 -*-
"""rlab/eval.py — 评测统一入口（委托给根目录已验证的 eval_vllm.py 调度器）。

评测协议（公共实验协议，附录B）：GSM8K test N=300 seed=42，vLLM 批量贪心。
注意教训：标签用 name=path 形式或依赖调度器的路径末3段去重，防同名 checkpoint 覆盖。

用法（--models 空格分隔多值，或整体加引号）：
    python -m rlab.eval --models grpo300=./rlab_out/grpo/step_300 dapo300=./rlab_out/dapo/step_300
    python -m rlab.eval --models "grpo300=./x, dapo300=./y"   # 引号+逗号亦可
    python -m rlab.eval --models ./rlab_out/grpo/step_300      # 不带 name 自动取路径末3段
    python -m rlab.eval --retool --models retool300=./rlab_out/retool/step_300
    python -m rlab.eval --algo retool_math --models m20=./rlab_out/retool_math/step_20  # 自动切 dapo_math+boxed
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", nargs="+",
                    help="一个或多个模型，支持 name=path 或纯 path；"
                         "空格分隔（各自成项）或整个加引号逗号分隔均可")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpus", default="0", help="评测用卡；默认 0 号（与生成端共卡需错峰）")
    ap.add_argument("--per_gpu", type=int, default=1,
                    help="每卡同时跑的模型进程数；1=卡内串行（默认，防OOM）")
    ap.add_argument("--split", default="test", choices=("test", "train"),
                    help="test=held-out（dapo_math=dev.jsonl；gsm8k=test split）；train=训练池抽样(过拟合诊断)")
    ap.add_argument("--base_path", default="/root/Qwen2.5-3B")
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--retool", action="store_true",
                    help="阶段2：多轮代码交织评测（兼容旧 flag，等价 --algo retool）")
    ap.add_argument("--algo", type=str, default=None,
                    help="算法名：grpo/retool/retool_math；指定后自动决定 prompt/预算/奖励口径与数据集")
    ap.add_argument("--eval_task", type=str, default=None, choices=["gsm8k", "dapo_math"],
                    help="评测数据集；None=自动（retool_math→dapo_math，其余→gsm8k）")
    args = ap.parse_args()

    # 兼容旧 --retool
    algo = args.algo
    if algo is None and args.retool:
        algo = "retool"
    models = ",".join(m.strip() for m in args.models if m.strip())

    cmd = [sys.executable, os.path.join(ROOT, "eval_vllm.py"),
           "--tuned", models, "--n", str(args.n), "--seed", str(args.seed),
           "--gpus", args.gpus, "--per_gpu", str(args.per_gpu), "--split", args.split,
           "--base_path", args.base_path]
    if args.skip_base:
        cmd.append("--skip_base")
    if args.retool:
        cmd.append("--retool")
    if algo is not None:
        cmd += ["--algo", algo]
    if args.eval_task is not None:
        cmd += ["--eval_task", args.eval_task]
    print("[eval]", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
