# -*- coding: utf-8 -*-
"""rlab/eval.py — 评测统一入口（委托给根目录已验证的 eval_vllm.py 调度器）。

评测协议（公共实验协议，附录B）：GSM8K test N=300 seed=42，vLLM 批量贪心。
注意教训：标签用 name=path 形式或依赖调度器的路径末3段去重，防同名 checkpoint 覆盖。

用法：
    python -m rlab.eval --models grpo=./rlab_out/grpo/step_300 dapo=./rlab_out/dapo/step_300
    python -m rlab.eval --models ./rlab_out/grpo/step_300   # 不带 name 自动取路径末3段
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="逗号分隔，支持 name=path 或纯 path")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpus", default="0", help="评测用卡；默认 0 号（与生成端共卡需错峰）")
    ap.add_argument("--per_gpu", type=int, default=3)
    ap.add_argument("--split", default="test", choices=("test", "train"))
    ap.add_argument("--base_path", default="/root/Qwen2.5-3B")
    ap.add_argument("--skip_base", action="store_true")
    args = ap.parse_args()

    cmd = [sys.executable, os.path.join(ROOT, "eval_vllm.py"),
           "--tuned", args.models, "--n", str(args.n), "--seed", str(args.seed),
           "--gpus", args.gpus, "--per_gpu", str(args.per_gpu), "--split", args.split,
           "--base_path", args.base_path]
    if args.skip_base:
        cmd.append("--skip_base")
    print("[eval]", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
