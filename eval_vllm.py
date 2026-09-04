# -*- coding: utf-8 -*-
# vLLM 多模型多进程自动调度器：单命令跑完所有模型
#   - 每个模型 = 一个独立子进程跑 eval_vllm_one.py（vLLM显存随进程退出干净释放，多进程是正确姿势）
#   - 模型 round-robin 分到各卡，每卡并发 --per_gpu 个进程，--gpu_mem 自动= 0.78/per_gpu
#   - 全部结束后自动合表（失败的不阻塞其余，最后汇总报告）
# 用法:
#   python eval_vllm.py --n 300 --gpus 0,1 --per_gpu 3 \
#       --tuned grpo200=/path/grpo,dapo200=/path/dapo,rfpp100=/path/100,rfpp200=/path/200,rfpp300=/path/300
#   （BASE 默认评；--skip_base 跳过；--gpus auto=全部可见卡）
import argparse, json, os, queue, re, subprocess, sys, threading, time

parser = argparse.ArgumentParser()
parser.add_argument("--tuned", default="", help="逗号分隔 checkpoint，支持 name=path；空=只评BASE")
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpus", type=str, default="auto", help="auto=所有可见卡，或 0,1")
parser.add_argument("--per_gpu", type=int, default=1,
                    help="每卡同时跑的模型进程数；1=卡内串行（推荐，防OOM）")
parser.add_argument("--gpu_mem", type=float, default=None, help="单进程vLLM显存占比，默认自动=0.78/per_gpu")
parser.add_argument("--skip_base", action="store_true")
parser.add_argument("--base_path", default="/root/Qwen2.5-3B")
parser.add_argument("--show", type=int, default=0)
parser.add_argument("--split", default="test", choices=["test", "train"], help="train=训练集内抽样(过拟合诊断)")
parser.add_argument("--out", default="eval_vllm_all.json", help="合并结果json")
args = parser.parse_args()

base_path = args.base_path
GPU_MEM = args.gpu_mem if args.gpu_mem is not None else round(0.78 / max(1, args.per_gpu), 3)


def _auto_label(p, used):
    parts = [x for x in p.strip().rstrip("/").split("/") if x not in ("", ".")]
    base = "_".join(parts[-3:]) if len(parts) >= 3 else "_".join(parts)
    base = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_.-]+", "_", base) or "ckpt"
    label, i = base, 2
    while label in used:
        label = f"{base}#{i}"
        i += 1
    return label


models, _used = [], set()
if not args.skip_base:
    models.append(("BASE", base_path)); _used.add("BASE")
for item in args.tuned.split(","):
    item = item.strip()
    if not item:
        continue
    if "=" in item:
        nm, p = (x.strip() for x in item.split("=", 1))
        if nm in _used:
            nm = _auto_label(nm, _used)
    else:
        p, nm = item, _auto_label(item, _used)
    _used.add(nm); models.append((nm, p))
assert models, "没有可评模型"

gpus = list(range(os.cpu_count() and __import__("torch").cuda.device_count())) \
    if args.gpus.strip().lower() == "auto" else [int(x) for x in args.gpus.split(",") if x.strip()]
assert gpus, "无可用GPU"
print(f"[调度] 模型数={len(models)}，GPU={gpus}，每卡并发={args.per_gpu}，单进程gpu_mem={GPU_MEM}")

one_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_vllm_one.py")
# 分配：模型 round-robin 均匀摊到各 GPU 队列；每 GPU 的任务严格串行处理
# （同卡多 vLLM 实例并发是 OOM/内存剖析竞态的重灾区，默认 per_gpu=1 卡内串行）
gpu_queues = {g: queue.Queue() for g in gpus}
for i, (nm, _) in enumerate(models):
    gpu_queues[gpus[i % len(gpus)]].put((i, nm))
print("  分配: " + "；".join(
    f"GPU{g}<-{[nm for _, nm in gpu_queues[g].queue]}" for g in gpus))

results = {}
failures = []
lock = threading.Lock()   # per_gpu>1 时同卡多线程共享 results/failures


def _assert_local_dir(p: str, what: str):
    """路径形式的参数必须在启动前确认存在（否则 transformers 会把它当 HF repo id，
    抛一堆 HFValidationError，难排查）。repo id 形式的名字（含 / 但不以 ./ ~ / 开头）跳过。"""
    if p.startswith(("./", ".\\", "/", "~")):
        exp = os.path.expanduser(p)
        if not os.path.isdir(exp):
            raise SystemExit(
                f"[错误] {what} 路径不存在: {p}\n"
                f"  提示: 训练端把 checkpoint 存到 <启动目录>/{'{out_dir}'}/<algo>/step_N，"
                f"请核对训练时的 cwd（可用: find ~ -maxdepth 5 -type d -name 'step_*' 2>/dev/null）")


def run_one(gpu, idx, name, path):
    _assert_local_dir(path, f"模型 {name}")
    out_json = f"eval_v_{re.sub(r'[^0-9A-Za-z_.-]+', '_', name)}.json"
    time.sleep(idx * 3)  # 错峰启动：避免同卡多实例同时剖析显存（vLLM 0.12 的内存自洽断言会撞车）
    cmd = [sys.executable, one_py, "--model", path, "--name", name,
           "--n", str(args.n), "--seed", str(args.seed), "--split", args.split,
           "--gpu_mem", str(GPU_MEM), "--out", out_json]
    if args.show:
        cmd += ["--show", str(args.show)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    # 竞态兜底：同卡实例退出释放显存撞上另一实例的初始化剖析 → AssertionError。
    # 该失败只发生在启动窗口期，冷却后重试几乎必成，最多试3次。
    for attempt in range(1, 4):
        print(f"  [启动][GPU{gpu}] {name} -> {out_json}" + (f"（第{attempt}次重试）" if attempt > 1 else ""))
        proc = subprocess.run(cmd, env=env)
        if proc.returncode == 0 and os.path.exists(out_json):
            with lock:
                with open(out_json, encoding="utf-8") as f:
                    results.update(json.load(f))
            return
        with lock:
            print(f"  [失败][GPU{gpu}] {name} (exit={proc.returncode})，第{attempt}次")
        time.sleep(10 * attempt)
    with lock:
        failures.append((name, gpu, proc.returncode))


def gpu_worker(gpu):
    """每 GPU 一条工作线程：从本卡队列逐个取任务，卡内严格串行。
    per_gpu>1 时同卡起多条线程（并发，旧模式，易 OOM，不推荐）。"""
    while True:
        try:
            idx, name = gpu_queues[gpu].get_nowait()
        except queue.Empty:
            return
        _, path = models[idx]
        run_one(gpu, idx, name, path)


threads = [threading.Thread(target=gpu_worker, args=(g,))
           for g in gpus for _ in range(args.per_gpu)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"\n{'='*60}\n[总表]（GSM8K test，N={args.n}，seed={args.seed}）")
print(f"{'模型':<16}{'准确率':>10}{'格式率':>10}{'双达标':>10}{'有效样本':>10}")
for name, r in results.items():
    print(f"{name:<16}{r['acc']*100:>9.1f}%{r['fmt']*100:>9.1f}%{r['both']*100:>9.1f}%{r['n']:>10}")
if failures:
    print(f"\n[警告] {len(failures)} 个模型失败: {[(n, f'GPU{g}', rc) for n, g, rc in failures]}")
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已存 {args.out}（单模型明细在各 eval_v_*.json）")
