# -*- coding: utf-8 -*-
# 定量评测：GSM8K **test** split 上对比 BASE vs 各 RL checkpoint
# train 在训练时被反复采样过(有泄漏)，必须用 held-out test 测泛化
# 用法:
#   双卡全自动(单命令跑完6模型): python eval_gsm8k_test.py --n 300 --gpus 0,1 --workers 3 --batch_size 8 \
#       --tuned grpo200=/path/grpo,dapo200=/path/dapo,rfpp100=/path/100,rfpp200=/path/200,rfpp300=/path/300
#   --gpus auto(默认)=所有可见卡；--workers=每卡并发模型数；退化成老行为: --gpus 0 --workers 1 --batch_size 1
import json, os, re, random, argparse, threading
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify, ExprExtractionConfig

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100, help="评测题数(test共1319，建议100-200)")
parser.add_argument("--tuned", default="./step_200", help="逗号分隔的一个或多个 checkpoint，可用 name=path 显式命名(推荐)；否则按路径末3段自动命名，同名自动加 #2/#3 后缀；传空字符串则只评 BASE")
parser.add_argument("--do_sample", action="store_true", help="开启采样(temperature=0.9)；默认greedy更稳定")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="eval_gsm8k_test_result.json", help="结果 json 输出路径(并发跑时每路指定不同文件，防互踩)")
parser.add_argument("--skip_base", action="store_true", help="跳过 BASE，只评 --tuned(并发第二路用，BASE 只跑一次)")
parser.add_argument("--workers", type=int, default=3, help="每卡并发评测的模型数(线程级，同CUDA上下文多流真并行；3B bf16每路约8G，H20上每卡3路≈25G)")
parser.add_argument("--gpus", type=str, default="auto", help="使用的卡：auto=所有可见卡(默认)，或显式如 0,1；模型按 round-robin 自动拆分到各卡")
parser.add_argument("--batch_size", type=int, default=8, help="每模型每轮 generate 的题数(greedy下与逐题数学等价，只提吞吐不改结果)")
parser.add_argument("--attn", type=str, default="sdpa", help="attention 实现：sdpa(默认快) / eager(参照路径，批量结果异常时切 eager 排查)")
parser.add_argument("--show", type=int, default=0, help="打印每模型第一批的前N个原始回答+单题得分(排障开箱验货用)")
args = parser.parse_args()

base_path = "/root/Qwen2.5-3B"
N = args.n
do_sample = args.do_sample
BATCH = max(1, args.batch_size)

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

# ---------- 与训练完全一致的奖励函数（0/1 化便于统计） ----------
def reward_correct(answer, ground_truth):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0: return 0.0
    try:
        # 【多线程适配·并行版全0的根因】math_verify 默认 parsing_timeout/timeout_seconds=5，
        # 内部用 signal.alarm 实现——signal 只能在主线程用，worker 线程里必抛
        # ValueError("signal only works in main thread")，被下面 except 吞掉 => 所有题 a=0。
        # 库方建议线程环境置 None（自管超时）；GSM8K 只解析短数字串，无挂起风险。
        ans = parse(nums[-1], extraction_config=[ExprExtractionConfig()], parsing_timeout=None)
        gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()], parsing_timeout=None)
        return 1.0 if verify(ans, gt, timeout_seconds=None) else 0.0
    except Exception:
        # 双保险：真出异常就退化为纯数值比较
        try:
            gt_nums = re.findall(pattern, ground_truth.replace(",", ""))
            if gt_nums and abs(float(nums[-1]) - float(gt_nums[-1])) < 1e-6:
                return 1.0
        except Exception:
            pass
        return 0.0

def reward_format(answer):
    if "reasoning process here" in answer.lower():  # 与新训练一致：占位符判格式失败
        return 0.0
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.0 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else 0.0

# ---------- 加载 GSM8K **test** split ----------
print("[1/3] 加载 GSM8K test split ...")
test_data = None
# 优先 modelscope（hf 不通）
# trust_remote_code=True：modelscope/gsm8k 是脚本型数据集，需要显式信任其仓库代码（安全已知的可信镜像）
try:
    from modelscope.msdatasets import MsDataset
    for did in ("modelscope/gsm8k",):
        try:
            ds = MsDataset.load(did, subset_name="main", split="test", trust_remote_code=True)
            if len(ds) > 0:
                test_data = [{"Q": x["question"], "A": x["answer"]} for x in ds]
                print(f"  modelscope {did}: {len(test_data)} 题")
                break
        except Exception as e:
            print(f"  {did} 失败: {e}")
except Exception as e:
    print(f"  modelscope 不可用: {e}")

# 兜底：hf（可能已缓存）
if test_data is None:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    test_data = [{"Q": x["question"], "A": x["answer"].split("####")[-1].strip()} for x in ds]
    print(f"  datasets: {len(test_data)} 题")

assert test_data is not None and len(test_data) > 0, "test split 加载失败"
random.seed(args.seed)
sample = random.sample(test_data, min(N, len(test_data)))
print(f"  固定 seed={args.seed}，抽 {len(sample)} 题")

# ---------- 待评模型列表（标签唯一，禁止静默覆盖） ----------
def _auto_label(p, used):
    parts = [x for x in p.strip().rstrip("/").split("/") if x not in ("", ".")]
    base = "_".join(parts[-3:]) if len(parts) >= 3 else "_".join(parts)
    base = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_.-]+", "_", base) or "ckpt"
    label, i = base, 2
    while label in used:
        label = f"{base}#{i}"
        i += 1
    return label


models = []
_used = set()
if not args.skip_base:
    models.append(("BASE", base_path))
    _used.add("BASE")
for _item in args.tuned.split(","):
    _item = _item.strip()
    if not _item:
        continue
    if "=" in _item:  # 显式命名：--tuned grpo200=/path/to/ckpt
        _name, _p = _item.split("=", 1)
        _name, _p = _name.strip(), _p.strip()
        if _name in _used:  # 显式名撞车也加后缀，禁止静默覆盖
            _name = _auto_label(_name, _used)
    else:
        _p = _item
        _name = _auto_label(_p, _used)
    _used.add(_name)
    models.append((_name, _p))

assert len(models) > 0, "--tuned 为空且 --skip_base，同时没有可评模型"
print(f"[2/3] 生成并评分 ... （模型数={len(models)}，每卡并发 workers={args.workers}，每轮批量 batch_size={BATCH}）")

# ---------- 单个模型的评测（线程入口：自带 tokenizer+model，与串行版数学等价） ----------
print_lock = threading.Lock()


def build_prompt(tok, q):
    return tok.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def eval_one(name, path, gpu):
    dev = f"cuda:{gpu}"
    torch.cuda.set_device(dev)  # 线程级当前设备 + 全显式 device，跨线程不串卡
    with print_lock:
        print(f"  [线程启动][GPU{gpu}] {name}: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # 批量生成必须左padding；单题ids与串行版逐字符一致
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, _attn_implementation=args.attn).to(dev).eval()
    try:
        acc, fmt, both, n_valid = 0.0, 0.0, 0.0, 0
        shown = []
        next_milestone = 20
        with torch.inference_mode():
            for s in range(0, len(sample), BATCH):
                chunk = sample[s:s + BATCH]
                prompts = [build_prompt(tokenizer, item["Q"]) for item in chunk]
                # 与串行版相同的逐题分词，只是拼成一个 batch（左pad不改变每题真实ids）
                enc = tokenizer(prompts, return_tensors="pt", padding=True)
                base_len = enc["input_ids"].shape[1]  # 左pad后全行等长Lmax；生成部分一律从 base_len 开始（注意不是每题真实长度Li！）
                out = model.generate(
                    input_ids=enc["input_ids"].to(dev),
                    attention_mask=enc["attention_mask"].to(dev),
                    max_new_tokens=512,
                    do_sample=do_sample,
                    temperature=0.9 if do_sample else None,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id)
                # 停止词用模型默认(与串行版一致，Qwen即151645 <|im_end|>)，绝不显式覆盖：
                # tokenizer.eos_token_id=151643(<|endoftext|>)≠模型默认，覆盖会导致冲过</answer>继续 rambling 到512 token
                _eos = model.generation_config.eos_token_id
                stop_ids = set(_eos if isinstance(_eos, list) else [_eos])
                stop_ids |= {tokenizer.eos_token_id, tokenizer.pad_token_id}
                for i, item in enumerate(chunk):
                    row_ids = out[i][base_len:].tolist()
                    # 去掉 batch 补齐的 pad 尾（greedy 到 EOS 即停，pad 只在 EOS 之后）
                    for _k, _t in enumerate(row_ids):
                        if _t in stop_ids:
                            row_ids = row_ids[:_k]
                            break
                    ans = tokenizer.decode(row_ids, skip_special_tokens=True)
                    if len(ans.strip()) == 0:
                        continue
                    n_valid += 1
                    gt = item["A"].split("####")[-1].strip()
                    a = reward_correct(ans, gt)
                    f = reward_format(ans)
                    acc += a; fmt += f; both += (a == 1.0 and f == 1.0)
                    if s == 0 and len(shown) < args.show:
                        shown.append((a, f, ans))
                done = min(s + BATCH, len(sample))
                if done >= next_milestone or done == len(sample):
                    with print_lock:
                        print(f"    {name} {done}/{len(sample)}  acc_sofar={acc/max(1, n_valid):.3f}")
                    while next_milestone <= done:
                        next_milestone += 20
                if shown and s == 0:  # 只打第一批，开箱验货
                    with print_lock:
                        print(f"  ---- {name} 首{len(shown)}个回答 ----")
                        for _a, _f, _t in shown:
                            print(f"  [a={_a:.0f} f={_f:.0f}] {_t[:400]}")
                    shown.clear()
        return name, {
            "acc": acc / n_valid if n_valid else 0,
            "fmt": fmt / n_valid if n_valid else 0,
            "both": both / n_valid if n_valid else 0,
            "n": n_valid,
        }
    finally:
        del model, tokenizer
        with torch.cuda.device(gpu):
            torch.cuda.empty_cache()


# ---------- 多卡自动拆分：模型按 round-robin 分到各卡，同进程线程级并发 ----------
# 单进程多线程(而非多进程)：同CUDA上下文多流真并行，无spawn/IPC开销，结果字典直接共享
if args.gpus.strip().lower() == "auto":
    gpus = list(range(torch.cuda.device_count()))
else:
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
assert len(gpus) > 0, "无可用 GPU（torch.cuda.device_count()=0，检查驱动/CUDA_VISIBLE_DEVICES）"
# 主线程逐卡初始化一次 CUDA 上下文，再起工作线程
for _d in gpus:
    with torch.cuda.device(_d):
        torch.cuda.init()
plan = {}
for _i, (_name, _p) in enumerate(models):
    plan.setdefault(gpus[_i % len(gpus)], []).append(_name)
print("  任务分配: " + "；".join(f"GPU{d}<-{','.join(ns)}" for d, ns in plan.items()))
pool_size = min(max(1, args.workers) * len(gpus), len(models))
results = {}
with ThreadPoolExecutor(max_workers=pool_size) as ex:
    futs = [(name, ex.submit(eval_one, name, path, gpus[i % len(gpus)]))
            for i, (name, path) in enumerate(models)]
    for name, fut in futs:  # 按提交顺序回收，表格顺序稳定
        n, r = fut.result()
        results[n] = r

print("\n[3/3] 结果（GSM8K test，N=%d）" % len(sample))
print(f"{'模型':<16}{'准确率':>10}{'格式率':>10}{'双达标':>10}{'有效样本':>10}")
for name, r in results.items():
    print(f"{name:<16}{r['acc']*100:>9.1f}%{r['fmt']*100:>9.1f}%{r['both']*100:>9.1f}%{r['n']:>10}")

# 存一份 json 备查
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已存 {args.out}")
