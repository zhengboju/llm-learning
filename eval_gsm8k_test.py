# -*- coding: utf-8 -*-
# 定量评测：GSM8K **test** split 上对比 BASE vs 各 RL checkpoint
# train 在训练时被反复采样过(有泄漏)，必须用 held-out test 测泛化
# 用法:
#   单卡3模型并行+8题批量: CUDA_VISIBLE_DEVICES=0 python eval_gsm8k_test.py --n 300 --workers 3 --batch_size 8 \
#       --tuned grpo200=/path/grpo,dapo200=/path/dapo
#   退化成老行为(串行逐题): --workers 1 --batch_size 1
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
parser.add_argument("--workers", type=int, default=3, help="同进程内并发评测的模型数(线程级，同CUDA上下文多流真并行；3B bf16每路约8G，H20上3路≈25G)")
parser.add_argument("--batch_size", type=int, default=8, help="每模型每轮 generate 的题数(greedy下与逐题数学等价，只提吞吐不改结果)")
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
        ans = parse(nums[-1], extraction_config=[ExprExtractionConfig()])
        gt = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
        return 1.0 if verify(ans, gt) else 0.0
    except Exception:
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
workers = max(1, min(args.workers, len(models)))
print(f"[2/3] 生成并评分 ... （模型数={len(models)}，同卡并发 workers={workers}，每轮批量 batch_size={BATCH}）")

# ---------- 单个模型的评测（线程入口：自带 tokenizer+model，与串行版数学等价） ----------
print_lock = threading.Lock()


def build_prompt(tok, q):
    return tok.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def eval_one(name, path):
    with print_lock:
        print(f"  [线程启动] {name}: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # 批量生成必须左padding；单题ids与串行版逐字符一致
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").cuda().eval()
    try:
        acc, fmt, both, n_valid = 0.0, 0.0, 0.0, 0
        next_milestone = 20
        with torch.inference_mode():
            for s in range(0, len(sample), BATCH):
                chunk = sample[s:s + BATCH]
                prompts = [build_prompt(tokenizer, item["Q"]) for item in chunk]
                # 与串行版相同的逐题分词，只是拼成一个 batch（左pad不改变每题真实ids）
                enc = tokenizer(prompts, return_tensors="pt", padding=True)
                plen_list = enc["attention_mask"].sum(dim=1).tolist()
                out = model.generate(
                    input_ids=enc["input_ids"].to("cuda"),
                    attention_mask=enc["attention_mask"].to("cuda"),
                    max_new_tokens=512,
                    do_sample=do_sample,
                    temperature=0.9 if do_sample else None,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
                for i, item in enumerate(chunk):
                    plen = int(plen_list[i])
                    row_ids = out[i][plen:].tolist()
                    # 去掉 batch 补齐的 pad 尾（greedy 到 EOS 即停，pad 只在 EOS 之后）
                    eos_id = tokenizer.eos_token_id
                    if eos_id is not None and eos_id in row_ids:
                        row_ids = row_ids[:row_ids.index(eos_id)]
                    ans = tokenizer.decode(row_ids, skip_special_tokens=True)
                    if len(ans.strip()) == 0:
                        continue
                    n_valid += 1
                    gt = item["A"].split("####")[-1].strip()
                    a = reward_correct(ans, gt)
                    f = reward_format(ans)
                    acc += a; fmt += f; both += (a == 1.0 and f == 1.0)
                done = min(s + BATCH, len(sample))
                if done >= next_milestone or done == len(sample):
                    with print_lock:
                        print(f"    {name} {done}/{len(sample)}  acc_sofar={acc/max(1, n_valid):.3f}")
                    while next_milestone <= done:
                        next_milestone += 20
        return name, {
            "acc": acc / n_valid if n_valid else 0,
            "fmt": fmt / n_valid if n_valid else 0,
            "both": both / n_valid if n_valid else 0,
            "n": n_valid,
        }
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()


# 主线程先初始化一次 CUDA 上下文，再起工作线程（同进程多流真并行；多进程只是分时片）
torch.cuda.init()
results = {}
with ThreadPoolExecutor(max_workers=workers) as ex:
    futs = [(name, ex.submit(eval_one, name, path)) for name, path in models]
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
