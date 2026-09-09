# -*- coding: utf-8 -*-
"""rlab/data.py — 数据集加载。

统一返回 QA 列表：[{"Q": 问题文本, "A": 标准答案文本(用于 reward 比对)}]。
默认走 modelscope（HF 及其镜像在训练机上网络不通），失败才回落 HF datasets。
fixture 模式供 CPU 冒烟测试，不触网。
"""

import json
import os

# 数据源选择：默认 ms（训练机 HF 网络不通）；RLAB_DATA_SOURCE=hf 可强制只走 HF，
# 其他值一律先 modelscope、失败回落 HF。
DATA_SOURCE = os.environ.get("RLAB_DATA_SOURCE", "ms").lower()

FIXTURE_QAS = [
    {"Q": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "A": "72"},
    {"Q": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "A": "10"},
    {"Q": "A deep-sea monster rises up once every 8 years to eat 2, but a village offers it 3 fish this time. How many fish will it have eaten after 8 more years?",
     "A": "5"},
    {"Q": "Ken had fifty pairs of shoes. If he sold half of his pairs, how many individual shoes did he remain with?",
     "A": "50"},
] * 8  # 32 条，够冒烟跑若干组


def load_qas(task: str = "gsm8k", fixture: bool = False):
    if fixture or task == "fixture":
        return list(FIXTURE_QAS)
    if task == "gsm8k":
        return load_gsm8k_train()
    if task in ("dapo_math", "dapo-math-17k", "math_dapo"):
        return load_dapo_math_train()
    raise KeyError(f"未知任务 {task!r}（阶段2/3 扩展 retool/search 任务时在此注册）")


def _patch_verification_mode():
    """兼容垫片（2026-09-04 真机实测两轮）：
    pod 里老 modelscope 仍向 datasets 传 verification_mode，但 datasets>=3 已删除
    该参数，导致 MsDataset.load 在 split 生成成功后死于 TypeError。
    第一轮垫片只打了 datasets.builder.Builder——datasets>=3 主类已改名
    DatasetBuilder，且 modelscope 常把 load_dataset 早绑定到自己模块里，
    所以没打中。本轮三处全打：
      1) datasets 侧所有 builder 类的 as_dataset（实例方法走 MRO，类上打必生效）；
      2) datasets.load_dataset 本体；
      3) 已加载的 modelscope.* 模块里早绑定的同名引用。
    该参数只控制校验失败时抛错还是警告，丢掉行为无损。"""
    import functools

    def _wrap(fn):
        @functools.wraps(fn)
        def w(*a, **k):
            k.pop("verification_mode", None)
            return fn(*a, **k)
        w._rlab_patched = True
        return w

    applied = []
    # 1) builder 类（新旧类名都试）
    try:
        from datasets import builder as _bd
        for _cls_name in ("DatasetBuilder", "Builder", "GeneratorBasedBuilder"):
            _cls = getattr(_bd, _cls_name, None)
            if _cls is not None and not getattr(_cls.as_dataset, "_rlab_patched", False):
                try:
                    _cls.as_dataset = _wrap(_cls.as_dataset)
                    applied.append(f"datasets.builder.{_cls_name}.as_dataset")
                except Exception:
                    pass
    except Exception:
        pass
    # 2) datasets.load_dataset 本体
    try:
        import datasets as _ds
        if callable(getattr(_ds, "load_dataset", None)) \
                and not getattr(_ds.load_dataset, "_rlab_patched", False):
            _ds.load_dataset = _wrap(_ds.load_dataset)
            applied.append("datasets.load_dataset")
    except Exception:
        pass
    # 3) modelscope 模块里早绑定的引用（from datasets import X 式导入打不到就靠这个）
    try:
        import sys as _sys
        for _name, _mod in list(_sys.modules.items()):
            if _mod is None or not (_name == "modelscope" or _name.startswith("modelscope.")):
                continue
            for _attr in ("load_dataset", "as_dataset"):
                _fn = getattr(_mod, _attr, None)
                if callable(_fn) and not getattr(_fn, "_rlab_patched", False) \
                        and getattr(_fn, "__module__", "").startswith("datasets"):
                    try:
                        setattr(_mod, _attr, _wrap(_fn))
                        applied.append(f"{_name}.{_attr}")
                    except Exception:
                        pass
    except Exception:
        pass
    print(f"[data] verification_mode 垫片已应用: {applied or '无（datasets/modelscope 未安装？）'}")


def load_gsm8k_train():
    """GSM8K train split。默认 modelscope，HF 仅作回落（训练机 HF 网络不通）。"""
    ms_err = None
    if DATA_SOURCE in ("ms", "auto"):
        try:
            _patch_verification_mode()
            from modelscope.msdatasets import MsDataset
            ds = MsDataset.load("modelscope/gsm8k", subset_name="main",
                                split="train", trust_remote_code=True)
            return [{"Q": x["question"], "A": x["answer"].split("####")[-1].strip()}
                    for x in ds]
        except Exception as e:
            ms_err = e
            if DATA_SOURCE == "ms":
                import traceback
                print(f"[data] modelscope gsm8k 加载失败（{e}），改走 HF\n"
                      + traceback.format_exc())
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return [{"Q": q, "A": a.split("####")[-1].strip()}
                for q, a in zip(ds["question"], ds["answer"])]
    except Exception as e:
        raise RuntimeError(
            f"[data] GSM8K train 加载失败：modelscope 错误={ms_err}，HF 错误={e}；"
            "可设置 RLAB_DATA_SOURCE=ms/hf 切换数据源") from e


# ---- DAPO-Math-17k（方案1：对齐 agentic-rl-lab/05-retool） ----
_DAPO_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
_DAPO_PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

# 本地 prepare_dapo_math 产物目录（train.jsonl 已剔除 dev、dev.jsonl 为 held-out）
_DAPO_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "dapo_math")


def _read_qa_jsonl(path: str) -> list:
    """读 prepare 脚本产物的 jsonl（question/Q + answer/A 键名都兼容）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = r.get("question") or r.get("Q")
            a = r.get("answer") or r.get("A")
            if q and a:
                rows.append({"Q": str(q), "A": str(a)})
    return rows


def load_dapo_math_dev() -> list:
    """held-out 评测池 = dev.jsonl（prepare_dapo_math 产物）。

    【2026-09-09 审查修复·训练/评测同池污染】原 eval 端 dev.jsonl 缺失时静默回落
    全量 17k train split——评的全是训练池内的题。现在 dev 缺失直接报错并给出
    指引，绝不静默用训练池充当 held-out。"""
    path = os.path.join(_DAPO_LOCAL_DIR, "dev.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[data] dapo_math held-out 集缺失: {path}\n"
            "  先运行: python -m rlab.prepare_dapo_math --dev-size 500\n"
            "  （不要用训练池充当 held-out——那是 2026-09-09 审查发现的同池污染）")
    rows = _read_qa_jsonl(path)
    if not rows:
        raise RuntimeError(f"[data] dev.jsonl 存在但清洗后为空: {path}")
    print(f"[data] dapo_math dev.jsonl (held-out): {len(rows)} 题")
    return rows


def dapo_exclude_dev(rows: list, dev_rows: list) -> list:
    """纯函数：从全量池中剔除 dev 题（按题面文本匹配，兼容 dev/train.jsonl 与
    全量加载两种行格式）。训练/评测不相交的契约所在，CPU 可测。"""
    dev_qs = {str(r.get("question") or r.get("Q") or "").strip() for r in dev_rows}
    return [r for r in rows if str(r.get("Q") or "").strip() not in dev_qs]


def load_dapo_math_train() -> list:
    """训练池。优先本地 train.jsonl（prepare 产物，已剔除 dev）；否则加载全量
    17k 并按 dev.jsonl 剔除 dev 题——保证训练池与 held-out 不相交。

    【2026-09-09 审查修复】旧版直接返回全量 17k：即使跑了 prepare 脚本，dev 题
    依然在训练池里（prepare 只写文件、训练路径根本不读 train.jsonl），dev=50 题
    被完整训过还拿来当评测集。"""
    ms_err = None
    train_jsonl = os.path.join(_DAPO_LOCAL_DIR, "train.jsonl")
    if os.path.exists(train_jsonl):
        rows = _read_qa_jsonl(train_jsonl)
        if rows:
            print(f"[data] DAPO-Math 训练池 via 本地 train.jsonl（已剔除 dev）: {len(rows)} 条")
            return rows
    # 1) 尝试 modelscope（训练机默认；HF 镜像也可能通）
    if DATA_SOURCE in ("ms", "auto"):
        try:
            _patch_verification_mode()
            # modelscope 上该数据集 id 可能是 BytedTsinghua-SIA/DAPO-Math-17k
            from modelscope.msdatasets import MsDataset
            # 先试 modelscope 官方 id；失败再试 HF id 的 ms 镜像
            for ms_id in ("BytedTsinghua-SIA/DAPO-Math-17k", "dapo-math-17k"):
                try:
                    ds = MsDataset.load(ms_id, split="train", trust_remote_code=True)
                    rows = []
                    for idx, row in enumerate(ds):
                        prompt = row.get("prompt")
                        q = _strip_dapo_template(str(prompt[0].get("content") or "")) if isinstance(prompt, list) and prompt else ""
                        rm = row.get("reward_model") or {}
                        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
                        if isinstance(gt, list):
                            gt = gt[0] if gt else None
                        a = str(gt or "").strip()
                        if q and a:
                            rows.append({"Q": q, "A": a})
                    if rows:
                        print(f"[data] DAPO-Math-17k via modelscope {ms_id}: {len(rows)} 条")
                        return _dapo_strip_dev(rows)
                except Exception:
                    continue
            raise RuntimeError("modelscope DAPO-Math-17k 均未命中")
        except Exception as e:
            ms_err = e
            import traceback
            print(f"[data] modelscope DAPO-Math-17k 加载失败（{e}），改走 HF\n" + traceback.format_exc())
    # 2) 回落 HF datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        rows = []
        for idx, row in enumerate(ds):
            prompt = row.get("prompt")
            q = _strip_dapo_template(str(prompt[0].get("content") or "")) if isinstance(prompt, list) and prompt else ""
            rm = row.get("reward_model") or {}
            gt = rm.get("ground_truth") if isinstance(rm, dict) else None
            if isinstance(gt, list):
                gt = gt[0] if gt else None
            a = str(gt or "").strip()
            if q and a:
                rows.append({"Q": q, "A": a})
        if not rows:
            raise RuntimeError("HF DAPO-Math-17k 清洗后为空")
        print(f"[data] DAPO-Math-17k via HF: {len(rows)} 条")
        return _dapo_strip_dev(rows)
    except Exception as e:
        raise RuntimeError(f"[data] DAPO-Math-17k 加载失败：modelscope 错误={ms_err}，HF 错误={e}") from e


def _dapo_strip_dev(rows: list) -> list:
    """全量池剔除 dev 题；dev.jsonl 缺失时打大字警告（此时评测协议已破坏）。"""
    dev_path = os.path.join(_DAPO_LOCAL_DIR, "dev.jsonl")
    if os.path.exists(dev_path):
        dev_rows = _read_qa_jsonl(dev_path)
        kept = dapo_exclude_dev(rows, dev_rows)
        print(f"[data] 训练池剔除 dev 题: {len(rows)} -> {len(kept)}（dev {len(dev_rows)} 题）")
        return kept
    print("\n" + "!" * 70)
    print(f"[data] 警告: dev.jsonl 缺失（{dev_path}），训练池无法剔除 held-out 题！")
    print("[data] 此训练池跑出的模型将没有可信的 held-out 评测。")
    print("[data] 请运行: python -m rlab.prepare_dapo_math --dev-size 500")
    print("!" * 70 + "\n")
    return rows


def _strip_dapo_template(q: str) -> str:
    if q.startswith(_DAPO_PROMPT_PREFIX):
        q = q[len(_DAPO_PROMPT_PREFIX):]
    if q.endswith(_DAPO_PROMPT_SUFFIX):
        q = q[:-len(_DAPO_PROMPT_SUFFIX)]
    return q.strip()


# ---- 离线难度预探测过滤（2026-09-10，probe_difficulty.py 配套） ----
# 动机：retool_math 丢弃率 81% 的主体是全错组 (1-p)^8；题目级 QuestionScheduler
# 只能出清"在线观察到连续零方差"的题（每题先烧 streak×n 条轨迹才出局），而离线
# 探针用 k 条短样本把 p≈0 / p≈1 的题在训练开始前一次性出清。两级过滤互补：
# 静态过滤出清"base 从未做对过的题"，在线调度出清"当前学不动的题"。


def load_difficulty_table(path: str) -> dict:
    """读 probe_difficulty.py 产出的 jsonl -> {Q: 行dict}（含 k/n_correct/fmt_rate…）。

    坏行静默跳过（探针是逐行追加写，崩溃可能留下截断行）；缺 k/n_correct 或
    k<=0 的行视为无效——过滤宁可保守（题进不了表 = 被丢弃，见 filter 的
    missing 口径），不允许半行数据混进训练池。"""
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = r.get("Q")
            k, nc = r.get("k"), r.get("n_correct")
            if q and isinstance(k, int) and k > 0 and isinstance(nc, int) and 0 <= nc <= k:
                table[str(q)] = r
    return table


def filter_qas_by_difficulty(qas: list, table: dict, lo: float = 0.0, hi: float = 1.0):
    """纯函数（CPU 可测）：保留通过率 n_correct/k 严格落在开区间 (lo, hi) 的题。

    默认 (0,1) = DAPO "accuracy neither 0 nor 1" 的离线版。表中缺失的题按
    丢弃计（missing）——过滤即选择，半覆盖的表不该让未探测题混进训练分布。
    返回 (kept, stats)；stats 口径：kept/p_zero/p_one/band_out/missing/total。"""
    kept, stats = [], {"kept": 0, "p_zero": 0, "p_one": 0, "band_out": 0,
                       "missing": 0, "total": len(qas)}
    for x in qas:
        row = table.get(str(x["Q"]))
        if row is None:
            stats["missing"] += 1
            continue
        rate = row["n_correct"] / row["k"]
        if rate <= lo:
            stats["p_zero" if rate == 0 else "band_out"] += 1
        elif rate >= hi:
            stats["p_one" if rate == 1 else "band_out"] += 1
        else:
            kept.append(x)
            stats["kept"] += 1
    return kept, stats
