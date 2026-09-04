# -*- coding: utf-8 -*-
"""rlab/data.py — 数据集加载。

统一返回 QA 列表：[{"Q": 问题文本, "A": 标准答案文本(用于 reward 比对)}]。
默认走 modelscope（HF 及其镜像在训练机上网络不通），失败才回落 HF datasets。
fixture 模式供 CPU 冒烟测试，不触网。
"""

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
