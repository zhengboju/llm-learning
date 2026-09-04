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


def load_gsm8k_train():
    """GSM8K train split。默认 modelscope，HF 仅作回落（训练机 HF 网络不通）。"""
    ms_err = None
    if DATA_SOURCE in ("ms", "auto"):
        try:
            from modelscope.msdatasets import MsDataset
            ds = MsDataset.load("modelscope/gsm8k", subset_name="main",
                                split="train", trust_remote_code=True)
            return [{"Q": x["question"], "A": x["answer"].split("####")[-1].strip()}
                    for x in ds]
        except Exception as e:
            ms_err = e
            if DATA_SOURCE == "ms":
                print(f"[data] modelscope gsm8k 加载失败（{e}），改走 HF")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return [{"Q": q, "A": a.split("####")[-1].strip()}
                for q, a in zip(ds["question"], ds["answer"])]
    except Exception as e:
        raise RuntimeError(
            f"[data] GSM8K train 加载失败：modelscope 错误={ms_err}，HF 错误={e}；"
            "可设置 RLAB_DATA_SOURCE=ms/hf 切换数据源") from e
