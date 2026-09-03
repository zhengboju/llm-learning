# -*- coding: utf-8 -*-
# 合并 eval_vllm_one.py 产出的多个 json 并打印总表
# 用法: python eval_merge.py eval_v_*.json
import json, sys

rows = {}
for f in sys.argv[1:]:
    with open(f, encoding="utf-8") as fp:
        rows.update(json.load(fp))

print(f"{'模型':<16}{'准确率':>10}{'格式率':>10}{'双达标':>10}{'有效样本':>10}")
for name, r in rows.items():
    print(f"{name:<16}{r['acc']*100:>9.1f}%{r['fmt']*100:>9.1f}%{r['both']*100:>9.1f}%{r['n']:>10}")
