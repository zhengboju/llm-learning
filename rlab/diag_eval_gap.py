# -*- coding: utf-8 -*-
"""对拍内嵌评测 vs 单独评测：定位 step100 +7.0pp 差值的来源。

判据分三层，逐层排除：
  ① 出处（provenance）：model_path / proto_src / started —— 测的是不是同一个 ckpt？
  ② 协议（protocol）：round_tokens/max_len/val_n/temperature/确定性档 —— 同一个协议？
  ③ 逐题（per-item）：qk 集合是否相同、翻转题数 —— 同 ckpt 同协议下只该翻几题
"""
import json
import os
import sys

from rlab.analysis import read_eval_result


def load(path, name=None):
    """读一个 eval json；name 给定时从合并 json（eval_vllm_all.json）里取该条。"""
    if not os.path.exists(path):
        return None, f"文件不存在: {path}"
    if name:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if name not in raw:
            keys = [k for k in raw if not k.startswith("_")]
            return None, f"{path} 里没有 {name!r}；现有条目: {keys}"
        return raw[name], None
    return read_eval_result(path), None


def show(tag, r):
    p = r.get("eval_protocol") or {}
    print(f"\n=== {tag} ===")
    print(f"  acc={(r.get('acc') or 0)*100:.1f}%  fmt={(r.get('fmt') or 0)*100:.1f}%  n={r.get('n')}")
    print(f"  model_path      = {r.get('model_path', '(缺，旧 json)')}")
    print(f"  proto_src       = {p.get('proto_src', '(缺)')}")
    print(f"  proto_from_run_info = {p.get('proto_from_run_info', '(缺)')}")
    print(f"  val_n={p.get('val_n')} greedy={p.get('greedy')} "
          f"temperature={p.get('temperature')}({p.get('temperature_src')}) top_p={p.get('top_p')}")
    print(f"  round_tokens={p.get('round_tokens')} max_len={p.get('max_len')} "
          f"max_rounds={p.get('max_rounds')} max_prompt_length={p.get('max_prompt_length')}")
    print(f"  batch_invariant={p.get('vllm_batch_invariant')} attn={p.get('vllm_attention_backend')}")
    print(f"  system_prompt_sha={p.get('system_prompt_sha')}  seed={p.get('seed')}")
    print(f"  split={r.get('split')}  n_requested={r.get('n_requested')} "
          f"dropped_long={r.get('n_dropped_long')} dropped_plen={r.get('n_dropped_plen')}")
    print(f"  metrics_version={r.get('metrics_version')}")


def compare(a, b, tag_a, tag_b):
    pa = a.get("eval_protocol") or {}
    pb = b.get("eval_protocol") or {}
    print(f"\n=== 协议逐键对比（{tag_a} vs {tag_b}）===")
    keys = ["val_n", "greedy", "temperature", "top_p", "round_tokens", "max_len",
            "max_rounds", "max_prompt_length", "vllm_batch_invariant",
            "vllm_attention_backend", "system_prompt_sha", "seed", "proto_src"]
    same = True
    for k in keys:
        va, vb = pa.get(k), pb.get(k)
        mark = "  " if va == vb else "❌"
        if va != vb:
            same = False
        print(f"  {mark} {k:<24} {va!r:<28} {vb!r}")
    for k in ("split", "model_path", "n", "n_dropped_long", "n_dropped_plen"):
        va, vb = a.get(k), b.get(k)
        mark = "  " if va == vb else "❌"
        if va != vb:
            same = False
        print(f"  {mark} {k:<24} {va!r:<28} {vb!r}")
    print(f"\n  → 协议{'完全一致' if same else '存在差异（见 ❌ 行，差异即根因）'}")
    return same


def pair(a, b, tag_a, tag_b):
    """逐题配对：同 ckpt 同协议下，翻转题数应该只有个位数。"""
    ia = {it["qk"]: it for it in (a.get("items") or []) if "qk" in it}
    ib = {it["qk"]: it for it in (b.get("items") or []) if "qk" in it}
    print(f"\n=== 逐题配对（{tag_a} vs {tag_b}）===")
    if not ia or not ib:
        print(f"  某一侧无 per-item（--dump_items 关了或旧 json）："
              f"{tag_a} {len(ia)} 题 / {tag_b} {len(ib)} 题 → 无法配对")
        return
    common = set(ia) & set(ib)
    print(f"  题面指纹: {tag_a} {len(ia)} 题 / {tag_b} {len(ib)} 题 / 交集 {len(common)} 题")
    if len(common) != len(ia) or len(common) != len(ib):
        print(f"  ❌ 题集不同（只 {tag_a} 有 {len(set(ia)-set(ib))} 题，"
              f"只 {tag_b} 有 {len(set(ib)-set(ia))} 题）"
              f"\n     → 抽到的题就不一样，acc 不可直接比（seed/split/剔题阈值有差异）")
    if not common:
        return
    b_only = sum(1 for q in common if (ia[q]["acc"] > 0) and not (ib[q]["acc"] > 0))
    c_only = sum(1 for q in common if not (ia[q]["acc"] > 0) and (ib[q]["acc"] > 0))
    agree = len(common) - b_only - c_only
    print(f"  一致 {agree} 题 | 只 {tag_a} 对 {b_only} 题 | 只 {tag_b} 对 {c_only} 题 "
          f"| 分歧合计 {b_only + c_only} 题")
    print(f"  Δacc = {(c_only - b_only) / len(common) * 100:+.1f}pp（配对口径）")
    if b_only + c_only <= 6:
        print("  → 分歧个位数 = 同权重同协议的 bf16/批调度抖动，正常噪声")
    else:
        print(f"  → ❌ 分歧 {b_only + c_only} 题远超抖动量级：**不是同一个权重或同一个协议**")


if __name__ == "__main__":
    inline_path, merged_path, merged_name = sys.argv[1], sys.argv[2], sys.argv[3]
    a, err_a = load(inline_path)
    b, err_b = load(merged_path, merged_name)
    if err_a:
        print(err_a); sys.exit(1)
    if err_b:
        print(err_b); sys.exit(1)
    show(f"内嵌评测 {inline_path}", a)
    show(f"单独评测 {merged_path}[{merged_name}]", b)
    compare(a, b, "内嵌", "单独")
    pair(a, b, "内嵌", "单独")
