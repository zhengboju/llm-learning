# -*- coding: utf-8 -*-
"""对拍内嵌评测 vs 单独评测：定位同 ckpt 两份读数的差值来源。

【起因·p11 step100 差 +7.0pp】内嵌 63.3% vs 单独 70.3%，而 step200/300 只差
+0.4/+0.8pp。只有一个 checkpoint 大幅偏离 → 排除"两条路径协议系统性不同"。

判据分四层，逐层排除（前三层全绿才走第四层）：
  ① 出处 provenance：model_path / proto_src / proto_from_run_info —— 同一个 ckpt？
  ② 协议 protocol：round_tokens/max_len/val_n/temperature/确定性档 —— 同一个协议？
  ③ 逐题 per-item：qk 题面指纹集合 + 翻转题数 —— 同 ckpt 同协议只该翻个位数
  ④ 代码执行分层：确定性档管不到的非确定性源（沙箱墙钟超时受负载影响）

【为什么需要第四层】p11 实测前三层全绿（题集交集 256/256、协议全键一致、两边
都 batch_invariant=True + greedy），却翻了 54 题且**方向偏斜**（内嵌对 18 /
单独对 36）。batch_invariant 只保证"结果不随 batch 组成变化"，管不到 vLLM 之外
的进程——retool 每个工具段都要跑 rlab/sandbox.py run_code，那是 5 秒**墙钟**
超时 + SIGKILL，而：
  · 内嵌评测跑在 checkpoint 刚存完、训练进程还活着时（还要和训练端
    sandbox_workers=8 抢 CPU），gpu_mem=0.20；
  · 单独评测跑在训练结束、机器空闲时，gpu_mem=0.78/per_gpu。
同一段代码在负载重时超时、空闲时跑完 → 工具返回错误而非数值 → 轨迹被带偏、
boxed 丢失。墙钟超时是负载敏感的，这正是确定性档覆盖不到的那类抖动。

用法：
    python -m rlab.diag_eval_gap <内嵌json> <合并json> <合并json里的条目名>
例：
    python -m rlab.diag_eval_gap \\
      rlab_out/retool_math_p11/step_100/eval_test.json eval_vllm_all.json step100
"""
import json
import os
import sys

from rlab.analysis import read_eval_result

BF16_JITTER_MAX = 6      # 同权重同协议下 bf16/批调度抖动的翻转题数上界（经验值）


def _pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def load(path, name=None):
    """读一个 eval json；name 给定时从合并 json（eval_vllm_all.json）里取该条。"""
    if not os.path.exists(path):
        return None, f"文件不存在: {path}"
    if name:
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            return None, f"读不出 {path}: {e}"
        if name not in raw:
            keys = [k for k in raw if not k.startswith("_")]
            return None, f"{path} 里没有 {name!r}；现有条目: {keys}"
        return raw[name], None
    r = read_eval_result(path)      # 统一解 {name: result} 嵌套壳与扁平壳
    if not r:
        return None, f"{path} 解不出有效 result（坏壳/空文件/超时哨兵）"
    return r, None


def show(tag, r):
    p = r.get("eval_protocol") or {}
    print(f"\n=== {tag} ===")
    print(f"  acc={_pct(r.get('acc'))}  fmt={_pct(r.get('fmt'))}  n={r.get('n')}")
    print(f"  model_path      = {r.get('model_path', '(缺，旧 json)')}")
    print(f"  proto_src       = {p.get('proto_src', '(缺)')}")
    print(f"  proto_from_run_info = {p.get('proto_from_run_info', '(缺)')}")
    print(f"  val_n={p.get('val_n')} greedy={p.get('greedy')} "
          f"temperature={p.get('temperature')}({p.get('temperature_src')}) "
          f"top_p={p.get('top_p')}")
    print(f"  round_tokens={p.get('round_tokens')} max_len={p.get('max_len')} "
          f"max_rounds={p.get('max_rounds')} "
          f"max_prompt_length={p.get('max_prompt_length')}")
    print(f"  batch_invariant={p.get('vllm_batch_invariant')} "
          f"attn={p.get('vllm_attention_backend')}")
    print(f"  system_prompt_sha={p.get('system_prompt_sha')}  seed={p.get('seed')}")
    print(f"  split={r.get('split')}  n_requested={r.get('n_requested')} "
          f"dropped_long={r.get('n_dropped_long')} dropped_plen={r.get('n_dropped_plen')}")
    print(f"  code_rate={_pct(r.get('code_rate'))} "
          f"code_ok_rate={_pct(r.get('code_ok_rate'))} "
          f"avg_rounds={r.get('avg_rounds')}")
    print(f"  metrics_version={r.get('metrics_version')}")


def norm_path(v):
    """路径归一：'./a/b' 与 'a/b' 是同一个目录，不该报成协议差异。"""
    if not isinstance(v, str):
        return v
    return os.path.normpath(v).replace("\\", "/")


# 路径类键按 normpath 比较；其余键严格相等
_PATH_KEYS = {"proto_src", "model_path"}


def compare(a, b, tag_a, tag_b):
    """协议逐键 diff。返回 (是否实质一致, 差异键列表)。

    路径类键只在 normpath 后仍不同才算差异——p11 实测两边只差一个 './' 前缀，
    旧版把它标成 ❌ 会把注意力引到无关处（真正的差异在 n_requested）。
    """
    pa = a.get("eval_protocol") or {}
    pb = b.get("eval_protocol") or {}
    print(f"\n=== 协议逐键对比（{tag_a} vs {tag_b}）===")
    proto_keys = ["val_n", "greedy", "temperature", "top_p", "round_tokens",
                  "max_len", "max_rounds", "max_prompt_length",
                  "vllm_batch_invariant", "vllm_attention_backend",
                  "system_prompt_sha", "seed", "proto_src"]
    top_keys = ["split", "model_path", "n", "n_requested",
                "n_dropped_long", "n_dropped_plen"]
    diffs = []
    for k, va, vb in ([(k, pa.get(k), pb.get(k)) for k in proto_keys]
                      + [(k, a.get(k), b.get(k)) for k in top_keys]):
        if k in _PATH_KEYS:
            differ = norm_path(va) != norm_path(vb)
            note = "" if differ else ("  (仅 ./ 前缀差异，等价)"
                                      if va != vb else "")
        else:
            differ = va != vb
            note = ""
        if differ:
            diffs.append(k)
        print(f"  {'❌' if differ else '  '} {k:<24} {va!r:<30} {vb!r}{note}")
    if not diffs:
        print("\n  → 协议实质完全一致（路径前缀差异不计）")
    else:
        print(f"\n  → 协议存在实质差异: {diffs}")
        if "n_requested" in diffs:
            print("     注：n_requested 不同但 n 相同 = 两边都取满了同一个池子，"
                  "题集可能仍相同（看下一层交集）")
    return (not diffs), diffs


def pair(a, b, tag_a, tag_b):
    """逐题配对。返回 (common, ia, ib, b_only, c_only)；无法配对时返回 None。"""
    ia = {it["qk"]: it for it in (a.get("items") or []) if "qk" in it}
    ib = {it["qk"]: it for it in (b.get("items") or []) if "qk" in it}
    print(f"\n=== 逐题配对（{tag_a} vs {tag_b}）===")
    if not ia or not ib:
        print(f"  某一侧无 per-item（--dump_items 关了或旧 json）："
              f"{tag_a} {len(ia)} 题 / {tag_b} {len(ib)} 题 → 无法配对")
        return None
    common = set(ia) & set(ib)
    print(f"  题面指纹: {tag_a} {len(ia)} 题 / {tag_b} {len(ib)} 题 / "
          f"交集 {len(common)} 题")
    if len(common) != len(ia) or len(common) != len(ib):
        print(f"  ❌ 题集不同（只 {tag_a} 有 {len(set(ia) - set(ib))} 题，"
              f"只 {tag_b} 有 {len(set(ib) - set(ia))} 题）"
              f"\n     → 抽到的题就不一样，acc 不可直接比"
              f"（seed/split/剔题阈值/n_requested 有差异）")
    else:
        print("  ✓ 题集完全相同 → acc 差异不来自抽题")
    if not common:
        return None
    b_only = sum(1 for q in common if (ia[q]["acc"] > 0) and not (ib[q]["acc"] > 0))
    c_only = sum(1 for q in common if not (ia[q]["acc"] > 0) and (ib[q]["acc"] > 0))
    agree = len(common) - b_only - c_only
    print(f"  一致 {agree} 题 | 只 {tag_a} 对 {b_only} 题 | "
          f"只 {tag_b} 对 {c_only} 题 | 分歧合计 {b_only + c_only} 题")
    print(f"  Δacc = {(c_only - b_only) / len(common) * 100:+.1f}pp（配对口径）")
    if b_only + c_only <= BF16_JITTER_MAX:
        print(f"  → 分歧 ≤{BF16_JITTER_MAX} = 同权重同协议的 bf16/批调度抖动，正常噪声")
    else:
        print(f"  → 分歧 {b_only + c_only} 题远超 bf16 抖动量级"
              f"（≤{BF16_JITTER_MAX}）→ 走第四层：代码执行分层")
    return common, ia, ib, b_only, c_only


def sandbox_verdict(b_only, c_only, ck_a, ck_b, d_rate, base_rate, n_disagree,
                    ck_min=5, enrich_min=10.0):
    """沙箱资源竞争判据（纯函数，CPU 可测）。

    返回 {"symmetric", "code_linked", "sandbox", "enrich", "skew"}。
    `sandbox=True` 仅当**两条同时成立**：
      ① 单向劣化（方向偏斜）——沙箱超时只会让轨迹变差，不会变好，所以对称翻转
         在定义上就不可能是它；
      ② 与代码执行相关——空闲侧 code_ok 显著更多，或翻转显著富集在代码题上。

    【2026-09-25 这个函数是为了修自己的 bug 才抽出来的】旧版判据写成
        ck_b - ck_a >= ck_min  or  (not sym and enrich >= enrich_min)
    第一个 clause 能**单独**成立 → 负例（方向 14/14 完全对称、富集 −38pp）照样
    印出"沙箱是主因 ✓"，只因 code_ok 差了 +11（fixture 里纯随机噪声）。
    ②必须被①**门控**（and），不能并联（or）——与 health.py 那条"判定规则有多个时
    先特例后一般，否则 elif 永远走不到"同源：条件实为 and 时写成 or，最弱的证据
    会独自定案。抽成纯函数后这条门控被单测锁死，不再只靠 fixture 肉眼看。
    """
    skew = abs(b_only - c_only)
    symmetric = skew < max(BF16_JITTER_MAX, 0.2 * max(n_disagree, 1))
    enrich = d_rate - base_rate
    code_linked = (ck_b - ck_a >= ck_min) or (enrich >= enrich_min)
    return {"symmetric": symmetric, "code_linked": code_linked,
            "sandbox": (not symmetric) and code_linked,
            "enrich": enrich, "skew": skew}


def code_layer(a, b, common, ia, ib, b_only, c_only, tag_a, tag_b):
    """第四层：代码执行分层——解释"协议全同、题集全同，却翻了几十题"。

    判据（两条同时成立才算沙箱竞争实锤）：
      ① 方向偏斜：沙箱超时只会让轨迹变差不会变好 → 分歧应显著偏向空闲那侧，
         而非对称（bf16 抖动是对称的）；
      ② 代码相关性：翻转富集在写过代码的题上，且空闲侧 code_ok 更高。
    """
    print(f"\n=== 代码执行分层（{tag_a} vs {tag_b}）===")
    cu_a = sum(1 for q in common if (ia[q].get("code_used") or 0) > 0)
    cu_b = sum(1 for q in common if (ib[q].get("code_used") or 0) > 0)
    ck_a = sum(1 for q in common if (ia[q].get("code_ok") or 0) > 0)
    ck_b = sum(1 for q in common if (ib[q].get("code_ok") or 0) > 0)
    print(f"  配对集内 code_used>0: {tag_a} {cu_a} 题 / {tag_b} {cu_b} 题"
          f"   （差 {cu_b - cu_a:+d}）")
    print(f"  配对集内 code_ok >0: {tag_a} {ck_a} 题 / {tag_b} {ck_b} 题"
          f"   （差 {ck_b - ck_a:+d}）")

    disagree = [q for q in common if (ia[q]["acc"] > 0) != (ib[q]["acc"] > 0)]
    if not disagree:
        print("  无分歧题，无需分层")
        return
    used_in = lambda q: ((ia[q].get("code_used") or 0) > 0
                         or (ib[q].get("code_used") or 0) > 0)
    d_code = [q for q in disagree if used_in(q)]
    d_rate = len(d_code) / len(disagree) * 100
    base_code = [q for q in common if used_in(q)]
    base_rate = len(base_code) / len(common) * 100 if common else 0.0
    print(f"  分歧 {len(disagree)} 题中写过代码 {len(d_code)} 题 ({d_rate:.0f}%)"
          f" | 全集写代码占比 {base_rate:.0f}%（基础率对照）")

    print("\n  判读：")
    v = sandbox_verdict(b_only, c_only, ck_a, ck_b, d_rate, base_rate, len(disagree))
    enrich = v["enrich"]
    if v["symmetric"]:
        print(f"    ① 方向基本对称 {b_only}/{c_only} → 不是单向劣化 ✗")
        print(f"    ② 代码成功数差 {ck_b - ck_a:+d}、翻转富集 {enrich:+.0f}pp"
              f"（① 未成立，本项不单独定案）")
        print("    → 沙箱竞争**不成立**（它只会单向变差）。对称的大量翻转指向"
              "另一类源：")
        print("       KV 池容量差异（gpu_mem 0.20 vs 0.78）导致抢占/重算路径不同、"
              "attn backend 实际生效值、\n"
              "       多轮上下文在不同 KV 容量下的分块边界不同。"
              "先核 eval 日志里两边的 KV cache blocks 与 preemption 计数。")
        return v
    hi = tag_b if c_only > b_only else tag_a
    print(f"    ① 方向偏斜 {b_only}/{c_only}（{hi} 明显更优）→ 单向劣化，"
          f"不是 bf16 抖动（那是对称的）✓")
    if v["code_linked"]:
        print(f"    ② 代码成功数差 {ck_b - ck_a:+d}、翻转富集 {enrich:+.0f}pp"
              f" → 劣化与代码执行相关 ✓")
        print("    → 综合：①+② 同时成立 = 内嵌评测与训练进程共卡共 CPU"
              "（gpu_mem 0.20 vs 0.78，且与 sandbox_workers=8 抢 CPU），"
              "\n       5s 墙钟沙箱超时被负载放大。")
        print("       结论：两个读数都不是 bug——内嵌是**带资源竞争的悲观读数**。")
        print("       用法：趋势判断用内嵌（同条件跨 step 可比）；"
              "绝对值/对外汇报用单独评测。")
    else:
        print(f"    ② 代码成功数差 {ck_b - ck_a:+d}、翻转富集 {enrich:+.0f}pp"
              f"（不显著）→ 劣化与代码执行无关 ✗")
        print("    → 单向劣化但不由沙箱解释：查 KV 抢占/重算、"
              "轮间上下文截断位置、两边 max_num_seqs 实际值。")
    return v


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    inline_path, merged_path, merged_name = argv
    a, err_a = load(inline_path)
    if err_a:
        print(err_a)
        return 1
    b, err_b = load(merged_path, merged_name)
    if err_b:
        print(err_b)
        return 1
    show(f"内嵌评测 {inline_path}", a)
    show(f"单独评测 {merged_path}[{merged_name}]", b)
    compare(a, b, "内嵌", "单独")
    res = pair(a, b, "内嵌", "单独")
    if res:
        common, ia, ib, b_only, c_only = res
        if b_only + c_only > BF16_JITTER_MAX:
            code_layer(a, b, common, ia, ib, b_only, c_only, "内嵌", "单独")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
