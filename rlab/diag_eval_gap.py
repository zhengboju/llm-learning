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


def cond_ok_rate(code_used_n, code_ok_n):
    """条件成功率 = code_ok | code_used（写了代码的题里执行成功的比例）。

    这是沙箱健康度的**唯一无混淆口径**：绝对数 code_ok 会随"写代码的题数"
    一起动，两侧写代码题数不同时绝对数差毫无信息量。None = 没题可算。"""
    if not code_used_n:
        return None
    return code_ok_n / code_used_n


def sandbox_verdict(b_only, c_only, cu_a, ck_a, cu_b, ck_b, n_disagree,
                    cond_drop_min=2.0):
    """沙箱资源竞争判据（纯函数，CPU 可测）。

    `sandbox=True` 仅当**两条同时成立**：
      ① 单向劣化（方向偏斜）——沙箱超时只会让轨迹变差不会变好，对称翻转在
         定义上就不可能是它；
      ② **条件成功率**在高负载侧更低：cond_a < cond_b − cond_drop_min(pp)。
         墙钟超时被 SIGKILL 的直接后果就是"写了代码但没拿到结果"。

    【2026-09-25 第二次修这个函数——上一版判据是被混淆的】
    上一版 ② 用 `ck_b - ck_a >= 5`（code_ok **绝对数**差）。在真机 p11 数据上
    它给出了自信的错误结论：
        内嵌 code_used=162 code_ok=157 → 条件成功率 96.9%
        单独 code_used=173 code_ok=165 → 条件成功率 95.4%
    绝对数差 +8 看着像"空闲侧代码成功更多 ✓"，但那 +8 完全是"单独侧写代码的题
    本来就多 11 题"的派生量——**条件成功率其实是内嵌更高 1.5pp**，与沙箱超时
    假设的方向相反。绝对数是混淆量，条件率才是无混淆量。
    教训与上一版（or 写成 and）不同源但同类：**判据必须用"若假设为真则必然
    单向变化"的量**，任何会被第三变量（此处=写代码题数）带动的量都不能当证据。
    """
    skew = abs(b_only - c_only)
    symmetric = skew < max(BF16_JITTER_MAX, 0.2 * max(n_disagree, 1))
    ca, cb = cond_ok_rate(cu_a, ck_a), cond_ok_rate(cu_b, ck_b)
    if ca is None or cb is None:
        cond_drop = None
        code_linked = False
    else:
        cond_drop = (cb - ca) * 100.0        # >0 = 高负载侧条件成功率更低
        code_linked = cond_drop >= cond_drop_min
    return {"symmetric": symmetric, "code_linked": code_linked,
            "sandbox": (not symmetric) and code_linked,
            "cond_a": ca, "cond_b": cb, "cond_drop": cond_drop,
            "used_gap": cu_b - cu_a, "skew": skew}


def code_layer(a, b, common, ia, ib, b_only, c_only, tag_a, tag_b):
    """第四层：代码执行分层——解释"协议全同、题集全同，却翻了几十题"。

    判据（两条同时成立才算沙箱竞争实锤）：
      ① 方向偏斜：沙箱超时只会让轨迹变差不会变好 → 分歧应显著偏向空闲那侧，
         而非对称（bf16 抖动是对称的）；
      ② **条件成功率**（code_ok | code_used）在高负载侧更低——绝对数 code_ok
         会被"写代码题数"带动，是混淆量，不能当证据（见 sandbox_verdict 注释）。
    """
    print(f"\n=== 代码执行分层（{tag_a} vs {tag_b}）===")
    cu_a = sum(1 for q in common if (ia[q].get("code_used") or 0) > 0)
    cu_b = sum(1 for q in common if (ib[q].get("code_used") or 0) > 0)
    ck_a = sum(1 for q in common if (ia[q].get("code_ok") or 0) > 0)
    ck_b = sum(1 for q in common if (ib[q].get("code_ok") or 0) > 0)
    print(f"  配对集内 code_used>0: {tag_a} {cu_a} 题 / {tag_b} {cu_b} 题"
          f"   （差 {cu_b - cu_a:+d}）")
    print(f"  配对集内 code_ok >0: {tag_a} {ck_a} 题 / {tag_b} {ck_b} 题"
          f"   （差 {ck_b - ck_a:+d}，**混淆量**，仅供参照）")
    _ca, _cb = cond_ok_rate(cu_a, ck_a), cond_ok_rate(cu_b, ck_b)
    print(f"  条件成功率 code_ok|code_used: {tag_a} {_pct(_ca)} / "
          f"{tag_b} {_pct(_cb)}   ← **无混淆口径**")

    disagree = [q for q in common if (ia[q]["acc"] > 0) != (ib[q]["acc"] > 0)]
    if not disagree:
        print("  无分歧题，无需分层")
        return None
    used_in = lambda q: ((ia[q].get("code_used") or 0) > 0
                         or (ib[q].get("code_used") or 0) > 0)
    d_code = [q for q in disagree if used_in(q)]
    d_rate = len(d_code) / len(disagree) * 100
    base_code = [q for q in common if used_in(q)]
    base_rate = len(base_code) / len(common) * 100 if common else 0.0
    print(f"  分歧 {len(disagree)} 题中写过代码 {len(d_code)} 题 ({d_rate:.0f}%)"
          f" | 全集写代码占比 {base_rate:.0f}%（基础率对照）")

    print("\n  判读：")
    v = sandbox_verdict(b_only, c_only, cu_a, ck_a, cu_b, ck_b, len(disagree))
    _cd = v["cond_drop"]
    if v["symmetric"]:
        print(f"    ① 方向基本对称 {b_only}/{c_only} → 不是单向劣化 ✗")
    else:
        hi = tag_b if c_only > b_only else tag_a
        print(f"    ① 方向偏斜 {b_only}/{c_only}（{hi} 明显更优）→ 单向劣化，"
              f"不是 bf16 抖动（那是对称的）✓")
    if _cd is None:
        print("    ② 无写代码题，条件成功率无从计算 ✗")
    elif v["code_linked"]:
        print(f"    ② 条件成功率 {tag_a} {_pct(v['cond_a'])} < {tag_b} "
              f"{_pct(v['cond_b'])}（低 {_cd:.1f}pp）→ 高负载侧沙箱确实在丢结果 ✓")
    else:
        print(f"    ② 条件成功率 {tag_a} {_pct(v['cond_a'])} vs {tag_b} "
              f"{_pct(v['cond_b'])}（差 {_cd:+.1f}pp）→ 高负载侧**并未**更差 ✗")

    if v["sandbox"]:
        print("    → ①+② 同时成立 = 内嵌与训练进程共卡共 CPU（gpu_mem 0.20 vs "
              "0.78，且与 sandbox_workers=8 抢 CPU），5s 墙钟沙箱超时被负载放大。")
        print("       两个读数都不是 bug——内嵌是**带资源竞争的悲观读数**。")
        print("       趋势判断用内嵌（同条件跨 step 可比）；绝对值用单独评测。")
        return v

    # 沙箱被否 → 把矛盾摆出来，并指向真正该查的地方
    print("    → 沙箱竞争**不成立**。")
    if v["used_gap"]:
        print(f"\n  ⚠ 真正的矛盾在更前面：两侧「写代码的题数」差 "
              f"{v['used_gap']:+d} 题（{cu_a} vs {cu_b}）。")
        print("     greedy(temp=0) + 同权重 + 同 prompt + batch_invariant=True 下，"
              "第一轮该逐 token 相同 →")
        print("     「写不写代码」本不该有任何差异。它变了，说明**分歧发生在生成层"
              "而非沙箱层**。")
        print("     batch_invariant 只保证「结果不随 batch 组成变化」，"
              "它不保证跨 KV 池容量一致：")
        print("       gpu_mem 0.20 vs 0.78 → KV 池块数不同 → chunked prefill 分块"
              "边界不同 → bf16 归约顺序不同")
        print("       → near-tie token 翻转（是否开 ``` 围栏正是这种 near-tie）。")
        print("     ⇒ 待验证的唯一假设：**gpu_mem 改变了生成本身**。"
              "判别实验见 docs（空闲机上跑 gpu_mem A/B）。")
    else:
        print("     两侧写代码题数相同，分歧不在代码路径 → 查 KV 抢占/重算计数、"
              "轮间上下文截断位置、max_num_seqs 实际值。")
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
