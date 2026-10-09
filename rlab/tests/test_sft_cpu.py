# -*- coding: utf-8 -*-
"""rlab/tests/test_sft_cpu.py — SFT 冷启动（数据/编码/损失/验收门）CPU 验收测试。

运行：python -m rlab.tests.test_sft_cpu

覆盖：
  A. sft_data：dump 行过滤 / 段结构校验 / 预算行分档改写（含 \\b 转义事故防护）
     / 低预算合成（cut=last 信息完备口径）/ 按题限额偏好短轨迹
  B. sft.encode_example（MockTok）：mask 契约（prompt/工具段 -100、assistant
     含收尾 im_end 进 loss）、前缀校验失败即抛
  C. masked_lm_loss_sum：数值 / **梯度探针**（masked 位梯度严格 0——
     "只测正向数值发现不了零梯度"的教训）/ n=0 计算图连通
  D. chunked_sft_loss：与 HF 内建 labels 路径数值一致（单 chunk）；
     collate 右 pad 契约
  E. sft_gate：四项门槛判定
  F. CLI/源码接线
"""
import json
import os
import tempfile

import torch

PASS = []


def check(name, cond):
    assert cond, f"[FAIL] {name}"
    PASS.append(name)
    print(f"  ok - {name}")


def _mk_row(qk="q1", acc=1, trunc=0, cu=1, clen=100, segs=None, status="ok"):
    return {"Q": f"题{qk}", "A": "42", "qk": qk, "status": status, "acc": acc,
            "trunc_final": trunc, "code_used": cu, "clen": clen,
            "segs": segs if segs is not None else [
                {"kind": "assistant", "len": 30,
                 "text": "先算一算，调用工具。"},
                {"kind": "tool", "len": 20,
                 "body": "42\n[budget] 7000 tokens left in this trajectory. "
                         "Finish with your final answer when you have enough."},
                {"kind": "assistant", "len": 25,
                 "text": "结果是 42。\n\\boxed{42}"}]}


# ---------------------------------------------------------------- A. sft_data ----
def test_sft_data():
    print("[A] sft_data：过滤 / 结构 / 预算行改写 / 低预算合成 / 限额")
    from rlab.rollout import budget_hint_line
    from rlab.sft_data import (build_dataset, final_boxed,
                               parse_budget_left, rewrite_budget_lines, row_ok,
                               split_turns, synthesize_low_budget)

    # 过滤：已上传+答对+未截断
    check("row_ok：ok+答对+未截断 → True", row_ok(_mk_row()))
    check("row_ok：截断/答错/uniform 三态都拒",
          not row_ok(_mk_row(trunc=1)) and not row_ok(_mk_row(acc=-1))
          and not row_ok(_mk_row(status="uniform")))

    # 段结构：必须 assistant 开头、严格交替
    check("split_turns：正常三段通过",
          [t["kind"] for t in split_turns(_mk_row())]
          == ["assistant", "tool", "assistant"])
    _bad = _mk_row(segs=[{"kind": "tool", "len": 1, "body": "x"}])
    check("split_turns：tool 开头 → None（结构非法不硬修）",
          split_turns(_bad) is None)
    _bad2 = _mk_row(segs=[{"kind": "assistant", "len": 1, "text": "a"},
                          {"kind": "assistant", "len": 1, "text": "b"}])
    check("split_turns：两个 assistant 连排 → None", split_turns(_bad2) is None)
    _bad3 = _mk_row(segs=[{"kind": "assistant", "len": 1, "text": "  "},
                          {"kind": "tool", "len": 1, "body": "x"},
                          {"kind": "assistant", "len": 1, "text": "\\boxed{42}"}])
    check("split_turns：空 assistant 文本 → None", split_turns(_bad3) is None)

    # 预算行解析与分档改写
    turns = split_turns(_mk_row())
    check("parse_budget_left：解析旧单行形态的剩余额度",
          parse_budget_left(turns[1]["body"]) == 7000)
    check("parse_budget_left：无该行 → None", parse_budget_left("只有回包") is None)
    need = 266 + 1024   # obs_est(500字符档) + answer_reserve
    n_rw = rewrite_budget_lines(turns, need)
    _expect = budget_hint_line(7000, need)     # 7000 ≥ 2·need → 宽松档
    check("预算行改写：数值不变、文案换成分档当前形态（宽松档逐字对拍）",
          n_rw == 1 and _expect in turns[1]["body"])
    # 幂等：新形态再改写仍得新形态
    n_rw2 = rewrite_budget_lines(turns, need)
    check("预算行改写幂等（分档形态再改写不漂移）",
          n_rw2 == 1 and turns[1]["body"].count("[budget]") == 1)

    # 【转义事故防护】低额度 → 强警档文案含 \boxed{}，若 re.sub 用字符串形态
    # 替换，\b 会被解析成退格符 \x08 —— 这里直接断言字节存活
    _low = split_turns(_mk_row())
    _low[1]["body"] = "42\n[budget] 100 tokens left in this trajectory."
    rewrite_budget_lines(_low, need)            # 100 < need → 强警档
    check("预算行改写：强警档的 \\boxed{} 字节存活（\\b 不被 re.sub 吃掉）",
          "\\boxed{}" in _low[1]["body"] and "\x08" not in _low[1]["body"])
    check("预算行改写：强警档文案确实是'再调用必然不执行'",
          "would NOT be executed" in _low[1]["body"])

    # final_boxed：取最后一个 assistant 段的 boxed
    check("final_boxed：取末段 boxed", final_boxed(turns) == "42")
    _nb = [dict(t) for t in turns]
    _nb[-1]["text"] = "没给答案"
    check("final_boxed：无 boxed → None", final_boxed(_nb) is None)

    # 低预算合成：cut=last（信息完备），强警档 + 简短收尾 + 原答案
    rng = __import__("random").Random(0)
    sv = synthesize_low_budget(turns, need, rng)
    check("低预算合成：cut 在最后一个工具段（前缀两段保留）",
          sv is not None and len(sv) == 3
          and sv[0]["text"] == turns[0]["text"])
    check("低预算合成：切口工具段被改成强警档",
          "would NOT be executed" in sv[1]["body"])
    check("低预算合成：收尾段给的是原轨迹自身的答案（不编造）",
          "\\boxed{42}" in sv[-1]["text"] and sv[-1]["kind"] == "assistant")
    check("低预算合成：零调用轨迹（无工具段）→ None",
          synthesize_low_budget([turns[0], turns[2]], need, rng) is None)
    # 前缀深拷贝语义：改合成样本不回污染原轨迹
    sv[1]["body"] = "改"
    check("低预算合成：不改原 turns（无共享可变引用）",
          "would NOT be executed" not in turns[1]["body"]
          or turns[1]["body"] != "改")

    # build_dataset：分类 + 每题限额偏好短 + 合成派生
    rows = [
        _mk_row(qk="q1", clen=300),               # 长 → 应被限额挤掉
        _mk_row(qk="q1", clen=100),
        _mk_row(qk="q1", clen=200),
        _mk_row(qk="q1", acc=-1),                 # 答错 → 不要
        _mk_row(qk="q2", cu=0, clen=80, segs=[    # 零调用答对
            {"kind": "assistant", "len": 60, "text": "直接算。\n\\boxed{42}"}]),
        _mk_row(qk="q3", trunc=1),                # 截断 → 不要
    ]
    samples, stats = build_dataset(rows, call_need=need, max_per_q=2,
                                   synth_per_q=1, seed=42)
    q1 = [s for s in samples if s["qk"] == "q1" and s["source"] != "low_budget"]
    check("build_dataset：每题限额 2 且偏好更短（clen=300 被挤掉）",
          sorted(s["clen"] for s in q1) == [100, 200])
    check("build_dataset：答错/截断不进", stats["ok"] == 4)
    check("build_dataset：三类齐（real_tool×2 + zero_call×1 + low_budget×1）",
          stats["per_source"] == {"real_tool": 2, "zero_call": 1,
                                  "low_budget": 1})
    _lb = next(s for s in samples if s["source"] == "low_budget")
    check("build_dataset：low_budget 派生自最短 real_tool 且收尾 boxed 在",
          _lb["qk"] == "q1" and "\\boxed{42}" in _lb["turns"][-1]["text"])

    # compose_system_prompt：预算段自动注入（与 RL 同入口）
    from rlab.sft_data import compose_system_prompt
    sp, sha = compose_system_prompt(max_traj_tokens=8192, answer_reserve=1024)
    check("compose_system_prompt：含预算段且带真实数字、返回 sha6",
          "Token budget:" in sp and "8192 tokens" in sp and len(sha) == 6)

    # main 端到端：临时 dump → jsonl，行内含系统提示与 turns
    with tempfile.TemporaryDirectory() as td:
        dp = os.path.join(td, "traj.jsonl")
        with open(dp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out = os.path.join(td, "sft.jsonl")
        from rlab.sft_data import main as _sd_main
        rc = _sd_main(["--dump", dp, "--out", out, "--max_traj_tokens", "8192",
                       "--answer_reserve", "1024"])
        got = [json.loads(x) for x in open(out, encoding="utf-8")]
        check("sft_data main：端到端产出 jsonl（含 sp_sha/turns/source）",
              rc == 0 and len(got) == 4
              and all("sp_sha" in g and g["turns"] and "source" in g
                      for g in got))
        check("sft_data main：产出行的预算行已是分档形态（含宽松档文案）",
              any("in this trajectory" in t.get("body", "")
                  for g in got for t in g["turns"] if t["kind"] == "tool"))


# ---------------------------------------------------------------- B. encode ----
def test_encode():
    print("[B] encode_example（MockTok）：mask 契约与前缀校验")
    from rlab.sft import collate_right_pad, encode_example
    from rlab.tests.test_native_protocol import MockTok

    tok = MockTok()
    turns = [{"kind": "assistant", "text": "先算，调用工具。"},
             {"kind": "tool", "body": "42\n[budget] 100 tokens left."},
             {"kind": "assistant", "text": "答案是。\n\\boxed{42}"}]
    e = encode_example(tok, "SYS", "问1", turns)
    ids, labels = e["input_ids"], e["labels"]
    check("encode：ids/labels 等长", len(ids) == len(labels) == e["n_tokens"])
    # prompt 前缀（system+user+生成头）全部 -100
    from rlab.protocol import initial_messages, render_chat_ids
    plen = len(render_chat_ids(tok, initial_messages("SYS", "问1"), True, None))
    check("encode：prompt 段全 -100", all(x == -100 for x in labels[:plen]))
    # assistant 段 labels == ids（进 loss），且收尾 IM_END 在 mask 内
    ast_pos = [i for i, x in enumerate(labels) if x != -100]
    check("encode：assistant 位 labels==input_ids",
          all(labels[i] == ids[i] for i in ast_pos))
    check("encode：收尾 im_end 进 mask（『答完就停』拿梯度）",
          ids[ast_pos[-1]] == MockTok.IM_END)
    check("encode：两段 assistant 都进 mask（段数×内容都对）",
          e["n_assistant_tokens"] == len(ast_pos)
          and len(ast_pos) > len("先算，调用工具。"))
    # 工具段：在 prompt 与第二段 assistant 之间的 -100 区间非空
    mid = labels[plen:ast_pos[-1]]
    check("encode：工具段是 -100（沙箱输出永不进 loss）",
          any(x == -100 for x in mid[len("先算，调用工具。") + 1:]))
    # roundtrip：解码含 boxed（文本层对拍）
    check("encode：解码 roundtrip 含 boxed 答案",
          "\\boxed{42}" in tok.decode([i for i in ids if i < 0x10000]))

    # 前缀校验失败 → 抛（模板改写历史时宁可丢样不训错位序列）
    class _BadTok(MockTok):
        rewrite_history = True
    try:
        encode_example(_BadTok(), "SYS", "问1", turns)
        check("encode：前缀校验失败必须抛", False)
    except ValueError:
        check("encode：前缀校验失败必须抛", True)

    # collate：右 pad，labels 补 -100，attn 0/1
    a = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    b = {"input_ids": [7], "labels": [7]}
    i2, l2, at = collate_right_pad([a, b], pad_id=0)
    check("collate：右 pad 成批（形状/pad 值/掩码）",
          i2.tolist() == [[1, 2, 3], [7, 0, 0]]
          and l2.tolist() == [[-100, 2, 3], [7, -100, -100]]
          and at.tolist() == [[1, 1, 1], [1, 0, 0]])


# ---------------------------------------------------------------- C. loss ----
def test_loss():
    print("[C] masked_lm_loss_sum：数值 / 梯度探针 / 空 mask")
    from rlab.sft import masked_lm_loss_sum
    torch.manual_seed(0)
    B, T, V = 2, 6, 11
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[0, :2] = -100     # prompt 位
    labels[1, 4:] = -100     # 工具段/pad 位
    ls, n = masked_lm_loss_sum(logits, labels)
    # 手算对拍：shift 后非 -100 位的 CE 和
    import torch.nn.functional as F
    sl = logits[..., :-1, :].float().reshape(-1, V)
    tl = labels[..., 1:].reshape(-1)
    expect = F.cross_entropy(sl, tl, ignore_index=-100, reduction="sum")
    check("loss：数值与手算 CE（shift+ignore_index）一致",
          abs(ls.item() - expect.item()) < 1e-5
          and n == int((tl != -100).sum()))

    # 【梯度探针】masked 位梯度严格 0、unmasked 位非 0——「只测正向数值
    # 发现不了零梯度/假信号」的教训（docs/08 §3.4 的 mask 数值后果）
    lg = torch.randn(1, 8, V, requires_grad=True)
    lb = torch.randint(0, V, (1, 8))
    lb[0, :3] = -100
    lb[0, 6:] = -100
    ls2, _ = masked_lm_loss_sum(lg, lb)
    ls2.backward()
    # shift 后位置 t 的梯度对应 logits[0, t]；labels[0, t+1]==-100 → 梯度 0
    masked_grad = lg.grad[0, :-1, :][lb[0, 1:] == -100]
    unmasked_grad = lg.grad[0, :-1, :][lb[0, 1:] != -100]
    check("梯度探针：masked 位梯度严格为 0（工具/prompt token 零信号）",
          masked_grad.abs().max().item() == 0.0)
    check("梯度探针：unmasked 位梯度非 0（assistant 确实在学）",
          unmasked_grad.abs().max().item() > 0)

    # 空 mask：loss 为 0 但计算图连通（backward 不炸）
    lg3 = torch.randn(1, 4, V, requires_grad=True)
    ls3, n3 = masked_lm_loss_sum(lg3, torch.full((1, 4), -100))
    ls3.backward()
    check("loss：全 -100 时 loss=0 且 backward 连通", n3 == 0
          and ls3.item() == 0.0 and lg3.grad is not None)


def test_chunked_loss():
    print("[D] chunked_sft_loss：与 HF 内建 labels 路径对拍（tiny Llama）")
    from rlab.sft import chunked_sft_loss
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=1,
                                         num_attention_heads=2,
                                         num_key_value_heads=2,
                                         vocab_size=50, max_position_embeddings=64))
    model.eval()
    ids = torch.randint(0, 50, (1, 20))
    labels = ids.clone()
    labels[0, :5] = -100
    attn = torch.ones_like(ids)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        mine_full, n1 = chunked_sft_loss(model, ids, attn, labels, chunk=64)
        mine_chunked, n2 = chunked_sft_loss(model, ids, attn, labels, chunk=8)
    check("chunked：单 chunk == HF 内建 labels loss（同一条 shift+mask 口径）",
          abs(mine_full.item() - ref.item()) < 1e-4 and n1 == 15)
    check("chunked：多 chunk 只丢边界预测对（20 token / chunk8 = 2 条边界），"
          "loss 不发散",
          n2 == 13 and abs(mine_chunked.item() - mine_full.item()) < 0.05)


# ---------------------------------------------------------------- E. gate ----
def test_gate():
    print("[E] sft_gate：四项门槛")
    from rlab.sft_gate import check_gate
    good = {"acc": 0.62, "fmt": 0.95, "code_rate": 0.8, "code_ok_rate": 0.64,
            "n": 300}
    rows, ok = check_gate(good, 0.60)
    check("gate：四项全过 → True", ok and len(rows) == 4)
    rows, ok = check_gate({**good, "code_rate": 0.5}, 0.60)
    check("gate：code_rate 不过 → False", not ok and not rows[0][3])
    rows, ok = check_gate({**good, "code_ok_rate": 0.4}, 0.60)
    check("gate：code_ok/code_used 比值不过 → False",
          not ok and not rows[2][3])
    rows, ok = check_gate({**good, "acc": 0.55}, 0.60)
    check("gate：acc 低于 BASE → False（冷启动不许伤基础能力）",
          not ok and not rows[3][3])
    rows, ok = check_gate({**good, "code_rate": 0.0}, 0.60)
    check("gate：code_rate=0 时比值门安全（ZeroDivision 不炸）",
          not ok)


# ---------------------------------------------------------------- F. 接线 ----
def test_wiring():
    print("[F] CLI/源码接线")
    import inspect
    from rlab import sft, sft_data, sft_gate
    src_d = inspect.getsource(sft_data)
    src_t = inspect.getsource(sft)
    src_g = inspect.getsource(sft_gate)
    check("sft_data CLI：dump/out/系统提示/预算三件套/限额",
          all(x in src_d for x in
              ('"--dump"', '"--out"', '"--system_prompt_file"',
               '"--max_traj_tokens"', '"--answer_reserve"', '"--max_per_q"',
               '"--synth_per_q"')))
    check("sft_data：预算行只从 budget_hint_line 派生（与线上同一函数）",
          "from rlab.rollout import budget_hint_line" in src_d)
    check("sft CLI：lr/epochs/8bit/max_seq_len/grad_accum",
          all(x in src_t for x in
              ('"--lr"', '"--epochs"', '"--optim_8bit"', '"--max_seq_len"',
               '"--grad_accum"')))
    check("sft：loss 走 chunked_sft_loss（不整物化 T×V logits）",
          "chunked_sft_loss(model" in src_t)
    check("sft：存盘 save_pretrained（vLLM 可直读）+ sft_info.json",
          "save_pretrained" in src_t and "sft_info.json" in src_t)
    check("sft_gate：读数经 analysis.read_eval_result（嵌套壳事故的教训）",
          "from rlab.analysis import read_eval_result" in src_g)
    check("sft_gate：exit code 接 PASS/FAIL（可 && 链进 RL）",
          "return 0 if ok else 1" in src_g)


def main():
    test_sft_data()
    test_encode()
    test_loss()
    test_chunked_loss()
    test_gate()
    test_wiring()
    print(f"\n[PASS] test_sft_cpu 全部 {len(PASS)} 项通过")


if __name__ == "__main__":
    main()
