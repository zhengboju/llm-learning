# -*- coding: utf-8 -*-
"""Search-GRPO 的 CPU 回归测试（docs/16）。

覆盖四层，全部不碰 GPU/网络：
  S 协议泛化：search 声明的渲染/解析（两形态）、错工具判 invalid、默认路径零变化；
  C 检索客户端：mock 注入 + 真 HTTP 服务端到端（线程内起服务）+ 故障必须抛异常；
  Q 打分：EM/F1/别名/空答案/载荷注入防护；
  D 数据：离线导入 → 落盘复读 → loader 核 held-out 契约；
  R rollout：FakeGen + mock 检索走完多轮（mask/工具段/打分三方对齐）。

【为什么必须测"故障不静默"】检索服务不通时若被当成"检索为空"→ 全题无答案 →
reward 全 -1 → 组内零方差 → 静默丢弃。日志里只看到丢弃率上涨，看不到任何异常
——这正是本项目最贵的一类 bug（基础设施故障伪装成模型能力差）。
"""
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rlab import protocol as P                                        # noqa: E402
from rlab.config import get_config                                    # noqa: E402
from rlab.search_client import (SearchClient, SearchRunner,           # noqa: E402
                                SearchServiceError, make_search_runner)
from rlab.search_qa import (aliases_of, evaluate_search_answer,       # noqa: E402
                            extract_final_answer, load_search_qa_dev,
                            load_search_qa_train, normalize_answer,
                            total_reward_search_qa)
from rlab.search_server import build_index, make_server               # noqa: E402

# ---- 从正则骨架重建渲染文本（零标签字面量铁律）----


def _render(name, param, payload):
    """按 protocol 的正则骨架拼出"渲染形态"文本——测试串与解析器同源。"""
    pre = P._FN_PRE.replace("\\s*", "\n")
    mid = P._FN_MID.replace("\\s*", "\n")
    head, tail = P._FN_POST.split("(.*?)")
    return (pre + name + mid + param + head.replace("\\s*", "\n")
            + payload + "\n" + tail.replace("\\s*", "\n"))


# =====================================================================
# S 组：协议泛化
# =====================================================================

def test_s1_default_regex_unchanged():
    """泛化后 (code_interpreter, code) 的正则与历史常量逐字相同——旧路径零变化。"""
    assert P._function_call_re("code_interpreter", "code").pattern \
        == P._RE_TOOL_CALL_FUNCTION.pattern


def test_s2_search_call_parses_function_style():
    txt = _render("search", "query", "who wrote hamlet")
    r = P.parse_assistant(txt, "auto", "search", "query")
    assert r.kind == "tool" and r.code == "who wrote hamlet"


def test_s3_search_call_parses_json_style():
    txt = P._TOOL_OPEN + '\n{"name": "search", "arguments": {"query": "peru capital"}}' \
        + P._TOOL_CLOSE
    r = P.parse_assistant(txt, "json", "search", "query")
    assert r.kind == "tool" and r.code == "peru capital"


def test_s4_wrong_tool_is_invalid():
    """工具错配必须显形：search 调用在 code_interpreter 档、code 调用在 search 档
    都判 invalid（有开标记 → 不是 answer）。静默判成 answer 会让"协议失效"伪装成
    "模型给了个错答案"。"""
    s = _render("search", "query", "q")
    c = _render("code_interpreter", "code", "print(1)")
    assert P.parse_assistant(s, "auto").kind == "invalid"
    assert P.parse_assistant(c, "auto", "search", "query").kind == "invalid"


def test_s5_code_path_still_parses():
    r = P.parse_assistant(_render("code_interpreter", "code", "print(1)"), "auto")
    assert r.kind == "tool" and r.code == "print(1)"


def test_s6_no_call_is_answer():
    assert P.parse_assistant("done \\boxed{x}", "auto", "search", "query").kind == "answer"


def test_s7_tool_declaration_and_ids():
    t = P.native_tool("search")
    assert t.name == "search" and t.param == "query"
    assert t.declaration["function"]["name"] == "search"
    assert P.tool_message("sid", "body", "search")["name"] == "search"
    assert P.make_call_id(0, 1, 2, "search") == "search-0-1-2"
    assert P.make_call_id(0, 1, 2) == "code-0-1-2"      # 历史 id 逐字不变


def test_s8_unknown_tool_fails_fast():
    with pytest.raises(KeyError):
        P.native_tool("browse")


def test_s9_resolve_tool_decls():
    assert P.resolve_tool_decls(True) == [P.CODE_TOOL]
    assert P.resolve_tool_decls([P.SEARCH_TOOL]) == [P.SEARCH_TOOL]
    assert P.resolve_tool_decls(False) is None


def test_s10_derives_both_styles_for_search():
    assert P.derive_tool_style(_render("search", "query", "q"), "search", "query") \
        == P.NATIVE_STYLE_FUNCTION
    js = P._TOOL_OPEN + '{"name": "search", "arguments": {"query": "q"}}' + P._TOOL_CLOSE
    assert P.derive_tool_style(js, "search", "query") == P.NATIVE_STYLE_JSON


# =====================================================================
# Q 组：打分
# =====================================================================

def test_q1_normalize():
    assert normalize_answer("The  United States!") == "united states"
    assert normalize_answer("  ") == ""


def test_q2_em_and_aliases():
    gold = {"A": "William Shakespeare", "aliases": ["Shakespeare"]}
    assert evaluate_search_answer(gold, "\\boxed{Shakespeare}")["em"] == 1.0
    assert evaluate_search_answer(gold, "\\boxed{william  shakespeare}")["em"] == 1.0
    assert evaluate_search_answer(gold, "\\boxed{Marlowe}")["em"] == 0.0


def test_q3_f1_partial_credit():
    gold = {"A": "William Shakespeare"}
    r = evaluate_search_answer(gold, "\\boxed{William Blake}")
    assert r["em"] == 0.0 and 0.0 < r["f1"] < 1.0


def test_q4_missing_boxed_counts_as_zero():
    gold = {"A": "Lima"}
    r = evaluate_search_answer(gold, "I am not sure")
    assert r["em"] == 0.0 and r["format"] == 0.0 and r["prediction"] == ""


def test_q5_tool_payload_boxed_not_answer():
    """调用载荷里的 boxed 是脚手架，不得当答案（完整块与未闭合块两种）。"""
    full = P._TOOL_OPEN + '\n{"name": "search", "arguments": {"query": "\\\\boxed{real}"}}' \
        + P._TOOL_CLOSE
    assert extract_final_answer(full) == ""
    unclosed = P._TOOL_OPEN + '\n{"name": "search", "arguments": {"query": "\\\\boxed{real}"}'
    assert extract_final_answer(unclosed) == ""
    # 反证：真的最终答案仍能取到（不是把整条都剥没了）
    assert extract_final_answer("blah \\boxed{real}") == "real"


def test_q6_reward_domain():
    gold = {"A": "Lima"}
    assert total_reward_search_qa(gold, "\\boxed{Lima}")["reward"] == 1.0
    assert total_reward_search_qa(gold, "\\boxed{Cusco}")["reward"] == -1.0
    empty = total_reward_search_qa(gold, "no answer")
    assert empty["reward"] == -1.0 and empty["format"] == -1.0   # 未交卷进分母


def test_q7_empty_gold_raises():
    """空 gold 不能定义对错：静默判 0 会让"数据坏了"伪装成"模型不行"。"""
    with pytest.raises(ValueError):
        evaluate_search_answer({"A": "  "}, "\\boxed{x}")


def test_q8_overlong_and_trunc_shaping():
    gold = {"A": "Lima"}
    base = total_reward_search_qa(gold, "\\boxed{Lima}")
    pen = total_reward_search_qa(gold, "\\boxed{Lima}", completion_len=9000,
                                 max_gen_tokens=8192, overlong_buffer=256,
                                 overlong_shaping=True)
    assert pen["reward"] < base["reward"] and pen["overlong"] > 0


# =====================================================================
# C 组：检索客户端 + 服务（真 HTTP）
# =====================================================================

def _corpus_file(tmpdir, rows):
    p = os.path.join(tmpdir, "corpus.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


CORPUS = [
    {"id": "1", "title": "Hamlet", "text": "Hamlet was written by William Shakespeare."},
    {"id": "2", "title": "Macbeth", "text": "Macbeth is a tragedy by Shakespeare."},
    {"id": "3", "title": "Peru", "text": "The capital of Peru is Lima."},
    {"id": "4", "title": "Lima", "text": "Lima is the largest city of Peru."},
]


@pytest.fixture()
def served(tmpdir):
    """真起一个 HTTP 服务（线程内），返回 (client, server, corpus_id)。"""
    idx = os.path.join(str(tmpdir), "index.sqlite")
    corpus_id = build_index(_corpus_file(str(tmpdir), CORPUS), idx)
    server = make_server(idx, "127.0.0.1", 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]
    client = SearchClient(f"http://127.0.0.1:{port}", timeout=5.0, top_k=3)
    try:
        yield client, server, corpus_id
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


def test_c1_health_and_corpus_pin(served):
    client, _srv, corpus_id = served
    h = client.check_health()
    assert h["ok"] is True and h["corpus_id"] == corpus_id
    assert client.corpus_id == corpus_id


def test_c2_search_returns_passages(served):
    client, _srv, _cid = served
    rows = client.search("capital of Peru")
    assert rows and any("Lima" in r["text"] for r in rows)
    assert all(set(r) == {"title", "text"} for r in rows)


def test_c3_batch_preserves_order(served):
    """批量端点必须按输入顺序回（错序会把 A 题的检索结果塞进 B 题的轨迹）。"""
    client, _srv, _cid = served
    groups = client.search_batch(["capital of Peru", "who wrote Hamlet", "zzzznope"])
    assert len(groups) == 3
    assert any("Lima" in r["text"] for r in groups[0])
    assert any("Shakespeare" in r["text"] for r in groups[1])
    assert groups[2] == []          # 空结果是合法返回，不是故障


def test_c4_empty_query_is_not_error(served):
    client, _srv, _cid = served
    assert client.search("") == []
    assert client.search_batch(["", "   "]) == [[], []]


def test_c5_runner_shape_matches_run_code(served):
    """runner 必须与 run_code 同型（rollout 的执行块按同一组键消费）。"""
    client, _srv, _cid = served
    runner = SearchRunner(client)
    res = runner("capital of Peru", timeout=5.0, mem_mb=256, max_chars=500)
    assert set(res) >= {"ok", "display", "duration", "error_type"}
    assert res["ok"] is True and "Lima" in res["display"]
    assert res["error_type"] == "ok"


def test_c6_display_truncated(served):
    client, _srv, _cid = served
    runner = SearchRunner(client)
    res = runner("Peru", max_chars=20)
    assert len(res["display"]) <= 20


def test_c7_network_failure_raises_not_zero(served):
    """服务不可达必须抛 SearchServiceError——绝不能返回"检索为空"（那会把基础设施
    故障变成全题零分，组内零方差 → 静默丢弃）。"""
    _c, srv, _cid = served
    port = srv.server_address[1]
    srv.shutdown()
    srv.server_close()
    dead = SearchClient(f"http://127.0.0.1:{port}", timeout=1.0, retries=0)
    with pytest.raises(SearchServiceError):
        dead.check_health()
    with pytest.raises(SearchServiceError):
        dead.search("anything")


def test_c8_local_endpoint_parity(tmpdir):
    """local:<path> 档与 HTTP 档同语义（pod 本地索引降级位）。"""
    idx = os.path.join(str(tmpdir), "index.sqlite")
    corpus_id = build_index(_corpus_file(str(tmpdir), CORPUS), idx)
    client = SearchClient(f"local:{idx}", top_k=3)
    assert client.check_health()["corpus_id"] == corpus_id
    rows = client.search("capital of Peru")
    assert rows and any("Lima" in r["text"] for r in rows)
    assert client.search("") == []


def test_c9_build_index_refuses_overwrite(tmpdir):
    """已有目标必须拒绝——防误删旧索引（重建代价是重跑一遍语料）。"""
    idx = os.path.join(str(tmpdir), "index.sqlite")
    build_index(_corpus_file(str(tmpdir), CORPUS), idx)
    with pytest.raises(FileExistsError):
        build_index(_corpus_file(str(tmpdir), CORPUS), idx)


def test_c10_corpus_change_detected(tmpdir):
    """服务端索引被换掉必须能被发现（训练中换语料 = 两个分布混在一起）。"""
    import shutil
    idx = os.path.join(str(tmpdir), "index.sqlite")
    cid1 = build_index(_corpus_file(str(tmpdir), CORPUS), idx)
    client = SearchClient(f"local:{idx}")
    client.check_health()
    other = os.path.join(str(tmpdir), "index2.sqlite")
    cid2 = build_index(_corpus_file(str(tmpdir), CORPUS + [
        {"id": "5", "title": "X", "text": "extra"}]),
        other)
    assert cid1 != cid2
    # 同路径**原地换成另一份内容** → 身份变了，必须 raise（不是静默用新语料）
    shutil.copyfile(other, idx)
    with pytest.raises(SearchServiceError):
        client.search("capital of Peru")


def test_c11_make_search_runner_reads_cfg(tmpdir):
    idx = os.path.join(str(tmpdir), "index.sqlite")
    build_index(_corpus_file(str(tmpdir), CORPUS), idx)
    runner = make_search_runner({"search_endpoint": f"local:{idx}",
                                 "search_top_k": 2, "search_timeout": 5.0,
                                 "search_retries": 0})
    assert runner.check_health()["ok"] is True
    res = runner.run_batch(["capital of Peru", "who wrote Hamlet"], max_chars=500)
    assert len(res) == 2 and all(r["ok"] for r in res)


# =====================================================================
# D 组：数据（离线导入 → 落盘 → loader 核 held-out）
# =====================================================================

def _write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_d1_offline_prepare_roundtrip(tmpdir):
    """上游 prepare 的真输出喂下游 loader（接口接缝用真实产物验证）。"""
    tr = os.path.join(str(tmpdir), "tr.jsonl")
    dv = os.path.join(str(tmpdir), "dv.jsonl")
    _write_rows(tr, [{"question": "Q1?", "answer": ["Lima", "City of Kings"]},
                     {"question": "Q2?", "answer": ["Cusco"]},
                     {"question": "Q2?", "answer": ["Cusco"]}])       # 重复题
    _write_rows(dv, [{"question": "D1?", "answer": ["Iquitos"]}])
    out = os.path.join(str(tmpdir), "out")
    _run_prepare(["--train_input", tr, "--dev_input", dv, "--output_dir", out])
    man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert man["holdout_ok"] is True and man["train_n"] == 2 and man["dev_n"] == 1
    # loader 读真产物
    rows = load_search_qa_train(os.path.join(out, "train.jsonl"),
                                os.path.join(out, "dev.jsonl"))
    assert len(rows) == 2
    # 别名保留（A 本身也是金标准，故含 A）
    assert aliases_of(rows[0]) == ["Lima", "City of Kings"]
    dev = load_search_qa_dev(os.path.join(out, "dev.jsonl"))
    assert len(dev) == 1


def _run_prepare(argv):
    from rlab.prepare_search_qa import main as pmain
    import sys as _s
    old = _s.argv
    _s.argv = ["prepare_search_qa"] + argv
    try:
        pmain()
    finally:
        _s.argv = old


def test_d2_train_dev_overlap_is_stripped(tmpdir):
    """train 里混进 dev 题面：loader 必须当场剔除（契约比文件名的语义重要）。"""
    out = os.path.join(str(tmpdir), "o")
    os.makedirs(out)
    _write_rows(os.path.join(out, "train.jsonl"),
                [{"Q": "same?", "A": "Lima"}, {"Q": "only-train?", "A": "Cusco"}])
    _write_rows(os.path.join(out, "dev.jsonl"), [{"Q": "same?", "A": "Lima"}])
    rows = load_search_qa_train(os.path.join(out, "train.jsonl"),
                                os.path.join(out, "dev.jsonl"))
    assert [r["Q"] for r in rows] == ["only-train?"]


def test_d3_missing_dev_refuses(tmpdir):
    """没有 held-out 就不许训（无法回答"有没有学到"）。"""
    tr = os.path.join(str(tmpdir), "tr.jsonl")
    _write_rows(tr, [{"Q": "q?", "A": "Lima"}])
    with pytest.raises(FileNotFoundError):
        load_search_qa_train(tr, os.path.join(str(tmpdir), "nope.jsonl"))


def test_d4_empty_answer_rejected(tmpdir):
    out = os.path.join(str(tmpdir), "o")
    os.makedirs(out)
    _write_rows(os.path.join(out, "train.jsonl"), [{"Q": "q?", "A": "-"}])
    _write_rows(os.path.join(out, "dev.jsonl"), [{"Q": "d?", "A": "Lima"}])
    with pytest.raises(RuntimeError):
        load_search_qa_train(os.path.join(out, "train.jsonl"),
                             os.path.join(out, "dev.jsonl"))


def test_d5_prepare_refuses_overwrite(tmpdir):
    tr = os.path.join(str(tmpdir), "tr.jsonl")
    dv = os.path.join(str(tmpdir), "dv.jsonl")
    _write_rows(tr, [{"Q": "q?", "A": "Lima"}])
    _write_rows(dv, [{"Q": "d?", "A": "Cusco"}])
    out = os.path.join(str(tmpdir), "o")
    _run_prepare(["--train_input", tr, "--dev_input", dv, "--output_dir", out])
    with pytest.raises(SystemExit):
        _run_prepare(["--train_input", tr, "--dev_input", dv, "--output_dir", out])


# =====================================================================
# R 组：配置与 rollout 接线
# =====================================================================

def test_r1_preset_loads_and_budget_consistent():
    cfg = get_config("search_qa", search_endpoint="http://127.0.0.1:9")
    assert cfg["native_tool_name"] == "search"
    assert cfg["data_task"] == "search_qa"
    assert cfg["tool_protocol"] == "native"
    # 预算自洽（validate_retool_budget 已跑过：能拿到 reserve 就是过了）
    assert cfg["_tool_reserve"] > 0


def test_r2_missing_endpoint_fails_fast():
    with pytest.raises(ValueError):
        get_config("search_qa")


def test_r3_wrong_tool_name_fails_fast():
    with pytest.raises(ValueError):
        get_config("search_qa", search_endpoint="x", native_tool_name="code_interpreter")


def test_r4_search_qa_requires_native():
    with pytest.raises(ValueError):
        get_config("search_qa", search_endpoint="x", tool_protocol="fence")


def test_r5_retool_math_unaffected():
    """泛化不得改动 retool_math 的任何预算/工具项（历史对照位）。"""
    c = get_config("retool_math")
    assert c["native_tool_name"] == "code_interpreter"
    assert (c["max_rounds"], c["round_gen_tokens"], c["max_context_tokens"]) \
        == (4, 2048, 14336)


def test_r6_system_prompt_has_no_protocol_tags():
    """提示里不得出现协议标签字面量（工具声明由模板 tools 段提供；提示再讲一遍
    就是自相矛盾，docs/09 §0.1 第三类灭绝）。允许 \\boxed{<占位>}。"""
    cfg = get_config("search_qa", search_endpoint="x")
    sp = cfg["system_prompt"]
    assert P._TOOL_OPEN not in sp and "tool_call" not in sp
    assert "\\boxed{" in sp


def test_r7_signature_includes_tool_and_endpoint():
    from rlab.train import run_signature
    a = get_config("search_qa", search_endpoint="http://a:1")
    b = get_config("search_qa", search_endpoint="http://b:1")
    for c in (a, b):
        c["all_steps"], c["save_steps"] = 20, 5
    assert run_signature(a) != run_signature(b)      # 换知识库必须可分辨
    assert "-tnsearch" in run_signature(a)


def test_r8_opt_suffix_prefixes_registered():
    """-tn/-se 必须登记为优化器段，否则 search_qa 的 run 崩溃后重启会被自己的
    护栏判成"换了配方"而拒跑。"""
    from rlab.train import _is_opt_suffix
    assert _is_opt_suffix("-tnsearch-seb44f33") is True


def test_r9_load_qas_dispatches_search_qa(tmpdir, monkeypatch):
    """data.load_qas 的 search_qa 分派必须走 search_qa 模块（带 aliases）。"""
    import rlab.data as D
    import rlab.search_qa as SQ
    tr = os.path.join(str(tmpdir), "train.jsonl")
    dv = os.path.join(str(tmpdir), "dev.jsonl")
    _write_rows(tr, [{"Q": "q?", "A": "Lima", "aliases": ["City of Kings"]}])
    _write_rows(dv, [{"Q": "d?", "A": "Cusco"}])
    monkeypatch.setattr(SQ, "_SEARCH_QA_DIR", str(tmpdir))
    rows = D.load_qas("search_qa")
    # aliases_of 归一：A 本身也是金标准，故列表为 [A, *aliases]
    assert len(rows) == 1 and rows[0]["aliases"] == ["Lima", "City of Kings"]


def test_r10_rollout_multi_turn_with_search_runner(tmpdir):
    """真多轮：FakeGen 发出 search 调用 → mock 检索回包 → 续写给答案。

    锁三条契约（与 test_native_protocol 的 N 组同构，这里换成 search 工具）：
      · **执行器被真的消费**（注入的 runner 收到模型写的 query）；
      · 工具段 mask=0、assistant 段 mask=1（工具 token 不进 loss）；
      · 第 2 轮 prompt == 第 1 轮 prompt + assistant ids + 工具段 ids（逐 token）。
    """
    from rlab.rollout import (build_prompt_ids, multi_turn_rollout_group,
                              retool_build_batch, strip_left_pad)
    from rlab.tests.test_native_protocol import MockTok
    from rlab.protocol import NATIVE_BAD_WORDS

    class _C:
        def __init__(self, text, t):
            self.text = text
            self.token_ids = t.encode(text)
            self.finish_reason = "stop"

    class _O:
        def __init__(self, text, t):
            self.outputs = [_C(text, t)]

    t = MockTok()
    # search 形态的调用文本（从 protocol 的 JSON 形态构造，零手写标签）
    search_call = (P._TOOL_OPEN + "\n"
                   + json.dumps({"name": "search",
                                 "arguments": {"query": "who wrote hamlet"}})
                   + "\n" + P._TOOL_CLOSE)

    class FakeGen:
        def __init__(self, rounds):
            self.rounds, self.r, self.seen = rounds, 0, []

        def generate(self, prompts, sps, use_tqdm=False):
            self.seen.append([list(p["prompt_token_ids"]) for p in prompts])
            texts = self.rounds[self.r]
            self.r += 1
            return [_O(x, t) for x in texts]

    class SP:
        def __init__(self):
            self.bad_words = list(NATIVE_BAD_WORDS)
            self.max_tokens = 512

    seen_queries = []

    def mock_runner(query, timeout=None, mem_mb=256, max_chars=500):
        seen_queries.append(query)
        return {"ok": True, "display": "[1] Hamlet\nby William Shakespeare",
                "duration": 0.01, "error_type": "ok"}

    cfg = get_config("search_qa", search_endpoint="http://mock:1")
    cfg = {**cfg, "num_pre_Q": 1, "max_rounds": 5}
    # MockTok 渲染工具声明用的是它自己写死的 code 参数名，故这里把**解析与声明**
    # 都对齐到它支持的口径是做不到的——改为直接验证 search 档的端到端：
    # MockTok._tools_block 只渲染 code 参数；用 search 名 + query 参数时，
    # 声明段不匹配会让 build_next_prompt 的校验① 触发（这正是"渲染与解析不同源
    # 必须炸"的护栏）。故本测试用 MockTok 的 json 形态工具，但把 cfg 的工具名
    # 换成 search，并让 MockTok 的声明同步（子类覆写）——保证同源。
    class SearchMockTok(MockTok):
        def _tools_block(self):
            decl = ('{"type": "function", "function": {"name": "search", '
                    '"parameters": {"type": "object", "properties": '
                    '{"query": {"type": "string"}}, "required": ["query"]}}}')
            return [self.TOOLS_DECL, *self._enc(decl)]

        def _msg_tokens(self, m):
            # 覆写：tool_calls 的载荷键是 query（不是 code）
            role, content = m.get("role"), m.get("content") or ""
            out = [self.IM_START, self.ROLE_PREFIX, *self._enc(role), *self._enc("\n")]
            if role == "assistant":
                if m.get("tool_calls"):
                    tc = m["tool_calls"][0]["function"]
                    args = tc.get("arguments")
                    q = args.get("query") if isinstance(args, dict) else args
                    body = (P._TOOL_OPEN + "\n"
                            + json.dumps({"name": "search",
                                          "arguments": {"query": str(q)}})
                            + "\n" + P._TOOL_CLOSE)
                    txt = (content.strip() + ("\n" if content.strip() else "") + body)
                else:
                    txt = content.strip()
                out.extend(self._enc(txt))
            elif role == "tool":
                out.extend(self._enc("<tool_response>\n"))
                out.extend(self._enc(content))
                out.extend(self._enc("\n</tool_response>"))
            else:
                out.extend(self._enc(content))
            out.append(self.IM_END)
            return out

    st = SearchMockTok()
    msgs = [[{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q1"}]]
    r1 = ["let me search\n" + search_call]
    r2 = ["thanks, the answer is \\boxed{Shakespeare}"]
    fg = FakeGen([r1, r2])
    segs, _full, cs = multi_turn_rollout_group(
        fg, [SP()], st, ["P"], cfg, code_runner=mock_runner, prompts_messages=msgs)

    assert seen_queries == ["who wrote hamlet"], seen_queries      # 执行器真被消费
    assert [s["kind"] for s in segs[0]] == ["assistant", "tool", "assistant"]
    assert cs[0]["code_used"] == 1 and cs[0]["code_ok"] == 1
    assert cs[0]["invalid_final"] == 0 and cs[0]["code_wasted"] == 0

    # 逐 token 续写契约
    base_ids = build_prompt_ids(msgs[0], st, None, tools=[P.SEARCH_TOOL])
    tool_ids = segs[0][1]["ids"]
    expect_r2 = [*base_ids, *st.encode(r1[0]), *tool_ids]
    assert fg.seen[1][0] == expect_r2

    # mask：assistant=1 / 工具段=0
    pids = None
    from rlab.rollout import build_prompt_batch
    pids = build_prompt_batch([{"Q": "Q1"}], cfg, st, prompts_messages=msgs)[1]
    _merged, mask, _per = retool_build_batch(pids, segs[:1], pids.shape[1],
                                             st.pad_token_id)
    a0 = len(st.encode(r1[0]))
    tl = len(tool_ids)
    m0 = mask[0].tolist()
    assert m0[:a0] == [1.0] * a0 and m0[a0:a0 + tl] == [0.0] * tl
    assert tl > 0

    # 打分：assistant 段拼接 → boxed 取到
    asst = "".join(s["text"] for s in segs[0] if s["kind"] == "assistant")
    sc = total_reward_search_qa(["Shakespeare"], asst)
    assert sc["reward"] == 1.0, sc
    assert strip_left_pad is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
