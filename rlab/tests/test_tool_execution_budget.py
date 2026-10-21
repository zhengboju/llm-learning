"""工具执行预算定向回归：python -m unittest rlab.tests.test_tool_execution_budget"""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from rlab.config import (get_config, inject_budget_prompt,
                         tool_execution_budget_tokens, validate_budget_target,
                         validate_tool_execution_budget)
from rlab.protocol import NATIVE_BAD_WORDS
from rlab.rollout import multi_turn_rollout_group, prompt_messages_for
from rlab.tests.test_native_protocol import MockTok, _SPStub, _json_form
from rlab.train import run_signature, write_run_info

ROOT = Path(__file__).resolve().parents[2]


class ToolExecutionBudgetTests(unittest.TestCase):
    def config(self, **overrides):
        cfg = dict(tool_protocol="native", max_traj_tokens=12288,
                   budget_target=8192, answer_reserve=1024,
                   tool_result_max_chars=500, max_context_tokens=20000,
                   max_rounds=5, round_gen_tokens=8192, sandbox_workers=1,
                   system_prompt="SYS", budget_hint=True)
        cfg.update(overrides)
        return cfg

    def rollout(self, cfg, rounds):
        tok = MockTok()
        class FakeGen:
            def __init__(self):
                self.r = 0
            def generate(self, prompts, sps, use_tqdm=False):
                text = rounds[self.r]
                self.r += 1
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text=text, token_ids=tok.encode(text), finish_reason="stop")])
                    for _ in prompts]
        def runner(code, **kwargs):
            return {"ok": True, "display": "42", "error_type": None}
        return multi_turn_rollout_group(
            FakeGen(), [_SPStub(list(NATIVE_BAD_WORDS))], tok, ["P"], cfg,
            code_runner=runner,
            prompts_messages=prompt_messages_for([{"Q": "Q1"}], cfg))

    def test_defaults_and_validation(self):
        cfg = self.config()
        self.assertEqual(tool_execution_budget_tokens(cfg), 8192)
        self.assertEqual(tool_execution_budget_tokens({"max_traj_tokens": 12288}), 12288)
        self.assertEqual(tool_execution_budget_tokens({}), 0)
        for value in (0, 8192, 12288):
            c = dict(cfg, tool_execution_budget=value)
            validate_tool_execution_budget(c)
            self.assertEqual(tool_execution_budget_tokens(c), value or 8192)
        for change in ({"tool_execution_budget": -1},
                       {"tool_execution_budget": 12289},
                       {"tool_execution_budget": 1290},
                       {"tool_execution_budget": 12288, "max_traj_tokens": 0},
                       {"tool_execution_budget": 12288, "tool_protocol": "fence"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_tool_execution_budget(dict(cfg, **change))
        # 通告容量与执行容量分离；显式执行预算可容纳调用就允许配置。
        c = dict(cfg, budget_target=500, tool_execution_budget=12288)
        validate_budget_target(c)
        validate_tool_execution_budget(c)

    def test_only_execution_gate_changes(self):
        cfg = self.config()
        rounds = ["x" * 7000 + "\n" + _json_form("print(42)"), "\\boxed{42}"]
        old, _, old_stats = self.rollout(cfg, rounds)
        wide, _, wide_stats = self.rollout(dict(cfg, tool_execution_budget=12288), rounds)
        self.assertEqual(old_stats[0]["code_wasted"], 1)
        self.assertEqual(old_stats[0]["code_used"], 0)
        self.assertEqual(wide_stats[0]["code_used"], 1)
        self.assertEqual(wide_stats[0]["code_wasted"], 0)
        self.assertEqual(old[0][0]["ids"], wide[0][0]["ids"])
        self.assertIn("\\boxed{42}", wide[0][-1]["text"])
        tools = [s["text"] for s in wide[0] if s["kind"] == "tool"]
        self.assertEqual(len(tools), 1)
        self.assertIn("[budget]", tools[0])
        # 与只把 budget_target 放宽相比，实际执行相同，但通告内容应不同。
        expanded, _, _ = self.rollout(dict(cfg, budget_target=12288), rounds)
        self.assertNotEqual(tools, [s["text"] for s in expanded[0] if s["kind"] == "tool"])
        explicit, _, explicit_stats = self.rollout(dict(cfg, tool_execution_budget=0), rounds)
        self.assertEqual(old, explicit)
        self.assertEqual(old_stats, explicit_stats)

    def test_execution_cap_still_rejects_late_call(self):
        cfg = self.config(tool_execution_budget=12288)
        rounds = ["x" * 7000 + "\n" + _json_form("print(42)"),
                  "x" * 4000 + "\n" + _json_form("print(42)")]
        _, _, stats = self.rollout(cfg, rounds)
        self.assertEqual(stats[0]["code_used"], 1)
        self.assertEqual(stats[0]["code_wasted"], 1)

    def test_prompt_signature_and_run_info(self):
        cfg = get_config("retool_math", use_wandb=False, tool_protocol="native",
                         max_traj_tokens=12288, budget_target=8192,
                         answer_reserve=1024, budget_hint=True,
                         max_context_tokens=13312)
        wide = copy.deepcopy(cfg)
        wide["tool_execution_budget"] = 12288
        validate_tool_execution_budget(wide)
        self.assertFalse(inject_budget_prompt(wide))
        self.assertEqual(cfg["system_prompt"], wide["system_prompt"])
        self.assertEqual(run_signature(cfg), run_signature(dict(cfg, tool_execution_budget=0)))
        self.assertEqual(run_signature(cfg), run_signature({k: v for k, v in cfg.items()
                                                            if k != "tool_execution_budget"}))
        self.assertIn("-teb12288", run_signature(wide))
        self.assertNotEqual(run_signature(cfg), run_signature(wide))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "run_info.json"
            write_run_info(str(p), wide)
            saved = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(saved["config"]["tool_execution_budget"], 12288)
            self.assertEqual(saved["tool_execution_budget"], 12288)

    def test_eval_config_seam_and_precedence(self):
        tree = ast.parse((ROOT / "eval_vllm_one.py").read_text(encoding="utf-8"))
        start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "args.tool_execution_budget is not None")
        end = next(i for i in range(start, len(tree.body))
                   if isinstance(tree.body[i], ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_tool_exec_effective"
                           for t in tree.body[i].targets))
        block = ast.Module(body=tree.body[start:end + 1], type_ignores=[])
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "mt_cfg" for t in n.targets))
        for cli, saved, effective, source in ((None, None, 8192, "default"),
                                              (None, 12288, 12288, "run_info"),
                                              (0, 12288, 8192, "cli"),
                                              (12288, 0, 12288, "cli")):
            ns = {"args": SimpleNamespace(tool_execution_budget=cli, max_rounds=5,
                                           round_tokens=8192),
                  "_rcfg": self.config(),
                  "_run_cfg": {} if saved is None else {"tool_execution_budget": saved},
                  "tool_execution_budget_tokens": tool_execution_budget_tokens,
                  "validate_tool_execution_budget": validate_tool_execution_budget,
                  "_tool_protocol": "native", "_mtj_eval": 12288, "_ctkw": None}
            exec(compile(block, "<eval-budget-resolution>", "exec"), ns)
            self.assertEqual(ns["_tool_exec_effective"], effective)
            self.assertEqual(ns["_tool_exec_src"], source)
            exec(compile(ast.Module(body=[assign], type_ignores=[]),
                         "<eval-rollout-seam>", "exec"), ns)
            cfg = ns["mt_cfg"]
            self.assertEqual(cfg["budget_target"], 8192)
            self.assertEqual(cfg["max_traj_tokens"], 12288)
            self.assertEqual(tool_execution_budget_tokens(cfg), effective)
            cfg["max_context_tokens"] = 20000
            cfg["system_prompt"] = "SYS"
            _, _, stats = self.rollout(cfg, ["x" * 7000 + "\n" + _json_form("print(42)"),
                                            "\\boxed{42}"])
            self.assertEqual(stats[0]["code_used"], int(effective == 12288))

    def test_audit_and_difficulty_fingerprint(self):
        from rlab.audit_config import budget_geometry
        from rlab.data import load_difficulty_table
        cfg = self.config(tool_execution_budget=12288)
        geometry = budget_geometry(cfg)
        self.assertEqual(geometry["announce"], 8192)
        self.assertEqual(geometry["tool_execution_budget"], 12288)
        self.assertEqual(geometry["tool_execution_cutoff"], 10998)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "difficulty.jsonl"
            row = {"Q": "q", "k": 8, "n_correct": 4,
                   "probe_meta": {"tool_execution_budget": 8192}}
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_difficulty_table(str(p), expected_meta={"tool_execution_budget": 12288},
                                      strict=True)
            del row["probe_meta"]["tool_execution_budget"]
            row["probe_meta"]["sp"] = "same"
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertIn("q", load_difficulty_table(
                str(p), expected_meta={"sp": "same", "tool_execution_budget": 12288}, strict=True))

    def test_eval_cli_and_real_transmission(self):
        # 执行三个入口中真实的 argparse 代码，不加载模型/引擎。
        for path, required in ((ROOT / "rlab" / "eval.py", []),
                               (ROOT / "eval_vllm.py", []),
                               (ROOT / "eval_vllm_one.py", ["--model", "dummy"])):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            scope = next((n.body for n in tree.body if isinstance(n, ast.FunctionDef)
                          and n.name == "main"), tree.body)
            nodes = []
            for node in scope:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
                        nodes.append(node)
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                        nodes.append(node)
            import argparse
            ns = {"argparse": argparse}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
            parser = ns.get("ap", ns.get("parser"))
            self.assertIsNone(parser.parse_args(required).tool_execution_budget)
            self.assertEqual(parser.parse_args(required + ["--tool_execution_budget", "0"])
                             .tool_execution_budget, 0)
            self.assertEqual(parser.parse_args(required + ["--tool_execution_budget", "12288"])
                             .tool_execution_budget, 12288)
        # 接缝锁：单模型评测真实 cfg -> rollout cfg，显式 0 也必须透传。
        src = (ROOT / "eval_vllm_one.py").read_text(encoding="utf-8")
        self.assertIn('"tool_execution_budget":', src)
        self.assertIn('"tool_execution_budget_effective":', src)
        self.assertIn('"tool_execution_budget_src":', src)
        self.assertIn("args.tool_execution_budget is not None", src)
        self.assertIn("validate_tool_execution_budget(_rcfg)", src)


if __name__ == "__main__":
    unittest.main()
