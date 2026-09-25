"""Stdlib-only tests executing the timer statements extracted from real train.py.

No train-module imports, tensor allocation, Neuron, host contact, or training.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[4]
RESEARCH = ROOT / "workflow/proposals/research-v2/train.py"
PINNED = ROOT / "submissions/research_v2/train.py"
SCORED = ROOT / "submissions/dense6x1024_direct_review_matrixlr015/train.py"
A2_DURATIONS = (1056.75, 903.47, 0.62, 0.63, 0.62)


def target_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def compile_nodes(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), "<actual timer AST>", "exec")


class ActualTimer:
    def __init__(self, source, budget=1800.0):
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.While))
        start = next(i for i, n in enumerate(loop.body) if target_name(n) == "dt")
        end = next(i for i, n in enumerate(loop.body) if target_name(n) == "stop_after_step")
        self.boundary_nodes = loop.body[start:end + 1]
        self.boundary_code = compile_nodes(self.boundary_nodes)
        self.stop_code = compile_nodes([n for n in self.boundary_nodes
                                        if target_name(n) == "time_budget_hit"])
        self.clock = 0.0
        self.env = {
            "time": SimpleNamespace(monotonic=lambda: self.clock),
            "_PROCESS_STARTED": 0.0,
            "args": SimpleNamespace(num_steps=1000000, max_train_seconds=budget),
            "last_loss": SimpleNamespace(item=lambda: 1.0),
        }
        prepare = ast.parse((ROOT / "prepare.py").read_text())
        prepare_names = {"TRAIN_TIME_BUDGET_SECONDS", "TRAIN_STARTUP_STEPS_EXCLUDED",
                         "TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS"}
        exec(compile_nodes([n for n in prepare.body if target_name(n) in prepare_names]), self.env)
        constants = {"CHECKPOINT_RESERVE_SECONDS", "STARTUP_LAUNCH_MARGIN_SECONDS",
                     "MAX_TRAIN_SECONDS"}
        exec(compile_nodes([n for n in tree.body if target_name(n) in constants]), self.env)
        initial = {"loop_started", "budget_started", "step", "recent_step_times",
                   "startup_allowance_est"}
        exec(compile_nodes([n for n in main.body if target_name(n) in initial]), self.env)

    def advance(self, duration):
        self.env["t0"] = self.clock
        self.clock += duration
        exec(self.boundary_code, self.env)
        self.env["step"] += 1
        return self.env["stop_after_step"]

    def predicts_stop(self, elapsed, history, reached_min=True):
        self.env.update(elapsed_after_step=elapsed, recent_step_times=list(history),
                        reached_min_steps=reached_min)
        exec(self.stop_code, self.env)
        return self.env["time_budget_hit"]


def prospective_source():
    """Apply the checked-in single unified hunk in memory; preserve the vehicle."""
    source = SCORED.read_text()
    patch = Path(__file__).with_name("prospective-116188.patch").read_text().splitlines(True)
    hunk = patch[3:]
    if not patch[2].startswith("@@") or any(line.startswith("@@") for line in hunk):
        raise AssertionError("Expected exactly one prospective patch hunk")
    old = "".join(line[1:] for line in hunk if line.startswith((" ", "-")))
    new = "".join(line[1:] for line in hunk if line.startswith((" ", "+")))
    if source.count(old) != 1:
        raise AssertionError("Prospective patch does not uniquely match the scored vehicle")
    return source.replace(old, new, 1)


class StartupPredictorTests(unittest.TestCase):
    def timer(self):
        return ActualTimer(RESEARCH.read_text())

    def test_official_policy_and_empty_initial_history(self):
        timer = self.timer()
        for name, value in (("TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS", 900),
                            ("TRAIN_STARTUP_STEPS_EXCLUDED", 5),
                            ("STARTUP_LAUNCH_MARGIN_SECONDS", 30.0),
                            ("CHECKPOINT_RESERVE_SECONDS", 15.0),
                            ("TRAIN_TIME_BUDGET_SECONDS", 1800),
                            ("MAX_TRAIN_SECONDS", 1800)):
            self.assertEqual(timer.env[name], value)
        self.assertEqual(timer.env["recent_step_times"], [])
        self.assertEqual(timer.env["startup_allowance_est"], 0.0)
        self.assertFalse(timer.predicts_stop(0.0, [], reached_min=False))
        self.assertFalse(timer.predicts_stop(0.0, []))

    def test_a2_trace_old_stops_at_five_fixed_continues_with_same_charge(self):
        old = ActualTimer(PINNED.read_text())
        fixed = self.timer()
        for duration in A2_DURATIONS:
            old_stop = old.advance(duration)
            fixed_stop = fixed.advance(duration)
        self.assertTrue(old_stop)
        self.assertFalse(fixed_stop)
        self.assertEqual(fixed.env["recent_step_times"], [])
        for key in ("observed_startup", "startup_allowance_est", "excess",
                    "budget_started", "elapsed_after_step"):
            self.assertEqual(fixed.env[key], old.env[key])
        self.assertAlmostEqual(fixed.env["observed_startup"], 1962.09)
        self.assertEqual(fixed.env["startup_allowance_est"], 900)
        self.assertAlmostEqual(fixed.env["excess"], 1092.09)
        self.assertAlmostEqual(fixed.env["elapsed_after_step"], 1092.09)
        self.assertFalse(fixed.advance(0.62))
        self.assertAlmostEqual(fixed.env["recent_step_times"][0], 0.62)

    def test_fifth_compile_sample_is_cleared_exactly_at_boundary(self):
        timer = self.timer()
        for duration in (1.0, 2.0, 3.0, 4.0):
            self.assertFalse(timer.advance(duration))
        self.assertEqual(timer.env["recent_step_times"], [1.0, 2.0, 3.0, 4.0])
        timer.advance(1100.0)
        self.assertEqual(timer.env["recent_step_times"], [])
        timer.advance(0.5)
        timer.advance(0.75)
        self.assertEqual(timer.env["recent_step_times"], [0.5, 0.75])

    def test_startup_excess_and_launch_margin_still_charge(self):
        for observed, excess in ((869.0, 0.0), (870.0, 0.0), (900.0, 30.0),
                                 (1056.75, 186.75), (2670.0, 1800.0)):
            with self.subTest(observed=observed):
                timer = self.timer()
                for duration in (0.0, 0.0, 0.0, 0.0, observed):
                    stopped = timer.advance(duration)
                self.assertEqual(timer.env["excess"], excess)
                self.assertEqual(timer.env["elapsed_after_step"], excess)
                self.assertEqual(stopped, excess >= 1785.0)

    def test_warm_steps_and_recent_twenty_max_keep_their_reserve(self):
        timer = self.timer()
        for _ in range(5):
            timer.advance(0.5)
        for duration in (0.62, 0.63, 0.62):
            self.assertFalse(timer.advance(duration))
        self.assertEqual(len(timer.env["recent_step_times"]), 3)
        self.assertFalse(timer.predicts_stop(1784.0, [0.62, 0.63, 0.62]))
        self.assertTrue(timer.predicts_stop(1784.5, [0.62, 0.63, 0.62]))
        self.assertTrue(timer.predicts_stop(1780.0, [10.0] + [0.5] * 19))
        self.assertFalse(timer.predicts_stop(1780.0, [10.0] + [0.5] * 20))

    def test_empty_history_reserves_checkpoint_even_at_near_zero_budget(self):
        timer = self.timer()
        for elapsed, expected in ((1784.999, False), (1785.0, True),
                                  (1799.999, True), (1800.0, True), (1800.001, True)):
            with self.subTest(elapsed=elapsed):
                self.assertEqual(timer.predicts_stop(elapsed, []), expected)
        self.assertTrue(timer.predicts_stop(1784.5, [0.5]))

    def test_negative_synthetic_timing_is_cleared_not_reinterpreted(self):
        timer = self.timer()
        for duration in (-0.1, 0.1, 0.0, 0.0, 0.0):
            timer.advance(duration)
        self.assertEqual(timer.env["recent_step_times"], [])
        self.assertEqual(timer.env["excess"], 0.0)
        self.assertFalse(timer.advance(-0.5))
        # A negative monotonic duration is nonphysical; existing semantics are unchanged.
        self.assertEqual(timer.env["recent_step_times"], [-0.5])
        old = ActualTimer(PINNED.read_text())
        self.assertEqual(timer.predicts_stop(1785.0, [-0.5]),
                         old.predicts_stop(1785.0, [-0.5]))

    def test_step_cap_still_stops(self):
        timer = self.timer()
        timer.env["args"].num_steps = 5
        for _ in range(5):
            stopped = timer.advance(0.1)
        self.assertTrue(stopped)
        self.assertTrue(timer.env["step_cap_hit"])
        self.assertFalse(timer.env["time_budget_hit"])

    def test_historical_pins_and_prospective_patch(self):
        self.assertEqual(hashlib.sha256(PINNED.read_bytes()).hexdigest(),
                         "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586")
        self.assertEqual(hashlib.sha256(SCORED.read_bytes()).hexdigest(),
                         "116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d")
        prospective = ActualTimer(prospective_source())
        research = self.timer()
        self.assertEqual(ast.dump(ast.Module(body=prospective.boundary_nodes, type_ignores=[])),
                         ast.dump(ast.Module(body=research.boundary_nodes, type_ignores=[])))
        for duration in A2_DURATIONS:
            stopped = prospective.advance(duration)
        self.assertFalse(stopped)
        self.assertEqual(prospective.env["recent_step_times"], [])
        self.assertAlmostEqual(prospective.env["excess"], 1092.09)


if __name__ == "__main__":
    unittest.main(verbosity=2)
