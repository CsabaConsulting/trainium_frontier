"""Stdlib-only exact-score port checks; does not import the model or allocate tensors."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'submissions/dense6x1024_direct_review_matrixlr015/train.py'
SOURCE = HERE / 'train.py'
REFERENCE_COMMIT = 'b4d83b5db66971f3c348837f04f53cba65aa6d33'
REFERENCE_PATH = 'workflow/proposals/research-v2/train.py'
REFERENCE_SHA = '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'
REF_TEXT = subprocess.check_output(['git', 'show', f'{REFERENCE_COMMIT}:{REFERENCE_PATH}'], cwd=ROOT).decode()


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def namespace(text):
    tree = ast.parse(text)
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
            except (TypeError, ValueError):
                continue
            nodes.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name in {'lr_multiplier', 'muon_momentum', 'muon_beta2', 'demon_beta1_scale'}:
            nodes.append(n)
    result = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<extracted>', 'exec'), result)
    return result, tree


def parsed(text, argv):
    ns, tree = namespace(text)
    body = function(tree, 'main').body
    end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    ns.update(argparse=argparse, os=SimpleNamespace(environ={}))
    exec(compile(ast.Module(body=body[:end], type_ignores=[]), '<parser>', 'exec'), ns)
    return vars(ns['parser'].parse_args(argv))


def schedule_block(tree):
    loop = next(n for n in function(tree, 'main').body if isinstance(n, ast.While))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'lrm' for t in n.targets))
    end = next(i for i in range(start, len(loop.body)) if isinstance(loop.body[i], ast.For)
               and isinstance(loop.body[i].target, ast.Name) and loop.body[i].target.id == 'group')
    return compile(ast.Module(body=loop.body[start:end + 1], type_ignores=[]), '<schedule>', 'exec')


class PortTests(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text()
        self.base = BASE.read_text()
        self.clean, self.tree = namespace(self.source)
        self.ref, self.ref_tree = namespace(REF_TEXT)

    def test_pinned_references(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(), '116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d')
        self.assertEqual(hashlib.sha256(REF_TEXT.encode()).hexdigest(), REFERENCE_SHA)

    def test_exact_long_launch_and_entire_parser(self):
        original = (BASE.parent / 'launch-command.txt').read_bytes()
        self.assertEqual((HERE / 'launch-command.txt').read_bytes(), original)
        argv = shlex.split(original.decode().split('train.py ', 1)[1])
        actual = parsed(self.source, argv)
        self.assertEqual(actual, parsed(self.base, argv))
        self.assertEqual((actual['n_embd'], actual['seq_len'], actual['pack_factor'], actual['seed'], actual['max_train_seconds'], actual['eval_public']), (1024, 1024, 32, 42, 1800, False))
        self.assertEqual(self.clean['DEPTH'], 8)
        self.assertEqual(self.clean['TOTAL_BATCH_SIZE'], 131072)
        self.assertEqual(self.clean['MATRIX_LR'], .015)
        self.assertEqual(1 * actual['pack_factor'] * actual['seq_len'] * 4, 131072)

    def test_no_new_flag_adapter_or_eval_subclass(self):
        for forbidden in ('--unet', '--initial-cache-refresh', '--recursive-schedule', '_SubmissionEvalGPT', '_install_submission_training', 'RECURSIVE_SCHEDULE'):
            self.assertNotIn(forbidden, self.source)
        self.assertIn('weights_only=True', ast.get_source_segment(self.source, function(self.tree, 'load_for_eval')))
        for name in ('load_for_eval', 'save_checkpoint', 'softcap_cross_entropy', 'lr_multiplier', 'muon_momentum'):
            self.assertEqual(dump(function(self.tree, name)), dump(function(ast.parse(self.base), name)), name)

    def test_unchanged_helpers_kernels_optimizer_and_model_components(self):
        changed = {'GPT', 'Block', 'main'}
        for node in ast.parse(self.base).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name not in changed:
                counterpart = next(n for n in self.tree.body if type(n) is type(node) and n.name == node.name)
                self.assertEqual(dump(node), dump(counterpart), node.name)

    def test_only_depth_and_attention_constant_changes(self):
        old, _ = namespace(self.base)
        keys = [k for k in old if k.isupper() and not callable(old[k])]
        for key in keys:
            self.assertEqual(self.clean[key], 8 if key == 'DEPTH' else old[key], key)
        self.assertEqual(self.clean['ATTENTION_LAYERS'], (0, 2, 4, 6, 7))

    def test_schedule_arithmetic_and_actual_group_updates_match_scored_profile(self):
        self.ref.update(MATRIX_LR=.015)
        points = [i / 1000 for i in range(1001)]
        for p in (.25, .955, 1.0):
            points += [math.nextafter(p, -math.inf), math.nextafter(p, math.inf)]
        for step in (0, 1, 2, 150, 299, 300, 3600):
            for p in points:
                self.assertEqual(self.clean['lr_multiplier'](step, p * 1800, 1800), self.ref['lr_multiplier'](step, p * 1800, 1800))
                self.assertEqual(self.clean['muon_momentum'](step, p), self.ref['muon_momentum'](step, p))
        for step in (0, 2, 300, 3600):
            for p in (0, .25, .5, .955, 1):
                groups = [dict(kind=k, initial_lr=lr, no_lr_decay=flat, initial_betas=(.9, .95), betas=(.9, .95)) for k, lr, flat in [('muon', .015, False), ('adamw', .3, False), ('adamw', .006, False), ('adamw', .2, True)]]
                results = []
                for ns, tree in ((self.clean, self.tree), (self.ref, self.ref_tree)):
                    ns.update(step=step, elapsed=p * 1800, batch_lr_scale=1.0, args=SimpleNamespace(max_train_seconds=1800), optimizer=SimpleNamespace(param_groups=copy.deepcopy(groups)))
                    exec(schedule_block(tree), ns)
                    results.append(ns['optimizer'].param_groups)
                self.assertEqual(*results)

    def test_main_only_initial_refresh_and_reviewed_timer_reset(self):
        restored = self.source.replace('        if step == 0:\n            for m in cached_linears:\n                m.refresh_weight_cache(COMPUTE_DTYPE)\n', '')
        restored = restored.replace('            recent_step_times.clear()\n', '')
        restored = restored.replace('max(recent_step_times[-20:], default=0.0)', 'max(recent_step_times[-20:])')
        self.assertEqual(dump(function(ast.parse(restored), 'main')), dump(function(ast.parse(self.base), 'main')))
        loop = next(n for n in function(self.tree, 'main').body if isinstance(n, ast.While))
        t0 = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 't0' for t in n.targets))
        self.assertEqual(dump(loop.body[t0 + 1].test), dump(ast.parse('step == 0', mode='eval').body))
        self.assertIn('optimizer.zero_grad', ast.unparse(loop.body[t0 + 2]))
        self.assertNotIn('no_grad', ast.unparse(loop.body[t0 + 1]))


spec = importlib.util.spec_from_file_location('timer_checks', ROOT / 'research/v2/proposals/20260925-startup-predictor-reset/test_startup_predictor.py')
timer_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timer_checks)


class CandidateTimerTests(timer_checks.StartupPredictorTests):
    def timer(self):
        return timer_checks.ActualTimer(SOURCE.read_text())

    def test_timer_matches_scored_research(self):
        self.assertEqual(dump(ast.Module(body=self.timer().boundary_nodes, type_ignores=[])), dump(ast.Module(body=timer_checks.ActualTimer(REF_TEXT).boundary_nodes, type_ignores=[])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
