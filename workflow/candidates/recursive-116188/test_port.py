"""Bounded source/arithmetic/timer checks. No imports of train.py or real data."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'submissions/dense6x1024_direct_review_matrixlr015/train.py'
RESEARCH = ROOT / 'workflow/proposals/research-v2/train.py'
CANDIDATE = HERE / 'train.py'
HELPERS = {'lr_multiplier', 'muon_momentum', 'muon_beta2'}


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def namespace(path):
    tree = ast.parse(path.read_text())
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
            except (TypeError, ValueError):
                continue
            nodes.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name in HELPERS | {'demon_beta1_scale'}:
            nodes.append(n)
    result = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), result)
    return result, tree


def schedule_block(tree):
    main = function(tree, 'main')
    loop = next(n for n in main.body if isinstance(n, ast.While))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'lrm' for t in n.targets))
    end = next(i for i in range(start, len(loop.body)) if isinstance(loop.body[i], ast.For)
               and isinstance(loop.body[i].target, ast.Name) and loop.body[i].target.id == 'group')
    return compile(ast.Module(body=loop.body[start:end + 1], type_ignores=[]), '<schedule>', 'exec')


class PortTests(unittest.TestCase):
    def setUp(self):
        self.clean, self.tree = namespace(CANDIDATE)
        self.ref, self.ref_tree = namespace(RESEARCH)
        self.ref['RECURSIVE_SCHEDULE'] = True

    def test_exact_reference_pins_and_no_schedule_option(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(),
                         '116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d')
        self.assertEqual(hashlib.sha256(RESEARCH.read_bytes()).hexdigest(),
                         '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3')
        self.assertNotIn('RECURSIVE_SCHEDULE', CANDIDATE.read_text())
        self.assertNotIn('--recursive-schedule', CANDIDATE.read_text())

    def test_bare_effective_argv_matches_entire_proven_launch_profile(self):
        def parsed(path, argv):
            ns, tree = namespace(path)
            body = function(tree, 'main').body
            end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
            ns.update(argparse=argparse, os=SimpleNamespace(environ={}))
            exec(compile(ast.Module(body=body[:end], type_ignores=[]), '<parser>', 'exec'), ns)
            return vars(ns['parser'].parse_args(argv))
        proven = ['--n-embd', '1024', '--no-ngram-ve', '--no-qk-shift',
                  '--compile-sdpa-direct', '--nki-local-conv', '--bf16-norm-output',
                  '--pack-factor', '32', '--clip-after-reduce', '--seq-len', '1024',
                  '--no-eval-public']
        self.assertEqual(parsed(CANDIDATE, []), parsed(BASE, proven))
        self.assertEqual((self.clean['DEPTH'], self.clean['N_EMBD'], self.clean['SEQ_LEN'],
                          self.clean['PACK_FACTOR'], self.clean['MATRIX_LR']),
                         (6, 1024, 1024, 32, .015))

    def test_schedule_bodies_are_exact_adopted_on_ast(self):
        for name in HELPERS:
            ref_fn = function(self.ref_tree, name)
            on = next(n for n in ref_fn.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == 'RECURSIVE_SCHEDULE')
            clean_fn = function(self.tree, name)
            self.assertEqual(dump(clean_fn.args), dump(ref_fn.args), name)
            self.assertEqual(dump(ast.Module(body=clean_fn.body, type_ignores=[])),
                             dump(ast.Module(body=on.body, type_ignores=[])), name)

    def test_exact_arithmetic_dense_grid_and_float_boundaries(self):
        points = [-1.0, 0.0, 0.05, 1.0 - 0.95, 0.35, 1.0 - 0.65, 0.955, 1.0, 1.2]
        points += [i / 1000 for i in range(1001)]
        for p in (1.0 - 0.95, 1.0 - 0.65, 1.0):
            points += [math.nextafter(p, -math.inf), math.nextafter(p, math.inf)]
        for warmup in (0, 8):
            self.clean['WARMUP_STEPS'] = self.ref['WARMUP_STEPS'] = warmup
            for step in (0, 1, 2, 7, 8, 150, 299, 300, 3600):
                for p in points:
                    for kind in ('muon', 'adamw'):
                        args = step, p * 1800, 1800, kind
                        self.assertEqual(self.clean['lr_multiplier'](*args),
                                         self.ref['lr_multiplier'](*args))
                    self.assertEqual(self.clean['muon_momentum'](step, p),
                                     self.ref['muon_momentum'](step, p))
                    self.assertEqual(self.clean['muon_beta2'](p), self.ref['muon_beta2'](p))

    def test_actual_group_assignments_match_adopted_dense_profile(self):
        clean_code, ref_code = schedule_block(self.tree), schedule_block(self.ref_tree)
        for step in (0, 1, 2, 150, 300, 3600):
            for p in (-0.01, 0, 0.05, 0.35, 0.5, 0.955, 1.0, 1.1):
                groups = [dict(kind=k, initial_lr=lr, no_lr_decay=flat,
                               initial_betas=(0.9, 0.95), betas=(0.9, 0.95))
                          for k, lr, flat in [('muon', .015, False), ('adamw', .3, False),
                                              ('adamw', .006, False), ('adamw', .2, True)]]
                results = []
                for ns, code in ((self.clean, clean_code), (self.ref, ref_code)):
                    ns.update(step=step, elapsed=p * 1800, batch_lr_scale=1.0,
                              WD_DECAY_TO_ZERO=True, DEMON_BETA1=False,
                              args=SimpleNamespace(max_train_seconds=1800),
                              optimizer=SimpleNamespace(param_groups=copy.deepcopy(groups)))
                    exec(code, ns)
                    results.append(ns['optimizer'].param_groups)
                self.assertEqual(*results)

    def test_optimizer_only_changes_dynamic_beta2_transport(self):
        base_tree = ast.parse(BASE.read_text())
        cls = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == 'MuonAdamW')
        original = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == 'MuonAdamW')
        restored = copy.deepcopy(cls)
        assignment = next(n for n in ast.walk(restored) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'one_beta2' for t in n.targets))
        self.assertEqual(dump(assignment.value), dump(ast.Name(id='one_b2_t', ctx=ast.Load())))
        assignment.value = ast.parse("1 - group['beta2']", mode='eval').body
        self.assertEqual(dump(restored), dump(original))
        # All other classes and non-schedule/non-main functions remain byte-equivalent AST.
        for node in base_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in HELPERS | {'main', 'MuonAdamW'}:
                candidate = next(n for n in self.tree.body if type(n) is type(node) and n.name == node.name)
                self.assertEqual(dump(candidate), dump(node), node.name)
        self.assertIn('weights_only=True', ast.get_source_segment(CANDIDATE.read_text(),
                                                                  function(self.tree, 'load_for_eval')))


spec = importlib.util.spec_from_file_location('timer_checks', ROOT /
    'research/v2/proposals/20260925-startup-predictor-reset/test_startup_predictor.py')
timer_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timer_checks)


class CandidateTimerTests(timer_checks.StartupPredictorTests):
    def timer(self):
        return timer_checks.ActualTimer(CANDIDATE.read_text())

    def test_candidate_timer_is_exact_reviewed_116188_reset(self):
        self.assertEqual(dump(ast.Module(body=self.timer().boundary_nodes, type_ignores=[])),
                         dump(ast.Module(body=timer_checks.ActualTimer(
                             timer_checks.prospective_source()).boundary_nodes, type_ignores=[])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
