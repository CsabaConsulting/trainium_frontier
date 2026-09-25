"""Stdlib exact-source/scalar checks; never import Torch or the candidate module."""
import argparse
import ast
import copy
import hashlib
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[4]
REL = 'workflow/proposals/research-v2/train.py'
BASE = subprocess.check_output(['git', 'show', 'b4d83b5d:' + REL], cwd=ROOT).decode()
SOURCE = (ROOT / REL).read_text()
SHA = '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'


def compile_nodes(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), '<exact-source>', 'exec')


def named(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def namespace(text):
    tree = ast.parse(text)
    ns = {'TRAIN_STARTUP_STEPS_EXCLUDED': 5, 'TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS': 900,
          '_PROCESS_STARTED': 100.0}
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            nodes.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name in {
                'startup_adjusted_schedule_elapsed', 'lr_multiplier', 'muon_momentum',
                'muon_beta2', 'demon_beta1_scale'}:
            nodes.append(n)
    exec(compile_nodes(nodes), ns)
    return ns, tree


def loop(tree):
    return next(n for n in named(tree, 'main').body if isinstance(n, ast.While))


def scalars(text, enabled, step, elapsed, budget_started):
    ns, tree = namespace(text)
    body = loop(tree).body
    first = next(i for i, n in enumerate(body) if assigned(n, 'schedule_elapsed' if text == SOURCE else 'lrm'))
    last = next(i for i in range(first, len(body)) if isinstance(body[i], ast.For) and
                isinstance(body[i].target, ast.Name) and body[i].target.id == 'group')
    groups = [dict(kind='muon', initial_lr=.015), dict(kind='adamw', initial_lr=.2, initial_betas=(.9, .95)),
              dict(kind='adamw', initial_lr=.3, no_lr_decay=True, initial_betas=(.8, .95))]
    ns.update(step=step, elapsed=elapsed, budget_started=budget_started, batch_lr_scale=1.,
              args=SimpleNamespace(startup_adjusted_schedule=enabled, max_train_seconds=1800),
              optimizer=SimpleNamespace(param_groups=groups))
    exec(compile_nodes(body[first:last + 1]), ns)
    return {k: ns[k] for k in ('lrm', 'lrm_flat', 'wd_progress', 'demon_b1', 'muon_b2')}, groups


class StaticTests(unittest.TestCase):
    def test_exact_pins_and_original_bytes_recovered(self):
        self.assertEqual(hashlib.sha256(BASE.encode()).hexdigest(), '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c')
        self.assertEqual(hashlib.sha256(SOURCE.encode()).hexdigest(), SHA)
        tree = ast.parse(SOURCE)
        helper = named(tree, 'startup_adjusted_schedule_elapsed')
        lines = SOURCE.splitlines(keepends=True)
        restored = ''.join(lines[:helper.lineno - 1] + lines[helper.end_lineno + 1:])
        restored = ''.join(line for line in restored.splitlines(keepends=True)
                           if "parser.add_argument('--startup-adjusted-schedule'" not in line
                           and not line.startswith('        schedule_elapsed = '))
        restored = restored.replace('elapsed=schedule_elapsed,', 'elapsed=elapsed,')
        restored = restored.replace('min(1.0, schedule_elapsed /', 'min(1.0, elapsed /')
        self.assertEqual(restored, BASE)

    def test_parser_only_new_default_off_option(self):
        node = next(n for n in ast.walk(named(ast.parse(SOURCE), 'main')) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and any(isinstance(a, ast.Constant)
                    and a.value == '--startup-adjusted-schedule' for a in n.value.args))
        parser = argparse.ArgumentParser()
        exec(compile_nodes([node]), {'parser': parser, 'argparse': argparse})
        self.assertIs(parser.parse_args([]).startup_adjusted_schedule, False)
        self.assertIs(parser.parse_args(['--no-startup-adjusted-schedule']).startup_adjusted_schedule, False)
        self.assertIs(parser.parse_args(['--startup-adjusted-schedule']).startup_adjusted_schedule, True)

    def test_off_all_optimizer_scalars_exact_over_grid(self):
        for step in (0, 1, 4, 5, 6, 299, 300, 3000):
            for elapsed in (0., .001, 449.99, 779.52, 790.37, 900., 1000., 1799., 1900.):
                self.assertEqual(scalars(SOURCE, False, step, elapsed, 110.),
                                 scalars(BASE, False, step, elapsed, 110.))

    def test_allowance_cap_margin_and_no_new_clock_read(self):
        ns, tree = namespace(SOURCE)
        fn = ns['startup_adjusted_schedule_elapsed']
        # process_age = elapsed + 10; existing margin30; cap900 => elapsed threshold860.
        for step in range(5):
            for elapsed in (0., 790.37, 859., 860., 861., 900., 1000.):
                self.assertEqual(fn(step, elapsed, 110.), max(0., elapsed - 860.))
        for step in (5, 6, 100):
            for elapsed in (0., 1e-8, 50., 1800.):
                self.assertEqual(fn(step, elapsed, 110.), elapsed)
        helper = named(tree, 'startup_adjusted_schedule_elapsed')
        self.assertEqual([ast.unparse(n.func) for n in ast.walk(helper) if isinstance(n, ast.Call)], ['max'])

    def test_recorded_compile_durations_change_old_not_new_early_schedule(self):
        a, b = 779.52, 790.37
        old_a, old_b = scalars(BASE, False, 1, a, 110.), scalars(BASE, False, 1, b, 110.)
        self.assertNotEqual(old_a, old_b)
        self.assertEqual(scalars(SOURCE, True, 1, a, 110.), scalars(SOURCE, True, 1, b, 110.))
        self.assertEqual(scalars(SOURCE, True, 1, a, 110.)[0]['lrm'], 1.)
        self.assertNotEqual(scalars(SOURCE, True, 1, 1000., 110.), scalars(SOURCE, True, 1, a, 110.))

    def test_reset_and_stop_accounting_exact_source_and_excess(self):
        ns, tree = namespace(SOURCE)
        baseline = ast.parse(BASE)
        body = loop(tree).body
        boundary = next(n for n in body if isinstance(n, ast.If) and
                        'step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED' == ast.unparse(n.test))
        ref_boundary = next(n for n in loop(baseline).body if isinstance(n, ast.If) and
                            'step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED' == ast.unparse(n.test))
        self.assertEqual(ast.dump(boundary), ast.dump(ref_boundary))
        for age, excess in ((800., 0.), (870., 0.), (871., 1.), (900., 30.), (1000., 130.)):
            env = dict(ns, step=4, time=SimpleNamespace(monotonic=lambda: 100. + age),
                       recent_step_times=[age], startup_allowance_est=0.)
            exec(compile_nodes([boundary]), env)
            self.assertEqual(env['excess'], excess)
            self.assertEqual(env['budget_started'], 100. + age - excess)
            self.assertEqual(env['recent_step_times'], [])
            after = 100. + age + .5 - env['budget_started']
            self.assertEqual(ns['startup_adjusted_schedule_elapsed'](5, after, env['budget_started']), excess + .5)
        for name in ('elapsed', 'ramp_progress', 't0', 'dt', 'elapsed_after_step', 'time_budget_hit', 'control'):
            actual = next(n for n in body if assigned(n, name))
            ref = next(n for n in loop(baseline).body if assigned(n, name))
            self.assertEqual(ast.dump(actual), ast.dump(ref), name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
