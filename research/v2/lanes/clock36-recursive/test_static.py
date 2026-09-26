"""Implementer stdlib source/scalar self-checks. No Torch/model import."""
import argparse
import ast
import copy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[4]
REL = 'workflow/proposals/research-v2/train.py'
BASE_REF = '7419927c7417b5666a984ec9c2b849feadabaaa9'
DONOR_REF = 'fe0969fd311d654e143dc6113614416c8970a424'
BASE = subprocess.check_output(['git', 'show', BASE_REF + ':' + REL], cwd=ROOT).decode()
DONOR = subprocess.check_output(['git', 'show', DONOR_REF + ':' + REL], cwd=ROOT).decode()
SOURCE = (ROOT / REL).read_text()
SOURCE_SHA = '4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea'


def code(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), '<exact-source>', 'exec')


def named(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def assigned(n, name):
    return isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)


@lru_cache(maxsize=4)
def template(text):
    tree = ast.parse(text)
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            nodes.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name in {'startup_adjusted_schedule_elapsed', 'lr_multiplier', 'muon_momentum', 'muon_beta2', 'demon_beta1_scale'}:
            nodes.append(n)
    return code(nodes), tree


def namespace(text):
    compiled, tree = template(text)
    ns = dict(TRAIN_STARTUP_STEPS_EXCLUDED=5, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900, _PROCESS_STARTED=100.)
    exec(compiled, ns)
    return ns, tree


def loop(tree):
    return next(n for n in named(tree, 'main').body if isinstance(n, ast.While))


@lru_cache(maxsize=4)
def schedule_block(tree):
    body = loop(tree).body
    start = next(i for i,n in enumerate(body) if assigned(n, 'schedule_elapsed'))
    end = next(i for i in range(start, len(body)) if isinstance(body[i], ast.For) and isinstance(body[i].target, ast.Name) and body[i].target.id == 'group')
    return code(body[start:end + 1])


def ref_lr(step, elapsed, budget, warmup, kind):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    fraction = .95 if kind == 'muon' else .65
    progress = min(1., max(0., elapsed / max(1e-6, budget)))
    if progress < 1. - fraction:
        return 1.
    remaining = (1. - progress) / fraction
    return remaining * 1. + (1 - remaining) * .05


def ref_momentum(step, progress):
    warm = min(step / 300, 1.)
    momentum = (1 - warm) * .85 + warm * .95
    progress = min(1., max(0., progress))
    if progress > 1. - .95:
        cooldown = (progress - (1. - .95)) / .95
        momentum = .95 + cooldown ** 2 * (.79 - .95)
    return momentum


def ref_beta2(progress):
    progress = min(1., max(0., progress))
    return .95 + max(0., (progress - (1. - .95)) / .95) * (.97 - .95)


def groups():
    return [dict(kind=k, initial_lr=.015, no_lr_decay=flat, initial_betas=(.9,.95), betas=(.9,.95))
            for k in ('muon', 'adamw', 'rmsprop', 'other') for flat in (False, True)]


def evaluate(text, clock, recursive, step, elapsed, budget_started=110., warmup=0, process_started=100.):
    ns, tree = namespace(text)
    ns.update(RECURSIVE_SCHEDULE=recursive, WARMUP_STEPS=warmup, _PROCESS_STARTED=process_started,
              step=step, elapsed=elapsed, budget_started=budget_started, batch_lr_scale=1.,
              args=SimpleNamespace(startup_adjusted_schedule=clock, max_train_seconds=1800.),
              optimizer=SimpleNamespace(param_groups=groups()))
    exec(schedule_block(tree), ns)
    return ns


class Tests(unittest.TestCase):
    def test_pins_and_donor_helpers(self):
        self.assertEqual(hashlib.sha256(BASE.encode()).hexdigest(), '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb')
        self.assertEqual(hashlib.sha256(DONOR.encode()).hexdigest(), '75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f')
        self.assertEqual(hashlib.sha256(SOURCE.encode()).hexdigest(), SOURCE_SHA)
        for name in ('lr_multiplier', 'muon_momentum', 'muon_beta2'):
            self.assertEqual(ast.dump(named(ast.parse(SOURCE), name)), ast.dump(named(ast.parse(DONOR), name)))

    def test_exact_original_bytes_recovered(self):
        # Reverse ONLY the selected donor additions/replacements; byte identity covers
        # every timer, stop, evaluator, model, optimizer and launcher line outside scope.
        text = SOURCE
        for line in ("RECURSIVE_SCHEDULE = False\n", "ARCH_TOGGLE_NAMES += ['RECURSIVE_SCHEDULE']\n\n", "    global RECURSIVE_SCHEDULE\n", "    RECURSIVE_SCHEDULE = args.recursive_schedule\n"):
            self.assertEqual(text.count(line), 1)
            text = text.replace(line, '')
        text = ''.join(line for line in text.splitlines(keepends=True) if "parser.add_argument('--recursive-schedule'" not in line and not line.startswith('        lrm_adam = ') and not line.startswith('            lrm_group = '))
        text = text.replace('RECURSIVE_SCHEDULE or MUON_BETA2_FINAL', 'MUON_BETA2_FINAL')
        text = text.replace('else lrm_group)', 'else lrm)')
        for name in ('lr_multiplier', 'muon_momentum', 'muon_beta2'):
            new = ast.get_source_segment(text, named(ast.parse(text), name))
            old = ast.get_source_segment(BASE, named(ast.parse(BASE), name))
            text = text.replace(new, old)
        text = ''.join(line for line in text.splitlines(keepends=True) if not line.startswith(('# Schedule formulas adapted from Recursive', '# https://github.com/recursive-org/', '# Only the separate LR warmdowns')))
        self.assertEqual(text, BASE)

    def test_cli_toggle_save_restore_and_old_missing_key(self):
        tree = ast.parse(SOURCE)
        option = next(n for n in ast.walk(named(tree, 'main')) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and any(isinstance(a, ast.Constant) and a.value == '--recursive-schedule' for a in n.value.args))
        parser = argparse.ArgumentParser()
        exec(code([option]), dict(parser=parser, argparse=argparse))
        self.assertIs(parser.parse_args([]).recursive_schedule, False)
        self.assertIs(parser.parse_args(['--no-recursive-schedule']).recursive_schedule, False)
        self.assertIs(parser.parse_args(['--recursive-schedule']).recursive_schedule, True)
        ns = dict(ARCH_TOGGLE_NAMES=['RECURSIVE_SCHEDULE'], RECURSIVE_SCHEDULE=False)
        exec(code([named(tree, 'current_toggles'), named(tree, 'restore_eval_toggles')]), ns)
        ns['_EVAL_TOGGLE_DEFAULTS'] = ns['current_toggles']()
        ns['restore_eval_toggles']({'toggles': {'RECURSIVE_SCHEDULE': True}})
        self.assertEqual(ns['current_toggles'](), {'RECURSIVE_SCHEDULE': True})
        ns['restore_eval_toggles']({'toggles': {}})
        self.assertEqual(ns['current_toggles'](), {'RECURSIVE_SCHEDULE': False})

    def test_scalar_four_toggle_grid(self):
        points = (-.1, 0., .05, .35, .5, 1., 1.2)
        points += tuple(math.nextafter(x, y) for x in (.05,.35,1.-.95,1.-.65) for y in (-math.inf,math.inf))
        for step in (0,1,4,5,6,299,300,301):
            for warm in (0,8,300):
                for p in points:
                    for clock in (False,True):
                        off = evaluate(SOURCE,clock,False,step,p*1800,warmup=warm)
                        old = evaluate(BASE,clock,False,step,p*1800,warmup=warm)
                        self.assertEqual(off['optimizer'].param_groups, old['optimizer'].param_groups)
                        on = evaluate(SOURCE,clock,True,step,p*1800,warmup=warm)
                        selected = max(0.,p*1800 + 10 + 30 - 900) if clock and step < 5 else p*1800
                        self.assertEqual(on['schedule_elapsed'], selected)
                        progress = min(1., selected/1800)
                        for g in on['optimizer'].param_groups:
                            mult = min(1.,(step+1)/max(1,warm)) if g['no_lr_decay'] else ref_lr(step, selected,1800,warm,g['kind'])
                            self.assertEqual(g['lr'], g['initial_lr'] * mult)
                            if g['kind'] == 'muon':
                                self.assertEqual(g['momentum'], ref_momentum(step,progress))
                                self.assertEqual(g['beta2'], ref_beta2(progress))

    def test_recursive_ignores_old_quantization_and_donor_identical(self):
        child,_ = namespace(SOURCE)
        donor,_ = namespace(DONOR)
        for ns in (child,donor):
            ns.update(RECURSIVE_SCHEDULE=True,LRM_QUANTUM=7,WARMUP_STEPS=0)
        for p in (-.1,0.,.051,.351,.6789,1.,1.2):
            for kind in ('muon','adamw','rmsprop','other'):
                result=child['lr_multiplier'](301,p*1800,1800,kind)
                self.assertEqual(result,donor['lr_multiplier'](301,p*1800,1800,kind))
                self.assertEqual(result,ref_lr(301,p*1800,1800,0,kind))

    def test_five_boundary_ages_offsets_and_accounting(self):
        ns, tree = namespace(SOURCE)
        fn = ns['startup_adjusted_schedule_elapsed']
        boundary = next(n for n in loop(tree).body if isinstance(n,ast.If) and ast.unparse(n.test) == 'step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED')
        for start in (0.,100.,123456.):
            ns['_PROCESS_STARTED'] = start
            for age in (779.,790.,869.999,870.,870.001,900.,1000.):
                for step in (0,1,4):
                    self.assertEqual(fn(step,age-10,start+10), max(0.,age+30-900))
                env = dict(ns, step=4,time=SimpleNamespace(monotonic=lambda:start+age),recent_step_times=[1000.],startup_allowance_est=0.)
                exec(code([boundary]),env)
                # The fake monotonic value is a rounded float. Independently use
                # its actual subtraction, not the unrounded requested duration.
                observed_age = (start + age) - start
                self.assertEqual(env['excess'],max(0.,observed_age+30-900))
                self.assertEqual(env['recent_step_times'],[])
                self.assertEqual(fn(5,.5+env['excess'],env['budget_started']),.5+env['excess'])

    def test_device_beta2_route_and_negative_mutants(self):
        tree = ast.parse(SOURCE)
        one = next(n for n in ast.walk(named(tree,'MuonAdamW')) if assigned(n,'one_beta2'))
        marker = object()
        env=dict(RECURSIVE_SCHEDULE=True,MUON_BETA2_FINAL=0.,one_b2_t=marker,group={'beta2':.96})
        self.assertIs(eval(compile(ast.Expression(one.value),'<beta2>','eval'),env),marker)
        self.assertIsNot(eval("one_b2_t if MUON_BETA2_FINAL > 0 else 1-group['beta2']",env),marker)
        # Each rejected alternative changes a specified observable, not just a string.
        correct=evaluate(SOURCE,True,True,1,1000.)
        raw=evaluate(SOURCE.replace("elapsed=schedule_elapsed, max_train_seconds=args.max_train_seconds, kind='adamw'", "elapsed=elapsed, max_train_seconds=args.max_train_seconds, kind='adamw'"),True,True,1,1000.)
        self.assertNotEqual(correct['optimizer'].param_groups,raw['optimizer'].param_groups)
        self.assertGreater(correct['schedule_elapsed'],0.)
        post=evaluate(SOURCE,True,True,5,200.)
        self.assertEqual(post['schedule_elapsed'],200.)
        self.assertNotEqual(post['schedule_elapsed'],max(0.,200.+10+30-900))
        flat=next(g for g in correct['optimizer'].param_groups if g['kind']=='muon' and g['no_lr_decay'])
        self.assertNotEqual(flat['lr'],flat['initial_lr']*correct['lrm'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
