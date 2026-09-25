"""Stdlib-only default-OFF port checks against exact3e79 and adopted6ef7."""
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
PATH = 'workflow/proposals/research-v2/train.py'
SOURCE = (ROOT / PATH).read_text()
BASE_COMMIT = 'b4d83b5db66971f3c348837f04f53cba65aa6d33'
SCHEDULE_COMMIT = 'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b'


def blob(commit, path=PATH):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT).decode()


BASE, REFERENCE = blob(BASE_COMMIT), blob(SCHEDULE_COMMIT)


def dump(node):
    return ast.dump(node, include_attributes=False)


def named(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def namespace(text):
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {'lr_multiplier', 'muon_momentum', 'muon_beta2', 'demon_beta1_scale'}:
            nodes.append(node)
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<schedule>', 'exec'), ns)
    return ns, tree


def schedule_block(tree):
    loop = next(n for n in named(tree, 'main').body if isinstance(n, ast.While))
    start = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'lrm' for t in n.targets))
    end = next(i for i in range(start, len(loop.body)) if isinstance(loop.body[i], ast.For) and isinstance(loop.body[i].target, ast.Name) and loop.body[i].target.id == 'group')
    return compile(ast.Module(body=loop.body[start:end + 1], type_ignores=[]), '<group-block>', 'exec')


class PortTests(unittest.TestCase):
    def setUp(self):
        self.base, self.bt = namespace(BASE)
        self.ns, self.tree = namespace(SOURCE)
        self.ref, self.rt = namespace(REFERENCE)

    def test_exact_source_pins(self):
        self.assertEqual(hashlib.sha256(BASE.encode()).hexdigest(), '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c')
        self.assertEqual(hashlib.sha256(REFERENCE.encode()).hexdigest(), '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3')
        self.assertEqual(hashlib.sha256(SOURCE.encode()).hexdigest(), '75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f')

    def test_patch_payload_exactly_existing_package(self):
        actual = subprocess.check_output(['git','diff',BASE_COMMIT,'--unified=0','--',PATH],cwd=ROOT).decode()
        existing = blob(SCHEDULE_COMMIT, 'research/v2/lanes/recursive-schedule/train.patch')
        def payload(text):
            return [line for line in text.splitlines() if line[:1] in ('+', '-') and not line.startswith(('+++','---'))]
        self.assertEqual(payload(actual), payload(existing))

    def test_all_three_helpers_exact_adopted_source(self):
        for name in ('lr_multiplier','muon_momentum','muon_beta2'):
            self.assertEqual(dump(named(self.tree,name)), dump(named(self.rt,name)), name)

    def test_default_off_and_boolean_cli(self):
        self.assertIs(self.ns['RECURSIVE_SCHEDULE'],False)
        node = next(n for n in ast.walk(named(self.tree,'main')) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='--recursive-schedule' for a in n.value.args))
        parser=argparse.ArgumentParser()
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<one-option>','exec'),{'argparse':argparse,'parser':parser})
        self.assertFalse(parser.parse_args([]).recursive_schedule)
        self.assertFalse(parser.parse_args(['--no-recursive-schedule']).recursive_schedule)
        self.assertTrue(parser.parse_args(['--recursive-schedule']).recursive_schedule)

    def test_dense_grid_on_and_off_values(self):
        points=[i/1000 for i in range(1001)]+[-1.0,1.2]
        for p in (1-.95,1-.65,.955,1.0):
            points += [math.nextafter(p,-math.inf),math.nextafter(p,math.inf)]
        for enabled in (False,True):
            self.ns['RECURSIVE_SCHEDULE']=enabled
            ref=self.ref if enabled else self.base
            self.ref['RECURSIVE_SCHEDULE']=enabled
            for warmup in (0,8):
                self.ns['WARMUP_STEPS']=ref['WARMUP_STEPS']=warmup
                for step in (0,1,2,7,8,150,299,300,3600):
                    for p in points:
                        for kind in ('muon','adamw'):
                            args=(step,p*1800,1800)
                            self.assertEqual(self.ns['lr_multiplier'](*args,kind),ref['lr_multiplier'](*args,kind) if enabled else ref['lr_multiplier'](*args))
                        self.assertEqual(self.ns['muon_momentum'](step,p),ref['muon_momentum'](step,p))
                        self.assertEqual(self.ns['muon_beta2'](p),ref['muon_beta2'](p))

    def test_actual_group_updates_including_flat_table_priority(self):
        for enabled in (False,True):
            self.ns['RECURSIVE_SCHEDULE']=enabled
            self.ref['RECURSIVE_SCHEDULE']=enabled
            ref,rt=(self.ref,self.rt) if enabled else (self.base,self.bt)
            for step in (0,1,2,150,300,3600):
                for p in (0,.05,.35,.5,.955,1):
                    groups=[dict(kind=k,initial_lr=.015,no_lr_decay=flat,initial_betas=(.9,.95),betas=(.9,.95)) for k,flat in [('muon',False),('adamw',False),('adamw',True),('rmsprop',True)]]
                    out=[]
                    for ns,tree in ((self.ns,self.tree),(ref,rt)):
                        ns.update(step=step,elapsed=p*1800,batch_lr_scale=1.,args=SimpleNamespace(max_train_seconds=1800),optimizer=SimpleNamespace(param_groups=copy.deepcopy(groups)))
                        exec(schedule_block(tree),ns)
                        out.append(ns['optimizer'].param_groups)
                    self.assertEqual(*out)

    def test_beta2_dispatch_only_optimizer_change(self):
        actual=copy.deepcopy(named(self.tree,'MuonAdamW'))
        expr=next(n for n in ast.walk(actual) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='one_beta2' for t in n.targets))
        sentinel=object()
        env=dict(RECURSIVE_SCHEDULE=True,MUON_BETA2_FINAL=0.,one_b2_t=sentinel,group={'beta2':.9})
        self.assertIs(eval(compile(ast.Expression(expr.value),'<beta2>','eval'),env),sentinel)
        expr.value=ast.parse("one_b2_t if MUON_BETA2_FINAL > 0.0 else 1 - group['beta2']",mode='eval').body
        self.assertEqual(dump(actual),dump(named(self.bt,'MuonAdamW')))

    def test_unchanged_architecture_loader_and_all_other_helpers(self):
        changed={'main','MuonAdamW','lr_multiplier','muon_momentum','muon_beta2'}
        for node in self.bt.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name not in changed:
                self.assertEqual(dump(node),dump(named(self.tree,node.name)),node.name)
        for key in ('TOTAL_BATCH_SIZE','MATRIX_LR','DEPTH','ATTENTION_LAYERS','MUON_NS_STEPS'):
            self.assertEqual(self.ns[key],self.base[key])


if __name__=='__main__':
    unittest.main(verbosity=2)
