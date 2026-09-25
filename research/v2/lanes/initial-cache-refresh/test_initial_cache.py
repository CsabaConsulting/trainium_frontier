"""Source-binding and bounded initial-cache equivalence checks (not native proof)."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ARCHITECTURE = os.environ.get('CACHE_TEST_VARIANT') == 'architecture'
SOURCE = Path(os.environ.get('CACHE_TEST_SOURCE', str(ROOT / 'workflow/proposals/research-v2/train.py')))
BASE_COMMIT = ('1e21c3d797eed764f108c41546b270065ad5650c' if ARCHITECTURE
               else 'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b')
BASE_SHA = ('93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5' if ARCHITECTURE
            else '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3')
RECEIPTS = HERE / 'architecture' if ARCHITECTURE else HERE
ADDED = [
    "    parser.add_argument('--initial-cache-refresh', action=argparse.BooleanOptionalAction, default=False, help='Refresh differentiable weight casts inside the timed first step before its first forward; default OFF.')\n",
    '        if step == 0 and args.initial_cache_refresh:\n',
    '            for m in cached_linears:\n',
    '                m.refresh_weight_cache(COMPUTE_DTYPE)\n',
]
ARGV = ['--n-embd', '128', '--no-ngram-ve', '--no-qk-shift', '--compile-sdpa-direct',
        '--nki-local-conv', '--bf16-norm-output', '--pack-factor', '32',
        '--clip-after-reduce', '--seq-len', '1024', '--no-eval-public',
        '--matrix-lr', '0.015', '--recursive-schedule', '--seed', '42',
        '--max-train-seconds', '1800']
if ARCHITECTURE:
    ARGV.remove('--recursive-schedule')
    # Do not use --unet-8x1024 on a laptop: it overrides --n-embd to 1024.
    # These existing component flags give the same architecture at width 128.
    ARGV += ['--depth', '8', '--unet-skip', '--partial-attn-8', '--attention-layers', '0,2,4,6,7']

def base_bytes():
    return subprocess.check_output(['git', 'show', BASE_COMMIT + ':workflow/proposals/research-v2/train.py'], cwd=ROOT)

def code(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), '<actual source AST>', 'exec')

def parts():
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.While))
    return tree, main, loop

def binding(n):
    return n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None

def parse(argv):
    tree, main, _ = parts()
    ns = {'argparse': argparse, 'os': SimpleNamespace(environ={}), 'TRAIN_TIME_BUDGET_SECONDS': 1800}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            ast.literal_eval(node.value)
        except (ValueError, TypeError):
            if binding(node) not in {'MAX_TRAIN_SECONDS', 'EVAL_PREFER_EMA'}:
                continue
        exec(code([node]), ns)
    stop = next(i for i, n in enumerate(main.body) if binding(n) == 'args')
    exec(code(main.body[:stop]), ns)
    return ns['parser'].parse_args(argv)

class SourceTests(unittest.TestCase):
    def test_only_four_lines_added_and_original_bytes_recoverable(self):
        base = base_bytes()
        self.assertEqual(hashlib.sha256(base).hexdigest(), BASE_SHA)
        restored = SOURCE.read_text()
        # Remove the complete guarded block once, not another post-update call.
        for addition in (ADDED[0], ''.join(ADDED[1:])):
            self.assertEqual(restored.count(addition), 1)
            restored = restored.replace(addition, '', 1)
        self.assertEqual(restored.encode(), base)

    def test_parser_default_off_explicit_off_on_no_other_default_change(self):
        baseline = vars(parse([]))
        self.assertFalse(baseline['initial_cache_refresh'])
        for argv, expected in ((['--no-initial-cache-refresh'], False),
                               (['--initial-cache-refresh'], True)):
            actual = vars(parse(argv))
            self.assertEqual(actual['initial_cache_refresh'], expected)
            actual['initial_cache_refresh'] = False
            self.assertEqual(actual, baseline)

    def test_refresh_placement_after_timer_and_before_micro_loop(self):
        _, _, loop = parts()
        start = next(i for i, n in enumerate(loop.body) if binding(n) == 't0')
        refresh = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                       and 'args.initial_cache_refresh' in ast.unparse(n.test))
        micro = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For)
                     and isinstance(n.target, ast.Name) and n.target.id == 'micro_index')
        self.assertLess(start, refresh)
        self.assertLess(refresh, micro)
        self.assertFalse(any(isinstance(n, ast.With) for n in ast.walk(loop.body[refresh])))
        self.assertEqual(ast.unparse(loop.body[refresh].test), 'step == 0 and args.initial_cache_refresh')
        self.assertEqual([n.func.attr for n in ast.walk(loop.body[refresh]) if isinstance(n, ast.Call)],
                         ['refresh_weight_cache'])

    def test_actual_guard_refreshes_once_and_is_noop_when_disabled(self):
        _, _, loop = parts()
        node = next(n for n in loop.body if isinstance(n, ast.If)
                    and 'args.initial_cache_refresh' in ast.unparse(n.test))
        for enabled in (False, True):
            calls = []
            linear = SimpleNamespace(refresh_weight_cache=lambda dtype: calls.append(dtype))
            for step in range(3):
                exec(code([node]), {'step': step, 'args': SimpleNamespace(initial_cache_refresh=enabled),
                                   'cached_linears': [linear], 'COMPUTE_DTYPE': 'bf16'})
            self.assertEqual(calls, ['bf16'] if enabled else [])

    @unittest.skipUnless((RECEIPTS / 'tiny-on.json').exists(), 'run --run-tiny first')
    def test_receipts_all_gradients_updates_optimizer_and_checkpoint_identical(self):
        receipts = {name: json.loads((RECEIPTS / f'tiny-{name}.json').read_text())
                    for name in ('base', 'default', 'off', 'on')}
        self.assertEqual(receipts['base']['train_py_sha256'], BASE_SHA)
        candidate_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        for name, receipt in receipts.items():
            self.assertEqual(receipt['verdict'], 'PASS', name)
            self.assertEqual(receipt['torchrun_rc'], 0)
            self.assertEqual(receipt['steps'], 3)
            self.assertEqual(receipt['micro_batch'], [2, 128])
            self.assertEqual(receipt['nproc'], 4)
            self.assertEqual(receipt['observation_driver_sha256'],
                             hashlib.sha256((HERE / 'g2_observe.py').read_bytes()).hexdigest())
            if name != 'base':
                self.assertEqual(receipt['train_py_sha256'], candidate_sha)
            self.assertEqual(receipt['plan_sha256'], receipts['base']['plan_sha256'])
            for ref, rank in zip(receipts['base']['ranks'], receipt['ranks']):
                self.assertEqual(rank['status'], 'PASS')
                self.assertEqual(rank['losses'], ref['losses'])
                self.assertEqual(rank['final_digests'] if rank['rank'] == 0 else None,
                                 ref['final_digests'] if ref['rank'] == 0 else None)
                obs, robs = rank['cache_observations'], ref['cache_observations']
                self.assertEqual(len(obs['steps']), 3)
                self.assertEqual(obs['steps'], robs['steps'])
                self.assertEqual(obs.get('checkpoint'), robs.get('checkpoint'))
                self.assertEqual(obs.get('checkpoint_file_sha256'), robs.get('checkpoint_file_sha256'))
                self.assertEqual(obs['mlp_custom_backward_calls'], 24 if ARCHITECTURE else 18)
                first = obs['forwards'][0]['caches']
                self.assertGreater(len(first), 0)
                self.assertEqual(len(obs['cache_refreshes']), len(first) * (4 if name == 'on' else 3))
                self.assertTrue(all(x['grad_enabled'] for x in obs['cache_refreshes']))
                for record in first.values():
                    if name == 'on':
                        self.assertTrue(record['connected_first_forward'])
                        self.assertTrue(record['requires_grad'])
                        self.assertEqual(record['master_dtype'], 'torch.float32')
                        self.assertEqual(record['value']['dtype'], 'torch.bfloat16')
                        self.assertNotEqual(record['grad_fn'], 'NoneType')
                    else:
                        self.assertIsNone(record)
                for i in (1, 2):
                    self.assertEqual(obs['forwards'][i], robs['forwards'][i])
                for s in obs['steps']:
                    self.assertEqual(len(s['parameters_after_update']), 65 if ARCHITECTURE else 62)
                    self.assertEqual(set(s['parameters_after_update']), set(s['gradients_before_clip']))
                    self.assertEqual(set(s['parameters_after_update']), set(s['gradients_after_clip']))
                if rank['rank'] == 0:
                    self.assertTrue(rank['checkpoint_check']['ok'])
                    self.assertFalse(any('cached' in k for k in obs['checkpoint']['model_state']))

def run_tiny():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH'] = str(ROOT / 'research/v2/lanes/recursive-schedule/test_support')
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='initial-cache-base-') as folder:
        base = Path(folder) / 'train.py'
        base.write_bytes(base_bytes())  # Exact, isolated test fixture; never a production edit.
        for name, source, delta in [('base', base, []), ('default', SOURCE, []),
                                    ('off', SOURCE, ['--no-initial-cache-refresh']),
                                    ('on', SOURCE, ['--initial-cache-refresh'])]:
            command = [sys.executable, '-B', str(HERE / 'g2_observe.py'), '--train-py', str(source),
                       '--stub-prepare', '--steps', '3', '--nproc', '4', '--micro-rows', '2',
                       '--micro-tokens', '128', '--timeout', '180', '--out', str(RECEIPTS / f'tiny-{name}.json'),
                       '--'] + ARGV + delta
            print(json.dumps({'name': name, 'command': command}), flush=True)
            subprocess.run(command, env=env, cwd=ROOT, check=True, timeout=200)

if __name__ == '__main__':
    if sys.argv[1:] == ['--run-tiny']:
        run_tiny()
    else:
        unittest.main(verbosity=2)
