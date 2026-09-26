"""Bounded checks of actual parser/accumulation AST; never import train.py/data."""
import argparse
import ast
import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name('train.py')
BASE_COMMIT = '7419927c7417b5666a984ec9c2b849feadabaaa9'
BASE_SHA = '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
ADDED = [
    '    global TOTAL_BATCH_SIZE\n',
    "    parser.add_argument('--total-batch-size', type=int, default=TOTAL_BATCH_SIZE, help='Global tokens per optimizer step; positive and divisible by per-microbatch world tokens. Changes gradient accumulation only; no automatic LR scaling. Default 131072.')\n",
    '    if args.total_batch_size <= 0:\n',
    "        parser.error('--total-batch-size must be positive')\n",
    '    TOTAL_BATCH_SIZE = args.total_batch_size\n',
]


def target(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id


def code(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), '<actual source AST>', 'exec')


def source_parts():
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.While))
    return tree, main, loop


def constants(tree):
    nodes = []
    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        try:
            ast.literal_eval(n.value)
        except (TypeError, ValueError):
            if target(n) not in {'MAX_TRAIN_SECONDS', 'EVAL_PREFER_EMA'}:
                continue
        nodes.append(n)
    ns = {'os': SimpleNamespace(environ={})}
    prepare = ast.parse((ROOT / 'prepare.py').read_text())
    exec(code([n for n in prepare.body if target(n) == 'TRAIN_TIME_BUDGET_SECONDS']), ns)
    exec(code(nodes), ns)
    return ns


def parse(argv):
    tree, main, _ = source_parts()
    ns = constants(tree)
    ns.update(argparse=argparse, os=SimpleNamespace(environ={}))
    stop = next(i for i, n in enumerate(main.body) if target(n) == 'args')
    exec(code(main.body[:stop]), ns)
    ns['args'] = ns['parser'].parse_args(argv)
    exec(code(main.body[stop + 1:stop + 3]), ns)
    return ns


def accumulation(batch):
    _, main, _ = source_parts()
    start = next(i for i, n in enumerate(main.body) if target(n) == 'world_tokens_per_micro')
    ns = dict(TOTAL_BATCH_SIZE=batch, DEVICE_BATCH_SIZE=1, PACK_FACTOR=32,
              SEQ_LEN=1024, world_size=4)
    exec(code(main.body[start:start + 3]), ns)
    return ns


class BatchTests(unittest.TestCase):
    def test_only_five_option_binding_lines_added_to_exact_adopted_source(self):
        base = subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:workflow/proposals/research-v2/train.py'], cwd=ROOT)
        self.assertEqual(hashlib.sha256(base).hexdigest(), BASE_SHA)
        restored = SOURCE.read_text()
        for line in ADDED:
            self.assertEqual(restored.count(line), 1)
            restored = restored.replace(line, '', 1)
        self.assertEqual(restored.encode(), base)

    def test_default_explicit_default_and_arm_parser(self):
        for argv, expected in (([], 131072), (['--total-batch-size', '131072'], 131072),
                               (['--total-batch-size', '262144'], 262144)):
            ns = parse(argv)
            self.assertEqual(ns['TOTAL_BATCH_SIZE'], expected)
            self.assertFalse(ns['args'].batch_ramp)
            # This option changes no other parsed defaults.
            baseline, selected = vars(parse([])['args']), vars(ns['args'])
            baseline.pop('total_batch_size')
            selected.pop('total_batch_size')
            self.assertEqual(baseline, selected)

    def test_invalid_nonpositive_or_noninteger_values_fail_parser(self):
        for value in ('0', '-1', '-262144', '1.5', 'nan', 'inf'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse(['--total-batch-size', value])
                self.assertEqual(caught.exception.code, 2)

    def test_actual_production_divisibility_accumulation_and_token_accounting(self):
        _, _, loop = source_parts()
        token_node = next(n for n in loop.body if target(n) == 'step_tokens')
        for batch, gas in ((131072, 1), (262144, 2)):
            ns = accumulation(batch)
            self.assertEqual(ns['world_tokens_per_micro'], 131072)
            self.assertEqual(ns['grad_accum_steps'], gas)
            ns['grad_accum_now'] = gas
            exec(code([token_node]), ns)
            self.assertEqual(ns['step_tokens'], batch)
        for batch in (1, 65536, 131073, 196608, 262143, 262145):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                accumulation(batch)

    def test_ramp_disabled_preserves_accumulation_and_lr_scale_one(self):
        tree, _, _ = source_parts()
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'batch_ramp_grad_accum')
        ns = dict(BATCH_RAMP=False)
        exec(code([fn]), ns)
        for gas in (1, 2):
            for progress in (0.0, 0.05, 0.5, 0.95, 1.0):
                self.assertEqual(ns['batch_ramp_grad_accum'](progress, gas), (gas, 1.0))

    @unittest.skipUnless(os.environ.get('RUN_BATCH_TENSOR_CHECKS') == '1', 'bounded tensor check opt-in')
    def test_actual_micro_loop_matches_global_batch_mean_gradient(self):
        import torch
        torch.set_num_threads(1)
        _, _, loop = source_parts()
        micro = next(n for n in loop.body if isinstance(n, ast.For)
                     and isinstance(n.target, ast.Name) and n.target.id == 'micro_index')
        for gas in (1, 2):
            for scale in (1.0, 128.0):
                local_grads, all_x, all_y = [], [], []
                for rank in range(4):
                    parameter = torch.tensor([[.2, -.1], [.3, .4], [-.2, .5]], dtype=torch.float64, requires_grad=True)
                    g = torch.Generator().manual_seed(42 + rank)
                    batches = [(torch.randn(2, 3, dtype=torch.float64, generator=g),
                                torch.randn(2, 2, dtype=torch.float64, generator=g)) for _ in range(gas)]
                    all_x.extend(x for x, _ in batches)
                    all_y.extend(y for _, y in batches)
                    calls = []
                    def fetch():
                        calls.append(1)
                        x, y = batches[len(calls)] if len(calls) < gas else batches[0]
                        return x, y, {'epoch': 1}
                    ns = dict(grad_accum_now=gas, grad_accum_steps=gas, BATCH_RAMP=False,
                              LOSS_SCALE=scale, x=batches[0][0], y=batches[0][1], step=0,
                              fetch=fetch, model=lambda x, y: ((x @ parameter - y) ** 2).mean(),
                              debug_numerics=lambda *args, **kwargs: None)
                    exec(code([micro]), ns)
                    self.assertEqual(len(calls), gas)
                    local_grads.append(parameter.grad / scale)
                reference = torch.tensor([[.2, -.1], [.3, .4], [-.2, .5]], dtype=torch.float64, requires_grad=True)
                ((torch.cat(all_x) @ reference - torch.cat(all_y)) ** 2).mean().backward()
                # Existing all-reduce AVG after local accumulation, before clipping.
                torch.testing.assert_close(torch.stack(local_grads).mean(0), reference.grad,
                                           rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
