"""Local-only checks for the c5c diagnostic copy; never launches a host job."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shlex
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workflow/tools/stub_prepare'))
import torch


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_source('hash50_test_c5c_base', ROOT / 'submissions/research_v2/train.py')
DIAG = load_source('hash50_test_c5c_diag', HERE / 'train.py')


def science_ast(path: Path) -> str:
    """Remove only the observer's imports, definitions and single hook call."""
    tree = ast.parse(path.read_text())
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body.pop(0)  # the deliberately changed diagnostic-only module header
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Import) and
                         set(alias.name for alias in node.names) <= {'hashlib', 'stat', 'sys'})
                 and not (isinstance(node, ast.FunctionDef) and
                          node.name in {'_table_hash50_scan', '_table_hash50_manifest'})]

    class RemoveOnlyHook(ast.NodeTransformer):
        def visit_If(self, node):
            if (ast.unparse(node.test) == 'step == 49 and args.num_steps == 50'
                    and len(node.body) == 1 and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Call)
                    and isinstance(node.body[0].value.func, ast.Name)
                    and node.body[0].value.func.id == '_table_hash50_manifest'):
                return None
            return self.generic_visit(node)

    RemoveOnlyHook().visit(tree)
    return ast.dump(tree, include_attributes=False)


class FakeParameter:
    def __init__(self, tag: str):
        self.tag = tag
        self.shape = (524288, 128)
        self.grad = None


class FakeModel:
    def __init__(self, keys):
        self.ngram_embeds = {key: types.SimpleNamespace(weight=FakeParameter(key)) for key in keys}
        self.owned = object()

    def named_parameters(self):
        yield 'owned.weight', self.owned
        for key, module in self.ngram_embeds.items():
            yield f'ngram_embeds.{key}.weight', module.weight


def fake_case(keys, rank, out_dir):
    model = FakeModel(keys)
    params = [model.ngram_embeds[key].weight for key in keys]
    state = {}
    for p in params:
        p.grad = FakeParameter(f'{p.tag}.grad')
        state[p] = {'step': 50, 'exp_avg_sq': FakeParameter(f'{p.tag}.moment')}
    adapter = types.SimpleNamespace(rank=rank, world_size=4,
                                    entries=[dict(kind='adamw', group=0, position=0, owner=0)],
                                    parameters=[model.owned], parameter_ids=[id(model.owned)],
                                    summary=dict(elements_by_owner=[1, 0, 0, 0]))
    optimizer = types.SimpleNamespace(param_groups=[dict(kind='adamw', params=[model.owned]),
                                                   dict(kind='rmsprop', params=params)],
                                      state=state, _fresh_start_owned=adapter)
    args = types.SimpleNamespace(num_steps=50, seed=42, eval_public=False,
                                 ngram_ve=True, ngram_ve_trigram=True,
                                 ngram_ve_bf16=True, ngram_ve_flat_lr=True,
                                 ngram_ve_clip_exclude=True, ngram_ve_opt='rmsprop',
                                 ngram_ve_table_mult=64, ngram_ve_dim=128,
                                 adam_every_n=1, ngram_rms_fp32=False,
                                 nki_ngram_rms=False,
                                 ngram_ve_layers='-1' if len(keys) == 2 else '1,-1',
                                 out_dir=str(out_dir))
    return model, optimizer, args


def fake_scan(tensor, *, digest):
    return dict(shape=list(tensor.shape), dtype='torch.bfloat16',
                byte_order=sys.byteorder, bytes=524288 * 128 * 2,
                finite=True, nonzero=1,
                sha256=hashlib.sha256(tensor.tag.encode()).hexdigest() if digest else None)


class Hash50LocalTests(unittest.TestCase):
    def test_contract_proposals_hashes_and_exact_queue_argv(self):
        contract = json.loads((HERE / 'GATE_CONTRACT.json').read_text())
        queue = json.loads((ROOT / 'research/v2/queue.json').read_text())
        template = json.loads((ROOT / 'research/agentic/EXPERIMENT_PROPOSAL.json').read_text())
        immutable = contract['immutable_source']
        for field, path in (
            ('c5c_base_sha256', ROOT / 'submissions/research_v2/train.py'),
            ('diagnostic_only_sha256', HERE / 'train.py'),
            ('prepare_sha256', ROOT / 'prepare.py'),
            ('queue_sha256', ROOT / 'research/v2/queue.json'),
        ):
            self.assertEqual(immutable[field], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIn('DISPATCH_HOLD', contract['status'])
        for authority in ('source_stage_authorized', 'cpu_g2_authorized',
                          'native_30_authorized', 'native_50_authorized',
                          'fresh_public_eval_authorized', 'scored_run_authorized',
                          'submission_authorized'):
            self.assertIs(contract[authority], False)
        base_argv = shlex.split(queue['base_command'])[5:]
        self.assertIn('--no-ngram-ve', base_argv)
        arms = {entry['id']: entry for entry in queue['queue'] if entry['id'] in ('A1', 'A2')}
        for profile in ('A1', 'A2'):
            proposal = json.loads((HERE / f'EXPERIMENT_PROPOSAL_{profile}.json').read_text())
            self.assertTrue(set(template).issubset(proposal))
            self.assertIsNone(proposal['human_approved_by'])
            self.assertIsNone(proposal['verdict'])
            self.assertTrue(all(value is False for value in proposal['gates'].values()))
            self.assertEqual(proposal['predicted_bpb_delta'], 0.0)
            self.assertIn('DISPATCH HOLD', proposal['candidate']['launch_command'])
            bound = contract['profiles'][profile]
            self.assertEqual(bound['label'], proposal['candidate']['run_label'])
            self.assertEqual(bound['ordered_train_argv'][-1], bound['planned_out_dir_must_be_absent'])
            replacement = arms[profile]['argv_delta']['with']
            idx = base_argv.index('--no-ngram-ve')
            expected = (base_argv[:idx] + replacement + base_argv[idx + 1:]
                        + ['--num-steps', '50', '--out-dir', bound['planned_out_dir_must_be_absent']])
            self.assertEqual(bound['ordered_train_argv'], expected)
            self.assertEqual(bound['table_keys'], ['b5', 't5'] if profile == 'A1'
                             else ['b1', 't1', 'b5', 't5'])
            self.assertIsNone(bound['exact_m64_cpu_g2_receipt'])
            self.assertIsNone(bound['independent_native_30_smoke_receipt'])
            self.assertIsNone(bound['one_shot_authorization'])

    def test_only_observer_changes_c5c_science_ast(self):
        self.assertEqual(science_ast(ROOT / 'submissions/research_v2/train.py'),
                         science_ast(HERE / 'train.py'))

    def test_raw_bf16_digest_finite_and_nonzero(self):
        values = torch.tensor([0.0, 1.0, -2.0], dtype=torch.bfloat16)
        result = DIAG._table_hash50_scan(values, digest=True)
        raw = memoryview(values.view(torch.int16).numpy()).cast('B')
        self.assertEqual(result['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['bytes'], 6)
        self.assertEqual(result['nonzero'], 2)
        self.assertTrue(result['finite'])
        self.assertFalse(DIAG._table_hash50_scan(torch.tensor([float('nan')], dtype=torch.bfloat16), digest=False)['finite'])

    def test_two_step_rmsprop_cpu_lockstep_against_c5c(self):
        outputs = []
        for module in (BASE, DIAG):
            table = torch.nn.Parameter(torch.zeros((4, 4), dtype=torch.bfloat16))
            group = dict(kind='rmsprop', params=[table], lr=0.1,
                         betas=(0.0, 0.999), eps=1e-10, weight_decay=0.0)
            optimizer = module.MuonAdamW([group])
            for step in range(2):
                table.grad = torch.full_like(table, 0.125 + step * 0.25)
                optimizer.step()
            weight = memoryview(table.detach().view(torch.int16).numpy()).cast('B')
            moment = memoryview(optimizer.state[table]['exp_avg_sq'].view(torch.int16).numpy()).cast('B')
            outputs.append((hashlib.sha256(weight).hexdigest(), hashlib.sha256(moment).hexdigest(),
                            optimizer.state[table]['step']))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0][2], 2)

    def test_a1_a2_atomic_rank_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='c5c-hash50-test-', dir='/dev/shm') as root:
            for keys, rank in ((('b5', 't5'), 0), (('b1', 't1', 'b5', 't5'), 1)):
                label = ('fs-20260925-v2-c5c-a1-m64d128-trn21-r1-hash50' if len(keys) == 2
                         else 'fs-20260925-v2-c5c-a2-m64d128-trn22-r1-hash50')
                out_dir = Path(root) / label
                model, optimizer, args = fake_case(keys, rank, out_dir)
                with patch.object(DIAG, '_table_hash50_scan', side_effect=fake_scan):
                    path = DIAG._table_hash50_manifest(model, optimizer, args, rank, 4)
                    payload = path.read_bytes()
                    manifest = json.loads(payload)
                    self.assertEqual(manifest['profile'], 'A1' if len(keys) == 2 else 'A2')
                    self.assertEqual(manifest['completed_optimizer_steps'], 50)
                    self.assertEqual(list(manifest['tables']), sorted(keys))
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                    self.assertEqual(stat.S_IMODE(out_dir.stat().st_mode), 0o700)
                    with self.assertRaises(FileExistsError):
                        DIAG._table_hash50_manifest(model, optimizer, args, rank, 4)
                    self.assertEqual(path.read_bytes(), payload)
                    self.assertEqual(list(out_dir.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
