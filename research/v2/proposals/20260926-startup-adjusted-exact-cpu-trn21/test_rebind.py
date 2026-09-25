"""Stdlib-only literal rebinding proof; never imports candidate or Torch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = REPO / 'research/v2/proposals/20260926-architecture-recursive-exact-cpu-trn21'
P = json.loads((HERE / 'CPU_PLAN.disabled.json').read_text())
Q = json.loads((OLD / 'CPU_PLAN.disabled.json').read_text())
PARENT = '9cb0d53ef81cd4668b9b3687fbacf5f6cc98ea2c'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def mapped(text):
    pairs = [(Q['stage_root'], P['stage_root']),
             (Q['source_candidate_sha256'], P['source_candidate_sha256']),
             ('--no-recursive-schedule', '--no-startup-adjusted-schedule'),
             ('--recursive-schedule', '--startup-adjusted-schedule')]
    pairs += [(old['label'], new['label']) for old, new in zip(Q['probes'], P['probes'])]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


class Rebind(unittest.TestCase):
    def test_parent_files_frozen_and_whole_driver_literal_identity(self):
        for file in ('CPU_PLAN.disabled.json', 'host_driver.py', 'validate_receipts.py', 'compare_checkpoints.py'):
            path = str((OLD / file).relative_to(REPO))
            original = subprocess.check_output(['git', 'show', PARENT + ':' + path], cwd=REPO)
            self.assertEqual(original, (OLD / file).read_bytes())
        expected = (OLD / 'host_driver.py').read_text().replace(Q['stage_root'], P['stage_root'])
        for old, new in zip(Q['probes'], P['probes']):
            expected = expected.replace(old['request_sha256'], new['request_sha256'])
        self.assertEqual(expected, (HERE / 'host_driver.py').read_text())
        for file in ('validate_receipts.py', 'compare_checkpoints.py'):
            self.assertEqual((OLD / file).read_bytes(), (HERE / file).read_bytes())

    def test_all_requests_and_commands_only_selected_factor(self):
        for old, new in zip(Q['probes'], P['probes']):
            self.assertEqual(json.loads(mapped(json.dumps(old['command']))), new['command'])
            self.assertEqual(sha(json.dumps(new['command'], separators=(',', ':')).encode()), new['command_sha256'])
            path = new['request_path']
            self.assertEqual(mapped((OLD / path).read_text()), (HERE / path).read_text())
            self.assertEqual(sha((HERE / path).read_bytes()), new['request_sha256'])
            self.assertNotIn('--recursive-schedule', new['command'])
            self.assertNotIn('--no-recursive-schedule', new['command'])
        for key, hash_key in (('expected_outer_command', 'outer_command_sha256'),
                              ('detached_dispatch_command', 'detached_command_sha256')):
            self.assertEqual(json.loads(mapped(json.dumps(Q[key]))), P[key])
            self.assertEqual(sha(json.dumps(P[key], separators=(',', ':')).encode()), P[hash_key])

    def test_model_protocol_guards_and_resources_unchanged(self):
        for key in ('ordering', 'minimum_available_ram_gib', 'minimum_free_disk_gib', 'full_model',
                    'nproc', 'steps', 'micro_batch', 'timeout_seconds', 'checkpoint_compare_timeout_seconds',
                    'seed', 'max_train_seconds', 'expected_plan_sha256', 'device_lock', 'prepare_dir',
                    'expected_hostname', 'source_global_tokens_per_update', 'source_grad_accum_steps',
                    'diagnostic_tokens_all_ranks_per_update', 'support_sha256'):
            self.assertEqual(P[key], Q[key], key)
        for name in ('g2', 'prepare', 'idle_guard', 'dispatch'):
            self.assertEqual(json.loads(mapped(json.dumps(Q['sources'][name]))), P['sources'][name])
        driver = (HERE / 'host_driver.py').read_text()
        self.assertEqual(driver.count('pass_fds=(lock_fd,)'), 2)
        self.assertEqual(driver.count('require_idle_under_inherited_lock('), 3)
        self.assertIn("'NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS': '1800'", driver)
        self.assertIn("'OMP_NUM_THREADS': '1'", driver)

    def test_real_prepare_five_but_only_three_updates(self):
        raw = (REPO / 'prepare.py').read_bytes()
        self.assertEqual(sha(raw), P['sources']['prepare']['sha256'])
        assignments = {}
        for node in ast.parse(raw).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value.value
        self.assertEqual(assignments['TRAIN_STARTUP_STEPS_EXCLUDED'], 5)
        self.assertEqual(assignments['TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS'], 900)
        scope = P['startup_boundary_scope']
        self.assertEqual(scope['real_prepare_startup_steps_excluded'], 5)
        self.assertEqual(scope['updates_per_rank'], 3)
        self.assertFalse(scope['fifth_boundary_exercised'])
        self.assertTrue(scope['no_stub_prepare'])
        for probe in P['probes']:
            request = json.loads((HERE / probe['request_path']).read_text())
            self.assertEqual(request['prepare_dir'], P['prepare_dir'])
            self.assertEqual(request['prepare_sha256'], sha(raw))
            self.assertNotIn('--stub-prepare', request['command'])
            self.assertNotIn('--real-data', request['command'])

    def test_complete_proposal_and_no_authorization(self):
        proposal = json.loads((HERE / 'EXPERIMENT_PROPOSAL.json').read_text())
        self.assertEqual(proposal['candidate']['commands'], [r['command'] for r in P['probes']])
        self.assertEqual(proposal['candidate']['run_labels'], [r['label'] for r in P['probes']])
        self.assertEqual(proposal['candidate']['source'], P['sources']['on'])
        self.assertEqual(proposal['control'], P['sources']['reference'])
        self.assertEqual(proposal['startup_boundary_scope'], P['startup_boundary_scope'])
        self.assertEqual(proposal['success_criteria'], P['acceptance'])
        self.assertEqual(P['source_candidate_commit'], '7419927c7417b5666a984ec9c2b849feadabaaa9')
        self.assertEqual(P['source_candidate_sha256'], '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb')
        for value in (P, proposal):
            self.assertFalse(value['launch_authorized'])
            self.assertFalse(value['native_launch_authorized'])
            self.assertFalse(value['source_adoption'])
        self.assertFalse(P['host_reserved'])
        self.assertTrue(P['local_tensor_execution_prohibited'])
        self.assertEqual(P['proposed_host'], 'trn21')
        self.assertIsNone(P['results'])
        self.assertEqual(len(set(r['label'] for r in P['probes'])), 4)

    def test_manifest_all_present_files_authenticate(self):
        manifest = json.loads((HERE / 'MANIFEST.sha256.json').read_text())
        self.assertFalse(manifest['launch_authorized'])
        self.assertFalse(manifest['native_launch_authorized'])
        for file, expected in manifest['files'].items():
            self.assertEqual(sha((HERE / file).read_bytes()), expected, file)
        expected_files = {str(p.relative_to(HERE)) for p in HERE.rglob('*')
                          if p.is_file() and p.name != 'MANIFEST.sha256.json' and '__pycache__' not in p.parts}
        self.assertEqual(set(manifest['files']), expected_files)


if __name__ == '__main__':
    unittest.main(verbosity=2)
