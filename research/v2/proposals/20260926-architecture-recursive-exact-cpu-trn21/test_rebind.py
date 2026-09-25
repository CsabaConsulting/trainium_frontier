"""Stdlib-only proof that the trn21 successor changes binding constants only."""
import ast
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = REPO / 'research/v2/proposals/20260926-architecture-recursive-exact-cpu'
P = json.loads((HERE / 'CPU_PLAN.disabled.json').read_text())
Q = json.loads((OLD / 'CPU_PLAN.disabled.json').read_text())
MAPPING = [
    (Q['stage_root'], P['stage_root']),
    (Q['prepare_dir'], P['prepare_dir']),
    (Q['expected_hostname'], P['expected_hostname']),
] + [(a['label'], b['label']) for a, b in zip(Q['probes'], P['probes'])]

def mapped(raw):
    for old, new in MAPPING:
        raw = raw.replace(old, new)
    return raw

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

class Rebind(unittest.TestCase):
    def test_original_frozen(self):
        for file in ('CPU_PLAN.disabled.json', 'host_driver.py', 'validate_receipts.py', 'compare_checkpoints.py'):
            original = subprocess.check_output(['git', 'show', '57511815:' + str((OLD/file).relative_to(REPO))], cwd=REPO)
            self.assertEqual(original, (OLD/file).read_bytes())
        self.assertEqual(P['source_candidate_sha256'], Q['source_candidate_sha256'])
        self.assertEqual(P['source_candidate_commit'], Q['source_candidate_commit'])

    def test_driver_only_constants(self):
        before = mapped((OLD/'host_driver.py').read_text())
        for old, new in zip(Q['probes'], P['probes']):
            before = before.replace(old['request_sha256'], new['request_sha256'])
        self.assertEqual(before, (HERE/'host_driver.py').read_text())
        for file in ('validate_receipts.py', 'compare_checkpoints.py'):
            self.assertEqual((OLD/file).read_bytes(), (HERE/file).read_bytes())

    def test_all_commands_and_science_unchanged(self):
        for old, new in zip(Q['probes'], P['probes']):
            self.assertEqual(json.loads(mapped(json.dumps(old['command']))), new['command'])
            self.assertEqual(sha(json.dumps(new['command'], separators=(',', ':')).encode()), new['command_sha256'])
            path = Path(new['request_path'])
            self.assertEqual(mapped((OLD/path).read_text()), (HERE/path).read_text())
            self.assertEqual(sha((HERE/path).read_bytes()), new['request_sha256'])
        for key in ('ordering','minimum_available_ram_gib','minimum_free_disk_gib','full_model','nproc','steps','micro_batch','timeout_seconds','checkpoint_compare_timeout_seconds','seed','max_train_seconds','expected_plan_sha256','acceptance','device_lock','required_gates'):
            self.assertEqual(P[key], Q[key], key)
        for key in ('expected_outer_command','detached_dispatch_command'):
            self.assertEqual(json.loads(mapped(json.dumps(Q[key]))), P[key])

    def test_host_evidence_and_no_release(self):
        binding = json.loads(subprocess.check_output(['git','show','af53137f:research/v2/operator/20260926-trn21-arch3e79-replicate/NATIVE_BINDING.disabled.json'],cwd=REPO))
        preflight = json.loads((REPO/'research/v2/operator/20260926-trn21-profile-recursive6ef7/launched/preflight.json').read_text())
        self.assertEqual(str(Path(binding['prepare_path']).parent),P['prepare_dir'])
        self.assertEqual(binding['prepare_sha256'],P['sources']['prepare']['sha256'])
        self.assertEqual(preflight['hostname'],P['expected_hostname'])
        self.assertEqual(P['proposed_host'],'trn21')
        self.assertFalse(P['launch_authorized'])
        self.assertFalse(P['native_launch_authorized'])
        self.assertFalse(P['host_reserved'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
