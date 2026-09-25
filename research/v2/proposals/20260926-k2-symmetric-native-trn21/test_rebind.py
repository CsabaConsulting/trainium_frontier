"""Mechanical successor proof only; no device/tensor/process launch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = '84d9f94f6b0341efcb7ddcd8ccaf2944481dc1e3'
OLD = 'research/v2/proposals/20260926-k2-name-binding-native/'
P = json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())


def blob(ref, path):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)


Q = json.loads(blob(PARENT,OLD+'NATIVE_BINDING.disabled.json'))


def mapped(text):
    for before,after in (
        ('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1',
         '/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2'),
        (Q['label'],P['label']),
        (Q['support_sha256']['consumer_nki.py'],P['support_sha256']['consumer_nki.py']),
        (Q['support_sha256']['rank_entry.py'],P['support_sha256']['rank_entry.py']),
        ('ip-172-31-32-253','ip-172-31-34-85')):
        text=text.replace(before,after)
    return text


class Mechanical(unittest.TestCase):
    def test_old_packet_all_bytes_preserved(self):
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',PARENT,'--',OLD],cwd=REPO).decode().splitlines()
        self.assertTrue(paths)
        for path in paths:self.assertEqual(blob(PARENT,path),(REPO/path).read_bytes(),path)

    def test_exact_only_identity_mapping(self):
        for name in ('host_driver.py','rank_entry.py','bound-command.txt'):
            self.assertEqual(mapped(blob(PARENT,OLD+name).decode()),(HERE/name).read_text(),name)
        self.assertEqual(blob(PARENT,OLD+'validate_receipts.py'),(HERE/'validate_receipts.py').read_bytes())
        for key in ('command','checker_args','environment','outer_command','dispatch_command'):
            self.assertEqual(json.loads(mapped(json.dumps(Q[key]))),P[key])
        for key in ('cache_keys','timeout_seconds','term_grace_seconds','completion_rule',
                    'postflight','scientific_scope','entry_wrapper','pending_gates'):
            self.assertEqual(Q[key],P[key],key)

    def test_exact_new_source_and_unchanged_checker_oracle(self):
        self.assertEqual(P['source_module_commit'],'9ce2e6a037212aaa76bd85b9b4ab7d311a481757')
        self.assertEqual(P['source_module_path'],'research/v2/lanes/k2-symmetric-consumer')
        for name in ('consumer_nki.py','check_native.py','reference.py'):
            raw=blob(P['source_module_commit'],P['source_module_path']+'/'+name)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),P['support_sha256'][name])
            if name!='consumer_nki.py':
                self.assertEqual(raw,blob(Q['source_module_commit'],Q['source_module_path']+'/'+name))
        kernel=blob(P['source_module_commit'],P['source_module_path']+'/consumer_nki.py').decode()
        self.assertIn('return resident_consumer[2]',kernel)

    def test_original_lock_wait_and_no_rearm(self):
        self.assertEqual(P['original_lock']['inode'],596823)
        self.assertEqual(P['original_lock']['device'],66305)
        self.assertEqual(P['original_lock']['path'],Q['outer_command'][2])
        self.assertIn('27768550',P['prerequisites']['assigned_profile'])
        self.assertIn('fresh20M',P['prerequisites']['current_training'])
        self.assertTrue(P['prerequisites']['no_automatic_release'])
        self.assertTrue(P['retry_classification']['distinct_design'])
        self.assertFalse(P['retry_classification']['rearm_old_d0f0'])
        self.assertFalse(P['retry_classification']['authorization'])
        for key in ('stage_authorized','launch_authorized','source_adoption','operator_assigned',
                    'host_reserved','automatic_full','automatic_eval','automatic_next_job'):
            self.assertFalse(P[key])
        self.assertTrue(P['postflight']['required'])
        self.assertEqual(P['postflight']['artifact'],'postflight-explicit-pids.json')


if __name__=='__main__':
    unittest.main(verbosity=2)
