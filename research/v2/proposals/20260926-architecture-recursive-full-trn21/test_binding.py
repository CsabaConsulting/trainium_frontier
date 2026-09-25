"""Stdlib-only frozen full binding checks; no model, host or actual driver execution."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = REPO / 'research/v2/operator/20260926-trn21-arch3e79-replicate'
B = json.loads((HERE / 'NATIVE_BINDING.disabled.json').read_text())
P = json.loads((HERE / 'EXPERIMENT_PROPOSAL.json').read_text())
Q = json.loads((OLD / 'NATIVE_BINDING.disabled.json').read_text())
DRIVER = (HERE / 'host_driver.py').read_text()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO)

def mapped(text):
    return text.replace(Q['label'], B['label']).replace(Q['source_sha256'], B['source_sha256'])

def constants():
    out = {}
    for node in ast.parse(DRIVER).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Path':
                value = value.args[0]
            out[node.targets[0].id] = ast.literal_eval(value)
    return out

class Binding(unittest.TestCase):
    def test_exact_predecessor_and_whole_driver_identity(self):
        raw = blob(B['derivation']['parent_commit'], B['derivation']['parent_driver_path'])
        self.assertEqual(sha(raw), B['derivation']['parent_driver_sha256'])
        self.assertEqual(raw, (OLD / 'host_driver.py').read_bytes())
        expected = mapped(raw.decode()).replace(
            'Exact cross-host architecture replicate then fresh public20M under one lifetime lock.',
            'Exact architecture Recursive stack then fresh public20M under one lifetime lock.')
        self.assertEqual(expected, DRIVER)
        self.assertEqual(sha(DRIVER.encode()), B['driver_sha256'])
        self.assertEqual(P['driver_sha256'], B['driver_sha256'])

    def test_exact_source_and_unchanged_support(self):
        raw = blob(B['source_commit'], B['source_git_path'])
        self.assertEqual(sha(raw), B['source_sha256'])
        self.assertEqual(B['source_sha256'], '75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f')
        self.assertEqual(sha((REPO / 'prepare.py').read_bytes()), B['prepare_sha256'])
        support = REPO / 'research/v2/operator/20260926-common-host-idle-guard'
        self.assertEqual(sha((support / 'idle_guard.py').read_bytes()), B['idle_guard_sha256'])
        self.assertEqual(sha((support / 'dispatch_driver.py').read_bytes()), B['dispatch_driver_sha256'])
        self.assertEqual(B['harness_sha256'], Q['harness_sha256'])
        self.assertNotIn(b'--startup-adjusted-schedule', raw)
        parent = blob(Q['source_commit'], Q['source_git_path']).decode()
        source = raw.decode()
        def section(text, start, end):
            return text[text.index(start):text.index(end, text.index(start))]
        self.assertEqual(section(source, '        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:', '        elapsed_after_step ='),
                         section(parent, '        if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:', '        elapsed_after_step ='))
        self.assertIn('math.isnan(train_loss) or train_loss > 100', source)
        self.assertIn('recent_step_times.clear()', source)

    def test_only_recursive_argv_and_exact_package(self):
        self.assertEqual(B['training_command'], Q['training_command'] + ['--recursive-schedule'])
        self.assertEqual(B['training_command'], P['argv'])
        self.assertEqual(shlex.split((HERE / 'package/launch-command.txt').read_text()), B['training_command'])
        prov = json.loads((HERE / 'package/provenance.json').read_text())
        self.assertEqual(prov['train_sha256'], B['source_sha256'])
        self.assertEqual(shlex.split(prov['launch_command']), B['training_command'])
        self.assertEqual(prov['source_commit'], B['source_commit'])
        self.assertFalse(prov['launch_authorized'])
        for name, expected in B['package_support_sha256'].items():
            self.assertEqual(sha((HERE / 'package' / name).read_bytes()), expected)
        for flag, expected in [('--matrix-lr', '0.015'), ('--seed', '42'), ('--max-train-seconds', '1800'),
                               ('--num-steps', '1000000'), ('--seq-len', '1024'), ('--pack-factor', '32')]:
            self.assertEqual(B['training_command'][B['training_command'].index(flag) + 1], expected)

    def test_env_cache_watchdogs_and_driver_binding(self):
        for key in ('command', 'evaluation_command', 'training_environment', 'evaluation_environment',
                    'outer_command', 'prepare_path', 'harness_path', 'device_lock', 'budgets', 'evaluation'):
            self.assertEqual(B[key], json.loads(mapped(json.dumps(Q[key]))), key)
        c = constants()
        for key, field in [('ROOT','root'), ('LABEL','label'), ('SOURCE_SHA','source_sha256'),
                           ('PREP','prepare_path'), ('HARNESS','harness_path'), ('TRAIN_COMMAND','command'),
                           ('EVAL_COMMAND','evaluation_command'), ('TRAIN_ENV','training_environment'),
                           ('EVAL_ENV','evaluation_environment')]:
            self.assertEqual(c[key], B[field])
        self.assertNotIn('-O', B['outer_command'])
        self.assertEqual(DRIVER.count('pass_fds=(lock_fd,)'), 1)
        self.assertEqual(DRIVER.count('require_idle_under_inherited_lock('), 2)
        self.assertIn("assert eval_fd==lock_fd,'LOCK_FD_CHANGED'", DRIVER)
        self.assertIn("charged_through_process_exit_within_budget", DRIVER)
        self.assertEqual(B['evaluation']['tokens'], 20971520)
        self.assertEqual(B['evaluation']['seq_len'], 1024)
        self.assertEqual(B['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'], '1800')

    def test_source_cpu_cold_receipts_not_new_acceptance(self):
        gate = B['cold_gate']
        data = json.loads((REPO / gate['archive']).read_text())
        raw = data['files'][gate['receipt_key']]
        self.assertEqual(sha(raw['text'].encode()), gate['receipt_sha256'])
        self.assertEqual(sha(raw['text'].encode()), raw['sha256'])
        report = json.loads(raw['text'])
        self.assertEqual(report['returncode'], 0)
        self.assertEqual(report['step_boundaries'], 5)
        self.assertEqual(report['observations'][-1]['step'], 4)
        self.assertEqual(report['observations'][-1]['observed_seconds'], gate['observed_uncapped_fifth_seconds'])
        self.assertLessEqual(gate['observed_uncapped_fifth_seconds'], 900)
        self.assertEqual(len(report['fresh_empty_sdk_caches']), 6)
        preflight = json.loads(data['files']['preflight.json']['text'])
        self.assertEqual(preflight['source_sha256'], B['source_sha256'])
        self.assertTrue(report['charged_through_process_exit_within_budget'])
        cpu = json.loads((REPO / 'research/v2/operator/20260926-trn21-architecture-recursive-cpu/OPERATOR_SUMMARY.json').read_text())
        self.assertEqual(cpu['source_candidate'], B['source_sha256'])
        self.assertEqual(cpu['terminal']['exit_code'], 0)
        self.assertEqual(cpu['strict']['status'], 'FULL_PARAMETER_CPU_RECEIPTS_MATCH')
        self.assertEqual(cpu['checkpoint_comparison']['parameters'], 113256724)
        self.assertFalse(B['stage_authorized'])
        self.assertFalse(B['launch_authorized'])

    def test_comparators_separate_and_no_eligibility_invention(self):
        adoption = json.loads((REPO / B['decision_comparator']['adoption_record']).read_text())
        self.assertEqual(adoption['source_sha256'], B['decision_comparator']['source_sha256'])
        self.assertEqual(adoption['public_20m'][0], B['decision_comparator']['public20m_bpb'])
        self.assertEqual(adoption['result_labels'][0], B['decision_comparator']['label'])
        self.assertEqual(P['primary_comparator'], B['decision_comparator'])
        self.assertEqual(P['secondary_comparator'], B['raw_parent_contrast'])
        self.assertEqual(B['raw_parent_contrast']['public20m_bpb'], .9931291951444726)
        self.assertIn('NOT an adopted', B['raw_parent_contrast']['role'])
        self.assertIn('eligibility', B['raw_parent_contrast']['eligibility'])
        self.assertEqual(B['host'], 'trn21')
        for obj in (B, P):
            self.assertFalse(obj['launch_authorized'])
            self.assertFalse(obj['stage_authorized'])
            self.assertFalse(obj['source_adoption'])
            self.assertFalse(obj['submission_authorized'])
        self.assertFalse(B['host_reserved'])
        self.assertTrue(B['no_auto_next_job'])

    def test_real_preflight_negative_fixtures(self):
        node = next(n for n in ast.parse(DRIVER).body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
        state = {'stop':False, 'disk':11 * 1024**3, 'bad_sha':None, 'cache_missing':False}
        class FakePath:
            def __init__(self, value): self.value = str(value)
            def __truediv__(self, other): return FakePath(self.value + '/' + str(other))
            @property
            def parent(self): return FakePath(self.value.rsplit('/',1)[0])
            def __str__(self): return self.value
            def glob(self, pattern): return []
            def is_dir(self): return not state['cache_missing']
        def digest(path):
            value = str(path)
            if state['bad_sha'] and state['bad_sha'] in value: return 'wrong'
            if value.endswith('package/train.py'): return B['source_sha256']
            if value.endswith('prepare.py'): return B['prepare_sha256']
            return B['harness_sha256']
        ns = {'Path':FakePath,'ROOT':FakePath(B['root']),'PREP':FakePath(B['prepare_path']),
              'HARNESS':FakePath(B['harness_path']),'SOURCE_SHA':B['source_sha256'],
              'PREP_SHA':B['prepare_sha256'],'HARNESS_SHA':B['harness_sha256'],'sha':digest,
              'os':SimpleNamespace(path=SimpleNamespace(lexists=lambda p:state['stop']),
                                   statvfs=lambda p:SimpleNamespace(f_bavail=state['disk'],f_frsize=1))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'exact-preflight','exec',optimize=0),ns)
        ns['preflight'](B['training_environment'])
        for field, bad in [('stop',True),('disk',0),('bad_sha','train.py'),('bad_sha','prepare.py'),
                           ('bad_sha','submission_rehearsal.py'),('cache_missing',True)]:
            old = state[field];state[field] = bad
            with self.assertRaises(AssertionError): ns['preflight'](B['training_environment'])
            state[field] = old

    def test_manifest(self):
        manifest = json.loads((HERE / 'MANIFEST.sha256.json').read_text())
        self.assertFalse(manifest['launch_authorized'])
        for file, expected in manifest['files'].items():
            self.assertEqual(sha((HERE / file).read_bytes()), expected, file)
        files = {str(p.relative_to(HERE)) for p in HERE.rglob('*')
                 if p.is_file() and p.name != 'MANIFEST.sha256.json' and '__pycache__' not in p.parts}
        self.assertEqual(files, set(manifest['files']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
