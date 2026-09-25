"""Implementer stdlib tests only; not independent acceptance, no hosts/tensors."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
P=json.loads((HERE/'WARM_BINDING.disabled.json').read_text())
OLD='research/v2/proposals/20260926-architecture-stable-source-pair/'

def sha(data):return hashlib.sha256(data).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
h=load('warm_identity',HERE/'identity_checks.py')
g=load('warm_guard',REPO/'research/v2/operator/20260926-common-host-idle-guard/idle_guard.py')
tree=ast.parse((HERE/'host_driver.py').read_text())
func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='classify_target')
ns={};exec(compile(ast.Module(body=[func],type_ignores=[]),'target-only','exec'),ns)
classify=ns['classify_target']

class Packet(unittest.TestCase):
    def test_exact_original_source_science_six_caches_and_support(self):
        old=json.loads((REPO/OLD/'PAIR_BINDING.disabled.json').read_text())
        phase=P['phases'][0]
        self.assertEqual(len(P['phases']),1)
        self.assertEqual(phase['name'],'W1')
        self.assertEqual(phase['training_command'],old['phases'][0]['training_command'])
        self.assertEqual(P['cache_environment'],old['cache_environment'])
        self.assertEqual(len(P['cache_environment']),6)
        for k,v in P['cache_environment'].items():self.assertEqual(phase['environment'][k],v)
        self.assertEqual(phase['environment']['NEURON_COMPILE_CACHE_URL'],P['cache_environment']['NEURON_COMPILE_CACHE_DIR'])
        for key in ('source_sha256','stable_directory','prepare_path','prepare_sha256','harness_path','harness_sha256','harness_git_object','device_lock'):
            self.assertEqual(P[key],old[key])
        oldenv=old['phases'][1]['environment'].copy();newenv=phase['environment'].copy()
        for k in ('TRN_EXP_LABEL','TORCH_NEURONX_METRICS_DIR'):
            self.assertNotEqual(oldenv.pop(k),newenv.pop(k))
        self.assertEqual(oldenv,newenv)
        self.assertNotIn('--fresh-caches',phase['command'])
        self.assertEqual(phase['command'],[part.replace(old['phases'][1]['root'],phase['root']) for part in old['phases'][1]['command']])
        self.assertEqual((HERE/'identity_checks.py').read_bytes(),(REPO/OLD/'identity_checks.py').read_bytes())
        for name,key in [('identity_checks.py','helper_sha256'),('EXPERIMENT_PROPOSAL.json','proposal_sha256'),('S1-pre-source-identity.json','s1_identity_sha256')]:
            self.assertEqual(sha((HERE/name).read_bytes()),P[key])
        for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
            self.assertEqual(sha((REPO/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes()),P[key])
        self.assertEqual(sha(subprocess.check_output(['git','show',P['source_commit']+':'+P['source_git_path']],cwd=REPO)),P['source_sha256'])
        self.assertEqual(sha(subprocess.check_output(['git','show',P['harness_git_object']],cwd=REPO)),P['harness_sha256'])
        self.assertEqual(sha((HERE/'w1/package/launch-command.txt').read_bytes()),phase['launch_sha256'])
        self.assertEqual(sha((HERE/'w1/package/provenance.json').read_bytes()),phase['provenance_sha256'])

    def test_original_anchor_is_byte_exact_raw_not_rounded_summary(self):
        bundle=json.loads((REPO/'research/v2/operator/20260926-trn22-stable-source-pair/completed/results-bundle.json').read_text())
        raw=bundle['files']['S1-pre-source-identity.json']
        value=(HERE/'S1-pre-source-identity.json').read_bytes()
        self.assertEqual(value,raw['text'].encode())
        self.assertEqual(sha(value),raw['sha256'])
        self.assertEqual(len(value),raw['bytes'])
        anchor=json.loads(value)
        self.assertEqual(anchor['executed_train']['st_ino'],2924146)
        self.assertEqual(anchor['executed_train']['st_mtime_ns'],1790376459638485637)

    def fixture(self,root,fifth=100,wall=103):
        sandbox=root/'train';(sandbox/'out').mkdir(parents=True)
        (sandbox/'out/final.pt').write_bytes(b'fixture-checkpoint')
        (sandbox/'train.py').write_bytes(b'fixture-source');(sandbox/'train.py').chmod(0o444)
        (sandbox/'prepare.py').write_bytes(b'fixture-prepare')
        receipt=dict(returncode=0,mode='train',step_boundaries=5,
                     observations=[dict(step=i,observed_seconds=fifth-4+i) for i in range(5)],
                     wall_seconds=wall,startup_allowance_observed=min(fifth,900),
                     charged_through_process_exit=wall-min(fifth,900),charged_through_process_exit_within_budget=True,
                     fresh_empty_sdk_caches={},checkpoint_sha256=sha(b'fixture-checkpoint'),checkpoint_bytes=len(b'fixture-checkpoint'))
        (sandbox/'rehearsal.json').write_text(json.dumps(receipt))
        (root/'training.log').write_text(''.join(f'step {i} | loss 2.0000 | lrm 1.000 | dt 0.100s | fixture\n' for i in range(5)))
        plan=dict(source_sha256=sha(b'fixture-source'),prepare_sha256=sha(b'fixture-prepare'))
        phase=dict(root=str(root),sandbox=str(sandbox),name='W1')
        return plan,phase,receipt

    def test_real_validator_target_boundary_and_900_excess_charge(self):
        for fifth,wall,expected in [(120,123,'USEFUL_WARM_START_OBSERVED'),(120.001,124,'WARM_TARGET_MISSED'),(1000,1003,'WARM_TARGET_MISSED')]:
            with self.subTest(fifth=fifth),tempfile.TemporaryDirectory(prefix='warm-five-fixture-') as tmp:
                plan,phase,_=self.fixture(Path(tmp),fifth,wall)
                observed=h.validate_phase(plan,phase)
                self.assertEqual(classify(observed),expected)
                self.assertEqual(observed['charged_through_exit_seconds'],wall-min(fifth,900))
                self.assertEqual(observed['runtime_reuse'],'UNPROVEN')

    def test_refuses_bad_core_gates_not_only_forensics(self):
        changes=[('returncode',1),('step_boundaries',4),('charged_through_process_exit',999),('charged_through_process_exit_within_budget',False),('fresh_empty_sdk_caches',{'wrong':'cold'}),('checkpoint_sha256','0'*64),('checkpoint_bytes',0)]
        for key,value in changes:
            with self.subTest(key=key),tempfile.TemporaryDirectory(prefix='warm-five-negative-') as tmp:
                root=Path(tmp);plan,phase,receipt=self.fixture(root)
                receipt[key]=value;(root/'train/rehearsal.json').write_text(json.dumps(receipt))
                with self.assertRaises(RuntimeError):h.validate_phase(plan,phase)
        with tempfile.TemporaryDirectory(prefix='warm-budget-negative-') as tmp:
            plan,phase,_=self.fixture(Path(tmp),100,1900.1)
            with self.assertRaisesRegex(RuntimeError,'SOURCE_BUDGET'):h.validate_phase(plan,phase)
        with tempfile.TemporaryDirectory(prefix='warm-nan-negative-') as tmp:
            root=Path(tmp);plan,phase,_=self.fixture(root)
            log=root/'training.log';log.write_text(log.read_text().replace('2.0000','nan',1))
            with self.assertRaisesRegex(RuntimeError,'NONFINITE'):h.validate_phase(plan,phase)

    def test_identity_rejects_wrong_bytes_symlinks_and_writable_source(self):
        with tempfile.TemporaryDirectory(prefix='warm-identity-negative-') as tmp:
            p=Path(tmp)/'train.py';p.write_bytes(b'exact')
            with self.assertRaisesRegex(RuntimeError,'WRITABLE'):h.identity(p,sha(b'exact'),True)
            p.chmod(0o444)
            with self.assertRaisesRegex(RuntimeError,'SHA_MISMATCH'):h.identity(p,'0'*64,True)
            q=Path(tmp)/'linked';q.symlink_to(p)
            with self.assertRaisesRegex(RuntimeError,'SYMLINK'):h.identity(q)

    def test_shared_idle_guard_busy_and_malformed_refuse(self):
        device=[dict(neuron_device=0,instance_type='trn2.3xlarge',nc_count=4,logical_neuroncore_config=2,neuroncore_ids=[0,1,2,3],neuron_processes=[])]
        for busy,bad in [(False,False),(True,False),(False,True)]:
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                text='1 0 python host_driver.py\n2 0 flock /usr/bin/flock\n'
                if busy:text+='3 0 torchrun torchrun train.py\n'
                return SimpleNamespace(returncode=0,stderr='',stdout=text if command==g.PS else ('{}' if bad else json.dumps(device)))
            if busy or bad:
                with self.assertRaises(RuntimeError):g.check_idle(run,1,2,{})
            else:g.check_idle(run,1,2,{})
            self.assertEqual(calls,[g.PS,g.NEURON_LS])

    def test_single_child_lifetime_lock_and_forensic_scope(self):
        text=(HERE/'host_driver.py').read_text()
        self.assertEqual(text.count('subprocess.call('),1)
        self.assertEqual(text.count('require_idle_under_inherited_lock('),3)
        self.assertIn('pass_fds=(lock_fd,)',text)
        self.assertIn("require(after==before,'SOURCE_OR_PACKAGE_CHANGED_DURING_PHASE')",text)
        self.assertIn("before[key]==anchor[key]",text)
        self.assertNotIn("require(evidence['exact_input_bytes_complete']",text)
        self.assertIn("native_cache_reuse='UNPROVEN'",text)
        self.assertEqual(P['useful_warm_fifth_target_seconds'],120)
        self.assertEqual(P['original_pair_status'],'PAIR_REFUSED_OR_INCOMPLETE')
        self.assertFalse(P['original_S2_launched'])
        for flag in ('stage_authorized','launch_authorized','host_reserved','source_adoption','submission_authorized'):self.assertFalse(P[flag])
        self.assertTrue(P['no_retry'] and P['no_eval'] and P['no_automatic_full'])

if __name__=='__main__':unittest.main(verbosity=2)
