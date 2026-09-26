"""Stdlib tests only. No host driver import, subprocess launch or tensors."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
B=json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())
SOURCE=(HERE/'host_driver.py').read_text()
TREE=ast.parse(SOURCE)


class BindingTests(unittest.TestCase):
    def test_disabled_source_and_support_pins(self):
        self.assertFalse(B['stage_authorized']);self.assertFalse(B['launch_authorized'])
        for commit,path,sha in [(B['source_commit'],B['source_git_path'],B['source_sha256']),(B['harness_source']['commit'],B['harness_source']['git_path'],B['harness_sha256'])]:
            raw=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),sha)
        for path,sha in [('host_driver.py',B['driver_sha256']),('lifetime_guard.py',B['lifetime_guard_sha256']),('package/launch-command.txt',B['package_launch_sha256']),('package/provenance.json',B['package_provenance_sha256'])]:
            self.assertEqual(hashlib.sha256((HERE/path).read_bytes()).hexdigest(),sha)
        for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes()).hexdigest(),B[key])

    def test_exact_unchanged_long_launch(self):
        launch=(HERE/'package/launch-command.txt').read_bytes()
        self.assertEqual(launch,subprocess.check_output(['git','show',B['source_commit']+':workflow/candidates/confirmed-clock36-116188/launch-command.txt'],cwd=ROOT))
        provenance=json.loads((HERE/'package/provenance.json').read_text())
        self.assertEqual(provenance['launch_command'],launch.decode().strip())
        self.assertEqual(B['training_command'],shlex.split(launch.decode())[1:])
        self.assertEqual(provenance['train_sha256'],B['source_sha256'])
        self.assertNotIn('--num-steps',B['training_command'])
        self.assertNotIn('--initial-cache-refresh',B['training_command'])

    def test_fresh_cache_routes_and_fixed_eval(self):
        for phase,cmd,key in [('train',B['command'],'training_environment'),('eval',B['evaluation_command'],'evaluation_environment')]:
            self.assertEqual(cmd.count('--fresh-caches'),1)
            self.assertEqual(B[key]['NEURON_COMPILE_CACHE_URL'],B['root']+'/'+phase+'/caches/compiler')
            self.assertNotIn('NEURON_COMPILE_CACHE_DIR',B[key])
            for p in B['fresh_cache_contract']['expected_'+('training' if phase=='train' else 'eval')+'_dirs'].values():
                self.assertTrue(p.startswith(B['root']+'/'+phase+'/caches/'))
        self.assertEqual(B['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'],'1800')
        self.assertEqual(B['evaluation_command'][-3:],['--eval-tokens','20971520','--fresh-caches'])
        self.assertEqual(B['evaluation']['batch_size'],1);self.assertEqual(B['evaluation']['seq_len'],1024)
        self.assertEqual(B['evaluation_environment']['NEURON_CC_FLAGS'],'--optlevel=1')
        self.assertTrue(B['cold_gate']['required_before_full_launch']);self.assertIsNone(B['cold_gate']['result'])

    def test_shared_guard_and_inherited_fd_control_flow(self):
        phase=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='phase')
        calls=[n for n in ast.walk(phase) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].func),'guardian.run')
        lifetime=ast.parse((HERE/'lifetime_guard.py').read_text())
        spawns=[n for n in ast.walk(lifetime) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
        self.assertEqual(len(spawns),1)
        self.assertIn('(self.fd,)',[ast.unparse(k.value) for k in spawns[0].keywords if k.arg=='pass_fds'])
        guards=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='require_idle_under_inherited_lock']
        self.assertEqual(len(guards),3)
        guards=sorted(guards,key=lambda n:n.lineno)[1:]  # Earlier one is final postflight's body.
        phasecalls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='phase']
        self.assertLess(guards[0].lineno,phasecalls[0].lineno)
        self.assertLess(guards[1].lineno,phasecalls[1].lineno)
        self.assertIn("assert eval_fd==lock_fd",SOURCE)
        self.assertEqual(B['outer_command'][:6],['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B'])

    def test_preflight_refusal_fixtures(self):
        function=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
        # Bound driver is non-optimized -I -B; explicitly compile its assertions intact.
        code=compile(ast.Module(body=[function],type_ignores=[]),'<preflight>','exec',optimize=0)
        class FakePath:
            def __init__(self,path):self.path=str(path)
            def __truediv__(self,name):return FakePath(self.path+'/'+name)
            def __str__(self):return self.path
            @property
            def parent(self):return FakePath(str(Path(self.path).parent))
            def glob(self,pattern):return []
            def is_dir(self):return True
        pins={B['root']+'/package/train.py':B['source_sha256'],B['root']+'/package/launch-command.txt':B['package_launch_sha256'],B['root']+'/package/provenance.json':B['package_provenance_sha256'],B['prepare_path']:B['prepare_sha256'],B['harness_path']:B['harness_sha256']}
        for bad in (None,'optimized','host','STOP','disk','disk11','disk13.49','source','launch','provenance','prepare','harness','sandbox','url'):
            env=copy.deepcopy(B['training_environment'])
            if bad=='url':env['NEURON_COMPILE_CACHE_URL']='/wrong'
            mapping={'source':B['root']+'/package/train.py','launch':B['root']+'/package/launch-command.txt','provenance':B['root']+'/package/provenance.json','prepare':B['prepare_path'],'harness':B['harness_path']}
            free={'disk':1,'disk11':11*1024**3,'disk13.49':13.49*1024**3}.get(bad,13.5*1024**3)
            ns=dict(ROOT=FakePath(B['root']),PREP=FakePath(B['prepare_path']),HARNESS=FakePath(B['harness_path']),SOURCE_SHA=B['source_sha256'],PREP_SHA=B['prepare_sha256'],HARNESS_SHA=B['harness_sha256'],TRAIN_ENV=env,Path=FakePath,sys=SimpleNamespace(flags=SimpleNamespace(optimize=int(bad=='optimized'))),socket=SimpleNamespace(gethostname=lambda:'bad' if bad=='host' else B['expected_hostname']),sha=lambda path:'bad' if str(path)==mapping.get(bad) else pins[str(path)],os=SimpleNamespace(path=SimpleNamespace(lexists=lambda path:(bad=='STOP' and str(path).endswith('/STOP')) or (bad=='sandbox' and str(path)==B['root']+'/train')),statvfs=lambda path:SimpleNamespace(f_bavail=free,f_frsize=1)))
            exec(code,ns)
            if bad is None:ns['preflight'](env)
            else:
                with self.assertRaises((AssertionError,RuntimeError),msg=bad):ns['preflight'](env)
        bad=None
        ns['TRAIN_ENV']={}
        env=copy.deepcopy(B['evaluation_environment'])
        for free in (10.5*1024**3,11*1024**3):
            ns['preflight'](env)
        free=10.49*1024**3
        with self.assertRaises(AssertionError):ns['preflight'](env)

    def test_exact_harness_creates_empty_caches_before_child_and_accounts_exit(self):
        raw=subprocess.check_output(['git','show',B['harness_source']['commit']+':'+B['harness_source']['git_path']],cwd=ROOT).decode()
        self.assertIn('cache.mkdir(parents=True, exist_ok=False)',raw)
        self.assertIn('assert not list(cache.iterdir())',raw)
        self.assertLess(raw.index('cache.mkdir('),raw.index('process = subprocess.Popen('))
        self.assertIn('env[key] = str(cache)',raw)
        self.assertIn('wall_seconds=time.monotonic() - started',raw)
        self.assertIn('charged_through_process_exit=result["wall_seconds"] - allowance',raw)
        self.assertIn('min(observed[4]["observed_seconds"], 900.0)',raw)
        self.assertIn('<= 1800',raw)
        self.assertIn('"--seq-len", "1024"',raw)
        self.assertIn('"--timeout-seconds", "1200"',raw)

    def test_exact_new_proposal_and_literal_derivation(self):
        proposal_raw=(HERE/'EXPERIMENT_PROPOSAL.json').read_bytes()
        proposal=json.loads(proposal_raw)
        self.assertEqual(proposal_raw,subprocess.check_output(['git','show',B['proposal_first_commit']+':'+str((HERE/'EXPERIMENT_PROPOSAL.json').relative_to(ROOT))],cwd=ROOT))
        self.assertEqual(hashlib.sha256(proposal_raw).hexdigest(),B['proposal_sha256'])
        self.assertEqual(B['source_gate'],proposal['gates'])
        self.assertFalse(B['old_scientific_approval_inherited'])
        self.assertEqual(B['original_lock'],{'path':'/home/ubuntu/.cache/trainium_freshstart/device0.lock','device':66305,'inode':832589})
        self.assertEqual(B['resource_gates']['disk_floor_gib'],10.5)
        self.assertEqual(B['resource_gates']['projected_compile_checkpoint_footprint_gib'],3)
        derivation=B['driver_derivation']
        parent=subprocess.check_output(['git','show',derivation['commit']+':'+derivation['path']],cwd=ROOT)
        self.assertEqual(hashlib.sha256(parent).hexdigest(),derivation['sha256'])
        expected=parent.decode()
        for old,new in derivation['replacements']:
            self.assertIn(old,expected);expected=expected.replace(old,new)
        baseline=subprocess.check_output(['git','show','3378ac0181f77610ae24d69281c5768f00b589e6:'+str((HERE/'host_driver.py').relative_to(ROOT))],cwd=ROOT).decode()
        self.assertEqual(expected,baseline)  # Historical four-literal derivation remains auditable.
        self.assertEqual(B['mechanics_revision']['baseline_driver_sha256'],hashlib.sha256(baseline.encode()).hexdigest())
        self.assertIn("(66305,832589)",SOURCE)
        self.assertIn('guardian.finish(postflight)',SOURCE)
        self.assertEqual(B['resource_gates']['final_training_disk_floor_gib'],13.5)
        self.assertEqual(B['resource_gates']['pre_eval_disk_floor_gib'],10.5)
        self.assertEqual(B['native_resolved_training_command'],proposal['argv'])
        self.assertEqual(B['native_resolved_training_command'][1:],B['training_command'][1:])
        self.assertEqual(B['native_resolved_training_command'][0],B['training_environment']['PATH'].split(':')[0]+'/torchrun')
        self.assertEqual(B['source_sha256'],proposal['source_sha256'])
        constants={}
        for node in TREE.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                value=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
                constants[node.targets[0].id]=ast.literal_eval(value)
        for field,key in [('ROOT','root'),('PREP','prepare_path'),('HARNESS','harness_path'),('LABEL','label'),('SOURCE_SHA','source_sha256'),('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment')]:
            self.assertEqual(constants[field],B[key],field)
        clean_ct=json.loads((ROOT/'research/v2/proposals/20260926-confirmed36-clean-cold-five-trn22-r2/CT_BINDING.disabled.json').read_text())
        self.assertEqual(clean_ct['training_command'][:-2],B['native_resolved_training_command'])
        expected_env=json.loads(json.dumps(clean_ct['training_environment']).replace(clean_ct['root'],B['root']).replace(clean_ct['label'],B['label']))
        self.assertEqual(expected_env,B['training_environment'])
        self.assertFalse(B['stage_authorized']);self.assertFalse(B['launch_authorized']);self.assertFalse(B['submission_authorized'])

    def test_actual_unchanged_harness_containment_old_fail_new_pass(self):
        raw=subprocess.check_output(['git','show',B['harness_source']['commit']+':'+B['harness_source']['git_path']],cwd=ROOT)
        predicates=[n.test for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Assert) and isinstance(n.test,ast.Call) and isinstance(n.test.func,ast.Attribute) and n.test.func.attr=='is_relative_to']
        self.assertEqual(len(predicates),1)
        self.assertEqual(ast.unparse(predicates[0]),"sandbox.is_relative_to(root / 'out')")
        predicate=compile(ast.Expression(predicates[0]),'<actual-pinned-harness-containment>','eval')
        root=Path(B['harness_path']).resolve().parents[2]
        for phase,key in [('train','sandbox'),('eval','eval_sandbox')]:
            old=(Path(B['harness_containment']['old_outside_root'])/phase).resolve()
            new=Path(B[key]).resolve()
            self.assertFalse(eval(predicate,{'__builtins__':{}},{'root':root,'sandbox':old}))
            self.assertTrue(eval(predicate,{'__builtins__':{}},{'root':root,'sandbox':new}))
            self.assertEqual(new,Path(B['root']).resolve()/phase)


if __name__=='__main__':unittest.main(verbosity=2)
