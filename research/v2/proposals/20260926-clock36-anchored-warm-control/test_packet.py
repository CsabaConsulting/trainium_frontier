"""Own stdlib-only composition tests. No host, SDK, model or data imports."""
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
PARENT=REPO/'research/v2/proposals/20260926-startup-adjusted-full-trn21-replication'
PLAN=json.loads((HERE/'ANCHORED_BINDING.disabled.json').read_text())
SOURCE=(HERE/'host_driver.py').read_text()
def module(text,name):
    m=types.ModuleType(name);m.__file__=str(HERE/(name+'.py'))
    # Reproduce the pinned normal-Python driver, including unchanged assertions,
    # even when this test suite itself is invoked with -O.
    exec(compile(text,m.__file__,'exec',optimize=0),m.__dict__)
    return m
def driver():return module(SOURCE,'host_driver')
CHECKS=module((HERE/'identity_checks.py').read_text(),'identity_checks')
def digest(data):return hashlib.sha256(data).hexdigest()
def create(path,data=b'x',mode=0o600):
    path.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode),'wb') as stream:stream.write(data)
    return path
def receipt(fifth=180.0,charge=1800.0):
    times=[fifth*i/5 for i in range(1,6)]+[fifth+0.25]
    return dict(returncode=0,mode='train',step_boundaries=6,
        observations=[dict(step=i,observed_seconds=t) for i,t in enumerate(times)],
        wall_seconds=fifth+charge,startup_allowance_observed=min(fifth,900),
        charged_through_process_exit=fifth+charge-min(fifth,900),
        charged_through_process_exit_within_budget=True,fresh_empty_sdk_caches={})
LOG=''.join('step %d | loss 2.0000 | lrm 0.001 | dt 0.25s | tok/s 1\n'%i for i in range(6))

class Fixture:
    def __init__(self,base):
        self.root=base/'new';self.root.mkdir()
        self.d=driver();d=self.d;d.ROOT=self.root;d.__file__=str(self.root/'host_driver.py')
        source=create(base/'historical/train/train.py',b'# source\n',0o444)
        prepare=create(source.parent/'prepare.py',b'# prepare\n',0o600)
        package=self.root/'package';package.mkdir()
        create(package/'train.py',source.read_bytes(),0o444)
        launch='torchrun --standalone --nproc_per_node=4 '+str(source)+' --out-dir out'
        create(package/'launch-command.txt',(launch+'\n').encode(),0o444)
        create(package/'provenance.json',json.dumps(dict(train_sha256=digest(source.read_bytes()),launch_command=launch)).encode(),0o444)
        protected=create(base/'project/prepare.py',prepare.read_bytes())
        harness=create(base/'project/rehearsal.py')
        create(self.root/'host_driver.py',SOURCE.encode())
        create(self.root/'identity_checks.py',(HERE/'identity_checks.py').read_bytes())
        create(self.root/'idle_guard.py')
        p=copy.deepcopy(PLAN)
        p.update(root=str(self.root),package=str(package),executed_source=str(source),
            expected_prepare_import=str(prepare),prepare_path=str(protected),harness_path=str(harness),
            source_sha256=digest(source.read_bytes()),prepare_sha256=digest(prepare.read_bytes()),
            harness_sha256=digest(harness.read_bytes()),training_command=shlex.split(launch))
        p['anchor_observation']={}
        mapping={'device':'st_dev','inode':'st_ino','mode':'st_mode','uid':'st_uid','size':'st_size','mtime_ns':'st_mtime_ns','ctime_ns':'st_ctime_ns','sha256':'sha256'}
        for file in (source,prepare):
            data=CHECKS.identity(file)
            p['anchor_observation'][str(file)]={raw:data[key] for raw,key in mapping.items()}
            p['anchor_observation'][str(file)]['stable_during_read']=True
        p['package_pins']={name:digest((package/name).read_bytes()) for name in ('launch-command.txt','provenance.json')}
        p['support']={name:digest((self.root/name).read_bytes()) for name in ('identity_checks.py','idle_guard.py')}
        metrics=self.root/'metrics-training';metrics.mkdir()
        train_env=copy.deepcopy(d.TRAIN_ENV);train_env['TORCH_NEURONX_METRICS_DIR']=str(metrics)
        p['training_environment']=train_env
        lock=create(base/'device0.lock');self.fd=os.open(lock,os.O_RDONLY)
        st=os.fstat(self.fd);p['original_lock']={'device':st.st_dev,'inode':st.st_ino}
        create(self.root/'ANCHORED_BINDING.disabled.json',(json.dumps(p)+'\n').encode())
        d.PLAN_SHA=digest((self.root/'ANCHORED_BINDING.disabled.json').read_bytes())
        d.TRAIN_ENV=train_env;self.plan=p;self.source=source;self.prepare=prepare
    def close(self):os.close(self.fd)
    def snapshot(self):return self.d.anchored_inputs(CHECKS,self.plan)

class Packet(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory(prefix='anchored-packet-fixture-');self.addCleanup(temp.cleanup)
        f=Fixture(Path(temp.name));self.addCleanup(f.close);return f
    def test_pins_and_exact_scope(self):
        d=driver();parent=json.loads((PARENT/'NATIVE_BINDING.disabled.json').read_text())
        self.assertEqual(digest((HERE/'ANCHORED_BINDING.disabled.json').read_bytes()),d.PLAN_SHA)
        self.assertEqual(PLAN['parent_training_command'],parent['training_command'])
        self.assertEqual([i for i,(a,b) in enumerate(zip(PLAN['parent_training_command'],PLAN['training_command'])) if a!=b],[3])
        self.assertEqual(len(PLAN['parent_training_command']),len(PLAN['training_command']))
        self.assertEqual(PLAN['training_command'][3],PLAN['executed_source'])
        self.assertEqual(shlex.split((HERE/'package/launch-command.txt').read_text()),PLAN['training_command'])
        for key,old,new,allowed in (
            ('training_environment',parent['training_environment'],PLAN['training_environment'],{'TRN_EXP_LABEL','TORCH_NEURONX_METRICS_DIR'}),
            ('evaluation_environment',parent['evaluation_environment'],PLAN['evaluation_environment'],{'TRN_EXP_LABEL'})):
            self.assertEqual(PLAN['parent_'+key],old)
            self.assertEqual(set(old),set(new))
            self.assertEqual({k for k in old if old[k]!=new[k]},allowed)
        for old,new in ((parent['command'],PLAN['train_command']),(parent['evaluation_command'],PLAN['eval_command'])):
            self.assertEqual([x.replace(parent['root'],PLAN['root']) for x in old],new)
        for name in ('preflight','phase'):
            def node(text):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(node(SOURCE)),ast.dump(node((PARENT/'host_driver.py').read_text())))
        self.assertEqual(digest((HERE/'identity_checks.py').read_bytes()),'85eef95494f6a4da41ee5284d5089b5fe6df7c2555c5bea5bc16e86965d7a8f4')
        called={n.func.attr for n in ast.walk(ast.parse(SOURCE)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(called & {'inputs','validate_phase','capture_inputs','chmod'})
        raw=subprocess.check_output(['git','show',PLAN['source_commit']+':'+PLAN['source_git_path']],cwd=REPO)
        self.assertEqual(digest(raw),PLAN['source_sha256'])
        tree=ast.parse(raw)
        roots={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}|{n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module}
        self.assertEqual(sorted(roots),PLAN['source_import_roots'])
        self.assertFalse(PLAN['launch_authorized']);self.assertFalse(PLAN['operator_assigned'])
        self.assertEqual(PLAN['original_lock']['inode'],596823)
        self.assertEqual(d.TRAIN_COMMAND,PLAN['train_command']);self.assertEqual(d.EVAL_COMMAND,PLAN['eval_command'])
        self.assertEqual(d.TRAIN_ENV,PLAN['training_environment']);self.assertEqual(d.EVAL_ENV,PLAN['evaluation_environment'])
        for key,pin in PLAN['package_pins'].items():self.assertEqual(digest((HERE/'package'/key).read_bytes()),pin)
        for key in ('EXPERIMENT_PROPOSAL.json','audit-normal.json','identity_checks.py'):
            self.assertEqual(digest((HERE/key).read_bytes()),PLAN['support'][key])
        for key in ('idle_guard.py','dispatch_driver.py'):
            path=REPO/'research/v2/operator/20260926-common-host-idle-guard'/key
            self.assertEqual(digest(path.read_bytes()),PLAN['support'][key])
    def test_prepare0600_anchor_and_field_failures(self):
        f=self.fixture();before=f.snapshot();f.d.unchanged(before,f.snapshot())
        self.assertEqual(before['files']['expected_imported_prepare']['st_mode']&0o777,0o600)
        self.assertFalse(before['actual_worker_import_origin_observed'])
        for raw in ('device','inode','mode','uid','size','mtime_ns','ctime_ns','sha256'):
            p=copy.deepcopy(f.plan);v=p['anchor_observation'][str(f.source)][raw]
            p['anchor_observation'][str(f.source)][raw]=('0'*64 if raw=='sha256' else v+1)
            with self.subTest(raw=raw),self.assertRaisesRegex(RuntimeError,'ANCHORED_FIELD_CHANGED'):
                f.d.anchored_inputs(CHECKS,p)
        with self.assertRaisesRegex(RuntimeError,'WRITABLE_IMMUTABLE_FILE'):
            CHECKS.identity(f.prepare,readonly=True)
    def test_reject_import_shadow(self):
        for name in ('torch','prepare','sitecustomize','usercustomize'):
            with self.subTest(name=name):
                f=self.fixture()
                if name=='prepare':create(f.source.parent/'prepare/__init__.py')
                else:create(f.source.parent/(name+'.py'))
                importlib.invalidate_caches()
                with self.assertRaisesRegex(RuntimeError,'IMPORT_SHADOW|PREPARE_RESOLUTION'):f.snapshot()
    def test_symlink_hardlink_hash_and_ancestor(self):
        f=self.fixture();link=f.root/'linked.py';link.symlink_to(f.source)
        with self.assertRaisesRegex(RuntimeError,'SYMLINK'):CHECKS.identity(link)
        os.link(f.source,f.root/'hardlink.py')
        with self.assertRaisesRegex(RuntimeError,'HARDLINK'):f.snapshot()
        f2=self.fixture();p=copy.deepcopy(f2.plan);p['source_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'SHA_MISMATCH'):f2.d.anchored_inputs(CHECKS,p)
        before=f2.snapshot()
        for field in f2.d.ANCESTOR_STABLE_FIELDS:
            after=copy.deepcopy(before);ancestor=next(iter(after['ancestors']))
            after['ancestors'][ancestor][field]+=1
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'ANCESTOR_IDENTITY'):f2.d.unchanged(before,after)
        for field in f2.d.STAT_FIELDS:
            after=copy.deepcopy(before);historical=next(iter(after['historical_directories']))
            after['historical_directories'][historical][field]+=1
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'ANCHORED_SNAPSHOT'):f2.d.unchanged(before,after)
        after=copy.deepcopy(before)
        for item in after['ancestors'].values():
            for field in ('st_nlink','st_mtime_ns','st_ctime_ns'):item[field]+=1
        f2.d.unchanged(before,after)
    def test_accounting_boundaries_and_negatives(self):
        d=driver()
        self.assertEqual(d.account_training(receipt(),LOG)['warm_target_disposition'],'WARM_TARGET_MET')
        self.assertEqual(d.account_training(receipt(180.00001),LOG)['warm_target_disposition'],'WARM_TARGET_MISSED')
        for field,value in (('returncode',1),('returncode',False),('mode','eval'),('step_boundaries',5),
            ('wall_seconds',float('nan')),('startup_allowance_observed',0),('charged_through_process_exit',1800.1),
            ('charged_through_process_exit_within_budget',False),('fresh_empty_sdk_caches',{'compiler':'new'})):
            r=receipt();r[field]=value
            with self.subTest(field=field),self.assertRaises((RuntimeError,ValueError)):d.account_training(r,LOG)
        with self.assertRaisesRegex(RuntimeError,'CHARGED'):d.account_training(receipt(180,1800.01),LOG)
        r=receipt();r['observations'][1]['step']=0
        with self.assertRaisesRegex(RuntimeError,'SEQUENCE'):d.account_training(r,LOG)
        r=receipt();r['observations'][1]['observed_seconds']=float('inf')
        with self.assertRaisesRegex(RuntimeError,'TIMES'):d.account_training(r,LOG)
        for log in (LOG.replace('loss 2.0000','loss nan',1),LOG.replace('step 0','step 9',1)):
            with self.assertRaises(RuntimeError):d.account_training(receipt(),log)
    def test_main_success_and_failure_paths_no_processes(self):
        cases=('success','warm_miss','train_rc','eval_rc','budget','anchor_after_train','initial_idle','idle_before_eval','final_idle','final_anchor','lock','hostname','optimized')
        for case in cases:
            with self.subTest(case=case):
                f=self.fixture();d=f.d;events=[];snapshots=[0];guards=[0];records={}
                original_snapshot=d.anchored_inputs
                def snapshot(checks,plan):
                    snapshots[0]+=1
                    if case=='anchor_after_train' and snapshots[0]>=2:raise RuntimeError('fixture anchor changed')
                    if case=='final_anchor' and snapshots[0]==5:raise RuntimeError('fixture final changed')
                    return original_snapshot(checks,plan)
                def guard(*args):
                    guards[0]+=1;events.append('guard')
                    if case=='initial_idle' and guards[0]==1:raise RuntimeError('fixture initial busy')
                    if case=='idle_before_eval' and guards[0]==3:raise RuntimeError('fixture busy')
                    if case=='final_idle' and guards[0]==4:raise RuntimeError('fixture final busy')
                    return f.fd
                def run_phase(name,cmd,env,fd):
                    self.assertEqual(fd,f.fd);events.append(name)
                    if name=='training':
                        create(f.root/'train/out/final.pt',b'checkpoint')
                        r=receipt(181 if case=='warm_miss' else 180)
                        if case=='budget':r['charged_through_process_exit']=1801
                        r.update(checkpoint_sha256=digest(b'checkpoint'),checkpoint_bytes=10)
                        create(f.root/'train/rehearsal.json',json.dumps(r).encode())
                        create(f.root/'training.log',LOG.encode())
                        for file in ('train.py','prepare.py'):
                            create(f.root/'train'/file,(Path(f.plan['package'])/'train.py').read_bytes() if file=='train.py' else f.prepare.read_bytes(),0o444 if file=='train.py' else 0o600)
                        return 7 if case=='train_rc' else 0
                    create(f.root/'eval/rehearsal.json',json.dumps(dict(returncode=0,mode='eval')).encode())
                    for file in ('train.py','prepare.py'):
                        create(f.root/'eval'/file,(f.root/'train'/file).read_bytes(),0o444 if file=='train.py' else 0o600)
                    return 8 if case=='eval_rc' else 0
                def loader(name,path):return types.SimpleNamespace(require_idle_under_inherited_lock=guard) if 'idle_guard' in name else CHECKS
                def lock(fd,plan):
                    if case=='lock':raise RuntimeError('fixture wrong lock')
                patches=dict(load_module=loader,preflight=lambda env:events.append('preflight'),phase=run_phase,
                    anchored_inputs=snapshot,check_lock=lock,record=lambda name,obj:records.setdefault(name,obj),
                    sys=types.SimpleNamespace(flags=types.SimpleNamespace(optimize=1 if case=='optimized' else 0)),
                    socket=types.SimpleNamespace(gethostname=lambda:'bad' if case=='hostname' else f.plan['expected_hostname']))
                with patch.dict(d.__dict__,patches):rc=d.main()
                good=case in ('success','warm_miss')
                self.assertEqual(rc,0 if good else 1)
                self.assertIn('terminal.json',records)
                self.assertEqual(records['terminal.json']['control_eligibility'],'PENDING_INDEPENDENT_RAW_BUDGET_EVAL_POSTFLIGHT_REVIEW')
                if case in ('train_rc','budget','anchor_after_train','initial_idle','idle_before_eval','lock','hostname','optimized'):
                    self.assertNotIn('fresh-public20m',events)
                if case=='initial_idle':self.assertEqual(guards[0],1);self.assertNotIn('training',events)
                elif case!='optimized':self.assertGreaterEqual(guards[0],2)
                if snapshots[0]:self.assertGreaterEqual(snapshots[0],2)
                if good:self.assertEqual(events.count('training'),1);self.assertEqual(events.count('fresh-public20m'),1)
                if case=='warm_miss':self.assertEqual(records['terminal.json']['training_observation']['warm_target_disposition'],'WARM_TARGET_MISSED')
    def test_inherited_preflight_negatives_and_phase_fd(self):
        d=driver()
        class FakePath:
            def __init__(self,value):self.value=str(value)
            def __truediv__(self,rel):return FakePath(self.value+'/'+str(rel))
            def __str__(self):return self.value
            @property
            def parent(self):return FakePath(self.value.rsplit('/',1)[0])
            def glob(self,pattern):return []
            def is_dir(self):return True
        root=FakePath('/fixture');env={key:'existing' for key in d.TRAIN_ENV}
        for failure in ('stop','disk','source','prepare','harness','cache'):
            def fake_sha(path):
                expected=d.SOURCE_SHA if 'package/train.py' in str(path) else d.PREP_SHA if 'prepare.py' in str(path) else d.HARNESS_SHA
                return 'bad' if failure in str(path) or (failure=='source' and 'package' in str(path)) else expected
            fake_os=types.SimpleNamespace(path=types.SimpleNamespace(lexists=lambda path:failure=='stop'),
                statvfs=lambda path:types.SimpleNamespace(f_bavail=0 if failure=='disk' else 20*1024**3,f_frsize=1))
            with self.subTest(failure=failure),patch.dict(d.__dict__,dict(Path=FakePath,ROOT=root,PREP=FakePath('/prepare.py'),HARNESS=FakePath('/harness.py'),sha=fake_sha,os=fake_os)):
                if failure=='cache':
                    with patch.object(FakePath,'is_dir',return_value=False),self.assertRaises(AssertionError):d.preflight(env)
                else:
                    with self.assertRaises(AssertionError):d.preflight(env)
        f=self.fixture();calls=[]
        with patch.object(f.d.subprocess,'call',side_effect=lambda *a,**kw:calls.append((a,kw)) or 0):
            self.assertEqual(f.d.phase('training',['not-executed'],{},f.fd),0)
        self.assertEqual(calls[0][1]['pass_fds'],(f.fd,));self.assertEqual(calls[0][1]['cwd'],f.root)

if __name__=='__main__':
    unittest.main(verbosity=2)
