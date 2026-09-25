"""Tiny stdlib filesystem/receipt fixtures only. Never imports train/prepare/Torch."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
P=json.loads((HERE/'PAIR_BINDING.disabled.json').read_text())
spec=importlib.util.spec_from_file_location('local_pair_checks',HERE/'identity_checks.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def h(raw):return hashlib.sha256(raw).hexdigest()

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.p=copy.deepcopy(P);self.phase=copy.deepcopy(P['phases'][0])
  self.stable=self.root/'stable';self.stable.mkdir()
  self.train=b'fixture_train = True\n';self.prep=b'fixture_prepare = True\n'
  for name,raw in [('train.py',self.train),('prepare.py',self.prep)]:
   (self.stable/name).write_bytes(raw);(self.stable/name).chmod(0o444)
  self.stable.chmod(0o555)
  self.protected=self.root/'protected_prepare.py';self.protected.write_bytes(self.prep)
  self.harness=self.root/'harness.py';self.harness.write_bytes(b'fixture harness')
  self.phase['root']=str(self.root/'phase');Path(self.phase['root']).mkdir()
  self.package=Path(self.phase['root'])/'package';self.package.mkdir()
  self.sandbox=Path(self.phase['root'])/'train';self.sandbox.mkdir()
  self.phase['package']=str(self.package);self.phase['sandbox']=str(self.sandbox)
  self.phase['training_command'][3]=str(self.stable/'train.py')
  launch=' '.join(self.phase['training_command'])+'\n'
  manifest=json.dumps(dict(train_sha256=h(self.train),launch_command=launch.strip()))
  for name,raw in [('train.py',self.train),('launch-command.txt',launch.encode()),('provenance.json',manifest.encode())]:
   (self.package/name).write_bytes(raw);(self.package/name).chmod(0o444)
  self.phase['launch_sha256']=h(launch.encode());self.phase['provenance_sha256']=h(manifest.encode())
  self.p.update(stable_directory=str(self.stable),source_sha256=h(self.train),prepare_sha256=h(self.prep),
                prepare_path=str(self.protected),harness_path=str(self.harness),harness_sha256=h(self.harness.read_bytes()))
  for name,raw in [('train.py',self.train),('prepare.py',self.prep)]:
   (self.sandbox/name).write_bytes(raw);(self.sandbox/name).chmod(0o444)
  (self.sandbox/'out').mkdir();(self.sandbox/'out/final.pt').write_bytes(b'not a real checkpoint')
  self.p['cache_environment']={k:str(self.root/'caches'/Path(v).name) for k,v in P['cache_environment'].items()}
  for path in self.p['cache_environment'].values():Path(path).mkdir(parents=True)
  self.metrics=Path(self.phase['root'])/'metrics-training';self.metrics.mkdir()
  self.phase['environment']['TORCH_NEURONX_METRICS_DIR']=str(self.metrics)
  self.receipt=dict(returncode=0,mode='train',observations=[dict(step=i,observed_seconds=10.0+i) for i in range(5)],
       step_boundaries=5,wall_seconds=15.,startup_allowance_observed=14.,charged_through_process_exit=1.,
       charged_through_process_exit_within_budget=True,checkpoint_bytes=21,checkpoint_sha256=h(b'not a real checkpoint'),
       fresh_empty_sdk_caches=self.p['cache_environment'])
  self.write_receipt()
  (Path(self.phase['root'])/'training.log').write_text(''.join('step %05d | loss %.4f | lrm 1.000 | dt 0.50s | diagnostic fixture\n'%(i,9-i/10) for i in range(5)))

 def tearDown(self):
  self.stable.chmod(0o755)
  self.tmp.cleanup()

 def write_receipt(self):
  (self.sandbox/'rehearsal.json').write_text(json.dumps(self.receipt))

 def graphs(self):
  debug=Path(self.p['cache_environment']['TORCH_NEURONX_DEBUG_DIR'])/'neuronx-cc';debug.mkdir()
  rows=[]
  for kind in ('forward','backward'):
   child=debug/kind;child.mkdir();raw=('binary fixture '+kind).encode()
   (child/'module.mlir').write_bytes(raw)
   rows.append(h(raw)+':'+('0'*64)+',model_default_0_'+kind+'_1,model_default,1\n')
  for rank in range(4):
   (self.metrics/('metric_table_graph_stats_rank'+str(rank)+'.csv')).write_text('graph_id,graph_name,model_name,torch_neuronx_compile_us\n'+''.join(rows))
  return debug

 def test_identity_pre_post_equal_and_no_import(self):
  before=M.inputs(self.p,self.phase);self.assertEqual(before,M.inputs(self.p,self.phase))
  self.assertNotIn('prepare',__import__('sys').modules)
  self.assertEqual(before['executed_train']['path'],str(self.stable/'train.py'))

 def test_hash_mode_hardlink_symlink_and_replacement_refusals(self):
  target=self.stable/'train.py'
  with self.assertRaisesRegex(RuntimeError,'SHA_MISMATCH'):M.identity(target,'0'*64,True)
  target.chmod(0o644)
  with self.assertRaisesRegex(RuntimeError,'WRITABLE'):M.identity(target,h(self.train),True)
  target.chmod(0o444)
  os.link(target,self.root/'hardlink')
  with self.assertRaisesRegex(RuntimeError,'HARDLINK'):M.identity(target,h(self.train),True)
  (self.root/'hardlink').unlink()
  alias=self.root/'alias';alias.symlink_to(self.stable,target_is_directory=True)
  with self.assertRaisesRegex(RuntimeError,'SYMLINK'):M.identity(alias/'train.py',h(self.train))
  self.stable.chmod(0o755)
  before=M.identity(target,h(self.train),True)
  target.unlink();target.write_bytes(self.train);target.chmod(0o444)
  after=M.identity(target,h(self.train),True)
  self.assertNotEqual(before,after)
  target.unlink();target.symlink_to(self.protected)
  with self.assertRaisesRegex(RuntimeError,'SYMLINK'):M.identity(target,h(self.prep))

 def test_extra_file_and_source_path_import_override_refuse(self):
  original=self.phase['training_command'][3];self.phase['training_command'][3]='train.py'
  with self.assertRaisesRegex(RuntimeError,'EXECUTED_SOURCE_PATH'):M.inputs(self.p,self.phase)
  self.phase['training_command'][3]=original;self.phase['environment']['PYTHONPATH']='.'
  with self.assertRaisesRegex(RuntimeError,'IMPORT_PATH_OVERRIDE'):M.inputs(self.p,self.phase)
  del self.phase['environment']['PYTHONPATH']
  self.stable.chmod(0o755);(self.stable/'other.py').write_text('fixture');self.stable.chmod(0o555)
  with self.assertRaisesRegex(RuntimeError,'CONTENTS'):M.inputs(self.p,self.phase)

 def test_five_step_acceptance_and_failure_boundaries(self):
  good=copy.deepcopy(self.receipt)
  self.assertEqual(M.validate_phase(self.p,self.phase)['fifth_boundary_seconds'],14.)
  for field,value in [('returncode',1),('returncode',False),('step_boundaries',4),('step_boundaries',5.0),('charged_through_process_exit',2.),('charged_through_process_exit',True),
                      ('charged_through_process_exit_within_budget',False),('wall_seconds',float('nan')),
                      ('checkpoint_sha256','0'*64),('checkpoint_bytes',0),('fresh_empty_sdk_caches',{})]:
   self.receipt=copy.deepcopy(good);self.receipt[field]=value;self.write_receipt()
   with self.assertRaises((RuntimeError,ValueError,KeyError)):M.validate_phase(self.p,self.phase)
  for fifth,accepted in [(900.,True),(900.00001,False)]:
   self.receipt=copy.deepcopy(good);self.receipt['observations'][-1]['observed_seconds']=fifth
   self.receipt.update(wall_seconds=fifth+1,startup_allowance_observed=min(fifth,900),
                       charged_through_process_exit=fifth+1-min(fifth,900))
   self.write_receipt()
   if accepted:self.assertEqual(M.validate_phase(self.p,self.phase)['fifth_boundary_seconds'],900)
   else:
    with self.assertRaisesRegex(RuntimeError,'OVER_900'):M.validate_phase(self.p,self.phase)
  self.receipt=good;self.receipt['observations'][0]['step']=1;self.write_receipt()
  with self.assertRaisesRegex(RuntimeError,'STEP_SEQUENCE'):M.validate_phase(self.p,self.phase)

 def test_source_budget_contradiction_s2_and_logged_nan(self):
  self.phase['name']='S2';self.receipt['fresh_empty_sdk_caches']={};self.write_receipt()
  self.assertEqual(M.validate_phase(self.p,self.phase)['status'],'FIVE_STEP_MECHANICAL_GATES_PASS')
  self.receipt.update(wall_seconds=1814.,charged_through_process_exit=1800.);self.write_receipt()
  M.validate_phase(self.p,self.phase)
  self.receipt.update(wall_seconds=1814.01,charged_through_process_exit=1800.01);self.write_receipt()
  with self.assertRaisesRegex(RuntimeError,'SOURCE_BUDGET'):M.validate_phase(self.p,self.phase)
  self.receipt.update(wall_seconds=15.,charged_through_process_exit=1.);self.write_receipt()
  log=Path(self.phase['root'])/'training.log';log.write_text(log.read_text().replace('loss 9.0000','loss nan'))
  with self.assertRaisesRegex(RuntimeError,'NONFINITE'):M.validate_phase(self.p,self.phase)

 def test_bounded_authenticated_binary_capture(self):
  self.graphs()
  result=M.capture_inputs(self.p,self.phase)
  self.assertTrue(result['exact_input_bytes_complete']);self.assertEqual(len(result['graphs']),4)
  self.assertEqual(len(result['authenticated_raw_inputs']),2);self.assertFalse(result['cache_hit'])
  for digest in result['authenticated_raw_inputs']:
   self.assertEqual(h((Path(self.phase['root'])/'compiler-evidence'/(digest+'.mlir')).read_bytes()),digest)

 def test_missing_binary_evidence_never_claims_complete(self):
  debug=self.graphs();(debug/'forward/module.mlir').write_bytes(b'wrong input')
  result=M.capture_inputs(self.p,self.phase)
  self.assertFalse(result['exact_input_bytes_complete']);self.assertEqual(len(result['missing_keys']),1)

 def test_exact_science_harness_guard_pins_and_driver_order(self):
  old=json.loads((REPO/'research/v2/operator/20260926-ct-architecture-startup-only/CT_BINDING.disabled.json').read_text())
  for phase in P['phases']:
   args=phase['training_command'].copy();args[3]='train.py';self.assertEqual(args,old['training_command'])
   self.assertEqual(phase['environment']['NEURON_CC_FLAGS'],old['training_environment']['NEURON_CC_FLAGS'])
   self.assertEqual(phase['environment']['NEURON_COMPILE_CACHE_URL'],P['cache_environment']['NEURON_COMPILE_CACHE_DIR'])
  self.assertIn('--fresh-caches',P['phases'][0]['command']);self.assertNotIn('--fresh-caches',P['phases'][1]['command'])
  for file,key in [('identity_checks.py','helper_sha256')]:self.assertEqual(h((HERE/file).read_bytes()),P[key])
  for path,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
   self.assertEqual(h((REPO/'research/v2/operator/20260926-common-host-idle-guard'/path).read_bytes()),P[key])
  raw=subprocess.check_output(['git','show',P['harness_git_object']],cwd=REPO)
  self.assertEqual(h(raw),P['harness_sha256'])
  self.assertEqual(h((REPO/'prepare.py').read_bytes()),P['prepare_sha256'])
  source=subprocess.check_output(['git','show',P['source_commit']+':'+P['source_git_path']],cwd=REPO)
  self.assertEqual(h(source),P['source_sha256'])
  driver=(HERE/'host_driver.py').read_text()
  driver_ast=ast.parse(driver)
  plan_pin=next(ast.literal_eval(n.value) for n in driver_ast.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PLAN_SHA' for t in n.targets))
  self.assertEqual(plan_pin,h((HERE/'PAIR_BINDING.disabled.json').read_bytes()))
  for phase in P['phases']:
   package=HERE/phase['name'].lower()/'package'
   self.assertEqual(h((package/'launch-command.txt').read_bytes()),phase['launch_sha256'])
   self.assertEqual(h((package/'provenance.json').read_bytes()),phase['provenance_sha256'])
  self.assertIn('pass_fds=(lock_fd,)',driver)
  self.assertLess(driver.index("if name=='S2':"),driver.index('subprocess.call('))
  self.assertIn("require(evidence['exact_input_bytes_complete']",driver)
  self.assertIn("require(after==before",driver)
  self.assertEqual(driver.count('require_idle_under_inherited_lock('),3)
  self.assertFalse(P['launch_authorized']);self.assertFalse(P['source_identity']['actual_import_origin_observed'])
  self.assertEqual(P['watchdogs']['S1_fifth_boundary_gate_seconds'],900)
  self.assertEqual(P['watchdogs']['per_phase_whole_harness_seconds'],1800)

if __name__=='__main__':unittest.main(verbosity=2)
