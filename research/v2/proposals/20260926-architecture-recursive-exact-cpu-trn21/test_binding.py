"""Stdlib-only mechanical checks and immutable full-G2 receipt fixtures."""
import ast,copy,hashlib,importlib.util,json,math,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
P=json.loads((HERE/'CPU_PLAN.disabled.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module():
 spec=importlib.util.spec_from_file_location('bound_fixture_validator',HERE/'validate_receipts.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Tests(unittest.TestCase):
 def test_exact_sources_harness_guard_and_support(self):
  self.assertFalse(P['launch_authorized']);self.assertFalse(P['native_launch_authorized']);self.assertTrue(P['local_tensor_execution_prohibited'])
  for name in ('reference','default','off','on','g2'):
   row=P['sources'][name];raw=subprocess.check_output(['git','show',row['commit']+':'+row['git_path']],cwd=REPO)
   self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
  for key in ('idle_guard','dispatch'):
   row=P['sources'][key];self.assertEqual(digest(REPO/row['git_path']),row['sha256'])
  for file,expected in P['support_sha256'].items():self.assertEqual(digest(HERE/file),expected)
  if 'driver_sha256' in P:self.assertEqual(digest(HERE/'host_driver.py'),P['driver_sha256'])
 def test_same_guarded_checks_and_sequential_child(self):
  before=subprocess.check_output(['git','show',P['driver_derivation']['commit']+':'+P['driver_derivation']['path']],cwd=REPO).decode()
  actual=(HERE/'host_driver.py').read_text()
  def function(s):return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='checks'),include_attributes=False)
  self.assertEqual(function(before),function(actual.replace('ip-172-31-34-85','ip-172-31-32-253')))
  self.assertIn("for name in ('reference', 'default', 'off', 'on'):",actual)
  self.assertIn("if phase.startswith('standard-compare'):",actual)
  self.assertIn("pass_fds=(lock_fd,)",actual)
  self.assertIn("('standard-compare-default'",actual);self.assertIn("('standard-compare-off'",actual)
  self.assertNotIn("('standard-compare-on'",actual)
  self.assertEqual(P['device_lock'],'/home/ubuntu/.cache/trainium_freshstart/device0.lock')
  self.assertEqual(P['minimum_available_ram_gib'],64)
 def test_only_schedule_cli_factor(self):
  requests={name:json.loads((HERE/name/'request.json').read_text()) for name in ('reference','default','off','on')}
  base=requests['reference']['argv']
  self.assertEqual(requests['default']['argv'],base)
  self.assertEqual(requests['off']['argv'],base+['--no-recursive-schedule'])
  self.assertEqual(requests['on']['argv'],base+['--recursive-schedule'])
  for name,r in requests.items():
   self.assertEqual(r['command'][r['command'].index('--')+1:],r['argv'])
   self.assertEqual(r['command'][r['command'].index('--nproc')+1],'4')
   self.assertEqual(r['command'][r['command'].index('--steps')+1],'3')
   self.assertEqual(r['command'][r['command'].index('--micro-tokens')+1],'256')
   self.assertEqual(r['parameters'],113256724);self.assertEqual(r['parameter_tensors'],65)
   self.assertNotIn('--stub-prepare',r['command']);self.assertNotIn('--real-data',r['command'])
   record=next(p for p in P['probes'] if p['name']==name)
   self.assertEqual(digest(HERE/name/'request.json'),record['request_sha256'])
 def fixture(self,root):
  raw=json.loads((REPO/'research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/results/candidate-on/g2.json').read_text())
  for name in ('reference','default','off','on'):
   arm=root/name;arm.mkdir();req=json.loads((HERE/name/'request.json').read_text());r=copy.deepcopy(raw)
   r.update(train_py_sha256=req['source_sha256'],argv=req['argv'])
   if name=='on':r['ranks'][0]['final_digests']['lm_head.weight']='intentional ON fixture difference'
   for file,value in [('request.json',req),('g2.json',r),('terminal.json',{'exit_code':0,'verdict':'PASS'}),('checkpoint-identity.json',{'bytes':453000000,'sha256':'f'*64})]:
    (arm/file).write_text(json.dumps(value))
 def test_true_full_shape_fixture_and_negative_gates(self):
  v=module()
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root)
   self.assertEqual(v.validate(root)['status'],'FULL_PARAMETER_CPU_RECEIPTS_MATCH')
   file=root/'default/g2.json';good=json.loads(file.read_text())
   for mutation in ('digest','loss','nan','shape','plan','prepare','params','rank','exit','checkpoint'):
    r=copy.deepcopy(good)
    if mutation=='digest':r['ranks'][0]['final_digests']['lm_head.weight']='wrong'
    if mutation=='loss':r['ranks'][0]['losses'][0]+=.01
    if mutation=='nan':r['ranks'][0]['losses'][0]=float('nan')
    if mutation=='shape':r['micro_batch']=[1,16]
    if mutation=='plan':r['plan_sha256']='wrong'
    if mutation=='prepare':r['prepare']['stub']=True
    if mutation=='params':r['ranks'][0]['n_params']=64
    if mutation=='rank':r['ranks'].pop()
    if mutation=='exit':r['torchrun_rc']=1
    if mutation=='checkpoint':r['ranks'][0]['checkpoint_check']['ok']=False
    file.write_text(json.dumps(r))
    with self.assertRaises((RuntimeError,KeyError)):v.validate(root)
   file.write_text(json.dumps(good))
 def test_checkpoint_three_off_sources_no_on_equivalence(self):
  source=(HERE/'compare_checkpoints.py').read_text()
  self.assertIn("for name in ('reference','default','off'):",source)
  self.assertNotIn("for name in ('reference','default','off','on'):",source)
  self.assertIn("weights_only=True",source)
  self.assertIn("torch.equal(logits[0],value)",source)
  self.assertIn("a.reshape(-1).view(torch.uint8)",source)
  self.assertEqual(P['source_global_tokens_per_update'],131072)
  self.assertEqual(P['diagnostic_tokens_all_ranks_per_update'],2048)
  self.assertFalse(P['production_update_equivalent'])
if __name__=='__main__':unittest.main(verbosity=2)
