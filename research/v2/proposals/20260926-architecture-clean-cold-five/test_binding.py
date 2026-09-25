"""Stdlib-only immutable cold-five binding checks. No workload imports or host calls."""
import ast, hashlib, json, shlex, subprocess, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
B=json.loads((HERE/'CT_BINDING.disabled.json').read_text())
D=(HERE/'host_driver.py').read_text()
class Tests(unittest.TestCase):
 def test_frozen_pins(self):
  for commit,path,expected in [(B['source_commit'],B['source_git_path'],B['source_sha256']),(B['harness_source_commit'],B['harness_source_git_path'],B['harness_sha256'])]:
   self.assertEqual(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)).hexdigest(),expected)
  self.assertEqual(hashlib.sha256((HERE/'host_driver.py').read_bytes()).hexdigest(),B['driver_sha256'])
  for file,expected in B['package_files'].items():self.assertEqual(hashlib.sha256((HERE/file).read_bytes()).hexdigest(),expected)
  for name in ('idle_guard','dispatch_driver'):
   self.assertEqual(hashlib.sha256((REPO/'research/v2/operator/20260926-common-host-idle-guard'/(name+'.py')).read_bytes()).hexdigest(),B[name+'_sha256'])
 def test_exact_long_plus_only_step_cap(self):
  proven=(REPO/'submissions/dense6x1024_direct_review_matrixlr015/launch-command.txt').read_text().strip()
  self.assertEqual((HERE/'package/launch-command.txt').read_text().strip(),proven+' --num-steps 5')
  self.assertEqual(B['training_command'],shlex.split(proven)[1:]+['--num-steps','5'])
  self.assertEqual(json.loads((HERE/'package/provenance.json').read_text())['launch_command'],B['launch_command'])
  for key in ('command','training_command','outer_command','detached_command'):
   self.assertEqual(hashlib.sha256(json.dumps(B[key],separators=(',',':'),ensure_ascii=True).encode()).hexdigest(),B[key+'_sha256'])
 def test_guard_single_child_and_failure_gates(self):
  t=ast.parse(D); calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  guards=[n for n in calls if n.func.attr=='require_idle_under_inherited_lock']
  jobs=[n for n in calls if n.func.attr=='call' and ast.unparse(n.func.value)=='subprocess']
  self.assertEqual(len(guards),1);self.assertEqual(len(jobs),1);self.assertLess(guards[0].lineno,jobs[0].lineno)
  self.assertEqual([ast.unparse(k.value) for k in jobs[0].keywords if k.arg=='pass_fds'],['(lock_fd,)'])
  for token in ("sys.flags.optimize","socket.gethostname()", "'SOURCE_SHA'","'LAUNCH_SHA'","'PROVENANCE_SHA'","'DISPATCH_SHA'","'COMPILER_URL'","'PREPARE_SHA'","'HARNESS_SHA'","'SANDBOX_ALREADY_EXISTS'","'IDLE_GUARD_SHA'","'DISK_BELOW_6_5_GIB'","'STOP: '"):
   self.assertIn(token,D)
  self.assertEqual(B['outer_command'][:6],['/usr/bin/flock','-n','/home/ubuntu/.cache/trainium_freshstart/device0.lock','/usr/bin/python3','-I','-B'])
 def test_cold_cache_accounting_no_automatic_full(self):
  self.assertEqual(B['command'][-3:],['--mode','train','--fresh-caches'])
  e=B['training_environment'];self.assertEqual(e['NEURON_COMPILE_CACHE_URL'],B['sandbox']+'/caches/compiler')
  self.assertEqual(e['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'],'1800');self.assertEqual(e['NEURON_COMPETITION_R1_SEED'],'42')
  caches={k:v for k,v in B['harness_injected_environment'].items() if k.endswith('_DIR') and not k.startswith('NEURON_COMPETITION')}
  self.assertEqual(len(set(caches.values())),6)
  self.assertTrue(all(v.startswith(B['sandbox']+'/caches/') for v in caches.values()))
  raw=subprocess.check_output(['git','show',B['harness_source_commit']+':'+B['harness_source_git_path']],cwd=REPO).decode()
  for s in ['cache.mkdir(parents=True, exist_ok=False)','assert not list(cache.iterdir())','charged_through_process_exit=result["wall_seconds"] - allowance','min(observed[4]["observed_seconds"], 900.0)']:
   self.assertIn(s,raw)
  self.assertLess(raw.index('cache.mkdir('),raw.index('process = subprocess.Popen('))
  self.assertTrue(B['no_eval']);self.assertTrue(B['no_automatic_full']);self.assertFalse(B['launch_authorized']);self.assertFalse(B['stage_authorized'])
  self.assertIn('REFUSE',B['status']);self.assertEqual(B['hard_cold_boundary_seconds'],900)
 def test_only_documented_driver_delta(self):
  der=B['driver_derivation'];raw=subprocess.check_output(['git','show',der['commit']+':'+der['path']],cwd=REPO).decode()
  self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),der['sha256'])
  tree=ast.parse(raw); labels={}
  for n in tree.body:
   if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('LABEL','SOURCE_SHA'):labels[n.targets[0].id]=ast.literal_eval(n.value)
  expected=raw.replace(labels['LABEL'],B['label']).replace(labels['SOURCE_SHA'],B['source_sha256'])
  # Strip only the added explicit refusal checks/import, leaving old control flow byte-identical.
  lines=D.splitlines(keepends=True)
  remove=('import socket','    if sys.flags.optimize:','    assert socket.gethostname()','    assert sha(ROOT/\'package/launch-command.txt\')','    assert sha(ROOT/\'package/provenance.json\')','    assert sha(ROOT/\'dispatch_driver.py\')','    assert ENV[\'NEURON_COMPILE_CACHE_URL\']')
  restored=''.join(line for line in lines if not line.startswith(remove))
  self.assertEqual(restored.rstrip('\n'),expected.rstrip('\n'))
if __name__=='__main__':unittest.main(verbosity=2)
