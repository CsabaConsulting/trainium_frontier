"""Own stdlib-only literal-delta checks; never imports bound source or Torch."""
import ast,hashlib,json,subprocess,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OLD=REPO/'research/v2/proposals/20260926-startup-adjusted-exact-cpu-trn21'
P=json.loads((HERE/'CPU_PLAN.disabled.json').read_text())
Q=json.loads((OLD/'CPU_PLAN.disabled.json').read_text())
def h(s):return hashlib.sha256(s).hexdigest()
class Rebind(unittest.TestCase):
 def test_whole_driver_identity_and_helpers(self):
  before=subprocess.check_output(['git','show','eb3590f3:'+str((OLD/'host_driver.py').relative_to(REPO))],cwd=REPO).decode()
  pairs=[(Q['stage_root'],P['stage_root']),(Q['prepare_dir'],P['prepare_dir']),(Q['expected_hostname'],P['expected_hostname'])]+[(a['request_sha256'],b['request_sha256']) for a,b in zip(Q['probes'],P['probes'])]
  for a,b in pairs:before=before.replace(a,b)
  self.assertEqual(before,(HERE/'host_driver.py').read_text())
  for file in ('validate_receipts.py','compare_checkpoints.py'):self.assertEqual((HERE/file).read_bytes(),(OLD/file).read_bytes())
  self.assertEqual(P['driver_sha256'],h((HERE/'host_driver.py').read_bytes()))
 def test_science_host_pins(self):
  base=json.loads((OLD/'on/request.json').read_text())['argv']
  for name in ('reference','default','off','on'):
   r=json.loads((HERE/name/'request.json').read_text())
   self.assertEqual(r['argv'],base+(['--recursive-schedule'] if name=='on' else ['--no-recursive-schedule'] if name=='off' else []))
   self.assertEqual(r['command'][r['command'].index('--')+1:],r['argv'])
   self.assertEqual(r['source_sha256'],P['sources'][name]['sha256'])
   self.assertEqual(r['prepare_dir'],P['prepare_dir']);self.assertNotIn('trn21',json.dumps(r))
   probe=next(p for p in P['probes'] if p['name']==name)
   self.assertEqual(probe['command_sha256'],h(json.dumps(r['command'],separators=(',',':')).encode()))
  self.assertEqual(P['source_candidate_commit'],'44926a6f35b7c221bb6cb6652eda3c77e9f10efd')
  self.assertEqual(P['sources']['reference']['commit'],'7419927c7417b5666a984ec9c2b849feadabaaa9')
  self.assertEqual(P['proposed_host'],'trn22');self.assertEqual(P['expected_hostname'],'ip-172-31-32-253')
  self.assertEqual(P['expected_lock_identity']['inode'],832589);self.assertEqual(P['expected_lock_identity']['device'],66305)
  self.assertEqual(P['device_lock'],'/home/ubuntu/.cache/trainium_freshstart/device0.lock')
 def test_resources_boundaries_no_authority(self):
  for k in ('nproc','steps','micro_batch','timeout_seconds','checkpoint_compare_timeout_seconds','seed','max_train_seconds','expected_plan_sha256','full_model','minimum_available_ram_gib','minimum_free_disk_gib'):self.assertEqual(P[k],Q[k],k)
  self.assertEqual(P['operator_minimum_free_disk_gib'],10.5)
  self.assertFalse(P['startup_boundary_scope']['fifth_boundary_exercised']);self.assertEqual(P['startup_boundary_scope']['real_prepare_startup_steps_excluded'],5)
  self.assertEqual(P['startup_boundary_scope']['updates_per_rank'],3)
  for text in ('PPID/PGID/SID','compiler','neuron-ls'):self.assertIn(text,P['postflight'])
  for value in (P,json.loads((HERE/'EXPERIMENT_PROPOSAL.json').read_text())):
   for key in ('launch_authorized','native_launch_authorized','source_adoption'):self.assertFalse(value[key])
  self.assertFalse(P['host_reserved']);self.assertTrue(P['local_tensor_execution_prohibited']);self.assertIsNone(P['results'])
  tree=ast.parse((REPO/'prepare.py').read_text())
  values={t.id:n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) for t in n.targets if isinstance(t,ast.Name)}
  self.assertEqual(values['TRAIN_STARTUP_STEPS_EXCLUDED'],5);self.assertEqual(values['TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS'],900)
if __name__=='__main__':unittest.main(verbosity=2)
