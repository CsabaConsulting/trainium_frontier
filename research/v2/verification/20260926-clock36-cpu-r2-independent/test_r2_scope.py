"""Own stdlib scope/pin proof for the r2 helper-only verification extension."""
import ast,hashlib,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'20260926-clock36-recursive-exact-cpu-trn22'
P=json.loads((HERE/'CPU_PLAN.disabled.json').read_text())
Q=json.loads((OLD/'CPU_PLAN.disabled.json').read_text())
class Scope(unittest.TestCase):
 def test_driver_exact_rebinding_only(self):
  old=(OLD/'host_driver.py').read_text().replace(Q['stage_root'],P['stage_root'])
  for a,b in zip(Q['probes'],P['probes']):old=old.replace(a['request_sha256'],b['request_sha256'])
  old=old.replace(Q['support_sha256']['compare_checkpoints.py'],P['support_sha256']['compare_checkpoints.py'])
  self.assertEqual(old,(HERE/'host_driver.py').read_text())
  self.assertEqual((OLD/'validate_receipts.py').read_bytes(),(HERE/'validate_receipts.py').read_bytes())
  for key in ('ordering','device_lock','full_model','minimum_available_ram_gib','minimum_free_disk_gib','operator_minimum_free_disk_gib','timeout_seconds','checkpoint_compare_timeout_seconds','expected_lock_identity','max_train_seconds','startup_boundary_scope','postflight'):
   self.assertEqual(P[key],Q[key],key)
 def test_requests_science_unchanged_and_pins(self):
  for name in ('reference','default','off','on'):
   old=(OLD/name/'request.json').read_text().replace(Q['stage_root'],P['stage_root']).replace('-trn22-g2-r1','-trn22-g2-r2')
   self.assertEqual(old,(HERE/name/'request.json').read_text())
   request=json.loads(old);probe=next(q for q in P['probes'] if q['name']==name)
   self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),probe['request_sha256'])
   self.assertEqual(hashlib.sha256(json.dumps(request['command'],separators=(',',':')).encode()).hexdigest(),probe['command_sha256'])
  for key in ('driver_sha256',):self.assertEqual(P[key],hashlib.sha256((HERE/'host_driver.py').read_bytes()).hexdigest())
  self.assertFalse(P['launch_authorized']);self.assertFalse(P['native_launch_authorized'])
  self.assertFalse(P['host_reserved']);self.assertIsNone(P['results'])
 def test_new_normal_loader_before_separate_old_shim(self):
  source=(HERE/'compare_checkpoints.py').read_text()
  self.assertLess(source.index('ordinary=mod.load_for_eval'),source.index('qualification=verify_trained_model'))
  self.assertLess(source.index('qualification=verify_trained_model'),source.index('mod.compute_dtype_for=lambda'))
  self.assertIn("if name=='on':",source)
  self.assertIn('weights_only=True',source)
  self.assertIn('113256724',source)
  tree=ast.parse(source)
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_trained_model')
  calls={ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call)}
  self.assertNotIn('causality_check',calls);self.assertNotIn('make_dataloader',calls)
  self.assertIn('torch.flip',calls);self.assertIn('model_state',calls)
  self.assertEqual(P['checkpoint_extension']['timeout_seconds'],600)
  self.assertFalse(P['checkpoint_extension']['quality_or_organizer_causality_claim'])
if __name__=='__main__':unittest.main(verbosity=2)
