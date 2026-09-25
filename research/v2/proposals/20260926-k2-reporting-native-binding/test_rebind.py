"""Proof that successor binding changes only fixed provenance/path/hash constants."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OLD=REPO/'research/v2/proposals/20260926-k2-resident-native-binding'
P=json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())
Q=json.loads((OLD/'NATIVE_BINDING.disabled.json').read_text())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def mapped(raw):
 for old,new in [(Q['label'],P['label']),(Q['support_sha256']['check_native.py'],P['support_sha256']['check_native.py']),
                 (Q['support_sha256']['rank_entry.py'],P['support_sha256']['rank_entry.py'])]:
  raw=raw.replace(old,new)
 return raw

class Mechanical(unittest.TestCase):
 def test_original_packet_frozen(self):
  names=subprocess.check_output(['git','ls-tree','-r','--name-only','8555da97','--',str(OLD.relative_to(REPO))],cwd=REPO).decode().splitlines()
  self.assertTrue(names)
  for name in names:self.assertEqual(subprocess.check_output(['git','show','8555da97:'+name],cwd=REPO),(REPO/name).read_bytes())
 def test_only_constant_changes(self):
  for file in ('host_driver.py','rank_entry.py','bound-command.txt'):
   self.assertEqual(mapped((OLD/file).read_text()),(HERE/file).read_text(),file)
  self.assertEqual((OLD/'validate_receipts.py').read_bytes(),(HERE/'validate_receipts.py').read_bytes())
  for key in ('command','checker_args','environment','dispatch_command','outer_command'):
   self.assertEqual(json.loads(mapped(json.dumps(Q[key]))),P[key],key)
  for key in ('expected_hostname','cache_keys','timeout_seconds','term_grace_seconds','pending_gates','completion_rule','scientific_scope'):
   self.assertEqual(P[key],Q[key],key)
 def test_source_checker_and_unchanged_kernel_reference(self):
  for file in ('consumer_nki.py','check_native.py','reference.py'):
   raw=subprocess.check_output(['git','show',P['source_module_commit']+':'+P['source_module_path']+'/'+file],cwd=REPO)
   self.assertEqual(sha(raw),P['support_sha256'][file])
   if file!='check_native.py':
    original=subprocess.check_output(['git','show',Q['source_module_commit']+':'+Q['source_module_path']+'/'+file],cwd=REPO)
    self.assertEqual(raw,original)
  self.assertEqual(P['support_sha256']['check_native.py'],'89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850')
  self.assertEqual(P['rebind']['original_source_verdict'],'REFUSE_PRESERVED_90db28e0')
  self.assertFalse(P['launch_authorized']);self.assertFalse(P['operator_assigned'])
if __name__=='__main__':unittest.main(verbosity=2)
