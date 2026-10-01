"""Read-only draft verification; metadata success is never package acceptance."""
import hashlib,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def load():
 s=importlib.util.spec_from_file_location('laneb_audit_evidence',H/'evidence.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify():
 assert not sys.flags.optimize
 manifest=json.loads((H/'MANIFEST.json').read_bytes())
 for name,pin in manifest['files'].items():
  p=Path(name);assert not p.is_absolute()and '..'not in p.parts
  assert hashlib.sha256((H/p).read_bytes()).hexdigest()==pin,'LOCAL_PIN:'+name
 assert set(manifest['files'])=={str(p.relative_to(H))for p in H.rglob('*')if p.is_file()and p.name!='MANIFEST.json'},'UNMANIFESTED_LOCAL_FILE'
 e=load();assert manifest['source_sha256']==e.SOURCE
 for k in('launch_authorized','export_authorized','submission_authorized','package_accepted'):assert manifest[k]is False
 refs=json.loads((H/'REFERENCES.json').read_bytes())
 for v in refs['files']:
  assert Path(v['path']).is_absolute();raw=e.checked(Path(v['path']),v['sha256']);assert len(raw)==v['bytes']
 assert e.checked(H/'train.py',e.SOURCE)==e.checked(e.SOURCE_PATH,e.SOURCE)
 assert e.checked(H/'launch-command.txt',e.COMMAND_SHA)==(e.COMMAND+'\n').encode()
 actual=e.evaluate()
 cold=json.loads((H/'cold-evidence.json').read_bytes())
 assert cold['receipts']==actual['receipts']and cold['mean_public20m']==actual['mean_public20m']and cold['source_sha256']==e.SOURCE
 assert json.loads((H/'ancestor-evidence.json').read_bytes())==actual['direct_research_proven_parent']
 assert json.loads((H/'source-cpu-evidence.json').read_bytes())==actual['source_cpu']
 assert json.loads((H/'SOURCE_INVENTORY.json').read_bytes())==actual['static_inventory']
 assert json.loads((H/'LEGACY_SCHEMA_DIAGNOSTIC.json').read_bytes())==actual['legacy_schema_diagnostic']
 for name,slot in(('LEGACY_FULL21_ADAPTER.diff','first21'),('LEGACY_PARENT_ADAPTER.diff','direct_parent21')):assert (H/name).read_text()==e.adapter_diff(slot)
 base=e.checked(e.PARENT_PATH,e.PARENT).decode();candidate=e.checked(e.SOURCE_PATH,e.SOURCE).decode()
 edits=json.loads(e.checked(e.SOURCE_PATH.parent/'EDITS.json','5d31e31da4b2abddd969da3b409e179d7669a6f64ba0c42367241f6b252c6752'))
 import difflib
 for before,after in edits:
  assert base.count(before)==1; base=base.replace(before,after)
 assert base==candidate,'EXACT_PARENT_DIFF_RECONSTRUCTION_FAILED'
 generated=''.join(difflib.unified_diff(e.PARENT_PATH.read_text().splitlines(True),candidate.splitlines(True),fromfile=str(e.PARENT_PATH.relative_to(e.R)),tofile=str(e.SOURCE_PATH.relative_to(e.R))))
 assert (H/'SOURCE.diff').read_text()==generated
 checklist=json.loads((H/'CHECKLIST.json').read_bytes());assert len(checklist['items'])==8
 assert [v['status']for v in checklist['items']]==['FAIL','FAIL','FAIL','PASS','FAIL','FAIL','FAIL','FAIL']
 assert checklist['fresh_independent_package_review']is checklist['claude_audit']is None and checklist['package_accepted']is False
 return dict(status='LOCAL_METADATA_AND_FIXED_RECEIPTS_VERIFIED',package_status='INCOMPLETE_AUDIT_NOT_ACCEPTED',source_sha256=e.SOURCE,command_sha256=e.COMMAND_SHA,manifest_sha256=e.sha((H/'MANIFEST.json').read_bytes()),local_files=len(manifest['files']),external_files=len(refs['files']),public20m_scores=[r['verifier_proof']['val_bpb']for r in actual['receipts']],mean_public20m=actual['mean_public20m'],same_seed=42,independent_seeds=False,checklist_pass_items=[4],checklist_nonpassing_items=[1,2,3,5,6,7,8],unknowns=actual['unknowns'],fresh_package_review=None,claude_audit=None,host_calls=0,model_or_torch_execution=False,package_accepted=False,launch_authorized=False,export_authorized=False,submission_authorized=False)
if __name__=='__main__':print(json.dumps(verify(),indent=2,allow_nan=False))
