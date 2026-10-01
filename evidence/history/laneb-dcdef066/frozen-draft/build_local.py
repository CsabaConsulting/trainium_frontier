"""One-use repository-local draft writer; apply_patch only, no export/model/host calls."""
import difflib,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def enc(v):return json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n'
def put(name,text):
 p=H/name
 if p.exists():assert p.read_bytes()==text.encode(),'NO_OVERWRITE:'+name;return
 patch='*** Begin Patch\n*** Add File: '+str(p)+'\n'+''.join('+'+line+'\n'for line in text.splitlines())+'*** End Patch\n'
 subprocess.run(['apply_patch'],input=patch,text=True,check=True,capture_output=True)
 assert p.read_bytes()==text.encode(),'EXACT_ADD_FILE_BYTES_REQUIRED:'+name
def main():
 assert not sys.flags.optimize and not(H/'MANIFEST.json').exists()
 s=importlib.util.spec_from_file_location('build_audit_evidence',H/'evidence.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
 actual=e.evaluate()
 source=e.checked(e.SOURCE_PATH,e.SOURCE).decode();put('train.py',source);put('launch-command.txt',e.COMMAND+'\n')
 assert e.sha((H/'train.py').read_bytes())==e.SOURCE and e.sha((H/'launch-command.txt').read_bytes())==e.COMMAND_SHA
 expected=e.checked(e.SOURCE_PATH.parent/'SOURCE.diff','894a76cc54f31fa870af864cdd239c6ecdae69d01492848eaa4818f414e0e910').decode()
 generated=''.join(difflib.unified_diff(e.checked(e.PARENT_PATH,e.PARENT).decode().splitlines(True),source.splitlines(True),fromfile=str(e.PARENT_PATH.relative_to(e.R)),tofile=str(e.SOURCE_PATH.relative_to(e.R))))
 assert expected==generated;put('SOURCE.diff',generated)
 for name,slot in(('LEGACY_FULL21_ADAPTER.diff','first21'),('LEGACY_PARENT_ADAPTER.diff','direct_parent21')):put(name,e.adapter_diff(slot))
 put('LEGACY_SCHEMA_DIAGNOSTIC.json',enc(actual['legacy_schema_diagnostic']))
 cold=dict(schema='r136-laneb-two-receipt-cold-evidence-v1',source_sha256=e.SOURCE,command=e.COMMAND,receipts=actual['receipts'],mean_public20m=actual['mean_public20m'],seed_scope=actual['seed_scope'],scope='Actual archived research run validity only; not package acceptance or official scores.',fresh_independent_package_review=None,claude_audit=None,package_accepted=False,launch_authorized=False,export_authorized=False,submission_authorized=False)
 put('cold-evidence.json',enc(cold));put('ancestor-evidence.json',enc(actual['direct_research_proven_parent']));put('source-cpu-evidence.json',enc(actual['source_cpu']));put('SOURCE_INVENTORY.json',enc(actual['static_inventory']))
 reasons=['Clean scoring-box dependency sufficiency UNKNOWN','End-to-end native train/eval identity UNKNOWN; fresh audit pending','Runtime isolation/eval-cache independence UNKNOWN','Both exact research FULL cold/budget receipts authenticated','Cumulative writes/current leftovers UNKNOWN','Minimal extra-environment sufficiency UNKNOWN','Fixed seed proved; timing-dependent table/stop behavior awaits audit','Concrete retained inactive paths/header/kernel-documentation gaps']
 put('CHECKLIST.json',enc(dict(schema='r136-laneb-eight-item-checklist-v1',source_sha256=e.SOURCE,items=[dict(item=n,status='PASS'if n==4 else'FAIL',reason=reason,details='AUDIT_BRIEF.md')for n,reason in enumerate(reasons,1)],unknown_is_fail_pending_evidence=True,fresh_independent_package_review=None,claude_audit=None,last_claude_audited_ancestor=None,package_accepted=False,submission_authorized=False)))
 references={}
 def add(path,pin,role):
  v=e.ref(path,pin);key=v['path']
  if key in references:assert references[key]['sha256']==pin;references[key]['roles'].append(role)
  else:references[key]=dict(v,roles=[role])
 def find_refs(v):
  if isinstance(v,dict):
   if set(('path','sha256','bytes'))<=set(v):add(Path(v['path']),v['sha256'],'authenticated_evidence_reference')
   for x in v.values():find_refs(x)
  elif isinstance(v,list):
   for x in v:find_refs(x)
 find_refs(actual)
 for path,pin,role in[(e.FULL,e.FULL_SHA,'pinned_full_adapter'),(e.ORIGINAL,e.ORIGINAL_SHA,'unchanged_original_full_gates'),(e.PARENT_PATH,e.PARENT,'direct_scientific_parent'),(e.SOURCE_PATH.parent/'SOURCE.diff','894a76cc54f31fa870af864cdd239c6ecdae69d01492848eaa4818f414e0e910','frozen_scientific_diff'),(e.SOURCE_PATH.parent/'EDITS.json','5d31e31da4b2abddd969da3b409e179d7669a6f64ba0c42367241f6b252c6752','exact_six_replacement_reconstruction'),(e.SOURCE_PATH.parent/'SPEC.md','f3cd4a4a5aadbbe82fbf349230555f10147f39ad452cc423880ab93a7b352f41','frozen_scientific_spec_not_fresh_audit'),(e.R/'prepare.py','5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb','immutable_provided_evaluator'),(e.R/'research/v2/operator/20260930-r113-laneb-wrapped-full-trn21-preparation/artifacts/snapshot/research/scratch/submission_rehearsal.py','7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b','unchanged_research_rehearsal_retained_snapshot'),(e.R/'workflow/REJECTION_CHECKLIST_2026-09-28_0315.md','71717878b1c070bb7bce459579ae31f8b26b6339db9c7661859db1f19b9b9612','required_eight_item_checklist'),(e.R/'research/v2/operator/20260930-r113-laneb-wrapped-full-trn21-preparation/evidence_r121.py','e1865c4b8eae480694dbf9c4bf5fca0935bb0a607351bc2645fcb58960e010c8','existing_source_cpu_binder'),(e.R/'research/v2/operator/20260930-r113-laneb-wrapped-model-native60-trn21-preparation/evidence.py','689c74ea4a6a298bf2b36c26bda7a3b26464cdccaac94977458bcf206fbb820d','existing_author_native_pair_binder')]:add(path,pin,role)
 put('REFERENCES.json',enc(dict(schema='absolute-canonical-references-v1',canonical_repository=str(e.R),files=[references[k]for k in sorted(references)],scope='Primary exact source/scaffold/helper and raw evidence files. Pinned helpers also retain their original transitive dependency checks. Raw archives are NOT copied.')))
 refusals=[]
 def refuse(name,fn):
  try:fn()
  except (AssertionError,KeyError,TypeError,ValueError,FileNotFoundError):refusals.append(name)
  else:raise AssertionError('DID_NOT_REFUSE:'+name)
 refuse('missing_explicit_archive_pin',lambda:e.verifier('first21').authenticate(e.JOBS['first21'],None))
 refuse('wrong_archive_pin',lambda:e.verifier('first21').authenticate(e.JOBS['first21'],'0'*64))
 refuse('wrong_packet_pin',lambda:e.verifier('first21').packet_for(dict(e.JOBS['first21'],packet_sha256='0'*64)))
 refuse('wrong_host_tuple',lambda:e.verifier('first21').packet_for(dict(e.JOBS['first21'],host='trn22',key='trn22-r113-laneb-wrapped-full-r1')))
 refuse('wrong_source_tuple',lambda:e.verifier('first21').packet_for(dict(e.JOBS['first21'],source_sha256=e.PARENT)))
 refuse('wrong_key_tuple',lambda:e.verifier('first21').packet_for(dict(e.JOBS['first21'],key='trn21-not-the-actual-key')))
 assert actual['legacy_schema_diagnostic']['status']=='REFUSED_EXPECTED_LEGACY_SCHEMA_MISMATCH'
 put('LOCAL_METADATA_TESTS.json',enc(dict(status='PASS_LOCAL_METADATA_TESTS_ONLY',source_copy_exact=True,command_exact=True,parent_diff_exact=True,original_full_verifier_unchanged=True,legacy_adapter_reverses_exactly=True,unmodified_legacy_mismatch_preserved=True,actual_receipt_validity=['first21','repeat22','direct_parent21'],source_cpu_existing_evidence_reauthenticated=True,new_model_or_cpu_tests_run=False,negative_refusals=refusals,fixtures_only_change_expected_metadata_no_raw_evidence_changed=True,fresh_independent_review=False,claude_audit=False,package_accepted=False,host_calls=0,model_or_torch_execution=False,export_performed=False)))
 files=sorted(p for p in H.rglob('*')if p.is_file());put('MANIFEST.json',enc(dict(schema='r136-laneb-two-receipt-repo-audit-draft-v1',source_sha256=e.SOURCE,status='INCOMPLETE_AUDIT_NOT_ACCEPTED',files={str(p.relative_to(H)):e.sha(p.read_bytes())for p in files},package_accepted=False,launch_authorized=False,export_authorized=False,submission_authorized=False)))
 proc=subprocess.run([sys.executable,'-I','-B',str(H/'verify_local.py')],cwd=e.R,text=True,capture_output=True);assert proc.returncode==0,proc.stdout+proc.stderr;print(proc.stdout)
if __name__=='__main__':main()
