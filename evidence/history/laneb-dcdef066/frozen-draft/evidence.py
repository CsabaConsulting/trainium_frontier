"""Read-only fixed-receipt LaneB audit evidence. Stdlib only; never imports train.py."""
import ast,base64,difflib,hashlib,importlib.util,json,math,re,sys,types
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
R=H.parents[3]
SOURCE='dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d'
PARENT='e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310'
SOURCE_PATH=R/'research/v2/proposals/20260930-r113-laneb-wrapped-ce2-flat-z/train.py'
PARENT_PATH=R/'research/v2/proposals/20260929-r109-ce2-flat-zbundle/train.py'
COMMAND='NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0'
COMMAND_SHA='223d4291ca69d7fc80b83f82d03e3733ddd2e99cd0dfd2659cc0e883258340bd'
FULL=R/'research/v2/verification/20260930-r132-current-full-receipts/verify.py'
FULL_SHA='ace0b4db184c9269d9d030b82a38314f7a7acbe4384a93c13c37e23575863b37'
ORIGINAL=R/'research/v2/verification/20260929-r73-a3-all3/verify.py'
ORIGINAL_SHA='35f75a833b909d78e7d3e2010939544eefdf4ca78db6ad712f552e82110f4ebe'
REVIEW=R/'research/v2/verification/20260930-r113-laneb-fresh-scored-review/REVIEW.json'
REVIEW_SHA='c5ebb3320e5c5b329beff5dcecac61eb80a6a98a23592d74c5e105caeac841f8'
AUTHOR_PATH='research/v2/operator/20260930-r113-laneb-wrapped-author-cpu-trn21-preparation/collect-20260929T194147190691Z.json'
AUTHOR_SHA='fbd77a18d2683ac09b3ec8e45c0c24a2498b4fb9014de2f4f5dc4faef1940fe1'
OWN_PATH='research/v2/operator/20260930-r113-laneb-fresh-ownref-cpu-trn22-preparation/collect-20260929T200655964822Z.json'
OWN_SHA='95cc6f1bafd0e50790f215121230cf387a46436f1ac9774c086b052ce0d3cae8'
JOBS={
 'first21':dict(folder='20260930-r113-laneb-wrapped-full-trn21-preparation',host='trn21',key='trn21-r113-laneb-wrapped-full-r1',packet_sha256='ee56667ee15e89ca386d6279b0e4b82b048df4985d3936f25f94738d85a9e5f4',source_sha256=SOURCE,archive_sha256='05a764083999229bb92dc7a9c3b177480dcdb3d0f526cdc78bdb884c73645bac',selection_member='R113_ROOT_SELECTION.json',expected_public20m=.9663196934983302),
 'repeat22':dict(folder='20260930-r127-laneb-samebyte-repeat-full-trn22-preparation',host='trn22',key='trn22-r127-laneb-samebyte-repeat-full-r1',packet_sha256='4d8b5abfb6a4ce749debec5ce7af14d67d498d4cb92e6f74f3846b54a192e9b8',source_sha256=SOURCE,archive_sha256='defb95ea5df8ac81778e4f0c637ab44846b779ceda006b3b6b16de748e962b38',selection_member='R98_ROOT_SELECTION.json',expected_public20m=.9659050462155375),
 'direct_parent21':dict(folder='20260929-r109-ce2-flat-z-full-trn21-preparation',host='trn21',key='trn21-r109-ce2-flat-z-full-r1',packet_sha256='28d29a6ea0325e320164672e46ea29bb3d4104a58cd8df94279dccce396a5584',source_sha256=PARENT,archive_sha256='088637f6f4db8ce993f5cc52e80ccc95f5e3441aacb5cc06ce88e59af2be1a0f',selection_member='R100_ROOT_SELECTION.json',expected_public20m=.9672630039667548)}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def checked(path,pin):
 assert isinstance(pin,str)and re.fullmatch('[0-9a-f]{64}',pin),'EXPLICIT_SHA256_REQUIRED'
 raw=Path(path).read_bytes();assert sha(raw)==pin,'PIN_DRIFT:'+str(path);return raw
def load(path,name,pin):
 checked(path,pin);s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def ref(path,pin):
 raw=checked(path,pin);return dict(path=str(Path(path).resolve()),sha256=pin,bytes=len(raw))
def raw_archive(path,pin):
 outer=json.loads(checked(path,pin));a=outer
 if 'stdout'in outer:
  assert outer['returncode']==0 and not outer['stderr'];a=json.loads(outer['stdout'])
 for name,v in a['files'].items():
  assert not Path(name).is_absolute()and '..'not in Path(name).parts
  raw=v['text'].encode();assert sha(raw)==v['sha256']and len(raw)==v['bytes'],'RAW_MEMBER:'+name
 return a
def members(a):return {n:dict(sha256=v['sha256'],bytes=v['bytes'])for n,v in sorted(a['files'].items())}

def verifier(slot):
 """Only adapt the real selection-member filename in two exact legacy packets."""
 job=JOBS[slot];raw=checked(FULL,FULL_SHA).decode();checked(ORIGINAL,ORIGINAL_SHA)
 old="f['R98_ROOT_SELECTION.json']";new="f['"+job['selection_member']+"']"
 assert raw.count(old)==1
 adapted=raw.replace(old,new)
 assert adapted.replace(new,old)==raw,'NON_MEMBER_ADAPTER_CHANGE'
 m=types.ModuleType('audit_full_'+slot);m.__file__=str(FULL)
 exec(compile(adapted,str(FULL),'exec'),m.__dict__)
 m.frozen_files()
 return m

def adapter_diff(slot):
 raw=checked(FULL,FULL_SHA).decode();member=JOBS[slot]['selection_member']
 new=raw.replace("f['R98_ROOT_SELECTION.json']","f['"+member+"']")
 return ''.join(difflib.unified_diff(raw.splitlines(True),new.splitlines(True),fromfile='pinned-ace0b4db/verify.py',tofile=slot+'-exact-legacy-member/in-memory-verify.py'))

def legacy_diagnostic():
 """Replay/preserve the real initial refusal; never fabricate an absent R98 member."""
 m=load(FULL,'audit_unmodified_full',FULL_SHA);m.frozen_files();job=JOBS['first21']
 try:m.packet_for(job)
 except KeyError as error:
  assert error.args==('R98_ROOT_SELECTION.json',)
  return dict(status='REFUSED_EXPECTED_LEGACY_SCHEMA_MISMATCH',exception='KeyError',missing_member='R98_ROOT_SELECTION.json',actual_member=job['selection_member'],packet_sha256=job['packet_sha256'],original_verifier_sha256=FULL_SHA,archive_reclassified=False,packet_modified=False,selection_synthesized=False,resolution='One exact member-name lookup adapted in memory; all original checks preserved.')
 raise AssertionError('LEGACY_SCHEMA_CHANGED_UNEXPECTEDLY')

def receipt(slot):
 job=JOBS[slot];m=verifier(slot);p=R/'research/v2/operator'/job['folder']/'artifacts'
 proof=m.authenticate(dict(job,slot=slot),job['archive_sha256'])
 assert proof['status']=='PASS'and proof['val_bpb']==job['expected_public20m']
 a=raw_archive(p/'results-bundle.json',job['archive_sha256']);packet=json.loads(checked(p/'packet.json',job['packet_sha256']));f=a['files'];b=packet['binding']
 original_selection=base64.b64decode(packet['files'][job['selection_member']]['b64'],validate=True)
 assert original_selection==f[job['selection_member']]['text'].encode()
 selection=json.loads(original_selection);assert (selection['host'],selection['key'],selection['source_sha256'])==(job['host'],job['key'],job['source_sha256'])
 run=json.loads(f['train/rehearsal.json']['text']);ev=json.loads(f['eval/rehearsal.json']['text']);pre=json.loads(f['operator-preflight.json']['text'])
 retained={n:json.loads(f[n]['text'])for n in('checkpoint.json','checkpoint-after-postflight.json')}
 safe=lambda env:{k:v for k,v in env.items()if k.startswith(('NEURON_','TORCH','NKI_','PYTHON','TRN_'))or k=='PATH'}
 return dict(verifier_proof=proof,raw_archive=ref(p/'results-bundle.json',job['archive_sha256']),actual_packet=ref(p/'packet.json',job['packet_sha256']),actual_selection_member=job['selection_member'],actual_selection_member_sha256=sha(original_selection),all_raw_members=members(a),
  source_command_and_environment=dict(source_sha256=job['source_sha256'],command=COMMAND,command_file_sha256=COMMAND_SHA,prepare_sha256=b['source_git_files']['prepare.py'],rehearsal_sha256=b['source_git_files']['research/scratch/submission_rehearsal.py'],training_environment=safe(b['train_environment']),evaluation_environment=safe(b['eval_environment']),environment_scope='Recorded research envelopes; not a clean minimal scoring-box proof.'),
  measured_cold_and_budget=dict(fifth_boundary=run['observations'][4],raw_cold_seconds=proof['raw_cold_seconds'],charged_through_process_exit_seconds=proof['charged_seconds'],charged_through_process_exit_within_budget=run['charged_through_process_exit_within_budget'],training_returncode=run['returncode'],evaluation_returncode=ev['returncode'],six_cache_creation_records=pre['fresh_cache_absent_then_created_empty'],scope='Six empty private training-cache roots; existing separate evaluation cache is not claimed empty.'),
  checkpoint=dict(records=retained,after_terminal=a['checkpoint_after_terminal'],remote_path=a['root']+'/train/out/final.pt',no_checkpoint_copy=a['no_checkpoint_copy'],currently_retained_on_host=None,backup_exists=None,scope='Stable at collection time only; no present-host or backup claim.'),
  disk_and_io=dict(collected_text_member_bytes=a['raw_bytes'],free_bytes_at_collection=a['free_bytes'],checkpoint_bytes=proof['checkpoint_bytes'],cumulative_checkpoint_bytes_written=None,cumulative_log_bytes_written=None,cumulative_cache_bytes_written=None,total_bytes_written=None,no_large_leftovers=None,scope='Collected text bytes/checkpoint length/free-space snapshot are NOT cumulative writes; current inventory UNKNOWN.'),
  scope='Authenticated research FULL metadata only, not independent source or submission acceptance.',launch_authorized=False,submission_authorized=False)

def source_cpu():
 proxy=types.SimpleNamespace(REPO=R,SOURCE=SOURCE,BASE=PARENT,CPU_WORKER='d378cfe9333fe89969f5345fdf0b55afa099a93df43120ae23355171670293e8',SCIENTIFIC_SPEC='f3cd4a4a5aadbbe82fbf349230555f10147f39ad452cc423880ab93a7b352f41',checked=checked,sha=sha,load=load)
 binder=load(R/'research/v2/operator/20260930-r113-laneb-wrapped-full-trn21-preparation/evidence_r121.py','audit_existing_source_cpu','e1865c4b8eae480694dbf9c4bf5fca0935bb0a607351bc2645fcb58960e010c8')
 author=binder.cpu({},proxy);review,raw=binder.review(dict(actual_independent_review=dict(path=str(REVIEW.relative_to(R)),sha256=REVIEW_SHA)),proxy,author)
 assert sha(raw)==REVIEW_SHA
 authorraw=raw_archive(R/AUTHOR_PATH,AUTHOR_SHA);ownraw=raw_archive(R/OWN_PATH,OWN_SHA)
 native=load(R/'research/v2/operator/20260930-r113-laneb-wrapped-model-native60-trn21-preparation/evidence.py','audit_existing_native_pair','689c74ea4a6a298bf2b36c26bda7a3b26464cdccaac94977458bcf206fbb820d').native_pair(proxy)
 return dict(existing_author=author,existing_independent_review=review,review_reference=ref(REVIEW,REVIEW_SHA),author_archive=ref(R/AUTHOR_PATH,AUTHOR_SHA),own_archive=ref(R/OWN_PATH,OWN_SHA),author_raw_members=members(authorraw),own_raw_members=members(ownraw),existing_standalone_native_pair=native,native_pair_archive=ref(R/native['collection_path'],native['collection_sha256']),source_cpu_scope='Existing 10-group author / 8-group independent eager-surrogate evidence. Not rerun. Native pair is isolated B32/H8/T1024/D128 benchmarking, not end-to-end model/eval numerical identity.',fresh_package_independent_review=None,claude_package_audit=None,submission_authorized=False)

def inventory():
 raw=checked(SOURCE_PATH,SOURCE).decode();tree=ast.parse(raw)
 a3=R/'research/v2/proposals/20260929-r69-a3-noproj-id/train.py';a3raw=checked(a3,'c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4').decode()
 def blocks(text):
  return [ast.get_source_segment(text,n)for n in ast.parse(text).body if isinstance(n,ast.If)and ast.unparse(n.test)=="__name__ == '__main__'" and any('NEURON_CC_FLAGS'in ast.unparse(c)for c in n.body)]
 original,current=blocks(a3raw),blocks(raw);assert len(original)==len(current)==1 and original==current,'A3_COMPILER_MAIN_BLOCK_CHANGED'
 imports=[dict(line=n.lineno,statement=ast.unparse(n))for n in ast.walk(tree)if isinstance(n,(ast.Import,ast.ImportFrom))]
 kernel_names={'_fused_mlp_kernel','_FusedMLP','_LaneBWrappedAttention','_lane_b_native_attention'}
 kernels=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,has_docstring=ast.get_docstring(n)is not None)for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.ClassDef))and(n.name in kernel_names or any('nki'in ast.unparse(d)for d in getattr(n,'decorator_list',[])))]
 calls=lambda predicate:[dict(line=n.lineno,expression=ast.unparse(n))for n in ast.walk(tree)if isinstance(n,ast.Call)and predicate(n)]
 return dict(source_reference=ref(SOURCE_PATH,SOURCE),line_count=len(raw.splitlines()),imports=sorted(imports,key=lambda v:v['line']),kernel_docstrings=sorted(kernels,key=lambda v:v['line']),
  environment_calls=calls(lambda n:'environ'in ast.unparse(n.func)or'getenv'in ast.unparse(n.func)),clock_calls=calls(lambda n:ast.unparse(n.func).startswith('time.')),
  explicit_io_calls=calls(lambda n:ast.unparse(n.func)in('open','torch.load','torch.save','make_dataloader','ensure_tokenizer')),
  training_mode_attribute_uses=[dict(line=n.lineno,expression=ast.unparse(n))for n in ast.walk(tree)if isinstance(n,ast.Attribute)and n.attr=='training'],
  targets_none_return_lines=[n.lineno for n in ast.walk(tree)if isinstance(n,ast.If)and ast.unparse(n.test)=='targets is None'],
  compiler_main_block=dict(byte_identical_to_A3=True,source_lines=[36,37],statement_sha256=sha(current[0].encode()),a3_reference=ref(a3,'c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4'),scope='Byte-preservation fact only; no claim that main/import compiler settings are identical.'),
  scope='Static AST inventory only, not dynamic compliance, dependency availability, semantic equivalence or dead-code proof.',source_imported_or_executed=False)

def evaluate():
 checked(SOURCE_PATH,SOURCE);checked(PARENT_PATH,PARENT)
 first,repeat=receipt('first21'),receipt('repeat22');parent=receipt('direct_parent21')
 mean=(first['verifier_proof']['val_bpb']+repeat['verifier_proof']['val_bpb'])/2
 assert mean==.9661123698569338 and first['raw_archive']['sha256']!=repeat['raw_archive']['sha256']
 return dict(schema='r136-laneb-exact-source-two-receipt-audit-evidence-v1',status='AUTHENTICATED_RESEARCH_RECEIPTS_ONLY_PACKAGE_AUDIT_PENDING',source_sha256=SOURCE,command=COMMAND,receipts=[first,repeat],mean_public20m=mean,seed_scope='Both seed42; two hosts, not independent seeds or official leaderboard scores.',direct_research_proven_parent=parent,last_claude_audited_ancestor=None,legacy_schema_diagnostic=legacy_diagnostic(),source_cpu=source_cpu(),static_inventory=inventory(),unknowns=['pristine_scoring_box_dependencies','minimal_launch_environment_sufficiency','end_to_end_native_train_eval_numerical_identity','cumulative_checkpoint_log_cache_bytes_written','current_large_leftover_inventory','present_checkpoint_retention_and_backup','fresh_independent_package_review','Claude_package_audit'],launch_authorized=False,adoption_authorized=False,export_authorized=False,submission_authorized=False,host_calls=0,model_or_torch_execution=False)
