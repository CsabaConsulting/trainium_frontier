"""Local mechanical rebinding of the executed cold P0 full train/eval packet."""
import ast,base64,copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
def sha(x):return hashlib.sha256(x).hexdigest()
def check(v,s):
 if not v:raise RuntimeError(s)
proposal_path='research/v2/proposals/20260926-batch-lr021-recursive-stack/EXPERIMENT_PROPOSAL.json'
pr=(REPO/proposal_path).read_bytes()
check(sha(pr)=='f923c591052d05830bdb1baecf1c602d2939e693ff8c13c6f8a8dafe5e17aa8e','PROPOSAL_PIN')
p=json.loads(pr)
raw=(HERE.parent/'20260926-final-sprint-p0/trn21-bound-packet.json').read_bytes()
check(sha(raw)=='4483ff52073b80a39dca8f52698b405bf290096aab2b6505123b61a7bf1ee500','EXECUTED_PARENT')
parent=json.loads(raw);old=parent['binding']
files={n:base64.b64decode(v['b64'],validate=True) for n,v in parent['files'].items()}
for n,v in files.items():check(sha(v)==parent['files'][n]['sha256'],'PARENT_FILE:'+n)
source=(REPO/p['source']).read_bytes();check(sha(source)==p['source_sha256'],'SOURCE_PIN')
command=p['candidate']['launch_command'];check(sha(command.encode())==p['candidate']['launch_command_sha256'],'COMMAND_PIN')
command_raw=(command+'\n').encode()
label=p['dispatch']['selected_label'];check(p['dispatch']['selected_host']=='trn21','HOST')
root=str(Path(old['root']).parent/(label+'-packet'))
prov={'label':label,'train_sha256':p['source_sha256'],'base_sha256':p['base_sha256'],
 'kind':'schedule_arm','run_purpose':'quality','launch_command':command,
 'same_host_scientific_comparator':p['control'],'cache_mode':'NEW_PRIVATE_COLD_TRAINING_EXISTING_RESEARCH_EVAL',
 'submission_authorized':False}
prov_raw=(json.dumps(prov,indent=2)+'\n').encode()
repl=[(old['root'],root),(old['label'],label),(old['source_sha256'],p['source_sha256']),
 (sha(files['package/launch-command.txt']),sha(command_raw)),(sha(files['package/provenance.json']),sha(prov_raw))]
driver=files['host_driver.py'].decode()
for a,b in repl:check(a in driver,'DRIVER_LITERAL:'+a);driver=driver.replace(a,b)
reverse=driver
for a,b in reversed(repl):reverse=reverse.replace(b,a)
check(reverse==files['host_driver.py'].decode(),'DRIVER_REVERSAL');ast.parse(driver)
def transform(o):
 if isinstance(o,str):
  for a,b in repl:o=o.replace(a,b)
  return o
 if isinstance(o,list):return [transform(v) for v in o]
 if isinstance(o,dict):return {k:transform(v) for k,v in o.items()}
 return o
b=transform(copy.deepcopy(old))
b.update(schema='prepared-batch-lr021-recursive-cold-full-v1',kind='schedule_arm',base_sha256=p['base_sha256'],
 run_purpose='quality',arm_id=p['arm_id'],launch_command=command,
 scientific_patch='batch262144_lr021_recursive_stack',stage_authorized=False,launch_authorized=False,
 operator_assignment_sha256=None,independent_review_sha256=p['code_evidence']['stack_review_sha256'],
 completed_proposal_path=proposal_path,completed_proposal_sha256=sha(pr),
 driver_sha256=sha(driver.encode()),cold_bound_driver_sha256=sha(driver.encode()),
 actual_same_host_comparator=p['control'],source_adoption=False,submission_authorized=False,
 no_automatic_next_job=True)
check(len(b['fresh_cache_directories'])==6 and len(set(b['fresh_cache_directories']))==6,'SIX_CACHES')
check(all(x.startswith(root+'/cold-training-caches/') for x in b['fresh_cache_directories']),'FRESH_ROOTS')
check(b['lock_identity']==[66305,596823] and b['budget']['source_seconds']==1800,'UNCHANGED_LOCK_BUDGET')
check(b['eval_environment']==dict(old['eval_environment'],TRN_EXP_LABEL=label),'UNCHANGED_EVAL_ENV')
files={n:v for n,v in files.items() if n not in ('RELEASE.json','NATIVE_BINDING.json','NATIVE_BINDING.disabled.json')}
files.update({'host_driver.py':driver.encode(),'package/train.py':source,'package/launch-command.txt':command_raw,
 'package/provenance.json':prov_raw,'EXPERIMENT_PROPOSAL.json':pr})
files['NATIVE_BINDING.disabled.json']=(json.dumps(b,indent=2)+'\n').encode()
for n in ('lifetime_guard.py','idle_guard.py','dispatch_driver.py'):check(sha(files[n])==parent['files'][n]['sha256'],'UNCHANGED_SUPPORT')
packet={'binding':b,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},
 'local_only':True,'host_calls':0,'mechanical_derivation':{'executed_parent_packet_sha256':sha(raw),
 'driver_literal_reversal_exact':True,'driver_logic_changed':False,'source_modified':False,'same_guard_prepare_harness_eval':True}}
out=HERE/'trn21-prepared.disabled.json'
with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
receipt={'status':'DISABLED_LOCAL_MECHANICAL_PASS','packet_sha256':sha(out.read_bytes()),
 'driver_sha256':sha(driver.encode()),'proposal_sha256':sha(pr),'source_sha256':sha(source),
 'command_file_sha256':sha(command_raw),'command_string_sha256':sha(command.encode()),
 'parent_packet_sha256':sha(raw),'six_private_caches':b['fresh_cache_directories'],'host_calls':0}
with (HERE/'PREPARED.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
