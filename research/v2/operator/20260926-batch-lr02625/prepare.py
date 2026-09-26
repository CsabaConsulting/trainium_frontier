"""LOCAL ONLY: exact executed8c6 full packet with one LR value/root/label rebind."""
import ast,base64,hashlib,json,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
def sha(x):return hashlib.sha256(x).hexdigest()
assert len(sys.argv)==3
proposal_path,proposal_sha=sys.argv[1:]
pr=(REPO/proposal_path).read_bytes();assert sha(pr)==proposal_sha
p=json.loads(pr)
raw=(HERE.parent/'20260926-final-sprint-successors/trn22-batch-lr14-bound-packet.json').read_bytes()
assert sha(raw)=='a0672a32dba0570e0ef0d434356ede50f775e774fd6bdd1cde0daceeb06053ff'
parent=json.loads(raw);old=parent['binding']
files={n:base64.b64decode(v['b64'],validate=True) for n,v in parent['files'].items()}
for n,v in files.items():assert sha(v)==parent['files'][n]['sha256'],n
source=(REPO/p['candidate']['source']).read_bytes()
assert source==files['package/train.py'] and sha(source)==p['candidate']['source_sha256']==old['source_sha256']
old_command=files['package/launch-command.txt'].decode().strip();command=p['candidate']['launch_command']
assert old_command.count('--matrix-lr 0.021')==1
assert command==old_command.replace('--matrix-lr 0.021','--matrix-lr 0.02625')
assert sha(command.encode())==p['candidate']['launch_string_sha256']
command_raw=(command+'\n').encode();label=p['dispatch']['selected_label']
assert sha(command_raw)==p['candidate']['launch_command_sha256']
assert label=='fs-20260926-batch262144-lr02625-trn22-r1' and p['dispatch']['selected_host']=='trn22'
root=str(Path(old['root']).parent/(label+'-packet'))
prov={'label':label,'train_sha256':p['candidate']['source_sha256'],'base_sha256':p['prospective_record']['base_sha256'],
 'kind':p['kind'],'run_purpose':'quality','launch_command':command,'same_host_comparator':p['control'],
 'cache_owner_label':old['cache_owner_label'],'cache_mode':'EXISTING_RESEARCH_ROOTS_NO_HIT_OR_COLD_CLAIM','submission_authorized':False}
prov_raw=(json.dumps(prov,indent=2)+'\n').encode()
repl=[(old['root'],root),(old['label'],label),(sha(files['package/launch-command.txt']),sha(command_raw)),(sha(files['package/provenance.json']),sha(prov_raw))]
driver=files['host_driver.py'].decode()
for a,b in repl:assert a in driver,a;driver=driver.replace(a,b)
reverse=driver
for a,b in reversed(repl):reverse=reverse.replace(b,a)
assert reverse==files['host_driver.py'].decode();ast.parse(driver)
def transform(o):
 if isinstance(o,str):
  for a,b in repl:o=o.replace(a,b)
  return o
 if isinstance(o,list):return [transform(v) for v in o]
 if isinstance(o,dict):return {k:transform(v) for k,v in o.items()}
 return o
b=transform(old)
b.update(schema='prepared-existing8c6-lr02625-full-v1',kind=p['kind'],base_sha256=p['prospective_record']['base_sha256'],
 run_purpose='quality',launch_command=command,scientific_patch='FLAG_ONLY_MATRIX_LR_02625',
 stage_authorized=False,launch_authorized=False,operator_assignment_sha256=None,
 completed_proposal_path=proposal_path,completed_proposal_sha256=sha(pr),
 driver_sha256=sha(driver.encode()),source_adoption=False,submission_authorized=False,no_automatic_next_job=True)
assert not b['fresh_cache_directories']
for phase in ('train_environment','eval_environment'):
 for k,v in old[phase].items():
  if 'CACHE' in k or k=='TORCH_NEURONX_DEBUG_DIR':assert b[phase][k]==v,(phase,k)
files={n:v for n,v in files.items() if n not in ('NATIVE_BINDING.json','RELEASE.json')}
files.update({'host_driver.py':driver.encode(),'package/train.py':source,'package/launch-command.txt':command_raw,
 'package/provenance.json':prov_raw,'EXPERIMENT_PROPOSAL.json':pr})
files['NATIVE_BINDING.disabled.json']=(json.dumps(b,indent=2)+'\n').encode()
for n in ('lifetime_guard.py','idle_guard.py','dispatch_driver.py'):assert sha(files[n])==parent['files'][n]['sha256']
packet={'binding':b,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},
 'local_only':True,'host_calls':0,'mechanical_derivation':{'executed_parent_packet_sha256':sha(raw),
 'driver_literal_reverse_exact':True,'source_identical':True,'only_scientific_argv_delta':'--matrix-lr0.021->0.02625',
 'all_train_eval_cache_paths_unchanged':True,'guardian_prepare_harness_unchanged':True}}
out=HERE/'trn22-prepared.disabled.json'
with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
receipt={'status':'LOCAL_MECHANICAL_PASS_NOT_LAUNCHED','packet_sha256':sha(out.read_bytes()),
 'driver_sha256':sha(driver.encode()),'proposal_path':proposal_path,'proposal_sha256':sha(pr),
 'command_file_sha256':sha(command_raw),'command_string_sha256':sha(command.encode()),'source_sha256':sha(source),'host_calls':0}
with (HERE/'PREPARED.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
