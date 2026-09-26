"""LOCAL ONLY: literal rebind of the proven full guardian/driver, never launch."""
import ast
import base64
import copy
import hashlib
import json
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
def sha(x): return hashlib.sha256(x).hexdigest()
def check(ok, why):
    if not ok: raise RuntimeError(why)
def unpack(raw):
    packet = json.loads(raw)
    files = {n: base64.b64decode(v['b64'], validate=True) for n,v in packet['files'].items()}
    for n,v in files.items(): check(sha(v)==packet['files'][n]['sha256'], 'FILE_PIN:'+n)
    return packet, files

proposal_path = 'research/v2/proposals/20260926-localconstants-k2-qualified-base-control/EXPERIMENT_PROPOSAL.json'
proposal_raw = (REPO/proposal_path).read_bytes()
check(sha(proposal_raw)=='56521727f3b43aa20cfe28d02bdeec4f5b7ab3eb4be7bdd3000ecd13305ab405', 'PROPOSAL_PIN')
proposal = json.loads(proposal_raw)
parent_raw = (HERE.parent/'20260926-k2-qualified-base-control/trn21-prepared.json').read_bytes()
check(sha(parent_raw)=='14d26df4f9cce72406ab5aaba139345b6c11ab3602d5cc640c6de0e13cebe3d1','PARENT_PACKET')
parent, files = unpack(parent_raw)
short_raw = (HERE.parent/'20260926-k2-sprint-timing/trn21-localconstants-native201-prepared.disabled.json').read_bytes()
check(sha(short_raw)=='dd2e7336323efb24f3d21c17968d7ef2db819d8b1c09a43befcda81a84e44343','NATIVE201_PACKET')
short, shortfiles = unpack(short_raw)
old = parent['binding']; sb = short['binding']; binding = copy.deepcopy(old)
source = (REPO/proposal['candidate']['source']).read_bytes()
source_sha = proposal['candidate']['source_sha256']
check(sha(source)==source_sha==sb['source_sha256'],'SOURCE_PIN')
check(source==shortfiles['package/train.py'],'EXACT_RUNNING_SOURCE')
short_argv=shlex.split(sb['launch_command']); full_command=proposal['candidate']['launch_command']
at=short_argv.index('--num-steps')
check(short_argv[at:at+2]==['--num-steps','201'],'CURRENT_COUNT')
check(short_argv[:at]+short_argv[at+2:]==shlex.split(full_command),'ONLY_STOP_PAIR_REMOVED')
command_raw=(full_command+'\n').encode()
check(sha(command_raw)==proposal['candidate']['launch_command_sha256'],'COMMAND_PIN')
check(command_raw==files['package/launch-command.txt'],'FULL_SCIENCE_ARGV_UNCHANGED')
label=next(x['label'] for x in proposal['host_alternatives'] if x['host']=='trn21')
root=str(Path(old['root']).parent/(label+'-packet'))
cache_keys=('NEURON_COMPILE_CACHE_URL','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR')
provenance={'label':label,'train_sha256':source_sha,'base_sha256':source_sha,'kind':'control',
 'arm_id':proposal['arm_id'],'run_purpose':'quality','launch_command':full_command,
 'prospective_control_role_frozen_before_launch':True,'no_duplicate_throughput_arm_record':True,
 'cache_owner_label':sb['label'],'cache_mode':'RETAINED_NATIVE201_SAME_HOST_RESEARCH_HIT_UNPROVEN',
 'submission_authorized':False}
prov_raw=(json.dumps(provenance,indent=2)+'\n').encode()
replacements=[(old['root'],root),(old['label'],label),(old['source_sha256'],source_sha),
 (old['cache_owner_label'],sb['label']),(sha(files['package/provenance.json']),sha(prov_raw))]
driver=files['host_driver.py'].decode()
for a,b in replacements:
 check(a in driver,'EXPECTED_LITERAL:'+a)
 driver=driver.replace(a,b)
restored=driver
for a,b in reversed(replacements): restored=restored.replace(b,a)
check(restored==files['host_driver.py'].decode(),'DRIVER_LITERAL_ONLY_REVERSIBLE')
ast.parse(driver)
def transform(obj):
 if isinstance(obj,str):
  for a,b in replacements: obj=obj.replace(a,b)
  return obj
 if isinstance(obj,list): return [transform(x) for x in obj]
 if isinstance(obj,dict): return {k:transform(v) for k,v in obj.items()}
 return obj
binding=transform(binding)
for k in cache_keys: check(binding['train_environment'][k]==sb['train_environment'][k],'SAME_HOST_ACTUAL_CACHE:'+k)
binding.update(schema='disabled-localconstants-k2-full-control-v1',kind='control',base_sha256=source_sha,
 arm_id=proposal['arm_id'],run_purpose='quality',prospective_record=proposal['prospective_record'],
 scientific_patch='K2_embedding_backward_local_constants',full_proposal_path=proposal_path,
 full_proposal_sha256=sha(proposal_raw),full_proposal_present=True,driver_sha256=sha(driver.encode()),
 stage_authorized=False,launch_authorized=False,release_file_created=False,host_reserved=False,
 independent_review_sha256=proposal['evidence_bar']['existing_single_review_sha256'],
 operator_assignment_sha256=None,source_adoption=False,submission_authorized=False,
 cache_owner_label=sb['label'],cache_mode=provenance['cache_mode'],
 cache_binding_status='ACTUAL_NATIVE201_OWNER_LIVE_QUALIFICATION_AND_EXIT_PENDING',
 current_native_packet_sha256=sha(short_raw),actual_same_host_on_qualification_pending=True,
 actual_k2_speed_pass=False,native_cold_startup_seconds=None,native_cold_startup_charge_seconds=None,
 no_automatic_next_job=True)
binding['conditional_requirements']=[
 'Actual native201 outcome must qualify exact precommitted t9-to-t44 >=1.03 with source/budget/finite/checkpoint evidence and explicit quiescence; no additional native test.',
 'Supervisor selects one full alternative and issues its exact release; this local packet creates no release file or host reservation.',
 'Reuse only this exact662 native201 owner cache on trn21 after exit. Warm directory is not hit proof or cold submission equivalence; preserve actual cold penalty.',
 'Original lifetime596823 lock, fresh ps/neuron-ls/fuser/STOP/hash/unused-root/RAM64GiB and initial13.5GiB/runtime10.5GiB admission remain.',
 'One actual full1800+fresh20M control row with base662, independent of BPB; no duplicate throughput row or automatic adoption/table/submission.']
files.update({'host_driver.py':driver.encode(),'package/train.py':source,
 'package/provenance.json':prov_raw,'EXPERIMENT_PROPOSAL.json':proposal_raw})
files['NATIVE_BINDING.disabled.json']=(json.dumps(binding,indent=2)+'\n').encode()
for n in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py','package/launch-command.txt'):
 check(sha(files[n])==parent['files'][n]['sha256'],'UNCHANGED_SUPPORT:'+n)
packet={'binding':binding,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},
 'local_only':True,'host_calls':0,'native_started':False,
 'mechanical_derivation':{'parent_packet_sha256':sha(parent_raw),'current_native201_packet_sha256':sha(short_raw),
 'driver_logic_changed':False,'driver_literal_reversal_exact':True,'source_modified':False,
 'scientific_change_from_native201':'Remove exactly --num-steps 201 only',
 'same_guard_prepare_harness_eval':True,'no_new_verification_algorithm':True},
 'missing_before_dispatch':['actual_native201_qualification_and_explicit_postflight','exact_Supervisor_release','fresh_machine_checks']}
out=HERE/'trn21-prepared.disabled.json'
with out.open('x') as f: json.dump(packet,f,indent=2);f.write('\n')
receipt={'status':'DISABLED_LOCAL_ONLY','host':'trn21','label':label,'kind':'control','base_sha256':source_sha,
 'packet_sha256':sha(out.read_bytes()),'driver_sha256':sha(driver.encode()),'proposal_sha256':sha(proposal_raw),
 'launch_command_sha256':sha(command_raw),'cache_owner_label':sb['label'],
 'literal_reverse_identity_pass':True,'source_argv_support_pins_pass':True,'host_calls':0,'launch_authorized':False}
with (HERE/'PREPARED.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
