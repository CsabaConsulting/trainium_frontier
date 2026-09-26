"""Local literal new-source native201 rebind; no scientific execution/host call."""
import ast,base64,hashlib,json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
def sha(x):return hashlib.sha256(x).hexdigest()
def check(ok,why):
 if not ok:raise RuntimeError(why)
raw=(HERE/'trn22-on-parentcontrol-prepared.json').read_bytes()
check(sha(raw)=='d3ab702d0839043ad6267d01a27a0aeaa02f81236a451e5155720f0255c5614e','PARENT_PACKET')
old=json.loads(raw);ob=old['binding'];files={n:base64.b64decode(v['b64'],validate=True) for n,v in old['files'].items()}
for n,v in files.items():check(sha(v)==old['files'][n]['sha256'],'FILE_PIN:'+n)
pp='research/v2/proposals/20260926-k2-runtime-groups-native201/EXPERIMENT_PROPOSAL.json'
pr=(REPO/pp).read_bytes();check(sha(pr)=='e2c199370c1a0ed8e9a9b9d840c4b2d810bcd5522db38adcd3ef3641c5534947','PROPOSAL')
p=json.loads(pr);s=(REPO/p['candidate']['source']).read_bytes();ss=p['candidate']['source_sha256']
check(sha(s)==ss=='f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e','SOURCE')
review=(REPO/'research/v2/verification/20260926-k2-runtime-groups/REVIEW.md').read_bytes()
check(sha(review)=='58a3343e253f72533ba43304741605bcacff5a45e2e6a4ffedbc583221bfb028','REVIEW')
label=next(x['label'] for x in p['host_alternatives'] if x['host']=='trn22')
binding=json.loads(json.dumps(ob).replace(ob['label'],label).replace(ob['source_sha256'],ss))
oldcommand=files['package/launch-command.txt'];command=(p['candidate']['launch_command']+'\n').encode()
check(oldcommand.replace(b'--num-steps 45',b'--num-steps 201')==command,'ONLY_COUNT_ARGV_CHANGE')
check(sha(command.rstrip(b'\n'))==p['candidate']['launch_command_sha256'],'COMMAND_PIN')
provold=files['package/provenance.json'];prov=json.loads(provold.decode().replace(ob['label'],label).replace(ob['source_sha256'],ss))
prov.update(launch_command=p['candidate']['launch_command'],run_purpose='cold-native201-net-work-and-loss-diagnostic-only',
 scientific_causal_change='Exact reviewed f804 runtime-group K2 source, no Operator source edit',
 native_gate='One native201 is the native test; no additional smoke/OFF/eval',
 continuation='Actual net-token ratio>=1.03, not speed-only; separate full/table selection required')
provnew=(json.dumps(prov,indent=2)+'\n').encode()
repl=[(ob['label'],label),(ob['source_sha256'],ss),(sha(oldcommand),sha(command)),(sha(provold),sha(provnew))]
driver=files['host_driver.py'].decode()
for a,b in repl:check(a in driver,'DRIVER_LITERAL');driver=driver.replace(a,b)
reverse=driver
for a,b in reversed(repl):reverse=reverse.replace(b,a)
check(reverse==files['host_driver.py'].decode(),'DRIVER_REVERSE_IDENTITY');ast.parse(driver)
binding.update(schema='disabled-runtimegroups-native201-v1',source_sha256=ss,steps=201,launch_command=p['candidate']['launch_command'],
 driver_sha256=sha(driver.encode()),independent_review_sha256=sha(review),operator_assignment_sha256=None,
 completed_proposal_path=pp,completed_proposal_sha256=sha(pr),preparation_status='DISABLED_PENDING_SUPERVISOR_RELEASE_AND_CURRENT_EVAL_POSTFLIGHT',
 timing=p['timing'],accounting=p['accounting'],run_purpose='compile_size_and_net_work_diagnostic_only',
 native_short_timing='One cold native201, unchanged t9-to-t44 primary window; no extra OFF/eval/full',
 native_gate_conditions=['Completed exact f804 proposal and Supervisor source/host selection; no new science review/smoke.',
 'Current trn22 batch+attached20M completes/retrieved; all observed descendants/compiler/device/original832589 lock quiescent.',
 'Fresh original lock, ps/neuron-ls/fuser/STOP/runtime/source/argv/resources and six new absent-to-empty cache roots.',
 'One native201 within unchanged1800/900/2730+60 accounting, no eval/full/retry/automatic successor.',
 'Cold fifth boundary, fixed rate, losses and measured net tokens required; speed alone grants no continuation.'])
binding['comparison_control']={**p['control']['primary'],'completed_bundle_sha256':'f17ea4ebbbef720cee5d02f810fcb103fc887a5d56fa7c906910f28346dae6c7',
 'actual_terminal_and_timing_pending':False,'same_host_rate_precommitted':True,'startup_reference':p['control']['startup_reference']}
binding['comparison_control']['label']=p['control']['primary']['run_label']
check(binding['root']==next(x['root'] for x in p['host_alternatives'] if x['host']=='trn22'),'ROOT')
check(binding['lock_identity']==[66305,832589] and binding['no_eval'] is True,'HOST_SCOPE')
check(len(set(binding['fresh_cache_directories']))==6,'SIX_CACHES')
check(all(x.startswith(binding['root']+'/cold-training-caches/') for x in binding['fresh_cache_directories']),'CACHE_OWNER')
check(all(binding[k] is False for k in ('stage_authorized','launch_authorized','host_reserved','release_file_created','source_adoption','submission_authorized')),'DISABLED')
files['HISTORICAL_MECHANICS_PROPOSAL.json']=files['EXPERIMENT_PROPOSAL.json']
files.update({'package/train.py':s,'package/launch-command.txt':command,'package/provenance.json':provnew,
 'host_driver.py':driver.encode(),'EXPERIMENT_PROPOSAL.json':pr,'EXACT_SOURCE_REVIEW.md':review})
files['NATIVE_BINDING.json']=(json.dumps(binding,indent=2)+'\n').encode()
for n in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py'):check(sha(files[n])==old['files'][n]['sha256'],'SUPPORT_UNCHANGED')
packet={'binding':binding,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},
 'local_only':True,'disabled':True,'host_calls':0,'native_started':False,
 'mechanical_derivation':{'parent_packet_sha256':sha(raw),'driver_literal_reverse_identity':True,
 'source_modified':False,'driver_logic_changed':False,'changes':['new source/label/root/cache namespace','45-to201 stop pair only','source/provenance/command hashes','completed proposal/net-work/control metadata']}}
out=HERE/'trn22-runtimegroups-native201-prepared.disabled.json'
with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
print(json.dumps({'status':'DISABLED_LOCAL_ONLY','label':label,'packet_sha256':sha(out.read_bytes()),'driver_sha256':sha(driver.encode()),'source_sha256':ss,'proposal_sha256':sha(pr),'command_file_sha256':sha(command),'host_calls':0}))
