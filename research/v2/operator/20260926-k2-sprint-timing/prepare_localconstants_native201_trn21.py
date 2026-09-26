"""New201-step binding for the direct human amendment; no host operations."""
import ast, base64, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(value):return hashlib.sha256(value).hexdigest()
def enc(value):return {'sha256':sha(value),'b64':base64.b64encode(value).decode()}
raw=(HERE/'trn21-localconstants-prepared.disabled.json').read_bytes()
assert sha(raw)=='e970d3806cef0b77d12faf0d52aec83609ecf8d30c4dfbfe003e43ddc322fd31'
old=json.loads(raw);old_label=old['binding']['label']
label='fs-20260926-k2-localconstants-native201-trn21-r1'
b=json.loads(json.dumps(old['binding']).replace(old_label,label))
files={name:base64.b64decode(value['b64']) for name,value in old['files'].items()}
for name,value in files.items():assert sha(value)==old['files'][name]['sha256'],name
old_command=files['package/launch-command.txt']
assert old_command.count(b'--num-steps 45')==1
files['package/launch-command.txt']=old_command.replace(b'--num-steps 45',b'--num-steps 201')
b['launch_command']=b['launch_command'].replace('--num-steps 45','--num-steps 201')
old_prov=files['package/provenance.json']
prov=json.loads(old_prov.decode().replace(old_label,label))
prov['launch_command']=b['launch_command']
prov['run_purpose']='201-step-native-whole-window-timing-and-first200-loss-not-quality'
files['package/provenance.json']=(json.dumps(prov,indent=2)+'\n').encode()
prior_driver=files['host_driver.py']
driver=prior_driver.decode().replace(old_label,label).replace(sha(old_command),sha(files['package/launch-command.txt'])).replace(sha(old_prov),sha(files['package/provenance.json']))
assert driver.replace(label,old_label).replace(sha(files['package/launch-command.txt']),sha(old_command)).replace(sha(files['package/provenance.json']),sha(old_prov))==prior_driver.decode()
ast.parse(driver);files['host_driver.py']=driver.encode()
b['driver_sha256']=sha(files['host_driver.py'])
b['steps']=201;b['run_purpose']='whole-step-timing-and-first200-loss-only'
b['native_short_timing']='One human-authorized201-step native job, retaining exact primary t9..t44; no extra OFF or gate run'
b['preparation_status']='DISABLED_PENDING_FINAL201_PROPOSAL_PARENT_TERMINAL_AND_SUPERVISOR_RELEASE'
b['superseded_compile_gate_record']=b.pop('additional_human_compile_gate')
b['superseded_compile_gate_record']['superseded_by']='Latest direct human: accepted base native forward evidence; one Trn2 native timing job is the native gate; no extra native tests or questions'
b['timing']['screen_pass']='201 finite completed steps with valid native/accounting/checkpoint; same-host t9..t44 rate>=1.03, plus report actual available losses0..200 versus base'
b['loss_reporting']='Actual source-emitted first-five and every10-step losses through200; no source/logger change or invented unlogged values'
b['native_gate_conditions']=[
 'Exact662c source and one accepted fresh source review; final human-amended201 proposal and Supervisor release must match.',
 'Use human-approved accepted-base native forward evidence and actual backward compiler/simulator evidence as bound by Supervisor; no additional native gate job.',
 'Current same-host genuine e1e2 control and attached20M must naturally finish, be retrieved and match the prospectively fixed t9..t44 science/runtime window.',
 'All-observed descendant/compiler/device/original-lock quiescence then fresh original596823 lock plus ps/neuron-ls/fuser/STOP/hash/resource checks.',
 'Six fresh private cold caches, actual four-rank environment, unchanged1800-second budget.',
 'Exactly201 steps, no OFF/eval/full/retry; report primary35*131072/(t44-t9), actual logged losses through200, uncapped startup and charged time.',
]
files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\n').encode()
assert sha(files['package/train.py'])==b['source_sha256']=='662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00'
assert b['stage_authorized'] is False and b['launch_authorized'] is False
assert '--num-steps 201' in b['launch_command'] and '--num-steps 45' not in b['launch_command']
for name in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py'):
 assert files[name]==base64.b64decode(old['files'][name]['b64'])
out=HERE/'trn21-localconstants-native201-prepared.disabled.json'
packet={'binding':b,'files':{name:enc(value) for name,value in files.items()},'mechanical_derivation':{'prior_disabled45_packet_sha256':sha(raw),'prior_driver_sha256':sha(prior_driver),'literal_changes':['new label/root/six-cache namespace','science argv num-steps45 to201 only','command/provenance hash pins'],'reverse_literal_driver_derivation_exact':True,'source_guardian_dispatch_budget_unchanged':True,'historical45_proposal_snapshot_not_current_launch_authority':True},'local_only':True,'host_calls':0,'disabled':True,'native_started':False}
with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
print(json.dumps({'packet_sha256':sha(out.read_bytes()),'driver_sha256':b['driver_sha256'],'launch_command_sha256':sha(files['package/launch-command.txt']),'label':label,'disabled':True,'host_calls':0}))
