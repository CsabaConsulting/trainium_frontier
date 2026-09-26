"""Exact scored-DISCARD checkpoint cleanup under Final Sprint; no cache removal."""
import ast, datetime, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
auth=REPO/'workflow/FINAL_SPRINT_2026-09-26.txt'
auth_raw=auth.read_bytes()
assert b'discarded, already-scored runs' in auth_raw
host=sys.argv[1]
assert host in ('trn21','trn22')
base='/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/'
literal={
 'trn21':[
  (base+'fs-20260926-arch3e79-primed-trn21-r1-full-packet/train/out/final.pt',453048594,'b29626439efda4ccf774ed99f18dd6c2baa9dfda1c4e08645a29b4bcf193b3a3','fs-20260926-arch3e79-primed-trn21-r1-full'),
  (base+'fs-20260926-arch75b9-recursive-trn21-r1-full-packet/train/out/final.pt',453048658,'689c2fbd64edb23852bb384c64e05f372c9787e05db973efd03a354de67f869b','fs-20260926-arch75b9-recursive-trn21-r1-full')],
 'trn22':[
  ('/home/ubuntu/trainium_l2_unet8x1024_93ae_20260925_trn22_r1/native/fs-20260925-l2-unet8x1024-93ae-trn22-full-r1/out/final.pt',453048594,'fb41241f812b0a357a36c18e0cbfbf81adc1df26ababf86d90fa40f51ded7c6b','fs-20260925-l2-unet8x1024-93ae-trn22-full-r1')]
}[host]
rows={r['label']:r for r in (json.loads(line) for line in (REPO/'research/v2/ledger.jsonl').read_text().splitlines())}
targets=[]
for path,size,digest,label in literal:
 r=rows[label]
 assert r['host']==host and r['status']=='evaluated' and r['public_20m'] is not None and r['checkpoint_sha256']==digest
 assert r.get('verdict')=='DISCARD' or r.get('selector_verdict')=='DISCARD'
 targets.append({'path':path,'expected_bytes':size,'expected_sha256':digest,'label':label,'score':r['public_20m'],'verdict':'DISCARD'})
prior=HERE.parent/'20260926-trn21-three-checkpoint-cleanup/cleanup_once.py'
tree=ast.parse(prior.read_text())
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
# Retain the previously used nofollow/descriptor-bound unlink logic; only target
# count/host and first fresh identity binding differ from the earlier 3-file job.
body=body.replace("'host':'trn21'", "'host':P['host']")
body=body.replace("socket.gethostname()!='ip-172-31-34-85'", "socket.gethostname()!=P['hostname']")
body=body.replace('(66305,596823)',"(66305,P['inode'])")
old="if not stat.S_ISREG(before.st_mode) or identity(before)!=item['expected_identity']:raise RuntimeError('ARCHIVED_IDENTITY_MISMATCH:'+item['path'])"
new="if not stat.S_ISREG(before.st_mode) or before.st_uid!=1000 or before.st_dev!=66305 or before.st_nlink!=1 or before.st_size!=item['expected_bytes']:raise RuntimeError('ARCHIVED_IDENTITY_MISMATCH:'+item['path'])\n   item['expected_identity']=identity(before)"
assert body.count(old)==1
body=body.replace(old,new).replace("if len(opened)!=3:","if len(opened)!=len(P['targets']):")
body=body.replace('NOT_ALL_THREE_VERIFIED','NOT_ALL_TARGETS_VERIFIED').replace('all_three_verified_before_any_unlink','all_targets_verified_before_any_unlink').replace('ALL_THREE_VERIFIED','ALL_TARGETS_VERIFIED').replace('THREE_APPROVED_FILES_REMOVED','EXACT_DISCARDED_SCORED_FILES_REMOVED')
ast.parse(body)
payload={'host':host,'hostname':{'trn21':'ip-172-31-34-85','trn22':'ip-172-31-32-253'}[host],'inode':{'trn21':596823,'trn22':832589}[host],'authorization_sha256':hashlib.sha256(auth_raw).hexdigest(),'targets':targets}
wire=('P='+repr(payload)+'\n'+body).encode()
spec=importlib.util.spec_from_file_location('strict_connector',REPO/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not m.SENSITIVE.search(wire)
if sys.argv[2:]==['--local-check']:
 print(json.dumps({'status':'LOCAL_CHECK_PASS','host':host,'targets':targets,'payload_sha256':hashlib.sha256(wire).hexdigest()}));raise SystemExit(0)
assert sys.argv[2:]==['--execute']
attempt=HERE/(host+'-checkpoint-cleanup-attempt.json');out=HERE/(host+'-checkpoint-cleanup.json')
assert not attempt.exists() and not out.exists()
state=m.local_state();assert not state['local_stops_present'] and not any(v is True for v in state['breaker_tripped'].values())
with attempt.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'controller':state,'targets':targets},f,indent=2);f.write('\n')
argv,*_=m.load_transport(host);argv[-1]='exec /usr/bin/python3 -I -B -S -'
p=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
assert not m.SENSITIVE.search(p.stdout+p.stderr)
events=[json.loads(line) for line in p.stdout.splitlines() if line.strip()]
with out.open('x') as f:json.dump({'returncode':p.returncode,'stderr':p.stderr.decode(),'payload_sha256':hashlib.sha256(wire).hexdigest(),'events':events},f,indent=2);f.write('\n')
final=events[-1]['result'] if events and events[-1]['event']=='FINAL' else {}
print(json.dumps({'host':host,'status':final.get('status'),'free_before':final.get('free_before'),'free_after':final.get('free_after'),'operations':final.get('operations'),'error':final.get('error'),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'receipt':str(out)}))
raise SystemExit(p.returncode)
