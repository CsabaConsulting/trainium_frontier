"""Exact existing staging route; disabled until the one Supervisor release."""
import ast,base64,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=Path('/home/poojith-devan/Desktop/Trainium')
ASSIGNMENT_PATH='research/v2/assignments/20260926-k2-runtimegroups-native201-trn22.md'
ASSIGNMENT_SHA256='25bdacab22b4d4223900f060666df9ba0cfa1608d3ad606f4fad396022452648'
CHOICE='trn22-runtimegroups-native201'
PACKET='20260926-k2-sprint-timing/trn22-runtimegroups-native201-prepared.disabled.json'
PIN='ddb614e5853e7242abe2757ee6271ae33d6492557fbeb5f69d2f1555fa7bcae3'
def sha(x):return hashlib.sha256(x).hexdigest()
assert sys.argv[1:2]==[CHOICE] and sys.argv[2:] in (['--local-check'],['--execute'])
raw=(HERE/'stage_next.py').read_bytes();assert sha(raw)=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
pr=(HERE.parent/PACKET).read_bytes();assert sha(pr)==PIN
packet=json.loads(pr)
for n,v in packet['files'].items():assert sha(base64.b64decode(v['b64'],validate=True))==v['sha256'],n
if not all((ASSIGNMENT_PATH,ASSIGNMENT_SHA256)):
 if sys.argv[2:]==['--local-check']:
  print(json.dumps({'status':'DISABLED_LOCAL_MECHANICAL_CHECK_PASS','packet_sha256':PIN,'host_calls':0,'release_pending':True}));raise SystemExit
 raise SystemExit('DISABLED_PENDING_EXACT_ONE_USE_SUPERVISOR_RELEASE')
text=raw.decode()
def replace_once(a,b):
 global text
 assert text.count(a)==1,a
 text=text.replace(a,b)
replace_once('choices={','choices={\n '+repr(CHOICE)+':('+repr(PACKET)+','+repr(PIN)+'),')
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth", "assignment=REPO/"+repr(ASSIGNMENT_PATH)+";auth=assignment.read_bytes();assert sha(auth)=="+repr(ASSIGNMENT_SHA256)+";assert b'fs-20260926-k2-runtimegroups-native201-trn22-r1' in auth and b'f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e' in auth")
replace_once("if choice in ('trn22-off','trn22-on'):","if choice in ('trn22-off','trn22-on',"+repr(CHOICE)+"):")
replace_once("review=(REPO/'research/v2/verification/20260926-k2-sprint-independent/REVIEW.md').read_bytes();assert sha(review)=='2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'","review=(REPO/'research/v2/verification/20260926-k2-runtime-groups/REVIEW.md').read_bytes();assert sha(review)=='58a3343e253f72533ba43304741605bcacff5a45e2e6a4ffedbc583221bfb028'")
insert="""proposal_path='research/v2/proposals/20260926-k2-runtime-groups-native201/EXPERIMENT_PROPOSAL.json'
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)=='e2c199370c1a0ed8e9a9b9d840c4b2d810bcd5522db38adcd3ef3641c5534947'
proposal=json.loads(proposal_raw)
assert b['source_sha256']==proposal['candidate']['source_sha256']==sha((REPO/proposal['candidate']['source']).read_bytes())
assert b['launch_command']==proposal['candidate']['launch_command'] and b['steps']==201 and b['no_eval'] is True
assert {'host':b['host'],'label':b['label'],'root':b['root']} in proposal['host_alternatives']
files['EXPERIMENT_PROPOSAL.json']=proposal_raw
files['SUPERVISOR_ASSIGNMENT.md']=auth
files['EXACT_SOURCE_REVIEW.md']=review
files['SOURCE_COMPOSITION.md']=(REPO/'research/v2/verification/20260926-k2-runtime-groups/COMPOSITION.md').read_bytes()
assert sha(files['SOURCE_COMPOSITION.md'])=='aac28c5d267a38650b63f5dc12418b6d5aa774a50cbc5f46d6ec0525d6e94460'
b['completed_proposal_path']=proposal_path;b['completed_proposal_sha256']=sha(proposal_raw)
"""
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()",insert+"files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')","predecessor=HERE/'trn22-batch-lr14-completed/results-bundle.json'")
gate="""assert terminal['label']=='fs-20260926-sprint-clean-batch-lr14-trn22-r1'
parent=HERE/'trn22-e1e2-control-completed/results-bundle.json'
acceptance=json.loads((HERE/'trn22-parent-control-timing-acceptance.json').read_bytes())
assert sha(parent.read_bytes())==acceptance['completed_bundle_sha256']=='f17ea4ebbbef720cee5d02f810fcb103fc887a5d56fa7c906910f28346dae6c7'
assert acceptance['control_label']==b['comparison_control']['label']=='fs-20260926-sprint-e1e2-control-trn22-r1'
assert acceptance['control_source_sha256']=='e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
# Original source639 acceptance stays historical, not edited or reused as f804 source approval.
# Exact f804 source review/COMPOSITION/proposal/release above supplies the new source binding.
for condition in ('same_host_runtime_science','first45_sequential_finite','t9_t44_trustworthy','window_no_recompile_or_stall','parent_train_eval_complete_and_budget_pass','explicit_postflight_quiescent'):
 assert acceptance['conditions'][condition] is True,'PARENT_COMPARISON_REFUSED:'+condition
assert acceptance['control_tokens_per_second']==b['comparison_control']['primary_tokens_per_second']==35*131072/(acceptance['t44']-acceptance['t9'])
assert p['lock_device']==66305 and p['lock_inode']==832589
"""
replace_once("attempt=HERE/(choice+'-attempt.json');",gate+"attempt=HERE/(choice+'-attempt.json');")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
