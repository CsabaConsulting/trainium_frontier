"""Existing full transport; no remote execution before exact result/release pins."""
import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
ASSIGNMENT_PATH=None
ASSIGNMENT_SHA256=None
NATIVE_QUALIFICATION_PATH=None
NATIVE_QUALIFICATION_SHA256=None
CHOICE='trn21-localconstants-k2-full-control'
PACKET='20260926-localconstants-k2-full-control/trn21-prepared.disabled.json'
PACKET_SHA='527a50707a716953912c5c503b07d315e556531f2c51a9851fe7f7756589d0ca'
PROPOSAL_PATH='research/v2/proposals/20260926-localconstants-k2-qualified-base-control/EXPERIMENT_PROPOSAL.json'
PROPOSAL_SHA='56521727f3b43aa20cfe28d02bdeec4f5b7ab3eb4be7bdd3000ecd13305ab405'
def sha(x):return hashlib.sha256(x).hexdigest()
def check(ok,why):
 if not ok:raise RuntimeError(why)
check(sys.argv[1:2]==[CHOICE] and sys.argv[2:] in (['--local-check'],['--execute']),'ARGS')
raw=(HERE/'stage_next.py').read_bytes()
check(sha(raw)=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318','PROVEN_TRANSPORT')
packet_raw=(HERE.parent/PACKET).read_bytes();check(sha(packet_raw)==PACKET_SHA,'PACKET_PIN')
p=json.loads(packet_raw)
for n,v in p['files'].items():check(sha(base64.b64decode(v['b64'],validate=True))==v['sha256'],'FILE_PIN:'+n)
check(sha((REPO/PROPOSAL_PATH).read_bytes())==PROPOSAL_SHA,'PROPOSAL_PIN')
check(p['binding']['stage_authorized'] is False and p['binding']['launch_authorized'] is False,'DISABLED_PACKET')
check(p['binding']['base_sha256']==p['binding']['source_sha256']=='662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00','SOURCE_PIN')
if not all((ASSIGNMENT_PATH,ASSIGNMENT_SHA256,NATIVE_QUALIFICATION_PATH,NATIVE_QUALIFICATION_SHA256)):
 if sys.argv[2:]==['--local-check']:
  ast.parse(raw)
  print(json.dumps({'status':'DISABLED_LOCAL_MECHANICAL_CHECK_PASS','packet_sha256':PACKET_SHA,'host_calls':0,
    'pending':['actual_native201_qualification_and_postflight','exact_Supervisor_release_pins','fresh_guarded_preflight']}))
  raise SystemExit
 raise SystemExit('DISABLED_PENDING_ACTUAL_NATIVE_QUALIFICATION_AND_EXACT_FULL_RELEASE')
check(sha((REPO/NATIVE_QUALIFICATION_PATH).read_bytes())==NATIVE_QUALIFICATION_SHA256,'SUPERVISOR_ACTUAL_QUALIFICATION_PIN')
text=raw.decode()
def replace_once(a,b):
 global text
 check(text.count(a)==1,'TRANSPORT_LITERAL:'+a)
 text=text.replace(a,b)
replace_once('choices={',"choices={\n "+repr(CHOICE)+':('+repr(PACKET)+','+repr(PACKET_SHA)+'),')
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth",
 "assignment=REPO/"+repr(ASSIGNMENT_PATH)+";auth=assignment.read_bytes();assert sha(auth)=="+repr(ASSIGNMENT_SHA256)+";assert b'fs-20260926-localconstants-k2-control-trn21-r1' in auth and b'662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00' in auth")
replace_once("if choice in ('trn22-off','trn22-on'):","if choice in ('trn22-off','trn22-on',"+repr(CHOICE)+"):")
replace_once("review=(REPO/'research/v2/verification/20260926-k2-sprint-independent/REVIEW.md').read_bytes();assert sha(review)=='2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'",
 "review=(REPO/'research/v2/verification/20260926-k2-compiler-local-constants/REVIEW.md').read_bytes();assert sha(review)=='2bbbd51f3a70a0fb1be5883a2a1413ea1b59a92f5643fac301fb365811124bf5'")
insert="""proposal_raw=(REPO/PROPOSAL_LITERAL).read_bytes()
assert sha(proposal_raw)==PROPOSAL_PIN
proposal=json.loads(proposal_raw)
assert b['kind']=='control' and b['base_sha256']==proposal['candidate']['source_sha256']==b['source_sha256']
assert b['launch_command']==proposal['candidate']['launch_command']
assert sha((REPO/proposal['candidate']['source']).read_bytes())==b['source_sha256']
files['EXPERIMENT_PROPOSAL.json']=proposal_raw
files['SUPERVISOR_ASSIGNMENT.md']=auth
files['EXACT_SOURCE_REVIEW.md']=review
files['NATIVE_QUALIFICATION.json']=(REPO/QUALIFICATION_LITERAL).read_bytes()
b['completed_proposal_path']=PROPOSAL_LITERAL;b['completed_proposal_sha256']=sha(proposal_raw)
b['native_qualification_sha256']=QUALIFICATION_PIN
""".replace('PROPOSAL_LITERAL',repr(PROPOSAL_PATH)).replace('PROPOSAL_PIN',repr(PROPOSAL_SHA)).replace('QUALIFICATION_LITERAL',repr(NATIVE_QUALIFICATION_PATH)).replace('QUALIFICATION_PIN',repr(NATIVE_QUALIFICATION_SHA256))
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()",insert+"files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')","predecessor=HERE/'trn21-localconstants-native201-completed/results-bundle.json'")
replace_once("if choice=='trn22-on':assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0", "if choice in ('trn22-on',"+repr(CHOICE)+"):assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
