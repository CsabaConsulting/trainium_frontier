"""Existing ON/parent transport; disabled until exact Supervisor release pins."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
# Populated mechanically only from the forthcoming exact Supervisor release.
ASSIGNMENT_PATH='research/v2/assignments/20260926-k2-native201-trn21.md'
ASSIGNMENT_SHA256='68d319c203e0512dc9bfe02598228f743e960364112f26b17cddaa35328406a8'
PROPOSAL_SHA256='3f0e622b0c1c7c19dbeaacd9674e4a088826b62078fde1eeee6fca482809ca70'
if not all((ASSIGNMENT_PATH,ASSIGNMENT_SHA256,PROPOSAL_SHA256)):
 raise SystemExit('DISABLED_PENDING_EXACT_SUPERVISOR_RELEASE_PINS')
raw=(HERE/'stage_next.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
assert sys.argv[1]=='trn21-localconstants-native201'
assert sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
def replace_once(old,new):
 global text
 assert text.count(old)==1,old
 text=text.replace(old,new)
replace_once('choices={',"choices={\n 'trn21-localconstants-native201':('20260926-k2-sprint-timing/trn21-localconstants-native201-prepared.disabled.json','dd2e7336323efb24f3d21c17968d7ef2db819d8b1c09a43befcda81a84e44343'),")
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth", "assignment=REPO/"+repr(ASSIGNMENT_PATH)+";auth=assignment.read_bytes();assert sha(auth)=="+repr(ASSIGNMENT_SHA256)+";assert b'fs-20260926-k2-localconstants-native201-trn21-r1' in auth and b'662c968aabd83dd4bfdbf497eb6491be30e6572f581cd4be15dd5c620f470a00' in auth")
replace_once("if choice in ('trn22-off','trn22-on'):","if choice in ('trn22-off','trn22-on','trn21-localconstants-native201'):")
replace_once("review=(REPO/'research/v2/verification/20260926-k2-sprint-independent/REVIEW.md').read_bytes();assert sha(review)=='2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'", "review=(REPO/'research/v2/verification/20260926-k2-compiler-local-constants/REVIEW.md').read_bytes();assert sha(review)=='2bbbd51f3a70a0fb1be5883a2a1413ea1b59a92f5643fac301fb365811124bf5'")
replace_once("'basis':'Unchanged prepared per-job13.5GiB initial/10.5GiB runtime floor; pair admission additional and explicit.'", "'basis':'Unchanged13.5GiB initial/10.5GiB runtime floor; retained genuine parent, no additional OFF job.'")
proposal_binding="""proposal_path='research/v2/proposals/20260926-k2-compiler-local-constants-timing/EXPERIMENT_PROPOSAL.json'
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)==PROPOSAL_PIN
proposal=json.loads(proposal_raw)
assert proposal['candidate']['source_sha256']==b['source_sha256']
assert proposal['candidate']['launch_command']==b['launch_command']
assert sha((REPO/proposal['candidate']['source']).read_bytes())==b['source_sha256']
assert b['label']=='fs-20260926-k2-localconstants-native201-trn21-r1'
assert b['steps']==201 and b['no_eval'] is True
assert b['lock_identity']==[66305,596823]
files['PRIOR_PROPOSAL_SNAPSHOT.json']=files['EXPERIMENT_PROPOSAL.json']
files['EXPERIMENT_PROPOSAL.json']=proposal_raw
files['CURRENT_EXPERIMENT_PROPOSAL.json']=proposal_raw
files['SUPERVISOR_ASSIGNMENT.md']=auth
files['EXACT_SOURCE_REVIEW.md']=review
b['completed_proposal_path']=proposal_path;b['completed_proposal_sha256']=sha(proposal_raw)
b['independent_review_sha256']=sha(review)
b['preparation_status']='SUPERVISOR_RELEASED_ONE_NATIVE201_PENDING_FRESH_MACHINE_CHECKS'
comparison_path=HERE/'trn21-parent-control-timing-acceptance.json'
if comparison_path.is_file():
 comparison=json.loads(comparison_path.read_text())
 b['comparison_control']['completed_bundle_sha256']=comparison['completed_bundle_sha256']
 b['comparison_control']['acceptance_sha256']=sha(comparison_path.read_bytes())
 b['comparison_control']['actual_terminal_and_timing_pending']=False
 b['comparison_control']['attached_public20m_and_quiescence_pending']=False
""".replace('PROPOSAL_PIN',repr(PROPOSAL_SHA256))
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()",proposal_binding+"files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')", "predecessor=HERE/'trn21-e1e2-control-completed/results-bundle.json'")
gate="""acceptance=json.loads((HERE/'trn21-parent-control-timing-acceptance.json').read_text())
assert acceptance['completed_bundle_sha256']==sha(predecessor.read_bytes()),'PARENT_RECEIPT_DRIFT'
assert acceptance['control_label']=='fs-20260926-sprint-e1e2-control-trn21-r1'
assert acceptance['control_source_sha256']=='e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
assert acceptance['candidate_source_sha256']==b['source_sha256']
assert acceptance['primary_statistic']=='35*131072/(t44-t9)'
for condition in ('same_host_runtime_science','first45_sequential_finite','t9_t44_trustworthy','window_no_recompile_or_stall','parent_train_eval_complete_and_budget_pass','explicit_postflight_quiescent'):
 assert acceptance['conditions'][condition] is True,'PARENT_COMPARISON_REFUSED:'+condition
assert acceptance['t44']>acceptance['t9']
assert acceptance['control_tokens_per_second']==35*131072/(acceptance['t44']-acceptance['t9'])
"""
replace_once("attempt=HERE/(choice+'-attempt.json');",gate+"attempt=HERE/(choice+'-attempt.json');")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
