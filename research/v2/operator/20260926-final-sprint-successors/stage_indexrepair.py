"""Exact reviewed repair source with the unchanged ON45 transport and guardian."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'stage_next.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
assert sys.argv[1]=='trn22-indexrepair'
assert sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
def replace_once(old,new):
 global text
 assert text.count(old)==1,old
 text=text.replace(old,new)
replace_once('choices={',"choices={\n 'trn22-indexrepair':('20260926-k2-sprint-timing/trn22-indexrepair-prepared.json','e09012030dd7a98b82e160998665a1bee1bb66deea467994f86251e1e0e162e8'),")
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth", "assignment=REPO/'research/v2/assignments/20260926-k2-native-index-repair.md';auth=assignment.read_bytes();assert b'fs-20260926-k2-indexrepair-on-parentcontrol-trn22-r1' in auth and b'a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71' in auth and b'14495514624' in auth")
replace_once("if choice in ('trn22-off','trn22-on'):","if choice in ('trn22-off','trn22-on','trn22-indexrepair'):")
replace_once("review=(REPO/'research/v2/verification/20260926-k2-sprint-independent/REVIEW.md').read_bytes();assert sha(review)=='2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'", "review=(REPO/'research/v2/verification/20260926-k2-native-index-repair/REVIEW.md').read_bytes();assert sha(review)=='b689fb6ed0dd86b73bb945da2fb6bb31a3e8a524b15dc3c984624a5dea08c6dc'")
replace_once("'basis':'Unchanged prepared per-job13.5GiB initial/10.5GiB runtime floor; pair admission additional and explicit.'", "'basis':'Unchanged per-job13.5GiB initial/10.5GiB runtime floor; allocated parent retained, no extra OFF45.'")
proposal_binding="""proposal_path='research/v2/proposals/20260926-k2-native-index-repair-timing/EXPERIMENT_PROPOSAL.json'
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)=='0b2f56940734765e9a85e46b250fd3d1ed8a989fa701c5ef7bfa0d908bcb7e5c'
proposal=json.loads(proposal_raw)
assert proposal['candidate']['source_sha256']==b['source_sha256']
assert proposal['candidate']['launch_command']==b['launch_command']
assert proposal['candidate']['run_label_template']==b['label']
assert proposal['control']['run_label']==b['comparison_control']['label']
assert sha((REPO/proposal['candidate']['source']).read_bytes())==b['source_sha256']
files['CURRENT_EXPERIMENT_PROPOSAL.json']=proposal_raw
b['completed_proposal_path']=proposal_path;b['completed_proposal_sha256']=sha(proposal_raw)
"""
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()",proposal_binding+"files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')", "predecessor=HERE.parent/'20260926-p3-component-timing/trn22-completed/results-bundle.json'")
replace_once("if choice=='trn22-on':assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0\nelse:assert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0", "assert terminal['label']=='fs-20260926-p3-component-trn22-r1'\nassert terminal['status'] in ('DIRECT_DIAGNOSTIC_EXIT_PENDING_RAW_INTERPRETATION','DIRECT_DIAGNOSTIC_FAILED','DIRECT_DIAGNOSTIC_DRIVER_FAILED')\nassert p['lock_device']==66305 and p['lock_inode']==832589")
gate="""parent=HERE/'trn22-e1e2-control-completed/results-bundle.json'
acceptance=json.loads((HERE/'trn22-parent-control-timing-acceptance.json').read_text())
assert sha(parent.read_bytes())==acceptance['completed_bundle_sha256']=='f17ea4ebbbef720cee5d02f810fcb103fc887a5d56fa7c906910f28346dae6c7'
assert acceptance['control_label']=='fs-20260926-sprint-e1e2-control-trn22-r1'
assert acceptance['control_source_sha256']=='e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
# Preserve the original candidate binding, then bind the reviewed two-line repair
# explicitly. The original acceptance receipt is never edited or relabelled.
assert acceptance['candidate_source_sha256']=='63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5'
assert b['source_sha256']=='a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71'
assert acceptance['precommit_commit']=='d3da2359'
assert acceptance['primary_statistic']=='35*131072/(t44-t9)'
for condition in ('same_host_runtime_science','first45_sequential_finite','t9_t44_trustworthy','window_no_recompile_or_stall','parent_train_eval_complete_and_budget_pass','explicit_postflight_quiescent'):
 assert acceptance['conditions'][condition] is True,'PARENT_COMPARISON_REFUSED:'+condition
assert acceptance['t44']>acceptance['t9']
assert acceptance['control_tokens_per_second']==35*131072/(acceptance['t44']-acceptance['t9'])
failed=HERE/'trn22-on-parentcontrol-completed/results-bundle.json'
assert sha(failed.read_bytes())=='52c017298afa972fb5731e479f1c3fa4eb72ee5ae7ed8eb39c067c41de4976de'
assert json.loads(json.loads(failed.read_bytes())['files']['terminal.json']['text'])['error']=='TRAIN_EXIT_1'
"""
replace_once("attempt=HERE/(choice+'-attempt.json');",gate+"attempt=HERE/(choice+'-attempt.json');")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
