"""Exact existing full transport: selected flag-only LR bracket, no new runner."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'stage_next.py').read_bytes();assert hashlib.sha256(raw).hexdigest()=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
assert sys.argv[1]=='trn22-batch-lr02625' and sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
def replace_once(a,b):
 global text
 assert text.count(a)==1,a
 text=text.replace(a,b)
replace_once('choices={',"choices={\n 'trn22-batch-lr02625':('20260926-batch-lr02625/trn22-prepared.disabled.json','c0714210defc6654c1bcf37150ee52f6a1041ab977a52de13df4ac5ab8ec8911'),")
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth", "assignment=REPO/'research/v2/assignments/20260926-batch262144-lr02625-trn22.md';auth=assignment.read_bytes();assert sha(auth)=='1fc3fef77ad5ec5169017616a597d4147bfc38e467c7fe359754deff9d30e3d1'")
insert="""proposal_path='research/v2/proposals/20260926-batch262144-lr02625-trn22/EXPERIMENT_PROPOSAL.json'
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)=='f3b1002296836887091d1d43383ada6eeee1884ad13d7abce0876bf3a82fc38e'
proposal=json.loads(proposal_raw)
assert files['EXPERIMENT_PROPOSAL.json']==proposal_raw
assert proposal['candidate']['source_sha256']==b['source_sha256']=='8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee'
assert sha((REPO/proposal['candidate']['source']).read_bytes())==b['source_sha256']
assert b['label']==proposal['dispatch']['selected_label'] and b['host']==proposal['dispatch']['selected_host']=='trn22'
assert files['package/launch-command.txt']==(proposal['candidate']['launch_command']+'\\n').encode()
assert sha(files['package/launch-command.txt'])==proposal['candidate']['launch_command_sha256']=='4097cc633b8cee9fe9869de5adc75770c61741209466730ad006d8950f2cfe04'
assert b['kind']=='flag_arm' and b['base_sha256']==proposal['prospective_record']['base_sha256']
assert sha(review)==proposal['existing_evidence']['source_review_sha256']
b['completed_proposal_path']=proposal_path;b['completed_proposal_sha256']=sha(proposal_raw)
b['independent_review_sha256']=sha(review)
files['SUPERVISOR_ASSIGNMENT.md']=auth;files['EXACT_SOURCE_REVIEW.md']=review
control=(HERE/'trn22-batch-lr14-completed/results-bundle.json').read_bytes()
assert sha(control)==proposal['control']['raw_archive_sha256']=='c05eaa57272719cb7ba3e2d24dae67f11ba6ca6e2de170ba7d0fbffe163bd91f'
b['same_host_actual_comparator_bundle_sha256']=sha(control)
"""
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()",insert+"files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')", "predecessor=HERE/'trn22-runtimegroups-native201-completed/results-bundle.json'\nassert sha(predecessor.read_bytes())=='025a588c82059b363a3f8aefeb1d4188d2fefbcf23e7106235408be9c03a0ba5'")
replace_once("if choice=='trn22-on':assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0\nelse:assert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0", "assert terminal['label']=='fs-20260926-k2-runtimegroups-native201-trn22-r1' and terminal['status']=='FULL_DRIVER_FAILED' and terminal['error']=='TRAIN_EXIT_1'\nassert p['lock_device']==66305 and p['lock_inode']==832589")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
