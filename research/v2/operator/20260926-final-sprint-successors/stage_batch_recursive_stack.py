"""Literal exact-byte full transport rebind; original guardian and guards unchanged."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'stage_next.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
assert sys.argv[1]=='trn21-batch-recursive-stack' and sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
def replace_once(a,b):
 global text
 assert text.count(a)==1,a
 text=text.replace(a,b)
replace_once('choices={',"choices={\n 'trn21-batch-recursive-stack':('20260926-batch-recursive-stack/trn21-prepared.disabled.json','945a1184fbfd21017482560e7ec351a216345f282d6d4f4c058d7d5d98b4f782'),")
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth", "assignment=REPO/'research/v2/assignments/20260926-batch-lr021-recursive-stack-trn21.md';auth=assignment.read_bytes();assert sha(auth)=='6a5f0cd9cdce5bfa8aab00cd2cbb1aa939846894db4c4b90035fab7aacc525b9'")
replace_once("else:review=(REPO/'research/v2/verification/20260926-p0-sprint-fresh-independent/REVIEW.md').read_bytes();assert b'PASS within bounded source/CPU scope' in review", "else:review=(REPO/'research/v2/verification/20260926-batch-recursive-composition/REVIEW.md').read_bytes();assert sha(review)=='4b5328aee65854d0d3dde58ad844956824750f85b20f9c494717502d5e66a669'")
insert="""proposal_path='research/v2/proposals/20260926-batch-lr021-recursive-stack/EXPERIMENT_PROPOSAL.json'
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)=='f923c591052d05830bdb1baecf1c602d2939e693ff8c13c6f8a8dafe5e17aa8e'
proposal=json.loads(proposal_raw)
assert files['EXPERIMENT_PROPOSAL.json']==proposal_raw
assert proposal['source_sha256']==b['source_sha256']=='c2e27ccde59e619d74102429e340716dbec16a6ba538ea9beca9c714d0d33f87'
assert sha((REPO/proposal['source']).read_bytes())==b['source_sha256']
assert b['label']==proposal['dispatch']['selected_label'] and b['host']==proposal['dispatch']['selected_host']=='trn21'
assert files['package/launch-command.txt']==(proposal['candidate']['launch_command']+'\\n').encode()
assert sha(proposal['candidate']['launch_command'].encode())=='17e182b51e862becb11b75ba26702812720470e5ac4f8045cfe1f2cf6bc32984'
assert b['kind']=='schedule_arm' and b['base_sha256']==proposal['base_sha256']
b['completed_proposal_path']=proposal_path;b['completed_proposal_sha256']=sha(proposal_raw)
b['independent_review_sha256']=sha(review)
b['resource_reserve']['initial_floor_bytes']=max(15032385536,b['resource_reserve']['initial_floor_bytes'])
files['SUPERVISOR_ASSIGNMENT.md']=auth;files['EXACT_SOURCE_REVIEW.md']=review
files['CPU_RECEIPT.json']=(REPO/proposal['code_evidence']['stack_cpu']).read_bytes()
assert sha(files['CPU_RECEIPT.json'])==proposal['code_evidence']['stack_cpu_sha256']
"""
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()",insert+"files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')", "predecessor=HERE/'trn21-localconstants-native201-completed/results-bundle.json'\nassert sha(predecessor.read_bytes())=='ef07d62a5ab66f0d44e7a3a091556446d72170e4434e7bdfc51e4be9ad747116'")
replace_once("if choice=='trn22-on':assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0\nelse:assert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0", "assert terminal['label']=='fs-20260926-k2-localconstants-native201-trn21-r1' and terminal['status']=='FULL_DRIVER_FAILED' and terminal['error']=='TRAIN_EXIT_143'\nassert p['lock_device']==66305 and p['lock_inode']==596823")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
