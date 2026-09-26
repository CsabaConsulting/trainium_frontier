"""Existing exact-byte full-job transport rebound to the selected trn22 LR arm."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'stage_next.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
assert sys.argv[1]=='trn22-batch-lr14' and sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
def replace_once(old,new):
 global text
 assert text.count(old)==1,old
 text=text.replace(old,new)
replace_once('choices={',"choices={\n 'trn22-batch-lr14':('20260926-final-sprint-p0/batch-lr14-trn22-successor-prepared.json','bb5d0a7977783b04568cb9bd4b54971002cd1121487a925a7136298e10d8bfdc'),")
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth", "assignment=REPO/'research/v2/assignments/20260926-batch-lr021-trn22-fallback.md';auth=assignment.read_bytes();assert sha(auth)=='efcf29b1287bd973500b6871bae1860fd18657869128b73780e52d28d5dcf378'")
proposal_binding="""proposal_path='research/v2/proposals/20260926-sprint-p0-clean-batch/EXPERIMENT_PROPOSAL.json'
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)=='785bfe4ea42bd8c82cacfeabf874de39bb7f253bc5bf1b23e2f7b112e6db8aef'
proposal=json.loads(proposal_raw)
assert proposal['candidate_sha256']==b['source_sha256']=='8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee'
assert 'batch-lr14-trn22' in proposal['variants']
assert files['EXPERIMENT_PROPOSAL.json']==proposal_raw
b['completed_proposal_path']=proposal_path;b['completed_proposal_sha256']=sha(proposal_raw)
b['independent_review_sha256']=sha(review)
b['first_source_native_timing_pending']=False
b['first_source_native_evidence']='research/v2/operator/20260926-final-sprint-successors/trn22-p0-completed/results-bundle.json'
b['first_source_native_evidence_sha256']='cc54af74150b6512cac5861c04b3908e5b21fecd2b4f85e8aa9b1e9e7b3d99a8'
qualified_raw=(HERE/'trn22-p0-completed/results-bundle.json').read_bytes()
assert sha(qualified_raw)==b['first_source_native_evidence_sha256']
qualified=json.loads(qualified_raw)
assert json.loads(qualified['files']['terminal.json']['text'])['eval_exit_code']==0
assert json.loads(qualified['files']['train/rehearsal.json']['text'])['charged_through_process_exit_within_budget'] is True
"""
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()",proposal_binding+"files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')", "predecessor=HERE/'trn22-indexrepair-completed/results-bundle.json'\nassert sha(predecessor.read_bytes())=='98d21e753ca955e67d3d78418ca7972b1804af0fede83c2826c946f0c8118164'")
replace_once("if choice=='trn22-on':assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0\nelse:assert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0", "assert terminal['label']=='fs-20260926-k2-indexrepair-on-parentcontrol-trn22-r1' and terminal['status']=='FULL_DRIVER_FAILED' and terminal['error']=='TRAIN_EXIT_1'\nassert p['lock_device']==66305 and p['lock_inode']==832589")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
