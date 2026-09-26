"""Build and check the proven staging payload locally; execution is not available."""
import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert len(sys.argv) == 3 and sys.argv[1] in ('trn21-warmdown060', 'trn22-warmdown085')
assert sys.argv[2:] == ['--local-check'], 'LOCAL_PREPARATION_ONLY_NO_HOST_EXECUTION'
old = HERE.parent / '20260926-final-sprint-successors/stage_next.py'
raw = old.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
text = raw.decode()


def replace_once(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)


replace_once('choices={', "choices={\n 'trn21-warmdown060':('20260926-operator-recovery/trn21-warmdown060-prepared.disabled.json','f402300af29399caf6aff0be6f9eb186a659a76679ff21ee05b5baa438582ab3'),\n 'trn22-warmdown085':('20260926-operator-recovery/trn22-warmdown085-prepared.disabled.json','61e3356f8c3fe0d1e30927bd1b64dc289d5e06724c1cfd0d7c90da40057eafb1'),")
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth", "assignment=REPO/'research/v2/assignments/20260926-adopted-batch-warmdown-brackets.md';auth=assignment.read_bytes();assert sha(auth)=='79edb9bc639e996488d899604895dd7f0ed53e0c0dd3f8d281b95bb1b326a824'")
replace_once("else:review=(REPO/'research/v2/verification/20260926-p0-sprint-fresh-independent/REVIEW.md').read_bytes();assert b'PASS within bounded source/CPU scope' in review", "else:review=(REPO/'research/v2/verification/20260926-warmdown-flag-fresh-review/REVIEW.md').read_bytes();assert sha(review)=='b8b22eb3c8384fc34ff18a8f04c012ddcb2f1ce6671dbba26a5077f8d4ceab68'")
replace_once('b.update(stage_authorized=True,launch_authorized=True,submission_authorized=False', 'b.update(stage_authorized=False,launch_authorized=False,submission_authorized=False')
replace_once("'launch_authorized':True,'submission_authorized':False", "'launch_authorized':False,'submission_authorized':False")
checks = """proposal_path=b['completed_proposal_path']
proposal_raw=(REPO/proposal_path).read_bytes()
assert sha(proposal_raw)==b['completed_proposal_sha256']
proposal=json.loads(proposal_raw)
assert proposal_raw==files['EXPERIMENT_PROPOSAL.json']
assert proposal['candidate']['source_sha256']==b['source_sha256']=='4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d'
assert (REPO/proposal['candidate']['source']).read_bytes()==files['package/train.py']
assert b['host']==proposal['dispatch']['selected_host'] and b['label']==proposal['dispatch']['selected_label']
assert files['package/launch-command.txt']==(proposal['candidate']['launch_command']+'\\n').encode()
assert sha(files['package/launch-command.txt'])==proposal['candidate']['launch_command_sha256']
assert sha(files['CPU_RECEIPT.json'])=='28799a7ac2fd3cd89fea92712fd42823db20df9fdf7320f3d452e9d71e998f67'
assert files['EXACT_SOURCE_REVIEW.md']==review and files['SUPERVISOR_ASSIGNMENT.md']==auth
if b['host']=='trn21':b['resource_reserve']['initial_floor_bytes']=max(15032385536,b['resource_reserve']['initial_floor_bytes'])
assert b['kind']=='schedule_arm' and b['base_sha256']=='8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee'
assert not b['launch_authorized'] and not b['stage_authorized']
"""
replace_once("files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()", checks + "files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\\n').encode()")
replace_once("'status':'LOCAL_CHECK_PASS'", "'status':'LOCAL_CHECK_PASS_DISABLED_NO_HOST_CALLS'")
ast.parse(text)
exec(compile(text, str(old), 'exec'), {'__file__': __file__, '__name__': '__main__'})
