"""Supervisor-released warmdown binding; terminal/eval/quiescence must precede dispatch."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
old = HERE / 'stage_warmdown.disabled.py'
raw = old.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ceae5db9440aa94841680dc7a307e6692cba071307c7ce0b7d66fcef009cc8e6'
text = raw.decode()


def replace_once(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)


replace_once("assert sys.argv[2:] == ['--local-check'], 'LOCAL_PREPARATION_ONLY_NO_HOST_EXECUTION'",
             "assert sys.argv[2:] in (['--local-check'], ['--execute'])")
replace_once('20260926-adopted-batch-warmdown-brackets.md', '20260926-warmdown-next-release.md')
replace_once('79edb9bc639e996488d899604895dd7f0ed53e0c0dd3f8d281b95bb1b326a824',
             'c0c63ef2a76f5d4b21a67393abf6b42b386ec2ba241531473211c4ecb21f5df5')
replace_once("replace_once('b.update(stage_authorized=True,launch_authorized=True,submission_authorized=False', 'b.update(stage_authorized=False,launch_authorized=False,submission_authorized=False')\n", '')
replace_once('replace_once("\'launch_authorized\':True,\'submission_authorized\':False", "\'launch_authorized\':False,\'submission_authorized\':False")\n', '')
replace_once("assert files['EXACT_SOURCE_REVIEW.md']==review and files['SUPERVISOR_ASSIGNMENT.md']==auth",
             "assert files['EXACT_SOURCE_REVIEW.md']==review\nassert sha(files['SUPERVISOR_ASSIGNMENT.md'])=='79edb9bc639e996488d899604895dd7f0ed53e0c0dd3f8d281b95bb1b326a824'\nfiles['SCIENCE_ASSIGNMENT.md']=files['SUPERVISOR_ASSIGNMENT.md']\nfiles['SUPERVISOR_ASSIGNMENT.md']=auth")
replace_once("assert not b['launch_authorized'] and not b['stage_authorized']",
             "assert b['launch_authorized'] and b['stage_authorized']")
replace_once('LOCAL_CHECK_PASS_DISABLED_NO_HOST_CALLS', 'LOCAL_CHECK_PASS_CONDITIONAL_RELEASE_NO_HOST_CALLS')
extra = '''
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')", "predecessor=HERE/(host+'-current-completed/results-bundle.json')")
replace_once("if choice=='trn22-on':assert terminal['status']=='TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' and terminal['train_exit_code']==0\\nelse:assert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0", "assert terminal['label']=={'trn21':'fs-20260926-batch-lr021-recursive-stack-trn21-r1','trn22':'fs-20260926-batch262144-lr02625-trn22-r1'}[host]\\nassert terminal['status']=='COMPLETE_PENDING_INDEPENDENT_REVIEW' and terminal['eval_exit_code']==0\\nassert (p['lock_device'],p['lock_inode'])==(66305,596823 if host=='trn21' else 832589)\\nreplica_path=REPO/'research/v2/proposals'/({'trn21':'20260926-batch262144-lr02625-crosshost-trn21','trn22':'20260926-batch-lr021-recursive-crosshost-trn22'}[host])/'EXPERIMENT_PROPOSAL.json'\\nassert json.loads(replica_path.read_bytes())['trigger']['satisfied'] is False,'QUALIFIED_CROSSHOST_REPLICA_HAS_PRIORITY'")
'''
replace_once("ast.parse(text)\nexec(", extra + "\nast.parse(text)\nexec(")
ast.parse(text)
exec(compile(text, str(old), 'exec'), {'__file__': __file__, '__name__': '__main__'})
