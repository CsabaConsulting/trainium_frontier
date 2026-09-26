import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

BASE=Path(__file__).parent
PACKET=Path('/home/poojith-devan/checkpoint-review-packet-Abknyi')
s=importlib.util.spec_from_file_location('review_helper',BASE/'candidate.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
results=[]
def check(name,truth):
    if not truth:raise RuntimeError(name)
    results.append(name)
for name,sha in h.SOURCES.items():
    raw=(PACKET/name/'train.py').read_bytes()
    check(name+'_source_sha',hashlib.sha256(raw).hexdigest()==sha)
    tree=ast.parse(raw)
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    loader=funcs['load_for_eval']
    calls=[n for n in ast.walk(loader) if isinstance(n,ast.Call)]
    load_calls=[n for n in calls if isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='torch.load']
    if name=='reference':
        check('reference_loader_calls_safe_wrapper',any(isinstance(n.func,ast.Name) and n.func.id=='torch_load_weights' for n in calls))
        load_calls=[n for n in ast.walk(funcs['torch_load_weights']) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.load']
    check(name+'_weights_only',len(load_calls)==1 and any(k.arg=='weights_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in load_calls[0].keywords))
    check(name+'_strict_load',any(isinstance(n.func,ast.Attribute) and n.func.attr=='load_state_dict' and any(k.arg=='strict' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in calls))
    check(name+'_eval',any(ast.unparse(n.func)=='model.eval' for n in calls))
    check(name+'_freeze',any(ast.unparse(n.func)=='model.requires_grad_' and isinstance(n.args[0],ast.Constant) and n.args[0].value is False for n in calls))
    if name=='candidate':check('candidate_constructs_ordinary_GPT',any(isinstance(n.func,ast.Name) and n.func.id=='GPT' for n in calls))
old=subprocess.check_output(['git','show','0930f2ae:research/v2/proposals/20260926-clock36-checkpoint-mask-r3/compare_checkpoints.py'],cwd='/home/poojith-devan/Desktop/Trainium')
check('baseline_sha',hashlib.sha256(old).hexdigest()=='1b3714e050e473b176a3e1f1fbd59387579867f3d06d14ecf71248a6d8a1714c')
oldf={n.name:n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
newtree=ast.parse((BASE/'candidate.py').read_text());newf={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
for name in ('raw_record','restored_mask_spec','mask_record','model_state','tensor_record','require_state_equal'):
    check('preserved_'+name,ast.dump(oldf[name])==ast.dump(newf[name]))
check('helper_direct_loader_call',any(isinstance(n,ast.Call) and ast.unparse(n.func)=='module.load_for_eval' for n in ast.walk(newf['run'])))
check('helper_no_model_or_module_assignments',not any(isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('model','module','mod','ordinary') for t in (n.targets if isinstance(n,ast.Assign) else [n.target])) for n in ast.walk(newtree)))
# Simulate a pair rejection before any source import or checkpoint allocation.
# Actual file identity/hash routines remain active, so finally reauthentication
# is exercised on reviewer-owned small input; no production paths are touched.
input_path=PACKET/'EXPERIMENT_PROPOSAL.json'
input_sha=h.file_record(input_path)['sha256']
calls=[]
original=h.authenticate_files
def authenticate(files):
    calls.append('auth')
    return original(files)
report={}
with patch.object(h,'validate_manifest',return_value={str(input_path):h.file_record(input_path)}),patch.object(h,'authenticate_files',side_effect=authenticate),patch.object(h,'qualify_pair',side_effect=h.CheckFailure('OWN_SYNTHETIC_PAIR_FAILURE')):
    try:h.run(input_path,input_sha,report)
    except h.CheckFailure as e:check('pair_error_preserved',str(e)=='OWN_SYNTHETIC_PAIR_FAILURE')
    else:raise RuntimeError('pair falsely accepted')
check('finally_rehash_on_pair_failure',calls==['auth','auth'] and report['inputs_unchanged_after'] is True)
print(json.dumps({'status':'SOURCE_AND_SMALL_BOUNDARY_CHECKS_ONLY','checks':results,'count':len(results),'real_input_acceptance':False},indent=2))
