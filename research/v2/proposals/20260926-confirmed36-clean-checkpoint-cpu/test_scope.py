"""Stdlib-only frozen proposal/source/helper scope checks; no scientific import."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
COUNT=0
def check(value,name):
    global COUNT
    COUNT+=1
    if not value:raise RuntimeError(name)
def blob(ref,path):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    proposal=json.loads((HERE/'EXPERIMENT_PROPOSAL.json').read_text())
    check((HERE/'EXPERIMENT_PROPOSAL.json').read_bytes()==blob('0930f2ae',str((HERE/'EXPERIMENT_PROPOSAL.json').relative_to(REPO))),'PROPOSAL_UNCHANGED')
    source_trees={}
    for name,entry in proposal['sources'].items():
        raw=blob(entry['commit'],entry['path'])
        check(sha(raw)==entry['sha256'],'EXACT_SCIENTIFIC_SOURCE:'+name)
        source_trees[name]=ast.parse(raw)
        loader=next(n for n in source_trees[name].body if isinstance(n,ast.FunctionDef) and n.name=='load_for_eval')
        loads=[n for n in ast.walk(loader) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.load']
        if name=='reference':
            check(any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch_load_weights' for n in ast.walk(loader)),'REFERENCE_WEIGHTS_HELPER_CALLED')
            wrapper=next(n for n in source_trees[name].body if isinstance(n,ast.FunctionDef) and n.name=='torch_load_weights')
            loads=[n for n in ast.walk(wrapper) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.load']
        check(len(loads)==1 and any(k.arg=='weights_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in loads[0].keywords),'SOURCE_WEIGHTS_ONLY:'+name)
    candidate=next(n for n in source_trees['candidate'].body if isinstance(n,ast.ClassDef) and n.name=='GPT')
    buffers=[ast.literal_eval(n.args[0]) for n in ast.walk(candidate) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self.register_buffer']
    check(buffers==['cos','sin'],'ACTUAL_CLEAN_ROOT_BUFFERS')
    loader=next(n for n in source_trees['candidate'].body if isinstance(n,ast.FunctionDef) and n.name=='load_for_eval')
    check(any(isinstance(n,ast.Call) and ast.unparse(n.func)=='GPT' for n in ast.walk(loader)),'CLEAN_ORDINARY_GPT')
    check(not any(isinstance(n,ast.Name) and n.id=='_SubmissionEvalGPT' for n in ast.walk(loader)),'NO_ADAPTER_LOADER')
    old=REPO/proposal['existing_helper']['path']
    check(sha(old.read_bytes())==proposal['existing_helper']['sha256'],'OLD_HELPER_UNCHANGED')
    code=(HERE/'compare_checkpoints.py').read_text()
    tree=ast.parse(code); oldtree=ast.parse(old.read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    oldfunctions={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
    for name in ('raw_record','restored_mask_spec','mask_record','model_state','tensor_record','require_state_equal'):
        check(ast.dump(functions[name],include_attributes=False)==ast.dump(oldfunctions[name],include_attributes=False),'PRESERVED_MASK_SEMANTICS:'+name)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                check(not (isinstance(target,ast.Attribute) and target.attr in ('forward','load_for_eval','compute_dtype_for','COMPUTE_DTYPE')),'NO_SOURCE_OVERRIDE')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    check(not any(ast.unparse(n.func) in ('causality_check','make_eval_dataloader','make_dataloader','torch.compile') for n in calls),'NO_DATA_OR_NATIVE_WORK')
    loads=[n for n in calls if ast.unparse(n.func)=='torch.load']
    check(len(loads)==1 and any(k.arg=='weights_only' and ast.literal_eval(k.value) is True for k in loads[0].keywords),'HELPER_WEIGHTS_ONLY')
    check("for batch,length in SHAPES:" in code and "((1,256),(2,256),(3,256),(1,257),(2,257),(3,257))" in code,'SIX_FIXED_SHAPES')
    check("if error>1e-2:" in code and "prefix_stop':cut-1" in code,'UNCHANGED_CAUSAL_TOLERANCE')
    check(code.count('authenticate_files(files)')==3,'FILE_HASH_PRE_POST_PLUS_DEFINITION')
    check("'source_changed':False" in code,'SCOPED_RECEIPT')
    pending=json.loads((HERE/'INPUTS.pending.json').read_text())
    check(pending['status'].startswith('PENDING_') and pending['files'] is None,'PENDING_REAL_HASHES')
    check(all(a['checkpoint_path'] is None and a['checkpoint_sha256'] is None and a['checkpoint_bytes'] is None for a in pending['arms'].values()),'NO_INVENTED_CHECKPOINT')
    for name in ('reference','candidate'):
        req=REPO/'research/v2/proposals/20260926-confirmed36-clean-cpu-trn21-r1'/name/'request.json'
        check(sha(req.read_bytes()) in code,'EXACT_REQUEST_PIN:'+name)
    check(sha((REPO/'prepare.py').read_bytes())=='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb','PREPARE_UNCHANGED')
    print(json.dumps({'status':'IMPLEMENTER_STATIC_ONLY','checks':COUNT,'helper_sha256':sha(code.encode()),
                      'source_imports':False,'tensor_allocations':False,'host_actions':False},indent=2))
if __name__=='__main__':main()
