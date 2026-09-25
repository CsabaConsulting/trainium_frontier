"""Source invariance corroboration; numerical evidence lives in exactness_test.py."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

paths = {n:Path('/work/source')/(n+'.py') for n in ('baseline','candidate','schedule_reference')}
texts = {n:p.read_text() for n,p in paths.items()}
trees = {n:ast.parse(t) for n,t in texts.items()}
def dump(node): return ast.dump(node,include_attributes=False)
def defs(tree): return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
ds = {n:defs(t) for n,t in trees.items()}
checks=[]
def check(name,actual,expected): checks.append(dict(name=name,passed=actual==expected,actual=actual,expected=expected))
changed = [n for n in ds['baseline'] if dump(ds['baseline'][n]) != dump(ds['candidate'][n])]
check('changed_top_level_definitions',changed,['MuonAdamW','lr_multiplier','muon_momentum','muon_beta2','main'])
check('same_definition_names',list(ds['candidate']),list(ds['baseline']))
for n in ['lr_multiplier','muon_momentum','muon_beta2']:
    check('schedule_reference:'+n,dump(ds['candidate'][n]),dump(ds['schedule_reference'][n]))
muon_methods = {name:{n.name:n for n in ds[name]['MuonAdamW'].body if isinstance(n,ast.FunctionDef)} for name in ds}
check('changed_optimizer_methods',[n for n in muon_methods['baseline'] if dump(muon_methods['baseline'][n]) != dump(muon_methods['candidate'][n])],['_muon_step'])
check('muon_step_exact_reference',dump(muon_methods['candidate']['_muon_step']),dump(muon_methods['schedule_reference']['_muon_step']))

class RemoveApprovedMainChanges(ast.NodeTransformer):
    def visit_Global(self,node):
        return None if node.names == ['RECURSIVE_SCHEDULE'] else node
    def visit_Expr(self,node):
        v=node.value
        if isinstance(v,ast.Call) and v.args and isinstance(v.args[0],ast.Constant) and v.args[0].value=='--recursive-schedule': return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['RECURSIVE_SCHEDULE','lrm_adam','lrm_group']: return None
        return self.generic_visit(node)
    def visit_Name(self,node):
        if node.id=='lrm_group': return ast.copy_location(ast.Name(id='lrm',ctx=node.ctx),node)
        return node

normal = RemoveApprovedMainChanges().visit(copy.deepcopy(ds['candidate']['main']))
check('main_unchanged_after_removing_only_approved_schedule_statements',dump(normal),dump(ds['baseline']['main']))

locations={n:{name:dict(first=node.lineno,last=node.end_lineno) for name,node in d.items()} for n,d in ds.items()}
changed_ranges=[]
a,b=texts['baseline'].splitlines(),texts['candidate'].splitlines()
for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
    if tag!='equal': changed_ranges.append(dict(change=tag,baseline_lines=[i+1,j],candidate_lines=[k+1,l],removed=a[i:j],added=b[k:l]))

# Closed-form parameter census inferred from source and packet's target config.
# No model, meta model or production tensor allocation.
def census(width):
    parts=dict(three_embedding_matrices=3*8192*width,eight_mlp_pairs=8*2*width*(4*width),
               five_attention_qkv_out=5*4*width*width,five_local_convs=5*2*width,
               last_attention_value_gate=32*(width//128),block_residual_scalars=8*2,unet_skip_scalars=4)
    return dict(parts=parts,total=sum(parts.values()),tensors=3+8*2+5*4+5+1+8*2+4)
check('target_parameter_census',census(1024)['total'],113256724)
check('reduced_parameter_census',census(128)['total'],4523316)
check('parameter_tensor_census',census(1024)['tensors'],65)
check('production_tokens_per_update',1*32*1024*4,131072)

# Avoid duplicating megabytes of AST; hashes preserve the compared values.
for item in checks:
    if isinstance(item['actual'],str) and len(item['actual'])>500:
        for k in ['actual','expected']:
            item[k+'_sha256']=hashlib.sha256(item.pop(k).encode()).hexdigest()
out=dict(checks=checks,changed_ranges=changed_ranges,definition_locations=locations,
         static_parameter_census={'target_8x1024':census(1024),'reduced_8x128':census(128)},
         notes=['Census assumes packet config, padded vocabulary 8192, attention layers 0/2/4/6/7, local conv kernel 2, one value embedding, no ngram or extra optional features. It is source arithmetic, not instantiated-model proof.',
                'No imports of candidate, evaluator or dependencies; source comments about earlier experiments are not treated as evidence.'])
Path('/work/artifacts/source_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[x for x in checks if not x['passed']],changed_ranges=len(changed_ranges),changed_definitions=changed,census=out['static_parameter_census']),indent=2))
raise SystemExit(any(not c['passed'] for c in checks))
