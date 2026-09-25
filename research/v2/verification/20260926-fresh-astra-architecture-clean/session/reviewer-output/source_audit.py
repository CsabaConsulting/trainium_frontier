"""Independent static source/receipt audit; no model allocation or native execution."""
from pathlib import Path
import ast,copy,hashlib,json,math
W=Path('/work');P=Path('/packet');A=W/'test_artifacts'
trees={n:ast.parse((W/(n+'.py')).read_text()) for n in ['vehicle','candidate','scored_reference']}
def index(tree):
 out={}
 def walk(node,prefix=''):
  for n in ast.iter_child_nodes(node):
   if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)):
    key=prefix+n.name;out[key]=n;walk(n,key+'.')
   else:walk(n,prefix)
 walk(tree);return out
nodes={k:index(v) for k,v in trees.items()}
class NoDocs(ast.NodeTransformer):
 def visit_Expr(self,n):
  if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):return None
  return self.generic_visit(n)
def normalized(n):
 n=NoDocs().visit(copy.deepcopy(n));n.name='NAME';return ast.dump(n,include_attributes=False)
pairs={n:n for n in ['_rope_norm_fwd','_rope_norm_bwd','_RopeNormNKI','_conv2_taps','_conv2_fwd_tile','_conv2_forward','_conv2_dw_partial','_conv2_bwd_tile','_conv2_backward','_Conv2NKI','reference_forward','_muon_math','_muon_lr_g','MuonAdamW._device_scalar','MuonAdamW._muon_compile_scalars','OwnedOptimizer.state_dict','OwnedOptimizer.load_state_dict','selected_lanes']}
pairs.update(_softcap_ce_fwd='build_kernels.forward',_softcap_ce_bwd='build_kernels.backward')
comparisons=[]
for cn,rn in pairs.items():
 c=nodes['candidate'][cn];r=nodes['scored_reference'][rn]
 comparisons.append(dict(candidate=cn,candidate_line=c.lineno,reference=rn,reference_line=r.lineno,ast_equal_ignoring_docs_and_root_name=normalized(c)==normalized(r)))
# Complete changed-definition inventory against full-context vehicle.
inventory=[]
for name,node in nodes['candidate'].items():
 old=nodes['vehicle'].get(name)
 inventory.append(dict(name=name,line=node.lineno,end_line=node.end_lineno,changed_from_vehicle=old is None or ast.dump(node,include_attributes=False)!=ast.dump(old,include_attributes=False)))
# Independently authenticate the raw receipts' internal consistency, not their conclusions.
c=json.loads((P/'raw/tiny-g2-clean.json').read_text());r=json.loads((P/'raw/tiny-g2-research.json').read_text())
receipt=dict(candidate_source_hash_match=c['train_py_sha256']==hashlib.sha256((W/'candidate.py').read_bytes()).hexdigest(),reference_source_hash_match=r['train_py_sha256']==hashlib.sha256((W/'scored_reference.py').read_bytes()).hexdigest(),rank_count=[len(c['ranks']),len(r['ranks'])],ranks=[x['rank'] for x in c['ranks']],argv_width128=c['argv'][c['argv'].index('--n-embd')+1]=='128',actual_micro_batch=c['micro_batch'],recorded_sequence_len=1024,vocab_size=8192,recorded_dtype='bfloat16 (shim)',steps=c['steps'],losses_identical_by_rank=all(x['losses']==y['losses'] for x,y in zip(c['ranks'],r['ranks'])),plans_identical_by_rank=all(x['plan_sha256']==y['plan_sha256'] and x['plan_summary']==y['plan_summary'] for x,y in zip(c['ranks'],r['ranks'])),rank0_digest_dicts_identical=c['ranks'][0]['final_digests']==r['ranks'][0]['final_digests'],digest_count=len(c['ranks'][0]['final_digests']),digest_hex_chars=16,candidate_shims=c['shims'],reference_shims=r['shims'],prepare_stub_hash_match=c['prepare']['sha256']==r['prepare']['sha256'],prepare_stub_source_supplied=False,full_parameter_digest_bytes_supplied=False,recorded_cpu_parameter_count=sum(c['ranks'][0]['plan_summary']['elements_by_owner']),static_width128_vocab8192_parameter_count=84*128**2+3*8192*128+10*128+32+20)
# These assertions verify internal receipt relationships, not host authenticity or rerun results.
assert receipt['losses_identical_by_rank'] and receipt['plans_identical_by_rank'] and receipt['rank0_digest_dicts_identical']
assert receipt['recorded_cpu_parameter_count']==receipt['static_width128_vocab8192_parameter_count']==4523316
# Source-extracted pure ownership planner executed on tensor-free shape objects.
class ShapeOnly:
 dtype='fp32'
 def __init__(self,shape):self.shape=tuple(shape)
 def numel(self):return math.prod(self.shape)
 def is_contiguous(self):return True
class TorchShapeStub:float32='fp32'
plans={}
for label in ['candidate','scored_reference']:
 tiny=json.loads((A/(label+'_optimizer_plan.json')).read_text());g=[]
 for group in tiny['groups']:
  params=[]
  for name in group['params']:
   if name.startswith('unet_skip_gates') or name.endswith(('resid_lambda','x0_lambda')):shape=[1]
   elif name in ['transformer.wte.weight','lm_head.weight','value_embeds.7.weight']:shape=[8192,1024]
   elif '.local_conv.' in name:shape=[1024,1,2]
   elif '.ve_gate.' in name:shape=[8,32]
   elif '.mlp.c_fc.' in name:shape=[4096,1024]
   elif '.mlp.c_proj.' in name:shape=[1024,4096]
   else:assert '.attn.' in name;shape=[1024,1024]
   params.append(ShapeOnly(shape))
  g.append({**group,'params':params})
 ns={'torch':TorchShapeStub()};exec(compile(ast.Module(body=[nodes[label]['make_plan']],type_ignores=[]),str(W/(label+'.py')),'exec'),ns)
 entries,_,summary=ns['make_plan'](g,4);plans[label]=dict(entries=entries,summary=summary)
 assert sum(summary['elements_by_owner'])==113256724
assert plans['candidate']==plans['scored_reference']
# Compare actual execution route evidence from the independent CPU probe.
results=json.loads((A/'results.json').read_text());by_case={x['case']:x for x in results}
routes=[]
for dtype in ['float32','bfloat16']:
 for t in [1,7,16,31,32]:
  a=by_case[f'candidate_eval_{dtype}_t{t}_route'];b=by_case[f'scored_reference_eval_{dtype}_t{t}_route']
  routes.append(dict(dtype=dtype,tokens=t,candidate_shapes=a['block0_input_shapes'],reference_shapes=b['block0_input_shapes'],arithmetic_batch_route_equal=a['block0_input_shapes']==b['block0_input_shapes'],cpu_outputs_equal=by_case[f'eval_{dtype}_t{t}/batch']['equal'],max_abs=by_case[f'eval_{dtype}_t{t}/batch']['max_abs'],max_rel=by_case[f'eval_{dtype}_t{t}/batch']['max_rel']))
audit=dict(kernel_and_helper_ast_comparisons=comparisons,vehicle_definition_inventory=inventory,receipt_internal_checks=receipt,full_size_static_ownership_plans=plans,eval_route_comparison=routes,blocking_finding=dict(id='EVAL_BATCH_WRAPPER_OMITTED',candidate_lines=[1280,1290,907],reference_lines=[3268,3290,4899,4908,4914],requirement='Preserve trained checkpoint load_for_eval semantics, including wrapper-installed evaluation path.',observed='Reference returns _SubmissionEvalGPT and processes each batch row with B=1; candidate returns GPT and processes B rows together.',native_numeric_difference_measured=False,cpu_numeric_difference_measured=False))
(W/'source_audit_results.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(ast_comparisons=comparisons,receipt_checks=receipt,full_static_plan_summary=plans['candidate']['summary'],route_comparison=routes),indent=2))
