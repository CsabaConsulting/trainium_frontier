"""Serial, reduced CPU integration + AOT exclusion; no native claim or data access."""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.parent / 'k2-runtime-groups/train.py'
BASE = REPO / 'research/v2/proposals/20260926-sprint-p0-clean-batch/train.py'
EXPECTED = '949f262c092c141f715bdd71532c42e0bcf502d926a1ee20bc885a2255062daa'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(a, b):
    require(torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().contiguous().reshape(-1).view(torch.uint8)), 'BITWISE_DIFFERENCE')


require(sha(ROOT/'train.py') == EXPECTED, 'CANDIDATE_DRIFT')
require(sha(PARENT) == 'f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e', 'KERNEL_PARENT_DRIFT')
require(sha(BASE) == '8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee', 'BASE_DRIFT')
nodes = [{n.name: ast.dump(n) for n in ast.parse(p.read_text()).body
          if isinstance(n, (ast.FunctionDef, ast.ClassDef))} for p in (PARENT, ROOT/'train.py')]
for name in ('_k2_build_kernel','_k2_prepare_metadata','_k2_build_registered_lookup'):
    require(nodes[0][name] == nodes[1][name], 'KERNEL_OR_HELPER_CHANGED:'+name)

sys.path.insert(0, str(REPO/'workflow/tools/stub_prepare'))
arrow, parquet = types.ModuleType('pyarrow'), types.ModuleType('pyarrow.parquet')
arrow.parquet = parquet
sys.modules['pyarrow'], sys.modules['pyarrow.parquet'] = arrow, parquet


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load('rescue_base', BASE)
candidate = load('rescue_candidate', ROOT/'train.py')
interpreter = load('rescue_instruction_interpreter', ROOT.parent/'k2-runtime-groups/cpu_instruction_model.py')
kernel_events = []


def kernel(g, counts, starts, positions, maximums):
    # Fail, rather than silently accepting an AOT/FakeTensor kernel invocation.
    from torch._subclasses.fake_tensor import FakeTensor
    require(not isinstance(g, FakeTensor), 'KERNEL_WAS_AOT_TRACED')
    require(not torch.compiler.is_compiling(), 'KERNEL_DYNAMO_CAPTURED')
    kernel_events.append(tuple(g.shape))
    features = g.shape[1]//2
    with interpreter.interpreted_kernel(counts.shape[0], g.shape[0], g.shape[1],
                                         counts.shape[0]//maximums.shape[0]) as (body, lang, isa):
        parts = []
        for owner in (0,1):
            lang.pid = owner
            out = body(interpreter.Tensor(g), *(interpreter.Tensor(x) for x in (counts,starts,positions,maximums)))
            parts.append(out.data[:,owner*features:(owner+1)*features])
        return torch.cat(parts, dim=1)


candidate._K2_LOOKUP = candidate._k2_build_registered_lookup(kernel=kernel, nki_op_factory=torch.library.custom_op)
candidate._K2_METADATA = lambda ids,vocab: candidate._k2_prepare_metadata(ids,vocab,4)
# Reused loader storage and two tables / two accumulation microsteps.
for dtype in (torch.int32,torch.int64):
    torch.manual_seed(777)
    weights=[torch.randn(8,8,requires_grad=True) for _ in range(2)]
    refs=[w.detach().clone().requires_grad_() for w in weights]
    for micro in range(2):
        ids=torch.randint(8,(2,8),dtype=dtype)
        original=ids.clone()
        meta=candidate._k2_prepare_metadata(ids,8,4)
        outputs=[candidate._K2_LOOKUP(w,*meta) for w in weights]
        expected=[torch.nn.functional.embedding(original,w).bfloat16() for w in refs]
        ids.fill_(1)
        for out,ref,w,wr in zip(outputs,expected,weights,refs):
            exact(out,ref)
            grad=torch.randn(2,8,8).bfloat16()
            out.backward(grad);ref.backward(grad)
            exact(w.grad,wr.grad)
receipts = []
for index_dtype in (torch.int32, torch.int64):
    models=[]
    for module in (base,candidate):
        torch.manual_seed(42)
        m=module.GPT(module.GPTConfig(sequence_len=8,vocab_size=64,n_layer=8,n_head=2,n_embd=64,mlp_ratio=2))
        m.init_weights();models.append(m)
    opts=[torch.optim.SGD(m.parameters(),lr=.01) for m in models]
    losses=[]
    for step in range(6):
        candidate.K2_EMBEDDING=step>=3
        torch.manual_seed(100+step)
        ids=torch.randint(64,(2,8),dtype=index_dtype);targets=torch.randint(64,(2,8))
        current=[]
        for m,opt in zip(models,opts):
            opt.zero_grad(set_to_none=True)
            loss=m(ids,targets);loss.backward();current.append(loss.detach())
        exact(*current)
        for pa,pb in zip(models[0].parameters(),models[1].parameters()):
            if pa.grad is None or pb.grad is None:require(pa.grad is pb.grad,'GRAD_PRESENCE')
            else:exact(pa.grad,pb.grad)
        for opt in opts:opt.step()
        for name,value in models[0].state_dict().items():exact(value,models[1].state_dict()[name])
        losses.append(float(current[0]))
    receipts.append({'dtype':str(index_dtype),'off3':'BIT_IDENTICAL','on3':'BIT_IDENTICAL_INTERPRETED_NKI',
                     'losses':losses,'optimizer':'SGD .01','shape':'8 layers/width64/context8/batch2'})
    before=len(kernel_events)
    for m in models:m.eval()
    with torch.no_grad():exact(models[0](ids),models[1](ids))
    require(len(kernel_events)==before,'EVAL_CALLED_CUSTOM_KERNEL')

# The AOT test uses the actual model/helper, not a toy substitution. Captured
# graphs are retained so exclusion can be inspected independently later.
candidate.K2_EMBEDDING=True
torch.manual_seed(123)
model=candidate.GPT(candidate.GPTConfig(sequence_len=8,vocab_size=64,n_layer=8,n_head=2,n_embd=64,mlp_ratio=2))
model.init_weights();model.train()
eager_reference=copy.deepcopy(model)
graphs=[]


def capture(kind):
    def compile_graph(gm, example_inputs):
        code=gm.code
        require('trainium_k2' not in code and '_k2_' not in code,'K2_OP_IN_AOT_GRAPH')
        graphs.append({'kind':kind,'code':code,'nodes':len(list(gm.graph.nodes))})
        return make_boxed_func(gm.forward)
    return compile_graph


compiled=torch.compile(model,backend=aot_autograd(fw_compiler=capture('forward'),bw_compiler=capture('backward')),dynamic=False)
torch.manual_seed(501)
ids=torch.randint(64,(2,8),dtype=torch.int32);targets=torch.randint(64,(2,8))
before=len(kernel_events)
for _ in range(3):
    model.zero_grad(set_to_none=True)
    eager_reference.zero_grad(set_to_none=True)
    expected=eager_reference(ids,targets)
    expected.backward()
    loss=compiled(ids,targets)
    require(bool(torch.isfinite(loss)),'NONFINITE_COMPILED_LOSS')
    loss.backward()
    exact(loss,expected)
    for actual,reference in zip(model.parameters(),eager_reference.parameters()):
        if actual.grad is None or reference.grad is None:require(actual.grad is reference.grad,'COMPILED_GRAD_PRESENCE')
        else:exact(actual.grad,reference.grad)
require(len(kernel_events)>before,'NO_EAGER_KERNEL_BACKWARD')
require(any(g['kind']=='backward' for g in graphs),'NO_COMPILED_BACKWARD')
require(any(g['kind']=='forward' and g['nodes']>100 for g in graphs),'TRANSFORMER_NOT_COMPILED')
result={'status':'PASS','source_sha256':EXPECTED,'torch_version':torch.__version__,
        'kernel_helpers_ast_unchanged':True,'model_tests':receipts,
        'aot_graphs':[{'kind':g['kind'],'nodes':g['nodes']} for g in graphs],
        'eager_kernel_backward_calls_after_aot':len(kernel_events)-before,
        'kernel_absent_from_captured_forward_and_backward':True,
        'aot_vs_eager_losses_and_all_gradients':'BIT_IDENTICAL_3_PASSES',
        'reused_ids_two_tables_two_accumulation_microsteps':'BIT_IDENTICAL_BOTH_INDEX_DTYPES',
        'native_measured':False,'independent_review':False}
(ROOT/'CPU_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'AOT_GRAPHS.json').write_text(json.dumps(graphs,indent=2)+'\n')
print(json.dumps(result))
