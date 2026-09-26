"""Reuse frozen FP32 collision/optimizer tests; test the new boundary with AOT.

Reduced synthetic CPU evidence only. No host, evaluator or training data access.
"""
import ast
import copy
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import types

import torch
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TABLE = HERE.parent/'tables-compact-k2'
PARENT = HERE.parent/'k2-outside-compile-rescue/train.py'
SOURCE = HERE/'train.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'ba5add518bfb93206313476b2cd929cd678b0ed63b0045d65ad547970a9b7ee7'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == '949f262c092c141f715bdd71532c42e0bcf502d926a1ee20bc885a2255062daa'
assert hashlib.sha256((TABLE/'train.py').read_bytes()).hexdigest() == '4b74ed48c772a5e82807d5509529704deaf1c8cd56692705efe9047ea7ae4a7b'
sys.path.insert(0, str(TABLE))
sys.path.insert(0, str(REPO/'workflow/tools/stub_prepare'))
arrow, parquet = types.ModuleType('pyarrow'), types.ModuleType('pyarrow.parquet')
arrow.parquet = parquet
sys.modules['pyarrow'], sys.modules['pyarrow.parquet'] = arrow, parquet
import test_tables as suite
suite.ROOT = HERE
suite.INDEX_DTYPE = getattr(torch, sys.argv[1])
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
a = suite.load('table_rescue_parent', PARENT)
b = suite.load('table_rescue_candidate', SOURCE)
nodes = [{n.name: ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n, (ast.FunctionDef,ast.ClassDef))}
         for p in (TABLE/'train.py', SOURCE)]
unchanged = ('_k2_build_kernel','_k2_prepare_metadata','_k2_build_registered_lookup',
             '_k2_table_build_kernel','_k2_table_build_registered_lookup','_p2_hashes',
             'MuonAdamW','OwnedOptimizer','load_for_eval','save_checkpoint')
for name in unchanged: assert nodes[0][name] == nodes[1][name], name
calls = []


def kernel(*args):
    from torch._subclasses.fake_tensor import FakeTensor
    assert not isinstance(args[0], FakeTensor) and not torch.compiler.is_compiling()
    calls.append(tuple(args[0].shape))
    return suite.cpu_kernel(*args)


lookup = b._k2_table_build_registered_lookup(kernel=kernel, nki_op_factory=torch.library.custom_op)
result = {'index_dtype': sys.argv[1], 'reference': suite.reference_probe(suite.INDEX_DTYPE),
          'hashes': suite.hashes(b), 'operations': suite.operations(b,lookup),
          'rms': suite.rms(b), 'clipping': suite.clipping(b),
          'unchanged_ast': list(unchanged)}
# The model fixture intentionally reduces table capacity; production-hash tests
# above must precede that fixture, as in the frozen parent suite.
with contextlib.redirect_stdout(sys.stderr):
    result['model'] = suite.models(a,b,lookup)
# AOT captures the transformer and projections, never either registered lookup.
torch.manual_seed(42)
model = b.GPT(b.GPTConfig(sequence_len=8,vocab_size=64,n_layer=8,n_head=2,n_embd=128,mlp_ratio=2,ngram_ve=True))
model.init_weights()
with torch.no_grad():
    for p in model.ngram_tables.values(): p.fill_(.125)
reference = copy.deepcopy(model)
graphs = []


def capture(graph, inputs):
    text = graph.code
    for token in ('k2_embedding','k2_table','nki','index_select','embedding.default','scatter'):
        assert token not in text.lower(), token
    graphs.append({'nodes': len(list(graph.graph.nodes)), 'code': text})
    return make_boxed_func(graph.forward)


compiled = torch.compile(model, backend=aot_autograd(fw_compiler=capture,bw_compiler=capture))
calls.clear()
losses = []
for step in range(3):
    torch.manual_seed(500+step)
    ids = torch.randint(64,(2,8),dtype=suite.INDEX_DTYPE)
    targets = torch.randint(64,(2,8))
    model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
    b.K2_EMBEDDING = False
    ref_loss = reference(ids, targets); ref_loss.backward()
    b.K2_EMBEDDING = True
    loss = compiled(ids, targets); loss.backward()
    suite.exact(loss,ref_loss)
    for p,q in zip(model.parameters(),reference.parameters()):
        if p.grad is None or q.grad is None: assert p.grad is q.grad
        else: suite.exact(p.grad,q.grad)
    losses.append(float(loss.detach()))
assert len(calls) >= 9 and len(graphs) >= 2
result.update(status='PASS_REDUCED_CPU', aot_vs_reference='EXACT_LOSS_AND_ALL_GRADIENTS_3_PASSES',
              losses=losses, eager_kernel_backward_calls=len(calls),
              captured_graph_nodes=[x['nodes'] for x in graphs], native_executed=False,
              torch_version=torch.__version__,
              input_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (SOURCE,PARENT,Path(__file__),TABLE/'train.py',TABLE/'test_tables.py',
                                      TABLE/'cpu_instruction_model.py',TABLE/'test_fp32_reference.py')},
              source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              scope='Reduced CPU/interpreted kernel, not full-shape Neuron compilation, speed or BPB')
print(json.dumps(result,indent=2))
