"""Execute unchanged original Function AST from supplied context, on CPU only."""
import ast
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('own_review', ROOT / 'output/review_cpu.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
review.authenticate()
review.disabled_test()
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)

namespace = dict(torch=torch, struct=struct, require=review.require, sha=review.sha)
own_tree = ast.parse((ROOT / 'output/review_cpu.py').read_text())
own_main = next(n for n in own_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
needed = {'raw', 'fingerprint', 'exact', 'f32', 'oracle', 'metadata', 'ordered_consumer'}
helpers = [n for n in own_main.body if isinstance(n, ast.FunctionDef) and n.name in needed]
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(ROOT / 'output/review_cpu.py'), 'exec'), namespace)

context_bytes = (ROOT / 'source-context/check_native.py').read_bytes()
context = ast.parse(context_bytes)
wrapper = next(n for n in ast.walk(context) if isinstance(n, ast.FunctionDef) and n.name == 'make_wrapper')
members = [n for n in wrapper.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
review.require(len(members) == 2, 'original class and enclosing function only')
review.emit('context_identity', source_sha=review.sha(context_bytes),
            stated_historical_original_sha='bd2bca7c29615df599094d3c618f5cdf0e4f4381e6d19213dc022691214ce788',
            source_identity_matches_historical=False,
            exact_member_ast_sha=review.sha(ast.dump(ast.Module(body=members, type_ignores=[])).encode()),
            native_backend_never_executed=True)

count = 0
for dtype in (torch.float32, torch.bfloat16):
    ns = dict(torch=torch, F=F, kernel=namespace['ordered_consumer'], output_dtype=dtype)
    exec(compile(ast.Module(body=members, type_ignores=[]), str(ROOT / 'source-context/check_native.py'), 'exec'), ns)
    original = ns['enclosing']
    routes = [('supplied_Function_eager', original),
              ('supplied_Function_AOT', torch.compile(original, backend='aot_eager', fullgraph=True, dynamic=False))]
    for index, ids_list in enumerate([list(range(8)), list(range(8)), [7, 0, 7, 0, 3, 7, 3, 0]]):
        ids = torch.tensor(ids_list, dtype=torch.int32)
        values = torch.arange(1, 33, dtype=dtype).reshape(8, 4) + index
        tangent = values.t().contiguous().t()
        meta = namespace['metadata'](ids_list)
        expected = namespace['oracle'](ids_list, tangent)
        for route, function in routes:
            weight = (torch.arange(32, dtype=torch.float32).reshape(8, 4) / 7).requires_grad_()
            seven = [weight, ids, *meta, tangent]
            before = [namespace['fingerprint'](t) for t in seven]
            out = function(weight, ids, *meta)
            grad, = torch.autograd.grad(out, weight, tangent)
            after = [namespace['fingerprint'](t) for t in seven]
            review.require(before == after, 'source Function input mutation')
            review.require(namespace['exact'](grad, expected), 'source Function byte mismatch')
            review.require(namespace['exact'](out, F.embedding(ids, weight).to(dtype)), 'source Function output mismatch')
            count += 1
            review.emit('exact_context_route', route=route, fixture=index, dtype=str(dtype),
                        seven_inputs_before=before, seven_inputs_after=after,
                        actual=namespace['fingerprint'](grad), expected=namespace['fingerprint'](expected))

review.require(not any(n == 'nki' or n.startswith(('nki.', 'torch_neuronx')) for n in sys.modules), 'native imported')
review.emit('context_summary', status='PASS_SUPPLIED_CONTEXT_CPU', exact_routes=count,
            torch=torch.__version__, threads=torch.get_num_threads(), native_tested=False)
