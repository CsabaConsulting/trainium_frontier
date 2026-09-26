#!/usr/bin/env python3
"""Small CPU-only isolation of exact checker wrapper, NOT the native NKI kernel."""
import ast
import hashlib
import json
import subprocess
import types
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
source = subprocess.check_output(['git', 'show', '9ce2e6a037212aaa76bd85b9b4ab7d311a481757:research/v2/lanes/k2-symmetric-consumer/check_native.py'],
                                 cwd='/home/poojith-devan/Desktop/Trainium')
pin = hashlib.sha256(source).hexdigest()
if pin != '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850':
    raise RuntimeError('source mismatch')
main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
node = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'make_wrapper')


def cpu_compile(fn, *, backend, fullgraph, dynamic):
    if (backend, fullgraph, dynamic) != ('neuron', True, False):
        raise RuntimeError('unexpected compile contract')
    return torch.compile(fn, backend='aot_eager', fullgraph=True, dynamic=False)


namespace = {'torch': types.SimpleNamespace(autograd=torch.autograd, compile=cpu_compile), 'F': F}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact-checker-wrapper>', 'exec'), namespace)


def tiny_cpu_kernel(g, counts, starts, positions, maximums):
    # Test substitute: one occurrence per row, no duplicates, permuted token order.
    # All four saved arguments participate; NOT an NKI emulator or SDK claim.
    return g.index_select(0, positions.reshape(-1).long()).float() * counts.float() + starts.float()*0 + maximums.sum()*0


ids = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4], dtype=torch.int32)
inverse = torch.argsort(ids).reshape(-1, 1).to(torch.int32)
metadata = (torch.ones(8, 1, dtype=torch.int32), torch.arange(8, dtype=torch.int32).reshape(-1, 1),
            inverse, torch.ones(2, 1, dtype=torch.int32))
results = []
for dtype in (torch.bfloat16, torch.float32):
    eager, compiled = namespace['make_wrapper'](tiny_cpu_kernel, dtype)
    for input_name, g in [('zero_first', torch.zeros(8, 4, dtype=dtype)),
                          ('nonzero_after_zero', (torch.arange(32).reshape(8, 4)-16).to(dtype)),
                          ('different_nonzero', (torch.arange(32).reshape(8, 4)+1).to(dtype))]:
        base = torch.randn(8, 4, dtype=torch.float32)
        expected = g[inverse.reshape(-1).long()].float()
        for mode, fn in [('eager', eager), ('aot_eager', compiled)]:
            weight = base.detach().clone().requires_grad_()
            output = fn(weight, ids, *metadata)
            output.backward(g)
            forward_ok = torch.equal(output.detach(), base[ids.long()].to(dtype))
            backward_ok = weight.grad is not None and torch.equal(weight.grad.view(torch.uint8), expected.view(torch.uint8))
            if not forward_ok or not backward_ok:
                raise RuntimeError(str((dtype, input_name, mode, forward_ok, backward_ok)))
            results.append(dict(dtype=str(dtype), input=input_name, mode=mode, forward_exact=True,
                                backward_byte_exact=True, grad_nonzero=int(torch.count_nonzero(weight.grad))))
print(json.dumps(dict(status='CPU_REFERENCE_ONLY_PASS', torch=torch.__version__, checker_sha256=pin,
    wrapper_ast_unchanged=True, backend_substitution='neuron -> aot_eager via compile proxy',
    kernel_substitution='explicit small unique-row CPU gather; NOT native NKI',
    shape=[8, 4], tensor_data_under_bytes=1048576, cases=results,
    meaning='No generic wrapper/dtype-history failure reproduced with this CPU substitute; cannot distinguish native capture, lowering, cache or device effects.'), indent=2))
