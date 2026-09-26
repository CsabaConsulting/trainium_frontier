"""Independent, tiny CPU review. Inputs are only the supplied diff/spec/receipt.

The NKI interpreter below models arithmetic and addressing; it does not compile
NKI, simulate Neuron, or prove native API support or hardware rounding.
"""
import ast
import collections
import hashlib
import json
import pathlib
import sys
import textwrap
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED = {
    "SPEC.md": "b5547916655fe0b5643cf51069e3e4ecd6f0c648f715ea356ad832c2761e5a3e",
    "parent.diff": "7b5bff632a616b84ae94f3d9fe17b1b5fc88802485bfdab1728300b9d8e3afea",
    "cpu-result.json": "4d50830392a7735298e08438542736a191f01ca0d4cc795a08f0c7432bbf11a1",
}
for filename, expected in EXPECTED.items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
diff = (ROOT / "parent.diff").read_text()
first_hunk = diff.split("@@", 2)[2].split("@@", 1)[0]
source = "\n".join(line[1:] for line in first_hunk.splitlines() if line.startswith("+"))
module = ast.parse(source)
namespace = {"torch": torch, "F": F}
exec(compile(module, "<actual-added-first-hunk>", "exec"), namespace)


def int32(x):
    y = x & 0xffffffff
    return y - 0x100000000 if y & 0x80000000 else y


def hash_oracle(rows):
    result = [[], []]
    for row in rows:
        b, t = [], []
        for i, token in enumerate(row):
            prev = row[i-1] if i else 8192
            prev2 = row[i-2] if i >= 2 else 8192
            b.append(int32(int32(prev*120193) ^ int32(token*128201)) % 524287)
            t.append(int32(int32(prev2*144203) ^ int32(prev*152197) ^ int32(token*160201)) % 524287)
        result[0].append(b)
        result[1].append(t)
    return tuple(torch.tensor(x, dtype=torch.int64) for x in result)


hash_cases = []
generator = torch.Generator().manual_seed(42)
for length in (1, 2, 3, 4, 17, 97):
    rows = torch.randint(0, 8193, (3, length), generator=generator, dtype=torch.int64)
    rows[0, 0] = 8192
    actual = namespace["_p2_hashes"](rows)
    expected = hash_oracle(rows.tolist())
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    hash_cases.append({"shape": list(rows.shape), "exact": True})
boundary = torch.tensor([[0, 8191, 8192, 2**31-1, -2**31, -1, 65535]], dtype=torch.int64)
assert all(torch.equal(a, b) for a, b in zip(namespace["_p2_hashes"](boundary), hash_oracle(boundary.tolist())))
causal_input = torch.randint(0, 8193, (2, 17), generator=generator)
causal_changed = causal_input.clone()
causal_changed[0, 9:] = (causal_changed[0, 9:] + 1) % 8193
causal_changed[1] = (causal_changed[1] + 1) % 8193
assert all(torch.equal(a[0, :9], b[0, :9]) for a, b in zip(namespace["_p2_hashes"](causal_input), namespace["_p2_hashes"](causal_changed)))


class Tile:
    def __init__(self, array, writes=None, dtype=None):
        self.array = array
        self.writes = writes
        self.dtype = dtype if dtype is not None else array.dtype

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, key):
        return Tile(self.array[key], None if self.writes is None else self.writes[key], self.dtype)

    def ap(self, **kwargs):
        return Address(self, **kwargs)


class Address:
    def __init__(self, tile, pattern, offset=0, vector_offset=None, indirect_dim=None):
        self.tile, self.pattern, self.offset = tile, pattern, offset
        self.vector_offset, self.indirect_dim = vector_offset, indirect_dim

    def indices(self):
        (stride0, count0), (stride1, count1) = self.pattern
        first = np.arange(count0, dtype=np.int64)[:, None]
        if self.vector_offset is not None:
            assert self.indirect_dim == 0
            first = self.vector_offset.array.astype(np.int64)
        return self.offset + first*stride0 + np.arange(count1)[None, :]*stride1

    def array(self):
        return self.tile.array.reshape(-1)[self.indices()]


class Register:
    value = 0


class Interpreter:
    def __init__(self):
        self.program = 0
        self.output = None
        self.store_history = []
        self.nl = types.ModuleType("nki.language")
        self.nisa = types.ModuleType("nki.isa")
        self.nki = types.ModuleType("nki")
        nl, nisa = self.nl, self.nisa
        nl.float32, nl.int32, nl.bfloat16 = np.float32, np.int32, "BF16"
        nl.shared_hbm, nl.sbuf = "shared_hbm", "sbuf"
        nl.add, nl.greater = np.add, np.greater
        nl.num_programs, nl.program_id = lambda: 2, lambda axis: self.program
        nl.dynamic_range = lambda n: range(n.value if isinstance(n, Register) else n)
        nl.ndarray = self.ndarray
        nl.full = lambda shape, value, dtype, buffer: Tile(np.full(shape, value, dtype=dtype))
        nl.where = lambda cond, a, b, dtype: Tile(np.where(self.read(cond), self.read(a), self.read(b)).astype(dtype))
        nisa.engine = types.SimpleNamespace(vector="vector")
        nisa.oob_mode = types.SimpleNamespace(skip="skip")
        nisa.dge_mode = types.SimpleNamespace(swdge="swdge")
        nisa.register_alloc = lambda value: Register()
        nisa.register_load = lambda dst, src: setattr(dst, "value", int(src.array.item()))
        nisa.memset = lambda dst, value: dst.array.fill(value)
        nisa.tensor_scalar = lambda dst, data, op0, operand0, engine: np.copyto(dst.array, op0(data.array, operand0), casting="unsafe")
        nisa.tensor_tensor = lambda dst, data1, data2, op, engine: np.copyto(dst.array, op(data1.array, data2.array), casting="unsafe")
        nisa.tensor_copy = lambda dst, src: np.copyto(dst.array, src.array)
        nisa.dma_copy = self.dma_copy
        self.nki.jit = self.jit
        self.nki.isa, self.nki.language = nisa, nl

    @staticmethod
    def read(x):
        return x.array() if isinstance(x, Address) else x.array

    def ndarray(self, shape, dtype, buffer):
        if buffer == "shared_hbm":
            if self.output is None:
                self.output = Tile(np.full(shape, np.nan, dtype=dtype), np.zeros(shape, dtype=np.int64))
            return self.output
        return Tile(np.full(shape, 73, dtype=dtype))

    def dma_copy(self, dst, src, **kwargs):
        if isinstance(src, Address):
            indices = src.indices()
            flat = src.tile.array.reshape(-1)
            valid = (indices >= 0) & (indices < flat.size)
            assert kwargs.get("oob_mode") == "skip"
            dst.array[valid] = flat[indices[valid]]
        else:
            np.copyto(dst.array, src.array)
        if dst.writes is not None:
            dst.writes += 1

    def jit(self, function):
        owner = self
        class Grid:
            def __getitem__(self, count):
                assert count == 2
                def run(*args):
                    owner.output = None
                    args = [Tile(arg.float().numpy() if arg.dtype == torch.bfloat16 else arg.numpy(),
                                 dtype="BF16" if arg.dtype == torch.bfloat16 else arg.numpy().dtype) for arg in args]
                    for program in range(count):
                        owner.program = program
                        result = function(*args)
                    assert np.all(result.writes == 1), result.writes
                    assert np.isfinite(result.array).all()
                    owner.store_history.append(int(result.writes.sum()))
                    return torch.from_numpy(result.array.copy())
                return run
        return Grid()


interpreter = Interpreter()
sys.modules.update({"nki": interpreter.nki, "nki.isa": interpreter.nisa, "nki.language": interpreter.nl})
kernel_ast = ast.parse(ast.unparse(next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_k2_table_build_kernel")))
geometry_edits = []
for node in ast.walk(kernel_ast):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
        names = [part.id for part in node.targets[0].elts]
        replacement = {( "vocab", "tokens", "channels"): (16, 8, 4),
                       ("partition", "features", "groups"): (4, 2, 4)}.get(tuple(names))
        if replacement is not None:
            geometry_edits.append({"names": names, "values": list(replacement)})
            node.value = ast.Tuple(elts=[ast.Constant(x) for x in replacement], ctx=ast.Load())
assert len(geometry_edits) == 2
ast.fix_missing_locations(kernel_ast)
exec(compile(kernel_ast, "<actual-kernel-tiny-geometry>", "exec"), namespace)
kernel = namespace["_k2_table_build_kernel"](enabled=True)
lookup = namespace["_k2_table_build_registered_lookup"](kernel=kernel, nki_op_factory=torch.library.custom_op)


def metadata(ids):
    flat = ids.reshape(-1)
    counts = torch.bincount(flat, minlength=16).to(torch.int32)
    starts = counts.cumsum(0).to(torch.int32) - counts
    positions = torch.argsort(flat, stable=True).to(torch.int32)
    maximums = counts.reshape(4, 4).max(1).values
    return tuple(x.reshape(-1, 1) for x in (counts, starts, positions, maximums))


ids = torch.tensor([[1, 2, 1, 3, 1, 2, 3, 1]], dtype=torch.int64)
cases = {
    "historical_bf16_cancellation": [1, 2**-8, -1, 0],
    "fp32_order_loses_unit": [2**25, 1, -(2**25), 0],
    "fp32_order_keeps_unit": [2**25, -(2**25), 1, 0],
    "positive_bf16_subnormal": [2**-133, 2**-133, -2**-133, 0],
    "negative_bf16_subnormal": [-2**-133, -2**-133, 2**-133, 0],
    "normal_cancellation": [1, -1, 2**-8, -2**-9],
}
results = []
for name, values in cases.items():
    actual_weight = torch.arange(64, dtype=torch.float32).reshape(16, 4).div(101).requires_grad_()
    expected_weight = actual_weight.detach().clone().requires_grad_()
    tangent = torch.zeros((1, 8, 4), dtype=torch.bfloat16)
    for position, value in zip((0, 2, 4, 7), values):
        tangent[0, position] = torch.tensor([value, -value, value, -value], dtype=torch.bfloat16)
    tangent[0, 1] = 0.75
    tangent[0, 3] = -0.25
    tangent[0, 5] = 0.125
    tangent[0, 6] = 0.5
    for microstep in range(2):
        actual = lookup(actual_weight, ids, *metadata(ids))
        expected = F.embedding(ids, expected_weight).to(torch.bfloat16)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        actual.backward(tangent)
        expected.backward(tangent)
        assert actual_weight.grad.dtype == torch.float32
        assert torch.equal(actual_weight.grad.view(torch.int32), expected_weight.grad.view(torch.int32)), name
        assert torch.all(actual_weight.grad[[0, 4, 15]] == 0)
    results.append({"case": name, "two_microsteps_bit_exact_to_torch": True,
                    "row1_gradient": actual_weight.grad[1].tolist()})

# Nonzero rows in several runtime groups detect group-offset mistakes that
# the cancellation fixtures, concentrated in group zero, cannot expose.
scattered_ids = torch.tensor([[14, 5, 14, 7, 5, 8, 7, 14]], dtype=torch.int64)
scattered_weight = torch.arange(64, dtype=torch.float32).reshape(16, 4).div(101).requires_grad_()
scattered_reference = scattered_weight.detach().clone().requires_grad_()
scattered_tangent = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4).div(16).to(torch.bfloat16)
lookup(scattered_weight, scattered_ids, *metadata(scattered_ids)).backward(scattered_tangent)
F.embedding(scattered_ids, scattered_reference).to(torch.bfloat16).backward(scattered_tangent)
assert torch.equal(scattered_weight.grad.view(torch.int32), scattered_reference.grad.view(torch.int32))

rms_start = diff.index("+    @torch.no_grad()\n+    def _rmsprop_step")
rms_chunk = diff[rms_start:].split("@@", 1)[0]
rms_source = textwrap.dedent("\n".join(line[1:] for line in rms_chunk.splitlines() if line.startswith("+")))
exec(compile(ast.parse(rms_source), "<actual-added-rms-method>", "exec"), namespace)
class OptimizerFixture:
    state = collections.defaultdict(dict)
    _rmsprop_step = namespace["_rmsprop_step"]
    def _device_scalar(self, group, name, value, parameter):
        return torch.tensor(value, dtype=parameter.dtype)

optimizer = OptimizerFixture()
p = torch.tensor([[0.25, -0.5], [1, 0]], dtype=torch.float32, requires_grad=True)
expected_p = p.detach().clone()
square_average = torch.zeros_like(p)
group = dict(params=[p], weight_decay=0.0, betas=(0.0, .999), lr=.2598076211353316, eps=1e-10)
rms_errors = []
for step, gradient in enumerate(([[1, 2], [0, -4]], [[.5, -.25], [1, 0]], [[-2, 1], [-.5, 3]]), 1):
    p.grad = torch.tensor(gradient, dtype=torch.float32)
    optimizer._rmsprop_step(group)
    # Independent recurrence uses explicit multiplication/addition, so small
    # rounding differences from lerp are measured with a stated tolerance.
    square_average = .999*square_average + .001*p.grad.square()
    expected_p -= group["lr"]*p.grad / ((square_average/(1-.999**step)).sqrt() + group["eps"])
    error = float((p.detach()-expected_p).abs().max())
    assert torch.allclose(p, expected_p, atol=1e-7, rtol=1e-6), error
    assert optimizer.state[p]["exp_avg_sq"].dtype == torch.float32
    assert optimizer.state[p]["step"] == step
    rms_errors.append(error)

receipt = {
    "status": "PASS_TINY_CPU_INDEPENDENT_CHECKS",
    "input_sha256": EXPECTED,
    "python": sys.executable,
    "torch": torch.__version__,
    "numpy": np.__version__,
    "hash_production_helper": {"cases": hash_cases, "int32_boundary_case_exact": True,
                               "same_shape_prefix_and_batch_independence_exact": True},
    "consumer": {"geometry_ast_edits_only": geometry_edits, "cases": results,
                 "nonzero_rows_in_multiple_runtime_groups_exact": True,
                 "output_elements_written_exactly_once_per_call": interpreter.store_history},
    "rms_independent_recurrence": {"steps": 3, "max_abs_errors": rms_errors, "atol": 1e-7, "rtol": 1e-6},
    "limitations": ["Tiny Python interpreter only: not NKI simulation, compilation, or native execution.",
                    "Interpreter assumes AP indirect row replacement and DMA BF16-to-FP32 semantics; native support unproved.",
                    "Metadata is an independent tiny Torch producer, not the unavailable unchanged K2 producer.",
                    "No full model, optimizer ownership, collective, checkpoint or throughput execution.",
                    "No full candidate source available: advertised train.py SHA cannot be independently recomputed."],
}
print(json.dumps(receipt, indent=2))
