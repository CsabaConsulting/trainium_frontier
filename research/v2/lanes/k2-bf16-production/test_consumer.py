"""Own tiny CPU interpretation of the actual consumer body, NOT an NKI simulator.

Only the six geometry constants are AST-replaced: V8/T16/C8/P4/F4/G2.
The fake instruction layer models the used DMA/elementwise semantics and tracks
writes. It cannot establish SDK legality, compiler ordering, or native rounding.
"""
import ast
import contextlib
import hashlib
import json
import pathlib
import sys
import types

import torch
from prototype import metadata_model
from test_model import bits, bf16, expected_cpu, require

ROOT = pathlib.Path(__file__).resolve().parent


class Tensor:
    def __init__(self, data, writes=None):
        self.data, self.writes = data, writes
        self.shape, self.dtype = tuple(data.shape), data.dtype

    def __getitem__(self, index):
        return Tensor(self.data[index], None if self.writes is None else self.writes[index])

    def ap(self, pattern, offset=0, vector_offset=None, indirect_dim=None):
        if vector_offset is not None:
            require(indirect_dim == 0, 'INTERPRETER_INDIRECT_DIM')
            return types.SimpleNamespace(base=self, pattern=pattern, offset=offset,
                                         indices=vector_offset.data.flatten())
        shape = tuple(x[1] for x in pattern)
        strides = tuple(x[0] for x in pattern)
        return Tensor(torch.as_strided(self.data, shape, strides, self.data.storage_offset() + offset))


class Instructions:
    engine = types.SimpleNamespace(vector='vector')
    oob_mode = types.SimpleNamespace(skip='skip')
    dge_mode = types.SimpleNamespace(swdge='swdge')

    def __init__(self):
        self.rounds = []
        self.add_count = 0

    def dma_copy(self, dst, src, **kwargs):
        if isinstance(src, Tensor):
            dst.data.copy_(src.data)
            if dst.writes is not None:
                dst.writes.add_(1)
            return
        require(kwargs == dict(oob_mode='skip', dge_mode='swdge'), 'INDIRECT_MODE')
        require(len(src.pattern) == 2 and src.pattern[0][1] == dst.shape[0], 'INDIRECT_SHAPE')
        flat = src.base.data.flatten()
        for lane, index in enumerate(src.indices):
            index = int(index)
            if not 0 <= index < src.base.shape[0]:
                continue
            offsets = index * src.pattern[0][0] + src.offset + torch.arange(dst.shape[1]) * src.pattern[1][0]
            require(bool(((offsets >= 0) & (offsets < flat.numel())).all()), 'INDIRECT_BOUNDS')
            dst.data[lane].copy_(flat[offsets])

    def register_alloc(self, value):
        return [value]

    def register_load(self, dst, src):
        dst[0] = int(src.data.item())

    def memset(self, dst, value):
        dst.data.fill_(value)

    def tensor_scalar(self, dst, data, op0, operand0, **kwargs):
        dst.data.copy_(op0(data.data, operand0))

    def tensor_tensor(self, dst, data1, data2, op, **kwargs):
        self.add_count += 1
        dst.data.copy_(op(data1.data, data2.data))

    def tensor_copy(self, dst, src):
        dst.data.copy_(src.data)


class Language(types.ModuleType):
    float32, int32, bfloat16 = torch.float32, torch.int32, torch.bfloat16
    shared_hbm, sbuf = 'hbm', 'sbuf'
    greater, add = staticmethod(torch.gt), staticmethod(torch.add)

    def __init__(self, instructions):
        super().__init__('nki.language')
        self.pid, self.instructions = 0, instructions
        self.max_elements = 0

    def ndarray(self, shape, dtype, buffer):
        require(torch.tensor(shape).prod().item() <= 128, 'LAPTOP_TENSOR_CAP')
        self.max_elements = max(self.max_elements, int(torch.tensor(shape).prod().item()))
        poison = float('nan') if dtype.is_floating_point else -19088743
        data = torch.full(shape, poison, dtype=dtype)
        return Tensor(data, torch.zeros(shape, dtype=torch.int32) if buffer == self.shared_hbm else None)

    def full(self, shape, value, dtype, buffer):
        out = self.ndarray(shape, dtype, buffer)
        out.data.fill_(value)
        return out

    def where(self, condition, a, b, dtype):
        return Tensor(torch.where(condition.data.bool(), a.data, b.data).to(dtype))

    def num_programs(self):
        return 2

    def program_id(self, axis):
        require(axis == 0, 'PROGRAM_AXIS')
        return self.pid

    def dynamic_range(self, register):
        self.instructions.rounds.append(register[0])
        return range(register[0])


class Jit:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        require(grid == 2, 'OWNER_GRID')
        return self.function


@contextlib.contextmanager
def interpreted_kernel():
    text = (ROOT / 'consumer_nki.py').read_text()
    tree = ast.parse(text)
    replaced = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = tuple(x.id for x in node.targets[0].elts)
            if names == ('vocab', 'tokens', 'channels'):
                node.value = ast.Tuple(elts=[ast.Constant(x) for x in (8, 16, 8)], ctx=ast.Load())
                replaced.append(names)
            if names == ('partition', 'features', 'groups'):
                node.value = ast.Tuple(elts=[ast.Constant(x) for x in (4, 4, 2)], ctx=ast.Load())
                replaced.append(names)
    require(len(replaced) == 2, 'GEOMETRY_REPLACEMENT')
    isa = Instructions()
    lang = Language(isa)
    nki = types.ModuleType('nki')
    nki.jit, nki.isa, nki.language = Jit, isa, lang
    injected = {'nki': nki, 'nki.isa': isa, 'nki.language': lang}
    previous = {key: sys.modules.get(key) for key in injected}
    sys.modules.update(injected)
    namespace = {}
    try:
        exec(compile(ast.fix_missing_locations(tree), 'tiny_geometry_consumer', 'exec'), namespace)
        yield namespace['build_kernel'](enabled=True), lang, isa
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def run_case(ids, gradient):
    require(len(ids) == 16 and tuple(gradient.shape) == (16, 8), 'FIXTURE_SHAPE')
    meta = metadata_model(ids, 8, group=4)
    args = [Tensor(torch.tensor(x, dtype=torch.int32).reshape(-1, 1))
            for x in (meta.counts, meta.starts, meta.positions, meta.maximums)]
    outputs, schedules = [], []
    with interpreted_kernel() as (kernel, lang, isa):
        for owner in (0, 1):
            lang.pid = owner
            isa.rounds = []
            result = kernel(Tensor(gradient), *args)
            first = owner * 4
            owned = result.writes[:, first:first + 4]
            unowned = result.writes[:, 4:8] if owner == 0 else result.writes[:, 0:4]
            require(bool((owned == 1).all()) and bool((unowned == 0).all()), 'WRITE_OWNERSHIP')
            outputs.append(result.data[:, first:first + 4].clone())
            schedules.append(list(isa.rounds))
        require(schedules[0] == schedules[1] == list(meta.maximums), 'ASYMMETRIC_SCHEDULE')
        require(isa.add_count == 2 * sum(meta.maximums), 'LOOP_ADDITION_COUNT')
        # Reinvoke the SAME callable with changed IDs/tangents, preserving old
        # outputs. This tests source-level fresh outputs and runtime arguments.
        old_output_bits = bits(result.data[:, 4:8])
        replay_ids = list(reversed(ids))
        replay_meta = metadata_model(replay_ids, 8, group=4)
        replay_args = [Tensor(torch.tensor(x, dtype=torch.int32).reshape(-1, 1))
                       for x in (replay_meta.counts, replay_meta.starts,
                                 replay_meta.positions, replay_meta.maximums)]
        replay_gradient = gradient.flip(0).clone()
        lang.pid = 1
        repeated = kernel(Tensor(replay_gradient), *replay_args)
        require(result.data.data_ptr() != repeated.data.data_ptr(), 'OUTPUT_ALIAS')
        require(bits(result.data[:, 4:8]) == old_output_bits, 'PRIOR_OUTPUT_MUTATED')
        replay_expected = expected_cpu(replay_ids, replay_gradient, 8)
        if bool(torch.isfinite(replay_expected).all()):
            require(bits(repeated.data[:, 4:8]) == bits(replay_expected[:, 4:8]), 'REPLAY_STALE')
        max_elements = lang.max_elements
    combined = torch.cat(outputs, dim=1)
    expected = expected_cpu(ids, gradient, 8)
    finite = bool(torch.isfinite(expected).all())
    if finite:
        require(bits(combined) == bits(expected), 'FINITE_EXACTNESS')
    else:
        require(torch.equal(torch.isnan(combined), torch.isnan(expected)), 'NAN_CLASSIFICATION')
        require(torch.equal(torch.isinf(combined), torch.isinf(expected)), 'INF_CLASSIFICATION')
        require(torch.equal(torch.signbit(combined[torch.isinf(combined)]),
                            torch.signbit(expected[torch.isinf(expected)])), 'INF_SIGN')
    return dict(verdict='EXACT_FINITE' if finite else 'FAIL_NONFINITE_DIAGNOSTIC',
                ids=ids, input_bf16=[f'{int(x) & 65535:04x}' for x in gradient.view(torch.int16).flatten()],
                output_bits=bits(combined), expected_bits=bits(expected), schedules=schedules,
                same_callable_replay_fresh_output=True,
                max_interpreter_tensor_elements=max_elements)


def main():
    fixtures = []
    ordinary = torch.arange(128, dtype=torch.float32).reshape(16, 8).to(torch.bfloat16)
    for name, ids in (
        ('balanced', list(range(8)) * 2),
        ('interleaved', [7, 0, 3, 0, 7, 3, 0, 7] * 2),
        ('all_one_row', [7] * 16),
        ('empty_first_group', [4, 7] * 8),
        ('boundary_and_uneven', [0, 7, 0, 1] + [7] * 12),
    ):
        fixtures.append(dict(name=name, **run_case(ids, ordinary)))
    for name, values in (
        ('positive_min_subnormal', [0x0001]),
        ('negative_min_subnormal', [0x8001]),
        ('both_min_subnormals', [0x0001, 0x8001]),
        ('positive_normal_cancellation', [0x0081, 0x8080]),
        ('negative_normal_cancellation', [0x8081, 0x0080]),
        ('large_order', [0x4b80, 0x3f80, 0xcb80]),
        ('small_order', [0x3f80, 0x3380, 0xbf80]),
        ('large_cancel_then_subnormal', [0x4b80, 0xcb80, 0x0001]),
        ('zero_signs', [0x8000, 0x0000]),
        ('positive_overflow', [0x7f7f, 0x7f7f]),
        ('negative_overflow', [0xff7f, 0xff7f]),
        ('infinity_cancel', [0x7f80, 0xff80]),
        ('nan_payload', [0x7fc1]),
    ):
        raw = [0] * 16
        ids = [4] * 16
        # Repeated row spans token offsets 0,5,10, not adjacent tile sums.
        for index, value in enumerate(values):
            raw[index * 5] = value
            ids[index * 5] = 0
        gradient = bf16(raw).expand(-1, 8).clone()
        fixtures.append(dict(name=name, **run_case(ids, gradient)))
    # Real data arguments change between calls; same source, no cached result.
    changed = ordinary.flip(0).clone()
    replay = run_case([0, 7] * 8, changed)
    fixtures.append(dict(name='changed_ids_and_tangent_replay', **replay))
    namespace = {}
    exec(compile((ROOT / 'consumer_nki.py').read_text(), 'original_consumer', 'exec'), namespace)
    try:
        namespace['build_kernel']()
    except RuntimeError as exc:
        disabled = str(exc)
    else:
        raise RuntimeError('DEFAULT_ENABLED')
    require('nki' not in sys.modules, 'NATIVE_IMPORT_LEAK')
    result = dict(status='IMPLEMENTER_TINY_BODY_INTERPRETATION_ONLY', fixtures=fixtures,
                  geometry=dict(production=[8192, 32768, 1024, 128, 512, 64], tiny=[8, 16, 8, 4, 4, 2]),
                  consumer_sha256=hashlib.sha256((ROOT / 'consumer_nki.py').read_bytes()).hexdigest(),
                  test_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  default_disabled_error=disabled, native_compiled=False,
                  torch=torch.__version__, python=sys.version)
    receipt = ROOT / ('consumer-raw-optimized.json' if sys.flags.optimize else 'consumer-raw-normal.json')
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], fixtures=len(fixtures), receipt=str(receipt))))


if __name__ == '__main__':
    main()
