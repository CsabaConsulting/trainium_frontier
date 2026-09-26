"""Own tiny CPU interpretation of the actual consumer body, NOT an NKI simulator.

Only the six geometry constants are AST-replaced for the requested small case.
The model test uses V64/T16/C64/P4/F32/G16; scalar fixtures use V8/T16/C8.
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
def require(condition, message):
    if not condition:
        raise AssertionError(message)

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
        require(torch.tensor(shape).prod().item() <= 65536, 'LAPTOP_TENSOR_CAP')
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
def interpreted_kernel(vocab=8, tokens=16, channels=8, partition=4):
    tree = ast.parse((ROOT / 'train.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_k2_build_kernel')
    function.name = 'build_kernel'
    tree = ast.Module(body=[function], type_ignores=[])
    replaced = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = tuple(x.id for x in node.targets[0].elts)
            if names == ('vocab', 'tokens', 'channels'):
                node.value = ast.Tuple(elts=[ast.Constant(x) for x in (vocab, tokens, channels)], ctx=ast.Load())
                replaced.append(names)
            if names == ('partition', 'features', 'groups'):
                node.value = ast.Tuple(elts=[ast.Constant(x) for x in (partition, channels//2, vocab//partition)], ctx=ast.Load())
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
