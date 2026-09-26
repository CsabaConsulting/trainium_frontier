"""Small FP32-table exactness, integration and parent-OFF identity checks."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import torch
import torch.nn.functional as F
from cpu_instruction_model import Tensor, interpreted_kernel
from test_fp32_reference import exact, run as reference_probe

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
INDEX_DTYPE = torch.int32


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def cpu_kernel(g, counts, starts, positions, maximums):
    features = g.shape[1]//2
    group = counts.shape[0]//maximums.shape[0]
    with interpreted_kernel(counts.shape[0], g.shape[0], g.shape[1], group) as (kernel, lang, isa):
        result = []
        for owner in (0, 1):
            lang.pid = owner
            out = kernel(Tensor(g), *(Tensor(t) for t in (counts, starts, positions, maximums)))
            result.append(out.data[:, owner*features:(owner+1)*features])
        return torch.cat(result, dim=1)


def hashes(m):
    # Independent Python integer oracle with explicit signed-int32 products/XOR.
    def i32(x):
        return (x + 2**31) % 2**32 - 2**31
    for ids in (torch.tensor([[0]]), torch.tensor([[8191, 0]]),
                torch.tensor([[0, 8191, 0, 8191], [8191, 8191, 1, 0]])):
        ids = ids.to(INDEX_DTYPE)
        b, t = m._p2_hashes(ids)
        br, tr = [], []
        for row in ids.tolist():
            rb, rt = [], []
            for j, cur in enumerate(row):
                p = row[j-1] if j else 8192
                pp = row[j-2] if j > 1 else 8192
                rb.append((i32(p*120193) ^ i32(cur*128201)) % 524287)
                rt.append((i32(pp*144203) ^ i32(p*152197) ^ i32(cur*160201)) % 524287)
            br.append(rb); tr.append(rt)
        exact(b, torch.tensor(br)); exact(t, torch.tensor(tr))
    return 'production_hashes_length1_2_4_and_signed_int32_exact'


def operations(m, lookup):
    fixtures = []
    for name, vals in [('bf16_old_witness', [1., 2**-8, -1.]),
                       ('fp32_large_order', [2**24, 1., -2**24]),
                       ('fp32_small_order', [1., 2**-24, -1.]),
                       ('subnormal_pos', [2**-133]), ('subnormal_neg', [-2**-133]),
                       ('normal_cancellation', [float.fromhex('0x1.02p-126'), -float.fromhex('0x1p-126')])]:
        p = torch.randn(8, 8, requires_grad=True)
        ref = p.detach().clone().requires_grad_()
        for step in range(2):
            ids = torch.tensor([[0, 3, 0, 3, 0, 3, 7, 7]*2], dtype=INDEX_DTYPE)
            g = torch.zeros(16, 8, dtype=torch.bfloat16)
            for j, value in enumerate(vals):
                g[j*2] = value
            meta = m._k2_prepare_metadata(ids, 8, 4)
            y, yr = lookup(p, *meta), F.embedding(ids, ref).bfloat16()
            exact(y, yr)
            y.backward(g.view_as(y)); yr.backward(g.view_as(yr))
            exact(p.grad, ref.grad)
            assert p.grad.dtype == torch.float32
        fixtures.append(name)
    return fixtures


def rms(m):
    p = torch.nn.Parameter(torch.zeros(8, 4))
    group = dict(kind='rmsprop', params=[p], lr=.2598076211353316,
                 betas=(0., .999), eps=1e-10, weight_decay=0., no_lr_decay=True)
    opt = m.MuonAdamW([group])
    ref, second = torch.zeros_like(p), torch.zeros_like(p)
    for step in range(1, 4):
        p.grad = torch.randn_like(p)
        second = torch.lerp(second, p.grad.square(), .001)
        denom = (second / (1-.999**step)).sqrt() + 1e-10
        ref = ref - (p.grad / denom) * torch.tensor(group['lr'], dtype=torch.float32)
        opt.step()
        exact(p, ref); exact(opt.state[p]['exp_avg_sq'], second)
        assert opt.state[p]['exp_avg_sq'].dtype == torch.float32
    # Every owner updates the full replicated table exactly once; fake collectives
    # test routing only, never claim distributed numerical/native evidence.
    for rank in range(4):
        table = torch.nn.Parameter(torch.zeros(8, 4))
        ordinary = torch.nn.Parameter(torch.ones(4, 4))
        rg = {**group, 'params': [table]}
        ag = dict(kind='adamw', params=[ordinary], lr=.01, betas=(.8,.95), eps=1e-10, weight_decay=0.)
        optimizer = m.MuonAdamW([ag, rg])
        with patch.object(m.dist, 'is_initialized', return_value=True), patch.object(m.dist, 'get_rank', return_value=rank), patch.object(m.dist, 'get_world_size', return_value=4), patch.object(m.dist, 'all_gather_into_tensor', side_effect=lambda out, inp: out.copy_(inp.repeat(4))):
            owned = m.OwnedOptimizer(optimizer)
            assert all(p is not table for p in owned.parameters)
            assert owned.groups[1]['params'] == [table]
            table.grad = torch.ones_like(table)
            ordinary.grad = torch.ones_like(ordinary)
            owned.step()
            assert optimizer.state[table]['step'] == 1
            assert bool((table != 0).all())
    return 'FP32_RMS_3_UPDATES_EXACT_AND_REPLICATED_ROUTING_4_FAKE_RANKS'


def clipping(m):
    model = torch.nn.Module()
    model.register_parameter('main', torch.nn.Parameter(torch.zeros(2)))
    model.register_parameter('b7', torch.nn.Parameter(torch.zeros(2)))
    model.register_parameter('t7', torch.nn.Parameter(torch.zeros(2)))
    for p, values in zip(model.parameters(), ([6.,8.], [3.,4.], [0.,0.])):
        p.grad = torch.tensor(values)
    refs = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    for p, ref in zip(model.parameters(), refs): ref.grad = p.grad.clone()
    torch.nn.utils.clip_grad_norm_(refs[:1], 1., foreach=None)
    torch.nn.utils.clip_grad_norm_(refs[1:], 1., foreach=None)
    calls = []
    with patch.object(m.dist, 'is_available', return_value=True), patch.object(m.dist, 'is_initialized', return_value=True), patch.object(m.dist, 'all_reduce', side_effect=lambda g, op: calls.append((g.dtype, op))):
        m.clip_and_sync_gradients(model, [model.main], max_norm=1., clip_table_params=[model.b7, model.t7])
    assert len(calls) == 3 and all(dtype == torch.float32 and op == m.dist.ReduceOp.AVG for dtype, op in calls)
    for p, ref in zip(model.parameters(), refs): exact(p.grad, ref.grad)
    return 'SEPARATE_MAIN_AND_JOINT_TABLE_CLIP_EXACT_FP32_AVG_ROUTING'


def models(a, b, lookup):
    b._P2_ROWS = 32  # Only table capacity reduced; true C128 and gate channels retained.
    b._K2_METADATA = lambda ids, vocab: b._k2_prepare_metadata(ids, vocab, 4)
    b._K2_LOOKUP = lookup
    b._K2_TABLE_LOOKUP = lookup
    def make(module, enabled=False):
        torch.manual_seed(42)
        kwargs = dict(sequence_len=8, vocab_size=64, n_layer=8, n_head=2, n_embd=128, mlp_ratio=2)
        if module is b:
            kwargs['ngram_ve'] = enabled
        model = module.GPT(module.GPTConfig(**kwargs))
        model.init_weights()
        return model
    parent, off, on = make(a), make(b), make(b, True)
    # Common parameters retain initial values when the new modules are enabled.
    for key, value in parent.state_dict().items():
        exact(value, on.state_dict()[key])
    assert on.config.ngram_ve and not off.config.ngram_ve
    opts = [torch.optim.SGD(m.parameters(), lr=.01) for m in (parent, off)]
    off_losses = []
    for step in range(3):
        torch.manual_seed(100+step)
        ids, targets = torch.randint(64,(2,8), dtype=INDEX_DTYPE), torch.randint(64,(2,8))
        losses = []
        for model, opt in zip((parent, off), opts):
            opt.zero_grad(set_to_none=True)
            loss = model(ids, targets); loss.backward(); losses.append(loss)
        exact(*losses)
        for pa, pb in zip(parent.parameters(), off.parameters()):
            if pa.grad is None or pb.grad is None:
                assert pa.grad is pb.grad
            else:
                exact(pa.grad, pb.grad)
        for opt in opts: opt.step()
        for key, value in parent.state_dict().items(): exact(value, off.state_dict()[key])
        off_losses.append(float(losses[0].detach()))
    # End-to-end ON compares actual interpreted registered backward with ordinary
    # FP32-master/BF16-gather model, including nonzero tables, gates and RMS update.
    ordinary = make(b, True)
    for model in (on, ordinary):
        with torch.no_grad():
            for p in model.ngram_tables.values(): p.fill_(.125)
    optimizers = [m.setup_optimizer() for m in (on, ordinary)]
    for step in range(2):
        losses = []
        for enabled, model, opt in zip((True, False), (on, ordinary), optimizers):
            b.K2_EMBEDDING = enabled
            opt.zero_grad(set_to_none=True)
            loss = model(ids, targets); loss.backward(); losses.append(loss)
        exact(*losses)
        for pa, pb in zip(on.parameters(), ordinary.parameters()):
            if pa.grad is None or pb.grad is None:
                assert pa.grad is pb.grad
            else: exact(pa.grad, pb.grad)
        for model, opt in zip((on, ordinary), optimizers):
            tables = list(model.ngram_tables.values()); tids = {id(p) for p in tables}
            b.clip_and_sync_gradients(model, [p for p in model.parameters() if id(p) not in tids], max_norm=1., clip_table_params=tables)
            opt.step()
        for key, value in on.state_dict().items(): exact(value, ordinary.state_dict()[key])
    on.eval(); on.requires_grad_(False)
    b.K2_EMBEDDING = True
    with torch.no_grad():
        output = on(ids)
        exact(output, on(ids))
        # Hold GEMM shapes fixed: changing future tokens or another batch row
        # must not change earlier outputs / this row at all.
        future_changed = ids.clone(); future_changed[:, 4:] = (future_changed[:, 4:]+7) % 64
        exact(on(future_changed)[:, :4], output[:, :4])
        other_row = ids.clone(); other_row[1] = (other_row[1]+13) % 64
        exact(on(other_row)[0], output[0])
        cross_shape_prefix_error = float((on(ids[:, :4])-output[:, :4]).abs().max())
    with tempfile.TemporaryDirectory(dir=ROOT, prefix='checkpoint-test-') as directory:
        path = Path(directory)/'model.pt'
        b.save_checkpoint(path, on, {})
        # Match the production Neuron BF16 compute policy in this CPU fixture.
        with patch.object(b, 'compute_dtype_for', return_value=torch.bfloat16):
            loaded = b.load_for_eval(str(path), torch.device('cpu'))
        with torch.no_grad(): exact(output, loaded(ids))
        assert loaded.config.ngram_ve
        old_path = Path(directory)/'old-parent.pt'
        a.save_checkpoint(old_path, parent, {})
        with patch.object(b, 'compute_dtype_for', return_value=torch.bfloat16):
            old_loaded = b.load_for_eval(str(old_path), torch.device('cpu'))
        assert not old_loaded.config.ngram_ve
        parent.eval(); parent.requires_grad_(False)
        with torch.no_grad(): exact(parent(ids), old_loaded(ids))
    b.K2_EMBEDDING = False
    return {'off_3_steps': 'BIT_IDENTICAL', 'off_losses': off_losses,
            'on_2_steps': 'REGISTERED_VS_ORDINARY_BIT_IDENTICAL',
            'common_init': 'BIT_IDENTICAL', 'checkpoint': 'STRICT_ON_AND_OLD_PARENT_ROUNDTRIP',
            'causality_and_batch_independence': 'SAME_SHAPE_BIT_EXACT',
            'cross_shape_prefix_max_abs_difference_diagnostic': cross_shape_prefix_error,
            'test_geometry': '8layers,C128,T16,V64,table32x128'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--index-dtype', choices=('int32', 'int64'), default='int32')
    args = parser.parse_args()
    INDEX_DTYPE = getattr(torch, args.index_dtype)
    torch.manual_seed(42)
    sys.path.insert(0, str(args.repo/'workflow/tools/stub_prepare'))
    arrow, parquet = types.ModuleType('pyarrow'), types.ModuleType('pyarrow.parquet'); arrow.parquet = parquet
    sys.modules['pyarrow'], sys.modules['pyarrow.parquet'] = arrow, parquet
    a, b = load('p2_parent', args.parent), load('p2_candidate', ROOT/'train.py')
    trees = [ast.parse(p.read_text()) for p in (args.parent, ROOT/'train.py')]
    for name in ('_k2_build_kernel', '_k2_prepare_metadata', '_k2_build_registered_lookup'):
        bodies = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for tree in trees]
        assert ast.dump(bodies[0]) == ast.dump(bodies[1]), 'original K2 helper changed'
    lookup = b._k2_table_build_registered_lookup(kernel=cpu_kernel, nki_op_factory=torch.library.custom_op)
    result = {'input_dtype': str(INDEX_DTYPE), 'reference': reference_probe(INDEX_DTYPE), 'hashes_test': hashes(b), 'op_cases': operations(b, lookup),
              'rms': rms(b), 'clip': clipping(b), 'model': models(a, b, lookup), 'native_executed': False,
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.parent, ROOT/'train.py', ROOT/'test_tables.py', ROOT/'test_fp32_reference.py', ROOT/'cpu_instruction_model.py')}}
    (ROOT/('cpu-result-' + args.index_dtype + '.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
