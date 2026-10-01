#!/usr/bin/env python3
"""Exact CPU model parity and native resampling checks against CPU references."""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--native', action='store_true'); args = ap.parse_args()
    models = {n: load('paircheck_'+n, ROOT/'candidates'/n/'train.py') for n in ('mlp', 'mlp6', 'mlpnki', 'mlp6nki')}
    receipts = []
    for old, new in [('mlp', 'mlpnki'), ('mlp6', 'mlp6nki')]:
        cfg = dict(sequence_len=32, vocab_size=256, n_layer=8, n_head=1, n_embd=128,
                   lane_b=False, coarse_mlp=True)
        torch.manual_seed(42)
        reference = models[old].GPT(models[old].GPTConfig(**cfg)); reference.init_weights()
        with torch.no_grad():
            for block in reference.transformer.h:
                block.mlp.c_proj.weight.uniform_(-.007, .007)
        candidate = models[new].GPT(models[new].GPTConfig(**cfg))
        candidate.load_state_dict(reference.state_dict(), strict=True)
        x = torch.randint(0, 256, (2, 32)); y = torch.randint(0, 256, (2, 32))
        w = torch.tensor([1., .2])
        assert torch.equal(reference(x), candidate(x)), (new, 'CPU logits')
        lr = reference(x, y, mtp_weights=w); lc = candidate(x, y, mtp_weights=w)
        assert torch.equal(lr, lc); lr.backward(); lc.backward()
        for (name, rp), (cn, cp) in zip(reference.named_parameters(), candidate.named_parameters()):
            assert name == cn
            assert torch.equal(rp.grad, cp.grad), (new, 'CPU parameter gradient', name)
        receipts.append(dict(variant=new, cpu_logits_and_all_gradients_exact=True,
                             nonzero_mlp_down_weights_tested=True))
    if args.native:
        m = models['mlpnki']; device = torch.device('neuron')
        for shape in [(2, 512, 128), (2, 1024, 1024), (32, 1024, 1024)]:
            for dtype in [torch.bfloat16]:
                cpu_x = torch.randn(shape, dtype=dtype)
                x = cpu_x.to(device)
                cpu_r = torch.randn(shape[0], shape[1]//2, shape[2], dtype=dtype)
                r = cpu_r.to(device)
                cpu_gfull = torch.randn(shape, dtype=dtype); gfull = cpu_gfull.to(device)
                cpu_ghalf = torch.randn(cpu_r.shape, dtype=dtype); ghalf = cpu_ghalf.to(device)
                for parity in [0, 1]:
                    select = m._pair_select_fwd_native[2](x, parity, 2).cpu()
                    reference = cpu_x[:, parity::2].contiguous()
                    assert torch.equal(select, reference), ('native select', shape, parity)
                    scattered = m._pair_select_bwd_native[2](ghalf, parity, 2).cpu()
                    reference = torch.zeros_like(cpu_x); reference[:, parity::2] = cpu_ghalf
                    assert torch.equal(scattered, reference), ('native scatter', shape, parity)
                    expanded = m._pair_expand_fwd_native[2](r, parity, 2).cpu()
                    reference = cpu_r.repeat_interleave(2, dim=1)
                    if parity:reference = torch.cat((torch.zeros_like(reference[:, :1]), reference[:, :-1]), dim=1)
                    assert torch.equal(expanded, reference), ('native expand', shape, parity)
                    reduced = m._pair_expand_bwd_native[2](gfull, parity, 2).cpu()
                    left = cpu_gfull[:, parity::2]; right = cpu_gfull[:, parity+1::2]
                    if parity:right = torch.cat((right, torch.zeros_like(left[:, :1])), dim=1)
                    reference = (left.float()+right.float()).to(dtype)
                    assert torch.equal(reduced, reference), ('native reduction', shape, parity, float((reduced.float()-reference.float()).abs().max()))
                    print('PAIR_NATIVE_CASE '+json.dumps(dict(shape=shape, parity=parity, exact=True)), flush=True)
                del x,r,gfull,ghalf
        receipts.append(dict(native_forward_backward_exact=True, shapes=3, parities=2,
                             all_four_ops=True, references_computed_on_cpu=True))
    print('PAIR_CHECK '+json.dumps(dict(passed=True, receipts=receipts)), flush=True)


if __name__ == '__main__': main()
