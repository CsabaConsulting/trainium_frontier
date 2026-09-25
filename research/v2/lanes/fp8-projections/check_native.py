#!/usr/bin/env python3
"""Operator-only, four-rank native correctness diagnostic; never a speed test.

Run only after the independent diff/CPU review and the normal machine
preflight/proposal gates. No host access or launcher is implemented here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch_neuronx  # Registers the native Neuron device/backend.


def metric(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('nonfinite native or reference result')
    error = actual - expected
    return dict(relative_rms=float(error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)),
                max_abs=float(error.abs().max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-py', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    path = Path(args.train_py).resolve()
    spec = importlib.util.spec_from_file_location('native_fp8_test_train', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _, rank, _, world, device = mod.init_runtime('neuron', seed=42)
    if world != 4:
        raise ValueError('native correctness gate requires all four LNC2 ranks')
    try:
        mod._ensure_fp8_projection_kernels()
        prepare, project = mod._FP8_PROJ_KERNELS
        cases = []
        # Square, MLP-up, MLP-down, and single-128-row weight packing.
        for channels, outputs in [(256, 256), (256, 1024), (1024, 256), (256, 128)]:
            torch.manual_seed(42)
            x = torch.randn(256, channels).to(torch.bfloat16)
            x[0].zero_()
            x[1] *= 1e-8
            x[2, 0] = 100.0
            w = torch.randn(outputs, channels) * (channels ** -0.5)
            gy = torch.randn(256, outputs).to(torch.bfloat16)
            qx, sx, ex = mod._fp8_row_reference(x)
            qw, sw, ew = mod._fp8_row_reference(w)
            reference = ((qx @ qw.T) * sw.T * sx).to(torch.bfloat16)
            xn, wn = x.to(device).requires_grad_(), w.to(device).requires_grad_()
            lanes = 2 if outputs % 256 == 0 else 1
            packed, scale, we = prepare[lanes](wn.to(torch.bfloat16), lanes)
            direct, xe = project[2](xn, packed, scale, 2)
            # Reconstruct packed bytes independently on CPU: K-partition,
            # K-block, output-channel -> ordinary [output-channel, K].
            raw = packed.cpu().permute(2, 1, 0).contiguous().reshape(outputs, channels)
            decoded = raw.view(torch.float8_e4m3fn).float()
            if ((raw >> 3) & 15).eq(15).any():
                raise ValueError('legacy E4M3 reserved exponent in actual weight bytes')
            packed_error = metric(decoded, qw)
            # Exercise the actual autograd path under the production compiler.
            compiled = torch.compile(mod._FP8Projection.apply, backend='neuron', fullgraph=True, dynamic=False)
            y = compiled(xn, wn)
            y.backward(gy.to(device))
            mod.synchronize(device)
            results = dict(shape=[256, channels, outputs], packed=packed_error,
                scales=metric(scale.cpu()[0], sw[:, 0]),
                saved_x=metric(xe, ex), saved_w=metric(we, ew), forward=metric(y, reference),
                dx=metric(xn.grad, gy @ ew), dw=metric(wn.grad, (gy.T @ ex).float()),
                compiled_vs_direct=metric(y, direct))
            # Quantizer reciprocal/rounding ties and native BF16 reductions may
            # differ; 1% RMS is a predeclared ceiling, not BF16-equality proof.
            if any(v['relative_rms'] > 0.01 for v in results.values() if isinstance(v, dict)):
                raise ValueError('native/reference RMS gate failed: ' + json.dumps(results))
            if not torch.equal(y.cpu()[:8], compiled(xn[:8], wn).detach().cpu()):
                raise ValueError('prefix/padding batch independence failed')
            cases.append(results)
        directory = Path(args.out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        receipt = dict(verdict='PASS_NATIVE_COMPONENT_NUMERICS_ONLY', rank=rank, world_size=world,
            train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases,
            torch=torch.__version__, native_kernels=True, whole_step_measured=False)
        with (directory / f'rank{rank}.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        print(json.dumps(receipt), flush=True)
        dist.barrier()
    finally:
        mod.cleanup_runtime()


if __name__ == '__main__':
    main()
