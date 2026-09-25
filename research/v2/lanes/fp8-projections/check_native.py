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


def tensor_summary(value):
    if value is None:
        return {'present': False}
    cpu = value.detach().float().cpu()
    finite = bool(torch.isfinite(cpu).all())
    return dict(present=True, shape=list(value.shape), dtype=str(value.dtype),
                finite=finite, nonzero=int(torch.count_nonzero(cpu)),
                rms=float(cpu.square().mean().sqrt()) if finite else None,
                l2=float(cpu.norm()) if finite else None,
                max_abs=float(cpu.abs().max()) if finite else None)


def diagnostic_comparison(actual, expected):
    stats = tensor_summary(actual)
    stats['reference'] = tensor_summary(expected)
    if stats.get('finite') and stats['reference'].get('finite'):
        error = metric(actual, expected)
        stats.update(relative_rms=error['relative_rms'], error_max_abs=error['max_abs'])
        stats['within_original_1_percent_rms'] = stats['relative_rms'] <= 0.01
    else:
        stats['within_original_1_percent_rms'] = False
    return stats


def inspect_saved_tensors(output, references):
    """Bounded observation of the actual autograd context, not a direct-kernel proxy.

    This synchronizes/copies tensors before backward and is therefore used only
    in explicitly marked inspection cases, never the first original-path case.
    Availability is diagnostic evidence, not assumed across SDK versions.
    """
    queue, visited, records, errors = [output.grad_fn], set(), [], []
    while queue and len(visited) < 16 and len(records) < 16:
        node = queue.pop(0)
        if node is None or node in visited:
            continue
        visited.add(node)
        try:
            saved = getattr(node, 'saved_tensors', ())
            for index, tensor in enumerate(saved):
                if len(records) == 16:
                    break
                comparisons = {name: diagnostic_comparison(tensor, reference)
                               for name, reference in references.items()
                               if tensor.shape == reference.shape}
                records.append(dict(node=type(node).__name__, index=index,
                                    summary=tensor_summary(tensor), comparisons=comparisons))
        except Exception as error:
            errors.append(dict(node=type(node).__name__, error=f'{type(error).__name__}: {error}'))
        queue.extend(parent for parent, _ in getattr(node, 'next_functions', ()))
    return dict(tensors=records, inspection_errors=errors,
                nodes_visited=len(visited), bounded_at_16_nodes_and_tensors=True,
                pre_backward_materialization=True,
                note='Ordering is not assumed; compare each saved tensor against named references.')


def first_case_diagnostic(mod, device, compiler=None):
    """One fixed 256-cube; no model, data, optimizer, timing or acceptance gate.

    The compiler injection exists only for CPU self-checks. The native CLI always
    uses the production Neuron backend and never falls back to CPU.
    """
    if compiler is None:
        compiler = lambda function: torch.compile(function, backend='neuron', fullgraph=True, dynamic=False)
    torch.manual_seed(42)
    x = torch.randn(256, 256).to(torch.bfloat16)
    x[0].zero_()
    x[1] *= 1e-8
    x[2, 0] = 100.0
    w = torch.randn(256, 256) * (256 ** -0.5)
    gy = torch.randn(256, 256).to(torch.bfloat16)
    qx, sx, ex = mod._fp8_row_reference(x)
    qw, sw, ew = mod._fp8_row_reference(w)
    reference = ((qx @ qw.T) * sw.T * sx).to(torch.bfloat16)
    dx_ref, dw_ref = gy @ ew, (gy.T @ ex).float()
    xn, wn = x.to(device).requires_grad_(), w.to(device).requires_grad_()
    prepare, project = mod._FP8_PROJ_KERNELS
    packed, scale, we = prepare[2](wn.to(torch.bfloat16), 2)
    direct, xe = project[2](xn, packed, scale, 2)
    raw = packed.cpu().permute(2, 1, 0).contiguous().reshape(256, 256)
    decoded = raw.view(torch.float8_e4m3fn).float()
    packed_error = metric(decoded, qw)
    reserved_exponent = bool(((raw >> 3) & 15).eq(15).any())
    bound = compiler(mod._FP8Projection.apply)
    cases = {}
    saved_references = dict(effective_x=ex, effective_w=ew, input_x=x, master_w=w, upstream=gy)

    def run_case(name, function, inputs, inspect=False):
        xx, ww = inputs
        case = dict(pre_backward_saved_tensor_inspection=inspect)
        try:
            y = function(xx, ww)
            case['grad_fn'] = type(y.grad_fn).__name__
            if inspect:
                case['saved_context'] = inspect_saved_tensors(y, saved_references)
            # Keep original bound-apply path free of pre-backward inspection.
            upstream = gy.to(device)
            y.backward(upstream)
            mod.synchronize(device)
            case.update(forward=diagnostic_comparison(y, reference),
                        dx=diagnostic_comparison(xx.grad, dx_ref),
                        dw=diagnostic_comparison(ww.grad, dw_ref),
                        upstream_roundtrip=diagnostic_comparison(upstream, gy),
                        compiled_or_eager_vs_direct=diagnostic_comparison(y, direct),
                        status='OBSERVED')
        except Exception as error:
            case.update(status='ERROR', error=f'{type(error).__name__}: {error}',
                        dx_summary=tensor_summary(xx.grad), dw_summary=tensor_summary(ww.grad))
        cases[name] = case
        print('FP8_VJP_DIAGNOSTIC_CASE ' + json.dumps(dict(case=name, **case), allow_nan=False), flush=True)

    # First computation is the original failed checker path, without retaining
    # or materializing compiled saved tensors before its original backward.
    run_case('original_compiled_bound_apply', bound, (xn, wn))

    def fresh_inputs():
        return (x.detach().clone().to(device).requires_grad_(), w.detach().clone().to(device).requires_grad_())

    def production_shaped_wrapper(xx, ww):
        # Same inner call made by Linear.forward; .apply is inside the compiled
        # Python frame, unlike the original checker's bound-apply entry point.
        return mod._FP8Projection.apply(xx, ww)

    run_case('compiled_enclosing_wrapper', compiler(production_shaped_wrapper), fresh_inputs())
    run_case('eager_apply_inspected', mod._FP8Projection.apply, fresh_inputs(), inspect=True)
    # No third compiler entry point: replay the already compiled bound graph.
    # A changed result here is an observation/materialization-effect clue.
    run_case('compiled_bound_apply_inspected_replay', bound, fresh_inputs(), inspect=True)
    with torch.no_grad():
        gn = gy.to(device)
        native_dx = gn @ we.detach()
        native_dw = (gn.T @ xe.detach()).float()
        # Independent ordinary BF16 device operands distinguish NKI-produced
        # operand handoff from the ordinary native BF16 matmul/upstream path.
        control_dx = gn @ ew.to(device)
        control_dw = (gn.T @ ex.to(device)).float()
        mod.synchronize(device)
    return dict(verdict='DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS',
        original_failure_remains='c1ac162c; no supersession or quality/timing authorization',
        shape=[256, 256, 256], seed=42, cases=cases,
        direct_operands=dict(packed=packed_error, reserved_exponent_present=reserved_exponent,
            scales=diagnostic_comparison(scale.cpu()[0], sw[:, 0]),
            effective_x=diagnostic_comparison(xe, ex), effective_w=diagnostic_comparison(we, ew)),
        manual_native_vjp=dict(dx=diagnostic_comparison(native_dx, dx_ref), dw=diagnostic_comparison(native_dw, dw_ref)),
        ordinary_bf16_control=dict(dx=diagnostic_comparison(control_dx, dx_ref), dw=diagnostic_comparison(control_dw, dw_ref)),
        original_rms_ceiling=0.01, native_kernel_executed=device.type == 'neuron',
        whole_step_measured=False, full_component_gate_passed=False,
        cost_bound='One 256-cube; two compiler entry points, one inspected replay, one eager apply; no optimizer/model.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-py', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--diagnose-first-case', action='store_true', help='Diagnostic only: retain original failure, inspect one 256-cube; never a component PASS.')
    p.add_argument('--label', help='Required unique diagnostic label; also recorded in each receipt.')
    args = p.parse_args()
    torch.set_num_threads(2)
    path = Path(args.train_py).resolve()
    if args.diagnose_first_case:
        if not args.label:
            p.error('--diagnose-first-case requires a unique --label')
        if hashlib.sha256(path.read_bytes()).hexdigest() != '921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858':
            raise ValueError('diagnostic requires unchanged exact source921271')
    spec = importlib.util.spec_from_file_location('native_fp8_test_train', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _, rank, _, world, device = mod.init_runtime('neuron', seed=42)
    if world != 4:
        raise ValueError('native correctness gate requires all four LNC2 ranks')
    try:
        mod._ensure_fp8_projection_kernels()
        if args.diagnose_first_case:
            receipt = first_case_diagnostic(mod, device)
            receipt.update(label=args.label, rank=rank, world_size=world,
                train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                argv=sys.argv[1:], torch=torch.__version__)
            directory = Path(args.out_dir)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f'diagnostic-rank{rank}.json').open('x') as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
                stream.write('\n')
            print(json.dumps(receipt, allow_nan=False), flush=True)
            dist.barrier()
            return
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
