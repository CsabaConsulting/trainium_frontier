#!/usr/bin/env python3
"""Gate each DMA/scan stage and the compiled composition before measuring it."""
import argparse
import json

import torch

from embedding_segmented_dma import build_kernels, dma_metadata
from embedding_segmented_nki import cpu_scan, metadata
from embedding_segmented_probe import check, gradient_for, indices_for, reference, time_arms


def run_case(args, kernels, shape, kind):
    tokens, channels, vocabulary = shape
    ids = indices_for(kind, tokens, vocabulary, 731)
    gradient = gradient_for(kind, tokens, channels, 732)
    rounded = gradient.bfloat16().float()
    meta = dma_metadata(ids, vocabulary)
    exact, bound, counts = reference(rounded, ids, vocabulary)
    cpu_result = cpu_scan(rounded, metadata(ids, vocabulary))
    check(cpu_result, exact, bound, counts, f'dma_cpu:{kind}:{shape}')
    print('DMA_SCAN_CASE '+json.dumps(dict(kind=kind, shape=shape,
          tiled_io=args.tiled_io,
          metadata_bytes=sum(t.numel()*t.element_size() for t in meta),
          arithmetic='BF16-rounded input gradients, FP32 sums, FP32 output')) ,flush=True)
    if args.device == 'cpu':
        return
    import torch_neuronx
    permute, scan, expand = kernels
    gd, idd = gradient.to('neuron'), ids.to('neuron')
    md = tuple(t.to('neuron') for t in meta)
    pp, cc, mm = md
    # Stage1 is a pure permutation and cast: bit equality is required.
    ordered = permute[2](gd, pp, 2, args.tiled_io)
    ordered_ref = gradient[meta[0].reshape(-1).long()].bfloat16()
    assert torch.equal(ordered.cpu(), ordered_ref), 'DMA permutation/cast mismatch'
    scanned = scan[2](ordered, cc, 2, args.tiled_io)
    # Independent CPU recurrence checks ALL scan positions, not only group ends.
    arr = ordered_ref.double().numpy()
    flags = meta[1].numpy().reshape(-1)
    import numpy as np
    scan_ref = np.empty_like(arr)
    scan_bound = np.empty_like(arr)
    accumulator = np.zeros(channels, np.float64)
    absolute_sum = np.zeros(channels, np.float64)
    length = 0
    for i in range(tokens):
        accumulator = accumulator * flags[i] + arr[i]
        absolute_sum = absolute_sum * flags[i] + np.abs(arr[i])
        length = length + 1 if flags[i] else 1
        gamma = length * 2.0**-24 / (1 - length * 2.0**-24)
        scan_ref[i] = accumulator
        scan_bound[i] = gamma * absolute_sum + 1e-30
    actual_scan = scanned.cpu()
    # Parallel and sequential FP32 prefix sums need not be bit-identical.
    # Check every position against the usual gamma_n * sum(abs(inputs)) bound.
    error = (actual_scan.double() - torch.from_numpy(scan_ref)).abs()
    bad_scan = int((error > torch.from_numpy(scan_bound)).sum())
    print('DMA_SCAN_STAGE '+json.dumps(dict(kind=kind, shape=shape,
          finite=bool(torch.isfinite(actual_scan).all()), bound_failures=bad_scan,
          max_abs=float(error.max()))),flush=True)
    assert torch.isfinite(actual_scan).all() and bad_scan == 0, 'FP32 scan mismatch'
    eager = expand[2](scanned, mm, vocabulary, 2)
    check(eager, exact, bound, counts, f'dma_eager:{kind}:{shape}')
    wrapped = tuple(torch_neuronx.wrap_nki(k) for k in kernels)

    def compose(gg, permutation, continuation, mapping):
        ordered_inner = wrapped[0][2](gg, permutation, 2, args.tiled_io)
        scanned_inner = wrapped[1][2](ordered_inner, continuation, 2, args.tiled_io)
        return wrapped[2][2](scanned_inner, mapping, vocabulary, 2)

    torch._dynamo.reset()
    compiled = torch.compile(compose, backend='neuron', fullgraph=True, dynamic=False)

    def native_backward(gg, ii):
        return torch.ops.aten.embedding_dense_backward.default(gg, ii, vocabulary, -1, False)

    native = torch.compile(native_backward, backend='neuron', fullgraph=True, dynamic=False)
    result = compiled(gd, *md)
    check(result, exact, bound, counts, f'dma_compiled:{kind}:{shape}')
    baseline = native(gd, idd)
    check(baseline, exact, bound, counts, f'dma_native_control:{kind}:{shape}')
    assert (compiled(torch.zeros_like(gd), *md).cpu() == 0).all()
    assert torch.equal(result.cpu(), compiled(gd, *md).cpu())
    difference = (result.cpu() - baseline.cpu()).abs()
    print('DMA_SCAN_NATIVE_DIFFERENCE '+json.dumps(dict(kind=kind, shape=shape,
          max_abs=float(difference.max()), exact=bool(torch.equal(result.cpu(), baseline.cpu())),
          stages_passed=True, compiled_composition_passed=True)),flush=True)
    if kind in ('one_id', 'zero', 'cancellation'):
        return

    def complete():
        new_meta = dma_metadata(ids, vocabulary)
        return compiled(gd, *(t.to('neuron') for t in new_meta))

    timed = time_arms(dict(native_full_vocab=lambda:native(gd, idd),
                          dma_scan_prepared_metadata=lambda:compiled(gd,*md),
                          dma_scan_metadata_creation_upload_included=complete),
                     args.steps, torch.neuron.synchronize)
    med = timed['median_ms']
    estimate = med['dma_scan_prepared_metadata'] + max(0,med['dma_scan_metadata_creation_upload_included']-med['dma_scan_prepared_metadata'])/4
    print('DMA_SCAN_BENCH '+json.dumps(dict(kind=kind,shape=shape,**timed,
          tiled_io=args.tiled_io,
          estimated_four_table_amortized_ms=estimate,input_id_d2h_included=False,
          all_intermediate_hbm_and_kernel_costs_included=True,
          estimate_is_not_whole_model_timing=True)),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu','neuron'), default='neuron')
    ap.add_argument('--real-shape', action='store_true')
    ap.add_argument('--training-row', action='store_true')
    ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--tiled-io',action='store_true',
                    help='Move dtype conversion intoSBUF and use explicit128x128 transposes.')
    args=ap.parse_args()
    torch.set_num_threads(4)
    kernels=None
    if args.device=='neuron':
        import torch_neuronx  # noqa:F401
        kernels=build_kernels()
    for shape in [(256,256,256)]+([(8192,1024,8192)] if args.real_shape else []):
        kinds=['uniform','zipf','one_id','zero','cancellation']
        if args.training_row and shape[0]==8192:kinds.append('training')
        for kind in kinds:run_case(args,kernels,shape,kind)
    print('PASS: segmented DMA component; no full-model or BPB claim',flush=True)


if __name__=='__main__':
    main()
