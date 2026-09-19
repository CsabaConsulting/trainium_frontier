#!/usr/bin/env python3
"""Gate the five-output FP8 forward against the original four-product authority.

Every retained forward tensor must exactly match the seven-output path. Reuse
the independent CPU quantizers, product bounds, two-microbatch accumulation and
updated-cache checks from the complete MLP gate, substituting only the compiled
autograd consumer of compact saves. This probe makes no timing or quality claim.
"""
import argparse
import json

import torch

from fp8_complete_mlp_autograd import make_autograd
from fp8_complete_mlp_probe import build_functions, run
from fp8_gradient_prepare_nki import build_kernels as build_gradient
from fp8_mlp_data_backward_nki import build_kernel as build_backward
from fp8_mlp_token_rows_nki import build_kernel as build_forward
from fp8_packed_weight_cache_nki import build_kernel as build_pack
from fp8_rms_scale_contract import policy
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_cache_probe import require_exact
from fp8_weight_gradient_nki import build_kernels as build_weight_gradient


def applications(channels, kernel, backward, partial, finish, product, stream_packed_weights=False,
                 output_batch_size=0):
    legacy, _, mixed, native = build_functions(channels, kernel, backward, partial, finish, product)
    scale = policy(channels)[0]

    def compact(x, u, d, um, dm, pu, pd):
        rows = x.reshape(-1, channels) if output_batch_size else x
        return kernel[2](rows, u, d, um, dm, 2, 128, False, scale, pu, pd,
                         True, False, True, stream_packed_weights, output_batch_size)

    def token_rows(values, shape):
        if output_batch_size:
            assert values[0].shape == (output_batch_size, shape[0] // output_batch_size, channels)
            return (values[0].reshape(shape), *values[1:])
        return values

    def paired_forward(x, u, d, um, dm, pu, pd):
        original = legacy(x, u, d, um, dm, pu, pd)
        candidate = token_rows(compact(x, u, d, um, dm, pu, pd), x.shape)
        repeated = token_rows(compact(x, u, d, um, dm, pu, pd), x.shape)
        if stream_packed_weights:
            resident = kernel[2](x, u, d, um, dm, 2, 128, False, scale, pu, pd, True, False, True)
            for index, (value, truth) in enumerate(zip(candidate, resident)):
                require_exact(f'streamed:resident_compact_tensor:{index}', value, truth)
        assert len(original) == 7 and len(candidate) == len(repeated) == 5
        for index, prior in enumerate((0, 1, 4, 5, 6)):
            require_exact(f'compact:retained_tensor:{index}', candidate[index], original[prior])
            require_exact(f'compact:repeat_tensor:{index}', repeated[index], candidate[index])
        print('FP8_COMPACT_SAVES_GATE ' + json.dumps(dict(
            tokens=x.shape[0], channels=channels, exact_output_relu_fp8_bytes_metadata=True,
            original_outputs=7, compact_outputs=5, stream_packed_weights=stream_packed_weights,
            output_batch_size=output_batch_size, direct_nki_output_rank=3 if output_batch_size else 2,
            repeated_call_exact=True, passed=True)), flush=True)
        # Independent staged backward still receives its original full tuple;
        # the candidate's autograd below consumes ONLY its own compact outputs.
        return original

    apply = make_autograd(compact, backward, partial, finish, product,
                          compact_forward_saves=True, preserve_input_rank=bool(output_batch_size))
    if output_batch_size:
        def shaped_complete(x, *rest):
            # Exercise rank-three autograd input/output and unchanged master
            # gradients, then expose the original gate's two-dimensional API.
            shaped = x.reshape(output_batch_size, x.shape[0] // output_batch_size, channels)
            return apply(shaped, *rest).reshape(x.shape)
        complete = torch.compile(shaped_complete, backend='neuron', fullgraph=True, dynamic=False)
    else:
        complete = torch.compile(apply, backend='neuron', fullgraph=True, dynamic=False)
    return paired_forward, complete, mixed, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--real-tokens', type=int, default=32768)
    parser.add_argument('--stream-packed-weights', action='store_true',
                        help='Additionally stream packed output groups in the compact candidate.')
    parser.add_argument('--preserve-output-rank', action='store_true',
                        help='Gate direct rank-three NKI output and autograd input gradients.')
    args = parser.parse_args()
    if args.real_tokens < 512 or args.real_tokens % 512:
        parser.error('real tokens must be a positive multiple of512')
    import torch_neuronx  # noqa: F401
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    kernel, backward, pack = (wrap_nki(fn()) for fn in (build_forward, build_backward, build_pack))
    partial, finish = (wrap_nki(k) for k in build_gradient())
    _, byte_pack, product = (wrap_nki(k) for k in build_weight_gradient())
    calibrate, quantize = (wrap_nki(k) for k in build_calibration())
    kernels = kernel, backward, partial, finish, product, calibrate, quantize, pack, byte_pack
    small = applications(256, kernel, backward, partial, finish, product, args.stream_packed_weights,
                         2 if args.preserve_output_rank else 0)
    cases = ('random', 'zero_input', 'zero_up', 'zero_down', 'zero_gradient',
             'tiny_gradient', 'outliers', 'changed_weights', 'norm_spikes', 'norm_tiny')
    for case in cases:
        run((512, 256, 512), case, kernels, small, 1, False)
    if args.real_shape:
        # The numerical gates are complete for this independent width. Only
        # this process's inactive Dynamo guards are dropped before the next;
        # default compiler limits and shared device caches remain unchanged.
        torch._dynamo.reset()
        print('FP8_COMPACT_PROCESS_GUARD_RESET ' + json.dumps(dict(
            completed_width=256, next_width=1024, shared_device_cache_purged=False)), flush=True)
        large = applications(1024, kernel, backward, partial, finish, product, args.stream_packed_weights,
                             32 if args.preserve_output_rank else 0)
        run((args.real_tokens, 1024, 4096), 'real_random', kernels, large, 1, False)
    print('FP8_COMPACT_SAVES_RESULT ' + json.dumps(dict(
        passed=True, stream_packed_weights=args.stream_packed_weights, cases=len(cases) + int(args.real_shape), real_shape=args.real_shape,
        preserve_output_rank=args.preserve_output_rank,
        real_tokens=args.real_tokens if args.real_shape else None,
        all_original_independent_forward_and_four_product_gates=True,
        exact_retained_forward_tensors_and_compiled_gradients=True,
        cache_updates_and_fp32_accumulation=True,
        whole_model_speed_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
