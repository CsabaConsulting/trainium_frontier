#!/usr/bin/env python3
"""Gate FP32 head accumulation and unchanged CE before forward-only timing.

The logits accumulation gate is gamma(C) times the absolute-product sum for
BF16 inputs and FP32 accumulation, plus32 FP32 subnormal units. There is no
BF16 output-rounding allowance. Loss/lse must then match the existing exact
adaptive CE bit-for-bit on these frozen FP32 logits. Independent FP64 loss
diagnostics and compiled-native differences are retained separately.

Loss nonnegativity is diagnostic, not an exact-compatibility requirement:
wave100 proves that the unchanged reference itself returns tiny negative
losses when fused target centering differs from its rounded target bias.
Require identical signs and bit-exact reference values; do not clamp either
kernel, add tolerances, or erase the original failed wave99 sign assertion.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from head_loss_fused_nki import build_kernel
from mlp_fused_bf16_probe import timing
from softcap_ce_parallel_nki import build_kernels


def emit(kind, record):
    print(kind + ' ' + json.dumps(record), flush=True)


def projection_gate(name, result, x, w):
    actual = result.detach().cpu().double()
    left, right = x.double(), w.double().T
    truth = left @ right
    product_sum = left.abs() @ right.abs()
    channels = x.shape[1]
    gamma = channels * 2.0**-24 / (1 - channels * 2.0**-24)
    bound = gamma * product_sum + 32 * 2.0**-149
    error = (actual - truth).abs()
    record = dict(name=name, shape=list(actual.shape),
                  finite=bool(torch.isfinite(actual).all()),
                  fp32_output=result.dtype == torch.float32,
                  bound_failures=int((error > bound).sum()),
                  max_abs=float(error.max()),
                  relative_rms=float(error.norm() / truth.norm().clamp_min(1e-30)),
                  fp32_gamma=gamma, bf16_output_rounding_allowed=False)
    record['passed'] = record['finite'] and record['fp32_output'] and record['bound_failures'] == 0
    emit('FUSED_HEAD_PROJECTION_GATE', record)
    assert record['passed'], record
    return truth


def check(case, functions, loss_reference, x, w, labels):
    inputs = (x.to('neuron'), w.to('neuron'), labels[:, None].float().to('neuron'))
    native = tuple(value.detach().cpu() for value in functions['native'](*inputs))
    weight_t = inputs[1].T.contiguous()
    result = functions['candidate'](inputs[0], weight_t, inputs[2])
    actual = tuple(value.detach().cpu() for value in result)
    projection_gate(case + ':fused', result[0], x, w)
    # Native output is diagnostic: exposing an output may create a precision
    # boundary. Do not silently use a BF16-rounded baseline for speed evidence.
    try:
        projection_gate(case + ':native', native[0], x, w)
        native_fp32_accumulation = True
    except AssertionError:
        native_fp32_accumulation = False
    frozen = tuple(value.cpu() for value in loss_reference(result[0], inputs[2]))
    exact = [torch.equal(a, b) for a, b in zip(actual[1:], frozen)]
    repeated = tuple(value.detach().cpu() for value in functions['candidate'](
        inputs[0], weight_t, inputs[2]))
    repeat_exact = all(torch.equal(a, b) for a, b in zip(actual, repeated))
    finite = all(bool(torch.isfinite(value).all()) for value in actual)
    sample = min(x.shape[0], 128)
    capped = 15.0 * torch.tanh(actual[0][:sample].double() / 15.0)
    shifted = capped - capped.gather(1, labels[:sample, None])
    truth_loss = shifted.exp().sum(1).log()
    loss_error = (actual[1][:sample, 0].double() - truth_loss).abs()
    record = dict(case=case, shape=[*x.shape, w.shape[0]],
                  exact_frozen_logit_loss_lse=exact,
                  frozen_max_abs=[float((a - b).abs().max()) for a, b in zip(actual[1:], frozen)],
                  repeated_exact=repeat_exact, finite=finite,
                  nonnegative_loss=bool((actual[1] >= 0).all()),
                  negative_loss_rows=int((actual[1] < 0).sum()),
                  minimum_loss=float(actual[1].min()),
                  reference_negative_loss_rows=int((frozen[0] < 0).sum()),
                  reference_minimum_loss=float(frozen[0].min()),
                  exact_reference_loss_sign=bool(torch.equal(actual[1] < 0, frozen[0] < 0)),
                  nonnegative_loss_diagnostic_only=True,
                  original_failed_sign_gate='fs-w99-fused-head-loss-forward-stage-gate-r1',
                  reference_rounding_diagnostic='fs-w100-ce-negative-stage-diagnostic-r1',
                  native_fp32_accumulation=native_fp32_accumulation,
                  native_exact_logits_loss_lse=[torch.equal(a, b) for a, b in zip(actual, native)],
                  native_max_abs=[float((a - b).abs().max()) for a, b in zip(actual, native)],
                  frozen_fp64_loss_max_abs=float(loss_error.max()),
                  independent_loss_diagnostic_only=True)
    record['passed'] = all(exact) and repeat_exact and finite and record['exact_reference_loss_sign']
    emit('FUSED_HEAD_LOSS_GATE', record)
    assert record['passed'], record
    assert torch.equal(inputs[0].cpu(), x) and torch.equal(inputs[1].cpu(), w)
    assert torch.equal(inputs[2].cpu().flatten(), labels.float())
    assert torch.equal(weight_t.cpu(), w.T.contiguous())
    if case == 'random':
        changed = x.clone()
        changed[127] = -changed[127] + .25
        changed[128] *= 2
        changed_result = tuple(value.cpu() for value in functions['candidate'](
            changed.to('neuron'), weight_t, inputs[2]))
        keep = torch.ones(x.shape[0], dtype=torch.bool)
        keep[127:129] = False
        independent = all(torch.equal(a[keep], b[keep]) for a, b in zip(actual, changed_result))
        emit('FUSED_HEAD_ROW_INDEPENDENCE', dict(shape=list(x.shape), changed_rows=[127, 128],
                                                passed=independent))
        assert independent
    del result, actual, repeated, inputs, weight_t
    gc.collect()
    return native_fp32_accumulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    assert not args.benchmark or args.real_shape
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    torch.manual_seed(7291)
    fused = wrap_nki(build_kernel())
    forward, _ = (wrap_nki(kernel) for kernel in build_kernels())

    def loss_reference(z, target):
        return forward[2](z, target, 2, True)

    def native(x, weights, target):
        z = F.linear(x, weights).float()
        loss, lse = loss_reference(z, target)
        return z, loss, lse

    def candidate(x, weight_t, target):
        return fused[2](x, weight_t, target, 2)

    functions = {name: torch.compile(fn, backend='neuron', fullgraph=True, dynamic=False)
                 for name, fn in (('native', native), ('candidate', candidate))}
    generator = torch.Generator().manual_seed(4917)
    shapes = [(512, 256, 1024)]
    if args.real_shape:
        shapes.append((512, 1024, 8192))
    count, native_eligible = 0, True
    with torch.no_grad():
        for tokens, channels, vocab in shapes:
            for case in ('zero', 'tiny', 'random', 'cancellation', 'outliers',
                         'confident_correct', 'confident_wrong'):
                x = torch.randn(tokens, channels, generator=generator).bfloat16()
                w = (torch.randn(vocab, channels, generator=generator) * .03).bfloat16()
                labels = torch.randint(vocab, (tokens,), generator=generator)
                labels[:4] = torch.tensor([0, vocab - 1, 511, 512])
                if case == 'zero':
                    w.zero_()
                elif case == 'tiny':
                    x *= 2.0**-12
                elif case == 'cancellation':
                    x[:, 1::2] = x[:, ::2]
                    w[:, 1::2] = -w[:, ::2]
                elif case == 'outliers':
                    x[:, 0] *= 64
                    w[:, 127] *= 64
                elif case.startswith('confident_'):
                    x.zero_(); x[:, 0] = 1
                    w.zero_(); w[:, 0] = -80; w[0, 0] = 80
                    labels.fill_(0 if case == 'confident_correct' else vocab - 1)
                native_eligible &= check(case, functions, loss_reference, x, w, labels)
                count += 1
        if args.benchmark:
            tokens, channels, vocab = 16384, 1024, 8192
            x = torch.randn(tokens, channels, generator=generator).bfloat16()
            w = (torch.randn(vocab, channels, generator=generator) * .03).bfloat16()
            labels = torch.randint(vocab, (tokens,), generator=generator)
            # Gate the real row count on a bounded FP64 row sample, while exact
            # frozen-CE and row checks cover the complete device output below.
            inputs = (x.to('neuron'), w.to('neuron'), labels[:, None].float().to('neuron'))
            prepared = inputs[1].T.contiguous()
            result = functions['candidate'](inputs[0], prepared, inputs[2])
            projection_gate('real_rows_sample', result[0][:128], x[:128], w)
            frozen = loss_reference(result[0], inputs[2])
            assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(result[1:], frozen))
            native_output = functions['native'](*inputs)
            projection_gate('real_rows_native_sample', native_output[0][:128], x[:128], w)
            assert native_eligible, 'Native comparison has a changed FP32 output boundary; do not time it'
            del result, frozen, native_output

            def native_pair():
                return functions['native'](*inputs), functions['native'](*inputs)

            def fused_pair():
                # One real BF16 transpose per update, two microbatches, all
                # logits saves. No prepacked-weight cost is hidden from timing.
                wt = inputs[1].T.contiguous()
                return (functions['candidate'](inputs[0], wt, inputs[2]),
                        functions['candidate'](inputs[0], wt, inputs[2]))

            measured = timing({'native_two_forwards': native_pair,
                               'fused_transpose_and_two_forwards': fused_pair},
                              args.steps, torch.neuron.synchronize)
            emit('FUSED_HEAD_FORWARD_BENCH', dict(shape=[tokens, channels, vocab], **measured,
                weight_transpose_charged=True, logits_save_included=True,
                forward_only=True, backward_optimizer_or_whole_model_included=False))
        emit('FUSED_HEAD_RESULT', dict(passed=True, cases=count, real_width=args.real_shape,
            native_precision_eligible=native_eligible,
            exact_ce_on_frozen_fp32_logits=True,
            backward_or_autograd_integration=False, quality_or_full_step_promotion=False))


if __name__ == '__main__':
    main()
