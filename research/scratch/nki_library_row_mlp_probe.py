#!/usr/bin/env python3
"""Stagewise ROW FP8 MLP screen, charging all online input/weight preparation.

Decode legacy bytes on CPU and audit the two GEMMs against the independently
characterized11-bit pair-add model with FP32 accumulation bounds. The model is
empirical, not an AWS arithmetic guarantee. Retain library scaling roundings.
Forward timing excludes saved activations/backward and cannot promote training.
"""
import argparse

import torch
import torch.nn.functional as F
import torch_neuronx

from fp8_mlp_island_probe import make_inputs
from fp8_packed_weight_cache_probe import decode_legacy
from fresh_start_fp8_probe import e4m3_legacy, eleven_bit_pair_reference
from mlp_fused_bf16_probe import timing
from nki_library_probe_common import bounded, emit, exact, verify_sources
from nki_library_row_mlp import install, row_mlp
from nki_library_row_quantize import quantize_rows
from nki_library_row_quantize_probe import check as quantizer_check


def selected_reference(left, right):
    rows, contraction = left.shape
    columns = right.shape[1]
    if rows <= 512:
        rr, cc = torch.arange(rows), torch.arange(columns)
    else:
        rr = torch.unique(torch.cat((torch.linspace(0, rows - 1, 33).long(),
                                     torch.tensor([127, 128, 255, 256]))))
        cc = torch.unique(torch.cat((torch.linspace(0, columns - 1, 257).long(),
                                     torch.tensor([127, 128, 255]))))
    order = torch.arange(contraction).reshape(-1, 2, 128).transpose(1, 2).reshape(-1)
    truth, bound = eleven_bit_pair_reference(left[rr][:, order], right[order][:, cc], 1.0)
    return rr, cc, truth, bound


def native(x, up, down):
    return F.linear(F.relu(F.linear(x, up.bfloat16())).square(), down.bfloat16())


def run_case(shape, kind, steps):
    torch._dynamo.reset()
    tokens, channels, hidden = shape
    x, up, down = make_inputs(shape, kind)
    if kind == 'tiny_weights':
        up, down = ((v.float() * 2.0**-30).bfloat16() for v in (up, down))
    elif kind == 'row_scale_range':
        x = torch.ldexp(x.float(), torch.linspace(-5, 5, tokens).round().int()[:, None]).bfloat16()
    up, down = up.float(), down.float()  # Ordinary FP32 master boundary; casts are timed.
    quantizer, mlp = (torch_neuronx.wrap_nki(k) for k in (quantize_rows, row_mlp))

    def prepare(xx, uu, dd):
        qx, xm = quantizer[2](xx, True)
        qu, um = quantizer[2](uu.bfloat16(), False)
        qd, dm = quantizer[2](dd.bfloat16(), False)
        return (qx, qu.T.contiguous(), qd.T.contiguous(),
                um[:, 1].reshape(1, hidden).expand(128, hidden).contiguous(),
                dm[:, 1].reshape(1, channels).expand(128, channels).contiguous(),
                xm, um, dm)

    def diagnostic(*values):
        return mlp[2](*values, True)

    def complete(xx, uu, dd):
        prepared = prepare(xx, uu, dd)
        return mlp[2](*prepared[:5], False)

    prepared_fn = torch.compile(prepare, backend='neuron', fullgraph=True, dynamic=False)
    debug_fn = torch.compile(diagnostic, backend='neuron', fullgraph=True, dynamic=False)
    complete_fn = torch.compile(complete, backend='neuron', fullgraph=True, dynamic=False)
    native_fn = torch.compile(native, backend='neuron', fullgraph=True, dynamic=False)
    values = tuple(v.to('neuron') for v in (x, up, down))
    prepared = prepared_fn(*values)
    px, pu, pd, su, sd, xm, um, dm = (v.detach().cpu() for v in prepared)
    quantizer_check(kind + ':input', x, (px, xm), True)
    quantizer_check(kind + ':up_weight', up.bfloat16(), (pu.T.contiguous(), um), False)
    quantizer_check(kind + ':down_weight', down.bfloat16(), (pd.T.contiguous(), dm), False)
    result = tuple(v.detach().cpu() for v in debug_fn(*prepared[:5]))
    output, raw_up, relu, intermediate_bytes, intermediate_scale = result
    assert all(bool(torch.isfinite(v).all()) for v in result)
    dx, du, dd = decode_legacy(px[:, :channels]), decode_legacy(pu), decode_legacy(pd)
    rr, cc, truth, bound = selected_reference(dx, du)
    bounded(f'{shape}:{kind}:up_pair_fp32', raw_up[rr][:, cc], truth, bound + 1e-30)
    # Independent per-stage algebra checks the library's two BF16 roundings:
    # PSUM * output-channel weight scale -> BF16; * token scale + ReLU -> BF16.
    rounded = (raw_up * um[:, 1].reshape(1, hidden)).bfloat16().float()
    expected_relu = (rounded * xm[:, 1:2]).relu().bfloat16()
    exact(kind + ':scaled_relu_rounding', relu, expected_relu)
    activated = relu.square().float()
    expected_scale = (activated.abs().amax(1, keepdim=True) *
                      torch.tensor(1 / 240, dtype=torch.float32)).clamp_min(1e-6)
    exact(kind + ':intermediate_scale', intermediate_scale, expected_scale)
    dh = decode_legacy(intermediate_bytes)
    # A predeclared interval handles FP32 reciprocal/multiply rounding before
    # discrete FP8 ties:4 unit roundoffs for reciprocal plus multiply margin.
    # This is an explicit experimental requirement, not an API guarantee.
    value = activated.double() / expected_scale.double()
    eps = 8 * 2.0**-24
    lo, hi = e4m3_legacy(value * (1 - eps)), e4m3_legacy(value * (1 + eps))
    mismatches = int(((dh.double() < lo) | (dh.double() > hi)).sum())
    emit('NKILIB_ROW_INTERMEDIATE_GATE', dict(name=kind, passed=mismatches == 0,
         outside_quantization_interval=mismatches, relative_interval=eps,
         all_decoded_coordinates=True, finite=True))
    assert mismatches == 0, kind
    rr, cc, truth, bound = selected_reference(dh, dd)
    for scale in (dm[cc, 1].double()[None, :], intermediate_scale[rr].double()):
        truth = truth * scale
        bound = bound * scale.abs()
        bound += 2.0**-24 * (truth.abs() + bound)  # FP32 scale multiplication.
        bound += 2.0**-8 * (truth.abs() + bound)   # The library's BF16 store.
    bounded(f'{shape}:{kind}:down_scaled_pair', output[rr][:, cc], truth, bound + 1e-30)
    plain = complete_fn(*values).detach().cpu()
    exact(kind + ':plain_diagnostic_output', plain, output)
    exact(kind + ':plain_repeat', complete_fn(*values), plain)
    changed = x.clone()
    changed[1] = (changed[1].float() * .5 + .125).bfloat16()
    other = complete_fn(changed.to('neuron'), values[1], values[2]).detach().cpu()
    keep = torch.arange(tokens) != 1
    exact(kind + ':independent_rows', other[keep], plain[keep])
    baseline = native_fn(*values).detach().cpu().double()
    delta = (plain.double() - baseline).norm() / baseline.norm().clamp_min(1e-30)
    emit('NKILIB_ROW_APPROXIMATION', dict(name=kind, relative_rms=float(delta),
         comparison='unquantized native BF16', quality_gate=False))
    measured = {}
    if kind in ('random', 'real_random'):
        measured = timing(dict(native_bf16=lambda: native_fn(*values),
                               row_fp8_with_online_preparation=lambda: complete_fn(*values)),
                          steps, torch.neuron.synchronize)
    emit('NKILIB_ROW_MLP_CASE', dict(shape=list(shape), kind=kind, passed=True,
         independent_decoded_pair_and_rounding_gates=True,
         full_gemm_coordinates=tokens <= 512, all_outputs_and_quantizer_coordinates_finite=True,
         plain_and_instrumented_exact=True, all_online_calibration_layout_and_master_casts_charged=True,
         saved_activations_and_backward_excluded=True, **measured))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_sources()
    install()
    cases = [((512, 256, 1024), kind) for kind in
             ('random', 'zero_input', 'zero_up', 'zero_down', 'outliers',
              'changed_weights', 'tiny_weights', 'row_scale_range')]
    cases += [((2048, 1024, 4096), 'real_random'), ((32768, 1024, 4096), 'real_random')]
    for shape, kind in cases:
        emit('NKILIB_ROW_MLP_START', dict(shape=shape, kind=kind))
        run_case(shape, kind, args.steps)
    emit('NKILIB_ROW_MLP_RESULT', dict(passed=True, cases=len(cases), forward_only=True,
         online_preparation_charged=True, full_training_or_bpb_claim=False))


if __name__ == '__main__':
    main()
