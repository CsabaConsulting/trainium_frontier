"""Restricted fixed FP8 input scale justified by a gain-free RMS norm boundary.

The executable contract is finite BF16 input with |x| <= 130/scale. For the
supported widths, scale=128/sqrt(D), leaving 1/64 relative headroom over the
exact RMS-normalization bound and substantial headroom below legacy E4M3's240.
This is not a general-input replacement for adaptive calibration. The kernel
does not secretly measure/clamp an invalid caller. Quantization is separately
audited against a CPU implementation; no BPB or full-model equivalence follows.
"""
import json

import torch
import torch.nn.functional as F

from fresh_start_fp8_probe import e4m3_legacy
from fp8_weight_cache_probe import require_exact

RMS_EPS = 2.0**-23


def policy(channels):
    if channels not in (256, 1024):
        raise ValueError('Fixed RMS input policy is gated only for D256/D1024')
    scale = 128.0 / channels**.5
    return scale, 130.0 / scale


def require_domain(values):
    scale, bound = policy(values.shape[-1])
    assert values.dtype == torch.bfloat16
    assert bool(torch.isfinite(values).all()), 'Nonfinite fixed-scale input'
    assert float(values.float().abs().max()) <= bound, 'Fixed-scale input exceeds declared norm bound'
    return scale, bound


def normalize(values):
    result = F.rms_norm(values.float(), (values.shape[-1],), eps=RMS_EPS).bfloat16()
    require_domain(result)
    return result


def reference(values):
    scale, bound = require_domain(values)
    meta = torch.tensor([scale, 1.0 / scale, bound], dtype=torch.float32).repeat(values.shape[0], 1)
    quantized = (e4m3_legacy(values.float() * scale) / scale).bfloat16()
    return meta, quantized


def gate_contract(quantize):
    """Gate every in-domain BF16 code and the actual compiled norm boundary.

    Error bound for round-to-nearest E4M3 is |error| <= |x|/16 + 2^-10/scale.
    The norm diagnostic uses FP64 truth and a predeclared FP32 gamma_(2D+8)
    plus BF16 unit-roundoff bound. Half-output negative controls are required.
    """
    all_codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    for channels in (256, 1024):
        scale, bound = policy(channels)
        finite_domain = torch.isfinite(all_codes) & (all_codes.float().abs() <= bound)
        codes = all_codes[finite_domain]
        values = torch.zeros(256, channels, dtype=torch.bfloat16)
        values.flatten()[:codes.numel()] = codes
        metadata, expected = reference(values)
        device_values, device_meta = values.to('neuron'), metadata[:128].to('neuron')
        actual = quantize[2](device_values, device_meta, 2)
        require_exact(f'rms_fixed_D{channels}:all_codes_quantizer', actual, expected)
        require_exact(f'rms_fixed_D{channels}:repeat', quantize[2](device_values, device_meta, 2), actual)
        require_exact(f'rms_fixed_D{channels}:unchanged', device_values, values)
        error = (expected.double() - values.double()).abs()
        error_bound = values.double().abs() / 16.0 + 2.0**-10 / scale
        assert bool((error <= error_bound).all())
        assert float(values.float().abs().max()) * scale <= 130.0 < 240.0
        negative = int(((expected.double() * .5 - values.double()).abs() > error_bound).sum())
        assert negative > 0
        bad = torch.zeros(1, channels, dtype=torch.bfloat16)
        rejected = 0
        for invalid in (bound * 2, float('nan'), float('inf')):
            bad[0, 0] = invalid
            try:
                require_domain(bad)
            except AssertionError:
                rejected += 1
        assert rejected == 3
        print('FP8_RMS_SCALE_QUANTIZER_GATE ' + json.dumps(dict(
            channels=channels, scale=scale, input_bound=bound,
            finite_bf16_codes=int(codes.numel()), exact_device_quantizer=True,
            max_absolute_error=float(error.max()), bound_failures=0,
            half_quantizer_negative_control_failures=negative,
            invalid_domain_negative_controls_rejected=rejected, passed=True)), flush=True)

        def actual_norm(x):
            return F.rms_norm(x.float(), (channels,), eps=RMS_EPS).bfloat16()

        compiled_norm = torch.compile(actual_norm, backend='neuron', fullgraph=True, dynamic=False)
        generator = torch.Generator().manual_seed(7921 + channels)
        raw = torch.randn(256, channels, generator=generator).bfloat16()
        raw[0].zero_()
        raw[1].zero_(); raw[1, 0] = 2.0**30
        raw[2].zero_(); raw[2, -1] = -(2.0**30)
        exponents = torch.linspace(-30, 30, 253).round().int()
        raw[3:] = torch.ldexp(raw[3:].float(), exponents[:, None]).bfloat16()
        normalized = compiled_norm(raw.to('neuron')).cpu()
        require_domain(normalized)
        truth = raw.double() / (raw.double().square().mean(dim=-1, keepdim=True) + RMS_EPS).sqrt()
        gamma = (2 * channels + 8) * 2.0**-24 / (1 - (2 * channels + 8) * 2.0**-24)
        error_bound = truth.abs() * (2.0**-8 + gamma + 2.0**-8 * gamma)
        error = (normalized.double() - truth).abs()
        failures = int((error > error_bound + 2.0**-134).sum())
        negative = int(((normalized.double() * .5 - truth).abs() > error_bound + 2.0**-134).sum())
        assert failures == 0 and negative > 0
        changed = raw.clone(); changed[127] *= -3; changed[128] += 7
        normalized_changed = compiled_norm(changed.to('neuron')).cpu()
        keep = torch.ones(256, dtype=torch.bool); keep[127:129] = False
        require_exact(f'rms_fixed_D{channels}:norm_row_independence', normalized_changed[keep], normalized[keep])
        print('FP8_RMS_SCALE_NORM_GATE ' + json.dumps(dict(
            channels=channels, cases='zero, signed spike, random, input exponents -30..30',
            actual_maximum=float(normalized.abs().max()), input_bound=bound,
            max_absolute_error=float(error.max()), bound_failures=failures,
            half_norm_negative_control_failures=negative,
            norm_reference='independent FP64 RMS norm with protected epsilon',
            domain_generalization_is_not_a_full_model_gate=True, passed=True)), flush=True)
