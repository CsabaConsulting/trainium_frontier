"""Independent FP64 autograd reference and propagated BF16/FP32 error bounds.

CPU BF16 and Neuron compiled arithmetic can round intermediate products at
different points. Compare both with the exact same quantized-weight real-valued
MLP, allowing a bounded FP32 dot accumulation and BF16 rounding at each declared
boundary. The storage-only FP8 experiment still separately requires exact
agreement with its paired NKI BF16-cache control.
"""
import json

import torch


def gamma(length):
    return length * 2.0**-24 / (1 - length * 2.0**-24)


def reference_and_bounds(x, up, down, gradient):
    """Full-coordinate bounds, with no fitted atol or post-result tolerance.

    ReLU is1-Lipschitz. Power-of-two multiplication by2 is exact in the tested
    finite normal range. Every matmul is allowed FP32 accumulation followed by
    BF16 rounding, even when the compiler retains greater intermediate precision.
    Products/squares each have a BF16 output rounding term. Inputs and quantized
    weights are treated as exact data; this does not bound model quantization.
    """
    x, up, down = (a.detach().cpu().double().requires_grad_() for a in (x, up, down))
    g = gradient.detach().cpu().double()
    with torch.enable_grad():
        hidden = (x @ up.T).relu()
        y = hidden.square() @ down.T
        y.backward(g)
    reference = tuple(a.grad.detach().clone() for a in (x, up, down))
    x, up, down, hidden = (a.detach() for a in (x, up, down, hidden))
    tokens, channels = x.shape
    hidden_width = up.shape[0]
    unit = 2.0**-8
    with torch.no_grad():
        up_acc_error = gamma(channels) * (x.abs() @ up.abs().T)
        hidden_error = up_acc_error + unit * (hidden.abs() + up_acc_error)
        del up_acc_error
        derivative_hidden = g @ down
        derivative_acc_error = gamma(channels) * (g.abs() @ down.abs())
        derivative_error = derivative_acc_error + unit * (
            derivative_hidden.abs() + derivative_acc_error)
        del derivative_acc_error
        derivative_pre = derivative_hidden * (2 * hidden)
        pre_input_error = 2 * (hidden.abs() * derivative_error +
                               derivative_hidden.abs() * hidden_error +
                               derivative_error * hidden_error)
        pre_error = pre_input_error + unit * (derivative_pre.abs() + pre_input_error)
        del pre_input_error, derivative_error, derivative_hidden

        input_error = pre_error @ up.abs()
        acc_error = gamma(hidden_width) * ((derivative_pre.abs() + pre_error) @ up.abs())
        bound_x = input_error + acc_error + unit * (reference[0].abs() + input_error + acc_error)
        del input_error, acc_error

        input_error = pre_error.T @ x.abs()
        acc_error = gamma(tokens) * ((derivative_pre.abs() + pre_error).T @ x.abs())
        bound_up = input_error + acc_error + unit * (reference[1].abs() + input_error + acc_error)
        del input_error, acc_error, derivative_pre, pre_error

        activation = hidden.square()
        activation_input_error = 2 * hidden.abs() * hidden_error + hidden_error.square()
        activation_error = activation_input_error + unit * (activation + activation_input_error)
        del activation_input_error, hidden_error, hidden
        input_error = g.abs().T @ activation_error
        acc_error = gamma(tokens) * (g.abs().T @ (activation + activation_error))
        bound_down = input_error + acc_error + unit * (reference[2].abs() + input_error + acc_error)
    return reference, (bound_x, bound_up, bound_down)


def audit(name, actual, reference, bounds):
    records = []
    for field, got, expected, bound in zip(('dx', 'dup', 'ddown'), actual, reference, bounds):
        got = got.detach().cpu().double()
        error = (got - expected).abs()
        rms = expected.square().mean().sqrt().clamp_min(1e-30)
        # This tiny absolute term covers exact zero, not ordinary BF16 drift.
        failures = int((error > bound + 1e-30).sum())
        negative = int(((got * .5 - expected).abs() > bound + 1e-30).sum())
        record = dict(name=name, field=field, shape=list(got.shape),
                      finite=bool(torch.isfinite(got).all()), bound_failures=failures,
                      max_abs=float(error.max()), relative_rms=float(error.square().mean().sqrt() / rms),
                      max_bound=float(bound.max()), rms_bound=float(bound.square().mean().sqrt()),
                      half_gradient_negative_control_failures=negative,
                      reference='independent FP64 autograd on exact quantized weights',
                      error_model='propagated FP32 gamma_K and BF16 u=2**-8; no empirical atol')
        record['passed'] = record['finite'] and failures == 0
        print('FP8_CACHE_BACKWARD_FP64_BOUND ' + json.dumps(record), flush=True)
        assert record['passed'], record
        # Confirm that the bound rejects a missing derivative factor for any
        # nonzero field instead of becoming a vacuous allowance.
        assert not bool(expected.abs().any()) or negative > 0, record
        records.append(record)
    return records
