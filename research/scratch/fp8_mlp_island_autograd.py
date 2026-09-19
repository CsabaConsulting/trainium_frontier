"""Declared straight-through BF16 backward for the actual FP8 forward island.

This is mixed precision, not a full FP8 training pipeline. Scale selection,
FP8 rounding, TensorEngine pair rounding and BF16 casts use identity surrogates.
Weight gradients reach each FP32 master exactly once; cache inputs are detached.
Forward gates separately validate every saved value and token independence.
"""
import torch


def make_autograd(forward, relu_token_major=False, token_row_backward=False):
    if token_row_backward and not relu_token_major:
        raise ValueError('Token-row backward requires token-row saved ReLU')
    class FP8Island(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, up_master, down_master, up_cache, down_cache, up_meta, down_meta):
            result, saved_relu, quantized_x, quantized_hidden = forward(
                x, up_cache, down_cache, up_meta, down_meta)
            ctx.save_for_backward(quantized_x, up_cache, down_cache, saved_relu, quantized_hidden)
            ctx.up_dtype, ctx.down_dtype = up_master.dtype, down_master.dtype
            return result

        @staticmethod
        def backward(ctx, gradient):
            x, up, down, saved_relu, activated = ctx.saved_tensors
            if token_row_backward:
                # Keep the saved ReLU and its derivative in token rows.
                # The alternative logical GEMM orientation retains all BF16
                # rounding boundaries and uses the same independent FP64 bound.
                derivative_hidden = gradient @ down
                derivative_pre = derivative_hidden * (saved_relu * 2.0)
                dx = derivative_pre @ up
                dup = (derivative_pre.T @ x).to(ctx.up_dtype)
            else:
                relu_t = saved_relu.T if relu_token_major else saved_relu
                derivative_hidden_t = down.T @ gradient.T
                derivative_pre_t = derivative_hidden_t * (relu_t * 2.0)
                dx = derivative_pre_t.T @ up
                dup = (derivative_pre_t @ x).to(ctx.up_dtype)
            ddown = (gradient.T @ activated).to(ctx.down_dtype)
            return dx, dup, ddown, None, None, None, None
    return FP8Island.apply


def reference_and_bounds(x, up, down, relu_t, activated, gradient):
    """Independent FP64 autograd VJP of the declared surrogate, with bounds.

    Inputs/weights/activations here are the quantized saved values already
    checked by the independent forward-stage gates. Detach offsets impose those
    forward values while ordinary FP64 autograd differentiates the surrogate.
    The bound covers BF16 backward computation, not forward quantization quality.
    No empirical tolerances or CPU BF16 rounding path are substituted.
    """
    from mlp_backward_rounding_bounds import gamma
    x, up, down = (a.detach().cpu().double().requires_grad_() for a in (x, up, down))
    r = relu_t.detach().cpu().double().T.contiguous()
    h = activated.detach().cpu().double()
    g = gradient.detach().cpu().double()
    with torch.enable_grad():
        pre = x @ up.T
        # The local derivative of ReLU^2 is2*r, including r=0. This value
        # injection has the same derivative without depending on CPU rounding
        # at a different preactivation. The forward stage was audited separately.
        relu_surrogate = pre + (r - pre).detach()
        hidden = relu_surrogate.square()
        quantized_hidden = hidden + (h - hidden).detach()
        value = quantized_hidden @ down.T
        value.backward(g)
    reference = tuple(a.grad.detach().clone() for a in (x, up, down))
    x, up, down = (a.detach() for a in (x, up, down))
    tokens, channels = x.shape
    hidden_width = up.shape[0]
    unit = 2.0**-8
    with torch.no_grad():
        derivative_hidden = g @ down
        accumulated = gamma(channels) * (g.abs() @ down.abs())
        derivative_error = accumulated + unit * (derivative_hidden.abs() + accumulated)
        derivative_pre = derivative_hidden * (2 * r)
        input_error = 2 * r * derivative_error
        pre_error = input_error + unit * (derivative_pre.abs() + input_error)

        error_x = pre_error @ up.abs()
        acc_x = gamma(hidden_width) * ((derivative_pre.abs() + pre_error) @ up.abs())
        bound_x = error_x + acc_x + unit * (reference[0].abs() + error_x + acc_x)
        error_up = pre_error.T @ x.abs()
        acc_up = gamma(tokens) * ((derivative_pre.abs() + pre_error).T @ x.abs())
        bound_up = error_up + acc_up + unit * (reference[1].abs() + error_up + acc_up)
        acc_down = gamma(tokens) * (g.abs().T @ h.abs())
        bound_down = acc_down + unit * (reference[2].abs() + acc_down)
    return reference, (bound_x, bound_up, bound_down)


def audit(name, actual, reference, bounds):
    import json
    for field, got, truth, bound in zip(('dx', 'dup', 'ddown'), actual, reference, bounds):
        got = got.detach().cpu().double()
        error = (got - truth).abs()
        failures = int((error > bound + 1e-30).sum())
        negative = int(((got * .5 - truth).abs() > bound + 1e-30).sum())
        record = dict(name=name, field=field, finite=bool(torch.isfinite(got).all()),
                      bound_failures=failures, max_abs=float(error.max()),
                      relative_rms=float(error.square().mean().sqrt()
                                         / truth.square().mean().sqrt().clamp_min(1e-30)),
                      rms_bound=float(bound.square().mean().sqrt()),
                      half_gradient_negative_control_failures=negative,
                      reference='independent FP64 autograd of declared STE using audited saved values',
                      forward_quantization_quality_bounded=False)
        record['passed'] = record['finite'] and failures == 0 and (
            not bool(truth.abs().any()) or negative > 0)
        print('FP8_ISLAND_BACKWARD_BOUND ' + json.dumps(record), flush=True)
        assert record['passed'], record
