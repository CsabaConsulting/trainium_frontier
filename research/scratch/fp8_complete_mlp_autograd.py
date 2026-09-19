"""Compose the gated FP8 forward and all four FP8 backward products.

This declares a new approximate VJP. Data products quantize gradients per row;
weight products use current matrix-wide gradient calibration. The ReLU-squared
derivative stays BF16 and both master-weight gradients are produced in FP32.
Saved forward FP8 bytes are reused, with no activation requantization pass.
All caches are detached inputs and must be refreshed from current masters.
"""
import torch


def make_autograd(forward, backward_data, partial, finish, product, compact_forward_saves=False,
                  preserve_input_rank=False):
    class CompleteFP8MLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, up_master, down_master, up_cache, down_cache,
                    up_meta, down_meta, packed_up, packed_down,
                    packed_up_transpose, packed_down_transpose, x_meta, one_meta):
            values = forward(
                x, up_cache, down_cache, up_meta, down_meta, packed_up, packed_down)
            if compact_forward_saves:
                result, relu, x_bytes, hidden_bytes, hidden_meta = values
            else:
                result, relu, _, _, x_bytes, hidden_bytes, hidden_meta = values
            ctx.save_for_backward(relu, x_bytes, hidden_bytes, hidden_meta,
                                  up_meta, down_meta, packed_up_transpose,
                                  packed_down_transpose, x_meta, one_meta)
            ctx.up_dtype, ctx.down_dtype = up_master.dtype, down_master.dtype
            if preserve_input_rank:
                ctx.input_shape = x.shape
            return result

        @staticmethod
        def backward(ctx, gradient):
            if preserve_input_rank:
                # Only the optional forward boundary has a different rank.
                # Every gated FP8 backward product still consumes token rows.
                gradient = gradient.reshape(-1, gradient.shape[-1])
            (relu, x_bytes, hidden_bytes, hidden_meta, up_meta, down_meta,
             packed_up_transpose, packed_down_transpose, x_meta, one_meta) = ctx.saved_tensors
            dx, pre = backward_data[2](
                gradient, relu, packed_down_transpose, packed_up_transpose,
                down_meta, up_meta, 2, 128, False, True)
            # Move the hidden's row inverse into G before the down-weight
            # reduction. This is the separately declared BF16 scale boundary.
            factors = hidden_meta[:, 1:2].contiguous()
            down_parts = partial[2](gradient, factors, 2, True)
            down_bytes, down_gradient_meta = finish[2](gradient, factors, down_parts, 2, True)
            ddown = product[2](down_bytes, hidden_bytes, down_gradient_meta,
                               one_meta, 2, True, True)
            up_parts = partial[2](pre, one_meta, 2, False)
            up_bytes, up_gradient_meta = finish[2](pre, one_meta, up_parts, 2, False)
            dup = product[2](up_bytes, x_bytes, up_gradient_meta, x_meta, 2, True, True)
            if preserve_input_rank:
                dx = dx.reshape(ctx.input_shape)
            return (dx, dup.to(ctx.up_dtype), ddown.to(ctx.down_dtype)) + (None,) * 10

    return CompleteFP8MLP.apply
