"""Experimental training MLP fusion for installed NKI0.5/Trainium2.

Normal PyTorch layouts enter the kernel: X[T,C] BF16 and weights[H,C]
and[C,H], either FP32 or an explicitly measured BF16 cache. Weight loading/casting
and all layout movement are included; cache creation/refresh is separate. Both
BF16 weight matrices remain in SBUF across token tiles. The up projection,
BF16 rounding, ReLU, square and down projection are fused. ReLU activations
are saved to HBM for backward, so this is not an inference-only benchmark.

Backward initially uses native compiled BF16 matmuls, preserving FP32 parameter
gradient types. This scratch implementation does not modify the training model.
API: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html
"""
import torch
import torch.nn.functional as F


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def fused_mlp(x, up, down, lanes: int = 2, token_tile: int = 512,
                  contiguous_weights: bool = False, tiled_io: bool = False,
                  transposed_relu: bool = False):
        tokens, channels = x.shape
        hidden, up_channels = up.shape
        assert up_channels == channels and down.shape == (channels, hidden)
        assert x.dtype == nl.bfloat16 and up.dtype in (nl.float32, nl.bfloat16)
        assert down.dtype == up.dtype
        assert channels % 128 == 0 and hidden % 128 == 0 and lanes in (1, 2)
        assert token_tile <= 512 and token_tile % 128 == 0
        assert tokens % (token_tile * lanes) == 0
        nc, nh = channels // 128, hidden // 128
        # Capacity estimate excludes allocator overhead/spills, which must be
        # checked in the actual profile. Reserve two MiB beyond live caches.
        live_bytes = 4 * channels * hidden + 2 * (channels + hidden) * token_tile
        assert live_bytes + 2 * 1024**2 <= nl.tile_size.sbuf_size_bytes
        result = nl.ndarray((tokens, channels), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        saved_shape = (hidden, tokens) if transposed_relu else (tokens, hidden)
        saved_relu = nl.ndarray(saved_shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        up_cache = nl.ndarray((128, nh, nc, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_cache = nl.ndarray((128, nc, nh, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for hi in nl.affine_range(nh):
            for ci in nl.affine_range(nc):
                # P is the contracting dimension, F the stationary output
                # dimension. The DMA reads ordinary row-major weight tensors.
                if contiguous_weights:
                    # Combining strided transpose and FP32->BF16 DMA produced
                    # scalar two-byte packets. Read normal contiguous rows,
                    # transpose the input dtype on TensorEngine, then cache in SBUF.
                    # Gen3 transpose requires matching source/destination
                    # dtypes; it is bit-preserving rather than an accumulation.
                    up_raw = nl.ndarray((128, 128), dtype=up.dtype, buffer=nl.sbuf)
                    down_raw = nl.ndarray((128, 128), dtype=down.dtype, buffer=nl.sbuf)
                    up_t = nl.ndarray((128, 128), dtype=up.dtype, buffer=nl.psum)
                    down_t = nl.ndarray((128, 128), dtype=down.dtype, buffer=nl.psum)
                    nisa.dma_copy(dst=up_raw, src=up[hi * 128:(hi + 1) * 128,
                                                    ci * 128:(ci + 1) * 128])
                    nisa.dma_copy(dst=down_raw, src=down[ci * 128:(ci + 1) * 128,
                                                        hi * 128:(hi + 1) * 128])
                    nisa.nc_transpose(dst=up_t, data=up_raw)
                    nisa.nc_transpose(dst=down_t, data=down_raw)
                    nisa.tensor_copy(dst=up_cache[:, hi, ci, :], src=up_t)
                    nisa.tensor_copy(dst=down_cache[:, ci, hi, :], src=down_t)
                else:
                    # Retained failed control: the small profile measured
                    #524288 two-byte packets for the two weight inputs.
                    nisa.dma_copy(dst=up_cache[:, hi, ci, :],
                                  src=up.ap(pattern=[[1, 128], [channels, 128]],
                                            offset=hi * 128 * channels + ci * 128))
                    nisa.dma_copy(dst=down_cache[:, ci, hi, :],
                                  src=down.ap(pattern=[[1, 128], [hidden, 128]],
                                              offset=ci * 128 * hidden + hi * 128))
        for mi in nl.affine_range(tokens // (token_tile * lanes)):
            m0 = (mi * lanes + nl.program_id(0)) * token_tile
            inputs = nl.ndarray((128, nc, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
            activated = nl.ndarray((128, nh, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
            for ci in nl.affine_range(nc):
                if tiled_io:
                    # Read whole contiguous BF16 rows before exchanging the
                    # token and channel axes on TensorEngine. Direct strided
                    # dma_copy is retained below as the measured control.
                    for ti in nl.affine_range(token_tile // 128):
                        input_rows = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                        input_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        nisa.dma_copy(dst=input_rows,
                                      src=x[m0 + ti * 128:m0 + (ti + 1) * 128,
                                            ci * 128:(ci + 1) * 128])
                        nisa.nc_transpose(dst=input_t, data=input_rows)
                        nisa.tensor_copy(dst=inputs[:, ci, ti * 128:(ti + 1) * 128], src=input_t)
                else:
                    nisa.dma_copy(dst=inputs[:, ci, :],
                                  src=x.ap(pattern=[[1, 128], [channels, token_tile]],
                                           offset=m0 * channels + ci * 128))
            for hi in nl.affine_range(nh):
                up_acc = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.psum)
                for ci in nl.affine_range(nc):
                    nisa.nc_matmul(dst=up_acc, stationary=up_cache[:, hi, ci, :],
                                   moving=inputs[:, ci, :], accumulate=(ci > 0))
                rounded = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                relu = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rounded, src=up_acc)
                nisa.tensor_scalar(dst=relu, data=rounded, op0=nl.maximum, operand0=0.0)
                nisa.tensor_tensor(dst=activated[:, hi, :], data1=relu, data2=relu, op=nl.multiply)
                if transposed_relu:
                    # Preserve the accumulator's channel-major layout across
                    # the autograd boundary. One contiguous512-wide store
                    # replaces four transposes/stores for a normal token tile.
                    # Backward must consume the transposed view in its timing.
                    nisa.dma_copy(dst=saved_relu[hi * 128:(hi + 1) * 128,
                                                 m0:m0 + token_tile], src=relu)
                elif tiled_io:
                    for ti in nl.affine_range(token_tile // 128):
                        relu_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        relu_rows = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.nc_transpose(dst=relu_t, data=relu[:, ti * 128:(ti + 1) * 128])
                        nisa.tensor_copy(dst=relu_rows, src=relu_t)
                        nisa.dma_copy(dst=saved_relu[m0 + ti * 128:m0 + (ti + 1) * 128,
                                                    hi * 128:(hi + 1) * 128], src=relu_rows)
                else:
                    nisa.dma_copy(dst=saved_relu.ap(pattern=[[1, 128], [hidden, token_tile]],
                                                    offset=m0 * hidden + hi * 128), src=relu)
            for ci in nl.affine_range(nc):
                down_acc = nl.ndarray((128, token_tile), dtype=nl.float32, buffer=nl.psum)
                for hi in nl.affine_range(nh):
                    nisa.nc_matmul(dst=down_acc, stationary=down_cache[:, ci, hi, :],
                                   moving=activated[:, hi, :], accumulate=(hi > 0))
                output = nl.ndarray((128, token_tile), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=output, src=down_acc)
                if tiled_io:
                    for ti in nl.affine_range(token_tile // 128):
                        output_t = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                        output_rows = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.nc_transpose(dst=output_t, data=output[:, ti * 128:(ti + 1) * 128])
                        nisa.tensor_copy(dst=output_rows, src=output_t)
                        nisa.dma_copy(dst=result[m0 + ti * 128:m0 + (ti + 1) * 128,
                                                ci * 128:(ci + 1) * 128], src=output_rows)
                else:
                    nisa.dma_copy(dst=result.ap(pattern=[[1, 128], [channels, token_tile]],
                                                offset=m0 * channels + ci * 128), src=output)
        return result, saved_relu

    return fused_mlp


def reference_forward(x, up, down):
    relu = F.relu(F.linear(x, up.to(x.dtype)))
    return F.linear(relu.square(), down.to(x.dtype)), relu


def make_autograd(kernel_call, transposed_relu=False):
    class FusedMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, up, down):
            result, relu = kernel_call(x, up, down)
            ctx.save_for_backward(x, up, down, relu)
            return result

        @staticmethod
        def backward(ctx, grad):
            x, up, down, relu = ctx.saved_tensors
            if transposed_relu:
                # Keep HxT saved activation and hidden derivatives throughout
                # backward. The previous wrapper eagerly restored TxH, paying
                # a layout conversion before the autograd boundary.
                activated_t = relu.square()
                dhidden_t = down.to(x.dtype).T @ grad.T
                dpre_t = dhidden_t * (relu * 2.0)
                dx = (up.to(x.dtype).T @ dpre_t).T
                dup = (dpre_t @ x).to(up.dtype)
                ddown = (grad.T @ activated_t.T).to(down.dtype)
            else:
                activated = relu.square()
                dhidden = grad @ down.to(x.dtype)
                dpre = dhidden * (relu * 2.0)
                dx = dpre @ up.to(x.dtype)
                dup = (dpre.T @ x).to(up.dtype)
                ddown = (grad.T @ activated).to(down.dtype)
            return dx, dup, ddown

    return FusedMLP.apply
