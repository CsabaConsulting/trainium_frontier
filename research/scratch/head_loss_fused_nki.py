"""Forward-only head/softcap-CE fusion with the ordinary FP32 logits save.

The first experiment removes a logits reread while retaining the existing
backward input. It changes no special-function approximation and performs no
gradient filtering. Weight transpose/preparation is a separate charged cost.
Installed NKI0.5, NC-v3, BF16 operands and FP32 TensorEngine accumulators.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def forward(x, weight_t, target, lanes: int = 2):
        """Return FP32 logits[N,V], loss[N,1] and saved logsumexp[N,1].

        x is BF16[N,C]; weight_t is contiguous BF16[C,V]. Targets are exact
        integer-valued FP32[N,1]. Supported rows are at least512 and divisible
        by256; widths are multiples128 and vocabulary is a multiple512.
        This is not an autograd integration or a recomputing backward kernel.
        """
        tokens, channels = x.shape
        weight_channels, vocab = weight_t.shape
        assert channels == weight_channels and channels % 128 == 0
        assert x.dtype == weight_t.dtype == nl.bfloat16
        assert target.dtype == nl.float32 and target.shape == (tokens, 1)
        assert lanes == 2 and tokens >= 512 and tokens % 256 == 0
        assert vocab % 512 == 0
        # Stream weights instead of retaining a16MiB cache beside the two
        # full-vocabulary FP32 row tiles. Count temporaries and reserve2MiB.
        live_bytes = 4 * 128 * channels + 8 * 128 * vocab + 6 * 128 * 512
        assert live_bytes + 2 * 1024**2 <= nl.tile_size.sbuf_size_bytes
        logits_out = nl.ndarray((tokens, vocab), dtype=nl.float32, buffer=nl.shared_hbm)
        loss_out = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        lse_out = nl.ndarray((tokens, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nc, nv = channels // 128, vocab // 512
        for tile in nl.affine_range(tokens // (128 * lanes)):
            rows = nl.ds((tile * lanes + nl.program_id(0)) * 128, 128)
            input_rows = nl.ndarray((128, channels), dtype=nl.bfloat16, buffer=nl.sbuf)
            inputs = nl.ndarray((128, nc, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=input_rows, src=x[rows, :])
            for ci in nl.affine_range(nc):
                transposed = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed, data=input_rows[:, ci * 128:(ci + 1) * 128])
                nisa.tensor_copy(dst=inputs[:, ci, :], src=transposed)
            logits = nl.ndarray((128, vocab), dtype=nl.float32, buffer=nl.sbuf)
            for vi in nl.affine_range(nv):
                columns = nl.ds(vi * 512, 512)
                accumulator = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for ci in nl.affine_range(nc):
                    weights = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=weights, src=weight_t[ci * 128:(ci + 1) * 128, columns])
                    nisa.nc_matmul(dst=accumulator, stationary=inputs[:, ci, :],
                                   moving=weights, accumulate=(ci > 0))
                nisa.tensor_copy(dst=logits[:, columns], src=accumulator)
            # Preserve the original FP32 accumulator boundary into loss and
            # the ordinary backward save. No BF16 rounding of the logits.
            nisa.dma_copy(dst=logits_out[rows, :], src=logits)
            capped = nl.ndarray((128, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=capped, data=logits, op=nl.tanh, scale=1.0 / 15.0)
            labels = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            indices = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            target_unit = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=labels, src=target[rows, :])
            nisa.tensor_copy(dst=indices, src=labels)
            nisa.nc_n_gather(dst=target_unit, data=capped, indices=indices)
            target_value = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_target = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=target_value, data=target_unit, op0=nl.multiply, operand0=15.0)
            nisa.tensor_scalar(dst=negative_target, data=target_unit, op0=nl.multiply, operand0=-15.0)
            partial = nl.ndarray((128, nv), dtype=nl.float32, buffer=nl.sbuf)
            for vi in nl.affine_range(nv):
                columns = nl.ds(vi * 512, 512)
                nisa.activation(dst=logits[:, columns], data=capped[:, columns], op=nl.exp,
                                scale=15.0, bias=negative_target, reduce_op=nl.add,
                                reduce_res=partial[:, nl.ds(vi, 1)],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            denominator = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            logarithm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            lse = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=denominator, data=partial, op=nl.add, axis=(1,))
            nisa.activation(dst=logarithm, data=denominator, op=nl.log)
            nisa.tensor_tensor(dst=lse, data1=target_value, data2=logarithm, op=nl.add)
            nisa.dma_copy(dst=loss_out[rows, :], src=logarithm)
            nisa.dma_copy(dst=lse_out[rows, :], src=lse)
        return logits_out, loss_out, lse_out

    return forward
