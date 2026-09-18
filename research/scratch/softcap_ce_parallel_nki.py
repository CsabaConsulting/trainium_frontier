"""Scratch loss scheduling/gather variants; preserve train.py's numerical math.

The original target-relative shift, 512-wide exp reductions and backward
derivative are retained. No function approximation is added here. This tests
whether avoiding a mask/reduction and using both physical cores is profitable.
"""


def build_kernels():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def forward(z, tgt, lanes: int = 2, target_gather: bool = False):
        """Compute unreduced loss and saved logsumexp with unchanged softcap15.

        Args:
            z: Contiguous FP32 [N, V] logits.
            tgt: FP32 [N, 1] class indices, exact integers in [0, V).
            lanes: Physical cores in the matching launch grid.
            target_gather: Gather one SBUF element per row instead of mask/sum.
        Returns:
            FP32 [N, 1] loss and logsumexp, as in the original kernel.
        """
        rows_count, vocab = z.shape
        partitions = 128
        softcap = 15.0
        assert lanes in (1, 2) and rows_count % (partitions * lanes) == 0
        assert z.dtype == nl.float32 and tgt.dtype == nl.float32
        assert tgt.shape == (rows_count, 1) and vocab % 512 == 0
        loss = nl.ndarray((rows_count, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        lse = nl.ndarray((rows_count, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        if not target_gather:
            indices = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=indices, pattern=[[1, vocab]], offset=0, channel_multiplier=0)
        chunk = 512
        chunks = vocab // chunk
        for tile in nl.affine_range(rows_count // (partitions * lanes)):
            first = (tile * lanes + nl.program_id(0)) * partitions
            rows = nl.ds(first, partitions)
            logits = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            labels = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=logits, src=z[rows])
            nisa.dma_copy(dst=labels, src=tgt[rows])
            capped_unit = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=capped_unit, op=nl.tanh, data=logits, scale=1.0 / softcap)
            target_unit = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            if target_gather:
                gather_indices = nl.ndarray((partitions, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gather_indices, src=labels)
                # One output per partition avoids the large-output gather
                # composition that failed in the earlier embedding experiment.
                nisa.nc_n_gather(dst=target_unit, data=capped_unit, indices=gather_indices)
            else:
                mask = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mask, data=indices, op0=nl.equal, operand0=labels)
                nisa.tensor_tensor(dst=mask, data1=mask, data2=capped_unit, op=nl.multiply)
                nisa.tensor_reduce(dst=target_unit, op=nl.add, data=mask, axis=(1,))
            target_value = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_target = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=target_value, data=target_unit,
                               op0=nl.multiply, operand0=softcap)
            nisa.tensor_scalar(dst=negative_target, data=target_unit,
                               op0=nl.multiply, operand0=-softcap)
            partial_sums = nl.ndarray((partitions, chunks), dtype=nl.float32, buffer=nl.sbuf)
            for part in nl.affine_range(chunks):
                columns = nl.ds(part * chunk, chunk)
                nisa.activation(dst=logits[:, columns], op=nl.exp,
                                data=capped_unit[:, columns], scale=softcap,
                                bias=negative_target, reduce_op=nl.add,
                                reduce_res=partial_sums[:, nl.ds(part, 1)],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            denominator = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=denominator, op=nl.add, data=partial_sums, axis=(1,))
            log_denominator = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=log_denominator, op=nl.log, data=denominator)
            logsumexp = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=logsumexp, data1=target_value,
                               data2=log_denominator, op=nl.add)
            nisa.dma_copy(dst=loss[rows], src=log_denominator)
            nisa.dma_copy(dst=lse[rows], src=logsumexp)
        return loss, lse

    @nki.jit
    def backward(z, tgt, lse, g, lanes: int = 2):
        """Return FP32 [N, V] logits gradients, partitioning independent rows.

        Args:
            z, tgt, lse: Same inputs and saved logsumexp as forward.
            g: Arbitrary FP32 [N, 1] upstream loss gradient.
            lanes: Physical cores in the matching launch grid.
        Returns:
            Original softcap15/CE derivative, with unchanged rounding points.
        """
        rows_count, vocab = z.shape
        partitions = 128
        softcap = 15.0
        assert lanes in (1, 2) and rows_count % (partitions * lanes) == 0
        assert z.dtype == nl.float32 and tgt.dtype == nl.float32
        assert lse.dtype == nl.float32 and g.dtype == nl.float32
        assert tgt.shape == (rows_count, 1) and lse.shape == (rows_count, 1)
        assert g.shape == (rows_count, 1)
        dz = nl.ndarray(z.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        indices = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=indices, pattern=[[1, vocab]], offset=0, channel_multiplier=0)
        for tile in nl.affine_range(rows_count // (partitions * lanes)):
            rows = nl.ds((tile * lanes + nl.program_id(0)) * partitions, partitions)
            logits = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            labels = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            gradient = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_lse = nl.ndarray((partitions, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=logits, src=z[rows])
            nisa.dma_copy(dst=labels, src=tgt[rows])
            nisa.dma_copy(dst=gradient, src=g[rows])
            nisa.dma_copy(dst=negative_lse, src=lse[rows])
            nisa.tensor_scalar(dst=negative_lse, data=negative_lse,
                               op0=nl.multiply, operand0=-1.0)
            capped_unit = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=capped_unit, op=nl.tanh, data=logits, scale=1.0 / softcap)
            nisa.activation(dst=logits, op=nl.exp, data=capped_unit,
                            scale=softcap, bias=negative_lse)
            temporary = nl.ndarray((partitions, vocab), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=temporary, data=indices, op0=nl.equal,
                               operand0=labels, op1=nl.multiply, operand1=gradient)
            nisa.scalar_tensor_tensor(dst=logits, data=logits, op0=nl.multiply,
                                      operand0=gradient, op1=nl.subtract, operand1=temporary)
            nisa.activation(dst=temporary, op=nl.square, data=capped_unit)
            nisa.scalar_tensor_tensor(dst=capped_unit, data=temporary,
                                      op0=nl.subtract, operand0=1.0, reverse0=True,
                                      op1=nl.multiply, operand1=logits)
            nisa.dma_copy(dst=dz[rows], src=capped_unit)
        return dz

    return forward, backward
