"""Joint finite-BF16 softcapped exponential and softcap-derivative primitive.

This is not yet a cross-entropy kernel. It measures all table loads, index
arithmetic, gathers, analytic tails and output traffic against Scalar Engine
composition. The table has 5120 entries per field, replicated over128 lanes
(5MiB total). Saturated inputs clamp to the last active BF16 code of each sign.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def joint_functions(x, tables, use_table: bool = True, lanes: int = 2):
        """Return exp(15*tanh(x/15)) and d(15*tanh(x/15))/dx.

        Args:
            x: Contiguous finite BF16 [R,C], R divisible128*lanes, C%128=0.
            tables: FP32 [128,2,5120], identical precomputed rows; no data fit.
            use_table: False uses the current Scalar tanh/exp/square path.
            lanes: Matching physical-core launch grid within host LNC2.
        Returns:
            Two FP32 [R,C] outputs. Derivative is an analytic surrogate for
            later loss integration, not differentiation of integer indexing.
        """
        rows, cols = x.shape
        assert x.dtype == nl.bfloat16 and rows % (128 * lanes) == 0
        assert cols % 128 == 0 and lanes in (1, 2)
        assert tables.shape == (128, 2, 5120) and tables.dtype == nl.float32
        exponential = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        derivative = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        if use_table:
            lookup = nl.ndarray((128, 2, 5120), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lookup, src=tables)
        chunk = min(512, cols)
        for row in nl.affine_range(rows // (128 * lanes)):
            row_start = (row * lanes + nl.program_id(0)) * 128
            rr = nl.ds(row_start, 128)
            for column in nl.affine_range(cols // chunk):
                cc = nl.ds(column * chunk, chunk)
                source = nl.ndarray((128, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
                exps = nl.ndarray((128, chunk), dtype=nl.float32, buffer=nl.sbuf)
                derivs = nl.ndarray((128, chunk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=source, src=x[rr, cc])
                if use_table:
                    bits = nl.ndarray((128, chunk), dtype=nl.uint32, buffer=nl.sbuf)
                    magnitude = nl.ndarray((128, chunk), dtype=nl.uint32, buffer=nl.sbuf)
                    sign_offset = nl.ndarray((128, chunk), dtype=nl.uint32, buffer=nl.sbuf)
                    index = nl.ndarray((128, chunk), dtype=nl.uint32, buffer=nl.sbuf)
                    tiny = nl.ndarray((128, chunk), dtype=nl.uint32, buffer=nl.sbuf)
                    linear = nl.ndarray((128, chunk), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=bits, src=source.view(nl.uint16))
                    nisa.tensor_scalar(dst=magnitude, data=bits, op0=nl.bitwise_and, operand0=32767)
                    nisa.tensor_scalar(dst=sign_offset, data=bits, op0=nl.bitwise_and, operand0=32768)
                    nisa.tensor_scalar(dst=sign_offset, data=sign_offset,
                                       op0=nl.multiply, operand0=0.078125)  #2560/32768
                    nisa.tensor_scalar(dst=index, data=magnitude, op0=nl.maximum,
                                       operand0=14720, op1=nl.minimum, operand1=17279)
                    nisa.tensor_scalar(dst=index, data=index, op0=nl.subtract, operand0=14720)
                    nisa.tensor_tensor(dst=index, data1=index, data2=sign_offset, op=nl.add)
                    nisa.nc_n_gather(dst=exps, data=lookup[:, 0, :], indices=index)
                    nisa.nc_n_gather(dst=derivs, data=lookup[:, 1, :], indices=index)
                    nisa.tensor_scalar(dst=tiny, data=magnitude, op0=nl.less, operand0=14720)
                    nisa.tensor_scalar(dst=linear, data=source, op0=nl.add, operand0=1.0)
                    nisa.tensor_copy_predicated(dst=exps, src=linear, predicate=tiny)
                    nisa.tensor_copy_predicated(dst=derivs, src=1.0, predicate=tiny)
                else:
                    tangent = nl.ndarray((128, chunk), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=tangent, data=source, op=nl.tanh, scale=1.0 / 15.0)
                    nisa.activation(dst=exps, data=tangent, op=nl.exp, scale=15.0)
                    nisa.activation(dst=derivs, data=tangent, op=nl.square)
                    nisa.tensor_scalar(dst=derivs, data=derivs, op0=nl.subtract,
                                       operand0=1.0, reverse0=True)
                nisa.dma_copy(dst=exponential[rr, cc], src=exps)
                nisa.dma_copy(dst=derivative[rr, cc], src=derivs)
        return exponential, derivative

    return joint_functions
