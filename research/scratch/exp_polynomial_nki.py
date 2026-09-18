"""Bounded exponential alternatives for the soft-capped loss's [-30,30] range.

This is an isolated experiment, not an installed loss or attention kernel.
Compare the Scalar Engine's built-in piecewise approximation with arithmetic
on the Vector Engine. Coefficients come from Chebyshev interpolation over
[-.694,.694], with FP32 coefficients and an exact constant term of one.
They are not claimed to be optimal minimax coefficients.
"""


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def exp_kernel(x, method: int = 0, lanes: int = 2):
        rows, cols = x.shape
        assert x.dtype == nl.float32 and cols <= 2048
        assert lanes in (1, 2) and rows % (128 * lanes) == 0
        assert method in (0, 1, 2)
        if method == 1:
            coefficients = (1.0, 1.0000007152557373, 0.49981650710105896,
                            0.1666405200958252, 0.042679835110902786,
                            0.008477761410176754)
        else:
            coefficients = (1.0, 1.0, 0.5000007152557373,
                            0.16666674613952637, 0.0416594035923481,
                            0.008332528173923492, 0.001412956160493195,
                            0.0002010832540690899)
        degree = len(coefficients) - 1
        output = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        for tile in nl.affine_range(rows // (128 * lanes)):
            first = (tile * lanes + nl.program_id(0)) * 128
            source = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            result = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=source, src=x[first:first + 128, :])
            if method == 0:
                nisa.activation(dst=result, op=nl.exp, data=source)
            else:
                exponent = nl.ndarray((128, cols), dtype=nl.int32, buffer=nl.sbuf)
                reduced = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
                temporary = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
                scale_bits = nl.ndarray((128, cols), dtype=nl.uint32, buffer=nl.sbuf)
                # Integer rounding mode need not be assumed: truncation,
                # floor or nearest all leave |x-n*ln2| <=ln2 plus rounding.
                # The chosen polynomial interval includes that margin.
                nisa.tensor_scalar(dst=exponent, data=source, op0=nl.multiply,
                                   operand0=1.4426950408889634, engine=nisa.engine.vector)
                # Split ln2 avoids cancellation of x against n*ln2 rounded
                # as a single constant. n*ln2_hi is exact for |n| <=44.
                nisa.scalar_tensor_tensor(dst=reduced, data=exponent,
                                          op0=nl.multiply, operand0=-0.693145751953125,
                                          op1=nl.add, operand1=source)
                nisa.scalar_tensor_tensor(dst=reduced, data=exponent,
                                          op0=nl.multiply, operand0=-1.428606765330187e-6,
                                          op1=nl.add, operand1=reduced)
                nisa.tensor_scalar(dst=result, data=reduced, op0=nl.multiply,
                                   operand0=coefficients[degree], op1=nl.add,
                                   operand1=coefficients[degree - 1], engine=nisa.engine.vector)
                for k in nl.static_range(degree - 1):
                    nisa.tensor_tensor(dst=temporary, data1=result, data2=reduced,
                                       op=nl.multiply, engine=nisa.engine.vector)
                    nisa.tensor_scalar(dst=result, data=temporary, op0=nl.add,
                                       operand0=coefficients[degree - k - 2],
                                       engine=nisa.engine.vector)
                # Construct exact powers of two using IEEE754 exponent bits.
                # n is small, and these integer values are exactly representable
                # by the Vector Engine's intermediate FP32 arithmetic.
                nisa.tensor_scalar(dst=scale_bits, data=exponent, op0=nl.add,
                                   operand0=127.0, op1=nl.multiply, operand1=8388608.0,
                                   engine=nisa.engine.vector)
                nisa.tensor_tensor(dst=result, data1=result,
                                   data2=scale_bits.view(nl.float32), op=nl.multiply,
                                   engine=nisa.engine.vector)
            nisa.dma_copy(dst=output[first:first + 128, :], src=result)
        return output

    return exp_kernel
