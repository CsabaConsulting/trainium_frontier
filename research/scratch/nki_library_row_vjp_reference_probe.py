#!/usr/bin/env python3
"""CPU-only gates for the ROW-FP8 VJP scale-axis contract; no Trainium run."""

import unittest

import numpy as np

from nki_library_row_vjp_reference import decode_legacy_e4m3, row_fp8_vjp


class RowVJPReferenceTest(unittest.TestCase):
    def test_legacy_decoder_and_reserved_exponent(self):
        raw = np.array([0x00, 0x80, 0x01, 0x38, 0x77, 0xB8], dtype=np.uint8)
        expected = np.array([0.0, -0.0, 2.0 ** -9, 1.0, 240.0, -1.0])
        np.testing.assert_array_equal(decode_legacy_e4m3(raw), expected)
        with self.assertRaises(ValueError):
            decode_legacy_e4m3(np.array([0x78], dtype=np.uint8))

    def test_distinct_token_scales_stay_inside_weight_reduction(self):
        # All byte patterns are finite, representable legacy E4M3 values.
        x = np.array([[0x38, 0x40], [0x40, 0xB8], [0x42, 0x3C]], dtype=np.uint8)
        h = np.array([[0x38, 0x40, 0x3C], [0x40, 0xB8, 0x38],
                      [0x42, 0x3C, 0x40]], dtype=np.uint8)
        up = np.array([[0x38, 0x40, 0xB8], [0x40, 0x38, 0x3C]], dtype=np.uint8)
        down = np.array([[0x38, 0x40], [0xB8, 0x3C], [0x40, 0x38]], dtype=np.uint8)
        sx = np.array([0.125, 2.0, 0.5])
        sh = np.array([1.0, 0.25, 4.0])
        su = np.array([0.5, 2.0, 0.25])
        sd = np.array([1.5, 0.75])
        relu = np.array([[0.0, 1.0, 2.0], [0.5, 0.0, 3.0], [1.0, 2.0, 0.25]])
        gy = np.array([[1e-30, 0.5], [2.0, -1.0], [-0.25, 1.5]])
        got = row_fp8_vjp(x, sx, h, sh, relu, up, su, down, sd, gy)

        qx, qh = decode_legacy_e4m3(x), decode_legacy_e4m3(h)
        qu, qd = decode_legacy_e4m3(up), decode_legacy_e4m3(down)
        xeff = qx * sx[:, None]
        heff = qh * sh[:, None]
        ueff = qu * su[None, :]
        deff = qd * sd[None, :]
        dpre = (gy @ deff.T) * (2.0 * relu)
        np.testing.assert_allclose(got.dpre_t_h, dpre, rtol=1e-14, atol=0)
        np.testing.assert_allclose(got.dx, dpre @ ueff.T, rtol=1e-14, atol=0)
        np.testing.assert_allclose(got.dup_c_h, xeff.T @ dpre, rtol=1e-14, atol=0)
        np.testing.assert_allclose(got.ddown_h_c, heff.T @ gy, rtol=1e-14, atol=0)

        # Independent local surrogate: at the saved point it uses Heff for
        # the down-gradient, and 2*ReLU for the up/input derivatives.
        def loss(xvalue, uvalue, dvalue):
            pre_delta = xvalue @ uvalue - xeff @ ueff
            return np.sum(((heff + 2.0 * relu * pre_delta) @ dvalue) * gy)

        eps = 1e-5
        for name, original, expected in (
            ("input", xeff, got.dx),
            ("up", ueff, got.dup_c_h),
            ("down", deff, got.ddown_h_c),
        ):
            for index in np.ndindex(original.shape):
                plus, minus = original.copy(), original.copy()
                plus[index] += eps
                minus[index] -= eps
                args_plus = dict(xvalue=xeff, uvalue=ueff, dvalue=deff)
                args_minus = args_plus.copy()
                key = {"input": "xvalue", "up": "uvalue", "down": "dvalue"}[name]
                args_plus[key], args_minus[key] = plus, minus
                measured = (loss(**args_plus) - loss(**args_minus)) / (2.0 * eps)
                self.assertAlmostEqual(measured, expected[index], places=8)

    def test_invalid_layout_or_scale_fails_closed(self):
        one = np.array([[0x38]], dtype=np.uint8)
        valid = [one, [1.0], one, [1.0], [[1.0]], one, [1.0],
                 one, [1.0], [[1.0]]]
        row_fp8_vjp(*valid)
        bad = list(valid)
        bad[1] = [-1.0]
        with self.assertRaises(ValueError):
            row_fp8_vjp(*bad)
        bad = list(valid)
        bad[5] = np.array([[0x78]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            row_fp8_vjp(*bad)


if __name__ == "__main__":
    unittest.main()
