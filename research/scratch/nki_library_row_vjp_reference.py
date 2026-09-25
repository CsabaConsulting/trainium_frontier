"""CPU-only algebra contract for a future NKI Library ROW-FP8 MLP backward.

This is deliberately *not* a training kernel or a measured speedup. It decodes
the legacy E4M3 bytes saved by ``nki_library_row_training_forward.py`` and
computes the straight-through VJP in float64, with all token-row and
output-channel scales on their correct axes. Device stage rounding and any
new gradient quantizer require separate, explicit numerical gates.

The forward has Xeff = diag(sx) Qx, Ueff = Qu diag(su),
Heff = diag(sh) Qh, Deff = Qd diag(sd). Its BF16 pre-square ReLU is saved.
The declared surrogate treats quantization/casts as identities locally but
uses these *actual quantized* operands and 2*saved_relu for the activation
derivative. The returned weight gradients have the transposed NKI layouts
Qup[C,H] and Qdown[H,C]; torch Linear masters use their transposes.

Do not use this module in a leaderboard submission or claim BPB/step gains
from its results. It exists to catch incorrect scale movement before any
device-side implementation is timed.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class RowVJP(NamedTuple):
    dx: np.ndarray
    dup_c_h: np.ndarray
    ddown_h_c: np.ndarray
    dpre_t_h: np.ndarray


def decode_legacy_e4m3(raw: np.ndarray) -> np.ndarray:
    """Decode finite legacy E4M3 bytes (bias 7, max 240) into float64.

    Exponent 15 is reserved by the legacy format and cannot be an operand in
    the gated ROW forward. Rejecting it catches a byte/layout mismatch rather
    than silently treating an invalid byte as a number.
    """
    bits = np.asarray(raw)
    if bits.dtype != np.uint8:
        raise TypeError("FP8 storage must be raw uint8 bytes")
    exponent = (bits >> 3) & 15
    if np.any(exponent == 15):
        raise ValueError("reserved legacy-E4M3 exponent in saved operand")
    mantissa = bits & 7
    normal = np.ldexp(1.0 + mantissa.astype(np.float64) / 8.0,
                       exponent.astype(np.int16) - 7)
    subnormal = mantissa.astype(np.float64) * (2.0 ** -9)
    magnitude = np.where(exponent == 0, subnormal, normal)
    return np.where((bits & 128) != 0, -magnitude, magnitude)


def _bytes(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != shape:
        raise ValueError(f"{name}: expected uint8 {shape}, got {array.dtype} {array.shape}")
    return decode_legacy_e4m3(array)


def _scale(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (size, 1):
        array = array[:, 0]
    if array.shape != (size,):
        raise ValueError(f"{name}: expected ({size},) or ({size}, 1)")
    if not np.all(np.isfinite(array)) or not np.all(array > 0):
        raise ValueError(f"{name}: scales must be positive and finite")
    return array


def _finite(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: expected finite {shape}")
    return array


def row_fp8_vjp(
    x_bytes: np.ndarray,
    x_scale: np.ndarray,
    h_bytes: np.ndarray,
    h_scale: np.ndarray,
    relu: np.ndarray,
    up_bytes: np.ndarray,
    up_scale: np.ndarray,
    down_bytes: np.ndarray,
    down_scale: np.ndarray,
    gradient: np.ndarray,
) -> RowVJP:
    """Return the four ROW-FP8 surrogate products with scale axes preserved.

    ``x_bytes[T,C]``, ``h_bytes[T,H]``, ``up_bytes[C,H]`` and
    ``down_bytes[H,C]`` are the actual saved/weight bytes. Input and hidden
    scales vary by token; weight scales vary by output channel. No new gradient
    rounding is hidden here. In particular, token scales remain *inside* the
    T-reduction of each weight-gradient product.
    """
    x_shape = np.asarray(x_bytes).shape
    h_shape = np.asarray(h_bytes).shape
    if len(x_shape) != 2 or len(h_shape) != 2:
        raise ValueError("input and hidden bytes must be matrices")
    tokens, channels = x_shape
    h_tokens, hidden = h_shape
    if tokens == 0 or channels == 0 or hidden == 0 or h_tokens != tokens:
        raise ValueError("incompatible nonempty token/channel/hidden dimensions")

    qx = _bytes(x_bytes, (tokens, channels), "x_bytes")
    qh = _bytes(h_bytes, (tokens, hidden), "h_bytes")
    qu = _bytes(up_bytes, (channels, hidden), "up_bytes")
    qd = _bytes(down_bytes, (hidden, channels), "down_bytes")
    sx = _scale(x_scale, tokens, "x_scale")
    sh = _scale(h_scale, tokens, "h_scale")
    su = _scale(up_scale, hidden, "up_scale")
    sd = _scale(down_scale, channels, "down_scale")
    saved_relu = _finite(relu, (tokens, hidden), "relu")
    gy = _finite(gradient, (tokens, channels), "gradient")
    if np.any(saved_relu < 0):
        raise ValueError("saved pre-square ReLU must be nonnegative")

    # Data gradients: output-channel scales can move into incoming columns.
    dh = (gy * sd[None, :]) @ qd.T
    dpre = (2.0 * saved_relu) * dh
    dx = (dpre * su[None, :]) @ qu.T

    # Weight gradients: sx/sh depend on the contraction (token) axis and
    # therefore cannot be multiplied onto the finished matrix product.
    ddown = qh.T @ (gy * sh[:, None])
    dup = qx.T @ (dpre * sx[:, None])
    return RowVJP(dx=dx, dup_c_h=dup, ddown_h_c=ddown, dpre_t_h=dpre)
