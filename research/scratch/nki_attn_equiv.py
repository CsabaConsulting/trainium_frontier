"""CPU reference + equivalence harness for the P1 NKI attention-forward port.

Target (train.py CausalSelfAttention.forward, lines ~2561-2568): q/k/v arrive
[b, t, h, d] post-rope, transpose to [b, h, t, d], causal scaled-dot-product
attention, transpose back. Locked shape: h=5, d=128, t=2048, bf16, scale None
(= 1/sqrt(d)). attn_mask path (non-None) is out of scope for the port: the
locked config trains with mask None (is_causal=True branch).

Usage:
  python3 research/scratch/nki_attn_equiv.py          # self-test (reduced sizes)
  from research.scratch.nki_attn_equiv import ref_attn # (import from repo root)
Unit under test (when written): research/scratch/nki_attn_fwd.py::nki_attn_fwd,
called with the same [b, t, h, d] bf16 inputs; compare with check().
"""

import numpy as np


def ref_attn(q, k, v, scale=None):
    """Causal scaled-dot-product attention. q/k/v: [b, t, h, d] float32/float64
    numpy (upcast bf16 before calling). Returns [b, t, h, d]."""
    b, t, h, d = q.shape
    s = 1.0 / np.sqrt(d) if scale is None else scale
    qt = q.transpose(0, 2, 1, 3)  # [b, h, t, d]
    kt = k.transpose(0, 2, 1, 3)
    vt = v.transpose(0, 2, 1, 3)
    scores = np.matmul(qt, kt.transpose(0, 1, 3, 2)) * s
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores[..., mask] = -1e9
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / e.sum(axis=-1, keepdims=True)
    y = np.matmul(p, vt)
    return y.transpose(0, 2, 1, 3)


def check(got, want, tag="", rel_gate=True):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    adiff = np.max(np.abs(got - want))
    denom = np.maximum(np.abs(want), 1e-3)
    rdiff = np.max(np.abs(got - want) / denom)
    # bf16 rows gate on abs only: rel explodes in near-zero outputs while
    # abs sits at the format floor (~4e-3); fp32 rows gate on both.
    ok = adiff < 5e-2 and (rdiff < 5e-2 or not rel_gate)
    print(f"{tag}: {'PASS' if ok else 'FAIL'} "
          f"(max abs {adiff:.6f}, max rel {rdiff:.6f})", flush=True)
    return ok


def selftest():
    rng = np.random.default_rng(42)
    allok = True
    for (b, t, h, d) in [(1, 128, 2, 32), (2, 256, 5, 128)]:
        q = rng.standard_normal((b, t, h, d)).astype(np.float32)
        k = rng.standard_normal((b, t, h, d)).astype(np.float32)
        v = rng.standard_normal((b, t, h, d)).astype(np.float32)
        y = ref_attn(q, k, v)
        assert y.shape == (b, t, h, d)
        # causality: output[t] must not depend on inputs[>t]
        k2 = k.copy()
        k2[:, -1] += 10.0
        y2 = ref_attn(q, k2, v)
        allok &= check(y2[:, :-1], y[:, :-1], f"causal b{b}t{t}")
        # scale path
        ys = ref_attn(q, k, v, scale=0.1)
        allok &= check(ys.shape, y.shape, "scale-shape")
    # bf16 round-trip tolerance at locked dims (reduced t for speed)
    q = rng.standard_normal((2, 512, 5, 128)).astype(np.float32)
    y32 = ref_attn(q, q.copy(), q.copy())
    q16 = q.astype(np.float16).astype(np.float32)
    y16 = ref_attn(q16, q16.copy(), q16.copy())
    allok &= check(y16, y32, "fp16-noise")
    print("ALL PASS" if allok else "FAILURES", flush=True)
    return allok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
