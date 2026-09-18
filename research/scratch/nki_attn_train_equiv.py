"""CPU durable check for the train.py --nki-attn integration (P1 NKI port).

Covers what is testable without a Neuron device:
  1. The flag exists and defaults OFF (baseline numerics unchanged).
  2. With the flag forced ON on CPU, nki_attention() still takes the eager
     SDPA path and matches F.scaled_dot_product_attention bit-exactly
     (device/dtype/shape guards fall back safely off-device).
  3. The mask-cache helper builds the exact causal pattern.

The on-device kernel numerics live in nki_attn_fwd.py --selftest /
nki_attn_bwd.py (--selftest numpy + --hw on hardware); the 300-step smoke
covers the integrated training path.

Run from the repo root:  python3 research/scratch/nki_attn_train_equiv.py
"""

import sys

import torch
import torch.nn.functional as F


def main() -> bool:
    sys.path.insert(0, ".")
    import train as T

    ok = True

    # 1. default OFF
    print("USE_NKI_ATTN default:", T.USE_NKI_ATTN, flush=True)
    ok &= T.USE_NKI_ATTN is False

    # 2. forced-ON fallback is bit-exact eager SDPA on CPU
    T.USE_NKI_ATTN = True
    try:
        torch.manual_seed(0)
        for shape in [(1, 2, 128, 128), (2, 5, 512, 128)]:
            q = torch.randn(*shape, dtype=torch.bfloat16)
            k = torch.randn(*shape, dtype=torch.bfloat16)
            v = torch.randn(*shape, dtype=torch.bfloat16)
            for scale in (None, 0.05):
                got = T.nki_attention(q, k, v, scale)
                want = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                      scale=scale)
                same = bool(torch.equal(got, want))
                print(f"fallback exact shape={shape} scale={scale}: {same}",
                      flush=True)
                ok &= same and got.shape == want.shape
    finally:
        T.USE_NKI_ATTN = False

    # 3. mask pattern (built on CPU here; device version is identical math)
    m = T._attn_nki_mask(256, torch.device("cpu"), torch.bfloat16)
    want = (torch.arange(256).unsqueeze(1) <= torch.arange(256).unsqueeze(0))
    same = bool(torch.equal(m, torch.where(
        want, torch.tensor(1.0), torch.tensor(0.0)).to(torch.bfloat16)))
    print("mask exact:", same, m.shape, m.dtype, flush=True)
    ok &= same and m.shape == (256, 256)

    print("TRAIN-EQUIV", "PASS" if ok else "FAIL", flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
