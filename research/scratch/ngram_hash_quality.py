#!/usr/bin/env python3
"""Does replacing the PRIME modulus with a power-of-two MASK degrade the n-gram hash?

T3.1's `_ngram_hash_indices` uses `h % 262139` (largest prime <= 2^18) rather than
`h & (2^18 - 1)`, on the stated reasoning that "h % 2^k keeps only h's low k bits, which
throws away every bit of mixing the primes just put into the high half". That reasoning is
half right and the half that is wrong is the half that matters:

  * `(a ^ b) & M == (a & M) ^ (b & M)` for M = 2^k - 1, so masking the XOR is the XOR of
    the masked products -- the structure survives.
  * For an ODD multiplier p, `x -> (x*p) mod 2^18` is a BIJECTION on Z_{2^18}, hence
    INJECTIVE on the token ids (all < 2^14). So each masked product alone loses nothing.
  * What masking does lose is the high 13 bits of each 31-bit product, which is only a
    loss for the XOR's *combination* of the two. One `h ^= h >> 18` fold carries every one
    of those bits back into the kept window, at a cost of 2 integer ops.

This script measures it instead of arguing it: for each of the real per-layer prime pairs,
count DISTINCT rows and the collision-load distribution over the full (prev, cur) bigram
space and over a Zipf-weighted realistic sample. Pure CPU, no device needed.

    python3 research/scratch/ngram_hash_quality.py
"""

import argparse
import sys

import numpy as np


def _next_prime(n: int) -> int:
    def is_prime(m: int) -> bool:
        if m < 2:
            return False
        if m % 2 == 0:
            return m == 2
        f = 3
        while f * f <= m:
            if m % f == 0:
                return False
            f += 2
        return True
    while not is_prime(n):
        n += 1
    return n


PRIMES = tuple(_next_prime(8192 + 8000 * k) for k in range(30))


def largest_prime_below(n: int) -> int:
    m = n
    while _next_prime(m) != m:
        m -= 1
    return m


def bigram_primes(layer_idx: int) -> tuple[int, int]:
    i = (2 * layer_idx) % (len(PRIMES) - 1)
    return PRIMES[i], PRIMES[i + 1]


def h_prime(prev, cur, p0, p1, mod, mask):
    return np.bitwise_xor(prev * p0, cur * p1) % mod


def h_mask(prev, cur, p0, p1, mod, mask):
    h = np.bitwise_xor(prev * p0, cur * p1)
    h = np.bitwise_xor(h, np.right_shift(h, 18))
    return np.bitwise_and(h, mask)


def h_mask_nofold(prev, cur, p0, p1, mod, mask):
    return np.bitwise_and(np.bitwise_xor(prev * p0, cur * p1), mask)


SPELLINGS = {"prime%": h_prime, "mask+fold": h_mask, "mask(raw)": h_mask_nofold}


def report(name, rowvals, rows, npairs):
    counts = np.bincount(rowvals, minlength=rows)
    used = int((counts > 0).sum())
    # Expected distinct rows if the hash were a uniform random function.
    ideal = rows * (1.0 - (1.0 - 1.0 / rows) ** npairs)
    # Expected #pairs sharing a row with >=1 other pair, i.e. actual collision load.
    colliding = int(npairs - used)
    print(f"    {name:11s} distinct {used:>8,} / ideal {ideal:>10,.0f} "
          f"({used / ideal:6.4f} of ideal)   max_load {int(counts.max()):>3d}   "
          f"colliding_pairs {colliding:>8,} ({100.0 * colliding / npairs:5.2f}%)")
    return used / ideal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=262144)
    ap.add_argument("--vocab", type=int, default=8193, help="padded_vocab_size + 1 sentinel")
    ap.add_argument("--tables", type=int, default=3)
    ap.add_argument("--zipf-sample", type=int, default=4_000_000)
    args = ap.parse_args()

    rows, mask = args.rows, args.rows - 1
    mod = largest_prime_below(rows)
    assert rows & mask == 0
    V = args.vocab
    print(f"rows={rows} prime_mod={mod} mask=0x{mask:x} vocab(with sentinel)={V}")

    # ---- 1. FULL bigram space: every (prev, cur) pair the hash can ever see -----------
    prev_full = np.repeat(np.arange(V, dtype=np.int64), V)
    cur_full = np.tile(np.arange(V, dtype=np.int64), V)
    npairs = prev_full.size
    print(f"\n[1] full bigram space, {npairs:,} pairs")
    ratios = {}
    for layer in range(args.tables):
        p0, p1 = bigram_primes(layer)
        print(f"  layer {layer} primes ({p0}, {p1})")
        for name, fn in SPELLINGS.items():
            r = report(name, fn(prev_full, cur_full, p0, p1, mod, mask).astype(np.int64),
                       rows, npairs)
            ratios.setdefault(name, []).append(r)

    # ---- 2. Zipf-weighted realistic token stream -------------------------------------
    # Real text is far from uniform over the vocabulary, so the collision load that
    # actually matters is the one over FREQUENT bigrams.
    rng = np.random.default_rng(0)
    zipf_w = 1.0 / np.arange(1, V + 1) ** 1.0
    zipf_w /= zipf_w.sum()
    toks = rng.choice(V, size=args.zipf_sample + 1, p=zipf_w).astype(np.int64)
    prev_z, cur_z = toks[:-1], toks[1:]
    print(f"\n[2] Zipf(1.0)-sampled stream, {prev_z.size:,} positions "
          f"({np.unique(np.stack([prev_z, cur_z]), axis=1).shape[1]:,} distinct bigrams)")
    for layer in range(args.tables):
        p0, p1 = bigram_primes(layer)
        print(f"  layer {layer} primes ({p0}, {p1})")
        for name, fn in SPELLINGS.items():
            report(name, fn(prev_z, cur_z, p0, p1, mod, mask).astype(np.int64),
                   rows, prev_z.size)

    print("\nSUMMARY (full space, mean distinct/ideal over layers):")
    for name, vals in ratios.items():
        print(f"  {name:11s} {np.mean(vals):.6f}")
    print("\nA mask spelling within ~0.001 of the prime spelling on distinct-row count and "
          "max_load is a quality-neutral swap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
