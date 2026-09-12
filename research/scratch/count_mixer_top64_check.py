"""T4.1 step 5: quantify the §6 "top-64 approximation" caveat directly.

The doc's §6 worried that mixing with only the top-64 neural probabilities introduces
error. `count_mixer_dump.py` sidesteps that by storing the EXACT log p_neural(target)
and the EXACT full-softmax entropy, so the mixture bpb it feeds `count_mixer_fit.py` is
exact. This script measures what the approximation would have cost, from the
`fulldist_*.npy` slice ([N, 8192] fp16 log-probs) that the dump also writes:

  1. how much probability mass the top-64 actually covers,
  2. bpb of the mixture computed from a top-64-truncated p_neural (residual mass spread
     uniformly over the tail, and the alternative of handing it to the count model),
     against the exact mixture,
  3. entropy computed from the top-64 vs the exact entropy (the gate feature).

Pure CPU. Run: python research/scratch/count_mixer_top64_check.py --dir ~/countmix
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
import prepare as P
from count_mixer_fit import CountModel, bpb_from_logp, features, load_stream

V = P.TOKENIZER_VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stream", default="public_val")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--discount", type=float, default=0.5)
    ap.add_argument("--discount3", type=float, default=0.75)
    ap.add_argument("--lam", type=float, default=0.0125, help="global lambda to evaluate at")
    ap.add_argument("--topk", type=int, default=64)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    d = os.path.expanduser(args.dir)
    tag = f"{args.stream}{args.suffix}_seq{args.seq_len}"
    full = np.load(os.path.join(d, f"fulldist_{tag}.npy"), mmap_mode="r")
    n = full.shape[0]
    print(f"full-dist slice: {n:,} positions x {full.shape[1]}")

    s = load_stream(d, args.stream, args.seq_len, args.suffix)
    for k in ("x", "y", "valid", "prev", "prev2", "has_prev2", "logp", "entropy", "topk_mass"):
        s[k] = s[k][:n]
    token_bytes = P.get_token_bytes("cpu").numpy().astype(np.int64)
    s["nbytes"] = token_bytes[s["y"]]

    cm = CountModel(d, args.discount)
    f = features(s, cm, args.discount3)
    p_c = f["p_c"]

    lp_full = np.asarray(full, dtype=np.float32)
    # top-K truncation of the neural distribution
    part = np.argpartition(-lp_full, args.topk - 1, axis=1)[:, :args.topk]
    topk_lp = np.take_along_axis(lp_full, part, axis=1).astype(np.float64)
    mass = np.exp(topk_lp).sum(1)
    tgt_in_topk = (part == s["y"][:, None]).any(1)
    p_tgt_exact = np.exp(np.take_along_axis(lp_full, s["y"][:, None], axis=1)
                         .astype(np.float64))[:, 0]
    del lp_full

    # (i) truncated p_neural: keep top-K, spread the residual mass uniformly over the tail
    tail_p = (1.0 - mass) / (V - args.topk)
    p_tgt_trunc = np.where(tgt_in_topk, p_tgt_exact, tail_p)
    # (ii) truncated p_neural, residual handed to the count model instead
    p_tgt_trunc2 = np.where(tgt_in_topk, p_tgt_exact, (1.0 - mass) * p_c)

    lam = args.lam
    out = {
        "positions": int(n),
        "topk": args.topk,
        "mean_top64_mass": float(mass[s["valid"]].mean()),
        "p10_top64_mass": float(np.quantile(mass[s["valid"]], 0.10)),
        "frac_target_in_top64": float(tgt_in_topk[s["valid"]].mean()),
        "max_abs_logp_err_vs_dump": float(np.abs(np.log(p_tgt_exact) - s["logp"]).max()),
    }
    for name, p_n in (("exact", p_tgt_exact), ("trunc_uniform_tail", p_tgt_trunc),
                      ("trunc_count_tail", p_tgt_trunc2)):
        lp_mix = np.log(np.maximum((1 - lam) * p_n + lam * p_c, 1e-300))
        out[f"bpb_mix_{name}"] = bpb_from_logp(lp_mix, s["nbytes"], s["valid"])
        out[f"bpb_neural_{name}"] = bpb_from_logp(np.log(np.maximum(p_n, 1e-300)),
                                                 s["nbytes"], s["valid"])

    # entropy feature: exact vs top-64 estimate (top-K terms + uniform tail)
    p_top = np.exp(topk_lp)
    ent_top = -(p_top * topk_lp).sum(1)
    tail = np.maximum(1.0 - mass, 0.0)
    ent_trunc = ent_top - tail * np.log(np.maximum(tail / (V - args.topk), 1e-30))
    out["entropy_mae_top64"] = float(np.abs(ent_trunc - s["entropy"][:n]).mean())
    out["entropy_mean_exact"] = float(s["entropy"][:n].mean())
    print(json.dumps(out, indent=2, sort_keys=True))
    with open(os.path.join(d, f"top64_check_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
