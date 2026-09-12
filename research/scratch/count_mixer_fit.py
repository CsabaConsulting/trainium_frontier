"""T4.1 steps 3-4: smooth the count tables, fit the mixing gate, report bpb.

Part of the §6 offline protocol in research/ngram-drafter-and-count-mixer-2026-09-12.md.
Consumes the outputs of `count_mixer_dump.py` (per-token neural log-probs) and
`count_mixer_tables.py` (counts from train shards 0-3) and reports

  bpb(neural alone) vs bpb((1-g)*p_neural + g*p_count)

on stream (a) = the public val shard and stream (b) = held-out train shard 4, with
every gate fitted on (b) ONLY.

Why this is exact rather than a top-64 approximation: both mixands are proper
distributions over the 8192 real tokens, so
    log p_mix(y_t) = log((1-g)*p_neural(y_t) + g*p_count(y_t))
needs only the two *target* probabilities, both of which are exact in the dump. The
top-64 arrays are used only for context-only gate features (and to quantify coverage).

Gates, in the doc's order of simplicity:
  1. global lambda
  2. per-order lambda (order = longest context suffix present in the tables)
  3. 3+-feature logistic gate on context-only features
  4. non-parametric (order x neural-entropy-decile) lambda table -- an upper bound on
     what any smooth gate of those features could do

Pure CPU. Run: python research/scratch/count_mixer_fit.py --dir ~/countmix
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import prepare as P

V = P.TOKENIZER_VOCAB_SIZE
LN2 = math.log(2.0)
EPS_UNIFORM = 1e-7   # floor mixed into the count unigram so p_count > 0 everywhere


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding dump_*.npz and the tables")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--discount", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    ap.add_argument("--discount3", type=float, default=0.75)
    ap.add_argument("--suffix", default="", help="dump-file suffix, matching count_mixer_dump.py")
    ap.add_argument("--out", default=None, help="where to write the result json")
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Smoothing


class CountModel:
    """Interpolated Kneser-Ney bigram over the dense count table, with an optional
    exact-trigram layer on top (stupid-backoff-free: proper interpolation)."""

    def __init__(self, d: str, discount: float, use_kn_unigram: bool = True):
        t0 = time.monotonic()
        self.discount = discount
        bigram = np.load(os.path.join(d, "bigram.npy"), mmap_mode="r")
        self.c2 = np.asarray(bigram, dtype=np.int32)
        self.unigram = np.load(os.path.join(d, "unigram.npy")).astype(np.float64)

        self.ctx_count = self.c2.sum(axis=1, dtype=np.int64)          # c(v)
        self.ctx_n1plus = (self.c2 > 0).sum(axis=1).astype(np.int64)  # N1+(v .)
        if use_kn_unigram:
            cont = (self.c2 > 0).sum(axis=0).astype(np.float64)       # N1+(. w)
            base = cont / max(1.0, cont.sum())
        else:
            base = self.unigram / max(1.0, self.unigram.sum())
        self.p_base = (1.0 - EPS_UNIFORM) * base + EPS_UNIFORM / V
        # entropy of the smoothed bigram row per context, a context-only gate feature
        self.ctx_entropy = self._row_entropies()
        self.build_seconds = time.monotonic() - t0

        self.tri_keys = None
        kp = os.path.join(d, "trigram_keys.npy")
        if os.path.exists(kp):
            self.tri_keys = np.load(kp, mmap_mode="r")
            self.tri_cnts = np.load(os.path.join(d, "trigram_cnts.npy"), mmap_mode="r")
            ctx = np.asarray(self.tri_keys) // V
            first = np.concatenate(([0], np.flatnonzero(ctx[1:] != ctx[:-1]) + 1))
            self.tri_ctx_ids = ctx[first]
            cnts = np.asarray(self.tri_cnts, dtype=np.int64)
            self.tri_ctx_count = np.add.reduceat(cnts, first)
            self.tri_ctx_n1plus = np.diff(np.concatenate((first, [ctx.size]))).astype(np.int64)
            del ctx, first, cnts

    def _row_entropies(self) -> np.ndarray:
        ent = np.zeros(V, dtype=np.float64)
        for v in range(V):
            p = self.p_bigram_row(v)
            ent[v] = -(p * np.log(np.maximum(p, 1e-30))).sum()
        return ent

    def p_bigram_row(self, v: int) -> np.ndarray:
        cv = self.ctx_count[v]
        if cv == 0:
            return self.p_base
        row = self.c2[v].astype(np.float64)
        lam = self.discount * self.ctx_n1plus[v] / cv
        return np.maximum(row - self.discount, 0.0) / cv + lam * self.p_base

    def p_bigram(self, prev: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Vectorised p_KN(tgt | prev) for arrays of positions."""
        cv = self.ctx_count[prev].astype(np.float64)
        c = self.c2[prev, tgt].astype(np.float64)
        seen = cv > 0
        cv_safe = np.where(seen, cv, 1.0)
        lam = self.discount * self.ctx_n1plus[prev] / cv_safe
        p = np.maximum(c - self.discount, 0.0) / cv_safe + lam * self.p_base[tgt]
        return np.where(seen, p, self.p_base[tgt])

    # -- trigram layer -------------------------------------------------------
    def tri_lookup(self, prev2: np.ndarray, prev: np.ndarray, tgt: np.ndarray):
        """Return (c3, c3_ctx, n1plus3_ctx) for each query, zeros where unseen."""
        ctx_id = prev2.astype(np.int64) * V + prev.astype(np.int64)
        key = ctx_id * V + tgt.astype(np.int64)
        keys = np.asarray(self.tri_keys)
        pos = np.searchsorted(keys, key)
        hit = (pos < keys.size) & (keys[np.minimum(pos, keys.size - 1)] == key)
        c3 = np.where(hit, np.asarray(self.tri_cnts)[np.minimum(pos, keys.size - 1)], 0).astype(np.float64)

        cpos = np.searchsorted(self.tri_ctx_ids, ctx_id)
        chit = (cpos < self.tri_ctx_ids.size) & (
            self.tri_ctx_ids[np.minimum(cpos, self.tri_ctx_ids.size - 1)] == ctx_id)
        safe = np.minimum(cpos, self.tri_ctx_ids.size - 1)
        c3ctx = np.where(chit, self.tri_ctx_count[safe], 0).astype(np.float64)
        n1p3 = np.where(chit, self.tri_ctx_n1plus[safe], 0).astype(np.float64)
        return c3, c3ctx, n1p3

    def p_trigram(self, prev2, prev, tgt, p_bi, d3: float):
        c3, c3ctx, n1p3 = self.tri_lookup(prev2, prev, tgt)
        seen = c3ctx > 0
        safe = np.where(seen, c3ctx, 1.0)
        lam = d3 * n1p3 / safe
        p = np.maximum(c3 - d3, 0.0) / safe + lam * p_bi
        return np.where(seen, p, p_bi), c3ctx


# -----------------------------------------------------------------------------
# bpb


def bpb_from_logp(logp: np.ndarray, nbytes: np.ndarray, valid: np.ndarray) -> float:
    """Reproduces prepare.evaluate_bpb's accounting exactly: nats are masked by
    (valid & nbytes>0), the byte denominator by valid alone."""
    mask = valid & (nbytes > 0)
    nats = float(-(logp[mask]).sum())
    denom = float(nbytes[valid].sum())
    return nats / (LN2 * denom)


def mix_logp(logp_n: np.ndarray, p_c: np.ndarray, g: np.ndarray) -> np.ndarray:
    p_n = np.exp(logp_n.astype(np.float64))
    return np.log(np.maximum((1.0 - g) * p_n + g * p_c, 1e-300))


# -----------------------------------------------------------------------------


def load_stream(d: str, stream: str, seq_len: int, suffix: str = "") -> dict:
    z = np.load(os.path.join(d, f"dump_{stream}{suffix}_seq{seq_len}.npz"))
    x = z["x"].astype(np.int64)
    y = z["y"].astype(np.int64)
    valid = y >= 0
    ysafe = np.where(valid, y, 0)
    n = y.size
    pos_in_row = np.arange(n) % seq_len
    prev = x
    prev2 = np.empty_like(x)
    prev2[1:] = x[:-1]
    prev2[0] = 0
    has_prev2 = pos_in_row >= 1
    return {
        "x": x, "y": ysafe, "valid": valid, "prev": prev, "prev2": prev2,
        "has_prev2": has_prev2, "logp": z["logp_target"].astype(np.float64),
        "entropy": z["entropy"].astype(np.float64), "topk_mass": z["topk_mass"].astype(np.float64),
        "max_logp": z["topk_logp"][:, 0].astype(np.float64),
    }


def features(s: dict, cm: CountModel, d3: float) -> dict:
    prev, y, prev2 = s["prev"], s["y"], s["prev2"]
    p_bi = cm.p_bigram(prev, y)
    ctx_count = cm.ctx_count[prev].astype(np.float64)
    order = np.where(ctx_count > 0, 2, 1)
    p_c = p_bi
    tri_ctx_count = np.zeros_like(ctx_count)
    if cm.tri_keys is not None:
        p_tri, c3ctx = cm.p_trigram(prev2, prev, y, p_bi, d3)
        use = s["has_prev2"]
        p_c = np.where(use, p_tri, p_bi)
        tri_ctx_count = np.where(use, c3ctx, 0.0)
        order = np.where(use & (c3ctx > 0), 3, order)
    return {
        "p_bi": p_bi, "p_c": p_c, "order": order,
        "ctx_count": ctx_count, "tri_ctx_count": tri_ctx_count,
        "ctx_entropy": cm.ctx_entropy[prev],
    }


def _best_lambda(pn: np.ndarray, pc: np.ndarray) -> tuple[float, float]:
    """Coarse-then-fine 1-D search for the lambda minimising summed nats. The objective
    is convex in lambda (log of an affine function), so a local refine is exact."""
    def nats(lam):
        return float(-np.log(np.maximum((1.0 - lam) * pn + lam * pc, 1e-300)).sum())
    coarse = np.linspace(0.0, 0.95, 20)
    vals = [nats(l) for l in coarse]
    i = int(np.argmin(vals))
    lo = coarse[max(0, i - 1)]
    hi = coarse[min(len(coarse) - 1, i + 1)]
    fine = np.linspace(lo, hi, 21)
    fvals = [nats(l) for l in fine]
    j = int(np.argmin(fvals))
    return float(fine[j]), fvals[j]


def fit_global_lambda(logp_n, p_c, nbytes, valid):
    mask = valid & (nbytes > 0)
    lam, nats = _best_lambda(np.exp(logp_n[mask]), p_c[mask])
    denom = float(nbytes[valid].sum())
    return lam, nats / (LN2 * denom)


def fit_per_group_lambda(logp_n, p_c, nbytes, valid, groups, n_groups):
    """Independent lambda per group; the objective is separable across groups because
    the nats sum is, and the byte denominator is a constant."""
    lams = np.zeros(n_groups)
    mask = valid & (nbytes > 0)
    p_n = np.exp(logp_n)
    for gi in range(n_groups):
        sel = mask & (groups == gi)
        if not sel.any():
            continue
        lams[gi] = _best_lambda(p_n[sel], p_c[sel])[0]
    return lams


def apply_per_group(logp_n, p_c, groups, lams):
    return mix_logp(logp_n, p_c, lams[groups])


def fit_logistic_gate(feats: np.ndarray, logp_n, p_c, nbytes, valid, steps=800, lr=0.05):
    mask = valid & (nbytes > 0)
    X = torch.tensor(feats[mask], dtype=torch.float64)
    pn = torch.tensor(np.exp(logp_n[mask]), dtype=torch.float64)
    pc = torch.tensor(p_c[mask], dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        g = torch.sigmoid(X @ w + b)
        loss = -torch.log(((1 - g) * pn + g * pc).clamp_min(1e-300)).mean()
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.detach()), float(loss.detach())


def gate_from_logistic(feats: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(feats @ w + b)))


def main() -> None:
    args = parse_args()
    d = os.path.expanduser(args.dir)
    token_bytes = P.get_token_bytes("cpu").numpy().astype(np.int64)

    streams = {}
    for name in ("train4", "public_val"):
        s = load_stream(d, name, args.seq_len, args.suffix)
        s["nbytes"] = token_bytes[s["y"]]
        s["neural_bpb"] = bpb_from_logp(s["logp"], s["nbytes"], s["valid"])
        streams[name] = s
        print(f"{name}: {s['valid'].sum():,} valid tokens, neural bpb {s['neural_bpb']:.6f}, "
              f"mean top-64 mass {s['topk_mass'][s['valid']].mean():.6f}", flush=True)

    results = {"neural_bpb": {k: v["neural_bpb"] for k, v in streams.items()},
               "top64_mass": {k: float(streams[k]['topk_mass'][streams[k]['valid']].mean())
                              for k in streams},
               "arms": []}

    for discount in args.discount:
        for kn in (True, False):
            tag = f"D={discount}{'/KN' if kn else '/MLE'}"
            cm = CountModel(d, discount, use_kn_unigram=kn)
            F = {k: features(streams[k], cm, args.discount3) for k in streams}
            for label, key in (("bigram", "p_bi"), ("trigram", "p_c")):
                if key == "p_c" and cm.tri_keys is None:
                    continue
                pc = {k: F[k][key] for k in streams}
                arm = {"tag": tag, "order": label, "discount": discount, "kn_unigram": kn}
                # count model alone
                for k in streams:
                    arm[f"count_only_bpb_{k}"] = bpb_from_logp(
                        np.log(np.maximum(pc[k], 1e-300)), streams[k]["nbytes"], streams[k]["valid"])
                # 1. global lambda, fitted on train4
                lam, fit_bpb = fit_global_lambda(streams["train4"]["logp"], pc["train4"],
                                                 streams["train4"]["nbytes"], streams["train4"]["valid"])
                arm["global_lambda"] = lam
                for k in streams:
                    arm[f"global_bpb_{k}"] = bpb_from_logp(
                        mix_logp(streams[k]["logp"], pc[k], np.float64(lam)),
                        streams[k]["nbytes"], streams[k]["valid"])
                # oracle: lambda fitted on the test stream itself (upper bound only)
                lam_a, _ = fit_global_lambda(streams["public_val"]["logp"], pc["public_val"],
                                             streams["public_val"]["nbytes"],
                                             streams["public_val"]["valid"])
                arm["oracle_lambda_public_val"] = lam_a
                arm["oracle_bpb_public_val"] = bpb_from_logp(
                    mix_logp(streams["public_val"]["logp"], pc["public_val"], np.float64(lam_a)),
                    streams["public_val"]["nbytes"], streams["public_val"]["valid"])
                # 2. per-order lambda
                grp = {k: F[k]["order"] - 1 for k in streams}
                lams = fit_per_group_lambda(streams["train4"]["logp"], pc["train4"],
                                            streams["train4"]["nbytes"], streams["train4"]["valid"],
                                            grp["train4"], 3)
                arm["per_order_lambda"] = lams.tolist()
                for k in streams:
                    arm[f"per_order_bpb_{k}"] = bpb_from_logp(
                        apply_per_group(streams[k]["logp"], pc[k], grp[k], lams),
                        streams[k]["nbytes"], streams[k]["valid"])
                results["arms"].append(arm)
                print(json.dumps(arm, sort_keys=True), flush=True)

    # ---- richer gates, only on the single best (discount, order) arm ----------
    best = min(results["arms"], key=lambda a: a["global_bpb_public_val"])
    cm = CountModel(d, best["discount"], use_kn_unigram=best["kn_unigram"])
    key = "p_bi" if best["order"] == "bigram" else "p_c"
    F = {k: features(streams[k], cm, args.discount3) for k in streams}
    pc = {k: F[k][key] for k in streams}

    def featmat(k):
        s, f = streams[k], F[k]
        return np.stack([
            s["entropy"] / 5.0,
            np.log1p(f["ctx_count"]) / 15.0,
            np.log1p(f["tri_ctx_count"]) / 10.0,
            (f["order"] == 3).astype(np.float64),
            (f["order"] == 1).astype(np.float64),
            f["ctx_entropy"] / 5.0,
            np.exp(s["max_logp"]),
        ], axis=1)

    t0 = time.monotonic()
    w, b, tr_loss = fit_logistic_gate(featmat("train4"), streams["train4"]["logp"], pc["train4"],
                                      streams["train4"]["nbytes"], streams["train4"]["valid"])
    logi = {"weights": w.tolist(), "bias": b, "train_nll": tr_loss,
            "seconds": time.monotonic() - t0,
            "feature_names": ["entropy/5", "log1p(bi_ctx_count)/15", "log1p(tri_ctx_count)/10",
                              "order==3", "order==1", "ctx_entropy/5", "max_p_neural"]}
    for k in streams:
        g = gate_from_logistic(featmat(k), w, b)
        logi[f"bpb_{k}"] = bpb_from_logp(mix_logp(streams[k]["logp"], pc[k], g),
                                         streams[k]["nbytes"], streams[k]["valid"])
        logi[f"mean_gate_{k}"] = float(g[streams[k]["valid"]].mean())
    results["logistic_gate"] = logi
    print("logistic:", json.dumps(logi, sort_keys=True), flush=True)

    # 4. non-parametric (order x entropy decile) lambda table
    edges = np.quantile(streams["train4"]["entropy"][streams["train4"]["valid"]],
                        np.linspace(0, 1, 11)[1:-1])
    grp = {}
    for k in streams:
        dec = np.searchsorted(edges, streams[k]["entropy"])
        grp[k] = (F[k]["order"] - 1) * 10 + dec
    lams = fit_per_group_lambda(streams["train4"]["logp"], pc["train4"],
                                streams["train4"]["nbytes"], streams["train4"]["valid"],
                                grp["train4"], 30)
    bucket = {"lambdas": lams.tolist(), "entropy_edges": edges.tolist()}
    for k in streams:
        bucket[f"bpb_{k}"] = bpb_from_logp(apply_per_group(streams[k]["logp"], pc[k], grp[k], lams),
                                           streams[k]["nbytes"], streams[k]["valid"])
    results["bucket_gate"] = bucket
    print("bucket:", json.dumps(bucket, sort_keys=True), flush=True)

    # ---- ORACLE ceilings: the same two gate families fitted on stream (a) ITSELF.
    # Not a valid result (it reads the test stream) -- reported only to bound what any
    # honestly-fitted gate of these features could ever reach, so that a small honest
    # gain cannot be dismissed as a fitting failure.
    ow, ob, _ = fit_logistic_gate(featmat("public_val"), streams["public_val"]["logp"],
                                  pc["public_val"], streams["public_val"]["nbytes"],
                                  streams["public_val"]["valid"])
    og = gate_from_logistic(featmat("public_val"), ow, ob)
    olams = fit_per_group_lambda(streams["public_val"]["logp"], pc["public_val"],
                                 streams["public_val"]["nbytes"], streams["public_val"]["valid"],
                                 grp["public_val"], 30)
    results["oracle_gates_public_val"] = {
        "logistic_bpb": bpb_from_logp(mix_logp(streams["public_val"]["logp"], pc["public_val"], og),
                                      streams["public_val"]["nbytes"], streams["public_val"]["valid"]),
        "logistic_mean_gate": float(og[streams["public_val"]["valid"]].mean()),
        "bucket_bpb": bpb_from_logp(
            apply_per_group(streams["public_val"]["logp"], pc["public_val"], grp["public_val"], olams),
            streams["public_val"]["nbytes"], streams["public_val"]["valid"]),
    }
    print("oracle gates:", json.dumps(results["oracle_gates_public_val"], sort_keys=True), flush=True)

    results["best_arm"] = best
    out = args.out or os.path.join(d, f"fit_results{args.suffix}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nwrote {out}")

    nb = streams["public_val"]["neural_bpb"]
    print("\n=== SUMMARY (stream (a) = public_val, gates fitted on (b) = train shard 4) ===")
    print(f"neural alone         (a) {nb:.6f}   (b) {streams['train4']['neural_bpb']:.6f}")
    for name, val in (("global lambda", best.get("global_bpb_public_val")),
                      ("per-order lambda", best.get("per_order_bpb_public_val")),
                      ("logistic gate", logi.get("bpb_public_val")),
                      ("bucket gate", bucket.get("bpb_public_val")),
                      ("[oracle] global lam", best.get("oracle_bpb_public_val")),
                      ("[oracle] logistic", results["oracle_gates_public_val"]["logistic_bpb"]),
                      ("[oracle] bucket", results["oracle_gates_public_val"]["bucket_bpb"])):
        if val is not None:
            print(f"{name:<20} (a) {val:.6f}   gain {nb - val:+.6f}")


if __name__ == "__main__":
    main()
