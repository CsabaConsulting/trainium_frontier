"""T4.1 step 1: dump per-token neural log-probs from a checkpoint, offline on CPU.

Part of the §6 offline protocol in research/ngram-drafter-and-count-mixer-2026-09-12.md.
Runs `train.load_for_eval` on the CPU (eager, fp32) and walks
`prepare.make_eval_dataloader` exactly as `prepare.evaluate_bpb` does, so the token
stream, the packing and the -1 padding convention are identical to the scorer's. For
every position it records

  * the input/target ids,
  * log p_neural(target)  -- EXACT, not a top-k approximation,
  * the exact entropy of the full 8192-way softmax,
  * the top-K (id, logp) pairs and the mass they cover (for the doc's §6 caveat check).

Because both the neural and the count distribution are proper distributions over the
8192 real tokens, the mixture's log-prob at the *target* only needs the two target
probabilities, so the mixture bpb computed downstream from this dump is exact.

Pure CPU, does not touch the Neuron device -- safe to run alongside a device job.

Run (on the box, inside the venv):
  python research/scratch/count_mixer_dump.py \
      --checkpoint ~/ckpt-keep/final-t31-l5tri-m64-2026-09-12.pt \
      --stream public_val --max-tokens 2000000 --out ~/countmix
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import prepare as P

STREAMS = {
    # name -> (split, shard_ids)
    "public_val": ("public_val", None),
    "train4": ("train", [4]),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-py", default="train.py")
    ap.add_argument("--stream", required=True, choices=sorted(STREAMS))
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-dist-tokens", type=int, default=0,
                    help="also dump the full [N, 8192] fp16 log-prob matrix for the first N "
                         "tokens (§6 approximation caveat); 0 disables")
    ap.add_argument("--apply-ckpt-toggles", action="store_true",
                    help="set the imported train module's arch toggles from the checkpoint's own "
                         "payload['toggles'] before load_for_eval, so a checkpoint trained under "
                         "different defaults (e.g. the pre-T3.1 control) still strict-loads. Only "
                         "mutates the throwaway imported module, never train.py on disk.")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    return ap.parse_args()


def apply_ckpt_toggles(train_mod, checkpoint: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    toggles = payload.get("toggles", {})
    changed = {}
    for name, value in toggles.items():
        if hasattr(train_mod, name) and getattr(train_mod, name) != value:
            changed[name] = [getattr(train_mod, name), value]
            setattr(train_mod, name, value)
    if changed:
        print(f"applied checkpoint toggles: {json.dumps(changed, sort_keys=True)}")
    return changed


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    train_mod = P._import_train_module(args.train_py)
    tokenizer = P.ensure_tokenizer(build_if_missing=False)
    device = torch.device("cpu")
    ckpt = os.path.expanduser(args.checkpoint)
    toggles_changed = apply_ckpt_toggles(train_mod, ckpt) if args.apply_ckpt_toggles else {}
    model = train_mod.load_for_eval(ckpt, device)

    split, shard_ids = STREAMS[args.stream]
    loader = P.make_eval_dataloader(tokenizer, args.batch_size, args.seq_len, split, device,
                                    shard_ids=shard_ids)

    xs, ys, lps, ents, tk_ids, tk_lps, tk_mass = [], [], [], [], [], [], []
    full_dist = []
    full_dist_left = args.full_dist_tokens
    seen = 0
    started = time.monotonic()
    for x, y, _state in loader:
        logits = model(x)[..., :P.TOKENIZER_VOCAB_SIZE].float()
        logp = F.log_softmax(logits, dim=-1)                       # [b, t, V]
        safe_y = y.clamp_min(0)
        lp_t = logp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)    # [b, t]
        p = logp.exp()
        ent = -(p * logp).sum(-1)                                  # [b, t] nats
        tv, ti = logp.topk(args.topk, dim=-1)
        mass = tv.exp().sum(-1)

        xs.append(x.to(torch.int32).numpy())
        ys.append(y.to(torch.int32).numpy())
        lps.append(lp_t.numpy().astype(np.float32))
        ents.append(ent.numpy().astype(np.float32))
        tk_ids.append(ti.to(torch.int16).numpy())
        tk_lps.append(tv.to(torch.float16).numpy())
        tk_mass.append(mass.numpy().astype(np.float32))
        if full_dist_left > 0:
            flat = logp.reshape(-1, P.TOKENIZER_VOCAB_SIZE)
            take = min(full_dist_left, flat.shape[0])
            full_dist.append(flat[:take].to(torch.float16).numpy())
            full_dist_left -= take

        seen += int((y >= 0).sum().item())
        if seen % (args.batch_size * args.seq_len * 8) < args.batch_size * args.seq_len:
            rate = seen / max(1e-9, time.monotonic() - started)
            print(f"  {seen:,} tokens, {rate:,.0f} tok/s, "
                  f"eta {(args.max_tokens - seen) / max(1.0, rate):.0f}s", flush=True)
        if seen >= args.max_tokens:
            break

    def cat(parts, dtype):
        return np.concatenate([p.reshape(-1) for p in parts]).astype(dtype, copy=False)

    payload = {
        "x": cat(xs, np.int32),
        "y": cat(ys, np.int32),
        "logp_target": cat(lps, np.float32),
        "entropy": cat(ents, np.float32),
        "topk_mass": cat(tk_mass, np.float32),
        "topk_ids": np.concatenate([t.reshape(-1, args.topk) for t in tk_ids]),
        "topk_logp": np.concatenate([t.reshape(-1, args.topk) for t in tk_lps]),
        "row_len": np.array([args.seq_len], dtype=np.int32),
    }
    tag = f"{args.stream}{args.suffix}_seq{args.seq_len}"
    path = os.path.join(out_dir, f"dump_{tag}.npz")
    np.savez(path, **payload)
    if full_dist:
        np.save(os.path.join(out_dir, f"fulldist_{tag}.npy"),
                np.concatenate(full_dist, axis=0))

    meta = {
        "checkpoint": args.checkpoint, "stream": args.stream, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "topk": args.topk, "tokens_valid": seen,
        "positions": int(payload["y"].size), "seconds": time.monotonic() - started,
        "full_dist_tokens": args.full_dist_tokens - full_dist_left,
        "toggles_changed": toggles_changed,
    }
    with open(os.path.join(out_dir, f"dump_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
