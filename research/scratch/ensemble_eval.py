"""Model-ensembling spot-check: standalone eval for the 2026-09-16 side-measurement.

Context (see the dated section in research/trn2-experiment-results.md this script's runs feed):
this repo's whole week of results says spending compute on more steps/capacity at the locked
config beats almost every alternative use of the same compute, and that halving throughput (a
2-way world_size split) needs an enormous quality win to break even
(forgone-steps penalty ~= 0.077 * ln(step_ratio)). A deep-research pass flagged "model
ensembling" (train N smaller/shorter models concurrently, average their PREDICTIONS) as a
technique used in the sister Slowrun community, tied to this repo's own measured seed-init
variance (~0.003 std across seeds). This script is the "ONE cheap check" the user asked for:
does averaging the LOGITS of two independently-seeded HALF-compute replicas beat one
FULL-compute replica, at EQUAL TOTAL COMPUTE (box-core-seconds)?

It does NOT modify train.py's training path. It only loads checkpoints (already produced by
three separate `torchrun ... train.py` launches, see the driver commands in the results-doc
section) and re-runs eval, reusing train.py's own `load_for_eval` and prepare.py's own
`evaluate_bpb` / `make_eval_dataloader` / `get_token_bytes` machinery -- the exact same trusted
CE-from-logits path the real scorer and every public_val_bpb number in this repo already uses.
The only NEW code here is `evaluate_ensemble_bpb`, a line-for-line mirror of
`prepare.evaluate_bpb` except it averages two models' LOGITS (not their losses) before the
cross-entropy, which is the actual mechanism under test.

All four numbers are read off the SAME deterministic, non-shuffled `public_val` eval stream
(make_eval_dataloader has no RNG and rank/world_size default to 1/0 outside torchrun), so
control-alone, replica-A-alone, replica-B-alone and ensemble(A,B) are computed on byte-identical
batches, up to the shared --eval-tokens cutoff -- a fair, low-noise comparison.

Run (single process, no torchrun -- eval-only, one NeuronCore is plenty):
    NEURON_RT_VISIBLE_CORES=0 python research/scratch/ensemble_eval.py \
        --control out/ensemble-control/final.pt \
        --replica-a out/ensemble-repA/final.pt \
        --replica-b out/ensemble-repB/final.pt \
        --json-out research/run-logs/ensemble-eval-result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import train as T  # noqa: E402
from prepare import EvalTimeoutError, get_token_bytes, make_eval_dataloader  # noqa: E402


@torch.no_grad()
def evaluate_ensemble_bpb(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    split: str = "public_val",
    max_tokens: int | None = None,
    timeout_seconds: int = T.EVAL_TIMEOUT_SECONDS,
) -> dict:
    """Line-for-line mirror of prepare.evaluate_bpb, except the trusted CE is computed on the
    AVERAGE of the two models' logits (post-softcap, pre-cross-entropy) rather than on either
    model's logits alone. This is the actual "true ensemble" mechanism the whole check is about:
    averaging PREDICTIONS (logits), never losses.
    """
    started = time.monotonic()
    dev = torch.device(device)
    token_bytes = get_token_bytes(dev)
    loader = make_eval_dataloader(tokenizer, batch_size, seq_len, split, dev)
    # Single process, no torchrun: get_dist_info() (called inside make_eval_dataloader) reports
    # world_size=1, so the local token limit is just max_tokens itself -- same simplification
    # prepare.evaluate_bpb would produce at world_size=1.
    local_token_limit = None if max_tokens is None else max(1, max_tokens)

    model_a.eval()
    model_b.eval()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=dev)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=dev)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=dev)
    local_tokens = 0

    for x, y, _state in loader:
        if time.monotonic() - started > timeout_seconds:
            raise EvalTimeoutError(f"ensemble evaluation exceeded {timeout_seconds}s")
        logits_a = model_a(x)
        logits_b = model_b(x)
        vocab = T.TOKENIZER_VOCAB_SIZE
        if logits_a.size(-1) < vocab or logits_b.size(-1) < vocab:
            raise RuntimeError(f"model vocab too small: {logits_a.size(-1)}/{logits_b.size(-1)} < {vocab}")
        logits_a = logits_a[..., :vocab].float()
        logits_b = logits_b[..., :vocab].float()
        # THE mechanism under test: average LOGITS, then cross-entropy -- not average of losses.
        logits = (logits_a + logits_b) * 0.5
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1, reduction="none"
        )
        y_flat = y.reshape(-1)
        valid = y_flat >= 0
        safe_y = y_flat.clamp_min(0)
        nbytes = token_bytes[safe_y]
        mask = valid & (nbytes > 0)
        total_nats += (loss * mask).sum()
        total_bytes += (nbytes * valid).sum()
        batch_tokens = valid.sum()
        total_tokens += batch_tokens
        local_tokens += int(batch_tokens.item())
        if local_token_limit is not None and local_tokens >= local_token_limit:
            break

    elapsed = time.monotonic() - started
    nats = total_nats.item()
    nbytes_total = int(total_bytes.item())
    val_bpb = nats / (math.log(2) * nbytes_total) if nbytes_total > 0 else float("inf")
    return {
        "val_bpb": float(val_bpb),
        "eval_seconds": float(elapsed),
        "num_tokens": int(total_tokens.item()),
        "num_bytes": nbytes_total,
        "total_nats": float(nats),
    }


def load_and_check(label: str, path: str, device: torch.device, tokenizer, batch_size: int, seq_len: int):
    model = T.load_for_eval(path, device)
    causal = T.causality_check(model, tokenizer, batch_size, seq_len, device)
    print(f"[{label}] loaded {path}: causality_passed={causal['passed']} "
          f"max_abs_diff={causal['max_abs_diff']:.6f}", flush=True)
    return model, causal


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", required=True, help="path to the world_size=4 full-compute checkpoint")
    p.add_argument("--replica-a", required=True, help="path to treatment replica A's checkpoint (world_size=2)")
    p.add_argument("--replica-b", required=True, help="path to treatment replica B's checkpoint (world_size=2)")
    p.add_argument("--eval-tokens", type=int, default=4 * 524288,
                   help="matches train.py's own --eval-tokens default, so these bpb numbers are "
                        "directly comparable to every other public_val_bpb in trn2-experiment-results.md")
    p.add_argument("--seq-len", type=int, default=0,
                   help="0 (default) = use train.py's own SEQ_LEN (2048, the trained context length)")
    p.add_argument("--split", type=str, default="public_val")
    p.add_argument("--json-out", type=str, default="", help="optional path to dump the 4-number result as JSON")
    args = p.parse_args()

    device_type = T.detect_device_type()
    device = torch.device(device_type)
    seq_len = args.seq_len or T.SEQ_LEN
    batch_size = T.DEVICE_BATCH_SIZE
    tokenizer = T.ensure_tokenizer(build_if_missing=False)

    print(f"device={device} seq_len={seq_len} batch_size={batch_size} eval_tokens={args.eval_tokens} "
          f"split={args.split}", flush=True)

    t0 = time.monotonic()
    control_model, control_causal = load_and_check("control", args.control, device, tokenizer, batch_size, seq_len)
    rep_a_model, rep_a_causal = load_and_check("replica-A", args.replica_a, device, tokenizer, batch_size, seq_len)
    rep_b_model, rep_b_causal = load_and_check("replica-B", args.replica_b, device, tokenizer, batch_size, seq_len)

    print("--- (a) control alone ---", flush=True)
    control_eval = T.evaluate_bpb(control_model, tokenizer, batch_size, seq_len, device,
                                   split=args.split, max_tokens=args.eval_tokens,
                                   timeout_seconds=T.EVAL_TIMEOUT_SECONDS)
    print(f"control_val_bpb    = {control_eval['val_bpb']:.6f}  "
          f"(tokens={control_eval['num_tokens']}, {control_eval['eval_seconds']:.1f}s)", flush=True)

    print("--- (b) replica A alone ---", flush=True)
    rep_a_eval = T.evaluate_bpb(rep_a_model, tokenizer, batch_size, seq_len, device,
                                 split=args.split, max_tokens=args.eval_tokens,
                                 timeout_seconds=T.EVAL_TIMEOUT_SECONDS)
    print(f"replica_a_val_bpb  = {rep_a_eval['val_bpb']:.6f}  "
          f"(tokens={rep_a_eval['num_tokens']}, {rep_a_eval['eval_seconds']:.1f}s)", flush=True)

    print("--- (b) replica B alone ---", flush=True)
    rep_b_eval = T.evaluate_bpb(rep_b_model, tokenizer, batch_size, seq_len, device,
                                 split=args.split, max_tokens=args.eval_tokens,
                                 timeout_seconds=T.EVAL_TIMEOUT_SECONDS)
    print(f"replica_b_val_bpb  = {rep_b_eval['val_bpb']:.6f}  "
          f"(tokens={rep_b_eval['num_tokens']}, {rep_b_eval['eval_seconds']:.1f}s)", flush=True)

    print("--- (c) TRUE ensemble: average LOGITS of A and B, then cross-entropy ---", flush=True)
    ensemble_eval = evaluate_ensemble_bpb(rep_a_model, rep_b_model, tokenizer, batch_size, seq_len, device,
                                           split=args.split, max_tokens=args.eval_tokens,
                                           timeout_seconds=T.EVAL_TIMEOUT_SECONDS)
    print(f"ensemble_ab_val_bpb = {ensemble_eval['val_bpb']:.6f}  "
          f"(tokens={ensemble_eval['num_tokens']}, {ensemble_eval['eval_seconds']:.1f}s)", flush=True)

    elapsed = time.monotonic() - t0
    print("=== SUMMARY ===", flush=True)
    print(f"control_val_bpb:     {control_eval['val_bpb']:.6f}", flush=True)
    print(f"replica_a_val_bpb:   {rep_a_eval['val_bpb']:.6f}", flush=True)
    print(f"replica_b_val_bpb:   {rep_b_eval['val_bpb']:.6f}", flush=True)
    print(f"ensemble_ab_val_bpb: {ensemble_eval['val_bpb']:.6f}", flush=True)
    print(f"ensemble_minus_control: {ensemble_eval['val_bpb'] - control_eval['val_bpb']:+.6f}", flush=True)
    print(f"total_eval_wall_seconds: {elapsed:.1f}", flush=True)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "control": {"checkpoint": args.control, **control_eval, "causality": control_causal},
            "replica_a": {"checkpoint": args.replica_a, **rep_a_eval, "causality": rep_a_causal},
            "replica_b": {"checkpoint": args.replica_b, **rep_b_eval, "causality": rep_b_causal},
            "ensemble_ab": ensemble_eval,
            "eval_tokens": args.eval_tokens,
            "seq_len": seq_len,
            "split": args.split,
        }, indent=2))
        print(f"wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
