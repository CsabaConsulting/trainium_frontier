"""P0 eval reconciliation: score ONE checkpoint under every historical eval definition.

The 1.0-barrier confusion (agents reported 1.018985 vs 1.004355 vs 1.003045 for
what looked like the same model) comes from three different evaluation recipes:
in-process RAW weights @2048, in-process EMA weights @2048 (public_val_bpb_ema),
and the prepare.py eval-public sidecar (load_for_eval, EMA-preferring, its own
seq-len/batch). This script loads the checkpoint ONCE and evaluates the full
{raw, ema} x {2048, 1024} matrix plus any custom defs, printing one labeled
table with tokens/bytes/seconds per cell -- so every past and future number can
be relabeled to its definition.

Needs Trainium + the dataset (runs on the EC2 boxes, not locally).
Local check:  python3 research/reconcile_eval.py --dry-run
Remote run:   python3 research/reconcile_eval.py --checkpoint checkpoints/<run>/ckpt.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as T
from prepare import evaluate_bpb, ensure_tokenizer

BASE_CELLS = [
    # (name, weights, seq_len) -- mirrors the historical recipes
    ("inprocess-raw-2048", "raw", 2048),
    ("inprocess-ema-2048", "ema", 2048),
    ("inprocess-raw-1024", "raw", 1024),
    ("inprocess-ema-1024", "ema", 1024),
]


def plan_cells(custom: list[dict]) -> list[dict]:
    cells = [{"name": n, "weights": w, "seq_len": s} for n, w, s in BASE_CELLS]
    cells.extend(custom)
    return cells


def run(checkpoint: str, cells: list[dict], batch_size: int, device: torch.device,
        args_max_tokens: int | None = None, args_timeout: int = 1200) -> list[dict]:
    tokenizer = ensure_tokenizer(build_if_missing=False)
    payload = T.torch_load_weights(checkpoint, map_location="cpu")
    has_ema = "ema_state" in payload
    model = T.load_for_eval(checkpoint, device)  # EMA-preferring, like the scorer
    rows = []
    for cell in cells:
        w = cell.get("weights", "ema")
        if w == "raw" and has_ema:
            state = payload["model_state"]
            model.load_state_dict(state, strict=True)
            model.to(device)
            model.eval()
        elif w == "raw":
            pass  # no ema in checkpoint: load_for_eval already loaded raw
        # else: keep EMA weights from load_for_eval
        if w == "ema" and not has_ema:
            w = "raw(no-ema)"  # label honestly: nothing to prefer
        r = evaluate_bpb(model, tokenizer, batch_size, cell.get("seq_len", 2048),
                         device, split=cell.get("split", "public_val"),
                         max_tokens=cell.get("max_tokens", args_max_tokens),
                         shard_ids=cell.get("shard_ids"),
                         timeout_seconds=cell.get("timeout_seconds", args_timeout))
        rows.append({"def": cell.get("name", "?"), "weights": w,
                     "seq_len": cell.get("seq_len", 2048),
                     "val_bpb": r["val_bpb"], "num_tokens": r["num_tokens"],
                     "num_bytes": r["num_bytes"], "eval_seconds": r["eval_seconds"],
                     "over_timeout": r["eval_over_timeout"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", help="checkpoint path (remote: checkpoints/<run>/ckpt.pt)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device-type", default="neuron")
    ap.add_argument("--max-tokens", type=int, default=524288,
                    help="tokens per cell (default 524288 = exactly one rank's share of the "
                         "historical 4-rank in-process eval, whose --eval-tokens is 2M; all "
                         "cells see the same window, so cross-def deltas are apples-to-apples)")
    ap.add_argument("--timeout-seconds", type=int, default=1200)
    ap.add_argument("--defs", default="[]",
                    help="JSON list of extra cells, e.g. '[{\"name\":\"sidecar-4096\","
                         " \"weights\":\"ema\", \"seq_len\":4096}]'")
    ap.add_argument("--dry-run", action="store_true", help="print planned cells, run nothing")
    args = ap.parse_args()
    custom = json.loads(args.defs)
    cells = plan_cells(custom)
    if args.dry_run:
        for c in cells:
            print(c)
        return
    if not args.checkpoint:
        ap.error("--checkpoint required (or --dry-run)")
    device = torch.device(args.device_type)
    rows = run(args.checkpoint, cells, args.batch_size, device,
               args_max_tokens=args.max_tokens, args_timeout=args.timeout_seconds)
    print(f"{'def':22s} {'wts':5s} {'seq':5s} {'bpb':>10s} {'tokens':>10s} "
          f"{'bytes':>10s} {'sec':>7s} {'overT'}")
    for r in rows:
        print(f"{r['def']:22s} {r['weights']:5s} {r['seq_len']:<5d} {r['val_bpb']:10.6f} "
              f"{r['num_tokens']:10d} {r['num_bytes']:10d} {r['eval_seconds']:7.1f} "
              f"{r['over_timeout']}")


if __name__ == "__main__":
    main()
