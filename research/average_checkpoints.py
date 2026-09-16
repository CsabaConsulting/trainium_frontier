"""Average N training checkpoints into ONE static checkpoint for load_for_eval.

Sub-1.0 plan P4 (2026-09-16): last-k / trajectory averaging as a free variance
reduction. The output payload has the same schema save_checkpoint() writes
(model_state, model_config, toggles, meta, plus ema_state when present in ALL
inputs), so load_for_eval() consumes it unchanged -- and the averaged artifact
is still a single stateless model, i.e. rules-safe (pure/stateless/causal).

Inputs may be raw checkpoints (averages model_state) or, when every input
carries ema_state, their EMA weights (averages ema_state). Mixed inputs are
refused: averaging raw weights with EMA weights would be a silent category
error. All inputs must share identical model_config and parameter keys/shapes.

CPU-only. Run:
  python3 research/average_checkpoints.py --out avg.pt a.pt b.pt c.pt [--ema|--raw]
  python3 research/average_checkpoints.py --selftest   # synthetic payloads, no data needed
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=True)


def average_checkpoints(paths: list[Path], use_ema: str = "auto") -> dict:
    """Average checkpoint payloads. use_ema: 'auto' (ema iff ALL have it), 'ema', 'raw'."""
    if not paths:
        raise ValueError("no input checkpoints")
    payloads = [_load(p) for p in paths]
    base_config = payloads[0]["model_config"]
    for p, payload in zip(paths[1:], payloads[1:], strict=True):
        if payload["model_config"] != base_config:
            raise ValueError(f"model_config mismatch: {paths[0]} vs {p}")
    has_ema = ["ema_state" in p for p in payloads]
    if use_ema == "auto":
        key = "ema_state" if all(has_ema) else "model_state"
        if any(has_ema) and not all(has_ema):
            print("WARNING: some inputs lack ema_state; averaging model_state for all",
                  file=sys.stderr)
    elif use_ema == "ema":
        if not all(has_ema):
            missing = [str(p) for p, h in zip(paths, has_ema) if not h]
            raise ValueError(f"--ema requested but missing ema_state in: {missing}")
        key = "ema_state"
    elif use_ema == "raw":
        key = "model_state"
    else:
        raise ValueError(f"use_ema must be auto/ema/raw, got {use_ema!r}")
    ref_keys = set(payloads[0][key].keys())
    for p, payload in zip(paths[1:], payloads[1:], strict=True):
        if set(payload[key].keys()) != ref_keys:
            raise ValueError(f"state key mismatch: {paths[0]} vs {p}")
        for k in ref_keys:
            if payload[key][k].shape != payloads[0][key][k].shape:
                raise ValueError(f"shape mismatch on {k}: {paths[0]} vs {p}")
    n = len(payloads)
    avg = {k: sum(p[key][k].to(torch.float32) for p in payloads) / n for k in ref_keys}
    out = {
        "model_state": avg if key == "model_state" else payloads[0]["model_state"],
        "model_config": base_config,
        "toggles": payloads[0].get("toggles", {}),
        "meta": {**payloads[0].get("meta", {}),
                 "averaged_from": [str(p) for p in paths],
                 "averaged_key": key},
    }
    if key == "ema_state":
        out["ema_state"] = avg
    return out


def selftest() -> None:
    import tempfile
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(3):
            payload = {
                "model_state": {"w": torch.full((4,), float(i)),
                                "b": torch.randn(4)},
                "ema_state": {"w": torch.full((4,), 10.0 * i),
                              "b": torch.randn(4)},
                "model_config": {"n_layer": 6, "n_head": 5},
                "toggles": {"N_KV_HEADS": 0},
                "meta": {"step": i},
            }
            p = Path(d) / f"c{i}.pt"
            torch.save(payload, p)
            paths.append(p)
        out = average_checkpoints(paths)  # auto -> all have ema -> ema
        assert out["meta"]["averaged_key"] == "ema_state", out["meta"]
        assert torch.equal(out["ema_state"]["w"], torch.full((4,), 10.0)), out["ema_state"]["w"]
        assert torch.equal(out["model_state"]["w"], torch.zeros(4))  # first raw kept
        out = average_checkpoints(paths, use_ema="raw")
        assert torch.equal(out["model_state"]["w"], torch.ones(4)), out["model_state"]["w"]
        assert "ema_state" not in out
        # config mismatch refused
        bad = dict(torch.load(paths[0], map_location="cpu", weights_only=True))
        bad["model_config"] = {"n_layer": 8, "n_head": 5}
        bp = Path(d) / "bad.pt"
        torch.save(bad, bp)
        try:
            average_checkpoints([paths[0], bp])
            raise SystemExit("FAIL: config mismatch not refused")
        except ValueError:
            pass
        # mixed ema/raw under --ema refused
        noema = dict(torch.load(paths[0], map_location="cpu", weights_only=True))
        del noema["ema_state"]
        np_ = Path(d) / "noema.pt"
        torch.save(noema, np_)
        try:
            average_checkpoints([paths[0], np_], use_ema="ema")
            raise SystemExit("FAIL: mixed ema not refused")
        except ValueError:
            pass
    print("AVERAGE SELFTEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=False, help="output checkpoint path")
    ap.add_argument("inputs", nargs="*", help="input checkpoint paths")
    ap.add_argument("--ema", dest="use_ema", action="store_const", const="ema",
                    help="average ema_state (refuse if any input lacks it)")
    ap.add_argument("--raw", dest="use_ema", action="store_const", const="raw",
                    help="average model_state even when ema_state exists")
    ap.set_defaults(use_ema="auto")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.out or not args.inputs:
        ap.error("--out and at least one input required (or --selftest)")
    out = average_checkpoints([Path(p) for p in args.inputs], use_ema=args.use_ema)
    torch.save(out, args.out)
    print(f"wrote {args.out}: averaged {args.out and len(args.inputs)} inputs "
          f"via {out['meta']['averaged_key']}")


if __name__ == "__main__":
    main()
