"""T2.2 correctness harness for the weight EMA (CPU only).

The EMA itself is three lines of arithmetic; the risk is entirely in the plumbing around it --
whose weights end up in the checkpoint, and whose weights the SCORER gets when it calls
load_for_eval(path, device). So the checks are:

  1. OFF IS IDENTITY: with USE_WEIGHT_EMA=False the payload has no "ema_state" key at all, and
     load_for_eval returns exactly the raw weights -- i.e. every checkpoint written before T2.2
     and every run with the toggle off is untouched, not merely equivalent.
  2. LERP MATH: torch._foreach_lerp_(shadow, live, 1-decay) is BIT-IDENTICAL to the textbook
     `s = decay*s + (1-decay)*p` reference over several updates with changing live weights, and
     the shadow tracks a CONSTANT live tensor monotonically towards it (no sign error, no
     shadow/live argument swap -- the failure that would silently average nothing).
  3. STATE SUBSTITUTION: weight_ema_state() returns exactly the key set of model_state, with
     every PARAMETER replaced by its shadow and every buffer carried over verbatim, and it
     load_state_dict(strict=True)s into a fresh model.
  4. SCORER CONTRACT: load_for_eval is still a 2-positional-argument call returning an
     nn.Module in eval mode; it hands back the EMA weights when "ema_state" is present and
     EVAL_PREFER_EMA is on, and the RAW weights when EVAL_PREFER_EMA is off -- both strict.
  5. NOT A NO-OP: the EMA weights actually differ from the raw ones, and produce a different
     (finite) loss, so a passing run cannot be silently evaluating the same thing twice.

Run: python research/scratch/weight_ema_check.py
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train as T  # noqa: E402


def build(seed: int = 1234):
    torch.manual_seed(seed)
    m = T.GPT(T.build_config())
    m.init_weights()
    m.eval()
    return m


def main() -> int:
    T.COMPUTE_DTYPE = torch.float32  # bf16 would hide real rounding differences on CPU
    failures = []
    tmp = Path(tempfile.mkdtemp(prefix="ema-check-"))
    b, t = 2, 128
    torch.manual_seed(0)
    idx = torch.randint(0, T.TOKENIZER_VOCAB_SIZE, (b, t), dtype=torch.int32)
    targets = torch.randint(0, T.TOKENIZER_VOCAB_SIZE, (b, t), dtype=torch.long)

    model = build()
    names = [n for n, _ in model.named_parameters()]
    live = [p for _, p in model.named_parameters()]

    # ---- 1. off is the identity ------------------------------------------------------
    T.USE_WEIGHT_EMA = False
    raw_path = tmp / "raw.pt"
    T.save_checkpoint(raw_path, model, {"step": 1}, ema_state=None)
    payload_raw = T.torch_load_weights(raw_path, map_location="cpu")
    no_key = "ema_state" not in payload_raw
    m_raw = T.load_for_eval(str(raw_path), torch.device("cpu"))
    raw_same = all(torch.equal(payload_raw["model_state"][n], dict(m_raw.named_parameters())[n])
                   for n in names)
    print(f"[1] toggle off: 'ema_state' absent from payload={no_key} "
          f"load_for_eval returns raw weights={raw_same} keys={len(payload_raw)}")
    if not (no_key and raw_same):
        failures.append("USE_WEIGHT_EMA=False is not the historical behaviour")

    # ---- 2. lerp math ---------------------------------------------------------------
    decay = 0.995
    w = 1.0 - decay
    torch.manual_seed(7)
    shadow = [p.detach().clone() for p in live]
    ref = [p.detach().clone() for p in live]
    for _ in range(5):
        with torch.no_grad():  # stand in for an optimizer step: move the live weights
            for p in live:
                p.add_(torch.randn_like(p) * 1e-3)
            torch._foreach_lerp_(shadow, live, w)   # exactly as the training loop spells it
            for r, p in zip(ref, live):
                r.mul_(decay).add_(p, alpha=w)
    exact = all(torch.equal(s, r) for s, r in zip(shadow, ref))
    worst = max(float((s - r).abs().max()) for s, r in zip(shadow, ref))
    # The shadow must stay a graph-free leaf. Without the loop's no_grad the in-place lerp into a
    # requires_grad=False shadow from requires_grad=True Parameters silently makes it a graph node
    # that pins a fresh autograd graph every step -- correct numbers, growing memory.
    clean = all((not s.requires_grad) and s.grad_fn is None for s in shadow)
    print(f"[2] _foreach_lerp_ vs decay*s+(1-decay)*p over 5 updates: bit-identical={exact} "
          f"max_abs_diff={worst:.3e} ({len(shadow)} tensors) shadow graph-free={clean}")
    if not (exact and clean):
        failures.append("the foreach lerp does not implement the EMA recursion cleanly")

    # Direction check on a scalar: with the live value pinned, the shadow must approach it,
    # by exactly (1-decay) of the remaining gap each update.
    s0 = torch.zeros(())
    p0 = torch.ones(())
    gaps = []
    for _ in range(3):
        with torch.no_grad():
            torch._foreach_lerp_([s0], [p0], w)
        gaps.append(float(p0 - s0))
    want = [(1 - w) ** (i + 1) for i in range(3)]
    ok_dir = all(abs(g - x) < 1e-9 for g, x in zip(gaps, want))
    print(f"[2] direction: gap to a pinned target {['%.6f' % g for g in gaps]} "
          f"(expect {['%.6f' % x for x in want]}) -> {ok_dir}")
    if not ok_dir:
        failures.append("shadow/live arguments look swapped (the shadow is not tracking the live weights)")

    # ---- 3. state substitution ------------------------------------------------------
    ema_state = T.weight_ema_state(model, names, shadow)
    plain = model.state_dict()
    same_keys = set(ema_state) == set(plain)
    params_are_shadow = all(torch.equal(ema_state[n], s) for n, s in zip(names, shadow))
    buffers_verbatim = all(torch.equal(ema_state[k], plain[k]) for k in plain if k not in set(names))
    fresh = build()
    fresh.load_state_dict(ema_state, strict=True)
    strict_ok = all(torch.equal(dict(fresh.named_parameters())[n], s) for n, s in zip(names, shadow))
    print(f"[3] weight_ema_state: same key set={same_keys} ({len(plain)} keys, "
          f"{len(names)} params, {len(plain) - len(names)} buffers) params=shadow={params_are_shadow} "
          f"buffers verbatim={buffers_verbatim} strict load={strict_ok}")
    if not (same_keys and params_are_shadow and buffers_verbatim and strict_ok):
        failures.append("weight_ema_state does not produce a drop-in replacement state_dict")

    # ---- 4. scorer contract ---------------------------------------------------------
    T.USE_WEIGHT_EMA = True
    ema_path = tmp / "ema.pt"
    T.save_checkpoint(ema_path, model, {"step": 1, "weight_ema_updates": 5}, ema_state=ema_state)
    payload = T.torch_load_weights(ema_path, map_location="cpu")
    both = "model_state" in payload and "ema_state" in payload
    raw_untouched = all(torch.equal(payload["model_state"][n], p.detach()) for n, p in zip(names, live))

    import inspect
    sig = list(inspect.signature(T.load_for_eval).parameters)

    T.EVAL_PREFER_EMA = True
    m_ema = T.load_for_eval(str(ema_path), torch.device("cpu"))
    got_ema = all(torch.equal(dict(m_ema.named_parameters())[n], s) for n, s in zip(names, shadow))
    T.EVAL_PREFER_EMA = False
    m_back = T.load_for_eval(str(ema_path), torch.device("cpu"))
    got_raw = all(torch.equal(dict(m_back.named_parameters())[n], payload["model_state"][n])
                  for n in names)
    T.EVAL_PREFER_EMA = True
    ok_contract = (both and raw_untouched and sig == ["checkpoint_path", "device"]
                   and got_ema and got_raw and not m_ema.training
                   and isinstance(m_ema, torch.nn.Module))
    print(f"[4] contract: signature={sig} both weight sets saved={both} raw untouched={raw_untouched} "
          f"prefer_ema->ema={got_ema} prefer_ema off->raw={got_raw} eval mode={not m_ema.training} "
          f"-> {ok_contract}")
    if not ok_contract:
        failures.append("load_for_eval's contract or weight selection is wrong")

    # ---- 5. not a no-op -------------------------------------------------------------
    diff = max(float((dict(m_ema.named_parameters())[n] - payload["model_state"][n]).abs().max())
               for n in names)
    with torch.no_grad():
        l_ema = float(m_ema(idx, targets))
        l_raw = float(m_back(idx, targets))
    finite = all(torch.isfinite(p).all().item() for p in m_ema.parameters())
    ok_real = diff > 0.0 and finite and l_ema != l_raw
    print(f"[5] not a no-op: max|ema-raw|={diff:.3e} loss ema={l_ema:.6f} raw={l_raw:.6f} "
          f"all-finite={finite} -> {ok_real}")
    if not ok_real:
        failures.append("the EMA weights are identical to the raw weights (averaging did nothing)")

    T.USE_WEIGHT_EMA = False
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
