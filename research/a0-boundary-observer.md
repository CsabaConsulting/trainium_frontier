# Direct A0 boundary observer

`research/a0_boundary_observer.py` is a read-only sidecar for the already-approved generic
fresh-start guard. It does not launch, signal, stop, or write any runtime file. Start it before the
guarded request and capture its sole stdout JSON outside the science checkout.

Example shape (paths and label must come from the independently verified A0 request):

```text
python research/a0_boundary_observer.py \
  --label <exact-label> \
  --process-json <checkout>/research/run-logs/<exact-label>.process.json \
  --boundary-file <exact-NEURON_COMPETITION_R1_STEP_LOG> \
  --postflight-json <checkout>/research/run-logs/<exact-label>.postflight.json
```

The observer binds the immutable process receipt to live guard/child PID start identities, requires
an append-only canonical `0,1,2,...` boundary stream, and records the first observation of every
boundary with monotonic and UTC clocks. Seeing multiple boundaries in one poll, a gap, duplicate,
rewrite, truncation, PID reuse, invalid terminal postflight, or timeout makes the result non-scored.
It still observes until terminal postflight (or timeout) and emits the complete record. The timing
object includes launch-UTC to fifth-boundary observation, process-receipt observation to fifth
boundary, and fifth-boundary to child-terminal/postflight observations.

Local test only (fake processes and temporary files; no Neuron or host access):

```text
python -m unittest -v research/test_a0_boundary_observer.py
```

This sidecar supplies timing evidence only. It does not establish G2, scientific correctness,
training success, budget compliance, evaluation validity, or a verdict.
