# Independent final-r3 late-label addendum — HOLD supersedes PASS

Reviewed 2026-09-25 04:59 UTC, static source/timing audit only. This addendum
**supersedes the final-r3 PASS** in
`INDEPENDENT_CONTINUITY_FINAL_R3_REVIEW.json` (SHA-256
`1069bba48756a5977498fb214094413add9a67c8d06633c9171d45f10fd5210f`)
and its companion narrative (SHA-256
`e9bb77cf69c601d9f642d7977cf9f230eeccb3ce203631e9dfaa6065345cf1b0`).
They remain preserved as historical evidence of the checks then performed,
but **must not be treated as current final authority**. No Supervisor
assignment or Operator dispatch may use that PASS. The reviewed enabled
request remains Implementer commit `fd8b0c74ed2c90905411b225471840c608cad17b`;
the affected remote writer is SHA-256
`e93efd99c0a58aa32537d945ddc8fc29f9d7ce0dabb89ed93526da30421f4086`.
No remote/transport/request bytes were changed by this review.

## Blocking source-level finding

In `stage_r4_remote.py`, the second `_live_prewrite_safety()` call at line
613 is the last full live gate. Inside it, the **last science-label scan** is
at line 563. It is then followed by checkpoint, shared-lock and Neuron
`fuser` subprocesses and `neuron-ls` at lines 564–567; each subprocess can
run for nearly its 20-second timeout and still return a passing result
(lines 444–457). After `_live_prewrite_safety()` returns, `write()` performs
another potentially lengthy STOP/process walk at line 614 (implementation
lines 546–555). Only then does it create the target with `O_EXCL` at lines
615–616. Neither the later fuser/Neuron calls nor the final STOP/process
walk repeats `_label_scan()`.

A competing process can create a path bearing the r4 science label just
after line 563's scan and exit before the final process walk. The remaining
holder, Neuron, STOP and process checks can all be clean while that label
path persists, so the exact target can be created with the no-concurrent-
science/unused-label condition already false. The recorded
`last_live_precreate_safety.science_label_scan` is the earlier clean snapshot;
`stage_r4_transport.py` lines 437–439 validate that snapshot and only a
subsequent STOP/process snapshot, not a label recheck. A later postflight
may discover the label and return HOLD, but it cannot undo target creation.
The prior **32/32** tests injected activity before the second full scan or
tested the receipt's fields; they did not cover label creation **after its
last scan but before `O_EXCL`**. Their PASS result therefore does not close
this timing gap. This is materially wider than the inherent tiny
check-to-create race explicitly acknowledged in the prior review.

**Disposition: FINAL ENABLEMENT HOLD; no transfer.** Preserve the existing
PASS JSON and narrative for provenance, but mark them superseded by this
addendum and do not issue or rely on a Supervisor one-shot assignment. The
existing wrapper does **not** read this addendum; if a matching assignment
were later issued, the old PASS could still satisfy its file checks. Thus
withholding assignment and dispatch is essential until repaired bytes receive
a new review. A read-only local existence check at this audit found the
canonical assignment and fixed marker/journal absent. A separate Implementer
must repair the final label-to-create ordering and
corresponding receipt/validator and exact hash bindings; a fresh independent
final review must assess the new bytes. Fresh host preflight remains a later
gate and cannot repair this static writer defect. This audit made no SSH,
trn22/AWS contact, marker/journal, host write, transfer, checkout, G2,
native run, evaluation, cleanup, breaker reset or submission. No adversarial
injector or other code edit was needed for the source-level conclusion.
