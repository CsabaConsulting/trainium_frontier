# Operator terminal receipt — trn21 r10 read-only timing, 2026-09-25 11:10 UTC

**Disposition: HOLD, one read-only timing attempt consumed. Do not retry this
label or fixed result path.** This is not source-stage, native-training,
evaluation, or submission authority. No host file write, bundle/target
creation, one-shot stage marker, preexisting-process signal, training,
evaluation, cleanup, breaker reset, or AWS action occurred.

The exact reviewed `invoke_timing_readonly.py` was invoked once from the
clean stage execution root
`/home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r3-active-root`
at HEAD `7205abf5469edc616c4e4b58caee27bc1957bbd4`. The separate timing
package was clean at HEAD `490bb4e533d41e095cea3a69eb041a6c42ff6085`.
Pinned science/stage/timing bytes and the enabled request, independent
review, Supervisor assignment and invoker hashes matched before contact;
the fixed result path was absent. Active Desktop controller launch flags
were false, global/trn21 breakers clear, and listed local STOP paths absent.

A separate reviewed read-only trn21 snapshot at 11:09:46 UTC observed
57,896,996,864 root free bytes, 31,563,301 free inodes, clean pinned e65
source, no matched training/evaluation process, no Neuron/device/host-lock
holder, and no observed host STOP. Its raw receipt SHA-256 was
`694102330ebb1a41c9bc5d47589cb517593a50be9962bed2745d00921700dad4`
on branch `codex/20260925-dual-host-readonly-1021`. This was a
point-in-time observation, not source-stage preflight.

The one-shot invoker exited **3** after creating the fixed
[raw result](OPERATOR_RAW_TRN21_R10_TIMING_RESULT.json), SHA-256
`9573fae597fcef03e49419cbe6886541d3109b476ad89af26f4887806b51e34f`.
The remote read-only SSH command itself exited 0 without timeout or stderr;
the local execution root and active-controller state remained stable, and
the JSONL parsed cleanly. The remote full read-only preflight took
`1.4248434960609302` seconds. **Trial 1 of 3 stopped before any actual
open:** `RuntimeError: final label scans too old for first O_EXCL`. The
recorded pre-open helper elapsed time was `1.1614667139947414` seconds;
the terminal status is `READONLY_TIMING_HOLD_NO_OPEN` with one completed
trial. No passing per-trial label/space/activity ages were returned, so
their margins are unknown. No second timing attempt is permitted.

A separate read-only postflight at 11:10:35 UTC observed 57,896,972,288
root free bytes, 31,563,301 free inodes, the same clean e65 source, and
no matched process, Neuron/device/host-lock holder, or host STOP. Its raw
receipt SHA-256 was
`38416da210cf9d3841033085dc98393279ea7bf779b2813108243dd09b74987f`
on `codex/20260925-dual-host-readonly-1021`.

The actual r10 source stage remains HOLD. Any revised timing or source-stage
design requires a distinct package and independent review; this result is
not a reason to weaken the one-second gate or reuse the consumed label.
