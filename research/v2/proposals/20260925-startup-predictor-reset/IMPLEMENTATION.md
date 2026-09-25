# Startup predictor reset: Implementer handoff

The research source now clears `recent_step_times` at the existing fifth-step
startup-exclusion boundary. The next-step maximum uses `default=0.0` until the
first post-startup sample arrives. These are the only two behavioral edits;
Recursive remains default-OFF and all model computations are unchanged.

Base commit: `3d9f794697cbc2c2260474f6b7676131507a4362`.
Branch: `codex/20260925-startup-predictor-reset`.
Original research SHA-256: `f9b6a9cbb1a1760fb8db422ffd79ffb91e9280899c40aa86028d935acf355f67`.
Corrected research SHA-256: `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.

The startup allowance still caps at 900 seconds, the existing launch margin is
30 seconds, and all excess is still charged. The training budget remains 1800
seconds, checkpoint reserve remains 15 seconds, and step-cap logic is unchanged.
All five excluded samples are removed, including a compile-heavy fifth sample.
The reset neither changes which time is charged nor provides an additional
startup allowance.

`test_startup_predictor.py` uses only the Python standard library. It parses the
real `train.py` and `prepare.py`, then executes the actual timer initialization,
boundary and stop statements with a synthetic monotonic clock. It never imports
either module, allocates tensors, starts training, or contacts a host.
Nine tests passed on Python 3.14.4 at 2026-09-25 16:03 UTC, in 0.779 seconds.
The exact command is:

```text
python3 research/v2/proposals/20260925-startup-predictor-reset/test_startup_predictor.py
```

The A2 durations `(1056.75, 903.47, 0.62, 0.63, 0.62)` reproduce the original
five-step stop in the preserved c5c timer. Corrected code continues, with the
same observed startup of 1962.09 seconds and charged elapsed/excess of 1092.09
seconds (`1962.09 + 30 - 900`). This replay isolates those step durations; the
real run additionally included initialization and process overhead. It is not
a replacement measurement for the recorded harness wall time.

Other tests cover exact boundary clearing, first warm samples, the twenty-sample
maximum, cap and margin charging, empty history, near-zero remaining budget,
checkpoint reserve, exhausted budget, step cap, and synthetic negative timing.
Negative post-boundary durations are nonphysical for the monotonic clock; their
pre-existing handling is deliberately unchanged. No new clock sanitization was
introduced under this narrow correction.

The pinned c5c source and scored 116188 source have the same bug, but both are
preserved. `prospective-116188.patch` is a separate one-hunk artifact intended
only for a **new isolated copy** of the 116188 vehicle. It is not applied to the
scored package or Downloads. The test applies that exact hunk in memory and
proves its timer AST equals the corrected research timer. It is prospective
source, not a submission candidate with cleared replay/audit gates.

Preserved SHA-256 identities:

- `submissions/research_v2/train.py`: `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
- `submissions/dense6x1024_direct_review_matrixlr015/train.py`: `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`
- `prepare.py`: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`
- Test source: `e732daf73db7fc099b3b54e06cffdf0009c35d98a4b04612db60f9feba3b5c14`
- Prospective patch: `939811d5f8580bc133065d33ddc1d4494aac3ccaedcbad4027a09e5b24130e3b`

`git diff --check` passed. This is an Implementer self-check, not an independent
Verifier approval, model G2, three-step model identity, native gate or BPB result.
Those gates and any future uniquely labelled host experiment remain separate.
