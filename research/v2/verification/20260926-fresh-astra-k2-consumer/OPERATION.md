# Fresh K2 consumer source-review operation

Actual independent verdict: **REFUSE**, for a concrete checker nonfinite failure-reporting defect. The reviewer was a new packet-only process; this operator is not that reviewer and did not supply prior conclusions, notes or tests.

- Exact candidate: `077481d0476cebfd79ddc751f8f63e61c1fbf82f`.
- Requested/CLI-selected model: `gpt-6-astra`, high reasoning. Provider-resolved model was not separately echoed; do not invent a second attestation.
- Session: `01a0da5c-e47c-76c2-997d-ab14ed900de1`.
- Start `2026-09-25T20:58:35.867000Z`; end `2026-09-25T21:03:43.380815Z`; exit0; 307.5138180610011 seconds.
- Input manifest: `5e0dde0832d3031b56ea4143a705808e1bd62326674799837b3444e1af440561`.
- Actual usage: 473,666 input tokens, 421,768 cached input, 51,856 cache-write input, 17,915 output, 3,803 reasoning-output tokens. These are the event fields, not mutually disjoint billable totals or a price estimate.

The input contains only a complete three-new-file diff (`consumer_nki.py`, `check_native.py`, fixture/reference support), neutral lane specification, and two raw stdlib receipts. The implementer's tests/README/CHECKS, branches, repository, conversation and all prior reviewer findings are absent. The neutral specification explicitly distinguishes suitability for a future bounded falsifier from native/full-model/adoption acceptance.

The previously reviewed operation mechanism was reused with only the packet path/hash pin changed: fresh home/work, external bubblewrap read-only packet/system/CPU-library mounts, no original repository/home/SSH or process-environment filesystem. Codex runs ephemeral, ignores user config/rules, and cannot fork/resume a previous session through this invocation. Its existing API key is loaded in memory only for the provider-facing CLI; shell tools receive a credential-free explicit environment. Provider networking is permitted to the CLI; tool networking is forbidden by the supplied specification rather than a claimed network-namespace isolation. Logged tool actions stayed in packet/work and standard CPU libraries. No Opus/Fable call, retry, refusal rewrite, source edit, host action or native execution occurred. CLI help and official documentation were checked before starting.

## What the independent reviewer actually executed

The reviewer wrote its own integer-based IEEE FP32 recurrence oracle, separate control-flow emulator and single-thread actual CPU embedding-backward comparisons. **42 cases passed**, including 22 independently chosen cases and 20 supplied fixtures; **10 negative controls** exposed occurrence reordering/grouping or BF16 output rounding. Five malformed metadata variants were rejected. Eight normal/optimized OFF checks observed zero forbidden native/tensor imports. This is tiny fixed-bound CPU evidence only; the actual NKI kernel was never executed.

It then extracted the checker's exact comparison helpers and reproduced the reporting defect with 1x1 CPU tensors: +Inf, -Inf and NaN generate nonfinite `max_abs`, and `allow_nan=False` throws at both case and final-receipt serialization. Three invalid partial rank JSON files and six tracebacks are preserved verbatim. Finite exact, finite mismatch and signed-zero mismatch controls serialize correctly. Nonfinite comparisons remain `exact:false`; the bug is incomplete failure evidence/cleanup, not false numeric acceptance. No tolerance was changed.

The full independent report and exact reproduction are in `session/reviewer-output/report.md`, `decision.json`, `checker_failure_probe.py`, and `checker_failure_results.json`. Resolution is a narrow JSON-safe diagnostic/failure-receipt repair plus retesting, not a kernel/threshold rewrite. `native_executed:false`, launch and production adoption false, whole-step speedup unmeasured remain mandatory.

## Durable evidence and integrity

449 verbatim archive files (35,640,291 bytes) were copied after terminal exit; archive manifest SHA256 `e00d07f9c5aa600d27f3b38c0ed0cfdda8fdc71a536261ed204231ccc287cc11`. `validate_archive.py` independently authenticates every archive entry, all 430 reviewer-manifest work files, all five packet files, the single new session/turn, actual usage, 42 case result and three malformed failure receipts. Later operator summary/validation files are supplementary, not members of that earlier immutable archive manifest.

Key hashes:

- Report: `bdc256d9dac56c0161e8f7b6fd48d6854be93c40b0373ca5685a0d030477aa3a`.
- Decision: `42e71663e4a6d66c372a41470afcef55899e31cb09b1691d81ecbd2adaf82fdd`.
- Independent reference test: `7a116f5c87ecaead143681d4474caf851e4d8ddb62ee4d02e2fccbcaa11c2cf5`.
- Reference results: `1a648a8d671ae2ac71ae165f05c5937bf7cc95b6dc48b9efd8c5c8036ce52981`.
- Exact defect reproducer: `9a3a2b33fd01623308f5bba4f7a81e608f84679202fed8130b4905466e5232ea`.
- Defect results: `563007a71ce8765aa8c90a783fbda5de687c20783d89503955dad913bf6016e4`.

No rerun of CPU tensor work was done by the operator to manufacture a second approval. The bounded CPU slot was released on terminal completion and Implementer was notified. The fresh REFUSE supersedes any earlier technical-only suggestion that the checker had no blocking issue. Source/compiler/native binding gates remain separate; no next native job is authorized by this archive.
