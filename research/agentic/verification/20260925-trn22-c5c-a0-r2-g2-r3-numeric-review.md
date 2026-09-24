# Independent numeric/integrity review: trn22 c5c A0 r2 CPU G2 retry r3

Reviewed 2026-09-24 23:15 UTC by a Verifier independent of the r3 package
Implementer and Operator. This is an archived-evidence review only. I did not
connect to or mutate a host, run native training/evaluation, reset a breaker,
adopt a model change, or submit anything.

## Verdict and authority boundary

**PASS for the exact-environment, four-rank, three-step CPU G2 numeric and
archive-integrity gate. Native A0 r2 remains HOLD.** The retrieved r3 receipt
matches the prior c5c CPU PASS anchor bit-for-bit on the comparison fields.
This is a synthetic CPU preflight, not a public BPB result or evidence of
Trainium compilation, throughput, budget compliance, or model improvement.
It does not turn `research/v2/state.json` launch authorization on. A separate
immutable native request, its independent review, fresh live source/host/disk/
cache/process checks, external step observer, and one-attempt Supervisor
authorization are still required before the unconsumed science label
`fs-20260925-v2-c5c-a0-control-trn22-r2` can run.

## Evidence binding and checks I re-ran

- R3 package: `research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/`.
  `GATE_CONTRACT.json` SHA-256
  `164ac39016622ac3918341835b43a53cf3bd94f03bc29596b574d11d1e8ed5bd`;
  `RUNBOOK.md` SHA-256
  `31eeeecdd8f70d3fd15adce3c13d2141236ab643d823598fd9138c305fbfa53a`.
  The independent prior static/safety review is
  `research/agentic/verification/20260925-trn22-c5c-a0-r2-g2-r3-static-safety-review.md`
  (SHA-256 `36a65541e46981e159b121134349fc4176c385bb743a98dd92af101eda581617`).
- Operator archive commit `a1614ead` adds only the r3 receipt and ten retrieved
  evidence files plus their manifest; no source or scoring file. The Operator
  receipt hashes to `c8d73cc447a4fb99c9aba1869f3df47accee550ea2e04d704b9a43cd1bd8e116`.
  Its `remote-run/SHA256SUMS` hashes to
  `bfceeacbf3df6dd24698434bee3d41ee01bd28229f7d55272e1f609093270c2a`.
  I ran `sha256sum -c SHA256SUMS` in that archive: **all ten files OK**.
  The r3 G2 receipt itself hashes to
  `38dd39a02d86d639f6333228048f8bdb9dcc77735f03c2d45963e1057bf945b9`.
- The prior c5c CPU PASS anchor at
  `research/v2/g2-probes/fs-20260925-v2-g2-baseline-s0-trn22-r1/remote-run/receipts/fs-20260925-g2-a0-research-v2-trn22-r1.json`
  hashes to `17ce1ec84ddc0d585ecd103237ecfcd86d3820cdd9aaf1c4fe8b60ed308b90e3`.
  The exact pinned `e65a3097a5b00f6178188a935bb2acb9a5042c6c` Git blobs
  independently hash to c5c research `train.py`
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and unchanged `prepare.py`
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The local comparison tool hashes to
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`,
  the same hash authenticated for the host G2 tool.
- I ran `python workflow/tools/preflight_dryrun.py --compare` on the
  hash-checked anchor and r3 receipts. It exited **0** with
  `identical: true`, `plan_equal: true`, `losses_identical: true`,
  `state_keys_only_in_one: []`, `param_digest_differences: []`, and
  `n_params_compared: 62`. An additional strict JSON comparison returned
  true for both source hashes, exact Track-A argv, four ranks, three steps,
  2x256 CPU micro-batch, shims, plan hash and summary, every rank's status/
  losses/changed count, rank-0 checkpoint check, and all 62 final digests.
  The expected r3 temporary `--out-dir` differs from the old anchor; both
  receipts append only `--device-type cpu --no-compile --num-steps 3` and an
  isolated temporary output path.
- The consolidated r3 receipt is `PASS`, `torchrun_rc=0`, and its four
  separately archived rank JSON files match its `ranks` objects exactly.
  Ranks 0, 1, 2 and 3 each report PASS, three finite numeric losses, one
  identical ownership-plan SHA-256
  `b89d62227c844eb13706fbe81bddfb9b8b7db5fefcd57b4e7d0ea3213dca5031`,
  62/62 changed parameters, and no cross-rank mismatch. Rank 0's model-only
  checkpoint/load check is `ok: true`, `deterministic: true`, with logits
  shape `[1,256,8192]`; all 62 final parameter digests match the anchor.
  The separately archived isolated step log is exactly `0`, `1`, `2` plus
  final newline, one boundary per synthetic optimizer step.
- The retrieved containment launch is a unique r3 label with `env -i`, the
  pinned clean checkout, exact Track-A argv, four effective science values
  (`LNC=2`, compiler optlevel-1 flags, seed 42, budget 1800), and CPU-only
  G2 options. The retrieved quiescence receipt reports a stable 10-second
  window, zero tagged processes and Neuron holders, and hashes of the G2
  receipt/run log matching the archive. The Operator records no native
  training, fresh public evaluation, or BPB.
- I independently compared the archived r2 stage-HOLD receipt to the r3
  Operator receipt. The retained G1 checkpoint SHA-256
  `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`,
  all six warm-cache directory device/inode identities, pinned commit,
  `train.py` hash and `prepare.py` hash agree. The r3 Operator postflight
  records unchanged checkpoint/caches, clean source, absent science output/
  log/step paths, no lock or device holder, and 58,452,930,560 B free on `/`.
  Those are authenticated archived assertions, **not** a fresh live native
  preflight; the Operator must recheck them immediately before any native run.

The earlier r2 CPU G2 failed because its step-log parent was missing; it is
not a negative model result. The independently reviewed safety classification
found no second native-launch or global correctness breaker from that CPU
infrastructure failure. This numeric review does not reset or reinterpret
that classification. Any changed source/protected artifact, active STOP,
missing observer, or failed live check returns the native request to HOLD.
