# Narrow receipt-validator repair — disabled, pending re-review

Original packet `54fd9a10d0beee22d2d8209a3dce9d3f200da2b0` remains REFUSED and
unlaunched. The independent reviewer supplied four authenticated synthetic
counterexamples; its original log is preserved verbatim in
`repair-evidence/original-reviewer-refuse.log`. No reviewer files were edited,
and no reviewer was contacted. Replaying the exact old validator still reproduces
all four false acceptances. The successor rejects all four and accepts the
reviewer's positive baseline. These are synthetic validator tests, not native
receipts, independent re-review or native qualification.

Only validation/tests/docs and affected hashes changed:

- Require the full same-input forward records to agree across registered eager,
  registered compiled and ordinary compiled routes.
- Reconstruct all six deterministic input tensor records, not only hashes:
  shape, dtype, finite state, nonzero count and SHA. The weight record cannot be
  reconstructed from its hash; it must remain well-formed and agree across
  routes/ranks. Route upstream must equal the separately checked roundtrip.
- Require typed, nonempty exact runtime identities on all four ranks. Prior raw
  bundle `db4fd0b148fd29930bc4925ba898a8f80108d1bd84c6ad3efc33e8694e15e2c6`
  from `015bacec` authenticates all31 files /213475 decoded bytes and four rank
  identities Torch2.11.0+cpu / NKI0.5.0+28631259367.ga768afa6. Authentication
  details are retained in `repair-evidence/runtime-authentication.json`.
- Validate old-route actual/expected tensor schemas, expected reference records,
  exact/status agreement, finite/nonfinite error-field conventions and explicit
  missing/error metadata. Preserve proper zero dW, nonfinite, missing output,
  execution/post-observation errors and signed-zero mismatches as separate old
  observations. Their presence neither substitutes for nor suppresses required
  candidate comparisons. Error records may truthfully retain partial observations;
  missing/malformed fields cannot masquerade as a completed mismatch.

The checker, adapter, kernel, reference, rank entry, dispatcher and shared guard
are byte-identical to54fd9a10. Driver executable structure/argv/env is unchanged:
only its `PINS['validate_receipts.py']` string changed. Binding/manifest and the
driver constants diff were refreshed. No production/scoring/source math/tolerance,
cache policy, host identity, label, budget, containment or launch authority changed.

Own final tests:114 stdlib checks normal/-O, including43 refusal cases, seven
proper old-failure variants, original nine mocked driver scenarios and the
unchanged source/guard checks. The four new all-rank mutations do not rely on
cross-rank disagreement to detect the defects. All five authenticated reviewer
inputs were separately replayed normal/-O against original and successor.
The original90 checks are retained as historical evidence, not sufficient proof.
No tensor/native imports, host calls or launches occurred during this repair.

Reproduce from the worktree:

```sh
python3 -I -B research/v2/proposals/20260926-k2-registered-native-trn22/test_packet.py
python3 -I -B -O research/v2/proposals/20260926-k2-registered-native-trn22/test_packet.py
python3 -I -B research/v2/proposals/20260926-k2-registered-native-trn22/replay_repair_repros.py --reviewer-dir /tmp/trainium-k2-registration-review-GS6xFb/native-packet/native-review-output
```

For the last command, an immutable archived copy of that reviewer directory can
be supplied instead; all five file hashes are pinned. Add `-O` for its optimized
replay. The script reads the evidence only and does not run the reviewer's writer.
Full error magnitudes for arbitrary failed old outputs cannot be reconstructed
from a SHA alone; finite nonnegative/nonfinite conventions and logical record
consistency are checked, while required candidate gradients are independently
reconstructed exactly. This is not an assertion that old outputs became correct.
