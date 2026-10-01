# LaneB package integrity review

Verdict: **PACKAGE_INTEGRITY_ONLY** for frozen package manifest
`2618fe90bd2580c76e08eca28c599401f1428c1ddcd224aa082713b6aad78ace`.
The independent stdlib reference completed 586 checks with exit 0. No integrity
discrepancy was found within the allowed input scope. This is not a model/source,
native-equivalence, Claude-compliance, clean-box or submission acceptance.

This is the reused isolated session `/root/r136_sb_l2_isolated_verifier`, not a
fresh review session. Root reported the exact fresh-spawn failure:
`collab spawn failed: agent thread limit reached`. The earlier S-B/L2 source
review was parked without CPU execution when its adoption condition failed.
The reviewer did not implement the LaneB source or this submission package.

The new `check_integrity.py` imports only Python standard-library modules. It
does not import, read or run the supplied package checkers/builders or read their
work notes, PARTIAL.md, branch or current-state files. It reads the allowed
source/diff/spec, checklist/manifest claims and raw evidence. Thirteen allowed
package contents are hash-checked; the eight deliberately unread manifest
members are listed in RESULT.json. Therefore this does not claim all 21 local
package members were independently inspected or rehashed. All 97 members of
the two raw FULL archives, 48 ancestor archive members and both seven-member
CPU archives were independently authenticated; unneeded raw members were only
hashed, not interpreted as instructions or review conclusions.

The package source is byte-identical to `dcdef066`, and its command bytes are
exactly the command recorded in both actual packets and FULL archives:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

Own unified-diff replay validates every context/removal line and reconstructs
every candidate byte from the exact `e1bd539f` parent. No author EDITS/checker is
used for this proof. The source AST contains no `.training` attribute; that
limited static observation does not establish whole-model native parity.

| Raw FULL | Public BPB | Fifth-boundary cold | Charged through exit | Steps |
| --- | ---: | ---: | ---: | ---: |
| trn21 / `05a76408` | 0.9663196934983302 | 663.9014979940839 s | 1793.414417522028 s | 2886 |
| trn22 / `defb95ea` | 0.9659050462155375 | 645.364225326106 s | 1792.7041728659533 s | 2895 |

Both evaluate 20,971,520 public tokens at context 1024 and batch 1, with recorded
zero training/evaluation exits. The independent reference recomputes BPB from
raw total nats and byte counts, and charged time from wall time minus the capped
fifth-boundary allowance. It validates all boundary step indices, source/command
bytes, seed42/full-chip environment, clean source snapshots, six empty private
training-cache receipts, terminal/postflight idle records, and matching stable
checkpoint hashes/sizes through collection. Their mean is
**0.9661123698569338**. These are same-seed research receipts, not official scores
or independent-seed evidence. Checkpoint bytes were never loaded or copied.

The first packet/archive truly contains `R113_ROOT_SELECTION.json`; the ancestor
contains `R100_ROOT_SELECTION.json`; repeat22 contains `R98_ROOT_SELECTION.json`.
The two adapter diffs each replace only one lookup string. Own checking reads
the real members directly, compares packet/archive bytes and their existing
host/source fields, and verifies no synthetic R98 member appears in the legacy
archives. No adapter or supplied helper is executed, and no field is invented.
The direct ancestor's separate raw public0.9672630039667548/cold678.5742937349714/
charged1793.4091268959455 receipt is authenticated and excluded from LaneB's mean.
It does not establish a Claude-audited ancestor.

The existing author10/independent8 CPU receipts are authenticated against their
raw result and exit records; own8 contains125 metrics. They explicitly use eager
CPU kernel surrogates and mark native execution false. Existing `c5ebb332` remains
the sourceCPU authority. This session adds no numerical/source acceptance and
does not independently re-evaluate the native benchmark or inherited code helpers.

The eight-item brief is appropriately incomplete: only item4 passes for these
measured research receipts. Items1/2/3/5/6/7/8 remain FAIL with the stated gaps:
clean-box imports, whole-model native train/eval identity, runtime/cache isolation,
cumulative writes/current leftovers, minimal-environment sufficiency, inherited
timing-dependent table/stop behavior, and retained disabled paths/documentation.
The brief distinguishes reused evaluation caches from empty training caches,
rounded token logs from exact counters, and stable-at-collection checkpoints
from present retention or backup. The A3-identical compiler guard is correctly
not presented as proof that main/import compiler policies are equivalent.
No missing claim is silently promoted to PASS. Claude audit and last audited
ancestor remain UNKNOWN; the package's acceptance/export/launch/upload flags
remain false. A new integrity report does not satisfy those separate gates.

Reproduce from the repository root:

```text
python3 research/v2/verification/20260930-r137-laneb-package-integrity-review/check_integrity.py
```

Test SHA256: `05f683bcd2d9a7988ab69e2fc7ba80607e763ed70d24ef3790ab4fea3f605221`.
No host calls, network, Torch/model execution, package edits or state writes
occurred. Only this new verification folder was written.
