# trn21 c5c A0 four-rank CPU G2: safety-hardened final-pending package

**Status: offline Implementer artifact only; dispatch HOLD.** This package is
not an Operator request, a Supervisor assignment, a CPU G2 result, or a native
training authorization. Its contract keeps CPU G2, training, evaluation and
submission authorization `false`, and direct execution of its builder exits
`DISPATCH HOLD`. No SSH, host operation, CPU G2 or Neuron run was done here.

## Why this successor exists

The independently reviewed stage-r2 HOLD draft remains immutable at
`research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2/`.
Its static reviewer found that its offline builder accepted three non-argv
mutations: `one_attempt_per_label=false`, CPU tool `required_mode=0777`, and
`preflight_inode=0`. Those values cannot be treated as launch-safe merely
because the constructed argv hash stays unchanged. This successor checks all
three against the pinned transfer result, the final reviewed checkout request,
and hard-coded safety invariants. It also checks the contained-tool inode,
ownership, device, link count, bytes, SHA-256, and regular-file type.

The immutable 64-word four-rank CPU launch argv and three companion command
arrays are constructed by the old reviewed builder and rehashed here. No
science parameter, train argv, effective environment, source checkout, CPU
label, scratch root or command digest changes. The 15-element science suffix
remains the ratified c5c A0 control on trn21, seed 42, LNC 2, 1800 s.

## Authenticated lineage and the still-missing result

- Source checkout: `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`;
  e65 commit, `train.py` SHA-256 `c5c410af…`, unmodified `prepare.py`
  SHA-256 `5807c46e…`.
- The reviewed transfer terminal receipt and separate transfer-result review
  authenticate the persistent CPU tools. The preflight tool is UID 1000,
  ext4 device 66305, inode **294946**, mode `0600`, one link; contained is
  inode **296647** with the same policy. The builder rejects inode zero,
  mismatches in either the cross-session result or final postflight, and
  mismatches against the later checkout request's pinned target inventory.
- The exact enabled checkout request, independent request-review PASS, and
  Supervisor **checkout-only** one-attempt assignment are SHA-bound here.
  None of them grants CPU G2 authority. This package does not infer a
  completed checkout from request-level approval.
- The checkout Operator terminal receipt and independent checkout-result
  review are deliberately `null`. When they arrive, a separate reviewer
  must authenticate both exact file hashes and prove a clean one-commit e65
  checkout, the same path, c5c/prepare hashes, preserved CPU tools, intact
  checkpoints/caches, absent science output, full postflight and no holder.
  They must be bound into a **new** exact CPU-only request. Do not mutate
  this reviewed disabled object in place or substitute an older checkout.

Even after a clean checkout, a later CPU action needs an independently
reviewed final request, a fresh full live host/source/tool/label/STOP/breaker/
disk preflight, and a separate one-attempt Supervisor CPU-only assignment.
The G2 scratch root and label must be absent, the scratch root must be on the
same persistent ext4 device, and root free space must be at least
10,737,418,240 bytes before launch. The CPU tool bytes and inode must be
rehash-verified with no symlink following immediately before and after.
The Operator must archive the exact command transports, four rank reports,
receipt, logs, step lines and an unconditional separate read-only postflight;
failure preserves partial artifacts and consumes the attempt. A distinct
Verifier then compares numerical output with the pinned c5c CPU anchor.
No CPU PASS can by itself authorize native training or public evaluation.

## Implementer-only offline checks

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2-final-pending/test_g2_argv.py -v
```

The nine test methods exercise the default frozen command digests; both old
and new forms of all three reviewer-found mutations; independent transfer,
postflight and checkout-request inode/mode drift; source/review/assignment
drift; attempted dispatch spoofing; and direct-execution refusal. These are
synthetic offline checks only, **not** an independent Verifier receipt.
