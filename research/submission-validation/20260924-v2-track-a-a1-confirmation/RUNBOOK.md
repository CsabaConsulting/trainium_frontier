# Track-A A1 promotion/confirmation lane

This package is preparation only. Both authorization files must remain invalid until exact archived
control/A1 evidence is bound and an independent Verifier changes the state. No pending command may
launch, and this Implementer package performs no SSH.

The fixed sequence is:

1. `pair-trn21` authenticates both trn21 retained checkpoints and the nominated fresh2M evidence,
   then evaluates each checkpoint at exactly 20,971,520 tokens / 83,462,186 bytes in separate fresh
   processes. A nonnegative candidate-minus-control delta discards A1 and blocks position 2.
2. `replicate-trn22` additionally authenticates the negative position-1 archive, evaluates the
   retained trn22 control at full20M, runs the exact A1 full1800 seed-42 command, and evaluates its
   finite retained checkpoint at fresh2M and fresh20M.

Final adoption requires both matched full20M deltas to be strictly negative and their Decimal mean
to be at most -0.001. A nonnegative delta discards. Two negative deltas with a shallower mean nominate
one separately approved repeat. The other-host result alone is never confirmation.

After independent finalization, local no-SSH materialization has this shape:

```text
python research/submission-validation/20260924-v2-track-a-a1-confirmation/launch_adapter.py \
  --arm ARM --concrete-request /tmp/UNIQUE.request.json --dry-run \
  --dry-run-origin-head EXACT_FINALIZATION_HEAD
```

Each runtime is protected by the standard host lock, 6.5-GiB operational/5-GiB hard floor, exact
Git/source/request/runtime fingerprints, no-follow retained-checkpoint hashes, private subprocess
sessions, bounded TERM-to-KILL cleanup and proven process-group disappearance before hashing. No
checkpoint or cache deletion is authorized.
