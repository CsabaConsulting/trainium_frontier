# Independent result review — trn22 r10 r5 source-stage attempt

Checked 2026-09-25 11:53 UTC. Verdict: **HOLD_ONE_ATTEMPT_CONSUMED_NO_ACCEPTED_STAGE**.
This is a result disposition only. It authorizes no second attempt, marker
reset, cleanup, native training, evaluation, AWS action, or submission. I made
no SSH or host contact.

## Exact evidence and binding

- Operator result `ad925338797fa34de93b1e3f76de216ee60955bd` descends from
  reviewed/assigned source `ed2ca7fc43d29f6909c001ee97e2875ef8ae4ee7`
  and adds only the archived raw journal and Operator receipt.
- Archived `RAW_ONE_ATTEMPT.jsonl`: 8,793 bytes, SHA-256
  `229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8`.
  It is byte-identical to the retained local one-attempt marker, whose
  point-in-time metadata was owner-only mode `0600`, one link. Operator
  `RECEIPT.md` SHA-256 is
  `066466c83c93bd3d62dbdee277ad6ab0cab346cbf9fc2d47b5b284ff7dd33176`.
- The first journal event binds the exact label and source-stage-only scope
  to request SHA-256 `25092856f0fef3c1e3cf85e40e45fa0d2d51971b7b8112f50dbc3761770971f2`,
  independent r5 review `da237ac25e8547c7cc9b0cc6dbf1fec9e2875af6be94b7e407d17616181a0994`,
  and Supervisor one-attempt assignment
  `806c89ad2aa03cd706d5c8548c2862b0c8d0eb7e4389e5486b65f0e3859037bc`.
  All three hashes match retained authority files.

## Recomputed terminal facts

The journal's four and only four events are `ONE_ATTEMPT_CONSUMED`,
`FRESH_PREFLIGHT`, `STAGE_HOLD`, and `UNCONDITIONAL_FULL_POSTFLIGHT`.
There is **no `STAGE_WRITE` receipt and no distinct-session readback**.
`STAGE_HOLD` records exactly `RuntimeError: SSH phase failed or emitted
stderr: write`. The Operator command exited 1/HOLD.

The raw fresh preflight reports host `ip-172-31-32-253`, e65 source commit
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`, science hashes
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
for `train.py` and
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`
for `prepare.py`, no target or bundle, no matching process/Neuron holder/STOP,
58,453,106,688 free root bytes and 31,660,014 free inodes. The preserved
checkpoint hash is
`1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`.

The unconditional postflight is a distinct remote process on the same host,
boot ID and root mount. It observes **bundle and target still absent**,
the same e65 source identity and science hashes, byte-identical protected
checkpoint/cache metadata, the same checkpoint hash, no matching
process/Neuron holder/STOP, 58,453,102,592 free root bytes and 31,660,014
free inodes. Both space samples exceed the predeclared floors. The raw
source snapshot does not contain a Git dirty-tree field; the Operator's
separate pre-marker receipt reports the source was clean, but this result
review claims only the source identity and protected assets proven by the
archived pre/post snapshots.

The reviewed `_ssh_phase()` collapses a nonzero return code, nonempty stderr,
or oversized stdout into the recorded generic write-phase error. The raw
remote return code, stdout and stderr were not archived. Thus the specific
remote cause is **unknown**; the earlier narrow one-second timing margin is
not evidence that timing caused this failure. Successful preflight and
postflight bound the failure to the write phase, but do not prove why it
failed. No completed source stage or BPB result exists.

## Safe disposition

Keep the consumed marker and all current host artifacts untouched. Do not
retry this label, silently reset the marker, clean a possible provisional
file, or launch native A0 from an unstaged r10 target. The Supervisor may
separately commission a bounded **read-only** diagnostic to learn the
write-phase failure class. Any future source-stage attempt needs a new
unique label/target/marker and a completed, independently reviewed proposal
and one-attempt assignment under the current human authority; this receipt
supplies none of those gates. Protected deletion and breaker reset remain
human-only decisions.
