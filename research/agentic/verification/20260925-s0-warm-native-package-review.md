# Independent review: pending warm S0 native package

Package branch `experiment/20260925-s0-warm-native-trn22-r1`, request commit
`008e9bd3101c8301e5bf2af4ea10511cf8a3b566`, immutable source parent
`c9bc9517166cd38cba23764af2756aab3f53d5a4`.

Verdict: **static package PASS; dispatch HOLD on disk and live-cache checks**. This is not a
Neuron result, cold replay, submission-ready file, or authorization to launch.

I read the proposal, request, runbook and local static test and independently reran
`python3 research/submission-validation/20260925-s0-warm-native-trn22-r1/test_static.py`
with PASS. Root `train.py` SHA is the exact S0 `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`;
unchanged `prepare.py` SHA is `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The external pending request SHA is `e7763e2d30ee22ede8ba6ab9e1bc25320ec36e7edb32cb78756a3fd3e81fe902`;
its science command is bare four-rank `torchrun ... train.py`, with no flags after `train.py`.
The request leaves `dispatch_authorized=false`, forbids pruning/cold-cache substitution,
and correctly separates a warm native check from the two later required cold replays.

The package's conservative predispatch floor is **8,833,482,752 bytes**: 6.5 GiB operational
floor plus the last observed dense footprint of 1,854,160,896 bytes. trn22's measured free
space before A0 was only about 7,950,000,000 bytes, so S0 cannot be dispatched now under this
request. The six historical 116188 warm-cache paths were not reauthenticated live for this
package. After the human EBS expansion, an Operator must perform the exact live cache/disk,
source, one-job, lock, label, observer and pre/postflight gates before any separate
request-bound authorization. A0 and S0 cannot overlap on trn22.

The historical 116188 public20M score is useful for CLI calibration but its training run
failed the disk gate; it is not a valid S0 advancement control. Even a successful warm S0
run would still require cold bare-command replays, loader/causality checks and reviewer audit
before asking the human to submit.
