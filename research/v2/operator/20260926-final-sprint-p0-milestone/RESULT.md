# Both P0 runs are training — 07:22:13 UTC / 12:52:13 IST

- trn21 schedule: latest printed step320, loss3.4668,265,338 tokens/s,
  source remaining1644s. Existing step-log indices run0 through322.
- trn22 batch262144/LR.015: latest step140, loss3.8301,282,596 tokens/s,
  source remaining1673s. Existing step-log indices run0 through140.

All four original workers per host remain live with unchanged start ticks, exact source,
rank/world/LNC2/cache environment, original guardian and held lock. No fatal traceback
or terminal is observed. No compiler descendant was still present in this snapshot.
Free disk is14,638,350,336 bytes ontrn21 and17,014,960,128 bytes ontrn22.

Both first five steps are present and finite. Printed step-duration sums are807.15s
and828.40s, respectively, and both source remaining clocks show1800s atstep4. These
are **not exact launch-to-fifth timestamps**: the existing `steps.jsonl` contains
indices only, and the harness writes its own uncapped observations at terminal.
That final receipt remains authoritative. The current984s total process age is not
984s of startup; both runs have already been training. No step-count/compile alarm
was used to stop a run or launch another.

Estimated training finish:07:49–07:51UTC /13:19–13:21IST. Prior comparable8layer
fresh20M evaluations took699.84s and733.00s, so an11–15minute evaluation allowance
gives approximately08:00–08:06UTC /13:30–13:36IST for scores if no new issue occurs.
These are estimates, not promises. There is no new BPB result yet.

One bounded read-only observation per host; no mutation, signal, launch or cleanup.
Both attached evaluations remain protected. Next useful boundary is training→eval;
do not repeatedly poll unchanged healthy training.
