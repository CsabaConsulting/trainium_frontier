# Clean e1e2 cold-r2 observed startup result

The corrected unique-root diagnostic completed naturally. This is an observed
startup result pending independent numeric acceptance, not a BPB/full-run,
submission or lifetime-launcher safety approval. The separate full launcher
review is REFUSE; no successor was staged or launched.

- Actual five observed rank0 boundaries: 855.5032516639913,
  857.6618753699586, 858.1637012680294, 858.6654968990479,
  859.1679093550192 seconds from harness child clock.
- Uncapped fifth 859.1679093550192s: below870 target and900 hard ceiling.
- Actual process wall863.638870780007s; official charged-through-exit
  4.470961424987763s; charged-budget receipt true, source over_budgetFalse.
- Five finite printed losses9.0101,8.7779,8.4047,8.0711,7.7681.
- Four actual rank0..3/world4/LNC2 worker identities and effective six private
  caches were captured at04:14:41; torchrun/harness/driver/outer all exited0.
  Aggregate torchrun success is the rank-exit proof; individual unobserved
  wait timestamps are not fabricated.
- Terminal04:28:48.701880UTC; outer exit04:28:48.707320UTC.
- Retained checkpoint453044626B, SHA256
  8e867c7a541f799e96502001e91ea06388e46fe0c2159a7c07c684d947b9dbde;
  full post-terminal rehash stable. No checkpoint download/deletion.

At04:30:16.865633, explicit postflight checks every observed dispatcher,
flock, driver, harness, torchrun, worker and compiler identity plus complete
PID/PPID/PGID/SID census. All29 observed identities are absent;255-row census
has no related or busy job/compiler, neuron-ls and fuser are idle, and the
original66305:832589 lock is free. Free disk18772877312B. These actual checks,
not parent exit or lock availability alone, establish current quiescence.

Complete bundle:26 files/156573B, including allfour closed graph_stats CSVs.
SHA256 c03997f87f4bb95f539b0e4ac10bfeff012e1ac0dc12b3ead27b3dcc25aa9c87.
Failed r1 and all sources/checkpoints/caches are preserved. No cleanup,
source/harness/prepare changes, retry, full/eval, AWS or submission action.

trn21 remains protected: latest authorized observation04:28:39 showed step2480,
loss2.7984,268251tok/s,582s source-clock remaining and original596823 held.
