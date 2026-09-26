# Clean full cold replay — actual terminal and postflight

Protected label: `fs-20260926-confirmed36-clean-coldfull-trn22-r2`.
This is Operator retrieval, not independent scientific acceptance or submission approval.

- Training: exit 0; 3,628 steps. Actual uncapped fifth boundary
  859.4735238229623 s; process wall through exit 2,649.2421065529343 s;
  charged 1,789.768582729972 s, within the 1,800-second budget.
- Attached fresh-cache public-20M/context-1024/batch-1 evaluation: exit 0;
  **0.9883890652263234 BPB** from 57,179,868 nats / 83,462,186 bytes /
  20,971,520 tokens. Evaluator time 720.3386545890244 s, timeout false.
  Evaluation ended 05:51:11.000739 UTC; outer dispatcher exited 0 at
  05:51:11.553877 UTC on 26 September 2026.
- Checkpoint remained 453,044,626 bytes with SHA-256
  `4d024cce5120189010d0017fcbf6aaac22a064506eaeb8ca98b453972997d2f2`
  before evaluation, after guarded postflight, and in a fresh stable read after
  terminal. No checkpoint bytes were transferred.
- The bounded final archive contains all 43 noncache regular artifacts
  (1,062,196 bytes), including all four closed graph-metric CSVs and the complete
  guardian lifetime log. Hidden files were included by enumeration; none were
  present. Only the two cache subtrees and the retained checkpoint were skipped.
  Every captured file was locally byte-counted and rehashed.
- Explicit postflight at 05:58:18.149745 UTC: all **702 actually observed**
  identities absent; 252-row full process census has no related lineage or
  live job/compiler rows; Neuron device idle and fuser empty. Original
  device66305/inode832589 was free and accepted a nonblocking lock probe.
  Guardian also recorded `waitpid_ECHILD_AND_POSTFLIGHT_PASS` before release.
  Free disk: 17,294,692,352 bytes.

Raw bundle SHA-256:
`20e1e5f920dff44a4d0a047790e02397ed916945e975a6822d6385efe2f9151d`.
Explicit postflight SHA-256:
`ebbbf6ce4732cc01ed6ead64787239d1208d5da3fa8db120f8ae82d0576da418`.

No successor was launched, no host files were changed, and no caches or
checkpoints were deleted by this task. Collector/retriever are consumed.
