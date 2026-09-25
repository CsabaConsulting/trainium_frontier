# Independent addendum: trn22 A0 r10 post-launch identity

This addendum reviews Operator commit `439e4535` against the earlier independent
HOLD at `14ee9af5`; it does not replace that receipt's separate terminal-result
gate. I did not operate or contact trn22 and did not author the Operator packet.

**Disposition: PASS for bounded post-launch source/argv/live-training identity;
HOLD for scored-result use.**

- The newly archived
  `research/v2/operator/20260925-trn22-a0-r10/train.progress-1341.log`
  independently rehashes to the Operator's stated SHA-256
  `1fb6903490e37ee5b2dfba0de8025874614bf3097920a25e39d839925dbaf57d`.
  Its `ISOLATED_SUBMISSION` line records the isolated training sandbox,
  source SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
  trusted `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
  and the full LNC2/four-rank launch. `REHEARSAL_COMMAND` has the ordered
  science argv, exactly matching the trn22 A0 assignment/preflight and trn21
  A0 flag sequence. Training passed finite step 1100 at ~0.49 s/step, so the
  original absent-live-log gap is closed and this is more than process start.
- The accompanying
  `research/v2/operator/20260925-trn22-a0-r10/LIVE_SNAPSHOT_20260925T1343Z.md`
  is an authenticated, committed **Operator observation**, not a raw `ps` or
  `fuser` transcript. It names the single driver PID 2998248, torchrun PID
  2998266, workers 2998283–2998286, and reports that only those processes
  held `/dev/neuron0` at 13:43 UTC with 57,110,691,840 bytes free. This is
  equivalent bounded process evidence for the post-launch identity check;
  the absence of machine-generated raw `ps`/`fuser` output remains a stated
  limitation and should not be exaggerated into exhaustive proof of all host
  processes.
- The September 24 rehearsal harness previously used for this manual launch
  is pinned by preflight SHA-256 `7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`.
  Its extra historical untracked log files are not part of the isolated
  sandbox and do not, by themselves, show altered science bytes. The new
  live log binds the sandbox train and evaluator hashes needed for this narrow
  launch review.

The job was still training at the live snapshot. **No A0 BPB/control verdict,
checkpoint validity or A2 comparison follows from this addendum.** Before
`decide.py` or a table comparison uses trn22 A0 r10, archive and independently
review the complete exit/log, finite steps and budget, model checkpoint hash,
fresh-process public-20M/context-1024 eval with exact trusted source, clean
postflight, and ledger record. If any of those fails, this bounded post-launch
PASS does not rescue the result.
