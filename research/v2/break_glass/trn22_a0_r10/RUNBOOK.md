# trn22 break-glass A0 manual path — disabled until independent review and assignment

This is the one `fs-20260925-v2-c5c-a0-control-trn22-r10` control selected
by proposal/ratification `1cf9a290`/`6a3aa8e9`, adapted only in its
operational route by Poojith's later `research/v2/BREAK_GLASS_2026-09-25.md`.
It does **not** launch from this repository or authorize its own execution.
The Implementer has not contacted trn22. A different Verifier must review the
package; the Supervisor must issue a new exact, expiring one-label
`SUPERVISOR_ASSIGNMENT.json`; only the Operator may stage/launch. No AWS
action, cleanup, source overwrite, submission, or second active launcher.

## Exact science and existing evidence

- Local source is the clean `22bfbab5` worktree's root `train.py`, SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
  Its `prepare.py` is SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- The old trn22 e65 checkout has the same model and evaluator bytes and the
  proven Sep 24 `submission_rehearsal.py` harness (SHA-256 `7a55cb87…`).
  Only a **new** private `out/` package/work directory is used; no checkout
  source file is replaced.
- The exact four-rank/three-step CPU G2 receipt
  `research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/remote-run/c5c-g2.json`
  hashes to `38dd39a02d86d639f6333228048f8bdb9dcc77735f03c2d45963e1057bf945b9`.
  The independent numeric PASS is in
  `research/agentic/verification/20260925-trn22-c5c-a0-r2-g2-r3-numeric-review.md`.
  This G2 receipt binds the exact c5c bytes, prepare hash, and ordered A0 argv;
  it is CPU-only, not a native result.
- G1's retained-checkpoint evaluation calibrated the unchanged evaluator on
  trn22: context 1024, exactly 20,971,520 public tokens, 83,462,186 bytes.
- Training uses full-chip LNC2/world4, seed 42, 1800 charged seconds, no
  in-process public eval, and the six existing trn22 warm cache directories
  after fresh ownership/space checks. A cold cache is reserved for S0.

## Operator only, after PASS review and Supervisor assignment

Resolve the working SSH credential/config without displaying or copying key
material. Target host is `ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com`;
expected hostname is `ip-172-31-32-253`. The canonical manual source is
`/home/poojith-devan/Desktop/Trainium-trn22-a0-manual/train.py`.
The host checkout is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2`.
The new package directory must be **absent** before one exclusive
`mkdir -m 700`; never reuse an earlier directory or run label. The work path
`out/fs-20260925-v2-c5c-a0-control-trn22-r10-manual-work` must also be absent.

Copy by plain `scp` (or `rsync`) into the new package directory, byte for
byte: root `train.py`; this package's `launch-command.txt`,
`provenance.json`, and `remote_manual_a0.py`; the G2 JSON above as
`G2_PASS.json`; and the separately reviewed Supervisor assignment as
`SUPERVISOR_ASSIGNMENT.json`. Do not send `.env`, keys, or user files.
On the host, run `sha256sum` on `train.py`, `G2_PASS.json`, `prepare.py`, and
the rehearsal harness; compare with the exact digests above before launch.
The runner repeats those checks mechanically.

Run the copied script once with the host native Python:

```
/home/ubuntu/workspace/native_venv/bin/python /home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2/out/fs-20260925-v2-c5c-a0-control-trn22-r10-manual-source-r1/remote_manual_a0.py
```

The Operator should detach it under `nohup`/equivalent so PC connectivity
does not end the run, then verify `train_DRIVER_STARTED` in `driver.log` and
the actual `torchrun` worker process before reporting the first launch.
Do not start another launcher while this host-wide lock is held. The script
will refuse a missing/mismatched G2/assignment, occupied Neuron device,
non-unique label, dirty/wrong host checkout, wrong SHA, or <6.5 GiB root free;
it checks free disk during both phases and preserves all files on failure.

Training and evaluation each use an isolated Sep 24 rehearsal sandbox. After
training exits cleanly, the runner checks at least five step boundaries,
charged budget and checkpoint hash, then runs a **new Python process** for
`prepare.py eval-public --device neuron --batch-size 1 --seq-len 1024
--eval-tokens 20971520 --timeout-seconds 1200`. It writes `train.log`,
`eval.log`, `preflight.json`, `attempt.json`, `quiescence.json`,
`run-record.json`, and `terminal.json` beside the staged package. The
checkpoint remains in the unique train sandbox. The raw `train.log`/`eval.log`
are the error evidence if `terminal.json` says FAILED; report the actual
error text, with any secrets redacted.

Retrieve these files and the rehearsal receipts/step log to the canonical
controller repo. A different Analyst must validate them, append exactly one
`run-record.json` row (success **or failure**) to
`research/v2/ledger.jsonl`, and commit/push evidence before A2 uses this
control. If success, report steps and public-20M BPB; a control has no delta
versus itself and is not a model improvement. Run `workflow/tools/decide.py`
unchanged for later arms. On any failure, the label is consumed and requires
a new proposal/label; do not silently retry or clean up.
