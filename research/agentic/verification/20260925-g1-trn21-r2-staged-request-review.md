# Independent review: trn21 G1 r2 staged calibration

Reviewed: 2026-09-24 23:08 UTC. Operator stage commit:
`7a3d0b0fe98986f8490313a904c0173ad83319e5`.

**Verdict: PASS for the staged source and request to proceed to a separate
Supervisor one-label evaluator-authorization decision. Evaluator dispatch is
still HOLD.** This review does not authorize a launch, establish trn21 G1 PASS,
produce a new BPB result, or permit A0 training. I neither edited nor ran
`prepare.py`, evaluated a checkpoint, trained, cleaned a host, reset a breaker,
or changed AWS resources. Host probes below were read-only.

## Bound inputs

| Input | SHA-256 |
| --- | --- |
| `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/request.pending.json` | `9804d3f76ac8b7495c919539e94d7022609162fc0d9dfc31377e46fb060a691d` |
| `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/RUNBOOK_PENDING.md` | `48d9b2e39e25e0281d3178a4a87733a421f9f13744e8bf9cfd7c7f9c9566be67` |
| Independent pending-package review | `c9a60d8d762bc1ae7f30bc370a2441513b0046d6442e136d4eea16c98c4fd5e9` |
| `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/operator-stage-receipt.json` | `ca417a419557c4891236716cb6d844c607dd1de22d9e1a8c4187a42a63015d52` |
| `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/request.staged.json` | `aba839eb87b041fbb266e2db3795d49f29afcf76c91ecd5eee96a5b3f978dd2a` |

The stage commit adds only the receipt and staged request; `git diff --check`
passes. A direct pending-versus-staged diff shows the evaluator argv and
science-effective environment unchanged. The staged document fills in the
exact commit, source/checkpoint/disk facts and broader unset-name gates; it
retains `launch_authorized=false` and
`STAGED_AWAITING_INDEPENDENT_REVIEW_NOT_DISPATCHABLE`.

## Independent live checks

Read-only SSH used the pinned trn21 endpoint, BatchMode, IdentitiesOnly, strict
host-key checking and the existing key *by path*. No key contents, credential
values, or `.env` files were read or output. At 23:06:16–23:07:06 UTC:

- Hostname was `ip-172-31-34-85`; effective UID 1000/user `ubuntu`/HOME
  `/home/ubuntu`. Native Python was executable. Root had **58,685,087,744
  bytes** free at the main audit, versus the unchanged required
  **9,126,805,504-byte** post-stage evaluator gate. This is a snapshot, not
  authority to skip the fresh pre-dispatch disk check.
- The retained source repository remained at trained commit
  `e7f11c38dc3b5e68b373322e10cfef580f1e8d36`. Its required tracked
  `prepare.py` and nested `train.py` paths had no index or worktree diff;
  their committed-blob and worktree SHA-256 values matched
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`
  and
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
  Source `git fsck` passed, with no object alternates, non-sample hooks, or
  suspicious local Git-config *names*. Its six other untracked run-output
  entries were still present; this source checkout must not be evaluated or
  copied in place. The source remote resolved to a plain existing local path
  without URL user-info/query. A narrow tracked-filename scan found no
  `.env`, PEM/private-key, or credential-file shapes; filename screening
  alone cannot guarantee arbitrary content is secret-free.
- The new staged checkout independently resolved to that same trained
  commit; it is shallow, has exactly one reachable commit, is clean including
  untracked paths, has **zero remotes**, no object alternates, no non-sample
  hooks or suspicious local Git-config names, and passed `git fsck`. Its
  nested `train.py` and trusted `prepare.py` are regular files (not symlinks)
  with exactly the two hashes above. These checks support the Operator's
  committed-object `file://` clone claim without relying on the dirty source
  worktree.
- Retained `final.pt` was a regular, unheld **469,831,009-byte** file with
  SHA-256
  `5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`.
  The six specified historical eval-cache directories (`neff`, `local`,
  `hlo`, `inductor`, `compiler`, `debug`) existed. Fixed tokenizer-cache
  hashes were `aacd6c6a838eaec2c662a7d0cba0b5412f930a6ddee95b9ecb2c8fac53f3dafc`
  for `tokenizer.pkl` and
  `3de03cf6f8e16a9397fecd0cebd04f435f11994b215a5c785787fcc15adb0e15`
  for `token_bytes.pt`. The public validation file was checked by identity
  and 91,699,537-byte size only; no validation contents were read.
- A names-only effective-environment scan found no inherited rank,
  distributed, `ROLE_*`, `MASTER_*`, `TORCHELASTIC_*`, `PYTHONHOME`,
  `NEURON_RT_*`, or `NEURON_COMPETITION_R1_*` overrides. No matching
  train/eval/G2/launcher PID was found. `fuser -s` returned no holder for the
  checkpoint, shared device lock, or the one present `/dev/neuron0` device.
  A full `/home/ubuntu` same-filesystem filename scan found **zero** r2-label
  artifacts; the local run/dispatch/output directories likewise had none.
  Workflow-v2 global and trn21 breakers were closed in the controller state,
  while `launch_authorized` and trn21 `launch_allowed` remained false.

## Exact evaluator contract and remaining hold

The unchanged trusted `prepare.py` at the verified SHA accepts precisely
`eval-public --train-py <nested-116188> --checkpoint <retained-final.pt>
--device neuron --batch-size 1 --seq-len 1024 --eval-tokens 20971520
--timeout-seconds 1200`. The staged command wraps that one fresh Python
process in a 1,800-second outer timeout and uses the historical
`--optlevel=1 --auto-cast matmult --auto-cast-type bf16` compiler flags
and six warm cache paths. The archived reference contains 20,971,520
tokens, 83,462,186 bytes and 57,379,240 nats; independent arithmetic gives
`0.9918353324459732` BPB. This is an evaluator calibration target, not a
fresh measurement.

The staged request says `continuous_floor_monitor_required_on_evaluator=true`,
and its runbook requires a continuous 6.5-GiB floor monitor. **No monitor is
running yet, and this request contains no executable launch authorization.**
Before any evaluator dispatch, the Supervisor must bind an exact one-label
guard/monitor implementation, reauthenticate the source/checkpoint/cache
hashes, complete fresh host/disk/holder/label and full inherited-env-name
preflight, and ensure a stop/quiesce plus two stable postflight checks.
Any attempted r2 evaluator launch consumes r2. Only a successful guarded
evaluation followed by a separate independent result review may mark
trn21 G1 PASS. Do not append the calibration to the model-result ledger,
start A0, prune the checkpoint, or submit on this receipt.
