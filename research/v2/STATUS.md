# Workflow installation and incremental launch status

Status: **Phase 0 installed; independent review and all runtime gates pending.**

## Latest human brief (25 September, 02:06 IST)

- [Active brief](CONTROLLER_BRIEF_2026-09-25.md): keep proven launches while piloting the repaired
  new runner on one host; do not gate independent legacy dense work on a wholesale migration.
- [Disk decision packet](DISK_DECISION_PACKET_2026-09-25.md): recommend human expansion of both
  exact root EBS volumes from 200 to 250 GiB. The historical 3.89 GB cleanup opportunity is gone:
  the old controller directory is empty. No deletion/AWS action occurred; LR0.015 checkpoints stay.
- Latest measured root free space: trn21 6.43 GiB, trn22 7.46 GiB. trn21 and table/cold replay work
  remain disk-blocked. trn22's warm dense path still needs its independent runtime gates.
- Independent static model review is integrated at `7150b59b`; the exact four CPU/G2 checks have
  a reviewed contained shell wrapper at `445cc9d0`. CPU checks are not scored training or BPB gains.
- Ready jobs: **zero**, not two per host yet. Proposals must not be counted as checked jobs.
- Use 2M screens only to triage; fresh public20M/context1024 and unchanged decide.py govern results.
  The active adoption threshold is mean delta `<= -0.0010`, not historical `-0.0008`.
- The local autonomous-loop service is inactive; its watchdog timer is active. A running unattended
  Bedrock scheduler has not been verified. Ideation cadence is now a 3–4-hour policy, not a claim
  of deployed scheduling.

- `launch_authorized` is `false`; no controller or launcher is active.
- The seeded queue is an exact copy of `reports/2026-09-24-queue-v2.json` and remains proposal-only.
- `ledger.jsonl` is empty.
- G1 source and paired-host metadata are recorded in `g1.json`; fresh checkpoint calibration remains pending.
- Independent G2 review and all native/Neuron gates remain pending.
- The old `research/agentic/autoloop/selection.py` selection/control loop is historical and inactive.
  Workflow v2 must not be run concurrently with it.
- Workflow-v2 selection requires a fresh-process context-1024 evaluation over 20,971,520 tokens and
  the unmodified `workflow/tools/decide.py` rules.

## Installed identities

| Artifact | SHA-256 |
|---|---|
| `04-train-research-35c9.py` | `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2` |
| `05-train-submission-clean-116188.py` | `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` |
| `submissions/research_v2/train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `submissions/candidates/S0/train.py` | `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` |
| `workflow/tools/decide.py` | `a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66` |
| `research/v2/queue.json` | `1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00` |

The imported CPU receipts and end-to-end transcript are upstream evidence. This installation did not
independently verify them and makes no native-hardware claim. See `UPSTREAM_PROVENANCE.json`.

## Current handoff (2026-09-24 20:27 UTC)

- Phase 0 exact-copy installation is integrated at `8b9fc841`; G1 facts at `d6629b61`.
- Main reran the upstream synthetic E2E successfully, but independent negative tests found real runner bugs:
  repeat failed labels could launch, effective environment/untracked source were not checked, and evaluation
  population was copied from config instead of verified from evaluator JSON. Reviewed repairs are in progress.
- Both hosts have the default five shards, 415 row groups and 424,960 documents. This is not a token count.
- Both retained organizer files match unchanged `prepare.py` SHA `5807c46e...`; retained calibration checkpoints
  exist. Fresh calibration has not run. Historical 20M evaluations took about 12 minutes each.
- Last live inventory: trn21 idle at 6.43 GiB free (below floor); trn22 idle at 7.46 GiB.
- The legacy local service was found auto-restarting and explicitly stopped (inactive/dead, PID 0). Its old
  breaker record was not reset. No instance or running training job was stopped.
- S0 and research-v2 passed independent static review at `0412d21f`; CPU/G2 and native checks are still required.
  The assigned Operator is preparing four baseline/S0 CPU checks, not a scored experiment.
- Best confirmed public20M remains single 0.9918353324, two-run mean 0.9919537735 on the older clean 116188 file.
  Installed S0 `ad223cdc...` is unscored and not ready to upload.
