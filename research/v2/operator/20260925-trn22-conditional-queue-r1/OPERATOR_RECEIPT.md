# trn22 conditional queue: empty waiting bootstrap

Installed by a distinct Operator after the newest resume/table correction and
independent queue PASS at controller `aaf9b0aa`. This is an operational waiting
service, **not a training launch, six ready jobs, or a table-quality verdict**.

## Deployment

- Started **2026-09-25 15:55:44 UTC / 21:25:44 IST**, detached queue PID
  **3050509**. At 15:56:08 UTC it remained alive with PPID 1. Its first event
  is `waiting_resolution`. It has no device-lock FD while waiting.
- Host `trn22`, hostname `ip-172-31-32-253`.
- Exact independently reviewed runner SHA-256
  `0ce40ad8aea85aa556b3b87e4753c25049751024a137c69947b8e688e482b50d`.
- Bootstrap SHA-256
  `092e0ba53863fe02758b5a4dc4c99423909d1c92438d99264d1b3c669eed781b`.
- New private host root:
  `/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1`.
- Manifest:
  `research/v2/operator/20260925-trn22-conditional-queue-r1/bootstrap.json`
  under that root. Run spool: `runs/`. Controller inbox:
  `next-resolution.json` under that root, currently absent.
- `armed: true`, `jobs: []`. No old flag-bank job is included or can fall
  through. The controller must publish a hash-pinned resolved manifest.
- Runner, exact c5c source and bootstrap were copied normally into the new
  directory and their SHA-256 values verified on the host. No model, SDK,
  evaluator, timing constant or existing cache contents were modified.
- Deployment preflight found no existing queue PID, no local/global/trn22
  breaker or STOP, and no STOP at any of 271 discovered applicable host paths.
  Free space was 51,519,066,112 bytes before deployment and 51,518,656,512
  bytes at launch, above the unchanged 6.5-GiB floor.
- Native Python and torchrun are under
  `/home/ubuntu/workspace/native_venv/bin`. Trusted `prepare.py` remains in
  `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1`.
- Warm cache roots are that same checkout's
  `out/fs-20260924-v2-35c9-lr015-control-trn22-r1-work/caches/train` and
  `/eval`, each with the existing six required subdirectories. This establishes
  configured reuse, not proof of an actual future compiler cache hit.
- Common device lock remains
  `/home/ubuntu/.cache/trainium_freshstart/device0.lock`. The waiting service
  holds no lock; the current A2 driver retains its train/eval sequencing.

The full bootstrap, post-copy hashes, preflight and process/event receipt are
archived beside this document. `run_root/PAUSE_NEXT` pauses between jobs.
Only the authorized controller/Operator writes the resolution inbox. New
training still checks exact sources/proposals, STOP, disk and device idleness.

## Existing A2 result: startup truncated, not a scientific negative

The existing `fs-20260925-v2-c5c-a2-m64d128-trn22-r1-full` was not altered.
Its training returned zero at 15:46:11 UTC but reached only five rank-0
boundaries. First two step durations were **1056.75 and 903.47 seconds**;
steps 2–4 took about 0.62–0.63 seconds (209,363–211,084 tokens/s). Startup
allowance hit 900.0 seconds. Source summary reports training 1116.4 seconds,
charged 1117.2 seconds, `over_budget: False`, five steps and 0.7M tokens.
The harness measured total process wall 1997.2943 seconds and charged process
exit 1097.2943 seconds after the 900-second allowance.

The source's unchanged rolling maximum lookahead includes the compilation-
heavy steps, explaining its early stop. It is a **startup-truncated attempt**,
not a completed 1800-second quality trial; its later BPB cannot establish
that A2 lost or select the no-table branch.

Fresh-process public-20M/context-1024 evaluation PID **3048593** was still
running at 15:56:08 UTC. At artifact retrieval 15:56:29 UTC no evaluation
exit code or BPB existed. Parent train-to-eval driver **3043794** remained
alive. The saved checkpoint stays on the host (1,008,809,893 bytes, SHA-256
`4e04be090e7f0655abca554e391f95ba25a825b097cd6f3b2ff7e48b833c121a`).
No checkpoint was downloaded or deleted.

Exact terminal training log, harness result, five step boundaries, train exit
code and explicitly non-final driver/eval log snapshots are archived in
`a2-startup-truncated/`. Their retrieval manifest has host paths, byte lengths,
SHA-256 and retrieval time. Evaluation remains pending, so no quality verdict
or canonical result-ledger insertion is made by this Operator.
