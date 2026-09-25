# trn21 break-glass Operator: S0 failure and A0 launch

Observation date: 2026-09-25 UTC. This is an Operator record, not a validation
verdict or submission authorization. The human's break-glass instruction allowed
the one-Operator base-control launch before independent receipt review.

## S0 cold replay r1 (failed; no retry)

- Label `fs-20260925-s0-cold-replay-trn21-r1`; host directory
  `/home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1`.
- Exact `train.py` SHA-256
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`;
  bare four-rank launch recorded in `launch.json` at 13:17:22.389804 UTC.
- Four rank processes were observed at 13:17:41 UTC. Rank 3 then failed at
  13:17:43 UTC, before any training step, with `nki_kernel_global` Neuron
  compilation: failure reading the newly compiled `module.neff` in the S0
  private cold debug directory. Torchrun returned exit code 1; no checkpoint
  was created. This label is consumed and will not be relaunched.
- Retrieved immutable artifacts: `launch.json` SHA-256
  `5a463f714b70025aafa6c49c426bc728b4ed008b5a52d83447cc256929ff11e8`,
  `training.log` SHA-256
  `19baf980a82d6c560513f07119fb6216e073aabeaa30e23e7db4b730c1913929`,
  and `run-exit-code.txt` SHA-256
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.
- Separate post-crash host snapshot at 13:24:45 UTC has no training process,
  Neuron holder, or host-lock holder; 57,824,866,304 bytes free. Snapshot
  SHA-256 `b0c7ed6d2bf92e4083f4ff63cc3f089894b728f20d5419b6b4fd73a4aa1f7bd4`.

## A0 c5c control r1 (launched; result pending)

- Supervisor break-glass assignment `research/v2/assignments/20260925-trn21-a0-breakglass-r1.json`
  at controller commit `b18fb3586494c21f10008a9c63fd2056d1e6c00a`;
  ratified proposal and independent exact c5c CPU G2 cited there.
- Label `fs-20260925-v2-c5c-a0-control-trn21-r1`; unique host directory
  `/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1`.
- Source `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  trusted `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Only `train.py` was copied into the unique directory; host rehash passed.
- Fresh preflight at 13:30:16 UTC and poststage check at 13:30:56 UTC passed:
  exact source, host identity, clean e65 dependency checkout, disk > 10 GiB,
  private destination, seed 42, five train shards plus public val shard,
  known warm-cache inode pins, no STOP, training process, Neuron process,
  device holder, or lock holder.
- Detached launch began 13:31:07.744399 UTC (19:01:07 IST). `host-launch.json`
  SHA-256 `f97a553b32d53da3526e850040b5c31ff004f1bbb98d8f3698de5de0337b19b4`
  records the exact `flock` + 3600-second outer timeout + full-chip
  `torchrun --standalone --nproc_per_node=4 train.py` command, the one-factor
  A0 flags, 1800-second train budget, optlevel-1 compiler flags, and pinned
  warm cache. No extra science flag was added.
- Independent postlaunch snapshot at 13:31:21 UTC showed four rank PIDs
  2866521–2866524 on Neuron, torchrun PID 2866506, and the device lock held;
  57,824,366,592 bytes free. Snapshot SHA-256
  `37ca0fc03f304cdd850f216e1efa66c9e84da87805a446607ddfa265e681a478`.
- A0 training remains in progress at this report. The fresh-process 20M-token
  public evaluation and full artifact retrieval are pending, and no BPB claim
  is made. Independent postlaunch receipt review is required before result use.

The initial A0 local preflight transport had a JSON-stdin construction error
before a host write. Its failed local receipt is retained. It did not create
the destination or consume the A0 run label; the corrected preflight and
stage/launch receipts are retained in this directory. The two earlier S0
launcher-construction errors were likewise local parser errors before SSH and
before the single S0 native launch, per Supervisor disposition.
