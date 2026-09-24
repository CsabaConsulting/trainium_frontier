# G1 drafting checklist analysis

Date: 2026-09-24  
Role: Analyst only  
Basis: corrected G0 comparator r4 and its independent PASS receipt  
Disposition: **G1 is scientifically necessary, but is currently unsafe and
unlaunchable until a distinct request/finalization chain passes independent review**

This report is analysis only. It changes no production or gate source, proposal,
request, authorization, host, or experiment state. It authorizes no host access,
native execution, G2, full A1 training, cleanup, pruning, or submission.

## Current authority boundary

G1 is scientifically necessary but currently unsafe/unlaunchable: the r4 proposal and
runbook still say comparator verification is pending and that only G0 can be finalized;
`requests.pending.json` remains `executable:false`; and no G1 request template,
finalizer, or authorization exists. Independent receipt commit
`8add2585d30b11ac2c50fb2701957657a1a6f30d` permits only drafting a separate
non-executable G1 request.

The accepted r4 and receipt identities are:

- comparator r4 commit:
  `994d302b0036c56aaab63797f0561797bc4cf431`;
- independent PASS receipt commit:
  `8add2585d30b11ac2c50fb2701957657a1a6f30d`;
- receipt path:
  `research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/g0-comparator-r4-independent-verification.json`;
- receipt SHA-256:
  `c4da83d6fe19fddab548ef49cb1a2d2eb69146e37bc36f452bcf37b446fc8c68`.

## Intended proof

G1 is the same hybrid owned-AdamW/Muon plus replicated-RMSProp run as G0, with
seed, data cursor, fresh model/optimizer initialization, source and scientific flags
held identical. The only intended gate difference is detailed observation OFF to ON.

G1 has two purposes:

1. Prove that detailed observation is byte-nonperturbing by requiring exact G0/G1
   equality for every shared record.
2. Capture hybrid b5/t5 `local_pre_avg`, `post_avg_preclip`, and `postclip` gradient
   metadata, clip scalar, table parameter, `exp_avg_sq`, step, bias and RMS-call
   metadata required for the later G1/G2 RMS equivalence comparison.

There is no scientific/model/optimizer change, no public evaluation, and no BPB claim.
G1 by itself cannot prove RMS or integrated hybrid/ordinary equivalence; that requires
the separately authorized G2 arm and final three-arm comparator.

## Supervisor drafting checklist

1. Create a distinct **non-executable** G1 finalization proposal, request template,
   runtime driver, authorization template and adversarial tests. Do not reuse the
   consumed G0 driver, request or authorization. Reconcile the proposal/runbook state
   fields that remain historically pending at r4. A new exact clean request-bearing
   head, committed request-template SHA-256 and authorization SHA-256 must be supplied
   through independent finalization. The authorization may enable G1 only; G2, full
   A1, cleanup, checkpoint pruning, cache deletion and submission must remain false.

2. Pin the accepted comparator and all protected scientific/gate identities exactly:

   - r4 commit:
     `994d302b0036c56aaab63797f0561797bc4cf431`;
   - PASS receipt commit:
     `8add2585d30b11ac2c50fb2701957657a1a6f30d`;
   - PASS receipt SHA-256:
     `c4da83d6fe19fddab548ef49cb1a2d2eb69146e37bc36f452bcf37b446fc8c68`;
   - A1 ownership manifest:
     `8119df41c8a9ecbfc3f9e69067c7e6ac162c2d5d8ba0a15635265304dcf865a2`;
   - r4 comparator:
     `00f1a7263bf7367fb7ab7568822396ff04839b726c9565bb1fd6d0d70ecca17c`;
   - production candidate:
     `9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`;
   - `prepare.py`:
     `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
   - observer:
     `89ad30056bf3fe207c43ea61f8c8a94583232f9f21e5158a80fd37cce42d02e7`;
   - hybrid gate source:
     `908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94`;
   - ordinary gate source:
     `6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0`;
   - model checker:
     `e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e`.

   The distinct G1 request must also pin the exact launcher, guard, host, backup, sync,
   runner, finalizer, request, proposal, runbook, test and authorization bytes at its
   future request-bearing head. Historical stable launcher-family hashes recorded by
   G0 are:

   - `research/fresh_start_launch.py`:
     `872c29d6703be64e3bcfdae4deff99acb459eca804f5a7bb574ac6d788e4a80c`;
   - `research/fresh_start_guard.py`:
     `ac6f7d23dbe998b1d9c8ae14e209c6ffc21fac5d927895b1b92b15dc493f6de6`;
   - `research/fresh_start_host.py`:
     `f5bc5034b4ac77e2af3234a9630ab8e4d0e3f1b24f7404ac22778d30016e6d16`;
   - `research/fresh_start_backup.py`:
     `c978fd20b6620d0d80c25f8982974e8c4b2ad009edd9da376d784f5cecc3ead3`;
   - `research/fresh_start_sync.py`:
     `ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c`;
   - `research/run_experiment.py`:
     `494e283f2464f1d08059eb69423165cbd40ff3fbf9ba9c89f746e0ce45096705`.

   These historical hashes are evidence, not permission to assume the future G1 head
   still contains them; the G1 finalizer and Verifier must hash the actual protected
   files at that head.

3. Pin the authenticated G0 dependency exactly:

   - evidence commit:
     `37aa3644e63e20a1427cfa69b44fd6de7e25cd81`;
   - corrected immutable-G0 replay status: `PASS`;
   - corrected replay result SHA-256:
     `7198d39b9c21f1eb7e47f84e52398ca85d239ad564a36e6d524674d83355ab4a`;
   - corrected replay result size: `326` bytes;
   - G0 Operator receipt SHA-256:
     `4cca77e78ee510a21c693a0ca285c259dab14e0ba69e89b8b63c555cf0b58d14`;
   - runtime result SHA-256:
     `d2d3dcafd5d31a3cc69a030f1db3798ba22aa77f69878a8be00919a803180eff`;
   - runtime job SHA-256:
     `400b339d4cc913a0bacabc8fd7c657cf3795f07bbd56b44ffde48d8a1e8c3d1a`;
   - clean postflight SHA-256:
     `425be4b5357410a3bd8da350ab7862f435490ab706a3ada425ca9a9254c31c6b`;
   - rank-0 report:
     `8aa51461a8cfba1efd1467d97d3ff4788728870b1809b93c2f27c8c8a1054064`;
   - rank-1 report:
     `bb72db43603c324eb6d8b4a54a7a50d58a6186b9763dec661a1681d7dc88cf63`;
   - rank-2 report:
     `db6e73e08a1154f4a62b488578cf36565b9accf951eeb40ebe4259e7ca069f7c`;
   - rank-3 report:
     `897e2fea281755d9ab655670bd56d7d36fae4d90353c98f5e05f29cfb5289aee`;
   - retained checkpoint SHA-256:
     `2d0891c569e6046bd4a3635c62c1ebdb4c6ea6ed0d9788212910a26e61ad3b13`;
   - retained checkpoint size: `739,322,371` bytes.

   Durable consumers must reauthenticate stored or regenerated G0 comparator-result
   bytes against the receipt digest and r4 publication contract. A bare path observed
   later is not authority.

4. Fix the G1 host and resource contract exactly:

   - sole host: `trn21` / `ip-172-31-34-85`;
   - full Trn2 chip through LNC2, world size 4 and four torchrun processes;
   - one run and one private process group on the host; no concurrent run;
   - initially absent remote:
     `/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r1`;
   - initially absent compile cache:
     `/home/ubuntu/neuron_cache_20260924_v2_a1_native5_g1_trn21_r1`;
   - initially absent output:
     `out/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1`;
   - initially absent report directory:
     `research/run-logs/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1/native-gate`;
   - minimum free space on the artifact filesystem: `6,979,321,856` bytes;
   - hard abort below: `5,368,709,120` bytes;
   - outer timeout: `4,200` seconds;
   - child timeout: `3,600` seconds;
   - TERM grace: `60` seconds, followed by bounded KILL/reap/quiescence proof;
   - exact clean origin-aligned tree, idle process inventory and no device holder;
   - no deletion, cache cleanup or checkpoint pruning authority.

5. Use only an independently materialized concrete G1 request. The eventual Operator
   entry point is the standard launcher over that exact concrete request, not raw
   `torchrun`:

   ```text
   python research/fresh_start_launch.py \
     /tmp/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1.request.json
   ```

   No such authorized concrete request exists yet. Its child environment must be
   exactly:

   ```text
   NEURON_LOGICAL_NC_CONFIG=2
   NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16
   NEURON_COMPETITION_R1_SEED=42
   PYTHONUNBUFFERED=1
   NEURON_COMPILE_CACHE_DIR=/home/ubuntu/neuron_cache_20260924_v2_a1_native5_g1_trn21_r1
   ```

   The exact intended child argv is:

   ```text
   /home/ubuntu/workspace/native_venv/bin/torchrun --standalone --nproc_per_node=4 \
     research/submission-validation/20260924-v2-a1-hybrid-native5-equivalence-gate/train_gate_hybrid.py \
     --num-steps 5 --max-train-seconds 1800 \
     --n-embd 1024 --ngram-ve --ngram-ve-layers=-1 \
     --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram \
     --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 \
     --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct \
     --nki-local-conv --bf16-norm-output --pack-factor 32 \
     --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 \
     --out-dir out/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1 \
     --native-gate-arm G1 --native-gate-observation \
     --native-gate-label fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1 \
     --native-gate-host trn21 \
     --native-gate-report-dir research/run-logs/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1/native-gate
   ```

   `--num-steps 5` means exactly five completed optimizer updates, not five
   microbatches. `--max-train-seconds 1800` remains the unchanged in-process cap; it
   does not convert G1 into a full scientific training run.

6. Require all of the following for a valid G1 runtime/evidence PASS:

   - return code zero, no timeout or interruption, exactly five completed optimizer
     updates and complete rank-0 through rank-3 archives;
   - exact source, config, host, seed, world-size, LNC, label, path and environment
     identity;
   - all reported losses, gradients, states, norms and scalars present and finite;
   - exact 66-entry A1 manifest with owner counts
     `[27405059,30540035,30408963,29362179]`, stride `30540160`, and the two BF16
     RMSProp tables excluded from ownership;
   - exactly one RMS call per update and exactly five FP32 owner exchanges per rank,
     each with input `[30540160]` and output `[122160640]` in the expected position;
   - required detailed local-pre-AVG, post-AVG/preclip, postclip, clip, table,
     `exp_avg_sq`, step and bias records for both tables at every update;
   - cross-hybrid-rank post-AVG/table/persistent-model agreement;
   - sealed model-only checkpoint and immutable evidence hashes after process-group
     quiescence;
   - clean postflight with no process survivor or device holder.

   Any crash, timeout, missing or nonfinite gradient, table ownership, missed or
   duplicated RMS call, extra/missing/malformed collective, rank disagreement,
   source/config/path drift, dirty tree, insufficient disk, incomplete artifact or
   surviving process is FAIL. Stop without G2 preparation.

7. Require the distinct scientific G1 non-perturbation verdict before G2. Normalize
   only the declared G0/G1 header fields `arm`, `detailed`, and `label`; then require
   exact equality for every shared `inventory`, `loss`, pre/post `update`,
   `collectives`, and `complete` record on every rank. G1-only gradient and clip
   records are validated for complete exact schema and finiteness, not compared with
   records absent from G0. Any shared-record mismatch is a G1 FAIL and must keep G2
   locked.

## Required package-gap resolution

The accepted r4 comparator CLI supports only `--validate-g0-only` or the complete
G0/G1/G2 comparison. It has no G0/G1-only durable verdict mode. Authenticating only
the structure and lifecycle of the G1 archive would not prove G1's stated
non-perturbation purpose.

The G1 finalization chain must therefore include a separately independently reviewed
G1 evidence adjudication path, or an equivalently durable independent verifier
procedure, before G2 may be drafted. It must reuse the accepted exact comparison
semantics without loosening any r4 ownership, I/O, publication, equality or lifecycle
gate. Any code change to add such a path is gate infrastructure only, but still needs
Implementer/Verifier separation and new exact hashes.

G1 is not redundant: it is the only hybrid observation-ON arm that supplies the table
details needed for the later ordinary-optimizer comparison. It is also not presently
safe to launch: receipt `8add2585` explicitly authorizes drafting only.

