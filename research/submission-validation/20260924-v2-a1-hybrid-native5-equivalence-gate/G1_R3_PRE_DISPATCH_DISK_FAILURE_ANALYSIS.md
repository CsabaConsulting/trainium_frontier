# G1 r3 pre-dispatch disk-floor failure analysis

Date: 2026-09-25  
Role: Analyst  
Verdict: **FAIL_PRE_DISPATCH_DISK_FLOOR; r3 is consumed; no scientific result**  
Disposition: **STOP. Obtain separate exact cleanup authority, then build and independently verify a fresh r4 package.**

This report is analysis and a drafting checklist only. It authorizes no host access, deletion,
cleanup, request materialization, launch, training, G2, full A1, threshold change, or submission.

## Controlling evidence and authority

- Failure evidence commit:
  `5e45a564bba0d04b052345a7c9c1812ddb71ce51`.
- Failure receipt:
  `operator-evidence/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3-pre-dispatch-disk-failure/OPERATOR_FAILURE_RECEIPT.json`,
  SHA-256 `2325354ee0660fe713a64b23e102c2b53b9c8cab4ca6a5ab081f8850cb322946`.
- Archived concrete request SHA-256:
  `688c6e37fd1753fb3b21f9fea5e77cbb04ed83e92f5a6057ecaba6a7028da0c6`.
- r3 implementation and independent verifier:
  `37071df8069835d363e51883e090ff594e5a6520` and
  `312a3ad689182a512d381a1cacc4d60e1bdc0a81`.
- r3 authorized head and authorization SHA-256:
  `c4fed0c1b307b6c22149f4d42a134ce531879815` and
  `307ed8a59521a22b32f790022062ae436f6543714520f3e8d6a93cf7d8db14c0`.
- Authorization verifier:
  `9ca2bded4b691b710eaf10331247f572cd8e0eca`, receipt SHA-256
  `7082d290f894c7b3cb5a90af1504f13d8f60ac0b2bf3f0abded18cbbd89eae0e`.
- Request verifier:
  `fe1a74996e1dbbe0b7642bc510513760cdc414da`, receipt SHA-256
  `d6acfcde7086b4f832d3aeade236b0d069b3687bc765eaa46c23a389443ccafa`.
- Frozen request-template SHA-256:
  `50b083be810557b4cc222520119006475bc59158659dfa7d63ba90fd4272b43a`.

The root contract requires unique labels, complete proposals, one run per host, full pre-run checks,
durable failure evidence, and role separation. The r3 proposal and runbook freeze an operational
floor of `6,979,321,856` bytes (6.5 GiB), a hard-abort floor of `5,368,709,120` bytes, and the action
`STOP without cleanup` below the operational floor. The executable r3 authorization expressly sets
checkpoint pruning and cache cleanup false. The standing autonomous authorization permits bounded
development runs, but its cleanup expansion excludes unknown worktrees, source, logs, caches and
unretrieved artifacts; it does not override an experiment-specific no-cleanup lock. The standing
mission retains the 6.5-GiB final-week gate. Therefore no existing authority permits deleting the
r3 checkout or bundle.

## What happened

The Operator correctly stopped after staging and before dispatch:

```text
initial free bytes                         7,094,439,936
frozen operational floor                  6,979,321,856
initial headroom above floor                 115,118,080
post-staging free bytes                    6,902,263,808
observed staging debit                       192,176,128
post-staging shortfall                        77,058,048
```

The initial check found an idle host, free workflow lock, no device holder, no training process,
and all r3 paths absent. Staging then created and preserved:

```text
/home/ubuntu/v2-a1-native5-g1-r3.bundle
  SHA-256 e319b4d1455a31dfa84594ff3fab6a6d4a3bf8ad8179f7a39bc2fe7d2504cd4c
  logical bytes 46,547,065

/home/ubuntu/trainium_freshstart_20260925_v2_a1_native5_g1_hybrid_observation_on_trn21_r3
  HEAD c4fed0c1b307b6c22149f4d42a134ce531879815
  clean checkout; strict git-fsck PASS
```

The second disk check failed. The launcher was not invoked, no dispatch receipt was created, the
remote runtime script did not execute, and training did not start. The r3 compile cache, job,
launch/result/postflight logs, output, native-gate reports and checkpoint are all recorded absent.
Post-failure observation again found the lock free, no holders and no exact-label process.

This is an operational capacity failure, not a gate comparison, numerical failure, candidate
failure, or training result. It provides no evidence for or against the A1 scientific hypothesis.

## Why r3 is consumed

The absence of a launcher invocation does not preserve r3 freshness. The independently verified
r3 concrete request was handed to an Operator attempt, and its unique bundle and remote checkout
were created. The r3 contract requires those identities to be initially absent. They are now
durably recorded as existing, preserved artifacts in a terminal failure receipt. Removing them
later cannot erase that history or make the request, authorization, label, branch, bundle, checkout,
cache, output/report names or request inode "first use" again.

Consequently:

- do not rematerialize, redispatch, amend in place, or retry the r3 request;
- do not reuse r3 label-derived remote, cache, output, report, log, job, result or dispatch paths;
- do not treat the r3 authorization or request verifier as authority for a descendant or r4;
- preserve the r3 receipt and concrete request as historical failure evidence.

The special r2 rematerialization ruling does not apply. That ruling concerned a lost local
serialization before any remote attempt artifact existed. Here remote staging occurred and a
terminal Operator failure receipt exists.

## Capacity math for cleanup planning

The Supervisor supplied a later scheduling observation of `6,902,231,040` free bytes. This report
does not independently verify that live value. Against the unchanged floor it is short by
`77,090,816` bytes.

The already archived r1 checkout+bundle pair is reported to recover `191,806,441` bytes. Removing
only that pair would project `7,094,037,481` pre-staging bytes, but a same-size r4 staging operation
would return the host to approximately `6,902,231,040`, still `77,090,816` below the floor. **r1
cleanup alone cannot make r4 safe.**

The r3 receipt observed a `192,176,128`-byte staging debit. Treat this as a capacity observation,
not as an exact deletion-size receipt: exact allocated bytes for each target still require a fresh
read-only inventory. The no-margin minimum before r4 staging is therefore:

```text
6,979,321,856 floor + 192,176,128 observed staging reserve
= 7,171,497,984 bytes
```

A safe plan must also preserve positive headroom. A fully evidence-derived target is to retain the
same `115,118,080`-byte cushion that r3 had before staging:

```text
6,979,321,856 floor
+ 192,176,128 maximum observed G1 staging debit
+ 115,118,080 observed initial r3 cushion
= 7,286,616,064 bytes required before r4 staging
```

Using the supplied later free-space value, the reported r1 recovery, and the observed r3 staging
debit gives a projection of `7,286,213,609` bytes before r4 staging. That is `402,455` bytes below
the evidence-derived target. Exact live allocated-byte accounting may change this projection, but
it must be measured before authority is drafted; it must not be rounded in a favorable direction.

Therefore the cleanup authority should cover enough **exact, independently inventoried, consumed
staging targets** to prove at least `7,286,616,064` free bytes after cleanup. The preferred order is:

1. consumed r1 staging pair:
   - `/home/ubuntu/v2-a1-native5-g1-head-149cc912-trn21-r1.bundle`;
   - `/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r1`;
2. consumed r3 staging pair:
   - `/home/ubuntu/v2-a1-native5-g1-r3.bundle`;
   - `/home/ubuntu/trainium_freshstart_20260925_v2_a1_native5_g1_hybrid_observation_on_trn21_r3`;
3. only if the exact post-cleanup projection still misses the target, another separately proven
   consumed controller pair. The r2 pair is a possible but more sensitive candidate:
   - `/home/ubuntu/v2-a1-native5-g1-r2.bundle`;
   - `/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r2`.

R2 reached the runtime and its checkout held failure artifacts. It is eligible only after an
independent verifier proves commit `5b392db4bc8c01a47da2f5969cb33df0c50ec52b` contains a complete,
hash-exact local archive of every required r2 runtime artifact and a fresh inventory finds no
unarchived file or process reference. The separate r2 compile cache is not part of the two paths
above and must remain untouched unless it is itself explicitly inventoried and separately
authorized. Never delete shared caches, data, checkpoints, logs, current worktrees, r0/G0 evidence,
unknown paths, or use a wildcard/broad cleanup command.

If the exact authorized deletions do not produce the frozen pre-staging free-space target, stop.
Do not lower `6,979,321,856`, reinterpret decimal/GiB units, count logical rather than allocated
recovery, or assume that r4 staging will be smaller. The repository-wide protocol also contains a
general 8-GiB preflight statement while the human-approved final-week mission and this exact gate
freeze 6.5 GiB. R4 finalization should explicitly record the controlling final-week override; if a
Supervisor instead applies the higher general threshold, it must be satisfied, not waived.

## Separate cleanup-only authorization and evidence gates

Cleanup must be its own non-training operation and must complete before any r4 request exists.
Minimum chain:

1. **Read-only inventory.** An Operator records current free bytes/inodes, lock state, holders,
   relevant processes and, for each proposed target, exact absolute path, parent, device/inode,
   UID/GID, type, link/symlink state, apparent and allocated bytes, process references, Git HEAD/
   cleanliness where applicable, and file hashes for bundles. Inventory performs no deletion.
2. **Archive completeness proof.** An independent Verifier binds r1/r3 failure evidence and, if
   considered, r2 evidence; proves every target is controller-created and consumed; proves no
   checkpoint, unique log, report, result, source change or other unretrieved artifact would be
   lost; and computes an exact conservative post-cleanup free-space projection using allocated
   bytes.
3. **Explicit target authorization.** A separate human/Supervisor state document lists only the
   exact approved absolute targets and their authenticated identities. It grants deletion of those
   targets only, sets training/evaluation/request materialization/AWS lifecycle/cache or checkpoint
   pruning/submission false, requires the host idle and unheld, and requires the post-cleanup free
   space target above. Standing authority alone is insufficient for these worktrees.
4. **Independent authorization verification.** A different Verifier checks the exact target set,
   identities, archive hashes, capacity calculation, no-follow/no-wildcard deletion mechanism,
   downstream false authorities, and fail-closed conditions.
5. **Cleanup Operator receipt.** Under the verified authority, an Operator reauthenticates the
   target identities immediately before deletion, removes no other name, records exact allocated
   bytes recovered and before/after free bytes/inodes, proves target absence and preservation of
   excluded paths, and records idle/lock/holder state. The receipt and any stdout/stderr are pushed.
6. **Independent post-cleanup verification.** A Verifier authenticates the cleanup receipt and
   confirms the achieved free-space value is at least `7,286,616,064` before r4 staging is allowed.

No cleanup step may be combined with r4 staging or launch. A failed cleanup capacity gate stops the
chain and does not consume an r4 identity that has not yet been created.

## Fresh r4 package

Recommended exact identities:

```text
label:          fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4
branch:         experiment/20260925-v2-a1-hybrid-native5-g1-r4-finalization
bundle:         /home/ubuntu/v2-a1-native5-g1-r4.bundle
remote:         /home/ubuntu/trainium_freshstart_20260925_v2_a1_native5_g1_hybrid_observation_on_trn21_r4
compile cache:  /home/ubuntu/neuron_cache_20260925_v2_a1_native5_g1_trn21_r4
output:         out/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4
report:         research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4/native-gate
child log:      research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4.child.log
runtime job:    research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4.job.json
runtime result: research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4.runtime-result.json
dispatch:       research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4.dispatch.json
request:        /tmp/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r4.request.json
```

R4 requires a new disk-retry amendment, new pending and executable r4 authorization paths, new
request bytes/inode, and new implementation/finalization/request-verifier receipts. R3 authority
cannot be copied forward.

The only permissible gate-code change is the exact G1 observer allowlist literal from r3 to r4,
plus identity/schema/provenance plumbing. Because the ownership manifest authenticates the observer,
the observer hash, its manifest pin and the comparator's manifest-hash pin must be mechanically
rebound and independently reviewed. This must not be described as byte-identical gate code.

All scientific and numerical bytes/criteria remain fixed:

- production candidate SHA-256
  `9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`;
- `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
- hybrid and ordinary gate sources
  `908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94` and
  `6ed69d2356c28e0dd8f88f494d988c0b87f7f86426a001945bfd958ed6f8f6d0`;
- checkpoint verifier
  `e1ead481f9786ec02a3c60d88e72820f084274200bccd1f7eeb16da108cb6e2e`;
- repaired launcher
  `7620fea10bb51c54481b8347ec9e97ebd40ce8f5ee206dc3f203c9483c35ee2f`;
- host `trn21`, seed 42, world size 4, LNC2, five completed optimizer updates, exact flags,
  candidate/model/optimizer/data arithmetic, exact-byte equality criteria, checkpoint/causality
  gates, the 6.5-GiB operational floor and approximately-5-GiB hard floor.

No tolerance, exactness condition, record count, decision outcome, or downstream lock may be
relaxed to make r4 pass.

## Required r4 independent gates

After verified cleanup, and before any r4 host action:

1. Implementer records r3 as consumed and prepares only the exact r4 identity/plumbing change.
2. Independent implementation Verifier proves the fixed science hashes above, exact r3-to-r4
   label-only observer diff, mechanical manifest/comparator pin rebound with no comparator logic or
   ownership-entry change, and an exact G0 replay under the rebound pins.
3. The exact request-template argv/environment must construct the real
   `NativeGateObserver` for ranks 0--3 under CPU; r4 must pass, while r1/r2/r3 and a wrong host must
   fail. The production wrapper's forwarding must remain exact.
4. Verifier proves every r4 bundle/remote/cache/output/report/log/job/result/dispatch/request path is
   distinct from r1/r2/r3 and initially absent, and that r1/r2/r3 authorities reject.
5. Full gate/G1/ownership tests, Python compilation, JSON parsing, hash maps, G0 corrected replay,
   secret-marker review and diff checks pass. Threshold literals must be unchanged.
6. A new state-only r4 authorization is independently verified. Only then may one O_EXCL concrete
   request be materialized and independently authenticated by bytes, hash, inode and canonical
   reconstruction.
7. Immediately before staging, Operator rechecks idle/lock/holders and proves free bytes at least
   `7,286,616,064`. After staging, and before launcher invocation, Operator again checks the exact
   filesystem and requires at least `6,979,321,856`. Any failure consumes r4, is archived, and
   stops without cleanup or retry.
8. Only a separately authorized one-time r4 execution can generate G1 evidence. It must preserve
   all existing lifecycle, process-group, artifact retrieval, checkpoint and postflight gates.
   A later Analyst may adjudicate it; a successful process alone is not a PASS.

G2, full A1 training, cleanup beyond the exact cleanup receipt, submission and leaderboard actions
remain locked throughout.

## Final recommendation

Archive r3 as a consumed operational failure. Do not call it a science result and do not repeat it.
First obtain and independently verify exact cleanup-only authority for enough consumed staging
artifacts to meet the evidence-derived pre-staging target with margin. After a separately verified
cleanup receipt, prepare a new r4 identity/package with unchanged science and unchanged thresholds,
then repeat the full implementation, authorization, request and two-stage disk gates. No training
advancement is presently authorized.
