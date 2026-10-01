# R207 final ranked table — Oct1 04:45 IST

Original FULL and own-checkpoint receipt validators replayed. This is an additive factual reconciliation, not a fresh independent package review. Frozen audit copies are unchanged.

| Rank / candidate | Public20M BPB | Exact-source FULLs | Own-CP LNC1 = unset (public2M) | Existing audit / acceptance |
| --- | --- | --- | --- | --- |
| 1. T3 seed42 | 0.96433779426305065 | 2 | 0.9590707374259858 | [audit copy](/home/poojith-devan/Downloads/Trainium-STATE-LIVE/audit-t3); R191 UPLOAD_READY |
| 2. T3 seed43 | 0.9649042783697576 | 1 | 0.9597143969855484 | [audit copy](/home/poojith-devan/Downloads/Trainium-STATE-LIVE/audit-t3-seed43); Claude package verdict NULL; not upload-accepted |
| 3. T3 seed45 | 0.9656747327089538 | 1 | 0.9607595726122182 | [audit copy](/home/poojith-devan/Downloads/Trainium-STATE-LIVE/audit-t3-seed45); Claude package verdict NULL; not upload-accepted |
| 4. T3 seed46 | 0.9667283945302776 | 1 | 0.9613701580783315 | [audit copy](/home/poojith-devan/Downloads/Trainium-STATE-LIVE/audit-t3-seed46); Claude package verdict NULL; not upload-accepted |
| — SEQ512 | CLOSED, no scored run | 0 | Not run | Cold1043.064s exceeds860s; no fallback underR207 |

Seed42 is the mean of two same-source, same-seed FULL receipts, not a two-seed mean. Its diagnostics bind only the second checkpoint, not both checkpoints. Seeds43/45/46 have one FULL each. R207 reports official seed42 BPB0.9647; this report does not independently authenticate platform evidence or change the frozen READINESS official-score NULL.

Seed45/46 now have complete authentic own-checkpoint pairs in canonical operator receipts. Those receipts are referenced here, not appended to or substituted into their frozen audit packages. Their historical AUDIT_BRIEF pending text is not a new verdict.

## FULL receipts

| Seed / host | Public20M BPB | Cold seconds | Charged seconds | Steps | Canonical FULL archive SHA256 |
| --- | --- | --- | --- | --- | --- |
| 42 / trn21 | 0.964447246764078 | 606.3875389539171 | 1793.3390622320585 | 3017 | `c1f0e66262d5691418e31a4718a67d958e6cd6e3e62acb5fcb89da48f3c67563` |
| 42 / trn22 | 0.9642283417620233 | 598.5196062619798 | 1793.7412270181812 | 3026 | `5cba9a846b346d24a173be29f3c45e015477c557c495edd22bd8c1e00032029d` |
| 43 / trn23 | 0.9649042783697576 | 596.9715136619998 | 1793.5737613750007 | 3024 | `18b70442269330c4b8727a1e63df269220c3f85a8e46c8ba1e412826e73b568e` |
| 45 / trn23 | 0.9656747327089538 | 601.8523181790006 | 1793.3114666969996 | 3019 | `3ef9f0941f8c576a3bddb60df414e4a927ceabb3de1d18fe9cd61ea39345c74d` |
| 46 / trn24 | 0.9667283945302776 | 591.4989819570001 | 1793.5043061659999 | 3017 | `5ff4546ccb885caa09d8b118fb6f1637775bc162aff27fade4149c53324653fb` |

All five listed FULL receipts authenticate fresh public20971520 tokens/context1024, six empty caches, stable checkpoint and original Guardian release. Source, packet, checkpoint, log, raw-envelope and audit evidence hashes/paths are retained in [EVIDENCE.json](EVIDENCE.json).

## Exact source and checkpoint bindings

### T3 seed42

Source SHA256: `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`.

FULL [trn21 canonical receipt](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20260930-r170-flags-t3-opt3-full-trn21-alternate-preparation/artifacts/results-bundle.json); checkpoint `71b16e3de821979c817749ee16baff706435d1e11b0cdc9ffd52044af9f401ce`; packet `7ffbee242c0ac0cf944ef433293b7e33f58a5e3a8c64d47091073ea8146be9f1`.

FULL [trn22 canonical receipt](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20260930-r170-flags-t3-opt3-full-trn22-preparation/artifacts/results-bundle.json); checkpoint `029510d8f1eba40e9e81f4e93b12665fea158785945b6fda33bfd3b85306baab`; packet `ec306779faf3055779aba6612837a3294e5376c0ca9449f735f2308b9346eadc`.

01-own-lnc1: [collect-20260930T152130433659Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20260930-r172-t3-opt3-second-postfull-eval-trn22-preparation/cases-r188-r2/01-own-lnc1/collect-20260930T152130433659Z.json), SHA256 `02a7384b54952bd99cd03c77b76dbc82fa63b1e400eef001231c9fbd275547f3`; BPB0.9590707374259858; own checkpoint `029510d8f1eba40e9e81f4e93b12665fea158785945b6fda33bfd3b85306baab`; exit0 and original Guardian released.

02-own-unset: [collect-20260930T152827824502Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20260930-r172-t3-opt3-second-postfull-eval-trn22-preparation/cases-r188-r2/02-own-unset/collect-20260930T152827824502Z.json), SHA256 `cd491f3855a7c68bc74d08972c508cb6c6077ffe26671f42483745b99ba1c858`; BPB0.9590707374259858; own checkpoint `029510d8f1eba40e9e81f4e93b12665fea158785945b6fda33bfd3b85306baab`; exit0 and original Guardian released.

### T3 seed43

Source SHA256: `10ec1169ae6eaf69f7b728c34758bb5e28e3fb8df78a795309f73c6a1a9b9362`.

FULL [trn23 canonical receipt](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r200-t3-seeds-newhost-full-preparation/trn23/artifacts/results-bundle.json); checkpoint `3f4d136b83e41ed2f0b35612f832e1936a4642190884d511e02ef5f72064aa25`; packet `7f578ae9dba776f7f9dbdd5ec01b1f50a9e98baded2af27e67ad10b4be19dfd9`.

01-own-lnc1: [collect-20260930T204642636467Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r202-t3-seed43-postfull-eval-trn23-preparation/cases/01-own-lnc1/collect-20260930T204642636467Z.json), SHA256 `90af0e1563163d51d4772719ff77595214fa8a1c186a21841d04e1055efba774`; BPB0.9597143969855484; own checkpoint `3f4d136b83e41ed2f0b35612f832e1936a4642190884d511e02ef5f72064aa25`; exit0 and original Guardian released.

02-own-unset: [collect-20260930T205024915390Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r202-t3-seed43-postfull-eval-trn23-preparation/cases/02-own-unset/collect-20260930T205024915390Z.json), SHA256 `c908c0ff336bcd774558609a16a6784d4bf1c45733d33d7db5c1df72a124d5a6`; BPB0.9597143969855484; own checkpoint `3f4d136b83e41ed2f0b35612f832e1936a4642190884d511e02ef5f72064aa25`; exit0 and original Guardian released.

### T3 seed45

Source SHA256: `fff845f2173308d77252a068587cf28718ea5c51e47fd7cc00ad389ba495aa5b`.

FULL [trn23 canonical receipt](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r202-t3-seeds45-46-newhost-full-preparation/trn23/artifacts/results-bundle.json); checkpoint `85e5ff85e848b00bbdfdf9904279f7d5619549b8d5410f940ab47c31aee68f26`; packet `c184cf2203e6fc697201ac82b384db9cabf7d2c64763a4e31d0140dd11774964`.

01-own-lnc1: [collect-20260930T230247070904Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r206-t3-seed45-post-qbundle-eval-trn23-preparation/cases/01-own-lnc1/collect-20260930T230247070904Z.json), SHA256 `4ac55dbfeeba483624244f68007087074c27216b8293210c045ef9e68f434264`; BPB0.9607595726122182; own checkpoint `85e5ff85e848b00bbdfdf9904279f7d5619549b8d5410f940ab47c31aee68f26`; exit0 and original Guardian released.

02-own-unset: [collect-20260930T230930385505Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r206-t3-seed45-post-qbundle-eval-trn23-preparation/cases/02-own-unset/collect-20260930T230930385505Z.json), SHA256 `b9a9a64af257a7109136348fad597f83375eba6ab340071a003c907bd0bc8edb`; BPB0.9607595726122182; own checkpoint `85e5ff85e848b00bbdfdf9904279f7d5619549b8d5410f940ab47c31aee68f26`; exit0 and original Guardian released.

### T3 seed46

Source SHA256: `54926ae7ca2c44867c94982b1cd471d031a1a5a1bc7daa8983f4b7e92768c415`.

FULL [trn24 canonical receipt](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r202-t3-seeds45-46-newhost-full-preparation/trn24/artifacts/results-bundle.json); checkpoint `285384eb354c5214a14eced7ac00ff51f03ffed736fae8865c7576cc23272a7e`; packet `6c0bd3e5a789e57a832d0e8a0538dc1d32afd62c17558934b8127a37a897b74f`.

01-own-lnc1: [collect-20260930T230247076899Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r206-t3-seed46-post-qbundle-eval-trn24-preparation/cases/01-own-lnc1/collect-20260930T230247076899Z.json), SHA256 `b803866f33a6964954be3424c8bb9538e04676ff649190c0af78fad9560d08a1`; BPB0.9613701580783315; own checkpoint `285384eb354c5214a14eced7ac00ff51f03ffed736fae8865c7576cc23272a7e`; exit0 and original Guardian released.

02-own-unset: [collect-20260930T230930394900Z.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r206-t3-seed46-post-qbundle-eval-trn24-preparation/cases/02-own-unset/collect-20260930T230930394900Z.json), SHA256 `dda27c493a22f6af68e4cc9b69b1797a68c1051f223d713adb146a2548dcda5b`; BPB0.9613701580783315; own checkpoint `285384eb354c5214a14eced7ac00ff51f03ffed736fae8865c7576cc23272a7e`; exit0 and original Guardian released.

## SEQ512 closed — actual native evidence

Both dummy shapes compiled, but total cold failed the gate. External torchrun elapsed1043.0640153740533s; per-rank process cold1033.522611/1033.522339/1033.522388/1033.522593s. The required cap was860s.
Exact terminal error: `RuntimeError: SEQ512_EXTERNAL_DUAL_COLD_AT_LEAST860`; rank assertion: `AssertionError: SEQ512_DUAL_COLD_AT_LEAST860`.
Canonical raw archive4b688d5eca98988b3970af18c271690e7fd1167e8376dd939acbe718c6cdcbde; terminal53302b55d3af5477d9d43832d6135a66a567dda5a09d4b17a16f3b5ac941d8e1. Original authenticator confirms failure, unchanged protected inputs, Guardian release, no related processes, idle device and free original lock. Zero checkpoint artifact bytes; no scored result or performance claim.
Root closure: [ROOT_FINAL_R207.json](/home/poojith-devan/Desktop/Trainium/research/v2/operator/20261001-r206-seq512-compile-trn21-post-r3-author-preparation/ROOT_FINAL_R207.json).
R207 closes all further science, retries and new seeds. All four hosts are now idle according to their latest original postflight receipts; no upload was performed. T3seed42 exact R191 UPLOAD_READY remains the selected file. Existing audit copies remain unchanged.

The repository's sealed reconciliation snapshot remains unchanged with its historical pendingSEQ row. This Downloads report adds only the later Root native closure; it does not rewrite that snapshot or any Claude verdict.

## Read-only replay

```sh
python3 -B research/v2/reconciliation/20261001-r207-t3-ranked-table/reconcile.py
```

The default replays original metadata validators and verifies this snapshot. `--emit` prints current evidence only. New evidence or package bytes cause snapshot drift, not permission to rewrite frozen packages.
