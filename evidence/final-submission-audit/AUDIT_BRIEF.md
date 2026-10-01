# T3 audit — R186, own-checkpoint diagnostics pending

Only train.py is the proposed source payload. Exact source `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`; exact LaneB-F1 parent `02c758a9bd980819333efeb48557e26e44c680c4e683f8eaf491782574d92c38`. SOURCE.diff changes the two source-owned NEURON_CC_FLAGS defaults from optlevel1 to3 (lines39 and41); every byte from import torch onward is unchanged. This can change compiler lowering/arithmetic order; no native bitwise claim is made. Existing caller overrides retain precedence. Do not accidentally override the O3 defaults when reproducing.

Two exact-source research FULLs, both seed42/public20971520 tokens/context1024:

| Host | BPB | Cold seconds | Charged through exit seconds | Stable checkpoint SHA prefix |
| --- | ---: | ---: | ---: | --- |
| trn21 | 0.964447246764078 | 606.3875389539171 | 1793.3390622320585 | 71b16e3de821979c |
| trn22 | 0.9642283417620233 | 598.5196062619798 | 1793.7412270181812 | 029510d8f1eba40e |

Exact pair mean: 0.96433779426305065. Both checkpoints587788862B; exact full hashes/remote paths and stable save/postflight/terminal records are in each host's cold-evidence.json. Checkpoint tensors were neither downloaded nor opened here. The ordinary FULL evaluations used LNC2 and source-owned O3 import default, not LNC1/unset. Their existing evaluation caches are disclosed; training used six freshly created empty cache directories.

**Own-checkpoint LNC1 and unset 2M diagnostics: PENDING.** Neither is fabricated from FULL20M, parent/source diagnostics or an old seed42 target. One actual selected T3 checkpoint must be bound later; firstLNC1 establishes its observed2M result, then unset is compared. No evaluator/source modifications are included.

R186 selects T3 as the current base/upload candidate and explicitly requests this export so source audit can begin now. This self-consistent export is not an independent package-integrity verdict, Claude package acceptance or upload permission. Those actual records remain NULL. No host or submission operation occurred. Root continues the diagnostics in an authorised idle slot; Poojith alone uploads.

The included independent source/CPU review88b564 has its original limited prefix/source scope. Existing raw author and own receipts are retained; no new self-review is claimed by the exporter (who authored T3). Both original full_receipt_r172 verifiers were replayed as read-only metadata checks.

## Human rejection checklist (unknowns remain explicit)

1. FAIL (pending Claude audit): exact train.py and command supplied; both FULLs pass in the research environment. Clean organiser-image import sufficiency is not independently proved. No new pristine-AMI gate is invented.

2. FAIL (own-checkpoint diagnostics PENDING): train.py1413 returns logits when targets=None; load_for_eval1922 uses the trained state. All post-import-Torch bytes equal LaneB-F1, but source/CPU prefix proof is not native tensor equivalence. Actual own-checkpoint LNC1/unset 2M receipts are absent.

3. FAIL (pending inherited compliance audit): exact diff changes only defaults at39/41; evaluator/prepare/data code is untouched. No new external weights/data or network path is added. Dynamic I/O/network telemetry was not measured; inherited cache/path behavior is disclosed in raw environments.

4. PASS: trn21 cold606.3875389539171s/charged1793.3390622320585s; trn22 cold598.5196062619798s/charged1793.7412270181812s. Both six-cache, stable-checkpoint, fresh20M and original Guardian gates authenticate.

5. FAIL (unmeasured telemetry): each stable final checkpoint is587788862B; cumulative/transient/overwritten checkpoint/log/cache writes, complete current leftovers and backup state are UNKNOWN. Disk snapshots are not total writes.

6. FAIL (pending required-versus-optional audit): exact command is supplied. Defaults at39/41 are O3; caller NEURON_CC_FLAGS remains optional setdefault precedence and was omitted by both FULL drivers. Their complete recorded environments are included; no claim that only command variables were present.

7. FAIL (pending compliance adjudication): seed42 atmain1994 is fixed for both runs; inherited elapsed-time schedule/stop/save behavior is unchanged. Same-seed cross-host outcomes are not a deterministic-replay or independent-seed guarantee.

8. FAIL (pending readability audit): source is frozen, not minified or cleanup-rewritten; two default-string edits only. Existing dead/optional/fallback paths are retained. Adequacy of inherited kernel/docstring/readability remains Claude judgment.

Finalization-r2 is included as the existing stage/unknown-telemetry policy, not as T3 acceptance. No pristine-image experiment, dynamic network trace or cumulative-I/O probe is added as an audit-handoff gate. No existing audit/Claude verdict file was overwritten; this is a new audit-t3 directory.
