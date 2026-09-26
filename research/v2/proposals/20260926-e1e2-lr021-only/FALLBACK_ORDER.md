# Ranked existing-code fallback plan

Researcher evidence snapshot, 26 September 2026, after actual P0 results.
This is not a shared queue, launch assignment or claim that three hosts/jobs
are ready. Both live jobs are protected. Admitted Final Sprint K2 work takes
priority; compiler-local-constant K2 still needs its one fresh review/native gates.

1. First fallback: `20260926-e1e2-lr021-only`, preferably trn21 after its
   actual e1e2 control plus fresh20M and quiescence. Source8c6d7af7, explicit
   batch131072, matrixLR.021 only. The existing factorial evidence has baseline,
   doubled-batch-only and doubled-batch/LR.021, but lacks this LR-only cell.
   This most directly tests whether the joint improvement depends on batch.
2. Second independent fallback: `20260926-e1e2-lr01875-only`, preferably
   trn22 after live batch.021 plus fresh20M. Same source/default batch, only
   matrixLR.01875: an untested1.25x intermediate point below the.021 endpoint.
   Lower priority than actual selector-required confirmation or admitted K2.
   Not a claim the batch arm won, and not retesting old6layer LR points.
3. A third unconditional high-information ready job cannot honestly be supplied.
   Conditional confirmation already exists: if actual trn21 e1e2control makes
   the existing Recursive arm NOMINATE (or the unchanged selector requires
   replication), use its already-proposed trn22 counterpart. Otherwise do not
   repeat it merely to fill a slot. Likewise replicate a new LR-only arm on the
   other host only after its actual selector verdict calls for that.

Important exact-source distinction: audited literal e1e2 submission source has
no `--matrix-lr` parser. The two new proposals use reviewed/native-tested
`research/v2/proposals/20260926-sprint-p0-clean-batch/train.py`, SHA
`8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee`,
whose defaults preserve e1e2 behavior and expose the two flags. No source edits.
MATRIX_LR changes both existing Muon matrix and local-convolution AdamW groups;
other group rates and the baseline schedule remain unchanged.

Existing Recursive conditional alternative:
`research/v2/proposals/20260926-sprint-p0-clean-recursive/EXPERIMENT_PROPOSAL.json`
SHA `ce03364846e30c156005703dcbd9bffc9a965bb5f8956e731365200cb3674c2c`;
source `50fa6d144f19758dcdf9980e03f2302453174b956f0fa1a3e0845c4c0b72e6fb`;
label `fs-20260926-sprint-clean-recursive-trn22-r1`;
existing `operator/20260926-final-sprint-p0/recursive-trn22-successor-prepared.json`
SHA `63ffc54e19571acdefa14d6256c3a74ebadc1dc658f2e52e39d320e88ba31dab`.
Actual trn21 Recursive BPB0.9880239239389417 is not a confirmed gain:
its genuine matched control is pending. Keep the original arm/base/roles.

No batch+Recursive stack is proposed. The accepted clean ports expose batch/LR
and Recursive in separate files, so that combination is not presently flag-only.
Neither an unconfirmed batch nor a schedule is an adopted stack winner, and
the batch quality-arm rows cannot be relabeled as batch-base control rows.
A later true batch-base LR optimization needs prospective comparable-control
and selector bindings, not an invented control or use of another host's score.

History retained: batch262144/LR.015 is DISCARD against actual trn22 e1e2
(delta+0.0000899543). Batch262144/LR.021 trn21 is0.9869854734569898,
still awaiting its matched control and live trn22 replication. Old clock36
LR.012 was a small ambiguous difference, not justification to rerun it.
Old6layer LR.0175 lost, but is not evidence that these untested8layer values win
or lose. No numeric gain is predicted here.

All proposals retain seed42, official1800 with >900 startup charged,
fresh public20971520/context1024/batch1, actual same-host e1e2 controls,
unchanged thresholds, original guardian/lock and resource floors:
initial14495514624 B, runtime11274289152 B, host RAM68719476736 B.
Only the existing sole Operator may mechanically prepare/admit a selected
unused label. All new stage/launch flags are false; no host facts are invented.
