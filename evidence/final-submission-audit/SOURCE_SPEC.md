# R168 T3-OPT3 — one flag on LaneB-F1

Frozen scientific parent: `02c758a9bd980819333efeb48557e26e44c680c4e683f8eaf491782574d92c38`. Candidate: `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`.

## Exact change

Replace --optlevel=1 by --optlevel=3 in the two existing compiler-default strings. There is one causal setting, shared by training and import/evaluation.
The complete byte suffix beginning with `import torch` is identical to LaneB-F1;
all model/kernel/parameter/optimizer/loader/loss/checkpoint/evaluator code is unchanged.
F1 LNC selectors, original V/O, and every numerical constant remain unchanged.
No L10a/L10b, E1, other flag, log statement, or evaluator override is stacked.

## Native environment contract

Omit only the existing NEURON_CC_FLAGS entries in both TRAIN_ENV and EVAL_ENV of the inherited matched60 driver. Do not inject a candidate compiler flag value. Exact source setdefault then supplies the main/import defaults. All other driver environment entries, original command, world=4/LNC=2, seed42 and guards remain unchanged.
For T1/T3, evaluate the pinned pre-Torch environment prefix with stdlib only and
record its values separately. The same operation on the unchanged parent must
reproduce the matched control TRAIN_ENV/EVAL_ENV flags. Such a record is source-
prefix evidence, not native-observed compiler argv. Bind real compiler argv/log
if present; do not fabricate it. Existing caller-supplied values retain setdefault
precedence in both scientific sources. No runtime/compiler is touched by the prefix proof.

## Evidence and limits

Compiler optimization can change lowered scheduling/arithmetic order. Source math is unchanged; no bitwise native-equivalence assertion is made. Cold >=900 s fails.
R168 is the authority excerpt; model runtime identifier is unknown. Static checks
prove exact source edits and source-body identity only. Instance-only author CPU
checks authenticate source/default/override semantics without Torch or tensor
numerics. Actual author CPU, independent review, native, FULL and release are NULL.

## Gate and resources

One 60-step original cold matched timing on trn22 against actual LaneB-F1 control
archive394f33b627e5ff1706aace3629d2f6d2e4c033e673f44acf8927548f50d9eb1b,
packetaad4317abef50b73f481191abb0308298e189a214851454e82d471f76a2151be,
327001.73740096996 tok/s. Require >=1.015 times this rate and cold <900 s.
Original entry22GiB/runtime20GiB, four ranks, lock/Guardian/STOP/process/device/
cache/charged-budget controls are not altered. Root alone admits and launches.
Timing is not FULL acceptance. Only actual passing flags may be combined in a
separate future FLAGS candidate; its one FULL must beat its applicable base by
at least0.0004. No combined source, T4 or T2 companion variant is built here.
