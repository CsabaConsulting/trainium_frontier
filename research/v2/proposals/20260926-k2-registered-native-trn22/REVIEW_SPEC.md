# Registered K2 native checker and one-use packet: neutral specification

This is preparation only under selected proposal 8bab89e1 and frozen adapter
aa3d4762. No stage/launch/host reservation is authorized. Exact label:
`fs-20260926-k2-registered-forward-trn22-r1`.

Protected adapter d761a727, symmetric kernel 2c8295a4 and reference 14572563
are byte-identical copies. The old Function wrapper and its tensor record,
comparison and route-observer ASTs are copied without change from the frozen
FP32-first checker bd2bca7c. No old file or failed native receipt is changed.

The new checker must:

1. Remain disabled without Torch/NKI imports unless explicitly enabled; require
   exact kernel/reference/adapter hashes, world4/LNC2 and unique output.
2. Execute the unique FP32/nonzero sentinel first. Record upstream roundtrip,
   direct/repeated consumer, dense CPU oracle, registered eager/compiled and
   ordinary compiled forward/backward; record old Function eager/compiled in
   a separate field. Candidate/control failure prevents the full suite.
3. After sentinel success, execute every original CASE_NAMES fixture for BF16
   then FP32, including both logical-empty direct fixed-storage checks. Preserve
   the original 150 comparisons per rank; add ordinary compiled controls and
   separate original-route observations. Never omit FP32/cancellation/subnormal
   cases or accept a partial/sentinel-only result.
4. Register once per dtype and reuse the same compiled callable across runtime
   metadata changes and a second distinct nonzero unique upstream (negated
   original) per dtype. No captured gradient or duplicate operator registration.
5. Record seven runtime input/metadata records before/after every route. These
   are host readbacks, not observed values inside the compiled backward.
6. Require strict finite shape/dtype/full-output bytes; safely encode NaN/Inf
   errors; always attempt final receipt and process-group cleanup. Old Function
   mismatches are preserved separately, neither candidate success nor a reason
   to suppress remaining candidate observations. An old execution error is
   recorded as such, not converted to an exact comparison.

The validator independently reconstructs every gradient and the fixture input
hashes from the protected stdlib reference. It requires all4 complete receipts,
all20 cases in order, both replays, sentinel, exact source/argv/label/environment,
zero exit consistency, strict comparisons and identical required observations
across ranks. It rejects missing/malformed/nonfinite/zero-gradient or contradictory
required evidence. A known old-route mismatch may be retained without preventing
separate candidate comparisons; missing old observations are not accepted.

The driver/rank-entry executable bodies are identical to the reviewed FP32-first
packet fa1a3652. Only constants/pins/argv/label/root are rebound. The adapter adds
one explicit support pin/argument. The shared guard and dispatcher are unchanged.
Keep the original trn22 lock device66305/inode832589, one synchronous world4 LNC2
child with 600s TERM/60s KILL grace, inherited lock FD, STOP checks, disk>=10.5GiB,
fresh in-lock ps and neuron-ls, and six absent/empty per-run caches (URL=DIR).
There is no component timing, training, evaluator, quality run or automatic next.

The sole Operator must separately capture actual process/descendant/compiler
identities while running and prove all of them quiescent plus fresh device idle
and original lock identity after EVERY outcome. The old driver is not a new
universal process-tree cleanup controller; failure/missing/ambiguous postflight
parks the route. No unrelated kill/cache deletion is authorized.

Fresh source review, independent checker/mechanics review, a separate one-use
Supervisor release and fresh remote checks are all still required. Raw observations
await independent result authentication; neither a zero exit nor this packet
automatically changes any historical failure, native qualification or adoption.
