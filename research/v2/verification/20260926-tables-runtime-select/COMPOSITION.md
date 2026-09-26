# Narrow runtime-address composition check

26 September 2026, approximately10:14 UTC; main Supervisor, not the Implementer.

Read the exact SPEC, complete SOURCE.diff, production kernel, current int32 CPU
receipt, production compiler receipt, all-groups official simulator receipt and
artifact manifest from `lanes/tables-runtime-select/`. This is a narrow repair
composition with the existing FP32 table design review, not an extra review gate.

Executed `python3 -B research/v2/verification/20260926-tables-runtime-select/check_addresses.py`.
Exit0: PASS_CHANGED_ADDRESS_EQUIVALENCE,524288 rows/67108864 elements covered by
contiguous owner intervals;119 other top-level AST nodes unchanged; four compiler
artifact hashes and all input hashes in both dtype CPU receipts verified.
Only `_k2_table_build_kernel` differs from4079. Source414cdd76 is frozen.

The new reshape/select addresses have identical row-major offsets and disjoint
64-channel LNC ownership to the intended previous slices. FP32 accumulation,
encounter order, masks, metadata, optimizer and eval code are unchanged. The
stdlib check does not prove native VirtualRegister support; actual production
backend plus official simulator receipts provide that separate local evidence.

Both CPU receipts assert current-source OFF3 bit identity and ON2 ordinary-vs-
registered identity. Compiler NEFF5a854075 and two official full-shape fixtures
are archived in2a51d40c5. The all-groups fixture has exact FP32 equality and
516096 untouched rows. No Trn2 speed, startup, native collective exactness or
full-quality result is asserted. Preserve one native timing as the agreed test;
do not add local framework-forward or another OFF/review gate.
