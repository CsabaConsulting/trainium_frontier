# Seed43 Recursive variance diagnostic — terminal raw receipts

Exact label `fs-20260926-recursive6ef7-seed43-trn21-r1-full`, trn21.
Source6ef7, same Recursive configuration, only seed43 in place of seed42.
Training receipts were committed separately in7a5847c6.

- 3646 training steps; actual process wall2594.4321609359467s.
- Uncapped fifth boundary804.1663017190294s; charged1790.2658592169173s.
- Fresh public20M/context1024/batch1: **0.9922035160012295 BPB**.
- Raw numerator57400540.0nats /83462186bytes /20971520tokens.
- Eval698.8205484380014s, process711.815239608055s, exit0/no timeout.
- Eval ended2026-09-25T21:19:10.613206Z; driver and outer flock exit0.
- Checkpoint469834913bytes; SHA ebe67c94d93d4fa7c1bab915bb472be7e6877ecf1610dc01799fe7ae7f0ee5fa.

Against the same-host seed42 Recursive control0.9899528600562257,
seed43 is +0.0022506559450038 BPB. This measures seed/run variability;
it is neither a new architecture result nor an adoption or submission claim.
Independent result review remains Supervisor-owned. Seed42 remains the
submission convention; do not select the best seed after seeing results.

The ten closed small artifacts were retrieved with pinned transport and
SHA-256 verified against the host. Manifest records each exact hash.
No checkpoint was downloaded or changed. `terminal-quiescence.json` records
21:21:11UTC: no descendants/Neuron users, original lock free, old runner
PID2983496 absent, and30.14GiB free. Subsequent architecture launch is separate.

Operator verification reproduced all ten hashes and BPB arithmetic within
1e-15. A first local strict float-equality assertion failed from arithmetic
ordering; the raw evaluator result, files and native run were unaffected.
