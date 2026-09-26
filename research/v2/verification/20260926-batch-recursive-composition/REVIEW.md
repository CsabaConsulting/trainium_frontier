# Independent batch/LR + Recursive composition review

Verdict: **PASS for the supplied composition diff**, 2026-09-26 10:33 UTC.
No blocking defect found in this scope. This is one bounded independent
Verifier review, starting from REFUSE and using only the four assigned
correctness inputs. Required repository contracts were read for authority;
historical scientific verdicts were not used to judge this change.

The candidate source is declared as
`c2e27ccde59e619d74102429e340716dbec16a6ba538ea9beca9c714d0d33f87`.
The Supervisor separately authenticates the full sources and exact diff
coverage. I did not open full candidate/parent/donor sources, the author test,
Implementer notes, or Implementer branch/history. The source hash here is a
binding from the supplied manifest/spec/receipt, not an independently read
source-file digest.

## Findings

The six diff hunks add the default-OFF flag, donor schedule branches, beta2
transport selection, CLI flag assignment, and group schedule routing.
The existing batch/LR parser, validation and assignments remain unchanged.
There are no edits to architecture, forward, clock/startup accounting,
checkpoint/eval loader, data, reduction/clipping, optimizer ownership or
the compiled Muon arithmetic body in this diff. Existing accepted forward
evidence is not reopened.

OFF preserves the pre-existing LR/momentum bodies. `lrm_adam` aliases `lrm`,
so the new nested routing expression produces the exact previous multiplier
for every supported kind, including `no_lr_decay`. OFF still supplies scalar
`1 - group['beta2']` to the Muon call and explicitly sets beta2 to `0.9`.
The additional `group['kind']` lookup is satisfied by every recorded optimizer
group; this review does not claim support for malformed groups lacking kind.

ON selects Muon's 0.95 warmdown and non-Muon's 0.65 warmdown with the 0.05
endpoint. It routes `no_lr_decay` to the existing warmup-flat multiplier,
retains the momentum recurrence and beta2 0.95-to-0.97 recurrence, and passes
the existing `one_b2_t` through the FP32 Muon path. The routing applies to the
local-convolution AdamW group as well as the matrix Muon groups at LR 0.021,
as specified. No donor/native arithmetic claim is reopened or broadened.

## Independently executed evidence

Command, exit 0:

```
python3 -B research/v2/verification/20260926-batch-recursive-composition/your_reference.py
```

The new stdlib reference extracts and executes the exact added schedule
branches and changed routing/transport expressions from the reviewed diff.
It compares them with separately written scalar references, using the
receipt's actual group initial LRs. It does not import candidate or author
code, Torch, NKI, a compiler, SDK, data or host interfaces.

- 3,696 scalar LR/momentum/beta2 comparisons passed; absolute tolerance
  `3e-16`, observed maximum error `1.1102230246251565e-16`. References
  independently rearrange the interpolation formulas, so bit identity is
  intentionally not claimed for these algebraically equivalent formulas.
- Cases include adjacent floats on both sides of both cooldown boundaries,
  progress below zero/above one, step 299/300/301, warmup values 0/1/20,
  and positive/zero/negative budget denominator handling. Warmup values are
  test parameters, not a claim about an unseen production constant.
- Four explicit final-value checks passed: Muon and AdamW LR multiplier
  0.05, momentum 0.79, beta2 0.97.
- 448 routing comparisons passed with bit-exact float equality, for OFF
  and ON, both flat-decay settings, receipt groups and an extra non-Muon kind.
- Ten transport comparisons passed: OFF gives the exact original scalar;
  ON preserves the identity of the supplied tensor-placeholder object.
  This is routing identity, not a native tensor arithmetic test.
- Input hashes agree with the manifest; receipt shape/step scope matches
  the stated OFF-default, OFF-target and donor-ON witnesses.

## Supplied receipt scope and limits

The author receipt reports actual MuonAdamW three-step parent/candidate OFF
identity at default LR 0.015/accumulation 1 and target LR 0.021/accumulation 2,
plus donor/candidate ON at LR 0.021/accumulation 2. It records eight layers,
width 128, vocabulary 256, sequence 32, 1,475,892 parameters, synthetic data,
seed 42, BF16 compute/FP32 parameters, and 12 Muon math calls per witness.
The OFF receipt records scalar beta2 transport and ON tensor transport.

Those model-update witnesses are supplied author evidence, not independently
rerun here. The receipt contains one compared-result record per witness,
not both raw execution traces. Its schedule/routing counts alone do not
provide numeric fixtures, so the independent execution above uses the exact
diff branches and a separately constructed reference. This distinction is
preserved rather than relabeling the author's PASS as an independent rerun.

The small model does not materialize the full global batch, test the full
113M model, distributed ownership, checkpoint I/O, native tensor transport,
compiler/startup behavior, throughput or BPB. These are limits of this
review, not newly invented gates. No heavy process, host access, training,
launch, adoption, selector change or submission was performed or authorized.
The Supervisor selects any run and the existing sole Operator operates it.

## Exact input SHA-256

| Input | SHA-256 |
| --- | --- |
| source.diff | `79d3e7e9cf0cda770a580d4e8ef2fd156a95965afd2851378803c381046a7516` |
| SPEC.md | `5a0f9fde3d09d2420a38905b7b5b8841660475a69992abd3d995d44ee192fa95` |
| CPU_RECEIPT.json | `8e1698a3e7d993c1c4fbc1d147e644ab6275b61097d24d483b8b069e60f32fef` |
| MANIFEST.json | `64f3ecea25e0e1cae828066696ef5b6ebe8692255b0f7bd31b2c2970df844d26` |

Independent reference SHA-256:
`44b375e32cc6a96ce113fe922c3458412c4ce65431722c42b109971b170e1e5b`.
