# P2 runtime-index HBM views: one minimal compiler repair

Supervisor selected this direction after the exact production parser failure
of source4079. This is one new derivative, not a modification of frozen4079,
K2 source662c, the evaluator or any data. No native launch authorization.

Candidate SHA-256:
`414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34`.
Parent SHA-256:
`4079e8e6bff1f8438e0430a8b8b2099085da5d6f908149268ac17e8b318fdbff`.

## Diagnosed failure and prospective fix

The exact NKI0.5.0+28631259367.ga768afa6 parser rejected parent line257
`first = group * partition` with `mul ... got (object, int)`. Its dynamic
loop index is an ISA VirtualRegister and cannot participate in ordinary
Python multiplication or Python slice endpoints. Existing CPU tests used
Python integer loop indices and could not detect that compiler restriction.
The failed receipt remains in the frozen parent lane, commit1899d6ea0.

The actual installed `NkiTensor.select` implementation accepts dynamic scalar
indices, including VirtualRegister, and records the selected dimension's
stride. Reshape HBM into explicit row groups and use `.select(0, group)`:

- counts/starts `[4096,128,1]` select group -> `[128,1]`;
- maxima `[4096,1,1]` select group -> `[1,1]`;
- output `[4096,128,128]` select group -> `[128,128]`, then statically slice
  owner0 channels0:64 or owner1 channels64:128.

No arithmetic on the runtime index remains. The4096-group dynamic loop and
dynamic occurrence loops remain; no4096-way unroll, graph expansion or new
fallback is introduced. The complete source diff only replaces these four
HBM access expressions and removes the invalid multiplication.

## Exact address/ownership claim to verify independently

For group `g` in `[0,4096)`, row lane `p` in `[0,128)` and owner `o` in `{0,1}`:

- counts/starts element address = `g*128+p`, unchanged from former slice;
- maxima element address = `g`;
- output element address = `(g*128+p)*128 + o*64+f`, `f` in `[0,64)`.

The two owner intervals are disjoint and cover each output row; all4096
groups cover524288 rows without overlap. The selected views preserve the
same backing HBM storage, dtype and dense write operations. Every token
load, mask, sentinel, FP32 addition and encounter ordering is unchanged.
FP32 master table/gradient/microstep accumulation/AVG/RMS and BF16 gathered
activations remain exactly as parent. Hashes, model, forward gather, optimizer,
timing/budget, checkpoint and evaluation code are byte-unchanged.

## Local checks and scope

Exact production backward input is BF16 `[32768,128]`, counts/starts int32
`[524288,1]`, positions int32 `[32768,1]`, maxima int32 `[4096,1]`, LNC2,
output FP32 `[524288,128]`. Real parser/backend retain these full dimensions.

`test_parser.py` is re-bound only to the new hash. `check_backend.py` adapts
the proven source-bound K2 harness to the table factory and exact geometry,
using the exact kernel's `CompileKernel._compile_opts()` production policy,
not raw experimental allocation options. Version-matched actual neuronx-cc
produces a NEFF; no mock or Python interpreter is called a compiler.

The official NKI simulator has two full-shape prospective fixtures: four
collisions on each of the first8192 rows (4032 entirely empty groups), and
four collisions on two rows per every128-row group including the final row.
Both have516096 untouched rows and BF16 signed token/channel-varying gradients.
The independent FP32 reference accumulates in original token order. Chunked
comparison avoids allocating unnecessary full-output temporaries under2GiB.

The existing small CPU suite is reused unchanged. Its instruction model adds
only ordinary Torch view implementations of `reshape` and `select`, including
the existing write-count views. This is a disclosed test-harness change:
CPU integer selection is not evidence of runtime VirtualRegister legality.
Real compiler/simulator checks supply that separate evidence. OFF3, ON2,
FP32-reference, hashes, RMS, mocked ownership, clipping and checkpoint limits
remain as previously documented. Cross-shape prefix error remains disclosed.

All heavy tests are sequential under the Supervisor's single local-process
slot, OMP/OpenBLAS/MKL threads1; compiler/simulator address space2GiB and bounded
timeouts. Source stays frozen during testing. Independent review, actual K2
native timing and table native checks still precede any scientific full run.
The human accepted unchanged ordinary framework forward evidence plus native
timing; this repair does not invent a new local full-model forward gate.
