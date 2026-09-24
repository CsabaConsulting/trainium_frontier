# Track B2 sparse candidate rejection

Date: 2026-09-24

The Implementer produced source-only candidate commit
`b2e615086ba0468720f42a53303081e24e16a343` from exact `35c9...` bytes. It changed only
`submissions/dense6x1024_public0992851/train.py`; candidate SHA-256 was
`3756a289a99a10471d300545833809df788ab3677b64fbabaa75a734d8c1d3ce`.

Independent verification rejected native-probe readiness before hardware use. The candidate lacked
the required probe export and fixed-world guard; zero-touch ranks could execute different
collective counts; its sparse COO coalescing, boolean compaction, dynamic slicing and device
`.item()` control flow were not credible fixed-shape Neuron paths; and exact lazy BF16 RMSProp
required an unbounded gap-proportional loop. Strict-load/key/table/logit and resume-next-update
identity were also not yet bound to a real checkpoint.

The Implementer then issued a STOP recommendation rather than weaken semantics or invent SDK
support. Exact sparse-gradient capture, deterministic fixed-shape duplicate reduction and bounded
bit-exact BF16 deferred decay remain unresolved. Do not run or push this candidate. Reopen sparse
training only after separately approved native component probes establish the missing primitives.

This rejection does not block the flag-only dense table A1/A2 experiments.
