# Base initial-cache refresh: bounded independent source review

Start at REFUSE. Review only the supplied full-context source diff, this lane
specification, and raw CPU receipts. Do not consult earlier implementation or
review notes. The requested decision concerns suitability for a later exact
full-size CPU qualification, not native execution or scientific adoption.

The proposed causal factor is a Boolean CLI option that defaults OFF. When ON,
it requests differentiable casts of the existing Linear weights immediately
before the first training forward. Preserve FP32 master parameters, the assigned
BF16 compute path, gradients and optimizer semantics; preserve the default/OFF
execution and saved-model/evaluation behavior. Do not change training data,
microshape, LR/schedule, total batch, model architecture or budget accounting.
All preparation inside the timed training loop must remain charged by the
existing clock. The normal later cache-refresh behavior must be preserved.

Base source SHA256: 6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3
Candidate source SHA256: bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522

The historical receipts are reduced CPU observations: four ranks, three steps,
width 128, depth 6, physical synthetic inputs 2x128 per rank, synthetic prepare,
no compilation, and a deterministic fake clock. They are not full117M G2, real
data, native correctness, compile-time, throughput, cold-start or budget proof.
Full117M standard G2 has not run for this exact source. Raw observer fields can
be used as evidence only within their documented/structural meaning; do not
infer execution counts merely from field names.

Independently reconstruct both sources from the diff and authenticate their
hashes. Write and actually run your own small single-thread CPU reference for
default/OFF identity and ON BF16-cache/FP32-master forward and backward/update
equivalence. Check repeated steps, cache connectivity, and state/load behavior
where relevant. Do not import the full training module or allocate a full model;
use extracted definitions and very small weights. Standard CPU libraries are
available. Do not access other repositories, branches, notes, hosts, network,
credentials, or hardware; do not modify candidate source. Do not launch training.

Return REFUSE with concrete counterexample if evidence is insufficient or any
required behavior differs. Otherwise state the precise limited CPU-qualification
scope supported, and keep full-size, fresh-session policy and native gates
distinct. Preserve your own source/test/results and actual commands. No prior
review verdict is supplied in this packet.
