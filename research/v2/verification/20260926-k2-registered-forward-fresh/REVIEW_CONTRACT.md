# Fresh independent source/CPU review

You receive only a full candidate adapter diff, neutral component specification,
unchanged source context, raw installed-SDK source evidence, and developer gate
receipts. No implementation branch, implementation notes, earlier reviewer
conclusions, conversation history or production repository is part of this review.
Do not inspect any repository or another agent's worktree. Work only in this packet
and an output/ subdirectory here; use the installed local CPU Python runtime.
No network, host, AWS, native device, external dataset/weights, SDK/package install
or source edits. Do not submit anything or claim native qualification.

Default REFUSE until evidence for this limited source/CPU gate is complete.
Reconstruct the 44-line adapter from its diff and independently write and execute
at least one exactness reference test; do not merely replay the developer's checks.
Use one-thread tiny synthetic CPU inputs, no model, <=64MiB tensor allocations.
The Python runtime itself is not included in that tensor limit.
The available local runtime is /home/poojith-devan/venv/bin/python
(Torch2.13 CPU), whereas the actual target is Torch2.11/NKI0.5; state limitations.

Assess the six-input schema, saved metadata, correct gradient return arity,
dtype/byte semantics, no mutation/cache, default-disabled no-Torch-import path,
unchanged automatic LNC2 backward callable and actual installed nki_op contract.
Use the archived unchanged nki_op AST with real local torch.library plus your OWN
tiny ordered FP32 consumer substitute if needed. No import of torch_neuronx/nki.
Independently test eager/AOT gradients, changed nonzero tangents and metadata, and
a poisoned consumer so that an ordinary embedding fallback cannot masquerade as
the registered derivative. Strict bytes are required for this diagnostic scope;
do not silently relax to a tolerance. Preserve failed attempts and exact errors.

The supplied native checker is unchanged context/intended future coverage, NOT a
passed native receipt. Native correctness remains unproven even if CPU suitability
passes. Future localization then all20 original fixtures/all4ranks and independently
bound hardware checks remain required. No speed/BPB/adoption claim follows here.

Leave output/REPORT.md, your own test source, exact commands/runtime information,
and raw outputs. Give a scoped PASS or REFUSE with concrete evidence and limits.
If a necessary claim cannot be checked, identify it precisely; do not invent
execution or reinterpret later-native checks as already completed.
