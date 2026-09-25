# K3: establish the active path before another kernel flag experiment

Supervisor/Analyst static inspection, 26 September IST. Frozen source is
`a4aa4351789115a2d0aa947d9ca00ff689f9bd0b:workflow/proposals/research-v2/train.py`,
SHA-256 `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
This is not the repository-root experimental `train.py`.

Training installs a replacement MLP forward across all layers before `main`.
Its backend is explicitly `native`; eligible BF16 training calls a native
linear/ReLU/square/linear forward and custom transposed backward. That path
never consults `USE_NKI_RELU2` or calls `relu_squared`. The original standalone
activation kernel therefore is not an active training lever just by adding
`--nki-relu2` to this base. Do not spend a host timing slot on that flag alone.

The original forward remains a fallback, including evaluation, non-BF16 inputs,
an explicit precomputed `fc`, and unsupported tile shapes. This is why the
finding is conditional on training eligibility, not a claim that the flag is
globally dead. The supplied stdlib AST audit checks exact source and guards;
its shape examples are arithmetic, not runtime observations or numerical proof.
The first audit attempt correctly failed closed with `RuntimeError: One main
entrypoint`: the file also has a separate main-only compiler-environment guard.
The corrected audit requires both guards and uniquely identifies the training
one by the installation call; the source was not changed.

A genuine fused activation proposal must modify the active adapter, preserve
its saved-ReLU layout and forward/backward semantics, and prove OFF identity,
ON correctness, cold startup and at least 3% whole-step gain. A standalone
component gain or a CPU fallback test does not satisfy those gates. No such
patch is implemented or assigned here.

The attention path is distinct: unmasked direct SDPA is selected only when
`COMPILE_SDPA_DIRECT` and not `USE_NKI_ATTN`; enabling the existing NKI attention
routes through a Dynamo-disabled wrapper with geometry checks. Existing
comments record a prior neighboring RoPE lowering failure. That comment is
historical evidence, not a reproduced current failure. Before proposing a new
flash kernel, inspect its existing native history and measured device cost.
No attention implementation, speed claim or train/eval divergence is authorized.

Next: use the pending P device decode to rank fused activation versus attention.
K1/K2 and an actual architecture decision-tree follow-up retain priority.
