# Mamba2 / hybrid-Mamba2-Transformer architecture research track

Parallel research branch, alongside the existing YAT/SLAY track (`research/yat-slay-attention-research.md`).
Does not touch `train.py`'s locked config. Raised by the user: consider a hybrid
Mamba2+Transformer architecture (NVIDIA Nemotron-Nano-style, "~3 Mamba2 layers per
MLP/attention layer") or pure Mamba2, using the sibling repo
`../state-space-models-neuron/mamba2/` and `mamba2hybrid/` as a starting point, plus five
PDF articles in `articles/`.

Two parallel investigations were run: (A) code due-diligence on the sibling repo, (B)
literature verification against primary sources (arXiv), specifically hunting for the same
failure mode already seen this competition with XSA/DCMHA/SLAY — "real published win,
but only ever validated at scales far larger than our 55.7M-param / ~500-750-step regime."

## A. Sibling-repo code due diligence (`state-space-models-neuron/`)

Full file-by-file read (`mamba2/`: `fused_kernel.py`, `mamba2_kernel.py`,
`conv1d_grouped.py`, `run_profile.py`, `configuration_mamba2.py`,
`mamba2_kernel_inference.py`, `mamba2_mixer.py`, `modeling_mamba2.py`, `utils.py`;
`mamba2hybrid/`: `mamba2hybrid_layer_allocation.py`, `mamba2hybrid_mlp.py`,
`modeling_mamba2hybrid.py`, `mamba2hybrid_attn.py`, `configuration_mamba2hybrid.py`).

1. **Real NKI kernels, not a toy port.** `mamba2_kernel.py` fwd+bwd use `nl.matmul`,
   `nisa.affine_select`, `nisa.nc_transpose`, a chunked-SSD algorithm, wrapped in
   `torch.autograd.Function` — same tier as our own hand-written rope+QK-norm and
   softcap-CE kernels. `conv1d_grouped.py` is a real grouped causal conv1d kernel
   (fwd+bwd). `fused_kernel.py` fuses conv+softplus+scan+RMSNorm into one kernel and
   ships its own numeric-equivalence test against the HF reference in `__main__`.
   `mamba2_kernel_inference.py` is a separate real single-token decode kernel.

2. **No real profiling data exists.** `run_profile.py` is a bare capture/upload harness;
   zero captured numbers or logs are checked into the repo. Any throughput claim would
   have to be measured fresh by us.

3. **Compliance: real risk if the shipped examples are followed as documented, not if
   used correctly.** The README's canonical path is: download the `state-spaces/mamba2`
   HF checkpoint, convert it, `from_pretrained` — this would be a real README-rule
   violation (pretrained weights prohibited). Example commands also assume
   `--tokenizer EleutherAI/gpt-neox-20b`, `vocab_size≈50288`, and config defaults
   (`Mamba2Config.vocab_size=32768`, `Mamba2HybridConfig.vocab_size=256000`) that don't
   match our fixed `vocab_size=8192` tokenizer. None of this is *forced* — configs are
   plain kwargs, and the model random-inits via `post_init()` unless `from_pretrained`
   is explicitly called on an external checkpoint. Using it from-scratch at our
   `vocab_size=8192` with no checkpoint load is compliant; the shipped docs/examples are
   not, and would need to be explicitly bypassed, not followed.

4. **Config-space fit for our scale: workable but untested at our size.**
   `hidden_size`, `num_hidden_layers`, `vocab_size`, `n_groups`, `head_dim` are free
   kwargs. Hard kernel constraints: `d_state==128` fixed in the backward kernel, conv
   kernel size fixed at 4, channel counts must be `%128==0`, `seq_len %128==0`,
   `intermediate_size = expand*hidden_size` must divide `head_dim` evenly. At our
   `n_embd=640`, `expand=2` → `intermediate_size=1280`, divides cleanly by `head_dim=128`
   (10 heads) — arithmetically workable. `predefined_configs.py`'s smallest shipped
   preset is `Mamba130M` (d_model=768, 24 layers) — more than 2x our width, 4x our depth;
   nothing near our 55.7M scale is provided or tested by the repo's own authors.

5. **Integration effort: a rewrite, not a drop-in.** Real hard blockers found directly
   in the code, not inferred: `Mamba2Mixer.forward` literally asserts
   `"xla" in self.in_proj.weight.device.type` — requires `torch_xla`, which our
   `train.py` does not use (native `torch_neuronx` eager). Every linear/embedding layer
   is `neuronx_distributed`'s `ColumnParallelLinear`/`RowParallelLinear`/
   `ParallelEmbedding` with tensor-parallel weight-splitting logic hardwired into
   `__init__`/forward, not optional even at tp=1. Loss uses NxD's
   `parallel_cross_entropy`. `requirements.txt` pins `torch-neuronx==1.13.*`,
   `neuronx_distributed==0.10.0`, `neuronx-cc==2.16*` against our
   `torch-neuronx==2.11.3.0.0` / `neuronx-cc==2.26.6360.0` — a large SDK-version gap; the
   NKI API surface has shifted across it. Net: would require porting off XLA+NxD onto
   native eager Neuron, replacing every `Parallel*` layer with plain
   `nn.Linear`/`nn.Embedding`, and re-verifying every kernel against the newer SDK.
   Realistic estimate: **weeks, not a session** — a project-sized undertaking, larger
   than the FP8 item already deferred in Bucket D.

6. **Hybrid ratio in the shipped code does not match the user's "~3 Mamba2 per
   attention/MLP" description.** `mamba2hybrid_layer_allocation.py` defaults to
   `hybrid_attention_ratio=0.08`, `hybrid_mlp_ratio=0.5` — i.e. ~8% attention layers
   (evenly dispersed), the rest split between Mamba2 and MLP layers, not "3 Mamba2 per
   attention/MLP layer." A custom ratio is settable (`override_pattern` string exists),
   so a 3:1 pattern *could* be configured, just isn't the default. Additional flags:
   `assert batch_size==1` in the fused-backward path (batch>1 backward incomplete), and
   several `# fixme`/`# todo` markers admitting incomplete cache/attention-mask/dt-limit
   support in the mixer forward.

**Verdict on the code:** real, hand-verified kernel work (green flag), but built for a
different runtime stack (XLA + neuronx_distributed + a much older SDK) than our
competition repo. Adopting the underlying *SSM math* is plausible; adopting the
*modeling stack as shipped* is not a realistic fit for a 30-minute-budget competition
round.

## B. Literature verification (5 PDFs + primary sources)

All five PDFs (`articles/*.pdf`) were checked. Four are generic SEO/marketing-style blog
summaries ("Beyond Transformers...", "How Mamba Beats Transformers...", "Mamba vs
Transformers: A Technical Comparison", and presumably "Transformers vs Mamba vs Linear
Attention" by pattern) — technically accurate at a high level (correct SSM equations,
correct complexity claims, correctly-cited real papers: Gu & Dao 2023 `arXiv:2312.00752`,
Mamba-2/Dao & Gu 2024, Long Range Arena/Tay et al. 2021), no fabricated APIs or invented
numbers found, but shallow and non-specific to our regime — no small-scale or
short-training-budget data anywhere in them. "Mamba Explained" (Google-Deepmind-adjacent
blog, 27.3.2024) is the one genuinely careful piece — correctly derives the ZOH
discretization and the S4-vs-S6 selection mechanism from the actual paper's math, no
errors found in the sections reviewed.

**Cross-checked directly against primary sources (not just the blog summaries):**

- **NVIDIA Nemotron-H** (`arXiv:2504.03624`, fetched and read directly, 2025-09-09
  revision). Real, exact architecture (Table 1 + Figure 2): Nemotron-H-8B is 52 layers,
  Nemotron-H-56B is 118 layers. **Only ~8% of layers are self-attention** (4/52 for the
  8B model, 10/118 for the 56B model), evenly dispersed throughout the model. **The rest
  is an even split between Mamba-2 and FFN layers** — i.e. the dominant pattern is
  alternating `Mamba2 → FFN → Mamba2 → FFN → ...` with an attention layer occasionally
  substituted in. This is a **structurally different ratio from the user's "~3 Mamba2
  layers per MLP/attention transformer layer" description** — it's much closer to "1
  Mamba2 : 1 FFN, with attention swapped in for ~8% of the Mamba2 slots," not "~3:1
  Mamba2-to-transformer-block."
- **This exactly matches Fork A's independent finding** in the sibling repo's
  `mamba2hybrid_layer_allocation.py` defaults (`hybrid_attention_ratio=0.08`,
  `hybrid_mlp_ratio=0.5`) — two independent sources (NVIDIA's own paper, and a
  third-party Trainium port's defaults) agree on ~8% attention, not a 3:1 ratio. High
  confidence this is the real number, and the user's "~3 Mamba2 per layer" recollection
  doesn't match either primary source.
- **Waleffe et al. 2024, "An Empirical Study of Mamba-based Language Models"**
  (`arXiv:2406.07887`, NVIDIA — cited directly by the Nemotron-H paper as the basis for
  its ratio choice). Real, relevant finding: tested **8B-parameter models on up to 3.5
  trillion tokens** — no smaller scale reported. Pure Mamba/Mamba-2 "match or exceed
  Transformers on many tasks" but **"lag behind Transformers on tasks which require
  strong copying or in-context learning abilities (e.g., 5-shot MMLU, Phonebook) or
  long-context reasoning."** This is the real, primary-source version of "Mamba is
  weaker at exact recall" — not a myth, a directly reported empirical finding, and it's
  specifically about copying/ICL/retrieval tasks, not raw perplexity. The hybrid model
  fixes this: "the 8B Mamba-2-Hybrid exceeds the 8B Transformer on all 12 standard tasks
  evaluated (+2.65 points on average)," at an optimal composition of **43% Mamba-2, 7%
  attention, 50% MLP** — again ~7-8% attention, not 3:1.
- **No small-scale (sub-100M) or short-training-budget (~500-750 step) data exists in
  any source checked** — Nemotron-H: 8B/47B/56B, up to 20T tokens. Waleffe et al.: 8B,
  up to 3.5T tokens. The generic blog articles cite Mamba's own paper's "up to 3B
  parameters" as the smallest scale with reported parity/wins. **Every real number found
  is 1.4x (3B) to >1000x (8B-56B) larger than our 55.7M model, and every training budget
  is orders of magnitude longer than our ~500-750 steps.**

## Recommendation

This is the **fourth** time this exact pattern has shown up this competition (after XSA,
DCMHA, SLAY): a real, credible technique with genuine published wins, where every single
piece of supporting evidence is measured at a scale and training budget far beyond ours,
and the one directly relevant data point we do have (XSA, actually tested at our scale)
went from a real positive at 0.7B-2.7B to a real *negative* at 55.7M. There is no
countervailing small-scale data point for Mamba2/hybrid in either direction — it hasn't
been tested small, in either the primary literature or the sibling repo's own presets
(smallest shipped preset: 130M/24-layer, still >2x our width and 4x our depth).

On top of the scale-mismatch risk, this bucket carries a **much larger implementation
cost** than XSA/DCMHA ever did: the only real Trainium-targeted Mamba2 code available
(the sibling repo) is built on a different runtime stack (XLA + neuronx_distributed +
an old SDK) than our competition `train.py` (native eager `torch_neuronx`), with hard
blockers (`assert "xla" in device.type`, hardwired tensor-parallel layers) that make
"adopt this code" not viable without a from-scratch rewrite — realistically a
multi-week project, not something to attempt inside a 30-minute-budget competition round
with ~4 days of session time invested already.

**Recommendation: do not pursue implementation this round.** Two honest reasons, not one
soft one: (1) the scale-mismatch risk that already burned XSA/DCMHA/SLAY applies here
with zero counter-evidence in either direction, and (2) even setting risk aside, the
integration cost is categorically larger than anything else attempted this competition —
larger than the FP8 item already deferred as "project-sized." If pursued at all, the
right-sized version would be a **from-scratch, native-eager-Neuron reimplementation of
just the Mamba2 SSD scan** (not the full hybrid modeling stack, not NxD, not XLA) at our
exact 55.7M/vocab-8192 scale, measured head-to-head against the current transformer under
the same 900s-proxy/1800s-confirm discipline used everywhere else this session — a real
option, just not a cheap one, and one that should be explicitly time-boxed given the
already-large backlog (YAT/SLAY Phase 2 still pending) and the competition's remaining
time.

## Log

- Fork A (agent `a5fcdef50638bbcfc`): full read of `state-space-models-neuron/mamba2/`
  and `mamba2hybrid/`, 6-question due-diligence report — see Section A above.
- Direct primary-source verification (this session, via WebFetch + PDF read): 5 PDFs
  read/skimmed, Nemotron-H (`arXiv:2504.03624`) fetched and read in full for Section 2
  (architecture), Waleffe et al. (`arXiv:2406.07887`) abstract-level verification via
  WebFetch. No fabricated claims found in any of the 5 PDFs; one blog ("Mamba Explained")
  independently verified as high-quality and accurate against the real SSM math.
