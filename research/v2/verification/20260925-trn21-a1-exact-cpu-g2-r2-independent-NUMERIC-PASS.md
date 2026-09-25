# Independent numeric review: trn21 A1 exact-size CPU G2 r2 — PASS

The retrieved four-rank, two-step CPU/Gloo evidence satisfies the exact A1
numeric rule in `research/v2/g2-probes/20260925-c5c-table-exact-g2-minimal/PACKETS.HOLD.json`.
This verdict covers CPU correctness evidence only. It proves no Neuron result,
throughput gain, BPB improvement, adoption, or submission readiness. The current
Final Push directive separately governs flag-only host launches. I did not
access a host, launch a job, or edit the candidate.

## Evidence and independent checks

- Operator artifact commit: `4030f4b748583df4afd004ecf179610340c34648`
  (controller copy `246285b51f3a9db8e786650477a84530c2d7a72d`). Raw directory:
  `research/v2/observations/20260925-trn21-a1-g2-r2/raw-host-root/`.
- Exact candidate SHA-256:
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
  Exact G2 tool SHA-256:
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`.
  Both independently match repository and retrieved bytes. The recorded trusted
  `prepare.py` hash is
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
  with `stub=false`. The ordered A1 argv exactly equals the packet: last-layer
  bigram plus trigram, multiplier 64, dimension 128, BF16 RMSProp, matrix LR 0.015.
  The launch environment records seed 42. Label:
  `fs-20260925-c5c-a1-exact-g2-trn21-r2`.
- All 14 entries in the Operator's `RAW_SHA256SUMS.txt` independently pass in
  the original retrieval worktree, including source, tool, four raw ranks and
  the model-only checkpoint. The aggregate JSON SHA-256 is
  `d2d280661297101495e1ee62652258aefba268d3b20d3b324921ce595984c01f`.
  Raw rank hashes in rank order are
  `8c4e8797d1891749e4ecb1304c7d48ac403371add68d67a66f936809221d9884`,
  `2b69f6cd0d1c51c3d35456d523a21ce74c4ed6e02d921731e6b27bd595c3c7c3`,
  `a1e06f539553a44452bf6b20914c030110614b6213dcdeb12d2ff23e683848c9`,
  `82618fa4fa341291d7551bd570fce284d7d940383a989bc35c527c1268b75269`.
- Independent JSON assertions passed: aggregate `PASS`, `torchrun_rc=0`,
  `timed_out=false`, four ranks, two steps; embedded ranks exactly equal the
  four raw rank objects. The detached monitor records exit zero and no disk
  floor breach. Raw ranks are exactly 0–3 and each has two finite losses.
- All ranks share ownership-plan SHA-256
  `82ad32341246717dbcca790065f1724fc70e40ee014130167e6faff19dfcaad4`
  and identical plan summaries. Each reports 68/68 parameters changed, no
  unchanged tables and no cross-rank parameter mismatch. Rank 0's final
  parameter digest map covers all 68 parameters.
- A1 has exactly **two** tables, `ngram_embeds.b5.weight` and
  `ngram_embeds.t5.weight`; no layer-1 tables. The source inserts bigram before
  trigram in its ordered `ModuleDict`, the sole table RMSProp group takes
  `list(self.ngram_embeds.parameters())`, and the frozen tracer iterates that
  group's parameters in order. Therefore each rank's four checks are exactly
  `b5,t5,b5,t5`, with step counters `1,1,2,2`, rather than A2's eight checks.
- Each rank has two RMSProp calls and four identical ordered traces. Every
  table has 67,108,864 elements, with 2,042–2,045 touched rows (allowed
  1–524,288). Untouched rows remain bit-identical in every check; all gradient
  medians and normalized update errors are finite. The worst
  `max_abs_err_over_lr` is **0.017789718444982852**, below 0.03.
- Rank 0's saved-checkpoint test reports strict load succeeds, deterministic
  finite logits, and shape `[1,256,8192]`. Inspection of the exact source and
  frozen tool confirms `weights_only=True`, `load_state_dict(..., strict=True)`,
  and repeated-logit equality/finiteness checks. I did not rerun the load.
  I independently hashed the retrieved 739,322,371-byte checkpoint at
  `/home/poojith-devan/Desktop/Trainium-trn21-s0-operator-preflight/research/v2/observations/20260925-trn21-a1-g2-r2/raw-host-root/tmp/g2-sg1ufc20/out/final.pt`:
  `c548b0863bb577de90c5e3efef3eecf4006ef7951561417f5b2189fcf3493a67`,
  matching the host/Operator manifest.

**Verdict: PASS for this exact CPU G2 gate.** The CPU synthetic micro-batch
and fake timing do not establish hardware runtime or competition performance.
Preserve and independently review the later Neuron run and fresh-process
public evaluation before using its result.
