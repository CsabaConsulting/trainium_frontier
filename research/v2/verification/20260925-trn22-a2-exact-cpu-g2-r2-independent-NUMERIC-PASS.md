# Independent numeric review: trn22 A2 exact-size CPU G2 r2 — PASS

**Scope and decision.** The retrieved, four-rank, two-step CPU/Gloo G2 result passes the
exact A2 numeric acceptance rule in
`research/v2/g2-probes/20260925-c5c-table-exact-g2-minimal/PACKETS.HOLD.json`.
This is permission to consider a *separately authorized* Neuron smoke. It is not
a Neuron result, public BPB, full-run approval, adoption, or submission approval.
I did not access a host, launch a job, or edit the candidate.

## Exact evidence reviewed

- Controller artifact commit: `00e1e9f4` (subsequent controller commits leave these
  raw files intact); raw directory:
  `research/v2/operator/20260925-trn22-a2-exact-g2-r2/final/`.
- One-use attempt has assignment SHA-256
  `196075917e95edb664704d7fbbbd82a1242c50c12751eee9fcf4bfb2b22ca7fd`,
  exact c5c source SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
  G2 tool SHA-256
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`,
  trusted `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
  seed 42 and the packet's ordered A2 arguments. The checked source and tool bytes
  in the repository match those pins. `g2.json` SHA-256 is
  `77037d646a865c8d2df910f70c409984cf94656046f8b43c8e96f7a76b5a29a1`;
  raw rank JSON SHA-256 values in rank order are
  `f2acb93d5bdab9b6539f971aae1e72344444f368aca83aa9fa5da10f182e778a`,
  `195d752cc3f019253f36fd8b1670836f9dda46bb2b0ce9ac488bcd14d3f275df`,
  `eb35d5c6ec47accb643dce370abb80623b26ca12136d42c8b4f55d68ce8f1b26ca`,
  `7d0a5a427f1c780084517fab64f28fe62e5fb795e22d8e36eb65dbd75e061ee7`.
- `g2.json` reports `verdict=PASS`, `torchrun_rc=0`, `timed_out=false`,
  `nproc=4`, `steps=2`. The separate launcher `exitcode` is `0`. The aggregate's
  four embedded rank records exactly equal the four raw rank JSON objects.
- Raw ranks are exactly 0, 1, 2, 3; all say `PASS` and each has exactly two
  finite losses. All share ownership-plan SHA-256
  `7cfeabba0d5ade8cebc24b103ad1297e190ad9c3a365edacba68e5c23a4937a1`.
  Each reports 74/74 parameters changed, no unchanged tables, and no cross-rank
  parameter mismatch. Rank 0's saved final digests cover all 74 parameters.
- All four raw `tables` lists are precisely `ngram_embeds.b1.weight`,
  `ngram_embeds.t1.weight`, `ngram_embeds.b5.weight`,
  `ngram_embeds.t5.weight`. The source inserts them in this ModuleDict order,
  and its sole RMSProp table group takes `list(self.ngram_embeds.parameters())`.
  The frozen tracer walks that group in order. Thus each rank's eight unnamed
  checks attribute to `b1,t1,b5,t5,b1,t1,b5,t5`, not merely to an aggregate
  table-change claim.
- Each rank has exactly two RMSProp calls, eight checks, and step sequence
  `1,1,1,1,2,2,2,2`. Every check has `numel=67108864` and 2,042–2,045 touched
  rows (required 1–524,288). Every untouched-row bit is true. All touched
  gradient medians and normalized errors are finite; worst
  `max_abs_err_over_lr=0.02045185503119651`, below the 0.03 limit.
  The complete trace is identical on all four ranks.
- Rank 0 reports checkpoint exists, strict `load_for_eval` succeeds, logits
  shape `[1,256,8192]`, finite deterministic repeated logits. Exact c5c
  `load_for_eval` uses `torch.load(..., weights_only=True)` and
  `load_state_dict(..., strict=True)`. The retrieved model-only checkpoint
  is 1,008,809,893 bytes and independently rehashes to
  `1f3e95b40aba32b468fcff7612ab85302fafcaed139b38995622db8fe9a7e6d4`,
  matching the Operator's host-side hash. This review did not re-run the
  checkpoint load; it independently checked the frozen tool's load procedure,
  raw rank-0 check, and retrieved checkpoint hash.

**Verdict: PASS for this exact CPU G2 gate only.** A Neuron compilation/run and
fresh-process public evaluation remain unproven. Keep the raw artifacts and
require their own separate review before any BPB claim.
