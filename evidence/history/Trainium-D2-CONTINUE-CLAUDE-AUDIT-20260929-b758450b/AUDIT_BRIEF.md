# D2 CONTINUE — READY FOR CLAUDE AUDIT; NOT APPROVED FOR UPLOAD

Both qualifying cold results are now collected. This is the audit package,
not Claude acceptance, upload approval or a new research gate. Unknown checklist evidence is FAIL
(pending evidence), never presumed PASS. Frozen scored bytes are not cleaned up.

## Exact source and command

`train.py` SHA256:
`b758450b599d4acc5b1fc17d37cccb87421d22bb99c213f6d8ed29fdede13fa8`

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

CONTINUE is selected. Flag0 is mandatory for this recipe; the same frozen file
defaults to FREEZE286000000 without it. Both runs use hardcoded seed42/every2,
FP32 duplicate-gradient sink, BF16 table/RMS state. The human seed42 clarification
supersedes the old seed43 request. There is no --seed option; no wrapper/env
monkeypatch is permitted. Same-seed confirmation is not across-seed robustness.

## Actual cold/score evidence

| Run | Host/seed/flag | BPB | Raw cold(s) | Charged(s) | Archive SHA256 |
|---|---|---|---|---|---|
| first | trn21 /42 /0 | 0.9726966326753679 | 645.214926446 | 1789.246907866 | `5a82d3568173b16eb73d74e5b1484e9835c2b519181c1c68166bee4ecdc6da72` |
| repeat | trn22 /42 /0 | 0.9724059578084576 | 676.220380932 | 1789.974079438 | `88acaeb5597adced2408f6e38fa840f3489ed48579695e97ea9c502e1ffe1fd8` |


First CONTINUE/trn21: BPB0.9726966326753679 over20971520 public tokens/context1024;
raw fifth-boundary cold645.2149264460895s; charged-through-exit1789.2469078660943s;
2562steps. Checkpoint588313401bytes,
SHAffd8d27b1e4d50ce5c6da65ab4fdc9884fa5395beb73240fceb1c01d8bf6ec27.
Archive5a82d3568173b16eb73d74e5b1484e9835c2b519181c1c68166bee4ecdc6da72.
Repeat CONTINUE22 archive is authenticated in the table above. Closed
FREEZE0.9761396502 is not a qualifying repeat. Both CONTINUE scores<=.9738 trigger
audit assembly, not automatic READY, acceptance or upload.

## Eight-item rejection checklist

S:L means line L of exact b758. These are assembled evidence statuses, not a
fresh independent or Claude verdict. Research review has narrower authority
than submission compliance. Exact input hashes are in DIFF_MANIFEST/PINNED_INPUTS.

| # | Check | PASS/FAIL | Evidence and limitation |
|---|---|---|---|
| 1 | Only train.py; exact command; clean dependencies | FAIL | Actual isolated full/eval used train.py plus supplied prepare.py. Research imports succeed; organizer clean-box equivalence remains unproved. S:36–44,180–182,446,471 and first archive isolated-submission log. No install proposed. |
| 2 | Strict trained load; shared causal train/eval | PASS | S:987–1052 shares model-owned lookup/core; S:1492–1514 saves full state and strict-loads before eval. Actual D2 fresh20M used its checkpoint. CPU14 covers learned checkpoint/causality; research ACCEPT covers shared path. Training compile/cached casts versus eager eval/shape dispatch are disclosed: NOT full D2 native FP32 parity or bit identity across shapes. |
| 3 | No validation/external/network/local-cache dependence | FAIL | Source precheck finds no candidate validation/network/pretrained path; S:1576–1608 uses train data. Provided tokenizer setup has an optional supplied URL; no minimal-runtime/no-network trace is claimed. Empty training caches are measured, not universal environment independence. |
| 4 | Empty-cache raw cold<=900; charged training<=1800 | PASS | First PASS: six created-empty private training caches and actual raw/charged receipts. Repeat PASS, actual receipt authenticated. Raw fifth boundary, not capped allowance. Eval reused declared evaluation cache; no cold-eval claim. |
| 5 | Total bytes written; no large leftovers | FAIL | Cumulative writes NOT MEASURED. First retained checkpoint588313401bytes and retrieved text sizes are not total writes or complete retained-cache census. Host free space is not per-run I/O. No zero-leftover/cleanup claim. |
| 6 | Required env only; features default ON | FAIL | Features default ON(S:1054–1056), but CONTINUE requires explicit flag0(S:1567–1577). Optional compiler/cache defaults S:36–39; research wrappers supply extra environment. Minimal clean-box necessity and human upload-command acceptance remain unproved. |
| 7 | Fixed seed; only declared time-based behavior | PASS | S:120–122,1570 seed42 and actual packet bindings. Research review records unchanged schedules/budget safety and synchronized control; flag0 never freezes. Not a cross-seed/bitwise-determinism claim. |
| 8 | CLEAN/readable source without dead paths | FAIL | Known drift: S:20–21 says output directory is the only CLI, but freeze CLI exists. S:11–15 broadly says FP32 optimizer state; table/RMS state is BF16. Unreachable injection alternatives S:733,751,757,841 remain (validation S:853–854). Frozen bytes preserved, no silent cleanup. |

## Diff and correctness lineage

DIFF_MANIFEST pins D1-parent and uploaded-b585 diffs and sources. D1 parent is
acfb3aefab0e6503498aca8387db63fff583930848fb16fcc9650887a57129ea.
D1 production-reference archive
1e0f075a18d92131cd1674d72121b6f9742b9bf79b584146b90ecb3503c23076
is lineage ONLY: D1 is524288x128; D2 is262144x256/two sites. It is NOT D2 native
production parity. D2 CPU14 archive
af2fec9279cc9d8ece735cfbe5dc30d453e54350cfc8e44bb5ce486a19a15e29
uses reduced transformer/full table geometry. Research-source review
b0af79a605172b5bf28e375fab47a375c52513e70a1d4972776ba955d009efc8
accepts research FULL, not submission compliance. These disclosures add no
new numerical experiment or research launch gate.

## Human boundary

Claude verdict PENDING; human final command/upload approval PENDING. Root exported this new Downloads audit folder after actual qualification.
READY means ready for Claude to review, not compliance acceptance or permission
to upload. The five FAIL/pending-evidence rows above remain deliberately visible. Checkpoints/TRAIN tensors remain on hosts; metadata
contains checkpoint identities only.
