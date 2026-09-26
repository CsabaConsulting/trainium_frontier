# Independent review: disabled bcbd full fallback

Disposition: **PASS_DISABLED_BINDING_MECHANICS_ONLY**. Reviewed frozen packet `7da2559f03cfda04af3a7858e1b2d4037f8b676b`, following proposal `43f721b2a0245e4af8d641696c10a9640ea2343a`. No launch, host reservation, source adoption, submission, new quality result or cold-full claim is granted. P/K2 retain priority. No host calls, tensors, candidate edits or provider calls occurred.

## Exact identities and contrast

- Source `b635c8b0cf2a10e1bcdd983e86192672ed6ad56f:workflow/proposals/research-v2/train.py`, SHA256 `bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522`: independently removing the four cache-refresh additions restores exact adopted 6ef7/a4aa source bytes. This is the 117,453,580-parameter six-layer baseline, not architecture 3e79.
- Driver SHA256 `7ae7f8f05c623a233d5700a7b56c1d54480373153d4fee1087f24d427c9f714b`. The **entire** driver equals the proven `2a36aa46` predecessor after only label/root, source hash and descriptive docstring substitutions. There is no hidden control-flow change.
- Binding SHA256 `bc492cf2fcf863a615835008dd7598cf03b58fa2f5c157661a2e5022434d6909`; manifest `67c41b49a435f4a0aafd198caae450d475ac3e4de52013b2a15bbadac0c19881`. All eleven manifest entries authenticate.
- Scientific command equals the same-host 6ef7 comparison command plus only `--initial-cache-refresh`. Recursive schedule was already ON; seed42, matrix LR0.015, width/context1024, pack32, full four ranks, MAX1800 and other graph flags are unchanged. No architecture, FP8, startup-adjusted schedule or batch change is introduced.

## Executed independent checks

`audit.py` ran under normal Python and `-O`: **196 checks each, identical result JSON**. It independently checks frozen pins, full-driver equality, actual raw evidence, and synthetic refusals, rather than relying on the author's PASS labels. Own guard fixtures exercise idle success, a Go/neuron-ls command failure, unknown/empty device schema, and a live torchrun process; both idle commands execute and every unsafe case refuses. The exact extracted runtime budget predicate rejects failed exits, false/missing acceptance, and integer `1` masquerading as boolean `True`. Runtime is pinned to normal Python; the extracted assertions explicitly compile with optimization disabled even when the audit runs under `-O`.

Author tests were also replayed: eight groups each, normal/`-O`, exit0. Two initial mistakes in **my audit fixture** incorrectly expected rank0-only checkpoint/digest fields on ranks1–3; both failures are retained, corrected only in the audit, and are not candidate failures.

Reproduce from this worktree:

```
python3 -I -B research/v2/verification/20260926-basebcbd-full-binding-independent/audit.py
python3 -I -B -O research/v2/verification/20260926-basebcbd-full-binding-independent/audit.py
```

## Actual prior evidence and its limits

All **88** raw full-parameter CPU files from `2718bc5c` were rehashed with sizes. Candidate default/OFF/ON each match reference on all four ranks' three recorded losses and ownership; rank0's all62 final digests and checkpoint reload pass, and all four retained checkpoint hashes/sizes agree. These are standard G2 CPU BF16, synthetic micro2x256, fixed-clock three-update checks—not full production token batches, native compilation or native first-five LR equivalence.

The exact prior trn22 cold receipt (`de21fab0…`) and launch pin authenticate: five steps, exit0, six fresh empty SDK cache roots; uncapped fifth **493.3377903739456s**, process wall **496.8015910499962s**, charge **3.463800676050596s**. Target600 passed on **trn22**, not proof of a trn21 cold run or of this full run. The new full binding deliberately retains existing train/eval caches; it cannot be called cold or a proved cache hit.

## Safety/accounting mechanics and remaining Operator obligations

Unchanged common guard/dispatcher, protected prepare (`5807c46e…`) and rehearsal (`7a55cb87…`) are pinned. The sole outer flock uses `/home/ubuntu/.cache/trainium_freshstart/device0.lock`; train and fresh eval each require fresh ps plus neuron-ls under that inherited live lock, and both immediate children receive the authenticated FD. Direct invocation cannot authenticate the outer parent. Stage/launch gates remain false. STOP checks, fresh output paths, source/prepare/harness hashes and 10.5GiB disk headroom are retained.

Training is capped by timeout2730+60s; its official observer charges process launch through exit minus `min(actual fifth boundary,900)`, and requires <=1800 with zero exit and an existing checkpoint before eval. Eval is a separate process using unchanged prepare, public20M=20,971,520 tokens, context1024/B1, internal1200 and outer1800+60s. No automatic successor is present. Final finite score, evaluator/checkpoint identities and complete raw receipts still need result acceptance; driver COMPLETE_PENDING_INDEPENDENT_REVIEW is not a quality PASS.

As in its predecessor, exact hostname, original trn21 lock device66305/inode596823, old-queue retirement, and explicit PID/PPID/PGID/SID/compiler/device/lock postflight on **every** outcome are Operator obligations, not newly implemented driver features. Immediate-child FD inheritance is not proof that arbitrary detached compilers retain it; no reuse is permitted until explicit quiescence is established. No second launcher or reserved host is created here.

## Comparator provenance

The current ledger contains two distinct same-source/same-argv trn21 observations: earlier `fs-20260925-l3-timer-dense-trn21-r1-full` at **0.9900560897366009**, kind **schedule_arm**; later `fs-20260925-l3-base-repeat-trn21-r2-full` at **0.9899528600562257**, kind **control**. The packet's earlier historical point is not promoted retrospectively into a control, and the genuine later control is not denied. Any future selector evaluation must retain the actual row kinds and unchanged selector rules. This receipt neither edits the ledger nor chooses a new adoption threshold.
