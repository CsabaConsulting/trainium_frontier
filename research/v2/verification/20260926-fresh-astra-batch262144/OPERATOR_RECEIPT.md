# Fresh Astra total-batch CLI review operation

This is the session operator's evidence archive, **not that operator's independent
source review**. The new reviewer has no implementation or earlier review context.
No source adoption, default change, merge, native execution, speed, quality or
submission authority follows from this review operation.

## Exact packet and new session

- Candidate commit: `4e361d15baef94f041501f558620422c55ae8c88`.
- Candidate source SHA256:
  `a489f7f9e935e4fe9ecd2ad9df5f0f327afcbe9475b71e7fbe0f4e71a48925d9`.
- Base commit: `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`.
- Base source SHA256:
  `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
- Input manifest SHA256:
  `eeda2cb3b99b75d614b8f937cfe5843453f77492fca9240b8abd6e7bca7c83d3`.
- NEW session: `01a0da21-3da3-7f12-a35a-25c0be650a72`, started
  `2026-09-25T19:53:26.511709+00:00`, explicitly selected `gpt-6-astra`, CLI
  `0.155.0-alpha.16.4`. Events do not independently echo a provider model identity.
  This is the authorized fresh same-family substitute, not a cross-family claim.

The 22 supplied files are the full-context source diff, neutral lane specification,
raw full-parameter CPU G2 records plus all four rank JSONs and raw torchrun logs for
the exact reference/default/262144 arm, and the raw default comparison/exit.
No implementation README, tests, hooks, notes, earlier review conclusions, branch,
repository or history is exposed. The supplied full117M-parameter receipts are
synthetic/shimmed CPU evidence, not native microshape or hardware evidence. No
mid-session feedback was sent and no previous model session was reused.

The existing reviewed outer bubblewrap mechanism was reused with a fresh empty
home, non-repository directory and ephemeral CLI. Only read-only OS/CPU libraries
and `/packet`, plus writable `/work`, are mounted. Original home, repository,
credentials, SSH files and process filesystem are absent. Only the fixed
`/proc/self/exe` symlink is present for the CLI. Credentials exist solely in the
CLI runtime environment; shell tools receive an explicit credential-free allowlist.
No key/.env was copied or printed. Provider network is permitted; model tool
network is prohibited by specification, not by a network-layer filter.

Outer isolation remained active while the redundant inner CLI sandbox was
disabled, as previously authorized. Built-in CLI instructions are not inherited
implementation context. Preflight, launch flags, actual commands and terminal
events are archived. This is a new assigned review, not a provider-refusal retry;
no Opus or Fable request was made. Candidate source is read-only.

## Result

Fresh reviewer verdict: **PASS, source/CPU-only** for this narrow CLI exposure in
the intended configuration. No blocking regression was found. The session finished
at `2026-09-25T19:59:28.740969+00:00`, elapsed `362.2292640449996 s`, operation exit0.
The test passed on its first run; no criterion was relaxed and no model refusal
or numerical test was retried. The source candidate was not edited.

The reviewer independently reconstructed both exact sources, audited all 22 input
hashes, and executed its own source/reference program. Removing only the inserted
CLI statements makes the candidate AST identical to the base. Actual production
integer arithmetic gives `1 * 32 * 1024 * 4 = 131072` global tokens per microbatch:
default131072 uses accumulation1 and262144 uses accumulation2. With the intended
batch-ramp OFF setting, this introduces no automatic LR scaling.

## Actual independent tiny CPU evidence

`reference_test.py` was authored inside the fresh session. Its recorded child
command ran `19:56:18.516721`–`19:56:20.657409` UTC, exit0: **352 checks / 11 cases**.
The test used Python3.14.4/PyTorch2.13.0+cpu, seed42, deterministic algorithms, all
thread counts1, width7/vocabulary11/context3/microbatch2 and165 parameters per toy
model. Four serial replicas use mocked deterministic AVG, a no-op debug hook and
transparent momentum-SGD updates; no full GPT, BF16/NKI/native path, real process
group, production optimizer ownership, dataset, timing or checkpoint load ran.

The reviewer executed AST-extracted argument, token, fetch, zero/accumulate,
unscale, clip/sync and update statements plus unchanged CE/schedule helpers.
Base default, candidate omitted argument and explicit131072 have byte-identical
FP32/FP64 three-step histories, including losses, gradients, clipped gradients,
parameters, momentum, fetch counts and token counts. This default identity has
no tolerance. Loss-scale8/unscale likewise matches scale1 exactly. Accumulation2
intentionally differs in final weights from the baseline.

Accumulation1/2 and a FP64 accumulation3 case were separately compared with an
independent token-summed logsumexp/global-mean gradient/update reference. This
differently ordered reference is **not called bit-identical**. Criteria were
declared before execution: FP64 atol1e-12/rtol1e-10 and FP32 atol3e-6/rtol3e-5.

| dtype | Largest gradient absolute error | Largest parameter absolute error |
| --- | ---: | ---: |
| FP64 | 2.7755575615628914e-17 | 5.551115123125783e-17 |
| FP32 | 1.4901161193847656e-8 | 1.4901161193847656e-8 |

The largest loss errors were4.440892098500626e-16 and2.384185791015625e-7,
respectively. Full per-step numerical values, errors, criteria, hashes, command
timestamps and warnings are preserved unchanged. No binary tensor artifacts were
generated; the complete numerical histories are JSON. The operation agent did not
rerun, implement or supply the fresh reviewer's test.

## Supplied receipt scope and preserved caveats

The reviewer checked actual fields rather than accepting PASS labels. All twelve
standalone rank JSONs match their embedded records and all three log tails match
the raw logs. Base/default's twelve micro-loss values and62 rank0 final digest
strings match. The262144 arm has six instead of three micro-losses per rank and
62 different final digests, consistent with the intentionally different batch.
Ownership totals117453580 parameters and has the same plan across modes.

These supplied full-parameter receipts nevertheless use synthetic2x256 forwards,
forced CPU BF16, skipped native CE and fake time. They do not establish native
production microshape, true timing or cold/budget compliance. Non-rank0 final
digest maps, initial tensors, checkpoint binaries and supplied harness source are
absent; checkpoint fields are audited records, not independent checkpoint loads.

Existing source caveats are explicitly retained: logged loss is the smoothed
rank0 last-microbatch loss, not an update/global mean; batch-ramp ON can override
accumulation and scale LR; optional byte-weight means are separately normalized;
weights-only load is warm start, not exact optimizer/RNG/loader resume. Divisibility
is correctly rejected but checked late and has reversed wording in its existing
error message. No unrelated correction was invented or applied.

## Boundaries and support hashes

The review explicitly REFUSES native correctness, compilation/startup, >=5% speed,
quality/adoption, changed default, full run, merge/staging/hardware authority and
independent verification of absent supplied checkpoints. The prospective105-step
diagnostic and its driver are not reviewed or launched by this source archive.
Only the sole Operator owns any separately assigned launch under common lock and
fresh process/device idle checks. No queue, ledger or new launcher was created.

The CPU slot was released after the fresh session exited; no child processes
remain. Final archive validation is stdlib hash/structure checking only. Original
reviewer `/work/` paths map to `session/reviewer-output/`, copied unchanged.

| Reviewer artifact | SHA256 |
| --- | --- |
| `REVIEW.md` | `45135385d5684213b8e67a15e6330918bfcb588511c7a3f0a69f52aeb747ea74` |
| `result.json` | `ad35c9f079ccd5328901e270f5bd820ca2366b8f097c0eba0658af3d50ce6852` |
| `reference_test.py` | `9f51fa844cf85660afa3138956037b202cf63e846d2860a96397c252d7a5c8af` |
| `reference_results.json` | `bac6fe83a370e496713aa436bbe1af0ad91680c5293a7d502c778e8b05dbd4d5` |

Returned usage counters: input826620, cached-input730300, cache-write-input96278,
output20962, reasoning-output2834 tokens. No cost estimate or unobserved provider
identity is inferred from those counters.
