# Table experiments: exact current gate, 25 September 2026

## Newer human scheduling amendment (25 September IST)

The active [controller brief](CONTROLLER_BRIEF_2026-09-25.md) incorporates the
newer human request to run **A1 on trn21 and A2 on trn22 in parallel** once each
has a valid same-host c5c A0 control and all exact-size G2, native-smoke,
disk, containment and authority gates pass. This supersedes the older
September-24 scheduling recommendation in `DECISION_LOG.md` that a full A2
must wait for an A1 result; it does **not** weaken any safety gate. A2 can be
evaluated as its predefined two-layer table *profile* against same-host A0.
An incremental claim about adding layer 1 still requires a separate same-host
A1 comparison, and neither profile is an adoption without the unchanged
two-host `decide.py` rule and human acknowledgement. No table run is currently
dispatchable.

Researcher read-only reconciliation after the first-place amendment. This is an evidence
inventory and queue recommendation, **not** a table result or launch authorization.

The active research-v2 source `submissions/research_v2/train.py` has SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
The separately accepted hybrid RMSProp ownership repair used a different source SHA
`9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`.
Its native five-step evidence cannot certify the exact c5c table path. The earlier
`35c9...` A1 attempt failed before step 1 at the RMSProp ownership adapter;
that is neither a negative BPB result nor evidence against tables.

The archived c5c table CPU receipts under `workflow/evidence/g2-a1m8-impl1.json` and
`workflow/evidence/g2-a2m8-impl1.json` use table multiplier **8** and stub data.
The independent [source review](../agentic/verification/20260925-upstream-s0-research-v2-static-verifier.md)
requires an exact-size multiplier-64/dim-128 four-rank CPU two-step RMS trace and a
30-step Neuron table smoke before a full A1/A2 experiment. The completed A0/S0 G2
receipts do not include this table argv. No current c5c table arm has a valid fresh
public20M BPB. A1/A2 entries in `research/v2/queue.json` are seeded profiles, not
fully checked runnable jobs.

After human disk expansion and a valid c5c same-host dense control, the next full
one-factor quality arm should be **A1 last-layer bigram+trigram tables** at multiplier
64/dim 128 with RMSProp, conditional on its exact G2 and native smoke gates. A2 adds
layer 1 and should be evaluated as a separate profile against a same-host control;
to claim the incremental effect of layer 1, compare A2 against same-host A1. The
old-stack table gains (roughly 0.015 BPB on last layer) are priors, not predicted
or measured gains on c5c.

trn21 and trn22 are currently disk-blocked; trn22's A0 label failed before a step,
so it did **not** create the required new valid c5c control. Any new attempt needs a
fresh label, reviewed failure diagnosis, intact gates and human breaker decision.
