# Research-only warm A2 repeat: queue resolution published

The newest direct human instruction and Supervisor selection at `3d9f7946`
authorize the exact c5c A2 profile again under the new label
`fs-20260925-v2-c5c-a2-m64d128-trn22-r2-warm-full`. This is **RESEARCH_ONLY**;
it is not a submission replay or a claim that startup is fixed. The five-step
r1 is ineligible for a table-quality conclusion.

The Operator copied the completed proposal and immutable resolution manifest
to the existing queue's private directory, verified both host SHA-256 values,
validated schema and unchanged bootstrap host/source/evaluator/cache/STOP
contract, then atomically published the previously absent inbox pointer at
**2026-09-25 16:00:23.363 UTC / 21:30:23 IST**. Existing queue PID **3050509**
loaded it at **16:00:24.099 UTC**. No second launcher was started.

- Manifest path:
  `/home/ubuntu/trainium_finalpush_queue_20260925_trn22_wait_r1/resolved-a2-warm-r2.json`
- Manifest SHA-256:
  `8a43595eb59ad7ddc99f60f542f751f100d21adf1cccc94b4166345b8dac6aa4`
- Proposal path: the same root's
  `proposals/fs-20260925-v2-c5c-a2-m64d128-trn22-r2-warm-full/EXPERIMENT_PROPOSAL.json`
- Proposal SHA-256:
  `44f105107fe7ec543b5f0874626ad51321985c32ed95ae7a9a8059eccacfe3f4`

All local breakers/STOPs and host STOPs were clear; free disk remained about
51.52GB. Exact c5c source and trusted prepare hashes match. The full-chip
four-rank LNC2, seed42, 1800-second profile remains unchanged. Existing warm
train/eval NEFF/HLO and other cache roots are reused without SDK environment
switches, timer changes, deletions or cache resets.

At the independent follow-up observation **16:01:48 UTC**, r1 fresh evaluation
PID3048593 was still active, its exit code absent, and r2 had not created an
attempt directory. The queue was correctly waiting for the common device
lock. This receipt proves resolution activation, not training start; a later
handoff receipt must identify the actual r2 process and launch time.

Cache counts at that observation: train NEFF251, local252, HLO633; eval
NEFF51, local51, HLO153. Both compiler directories contained zero files.
These populated paths establish reuse configuration, not measured cache hits.
The machine-readable receipt is `a2-warm-r2-resolution.json`.
