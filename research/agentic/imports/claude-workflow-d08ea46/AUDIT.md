# Audit: private workflow d08ea46 through 752ea5e

## Provenance

- Source URL:
  <https://github.com/poojithdevan4D/trainium-research-context/tree/claude/trainium-frontier-research-nkp5zd/workflow>
- Initial source commit: `d08ea46dd8135a6ceafa1b0856e5a2f0fc1fe474`
- Fetched source tip: `752ea5e00d020e1f2e601ee633448bf5ea52f522`
- Added commits reviewed read-only:
  - S0: `c30422e2a9f0ed7203c85597eef1cb78c9480b62`
  - research-v2: `dd4c05002ef269a0fae292ae90bf60cda5182104`
  - run_job: `752ea5e00d020e1f2e601ee633448bf5ea52f522`
- Audit date: 2026-09-24
- Canonical integration base: `2c086929483de88095d0c627e572fdf4af9a1f72`
- Audit method: inspect immutable Git objects from the read-only private checkout; do not
  execute or copy its tools, candidate files, patches, reports, evidence, queue, or state.

The private repository is evidence only. Its branch, commits, reports, and claims have no
authority over the canonical repository.

## Reviewed identities

### Initial d08ea46 import

| Private source path | SHA-256 |
|---|---|
| `workflow/AGENTS.md` | `d5823b32a299109c18b5fbe4bf46e96879628caee8e02138e56005b8a98dc40f` |
| `workflow/START_HERE.md` | `adc197c3bdf6c7eaa0de7044d71b282173c7812e59a3db490b504c74b92eafc0` |
| `workflow/tools/decide.py` | `e4288ab327ba3a761d5c9ca46d39f9b874bce55aa218331d31b47c1e54489ace` |
| `workflow/tools/preflight_dryrun.py` | `0266a67a315f3f454e9e183d3659de2aaf541f36faecaa80e196af259f4b57fd` |
| `workflow/tools/contained.py` | `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d` |
| `workflow/templates/proposal.json` | `e20f8ef545dd6fed9e8086f0e2c252df9d765a2a972b2accbbbf5eab888712a4` |
| `workflow/roles/supervisor.md` | `4729628c73ddb8b67da9b515aadb507d57954f4f6850cce80ee1b93a48788873` |
| `workflow/roles/researcher.md` | `701c746f26c7af1f02beae3433b1ff8a7605217aa503770f0fc4866792dbd610` |
| `workflow/roles/implementer.md` | `2b6fb448d24f67432a3a257124117c7cecdb669474c2bd4c99ba3d5bb22cb4e6` |
| `workflow/roles/verifier.md` | `918dfd80880d2a92958a30a8fca60d7eed220db6e0c61f43890df69d9578073e` |
| `workflow/roles/operator.md` | `1dbe2d168573bb568941283c526ac9e7942f5d7e533c5ef157dbaed196f6786d` |
| `workflow/roles/analyst.md` | `275500a474df628faf95ed6354110cd18b540a6fb76d0d66b3460e63c2e2d7f2` |
| `workflow/roles/porter.md` | `89ffb0c6fb5b3e113bdcf71f892b204c8e3b557f100af988caeb5b8895d1f783` |
| `reports/2026-09-24-queue-v2.json` | `2edf0940524113aad52a34c1b528629985f4d243a219e9e9f76df31c2657205a` |
| `workflow/proposals/IMPL-1/impl1.patch` | `f8787b6633d4c9cd31adf58c04413049e8e270774072201e9e234cdc8569a2ee` |

### c30422e S0

| Private source path | SHA-256 |
|---|---|
| `workflow/candidates/S0/EVIDENCE.md` | `6a8e962916059e3069d5a7d398674d09bda368f8eed2dd03a800cd394e2fef4d` |
| `workflow/candidates/S0/SHA256SUMS` | `ba18bef9e1d5a0961c0a694cf890b638e966112065872c422300da12ac0d8d0e` |
| `workflow/candidates/S0/launch-command.txt` | `61a3348a6dc5795ebd8f1a0da90262b19a9a33ba4cb941c09f7c7aa5828fc7db` |
| `workflow/candidates/S0/make_s0.py` | `23e65aed86b8387b37eb13d2f1436c0a28ad6f1caf7742ee2274b37e16fb722b` |
| `workflow/candidates/S0/train.py` | `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` |
| `workflow/evidence/lockstep3-116188-vs-S0.json` | `abc898449fba338032c9a7c77d3ea1933b1309ad69bb29f0701805df91d78fa4` |

### dd4c050 research-v2

| Private source path | SHA-256 |
|---|---|
| `workflow/proposals/research-v2/README.md` | `d0587b587bd8d4af8bbb55ada755a3a6b1f3fb737bafc04b279024c92e48798c` |
| `workflow/proposals/research-v2/impl2.patch` | `c6b0a3419d8ab0badfb84db1e7fcd651d73c65b54eb24db7a9faf7d4ec523645` |
| `workflow/proposals/research-v2/impl3.patch` | `f361d954a48b108601122e1fd62cf641d5959c0c34b4fca3d40154180a3c1160` |
| `workflow/proposals/research-v2/make_v2.py` | `a830420451bbf951fe5594d5655fe212cdd508d753a5ca3998a4c874a5d54fd5` |
| `workflow/proposals/research-v2/train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `reports/2026-09-24-queue-v2.json` | `62a4d84353556b499854c523ae5c5ca6285b1945ef0808e77a1503314c4c9bd8` |

### 752ea5e run_job

| Private source path | SHA-256 |
|---|---|
| `workflow/tools/run_job.py` | `3c225d9acb7a835939eeda36588851c6ff062df8adb4ab8e6dacc529caf642f1` |
| `workflow/tools/decide.py` | `a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66` |
| `workflow/templates/host.json` | `ae681a79965a3fd1949ce587cee18dddc8ce9dddcd45d4ddc727a4a3f5095019` |
| `workflow/templates/job.json` | `6b9baed323f84063fdfd92484f36e1b03280d290a21365373ad86d64af55604d` |
| `workflow/tools/test_run_job.sh` | `6aef35f18913050f34d4bb162e7eaddc9838e70ed5342d11ce1298d72e109b30` |
| `workflow/AGENTS.md` | `b2efea1e5c4ced897be593c20fbab7bc20707f264caa523a0cd1ca1628e3ed37` |
| `workflow/START_HERE.md` | `c27e54fafc568491495440d8ec044788d4a4f59535a8851a8f8b661a4dbca25f` |

## Commit-level dispositions

### c30422e S0 — PATCH / HOLD

- **ADOPT:** the review-clean S0 specialization is a plausible cleanup direction worth
  evaluating under the canonical protocol.
- **PATCH:** it is not a candidate on the private CPU lockstep claim. Before candidacy it needs
  independently verified Neuron lockstep, two cold comparable public20M runs, and reviewer
  verification. The canonical `116188cf` source remains current while those gates are absent.
- **REJECT:** importing the private S0 candidate bytes, evidence, or conclusions as canonical.

### dd4c050 research-v2 — ADOPT HYPOTHESIS / PATCH / REJECT IMPLEMENTATION

- **ADOPT:** fresh-tail is the only new hypothesis in this commit worth carrying into a future
  canonical Researcher proposal.
- **PATCH:** keep the later table-freeze/epoch ablation frozen until its prerequisites are
  established, then evaluate it as its own one-factor proposal through the canonical workflow.
- **REJECT:** the private IMPL-1/research-v2 implementation, its bundled changes, evidence,
  queue, and its A2-before-A1 ordering. Canonical A1 dependency and verification gates remain
  controlling.

### 752ea5e run_job — REJECT

Reject `run_job.py`, its modified `decide.py`, templates, and tests. The implementation lacks:

- an externally authorized, digest-bound request and exact schema;
- a durable host lock and exact one-run-per-host lease;
- no-follow safe-root path confinement for labels, worktrees, outputs, archives, and deletes;
- a fail-closed causality gate (`None` can pass);
- fail-closed escaped-descendant handling (the test treats escape as an anomaly rather than a
  run-invalidating gate); and
- safe archive/delete ordering with immutable evidence, verified durable archive identity, and
  deletion receipts before source checkpoint removal.

These are authority and evidence-integrity failures, not local patches to import. The canonical
controller, Operator protocol, and `selection.py` remain authoritative.

## Consolidated disposition

### ADOPT

- Explicit deterministic lifecycle documentation and strict separation between Researcher,
  Implementer, Verifier, Operator, Analyst, and Supervisor.
- Research ranking inputs: probability of improvement, estimated host-hours, concrete Trainium
  feasibility, cheapest falsifier, and durable prior-result references.
- Treat review-clean porting as an explicit duty after adoption, with equivalence and independent
  review before any human submission decision.
- The S0 review-clean specialization as a hypothesis only.
- Fresh-tail as a separate hypothesis only.

### PATCH

- Add the useful metadata to the existing backward-compatible `Proposal` schema and current
  Researcher prompt, rather than adopting the private proposal schema.
- Map “Porter” onto an approved Implementer port, independent Verifier review, Operator replay,
  and Supervisor state recording. It gains no standalone authority.
- Describe the existing restart-safe controller stages and use the current
  `autoloop/selection.py` screen/confirmation engine. Preserve the repository's existing policy,
  thresholds, controls, artifacts, and authority.
- Keep `116188cf` current unless S0 clears independent Neuron lockstep, two cold public20M runs,
  and reviewer verification.
- Keep the later table-freeze ablation frozen pending its prerequisite result.

### REJECT

- Replacing root `AGENTS.md`, or treating private prose as authorization.
- Importing `workflow/tools/decide.py`, `preflight_dryrun.py`, `contained.py`, or `run_job.py`.
- Creating `research/v2/queue.json`, `ledger.jsonl`, `state.json`, or any parallel queue/ledger.
- Importing the private queue report, evidence directory, run records, claimed results,
  candidate code, or patches. Their hashes establish provenance only, not correctness or
  approval.
- The private IMPL-1/research-v2 implementation and A2-before-A1 ordering.
- Promoting S0 from private CPU evidence alone.
- Allowing a 2M screen to promote or adopt a candidate directly.
- Any private threshold, idle policy, launch command, science claim, or proposed schedule that
  differs from current canonical configuration or a human-approved proposal.

No private implementation or evidence has been copied into the canonical repository.
