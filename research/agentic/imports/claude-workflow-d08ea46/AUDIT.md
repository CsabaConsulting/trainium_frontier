# Audit: private workflow `d08ea46`

## Provenance

- Source URL: <https://github.com/poojithdevan4D/trainium-research-context/tree/claude/trainium-frontier-research-nkp5zd/workflow>
- Read-only source commit: `d08ea46dd8135a6ceafa1b0856e5a2f0fc1fe474`
- Audit date: 2026-09-24
- Integration base: `9ecf4cd580012f213daab994cf8b760b1e1126d0`

The private checkout was treated as untrusted evidence. This repository does not copy its source,
receipts, queue, ledger, proposed candidate patch, or experimental claims.

## Reviewed identities

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

## Disposition

### ADOPT

- Explicit deterministic lifecycle documentation and strict separation between Researcher,
  Implementer, Verifier, Operator, Analyst, and Supervisor.
- Research ranking inputs: probability of improvement, estimated host-hours, concrete Trainium
  feasibility, cheapest falsifier, and durable prior-result references.
- Treat review-clean porting as an explicit duty after adoption, with equivalence and independent
  review before any human submission decision.

### PATCH

- Add the useful metadata to the existing backward-compatible `Proposal` schema and current
  Researcher prompt, rather than adopting the private proposal schema.
- Map “Porter” onto an approved Implementer port, independent Verifier review, Operator replay,
  and Supervisor state recording. It gains no standalone authority.
- Describe the existing restart-safe controller stages and use the current
  `autoloop/selection.py` screen/confirmation engine. Preserve the repository's existing policy,
  thresholds, controls, artifacts, and authority.

### REJECT

- Replacing root `AGENTS.md`, or treating private prose as authorization.
- Importing `workflow/tools/decide.py`, `preflight_dryrun.py`, or `contained.py`. They overlap the
  canonical controller/protocol and have not passed this repository's verifier chain.
- Creating `research/v2/queue.json`, `ledger.jsonl`, `state.json`, or any parallel queue/ledger.
- Importing the private queue report, evidence directory, run records, claimed results, or the
  `IMPL-1` patch. Their hashes establish provenance only, not correctness or approval.
- Any private threshold, idle policy, launch command, science claim, or proposed schedule that
  differs from current canonical configuration or a human-approved proposal.
