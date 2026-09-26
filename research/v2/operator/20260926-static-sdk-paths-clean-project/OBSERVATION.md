# Static import mapping and existing-project search

Completed 2026-09-26 06:19:30 UTC, within the assigned five-minute window.
Original host locks and fresh idle ps/neuron-ls/fuser checks protected both
read-only inspections. No SDK, .pth or editable finder was executed. No
repository, source, ignore rule, log, checkpoint or cache was changed.

Both hosts have the same statically parsed editable mapping:

- `torch_neuronx` maps to
  `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/__init__.py`,
  20,941 bytes, SHA-256
  `91f3fd461b084f1426fbf1a2df9a910950d37c662e52270c1822a6338f31d99c`.
- `torch_mlir` is present in the editable finder's MAPPING and NAMESPACES,
  pointing under
  `/home/ubuntu/workspace/torch_neuron_eager/third_party/torch-mlir/python/torch_mlir`.
  There is no `__init__.py` at that mapped directory. This is retained as a
  namespace mapping without an executed runtime-origin proof, not a native SDK
  import failure.
- NKI's ordinary source origin remains the installed site-packages
  `nki/__init__.py`, SHA-256
  `747cdfb1638e7585ac412c1cff036e73613abe6f1325da0cb1ed0ce383f7b93f`.

Wheel RECORD, relevant .pth, static editable-finder text, literal mappings and
source hashes are preserved in each r2 raw receipt. Metadata interpretation
does not attest to actual native execution. Neither module nor finder was imported.

The first collector assumed every mapped package had an `__init__.py` and
stopped with the exact `FileNotFoundError` for the torch_mlir pathname on both
hosts. Those initial raw failures remain in trn21.json/trn22.json. A narrow
read-only r2 collector recorded missing/namespace origins as unknown and
completed the remaining checks; no scientific or host code was edited.

## No clean existing trn22 project found

The original project's worktree list names only that same original checkout.
Exactly the three permitted roots were inspected:

1. `/home/ubuntu/trainium_frontier`, HEAD
   `dab464a2ab1d52c52f1f470ad986b4c420198ba2`: dirty tracked experiment log and
   numerous untracked files.
2. `/home/ubuntu/trainium_freshstart_20260918`, HEAD
   `ca635670f67b067201fb9b6167eccf414b70bacd`: dirty tracked experiment log and
   two untracked historical run logs.
3. The worktree-listed original
   `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1`,
   HEAD `65155c668efaa215a2136f67617b44ed32d4c9f6`: unchanged six untracked
   historical Sep24 control receipts. No tracked changes were reported there.

No clean-status exception, new worktree, commit, ignore rule or deletion was
invented. Revised project-binding scope remains the Supervisor's decision.

Raw r2 SHA-256:

- trn21-r2.json: `94f143ee278692734612aa75b0b120fcf2c3380f84f75ae7d20c977fee8fbfc3`
- trn22-r2.json: `13765850ed090e1ebc9af465831617178c0ddfc601ffd5684255f59c860eda16`

No further host call is authorized by this completed diagnostic.
