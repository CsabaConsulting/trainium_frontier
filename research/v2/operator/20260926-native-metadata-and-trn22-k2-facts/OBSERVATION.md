# Installed metadata and prospective K2 host facts — read only

Observed 2026-09-26 06:15:14.972172 UTC (trn21) and 06:15:15.243655 UTC
(trn22). Both observations held the original host lock through fresh successful
ps/neuron-ls/fuser idle checks and the bounded reads, then released it. No
staging, reservation, package install, SDK import, source edit, tensor creation,
helper invocation, native job, checkpoint access, deletion or AWS action occurred.

The selected K2 proposal/spec were read from immutable Git commit
`d6ded1f925bf6b929407ff36cd365c8d50655ea7`; those paths were not yet materialized
in the canonical checkout when first requested. This observation is not release
authority and does not treat the partial design as launchable.

## Installed metadata, both hosts

The exact invoked interpreter was
`/home/ubuntu/workspace/native_venv/bin/python -I -B -S -`, resolving to
`/usr/bin/python3.12` (3.12.3). `-S` prevents site/.pth startup execution.
Standard-library distribution metadata and explicit-directory filesystem
resolution were used; no selected native module appeared in `sys.modules`.
All eight executable resolved-path/link/hash/size pins are in each raw receipt.

Both hosts report the same selected distributions and METADATA hashes:

- torch: `2.11.0+cpu`.
- torch-neuronx: `2.11.3.0.0+unknown.dev`.
- neuronx-cc: `2.26.6360.0+6f180f47`.
- nki: `0.5.0+28631259367.ga768afa6`.
- neuron-agentic-development: `1.0`.

`neuronx-distributed` was not observed in the selected venv dist-info metadata.
Filesystem origins and source identities for resolved top-level modules are
recorded under `remote.native_package_version_inspection.selected_packages`.
**torch_neuronx and torch_mlir did not resolve from the explicit site-packages
directory under this nonimport check**, despite torch-neuronx metadata being
present. No .pth/editable finder was executed or inferred. Their actual runtime
import origins remain unproven and require separately scoped evidence before a
binding may claim them. Metadata is not proof of native execution or compatibility.

## trn22 prospective K2 facts

- Original lock: device66305/inode832589. Idle ps/neuron-ls/fuser; remote STOP
  paths clear; canonical global-correctness and trn22 breakers false.
- Project HEAD: `65155c668efaa215a2136f67617b44ed32d4c9f6`.
- **Project status is not clean.** No tracked changes were reported, but six
  untracked retained Sep24 control receipts exist under `research/run-logs/`:
  `fs-20260924-v2-35c9-lr015-control-trn22-r1.job.json`, `.launch.json`, `.log`,
  `.postflight.json`, `.preflight.json`, and `.process.json`. All were preserved.
  This fails the proposal's unqualified clean-status claim; no exception or
  cleanup was invented by the Operator.
- Packet and output roots `/home/ubuntu/k2-f0f2-trn22-evidence-r1-packet` and
  `/home/ubuntu/k2-f0f2-trn22-evidence-r1-output` are both absent.
- Exact retained `.../fs-20260926-confirmed36-clean-coldfull-trn22-r2-packet/train/train.py`
  matches source SHA-256 `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`,
  87,986 bytes; stable device66305/inode2932616, uid1000. Source was not executed.
- Free disk: **17,296,404,480 bytes**, above the unchanged 16 GiB floor by
  **116,535,296 bytes** only. MemAvailable: **129,079,549,952 bytes**, above 24 GiB.
  These are observations, not reserved space or a post-staging preflight.

trn21's project remains clean at HEAD `e65a3097a5b00f6178188a935bb2acb9a5042c6c`.
Its disk was 14,746,189,824 bytes free, MemAvailable 131,408,203,776 bytes.
Both top-level launch flags remained false; controller snapshots are not releases.

Raw receipt SHA-256:

- trn21.json: `3e1156c526bb6d9172ee4bf9a7ee2bf09e5b190fdce1d6dd5c6e42880606adb9`
- trn22.json: `f02e57f6bfd37e71a2587820148555c4afa102c48ed5a592c91a469cd62ca963`

The two collectors are consumed. No additional host inspection is implied.
