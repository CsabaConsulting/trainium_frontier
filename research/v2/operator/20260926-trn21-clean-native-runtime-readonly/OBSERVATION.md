# trn21 native-consistency preparation — read-only identities

Observed 2026-09-26 06:05:54.020812 UTC. No staging, helper invocation, native
imports, tensors, compilation, data access, cleanup, RELEASE creation or launch.

The packet's actual binding is
`research/v2/proposals/20260926-confirmed36-clean-native-trn21-packet-r1/packet/NATIVE_BINDING.disabled.json`.
The initially supplied top-level binding path was absent; the existing nested
binding was located without editing the packet. Binding SHA-256:
`c29fe99533c21cae045a4cd7b5116f40808feda65bdab14e5ed38e1ab0b11760`.

The immutable raw receipt contains the eight runtime executable pins in
`remote.runtime_pins`, each with exactly `resolved_path`, `link_target`, `sha256`
and `bytes`. Runtime links were observed explicitly; system Python and the
venv Python resolve to `/usr/bin/python3.12`. Original symlink and resolved-file
metadata were stable. No executable or SDK was imported to determine these pins.

All 17 retained input files match the unchanged INPUTS manifest
`0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e`,
including the real prepare.py, reference/candidate sources and both full CPU
checkpoints. Checkpoint contents were streamed only for SHA-256, not copied.
The accepted helper/foundation local bytes match their support pins; their
proposed remote paths and the entire prospective root remain absent.

Original device66305/inode596823 was held exclusively during observation.
Fresh ps, neuron-ls and fuser show idle hardware and no job. Remote and local
STOP checks are clear; canonical global-correctness and trn21 breakers are false.
The authoritative host launch flag remains false. This historical controller
snapshot is **not** a release or a replacement for fresh staging/launch gates.

- Free disk: 14,746,206,208 bytes; initial floor 14,495,514,624 bytes passes
  with only **250,691,584 bytes** headroom.
- MemAvailable: 131,413,176,320 bytes; 68,719,476,736-byte floor passes.
- Prospective root absent:
  `/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1`.

Native package/version inspection remains unperformed because this task selected
only the eight explicit executable identities. Actual native compatibility,
resource usage and numerical correctness remain unproven. No active training
queue or host reservation is claimed.

Raw `runtime-inspection.json` SHA-256:
`8ca95247b8f6a0699a4c198c514a4d6ff458e7f322963180d448fdf2af56e9da`.
The collector is consumed; do not rerun it as a release/launch action.
