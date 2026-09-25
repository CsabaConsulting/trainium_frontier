# Completed P profile: read-only offline CLI discovery

Observed 2026-09-25 20:30:51–20:31:28 UTC. No trace decoding, SDK import, device work, large transfer or host-file change occurred. Native seed43 and architecture runs remained untouched.

- `neuron-profile` is a deprecated/removed shim. Version/help/view-help all returned exit1 with the migration message preserved verbatim.
- Installed replacement: `neuron-explorer 2.32.0.498-b1f3998`, built 2026-08-10, commit `b1f3998a`.
- Its view help advertises `--session-dir` for an unpacked system/device profile, `--output-format=summary-json`, and options to skip instruction hierarchy/buffer/DMA detail. Help returns0 although its prefix says “exited with an error”; no actual decoder behavior was tested.
- Completed P inventory includes80 NEFF companions, total167,863,163bytes, under the rank0 native `neffs/` tree, and the4,130,966,280byte NTFF. The exact paths and sizes are archived; contents were not read or copied.
- This proves available CLI options and companion presence, NOT successful multi-model matching, bounded RAM/CPU cost, or accurate phase/MFU attribution. Summary output does not itself bound the amount of input processing. Do not arbitrarily pair one NEFF with the multi-model NTFF.
- The complete62764byte view-help output is preserved with whitespace normalization and a SHA256 of the original unnormalized output. The earlier prefix-only capture is explicitly named as such.

A future decoder probe needs a separately assigned noncompeting CPU slot and resource/output limits. No such probe was launched here.
