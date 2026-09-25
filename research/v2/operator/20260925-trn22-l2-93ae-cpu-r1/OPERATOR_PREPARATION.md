# trn22 exact 93ae CPU gate preparation

Supervisor assignment: architecture CPU gates take the slot after the active
`fs-20260925-l3-timer-dense-trn22-r1-full` training **and** fresh public-20M
evaluation finish. Neither phase may be interrupted. Four serial CPU probes
only are authorized by this assignment; a native architecture run still needs
independent numeric acceptance and a separate exact native binding.

Packet commit: `297c80441eb4c4e43ef8bc6a21bbc73c5275c4ea`.
Candidate source commit: `1e21c3d797eed764f108c41546b270065ad5650c`.
Candidate source SHA-256:
`93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
The packet files are retained verbatim; their prepared/disabled status records
the packet Implementer's state before the subsequent Supervisor assignment.

The per-probe `cpu_driver.py` files are operational bindings of the proven
`research/v2/operator/20260925-trn21-l3-timer-dense-g2-r1/cpu_driver.py`:

- The host and unique probe root are changed to trn22's assigned values.
- Source and G2 tool paths come from each exact packet command; the same SHA,
  evaluator, idle-process, Neuron-holder, disk and RAM checks remain.
- STOP roots are freshly discovered under `/home/ubuntu/trainium*` in addition
  to the common root, prepare root and probe root.
- The exact command array in the packet replaces the prior inline assembly.
- All other environment, logging, subprocess and terminal handling is unchanged.

Each `request.json` is the corresponding verbatim packet probe entry plus
the packet's prepare directory, prepare SHA and G2-tool SHA. No source,
framework, harness or evaluation modification is made.

Launch uses the proven detached `flock -n` pattern on
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`. The Operator checks the
old queue has exited at its owned `PAUSE_NEXT` boundary and the active run has
a terminal result before any CPU process. The next CPU probe is started only
after the preceding one has exited successfully and postflight is clean.
The six existing fallback jobs and resolution manifest are preserved.

All five source/tool Git objects were checked locally against `sources.json`.
At preparation time no stage, CPU probe or native architecture launch has
occurred on trn22. Host observations and later execution receipts are separate.
