# Conditional next-job binding and final retrieval readiness

Supervisor releaseab2b36569, actual11:03:49 UTC, is authenticated by
assignment SHA c0c63ef2a76f5d4b21a67393abf6b42b386ec2ba241531473211c4ecb21f5df5.
The two immutable prepared warmdown packets remain unchanged and disabled.
`stage_warmdown.py` mechanically binds the existing staging body to this
release. Both local payload checks pass; no remote staging or new launch
has occurred.

The release-bound transport preserves exact scientific source/argv/proposal/
review, original locks/guardians and resource guards. Before execution it
requires the actual current host label's successful terminal plus attached
eval, a retrieved postflight proving all observed IDs absent and ps/device/
fuser/original-lock quiescence, and no activated cross-host replica priority.
The existing fresh machine checks still run at staging and final lifetime
acquisition. An actual failure is preserved and referred to Supervisor; this
binding does not weaken its successful-terminal condition or retry a label.
No fixed-source-path option is included.

Local release-bound payload hashes at these checks (controller snapshot
included, not a future immutable dispatch hash):

- trn21: 4715d3e8e8f2fb41d32a74ac31a6d3bd388e625281ca52889203573cdb98f151.
- trn22: 01196ad94fecb8fbcb2b5283d83299010be29c0b0bbe190ce77f0b293dfae197.

The 11:05:07 UTC bounded full snapshots authenticate original four-rank
workers, c2e27ccd/8c6d7af source bytes, original held locks and actual cache
environment. Free space was163934932992/166691291136 bytes. Both terminal
files remained absent.

Raw snapshot hashes:

- trn21-full-recovery1105.json:
  dd0d58dae753093b8501c3a4d86b4dede2d273e76679b44c0206b2654536ef82.
- trn22-full-recovery1105.json:
  9665bf6ef085b11a321f9c93ff32e5c454ab94dc5d7fea1cc5875fdde357c612.

`retrieve_current.py` reuses the prior terminal collector and remote body
byte-for-byte, with current paths and the union of actual observed identities.
The two legacy templates are archived exactly under `proven_terminal_*` so
collection does not depend on untracked scripts in the old worktree. Their
SHA values remain86c96c314317f591939ae2c84f891027580963bc46d433bd08f029000268ac4c
andd71d9b710b33a5e8f9051d079e3e8a7d13b83676f3de16d32be02ae5cfb1c482.
Local checks currently bind519 trn21 and144 trn22 actual identities, including
guardian lifetime events. These are evidence, not invented transient PIDs.
No terminal retrieval has been attempted early. Checkpoints will be hashed
and retained remotely by the existing bounded collector, not copied or deleted.
