# Local warmdown preparation, 26 September 2026

Both exact approved warmdown packets are locally prepared and mechanically
checked. No host was contacted by preparation, staged or launched. Both saved
bindings have `stage_authorized: false` and `launch_authorized: false`; neither
contains a RELEASE file. The local staging checker rejects `--execute`.

| Host | Ratio | Disabled packet SHA-256 | Driver SHA-256 |
| --- | --- | --- | --- |
| trn21 | 0.60 | f402300af29399caf6aff0be6f9eb186a659a76679ff21ee05b5baa438582ab3 | 63ed9df58d134cfb916dfe030e180f80eb412f47dccbdc9a237535707e2b3e9b |
| trn22 | 0.85 | 61e3356f8c3fe0d1e30927bd1b64dc289d5e06724c1cfd0d7c90da40057eafb1 | b4d4a1a3dca6f3fa8ddbb4729d20b12323adde4a996d04e2f4591aaf440e06f7 |

`prepare_warmdown.py` authenticates all members of the actually executed
same-host parent packets, reads the exact approved source/proposal/review/CPU
receipts, changes only root/label/source/command/provenance literals in each
driver, and proves reversal restores the complete prior driver. Guardian,
idle and dispatch code, prepare.py, harness, evaluation, full1800 accounting,
original lock and resource floors remain unchanged. No scientific source was
edited and no source review, smoke, model or native test was repeated.

Source is exact 4ec8b1b4, ordinary schedule with batch262144/LR.021. Each
packet has 12 authenticated members. trn21 keeps the higher 15032385536-byte
initial floor and six unique private cold training caches; trn22 retains
14495514624-byte initial floor and its actual same-host research cache paths.
Both retain 11274289152-byte runtime and 68719476736-byte RAM floors.

The proven staging body also passed both local-only payload checks:

- trn21 disabled payload SHA-256:
  736263a967d803ca2ace87bbfeb0952919d557ef4d084fbe1677d32d564646db.
- trn22 disabled payload SHA-256:
  84027cc7149453f597c1ae9b9fe30ab8001b01185dc54fbe99dd3b6941cf5150.

Payload hashes include the current controller snapshot and therefore describe
these checks, not a future fresh launch. The read-only recovery assignment
does not release staging or launch. Current full jobs and attached20M remain
protected; machine admission and current-result replica priority are pending.
No fixed-source-path change has been applied or released.
