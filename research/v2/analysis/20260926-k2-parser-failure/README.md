# K2 parser failure diagnosis (CPU only)

Diagnostic work area, started 2026-09-26. No model candidate, review gate,
host command, launch, data access, or adoption is part of this investigation.
The candidate remains untouched at SHA-256
`a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71`.

Root authorized one isolated local diagnostic environment using exact archived
host NKI/compiler versions, no existing-environment mutation, a 5 GiB total
artifact limit, and the existing 30-minute diagnosis limit. Initial local free
space was 70,229,852,160 bytes at 2026-09-26T09:31:37Z.

Official AWS package source is documented at
https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/get-started/setup-env.html
as https://pip.repos.neuron.amazonaws.com . The exact archived host NKI wheel
is present in that index for CPython 3.12 and Linux x86_64.

Completed at 2026-09-26T09:35Z, within the 30-minute bound. See `REPORT.md`,
`parser-results.json`, and `diagnose.py`. The isolated environment uses about
164 MB and is ignored by Git; no package installation outside it occurred.
