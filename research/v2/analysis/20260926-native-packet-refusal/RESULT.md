# Root evidence authentication

2026-09-26 05:41 UTC. `python3 -B audit.py` and `python3 -O -B audit.py`
both completed with 1,963 checks. These primarily authenticate files, manifest
coverage and the independently observed defect; they are not new model tests.
The 271-entry original manifest, 266 archived regular files including that
manifest, six literal negative-test links and six explicitly omitted sparse
test stimuli are accounted for as described in ARCHIVE_SCOPE.md.

The fresh reviewer independently passed 15 grouped mechanical/exactness tests
in both modes, but reproduced a separate required failure: a PermissionError
scanning a directory containing a 17-byte hidden receipt became zero observed
artifact bytes and passed the gate. **REFUSE remains the verdict.**

Report SHA256:
`3363b898a0042d0634d1132ad31471e985c8dd3055ac4a331354c1ca08b0259c`.
Original evidence manifest SHA256:
`d316d2ea1296df6381641591651ccd4f275fa27a21123ade55ad303a6576add0`.
New disabled driver SHA256:
`132fa83ae654a3d7921ef8b907f759c6c69d2d3be11035170b56d1d98ff9c54c`.

This finding does not modify or invalidate the separately reviewed guardian
and different fixed driver in the live full cold replay. No native run,
launch authority, cleanup approval or submission readiness is established.
