# Actual P3 fallback launch

The repair was source-qualified before the 09:14:07 UTC cutoff, but its bound
transport was not locally validated by that cutoff. The Supervisor explicitly
selected this already-reviewed P3 fallback. No K2 retry was launched.

- Actual dispatch: 2026-09-26 09:14:36.679448 UTC; PID 3370167.
- Label: `fs-20260926-p3-component-trn22-r1`.
- Source: `3637166b43a0b996e7eb7c71bb3b5d56e7f8119241b0278107a72787044dcc99`.
- At 09:15:19.452258 UTC, ranks 0–3 were actual PIDs 3370203–3370206,
  on the exact requested argv; original lock device66305/inode832589 held.
- Free disk at that observation: 171822510080 bytes.
- This is the direct 900-second component diagnostic with external 930-second
  timeout and the unchanged lifetime guardian. No training harness, checkpoint,
  evaluation, automatic successor, speed claim, or quality claim.
- The exact dispatch used assignment SHA
  `b474bd86479da5ef12e8812d8ad2f3a296c3bd825a093582a2fe439d2673b0c4`,
  before the Supervisor's later literal-label appendix. Local release-text
  recognition was narrowed to that whole-file hash plus the explicit P3/source
  fallback wording; no driver or remote staging logic changed.
- The first live collector retained actual process/source/cache/lock evidence.
  Its generic optional `diagnostic.log` name did not match this driver's
  `component-diagnostic.log`; it does not establish absence of diagnostic errors.
  The terminal collector will retrieve the actual component log and outputs.

Actual idle interval from K2 terminal09:04:07.164509 to dispatch: 629.514939s.
All original failed K2 artifacts and both hosts' protected artifacts remain.
