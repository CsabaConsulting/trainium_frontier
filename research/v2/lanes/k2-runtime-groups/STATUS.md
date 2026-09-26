# K2 runtime groups status

2026-09-26, completed local evidence at 10:18:53 UTC; see REPORT.md.

- Source frozen: `f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e`.
- Exact narrow diff prepared against frozen/running 662c.
- After explicit slot release, both-dtype CPU finite/reference/OFF3/ON3 tests
  exited 0; the four historical nonfinite diagnostics per dtype stay labelled.
- Actual production parser/BIR/neuronx-cc/NEFF compile PASS; 61,338-byte backward
  executable, 3.52149 seconds. Not a full-model startup or throughput result.
- Both exact-production official simulator fixtures PASS, maximum error 0;
  sparse fixture includes 4096 untouched rows. Artifact/receipt hashes archived.
- All processes exited. The one local-heavy slot has been released to root.
- No host calls, running-source changes, launch, retry, native timing or BPB.
  Existing table/native packets are untouched; no native readiness asserted.
- Source identity alone is not launch authority. Root selects from actual
  evidence; the existing accepted framework forward applies without extra gate.
