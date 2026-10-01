# Human rejection checklist — Sep28 03:15 IST

Required in AUDIT_BRIEF.md of every newly built submission package. Each row
must say PASS or FAIL, cite exact source lines and/or hash-bound measured
evidence, and describe missing evidence. Unknown is FAIL (pending evidence),
never an assumed PASS. Do not alter immutable, already exported packages.

1. Only train.py is required; it runs with the exact launch command; no extra
   files, pip installs, or imports missing from a clean scoring box.
2. load_for_eval returns the trained model; evaluation and training use the
   same forward/attention/MLP/kernels; targets=None is pure causal, with no
   evaluation-only model behavior.
3. No reads of validation data, network access, external data or weights,
   hardcoded local paths, or dependence on local NEFF/compile caches.
4. Empty-cache cold startup is measured <=900 s; charged training <=1800 s.
5. Report total bytes written for checkpoints, logs and caches; no large
   leftover files. Final disk occupancy alone is not total bytes written.
6. No required environment variables beyond the launch command; features
   are default ON in code. Distinguish optional defaults from requirements.
7. Fixed seed; no timing-dependent behavior beyond the time-based schedule.
8. Reviewer-readable CLEAN source: no dead/disabled experiment paths; short
   docstring explaining each kernel and the shared training/evaluation path.

The checklist supplements, and does not waive, numerical equivalence,
source provenance, cold-run evidence or the human Claude audit/upload gate.
Existing evidence-only exports remain evidence-only until all required
checks are resolved; never relabel a failed score gate as passed.
