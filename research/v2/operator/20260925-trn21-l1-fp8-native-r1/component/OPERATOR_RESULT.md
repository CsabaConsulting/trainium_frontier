# FP8 native component: FAIL, no timing or quality run

Exact source921271, existing checker7ba967b0, LNC2/world4.
Launched2026-09-25 17:28:02UTC; process exited1 at17:28:30UTC.
All four ranks reported the same first-shape [256,256,256] correctness error:

`ValueError: native/reference RMS gate failed`

Forward relative RMS0.005053456407040358, packed0.004039883613586426,
saved_x0.002658416284248233, saved_w0.0038507836870849133.
Both dx and dw relative RMS are1.0, above the predeclared0.01 ceiling;
maximum absolute errors4.875 and326 respectively. Compiled/direct forward
relative RMS0.0. This does not establish the cause of the backward failure.

No OFF100, ON100, checkpoint-forward or full quality run was launched.
The exact raw four-rank trace is in component.log. All retrieved files were
verified byte-for-byte against host SHA-256 records. Postflight confirms no
remaining training/checker/evaluator processes or Neuron device holders.
Logs and partial outputs remain on the host; no cleanup or SDK/cache change.
The six previously installed fallback jobs remain unchanged and are eligible
for serial handback after normal local/host guards. FP8 requires diagnosis and
applicable source review before any further native attempt with a new label.
