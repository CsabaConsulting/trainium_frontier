# Dense 6×1024 submission

Upload **train.py** from this directory and paste the entire line in
**launch-command.txt** into the launch-command field. All required custom kernels
are included in train.py. The organizer supplies prepare.py and the fixed assets.
The command trains from fresh initialization and saves a consolidated `final.pt`
under the organizer-selected output directory (locally, `out/final.pt`).

The launch line is **245 characters**, excluding its trailing file newline, and
fits the 256-character field. It omits only values supplied by frozen `train.py`
defaults: compiler flags, 1,000,000 maximum steps, the 1,800-second budget,
depth 6, seed 42, and enabled model/Muon compilation. A Trainium2 argument probe
confirmed that the shortened and original commands produce identical complete
argument namespaces and compiler environments; the organizer static check passes.

The source recipe achieved **0.9928513116311253 public validation BPB** over
20,971,520 tokens at sequence length 1024. This is a public development result;
the private leaderboard score has not been measured. The model has 117,453,580
parameters, no ngrams, and no FP8. It uses BF16 compute, custom MLP backward,
two-core exact softcap/cross-entropy, optimizer ownership across four LNC2 ranks,
and gradient clipping after averaging.

The winning run completed 3,660 updates in 1,785.0 estimated charged seconds.
`provenance.json` pins the source commit, input hashes, launch command and
packaging changes. `research/build_dense_submission.py` reproduces this file.
The original training loop, data loader and timing hooks are AST-identical to
the winning source. The evaluation hook strictly loads the same weights and
freezes the model; training adapters are not installed during evaluator import.
It evaluates independent sequences with batch dimension one, preserving the
original B1 logits while avoiding backend rounding differences between batch
sizes. The parameter names, checkpoint and trained sequence model are unchanged.

The [official rules](https://trainium-frontier.devpost.com/rules) exclude startup
and compilation. The current organizer `prepare.py` implements this through
the first five real optimizer-step boundaries, capped at 900 seconds. The
submission retains those calls and reserves time for checkpoint writing.
The command disables optional in-process public evaluation; the organizer's
standalone scorer calls `load_for_eval` and computes its own score.

Packaging validation: the static checker has no warnings; CPU and Trainium
checkpoint-contract gates pass exactly, including original B1 logits, causality,
batch independence, repeatability, frozen parameters and unchanged buffers.
The isolated 20M public-score replay reproduced **0.9928513116311253** exactly
in 692.86 evaluation seconds, with a clean postflight. The isolated cold-cache
training rehearsal completed 3,634 updates and wrote its checkpoint within
1,790.34 charged seconds through process exit (startup 855.09s, below the 900s cap).
The freshly trained checkpoint scored **0.9938556748843932 public BPB** over
the same 20,971,520 tokens in 697.48 evaluation seconds. Training and evaluation
both have clean postflights. The checkpoint is preserved locally and in a
digest-verified GitHub backup. This package is ready for user submission;
see `validation.json` for the recorded checks and provenance. No competition
submission has been made, and the private leaderboard score remains unknown.
The organizer README and prepare.py were authenticated against upstream commit
`c907da4c74bbaafeca7d13765c49610b7e77436c` on 2026-09-19 UTC; both match this repo.
