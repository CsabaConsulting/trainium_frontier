# Dense 6x1024 review-clean MATRIX_LR 0.025 development package

This package is byte-identical to `dense6x1024_lean_review` except for the single
source change `MATRIX_LR = 0.02` to `MATRIX_LR = 0.025`. Its launch command is
unchanged. It is a development candidate, not a validated submission.

The approved cohort requires a full 1800-second run on trn21 followed by a new
process running protected public evaluation at sequence length 1024. No smoke run
is authorized.
