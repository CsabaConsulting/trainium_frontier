# Fresh independent R66 derivative review

Verdict: **ACCEPT for the authorized R66 derivative and package export only.**
The default REFUSE position is cleared by the independent checks below. This
review does not authorize Codex submission, training, host operations or cleanup.

Reviewed on 2026-09-28 22:03 UTC / 2026-09-29 03:33 IST by the fresh
`/root/r66_upload_fresh_review` agent. I was not the proposer, implementer or
operator. I did not read implementation notes, `compose.py`, `EDITS.json`,
`verify_structural.py` or a shared implementation branch. Inputs were the exact
source diff, user-lane R66 specification, original Claude conditional audit,
latest R66/67 authority text, gate receipt, and the source/archive context needed
to check the claims independently.

The reviewed derivative is
`6d197b3cd08a4e107020e5bb03a946a06a361ed36c6d7dfc12b35ca1484be14c`
(99,747 bytes). Its scored parent remains
`b758450b599d4acc5b1fc17d37cccb87421d22bb99c213f6d8ed29fdede13fa8`.

The independent command completed with exit code 0:

```text
python3 -I -B research/v2/verification/20260929-r66-upload-fresh-review/check_independent.py
```

`check_independent.py` SHA256 is
`972dbc87bbaae3dc9db013a51c7272bbe6dd92e2dfa4947217c08e6e46b81432`.
Its exact JSON output is retained in `INDEPENDENT_RECEIPT.json`.

## Checks and findings

1. Complete-byte reconstruction permits exactly three replacements: the FP32
   header correction, the CLI header correction, and the argparse freeze default
   286000000 to 0. Reconstructing from the authenticated b758 source yields the
   candidate byte-for-byte. There are no other source edits.
2. A reviewer-written AST interpreter reconstructs the two argparse options and
   executes stdlib argparse against empty argv. It returns out_dir `out` and
   tbl_freeze_tokens `0`. The actual `main` assignment is checked to pass that
   value directly to `build_config`. The interpreter then reconstructs the
   dataclass/config from its scalar AST, without executing candidate code.
3. That complete config equals the independently extracted `model_config` JSON
   in BOTH actual training logs: trn21 `d3c2c030...` and trn22 `1c56139d...`.
   Both archives contain exact b758 package source bytes and executed
   `--tbl_freeze_tokens 0`. All 47 text members in each pinned archive pass their
   byte-length and SHA256 checks. Recorded training and eval exit codes are 0;
   cold times are 645.214926 s / 676.220381 s and charged times are
   1789.246908 s / 1789.974079 s. Checkpoint identities are recorded, not loaded.
4. The complete module AST is identical after normalizing only the module header
   docstring and the single authorized argparse constant. All 43 top-level
   functions/classes apart from `main` also match without normalization, covering
   the model, config, optimizer, table update, save and evaluation paths.
5. Explicit freeze values 0 and 286000000 produce identical parent/derivative
   arguments and configs. The direct `build_config()` default remains 286000000,
   as required by the byte allowlist; it does not determine bare CLI behavior
   because `main` supplies the parsed value explicitly.
6. The header's BF16 table/RMS wording is consistent with persistent BF16 table
   allocation and BF16 `square_avg` initialization in the unchanged source. The
   header correctly names both parser options.

## Conditional audit and remaining scope

Claude's original audit SHA256 is
`55be15b181201ef89a420d6048081a60c417e55836a51d5e8d7a7a7bfa745df9`.
It conditionally accepted the scored recipe and required this exact derivative.
The source exactness/config conditions (a), (b), and (c) are satisfied; Root must
record this derivative SHA in the audit and export the same bytes under
`train_UPLOAD_6d197b3c.py` with a one-line README, as the latest R66 file directs.
The existing scored `train.py` is to remain immutable. The human's eventual
upload must use the derivative as `train.py` and declare:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
```

This derivative has no separately measured score. Its evidence is exact source
and recorded-config equivalence to the two accepted CONTINUE runs; R66 explicitly
requires no retraining. This review does not expand Claude's source audit into
an organizer clean-box portability proof, cumulative-write measurement, or a new
native numerical verification. Those limits remain as stated in the audit.

No Torch import, model/tensor construction, laptop model numerics, host call,
training, package edit, or upload was performed. Only this reviewer-owned
verification directory was created. Parent and candidate bytes were rechecked
unchanged at the end of the independent execution.

Other reviewed input hashes: SPEC `7700414f9c02e10072d534f80fe30ce9b087884e9b33dc021d32712786752d71`;
SOURCE.diff `1a0b57f2cffa07434091bd76160e1444a1f7869122f5d45d0f196bddd89abcc3`;
author gate receipt `2f3020a86911765f8414618839e366171547e2fccce0f36e0ac5fa2cf37da4c3`.
The author proof was not rerun or used as the independent reference.
