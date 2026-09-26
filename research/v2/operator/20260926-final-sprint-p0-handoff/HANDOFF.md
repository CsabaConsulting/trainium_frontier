# P0 training complete; attached public-20M protected

Actual bounded read-only observations: trn21 2026-09-26 07:51:15.038799 UTC,
trn22 07:51:15.096373 UTC. Both training phases exited zero and produced their
checkpoint. Both attached fresh evaluations are live, not complete. No source,
process, cache, queue or remote file was changed by this observation.

| Measurement | trn21 schedule | trn22 batch262144/LR.015 |
| --- | ---: | ---: |
| Complete sequential observed boundaries | 3633 | 1919 |
| Uncapped launch-to-fifth observation, seconds | 825.4130105050281 | 853.7422744090436 |
| Capped observed startup allowance, seconds | 825.4130105050281 | 853.7422744090436 |
| Training process through-exit wall, seconds | 2614.964834033046 | 2642.6125920159975 |
| Through-exit charged duration, seconds | 1789.5518235280178 | 1788.870317606954 |
| `charged_through_process_exit_within_budget` | true | true |
| Source `over_budget` | false | false |
| Source `training_seconds` | 1784.9 | 1784.3 |
| Source `startup_allowance_est` | 823.3 | 851.7 |
| Source `charged_seconds_est` | 1785.2 | 1784.7 |
| Training end UTC | 07:49:26.737113 | 07:49:54.802373 |
| Attached fresh-20M launch UTC | 07:49:26.809892 | 07:49:54.875529 |
| Available bytes at observation | 14184493056 | 16561147904 |

Observed allowance is `min(900, fifth observation)` and both fifth boundaries
are below 900 seconds. Wall minus allowance equals reported charged duration.
The source estimates are retained separately, not substituted for through-exit
measurement. All 3633/1919 boundary records are present without truncation.

The actual launch and earlier four-rank environment receipts establish the
six new private P0 training cache roots. The harness's own
`fresh_empty_sdk_caches: {}` field is preserved verbatim; it is not standalone
proof of cold cache creation. No cache-hit claim is made. Current checkpoint
and caches remain protected pending the final score and verdict.

## Checkpoints at training completion

Both are 453044626 bytes, recorded by the harness and independently matching
the driver's checkpoint receipt. No checkpoint was downloaded.

- trn21: `fbf2db730ced558129388997e1289e6e57d14935c5694463c93ef9bc71b1b1b7`
- trn22: `a648decd94a8d8d0efca7abfc3ce9aef70b10806bddaad4a52e7047d93c53951`

Post-evaluation stable checkpoint rehash remains pending, as does whole-job
terminal and explicit union-of-observed-descendants/device/lock postflight.

## Actual evaluation identities

trn21: timeout3229806/start94935327; harness3229807/start94935327;
evaluator3229821/start94935565. Parent guardian3203817/start94673537 remains
under original held lock66305:596823.

trn22: timeout3343121/start102305408; harness3343122/start102305408;
evaluator3343136/start102305664. Parent guardian3315643/start102040839 remains
under original held lock66305:832589.

Both actual evaluator commands use `prepare.py eval-public`, batch size1,
sequence length1024 and20971520 tokens, reading the corresponding new
checkpoint. No fatal error is present. Existing int64-to-int32 SDK warnings
are preserved; they are not presented as a fatal error or a scientific pass.
No BPB has been emitted in this snapshot. Neither host is free.

## Immutable raw evidence and local authentication

- `trn21-live.json` SHA256
  `016de27f39e91e70c9ef5bd5d3a5b90106b60505e1efa2b1fea6c20172f4f03b`
- `trn22-live.json` SHA256
  `e098d8f568f6cd819c5818fa57158d32b5c8b662801deb1706ab767b6ff66a9f`
- Embedded trn21 complete rehearsal SHA256
  `6ff345ada8d7f4a40bef3881b78f1a40440cb808ecca70e3c04cf0879c4f42de`
- Embedded trn22 complete rehearsal SHA256
  `f69f3ff5323d6b69d9e7ad5c9b1d067a90d071938424fce3830a59dcdf560d43`

Local authentication passed94 checks: all16 present files per host rehashed
and untruncated; all sequential finite strictly increasing step observations;
startup-cap/through-exit arithmetic; clean exit; matching checkpoint identity;
actual evaluator flags; held original locks; and final evaluation still absent.
The first local parser used the wrong flag name `--eval-seq-len` and raised
`ValueError: list.index(x): x not in list`. It was corrected to the actual
unchanged `--seq-len` in the already captured argv, then passed. No host action
or scientific change accompanied this local parser correction.

Next useful result check is approximately08:01–08:04 UTC. Existing scoped
cleanup/next-dispatch authorization applies only after actual whole-job
terminal, retrieval and explicit quiescence, then fresh original-lock guards.
