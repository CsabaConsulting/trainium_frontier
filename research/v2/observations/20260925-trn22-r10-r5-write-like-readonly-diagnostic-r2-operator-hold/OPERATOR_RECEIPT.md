# trn22 r10 r5 write-like read-only diagnostic r2 — Operator result

**HOLD. The one permitted diagnostic call is consumed. Do not retry this label.**
This is a current no-write stability-gate result, not evidence that the
historical r5 write-phase failure had the same cause. It gives no source-stage,
training, evaluation, cleanup, breaker-reset, or AWS authority.

## Exact authority and preflight

- Assignment checkout: `/home/poojith-devan/Desktop/Trainium-trn22-r10-diag-r2-supervisor-assignment`, clean at `37ee50c66057888b105fbf164c8c91b181f57d04`; it remained clean after the call.
- Independent assignment PASS: `08361ffddcd7b27f79bcc75e037d6bd3887bf85c`; assignment JSON SHA-256 `5926a4f028ccbcf3560c1979ef65c300de4a3bfef965d973b768eb04e14b6f12`. Exact enabled request SHA-256 `8d1393dcb577b3584f532656126216513517155896717adca34b0793c322b04c`; independent final review JSON SHA-256 `eea65a2e1f9660cd1e71cc33f7d92bc4a65051ee39334ee5bdd75c0a3f2c8fb9`.
- Source SHA-256: `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`; `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`; `sha256sum -c SHA256SUMS` passed 13/13 package entries.
- Independent local checks immediately before dispatch passed exact source/authority, Desktop/former/package STOP and breaker state, consumed-marker hash, and pinned SSH identity metadata. The consumed r5 marker SHA-256 remained `229a331b1aa8b302c607933273a7b09d6b35df9d3892e6ddbad41fb894bb1da8` after the call.

## One command and sanitized outcome

The Operator invoked exactly once, without arguments:

`/usr/bin/python3 -B /home/poojith-devan/Desktop/Trainium-trn22-r10-diag-r2-supervisor-assignment/research/v2/authorizations/trn22-r10-r5-write-like-readonly-diagnostic-r2-hold/operator_readonly.py`

Exit code: `1`. The program's UTC receipt time is
`2026-09-25T12:52:41.894870+00:00`. The full bounded, sanitized output is
`SANITIZED_DIAGNOSTIC_RESULT.json`, SHA-256
`64dab354b4b70a7805f1f585d6fa0e80610073dfda949fcd3be8ff8e1dbb5af6`.
No raw SSH stream, exception text, key, credential, or environment was
archived. There was no second diagnostic call.

1. Fresh read-only host preflight **PASS_NO_WRITE**: original e65 source, root free `58,452,955,136` bytes, `31,660,010` free inodes, no bundle or target. Its full probe includes host identity, source/protected assets, process/Neuron activity, STOP, disk/inodes and host lock.
2. The `write_like` phase **FAIL_NO_WRITE** at `PREFLIGHT_STABLE`, after `HOST_LOCK` and `FULL_PROBE`. Sanitized class `PREFLIGHT_STABILITY`, `RuntimeError`, error SHA-256 `61beb1d0e08a512b90ba8169d2b5ddea3478de488fffe983e7b0367f44f7e4a5`, 40 error bytes. The second full probe took `0.9455387600464746` seconds; transport returned 0 with zero stderr. It never reached the final would-open timing gate.
3. A distinct SSH/process read-only **postflight PASS_NO_WRITE**: original e65 source, root free `58,452,942,848` bytes, same free inodes, no bundle or target. Transport returned 0 with zero stderr.

The exact changed field behind `PREFLIGHT_STABLE` is not determined by this
sanitized receipt. This attempt did not stage r10, start training, or produce
new BPB. Further host writes require a separately diagnosed and reviewed
request under the human's conditional consent; this one-call assignment
cannot be reused.
