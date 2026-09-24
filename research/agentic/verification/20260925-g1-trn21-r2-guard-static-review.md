# Independent static review: trn21 G1 r2 guarded evaluator package

Reviewed 2026-09-24 23:25 UTC. Implementer package commit:
`b8af19b3f67bdd12a2b35a5ac097a363d995dbb7`.

**Verdict: HOLD / not dispatchable.** The package preserves the staged public-20M
evaluator inputs and its disk monitor passed local safety checks, but the
controller-side adapter does not enforce the authorization schema promised by
`RUNBOOK_GUARD.md`. This receipt grants no evaluator launch, trn21 G1 PASS, A0
training, model result, cleanup, breaker reset, AWS action, or submission. I
made no host connection or host change.

## Bound package and checks

| File | SHA-256 |
| --- | --- |
| `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/request.guard-pending.json` | `921498a53b010006649a4570f10513d43988a1f0ea770031f1c739158519ff91` |
| `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/guarded_eval.py` | `5438d644513b0cddb554727e9c50f4b5e09bba43628ce5869109d17c1c598e70` |
| `research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/dispatch_one_label.py` | `52a6099c501c3017949254d27281f1073cb41bbe5351d5ea2c762923aca1f09f` |
| Prior staged request | `aba839eb87b041fbb266e2db3795d49f29afcf76c91ecd5eee96a5b3f978dd2a` |
| Prior Operator stage receipt | `ca417a419557c4891236716cb6d844c607dd1de22d9e1a8c4187a42a63015d52` |
| Prior independent stage review | `bf3c78a7db65ea7829386129969e7bb06cc700f11c2d4b2f251c448f04878a21` |

`jq` comparison independently confirms the pending request retains the
staged request's label, host, trained commit, nested `train.py` and trusted
`prepare.py` hashes, checkpoint path/size/hash, environment, and exact
evaluator suffix after the original outer timeout. It remains
`launch_authorized=false` with null authorization. The requested 1,024-token
context, 20,971,520 tokens, single Neuron process, six historical warm-cache
paths, and 1,200-second evaluator/1,800-second outer timeouts are unchanged.
The monitor samples free bytes once per second, uses the 6.5-GiB hard floor,
and requests an authenticated stop below the floor plus 2 GiB. The six cache
directories are presence/path-checked, not content-SHA-bound; the staged
contract contains no cache-content hashes, so this review makes no such claim.

The supplied five local unit tests passed. My separate mocked threshold test
observed one low-space sample and one stop call with reason
`reserved_headroom_crossed`. A separate local test spawned **only my own**
`sleep` child with the matching marker/session, then verified the positive
UID/PID-birth/process-group/marker/argv TERM path. No host evaluator was run.
The exact e7 launch/sync/guard/host/backup runtime-closure hashes matched the
pending request, and `git diff b8af19b3^ b8af19b3 --check` passed.

## Blocking schema gap

The guard runbook says a Supervisor-authorized copy may change only the three
authorization fields and that `authorization` must have **exactly** the five
listed keys, with `guard_package_verifier` containing only `path` and
`sha256`. In `dispatch_one_label.py` lines 46–69, the adapter removes the
whole `authorization` object before comparing requests, then checks only the
presence/values of its expected members. An authorized JSON with additional
authorization or verifier keys therefore passes this boundary. That is not
the specified one-label, exact-schema authorization contract; the contents of
an unexpected field would not be reviewed or bound. The timestamp is also
accepted as any truthy value rather than a UTC ISO-8601 value.

**Required correction:** reject any extra or missing keys in the authorization
and nested verifier objects, parse and validate the timestamp, and add negative
tests proving extra-key and malformed-timestamp authorized copies fail before
key-path access or remote dispatch. Re-run the exact-binding and monitor
tests, commit a new candidate, and obtain a fresh independent review. Until
then the pending request, any authorized copy, and the r2 evaluator remain
HOLD. Even after static PASS, a separate Supervisor one-label authorization,
fresh Operator host preflight, external monitor installation/hash check,
continuous observation, two stable quiescent postflight checks, and independent
result review remain mandatory. Any attempted r2 evaluator launch consumes
the label; it must not be silently retried.
