# Direct-wiring Neuron component gate

This component gate does not authorize training, submission, or a scientific verdict. Hardware
execution must use the repository fresh-start launcher/guard, host lock, preflight, bounded command,
cleanup and postflight.

## External authorization is the trust root

The checkout cannot authorize itself. Both static and full modes require two literal values:

- `--authorized-head`: the exact 40-character HEAD authorized by the independent Verifier.
- `--authorization-sha256`: the exact SHA-256 of
  `neuron-component-authorization-request.json` at that HEAD.

The same literals must occur in the environment and manifested request. The harness requires
`git rev-parse HEAD == --authorized-head == --expected-commit`; ancestry is insufficient. It also
requires the authorization-request bytes to equal the external SHA exactly. Therefore every later
commit—including a harmless descendant or a descendant that recomputes its own request SHA—is
rejected when the original external literals are used.

The current request has `template_state: pending_external_verifier_literals`. Full mode always
rejects that state. Neither this runbook nor any script in the checkout computes, prints, fills, or
guesses the two authorization literals. A role/status string inside the request is not authority.

The independent Verifier's final out-of-band report is the only permitted source of:

1. the exact finalized checkout HEAD;
2. the exact finalized authorization-request SHA-256;
3. the complete generated request and exact `fresh_start_launch.py` command.

Do not commit that final report on top of the authorized checkout: doing so creates a descendant
whose HEAD is intentionally rejected. Preserve the report externally with the Operator record.

## Hard-coded workload contract

These values are constants in `neuron_component_direct.py`; the mutable authorization request must
equal them byte-for-value before any test:

- host `trn22`; hostname `ip-172-31-32-253`;
- remote `/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_component`;
- timeout argv exactly `timeout --foreground --signal=TERM --kill-after=60s 1200s`;
- torchrun prefix exactly `/home/ubuntu/workspace/native_venv/bin/torchrun --standalone
  --nproc_per_node=4`;
- seed 42, host threads 4, world/local-world size 4 and LNC2;
- compiler flags exactly `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`;
- exact PATH/NKI/thread/unbuffered/fallback environment and complete ordered script argv;
- candidate SHA `c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537`;
- unique label-derived cache/report paths and exact protected fingerprint map.

Both launch and job manifests must equal the independently authorized request as parsed JSON, with
only the launcher's six fixed augmentation fields. All complete argv positions, order, environment
keys/values, request keys, paths and hashes are exact. A `9999s` timeout fails even if someone
changes the request, commits it and supplies that descendant's recomputed SHA.

Before candidate import, every protected path must be tracked and clean at exact HEAD. Actual bytes
are compared with the request's fixed SHA map for candidate, harness, runbook, proposal,
launcher/guard/host/sync helpers, launcher-default schema inputs and the prior independent report.
The authorization file itself is pinned only by the external literal, avoiding a self-hash cycle.

## Deliberately non-executable pre-authorization template

This block documents the two required external fields but cannot launch. Operators must not replace
the sentinels themselves; they must use the complete command copied verbatim from the final
independent-Verifier report.

```bash
AUTHORIZED_HEAD='PENDING_EXTERNAL_VERIFIER_LITERAL_DO_NOT_FILL'
AUTHORIZATION_SHA256='PENDING_EXTERNAL_VERIFIER_LITERAL_DO_NOT_FILL'
echo "No hardware authority: obtain the complete exact command from the final independent Verifier report." >&2
exit 64
```

There is intentionally no request generator here. In particular, forbidden constructions include
`git rev-parse HEAD` for the authorized value and `sha256sum`/`hashlib` over the current request for
the expected SHA. The final Verifier must build the exact request independently, embed literal
values in all of these locations, and publish it without deriving them at Operator runtime:

- request `expected_commit` and `authorized_head`;
- command `--expected-commit`, `--authorized-head`, and `--authorization-sha256`;
- environment `TRAINIUM_COMPONENT_EXPECTED_COMMIT`, `TRAINIUM_COMPONENT_AUTHORIZED_HEAD`, and
  `TRAINIUM_COMPONENT_AUTHORIZATION_SHA256`;
- request `authorization_sha256`.

The final workload command still begins with the hard-coded timeout array and is dispatched only as
`python research/fresh_start_launch.py <verifier-supplied-request-path>`. The request path and exact
launch command must come from the external report, not this template.

## Finalization and execution sequence

1. Implementer commits this pending template and stops. No hardware authority exists.
2. Independent Verifier reviews source, negative tests, protected hashes and the hard-coded contract.
3. If satisfied, the Verifier changes only the template state to
   `finalized_by_external_verifier`, commits that exact request, and records the resulting HEAD and
   request SHA externally. The change requires another hash-map review because the request bytes
   changed; protected source hashes must remain unchanged.
4. Without adding a descendant commit, the Verifier constructs the complete label-specific request
   using the two recorded literals and publishes the exact request bytes/hash and launch command in
   the external final report.
5. Operator checks out that exact HEAD, verifies the externally supplied request SHA, and executes
   only the verbatim command. No recomputation or substitution is allowed.
6. After the guard exits, Operator retrieves launch/job/process/component/postflight/log receipts
   with `fresh_start_sync.py`. The component report, request, process command and postflight must all
   contain the same authorized HEAD/request SHA and exact argv. Missing postflight is incomplete,
   never pass.

Any later commit requires a new independent review and new external literals. A descendant passing
its own newly computed HEAD/SHA is not an authorized command, even though those values describe
that descendant; this runbook and the final Verifier report must never emit such a command.

## Numerical and repeated-execution gates

Every comparison requires finite candidate/reference tensors, a nonzero reference above the
declared minimum, nonzero candidate norm, bounded norm/error ratios and elementwise agreement.

| Quantity | atol | rtol | min ref L2 | max error/ref L2 | candidate/ref L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP output | 0.015625 | 0.01 | 1 | 0.01 | 0.98–1.02 |
| MLP input gradient | 0.00025 | 0.02 | 0.000001 | 0.02 | 0.90–1.10 |
| MLP up/down gradient | 0.002 | 0.02 | 0.00001 | 0.02 | 0.90–1.10 |
| CE loss | 0.00002 | 0.00002 | 1 | 0.0001 | 0.999–1.001 |
| CE logit gradient | 0.000002 | 0.0002 | 0.00001 | 0.0005 | 0.99–1.01 |
| Optimizer parameter | 0.0000001 | 0.000001 | 0.001 | 0.000001 | 0.99999–1.00001 |
| Optimizer update delta | 0.0000001 | 0.00001 | 0.00000001 | 0.0001 | 0.99–1.01 |

The production-trigger MLP executes twice with a real update/cache refresh; both CE grids execute
twice; OwnedOptimizer performs two schedules and proves replicated parameters, complete state and
round-trip after update two. Static adversarial checks reject zero, attenuation, permutation,
corruption and nonfinite values. None of these results alone establishes full-model performance,
checkpoint validity, BPB, reviewer acceptance or submission readiness.
