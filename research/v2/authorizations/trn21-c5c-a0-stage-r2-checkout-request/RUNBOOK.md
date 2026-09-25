# trn21 c5c A0 stage-r2 checkout request — repaired HOLD-only draft

This package binds a **single possible checkout attempt**, but it does not
authorize one. `CHECKOUT_REQUEST.HOLD.json` has
`checkout_authorized=false`, `INVOCATION_BINDING.HOLD.json` has
`dispatch_authorized=false`, and there is no Supervisor checkout-only
assignment. Direct execution of `checkout_invocation.py` exits `DISPATCH
HOLD`; the remote adapter also refuses this disabled request. **Do not SSH or
extract from this package.** The earlier transfer-only assignment cannot be
reused for checkout.

The earlier disabled request at `5fc762f8` received independent **HOLD**
(receipt SHA-256
`8656f59af4ca658d558525b69ed4c2a3b8d54a382ccc88d8603f8c20a0bb1678`):
one assignment could be replayed with a different caller journal path, and
an invented 64-character review digest passed its assignment check. This
revision repairs both mechanics but is a **new, unreviewed draft**, not a
retroactive PASS or checkout authority.

The transfer Operator's terminal receipt SHA-256 is
`9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e`;
its independent transfer-result PASS SHA-256 is
`bfafcb56ff449708dc31a1ebbe9023448018ea505e37a15847ae78bc3a3a2916`.
The nine raw write/first/final distinct-process receipts reassemble into
canonical SHA-256
`13db20bb2887eb6ff1c9feda31ea8453c9252ace18ff571021f866780e7cfe86`.
That PASS accepts only the three persistent files. The extraction code is
frozen at SHA-256
`e66afb7a1e3368ed7854eaa679e2bad9a6b8529e3b793e0f8fff2c69c496785d`;
its independent **implementation-only** PASS receipt SHA-256 is
`1a05c6dab4cbae6f49bd7e58760155d26562ba39df6a85524fe4a50289679df4`.
Neither PASS is an Operator invocation or a live host preflight.

## Exact bound action and audit

The only source input is the previously persisted, mode-`0600` capsule at
`/home/ubuntu/trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz`,
104,005,787 bytes, SHA-256
`32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264`.
The only potential new checkout is
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`,
required absent beforehand and owner-private `0700` if created. The code
requires a 6,338-member safe tar scan, exact e65 shallow one-commit clean
Git, pinned `train.py`/`prepare.py` hashes, and unchanged protected
checkpoints, caches and persistent CPU tools. No old checkout or `/dev/shm`
target is a fallback. No G2 or science command is in this invocation.

The deterministic builder in `checkout_invocation.py` reconstructs the nine
receipt bundle from the archived raw files and validates it locally. It
reuses the already archived **full read-only trn21 host probe** as the exact
fresh preflight and separate postflight SSH argv. That probe reports hostname,
UID, boot and mount namespace, root ext4/parent identity, all three persistent
artifact hashes/inodes, 10-GiB free-space gate, absent checkout/science/G2
paths, STOP/label/lock/process/Neuron checks, the two checkpoint identities
and six warm-cache identities. The request pins the full historical host
identity; a reboot, namespace/mount/inode drift or any ambiguous fresh
observation is HOLD, not a reason to substitute a source or clean assets.

The one checkout SSH argv transports the exact reviewed remote adapter as
`python3 -c` source; the 69,949-byte stdin payload contains the disabled
request bytes, frozen extraction source and nine-receipt bundle. The adapter
rehashes all three **before** it can call `stage_fixed_checkout` once. Exact
preflight/postflight argv SHA-256 is
`cc1997adb5b815a739a12cbb09319f3869e08f7d8318899fea9496bee40f66a8`;
stage argv SHA-256 is
`621078d7a2563c1249500532070b921712f63a7356c50d43991e57f96eb876fe`;
stage stdin SHA-256 is
`98f91c7be055660d194ca148957ed0885c49bc8e7525c129cb23421703e8a8a6`.
The full request/binding hashes and SHA-256-of-JSON-argv rule are in
`GATE_CONTRACT.json` and `INVOCATION_BINDING.HOLD.json`. These bytes are
review targets, **not** a dispatch recipe.

## Failure path and remaining authority gates

The later Supervisor assignment must name the request's **one fixed** local
attempt-marker/journal path:
`/home/poojith-devan/.local/state/trainium-trn21-c5c-a0-stage-r2-checkout-attempt-once.jsonl`.
The parent must already be owner-private `0700`; it is not a caller-selected
scratch directory. The wrapper refuses a different caller path, opens the
fixed marker with `O_EXCL|O_NOFOLLOW`, fsyncs the new marker and parent, and
fsyncs the bound assignment as its first journal record **before any SSH,
including the read-only preflight**. A failed preflight or pre-creation stage
failure still consumes this marker. A second invocation with the same path
fails at `O_EXCL`; a different path fails before opening it. No deletion or
replacement of the marker is authorized.

The assignment must also name
`research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/INDEPENDENT_FINAL_REQUEST_REVIEW.PASS.json`
and supply its SHA-256. The wrapper opens the *actual* on-disk owner-controlled,
non-symlink regular receipt, checks its SHA-256 and structured Verifier PASS
fields against the exact final request, binding, wrapper, adapter, audit/stage
argv, stage stdin, source checkout and marker path. The request records only
the future receipt **path**; the later assignment supplies the digest, so
there is no request/review hash cycle. A missing, non-hex, mismatched or HOLD
receipt stops before marker creation or host contact. The old independent
HOLD receipt is not a substitute for this future PASS.

After these local gates, the Operator flow permits at most one checkout SSH
process.
Whether fresh preflight fails, extraction returns nonzero, raises or succeeds,
its `finally` launches a **new, separate, read-only full postflight SSH
process** and archives its transport, snapshot and any failure. A failed
postflight is itself HOLD. On any failure, leave the capsule, both CPU tools,
protected assets and any partial checkout exactly in place. Do not delete,
overwrite, rename, clean, retry the same path or advance to G2.

Before even that one checkout attempt, a different Verifier must review this
exact request, builder/adapter, argv, marker/review interlocks and
failure-path tests. The Supervisor must then issue a new checkout-only
one-attempt assignment bound to the final request and invocation-binding
hashes and the actual Verifier PASS receipt. The Operator must verify the
local breaker/STOP gate and record a fresh complete live preflight within
that one assigned flow. Merely flipping either false
authorization field changes the request/argv/hash chain and requires a new
frozen package and separate review. After any host attempt, another Verifier
must review the full result and unconditional postflight. CPU G2, native A0,
evaluation, adoption and submission retain their own separate HOLD gates.

Local Implementer checks only (no SSH or extraction on trn21):

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/authorizations/trn21-c5c-a0-stage-r2-checkout-request/test_checkout_invocation.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest research/v2/authorizations/trn21-c5c-a0-stage-r2-extraction/test_stage_r2_checkout.py -v
```
