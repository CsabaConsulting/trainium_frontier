# G1 r1 pre-dispatch transport failure analysis

Date: 2026-09-24  
Role: Analyst only  
Evidence head: `e1550203a61c53cd7c73b0c543ced8a157312612`  
Disposition: **FIX LAUNCHER TRANSPORT; NEW R2 AUTHORITY CHAIN; NO R1 RETRY**

This report is documentation only. It changes no launcher, request, authorization, gate source,
candidate, host, key, or experiment state. It performs and authorizes no SSH, host access, cleanup,
retry, G2 action, full A1 run, or submission. No private-key contents were opened, read, hashed,
printed, copied, linked, or committed during this analysis.

## Finding

The archived r1 failure is a deterministic controller-side launcher defect, not a G1 numerical,
candidate, gate-program, Neuron, or remote-host result.

At authorized head `149cc9124ff0b9306eff9747182079ab708fc4ed`,
`research/fresh_start_launch.py` imports `SSH` from `research/fresh_start_sync.py` and invokes:

```text
[*SSH, HOSTS[request["host"]], <remote Python command>]
```

The imported constant is exactly:

```text
ssh -i /home/csaba/.ssh/trn2.pem -o BatchMode=yes -o ConnectTimeout=12
```

Thus `/home/csaba/.ssh/trn2.pem` was selected because it is a literal in the reviewed source, not
because of the request, the current user's `HOME`, an environment variable, or SSH host config.
The exact pinned hashes are:

- `research/fresh_start_launch.py`:
  `872c29d6703be64e3bcfdae4deff99acb459eca804f5a7bb574ac6d788e4a80c`;
- `research/fresh_start_sync.py`:
  `ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c`.

On this controller the literal `/home/csaba/.ssh/trn2.pem` is absent. The already-present team-key
path is `/home/poojith-devan/Desktop/Trainium/trn2.pem`; stat-only inspection found a regular,
single-link, current-user-owned file with mode `0600`. Its contents were not accessed. Historical
commit `5ee1c1f28356623746fb4d46952c307f4d0a5208` records that this repository-root key was reduced
from mode 0664 to 0600 and must never be printed, copied into agent context, or committed.

The r1 dispatch at evidence head `e1550203` returned 255 with the missing-identity warning and
`Permission denied (publickey)`. `REMOTE_SCRIPT` never executed, no guard or
torchrun process started, no native gate record or checkpoint was created, and the post-failure
host observation found no device holder or relevant process. The already-staged bundle and clean
remote checkout were preserved. Therefore r1 is `FAIL_PRE_DISPATCH_TRANSPORT` and is not an
experiment result.

## Why G0 could succeed with the same defect

The G0 dispatch archived at evidence commit
`37aa3644e63e20a1427cfa69b44fd6de7e25cd81` used the same launcher family. Its dispatch stderr
contains the same missing `/home/csaba/.ssh/trn2.pem` warning, but it returned zero and started the
remote guard. Local OpenSSH expansion for the r1 argv has `IdentitiesOnly no` and lists default
user identities. Consequently OpenSSH was allowed to fall back to an agent/default credential
after the explicit identity was absent.

The exact fallback credential accepted for G0 is not captured in the durable dispatch, so it must
not be guessed. G0 proves only that some fallback credential was accepted at that time. G1 proves
that no available fallback credential was accepted at its dispatch. This nondeterministic fallback
is the reason the hard-coded defect was latent; it is not evidence that the launcher transport is
correct.

## No pre-existing supported override

There is no supported override for `fresh_start_launch.py` at `149cc912`:

- its CLI accepts only the concrete request path;
- the concrete request has no local identity-file field;
- the request's `environment` is applied inside `REMOTE_SCRIPT`, after SSH has already connected;
- `GIT_SSH_COMMAND` does not control a direct `subprocess.run(["ssh", ...])` call;
- the current user's SSH config has no matching Trainium host stanza or `IdentityFile`, and relying
  on default/agent fallback would still be outside the reviewed request contract;
- changing SSH config, wrapping/monkeypatching the imported `SSH` value, or manually invoking raw
  SSH would bypass the exact reviewed launcher bytes.

The canonical control-plane autoloop has a separate safe key-path implementation: repository-root
`trn2.pem`, stat-only regular-file/mode validation, and `IdentitiesOnly=yes`. It is useful as an
implementation and test pattern, but it is a different code path and is not an override for this
launcher.

Do not repair this by copying or symlinking the key into `/home/csaba`, copying it into a new
worktree, changing global/user SSH config, relying on an agent/default identity, or bypassing the
launcher. Those actions would leave the reviewed source defect in place and make the credential
used by a run non-deterministic or unaudited.

## Minimal compliant repair

An Implementer should make a one-file, transport-only repair to
`research/fresh_start_launch.py`:

1. Add a required controller-local identity-path argument to the standard launcher. The future
   Operator supplies the already-present absolute path
   `/home/poojith-devan/Desktop/Trainium/trn2.pem`; the key is not copied into the request-bearing
   worktree or repository.
2. Validate the path with `lstat` only before transport: absolute path, present, regular file,
   not a symlink, current-user owned, one hard link, and exact mode `0600`. Never open, read, hash,
   serialize, or print the key contents.
3. Construct the local SSH argv from that validated path and add `-o IdentitiesOnly=yes` alongside
   the existing BatchMode and timeout controls. This removes fallback to agent/default identities
   and makes the selected credential explicit.
4. Keep the existing host allowlist, exact request parsing, bounded 90-second local dispatch,
   exclusive dispatch receipt, and remote script unchanged. Record at most non-secret transport
   metadata (selected path and stat validation result) in the local receipt; never key bytes or a
   key digest.
5. Stop importing the broken `SSH` constant into the launcher. `fresh_start_sync.py` may remain
   byte-identical for this minimal G1 repair because it is not the G1 dispatch path; its own
   hard-coded transport remains a separate defect and must not be used as a substitute launcher.

This is a launcher code change. There is no request-only, environment-only, or operator-only repair
that preserves the reviewed launcher contract.

## R2 authority and identity requirements

No new scientific hypothesis is needed: G1 remains the exact five-update, observation-ON hybrid
gate against authenticated G0. A new completed **transport-retry proposal/amendment** is still
required because the launcher is protected, r1 was attempted, and all non-overwrite identities
must be fresh. It must record r1's archived transport failure and declare the launcher transport
change as the sole operational change.

The next attempt must use at least:

- label `fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2`;
- remote
  `/home/ubuntu/trainium_freshstart_20260924_v2_a1_native5_g1_hybrid_observation_on_trn21_r2`;
- compile cache `/home/ubuntu/neuron_cache_20260924_v2_a1_native5_g1_trn21_r2`;
- r2 output, report, child-log, runtime-result, runtime-job and dispatch paths derived from that
  label; and
- a new bundle name and exact request-bearing head.

The r1 remote checkout already exists, its local dispatch identity was consumed, and the contract
requires initially absent non-overwrite paths. Even though training never started, r1 must not be
retried under the old label or paths. The r1 bundle, checkout, request, dispatch, and failure receipt
remain immutable evidence; no cleanup is implied or authorized.

Required sequence:

1. Implement the transport-only launcher repair and r2 proposal/template/finalization/test updates
   on a new clean branch; leave candidate, prepare, gate sources, observer, comparator, ownership
   manifest, and G0 evidence unchanged.
2. An independent Verifier reviews the exact diff and transport tests, rehashes every protected
   path, replays the immutable G0 comparator result, and issues a new r2 draft PASS.
3. A distinct state-only r2 authorization pins the new clean request-bearing HEAD, r2 request
   template, repaired launcher hash, unchanged scientific hashes, new finalization hashes, r2
   label/paths, and all downstream false authorities.
4. A different independent verification receipt validates that authorization before
   materialization.
5. Materialize a new O_EXCL mode-0600 r2 concrete request from the exact external literals, then
   independently verify its bytes, inode identity, schema, command, environment, fingerprints,
   label and paths.
6. Only a separately authorized Operator may invoke the repaired standard launcher once with the
   validated external key path. G2, cleanup, pruning, full A1 and submission remain locked.

## R1 identities that are historical and non-reusable

The following remain valid evidence of r1 but cannot authorize r2:

| Identity | R1 value / receipt | R2 disposition |
|---|---|---|
| Request-bearing head | `149cc9124ff0b9306eff9747182079ab708fc4ed` | superseded by repaired exact head |
| Request template | `32f69599970589c440f1679c3497f836f552b3fe86ba5af8bb14b9bd916b909d` | changes for launcher hash and r2 identities |
| Authorization | `f3cee7fd7b5f3a10eebaf6b457b5f58b13ea5fe9e4a8a9b6f912285d8a67f41e` | consumed; launcher/hash/head/label binding obsolete |
| Concrete request | `b1d7d037c34c9ef21b9ad093633e2d710b130c366bc214a912bfb2592a3318f0` | consumed r1 request; never edit or reuse |
| Dispatch | `6bc760190b86ab95f2c6844794063d34d682e2b609679360071d63b98c69fa44` | preserve transport failure only |
| Authorization verifier | commit `59d5a7452e5669a3f64076b3515de1ad6406da1a` | historical r1 PASS only |
| Request verifier | commit `53888cf0820ed025ff38e126c69ad52fa872f1f0` | historical r1 PASS only |
| Draft verifier | commit `7ee3be474d509e2a8b350b128ed02649702c86a2` | rerun because a protected launcher/finalization package changes |
| Failure archive | commit `e1550203a61c53cd7c73b0c543ced8a157312612` | controlling r1 failure provenance; must be cited, not replaced |

The authorization's protected-finalization hashes for proposal, runbook, request template,
pending requests, tests and authorization template also require new r2 values. The finalizer and
sync hashes require new values only if their bytes change, but they must still be independently
rehash-verified. Any receipt that asserts the exact 149cc tree or r1 label is not transferable.

The following scientific evidence remains reusable and should be repinned unchanged:

- comparator r4 commit `994d302b0036c56aaab63797f0561797bc4cf431` and comparator SHA-256
  `00f1a7263bf7367fb7ab7568822396ff04839b726c9565bb1fd6d0d70ecca17c`;
- G0 PASS receipt commit `8add2585d30b11ac2c50fb2701957657a1a6f30d`, receipt SHA-256
  `c4da83d6fe19fddab548ef49cb1a2d2eb69146e37bc36f452bcf37b446fc8c68`, and corrected replay
  SHA-256 `7198d39b9c21f1eb7e47f84e52398ca85d239ad564a36e6d524674d83355ab4a`;
- G0 evidence commit `37aa3644e63e20a1427cfa69b44fd6de7e25cd81` and its authenticated artifacts;
- production candidate, prepare, observer, hybrid/ordinary gate sources, ownership manifest,
  comparator and model-checker hashes, provided independent rehashing confirms no byte change.

## Required independent tests and review

The r2 verifier must, without network or key-content access:

1. prove the implementation diff is transport/finalization documentation/tests only and that no
   candidate, prepare, evaluator, gate science, seed, flags, five-update contract, G0 evidence, or
   comparator bytes changed;
2. mock the subprocess boundary and require the exact SSH argv to contain the selected path,
   `IdentitiesOnly=yes`, BatchMode, connect timeout, pinned host, and unchanged remote script;
3. prove request/environment fields cannot select or replace the local identity;
4. prove absent, relative (if absolute is required), symlink, non-regular, wrong-owner,
   multi-link, and non-0600 identity paths fail before SSH;
5. instrument file APIs so validation passes by `lstat` only and fails if code opens, reads, or
   hashes the private key;
6. prove no default/agent fallback remains and no key bytes/digest enter stdout, stderr, request,
   dispatch, exception text, or durable receipts;
7. preserve exclusive dispatch receipt behavior and test success, SSH 255, timeout, and malformed
   request paths fail closed without remote-script or training claims;
8. run `py_compile`, `git diff --check`, all G1 package/gate/ownership regressions, exact G0 replay,
   finalizer mutation tests, and deterministic concrete-request reproduction;
9. verify r2 label, remote/cache/output/report/log paths are initially absent in the request
   contract and differ from every r1 path; and
10. independently rehash the repaired launcher and every source/finalization file named by the new
    template and authorization before granting any Operator disposition.

## Final verdict

**FIX_LAUNCHER / NEW_R2_CHAIN.** The already-present local key can be used safely only through a
reviewed launcher capability that validates its path without reading it and forces
`IdentitiesOnly=yes`. There is no supported r1 override. Do not retry r1, reuse its authority,
copy/symlink the key, mutate SSH config, or bypass the launcher. No scientific or training
advancement is authorized until the repaired r2 request and authorization independently pass.
