# G1 r2 correctness-failure diagnosis

Date: 2026-09-25  
Role: Analyst (read-only scientific/correctness adjudication)  
Verdict: **FAIL — gate-contract implementation error; no G1 scientific result**  
Disposition: **STOP. Build and independently verify a new r3 chain; do not reuse or retry r2.**

This report records analysis only. It does not authorize code changes, request materialization,
host access, cleanup, a retry, G2, full A1, training, or submission.

## Controlling evidence

- Failure evidence commit: `5b392db4bc8c01a47da2f5969cb33df0c50ec52b`.
- Evidence label: `fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2`.
- Evidence operator receipt:
  `operator-evidence/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2/OPERATOR_RECEIPT.json`,
  SHA-256 `c6db31c323c545c851c63fb1c8313b46a0b2d0aeafc1e2a1ea8045e43acd8fb5`.
- Child log SHA-256:
  `87dedc4345bb1605e115acf995ceb5dedb56c20bfddb423343405a85632e9f4a`.
- Runtime-result SHA-256:
  `705547ff56258ac69232b0dd515d07a52056445c32ce5e513ac15613ffba8596`.
- Authorized r2 head: `4724f67d9a55781a0dc7ef0864c93e8db2ea0fa0`.
- r2 implementation and transport verifier: `a455a207f0f19603b4d3dd52bd7a93fa5a445f39`
  and `8096e47c`.
- r2 authorization and its verifier: `4724f67d9a55781a0dc7ef0864c93e8db2ea0fa0`
  and `3a5e47f917f39063885500d9a3c00bd1417e8d8c`.
- Exact request verification, rematerialization ruling, and rematerialized-inode verification:
  `65b807dab3ea6a785e44aef91fef32c5168b3879`, `2ddcddd30b5df9297735d5631513c173998d8fda`,
  and `60c108bd`.
- Exact r2 request-template, authorization, and materialized-request SHA-256 values:
  `a2d1a0a60e8672d816e0ef94ee77d04af771d70b05fc72fb416f3ee252bc7fee`,
  `3d98b81c75c020418c2b61349b8f1b42ff937416b8b605c27542eb097f877771`, and
  `dfde895a86c66a71eb6a450c4f82bacf44209c6c815fc81a368af23499e921fb`.

The preserved G0 evidence remains commit
`37aa3644e63e20a1427cfa69b44fd6de7e25cd81`; its corrected comparator-r4 PASS is independently
recorded at `8add2585d30b11ac2c50fb2701957657a1a6f30d`, with replay SHA-256
`7198d39b9c21f1eb7e47f84e52398ca85d239ad564a36e6d524674d83355ab4a`. G1 r1's separate
pre-dispatch transport failure remains immutable at `e1550203a61c53cd7c73b0c543ced8a157312612`.

## Exact root cause

The host did not mismatch. The label did.

At authorized head `4724f67d`, the request template, finalizer, authorization, concrete request,
runtime environment, and actual runtime command all name:

```text
host  = trn21
label = fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r2
```

The runtime command passes those values as `--native-gate-host trn21` and
`--native-gate-label ...-r2`; `TRAINIUM_FRESH_START_RUN_ID` is the same r2 label. In contrast,
`research/a1_native_gate_observer.py` at SHA-256
`89ad30056bf3fe207c43ea61f8c8a94583232f9f21e5158a80fd37cce42d02e7` contains:

```python
ARM_LABELS = {
    "G0": "...-g0-hybrid-observation-off-trn21-r1",
    "G1": "...-g1-hybrid-observation-on-trn21-r1",
    "G2": "...-g2-ordinary-observation-on-trn21-r1",
}
...
if host != "trn21" or label != ARM_LABELS[arm]:
    raise ValueError("native gate host/label contract mismatch")
```

The relevant literal is line 25 and the rejection is lines 77--78. The `or` combines two checks
under one message, but the supplied host equals the required host while the supplied r2 label does
not equal the hard-coded r1 G1 label. Therefore the G1 label comparison is deterministically false
on every rank.

`train_gate_hybrid.py` does not transform these values. At lines 3862--3871 it constructs the real
`NativeGateObserver`, forwarding `args.native_gate_label` and `args.native_gate_host` directly.
Initialization validates arm/mode/world first, then rejects host/label before creating the report
directory/file, installing the collective spy, binding the optimizer, or entering step 0.

The child log proves all four ranks reached the same boundary: rank 2 at lines 67--75, rank 0 at
78--86, rank 1 at 88--96, and rank 3 at 98--106. Runtime evidence contains zero optimizer updates,
zero rank reports, and zero checkpoints. Model and optimizer construction did occur, so this is not
a transport or pre-Neuron failure, but no training update or equality observation occurred.

Classification: **gate harness/contract integration defect**, not a candidate/source semantic
failure, numerical mismatch, compiler issue, host issue, or environment issue.

## Independent local reproduction

Using a detached worktree at exact head `4724f67d`, I loaded the exact request template and exact
observer source under CPU PyTorch. I extracted arm, label, and host from `train_argv`, and run ID
from `child_environment`, then invoked the real `NativeGateObserver` constructor for ranks 0--3
with a minimal fake distributed interface. No training or Neuron action was used. The result was:

```text
runtime arm=G1 label=...-trn21-r2 host=trn21 run_id=...-trn21-r2
observer_expected ...-trn21-r1
r2_rank 0 ValueError native gate host/label contract mismatch
r2_rank 1 ValueError native gate host/label contract mismatch
r2_rank 2 ValueError native gate host/label contract mismatch
r2_rank 3 ValueError native gate host/label contract mismatch
r1_exact PASS rank-0.jsonl
```

This distinguishes the two operands without changing the source: the same constructor accepts the
observer's exact r1 label with host `trn21`, while it rejects the exact r2 runtime label with the
same host.

## Why verification did not exercise the runtime boundary

The passing tests covered internally consistent islands, not their production seam:

1. `test_gate_package.py:1558-1582` constructs only a successful G0 observer and an intentionally
   invalid G2 mode. It supplies `label=OBSERVER.ARM_LABELS[...]`, so its expected value and input
   come from the same module constant. It never constructs G1 and never supplies the external r2
   label.
2. `test_g1_r2_package.py:205-283` checks `g1-request.template.json` against
   `g1_finalization.py`'s r2 `G1_LABEL`. It pins the observer's old hash as "protected science"
   but neither imports nor constructs the observer. Hash equality proved the stale bytes were
   unchanged; it did not prove that those bytes accepted r2.
3. `g1_finalization.py:473-488` proves the train argv contains `template["label"]`. It does not
   compare that label with `NativeGateObserver.ARM_LABELS["G1"]`.
4. `test_g1_package.py` manufactures synthetic G1 records from G0 records, rewriting their header
   to `adjudicate_g0_g1.py`'s label. It tests evidence adjudication, not runtime observer startup.
5. `adjudicate_g0_g1.py:285-299` intentionally translates a fresh retry label to the comparator's
   canonical r1 label. This code is never reached without real rank reports.

The r2 transport verifier expressly treated the unchanged observer SHA as protected science while
normalizing r1/r2 command fields elsewhere. Thus its "science unchanged" check preserved the stale
literal. The verification chain exhaustively authenticated transport, serialization, hashes,
request identity and authorization, but no test followed the exact production data flow
`request template -> runtime argv -> parser values -> NativeGateObserver.__init__`.

## Other stale or apparently stale contracts

- `compare_gate_reports.py:30-34` still uses the canonical r1 G1 label. This is intentional, not
  this bug: the accepted comparator remains immutable and `adjudicate_g0_g1.py` validates the retry
  label before translating only the header label to the comparator's canonical identity. An r3
  adjudicator must update the input-side label and retain/test this narrow normalization.
- G1 r1 authorization, pending authorization, drafting checklist and failure evidence are
  historical and must remain unchanged. Their r1 literals are not active r3 inputs.
- At the r2 evidence tip, `EXPERIMENT_PROPOSAL.json`, `RUNBOOK.md`, and `requests.pending.json`
  still describe r2 as pending/non-executable or absent. Those are now administratively stale and
  must be superseded for r3, but they did not cause line 78.
- `g1_finalization.py`, `adjudicate_g0_g1.py`, r2 schemas, the r2 amendment, and all r2
  label-derived paths are now consumed identities. They require r3-specific constants or a new
  r3 document; they cannot simply be relaunched.
- The observer's G2 r1 label is not currently stale because the planned first G2 identity remains
  r1 and G2 is locked. It is the same design hazard: any later G2 retry with a unique label must
  update the independent observer allowlist and execute the same cross-boundary test before
  authorization.

No second active r2 runtime label consumer was found disagreeing with r2. The observer was the
single active seam mismatch; the comparator's r1 literal is deliberate canonicalization.

## Minimal, scientific-neutral r3 repair

The repair must remain fail-closed. Do not accept a suffix/prefix pattern, derive expected and
actual labels from the same argv value, bypass the observer, or relax validation.

Recommended exact new identity:

```text
label:         fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3
branch:        experiment/20260925-v2-a1-hybrid-native5-g1-r3-finalization
bundle:        /home/ubuntu/v2-a1-native5-g1-r3.bundle
remote:        /home/ubuntu/trainium_freshstart_20260925_v2_a1_native5_g1_hybrid_observation_on_trn21_r3
compile cache: /home/ubuntu/neuron_cache_20260925_v2_a1_native5_g1_trn21_r3
output:        out/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3
report:        research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3/native-gate
child log:     research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3.child.log
runtime job:   research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3.job.json
runtime result: research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3.runtime-result.json
dispatch:      research/run-logs/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3.dispatch.json
request:       /tmp/fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3.request.json
```

The smallest code correction is one exact allowlist-literal change:
`ARM_LABELS["G1"] = <exact r3 label>`. This changes only gate instrumentation metadata validation,
not model, optimizer, data, gradient, collective, or training arithmetic. It necessarily produces
a new observer hash and must be called out as a scientific-neutral gate-instrumentation correction,
not hidden as byte-identical science.

The r3 package must also:

1. create a new `G1_R3_CORRECTNESS_RETRY_AMENDMENT.json` and distinct
   `g1-r3-authorization.pending.json` / `g1-r3-authorization.json`; never overwrite or reuse r2
   authority;
2. update the active request template, finalizer, proposal/runbook/request state, adjudicator,
   schemas/error text as appropriate, and every label-derived path to r3;
3. update every protected-source/fingerprint map to the new observer, finalizer, template,
   adjudicator, test, amendment and authority hashes;
4. keep the production candidate, `prepare.py`, hybrid gate math, seed 42, world size 4, LNC2,
   five updates, flags, comparator r4, ownership manifest, and all numerical PASS criteria exact;
5. preserve r1 and r2 remote artifacts/caches/evidence. No cleanup is implied or authorized;
6. obtain a new Implementer commit, independent implementation verification, state-only
   Supervisor authorization, independent authorization verification, O_EXCL concrete request,
   and independent exact request/inode verification before any Operator decision.

No G0 rerun is scientifically required only if independent review proves the r3 observer diff is
limited to the G1 expected-label literal and G0 computational/recording behavior is unchanged.
The explicit amendment must acknowledge that the observer file hash changed. If the observer,
hybrid wrapper, candidate, arithmetic, flags, or decision criteria change beyond that literal and
the r3 contract plumbing, the existing G0 comparison is no longer sufficient and the proposal
must be re-adjudicated rather than assumed reusable.

## Required regression and integration gates

Before r3 authorization, all of these must pass:

1. **Actual constructor / exact argv test.** Load the exact r3 request template. Extract
   `--native-gate-arm`, `--native-gate-label`, `--native-gate-host`, observation mode and report
   path from its exact `train_argv`; verify `TRAINIUM_FRESH_START_RUN_ID` from the exact
   `child_environment`; then invoke the real `NativeGateObserver` constructor for ranks 0--3.
   Run under a temporary working directory so the unchanged relative report argv creates only
   temporary files. Require four exclusive rank headers with exact r3 label/host/arm/mode/world.
2. **Independent-literal seam assertion.** Require the template/finalizer r3 label to equal
   `NativeGateObserver.ARM_LABELS["G1"]`. Do not obtain both sides from one shared test fixture.
3. **Negative identity tests.** With host fixed to `trn21`, require r1 and r2 labels to fail. With
   label fixed to exact r3, require a wrong host to fail. This separates the two operands hidden by
   the common exception text.
4. **Production forwarding test.** Statical/AST review or a small production-used constructor
   helper must prove the hybrid wrapper forwards parsed label and host unchanged. If a helper is
   introduced, production `main` and the test must call the same helper; do not duplicate logic.
5. **r3 uniqueness scan.** Require all remote/bundle/cache/output/report/log/job/result/dispatch/
   request identities to be new and initially absent, and reject every consumed r1/r2 identity.
6. **Adjudicator boundary.** Feed real-format r3 headers to the r3 adjudicator, require exact r3
   validation before the existing narrow canonical-r1 comparator translation, and reject r1/r2.
7. **Full existing suites and hash verification.** Re-run gate, G1, transport, ownership and
   py-compile suites. A test whose only failure is "authorization now exists" must be revised into
   an explicit phase-appropriate assertion; expected failures are not a substitute for a passing
   package.

A low-cost controller/CPU test at item 1 would have deterministically caught this defect before
host use. No Neuron run is needed to discriminate or verify the repair itself.

## Reusable and consumed evidence

Reusable if unchanged and freshly pinned in r3:

- production candidate SHA-256
  `9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`;
- `prepare.py`, hybrid gate SHA-256
  `908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94`, ownership manifest
  SHA-256 `8119df41c8a9ecbfc3f9e69067c7e6ac162c2d5d8ba0a15635265304dcf865a2`, accepted comparator
  SHA-256 `00f1a7263bf7367fb7ab7568822396ff04839b726c9565bb1fd6d0d70ecca17c`, and corrected G0
  evidence/replay;
- repaired launcher SHA-256
  `7620fea10bb51c54481b8347ec9e97ebd40ce8f5ee206dc3f203c9483c35ee2f`, subject to fresh r3
  pinning and verifier checks;
- r1/r2 failure archives as immutable historical provenance; r2 additionally proves the repaired
  transport reached the remote runtime and clean postflight.

Consumed or unusable as r3 authority/science:

- r2 label, branch, remote, bundle, compile cache, output, report, log, job, result, dispatch,
  concrete-request inode/bytes, authorization and all r2 launch permissions;
- observer SHA-256 `89ad...` as an executable r3 source identity;
- r2 rank reports/checkpoint/scientific outcome, because none exist;
- r1 and r2 verifier receipts as authorization for a new r3 head or request. They remain evidence,
  not authority.

## Final disposition

G1 remains unproven, G2 remains locked, and no training advancement is permitted from this run.
The evidence is decisive enough that repeating r2 would be wasteful and unauthorized. The next
legitimate action is an independently verified, fail-closed r3 package with the exact cross-boundary
observer initialization test above. Only a later, separately authorized r3 Neuron run can produce
the missing G1 evidence.
