# Fresh independent fixed-job lifetime review

Verdict: **ACCEPT** for the frozen fixed-job lifetime repair and disabled transport repin delta only. No launch, stage, native, scientific, score, adoption, or submission authorization is granted.

Reviewed exclusively the frozen packet `/home/poojith-devan/clean-fullcold-lifetime-fresh-TOCGAp`, beginning with SPEC.md. Did not inspect Implementer workspace, notes, tests, or surrounding repository state; did not contact any host. Author gate receipts were not substituted for independent tests.

Exact reviewed SHA-256 pins:

- host_driver.py: `23c8ad2f6037a6f479e0049473686146bc249b1edf535427725e107da1ac5061`
- lifetime_guard.py: `e461ec5c76c784fd51471c9e8c268cdde42fe1def68b31bfc9524480f980c312`
- NATIVE_BINDING.disabled.json: `2d71861029094a7f2e8a4ee640fa0c450bc9650f65142f7bd38854fbe7a16a72`
- stage-dispatch.py transport supplement: `94524dd3c7fd25f01952147f766bb0c6ecd75274ae0b198847a41d656782c338`
- transport.patch: `12e8aca43c134f1dd909cbe5928bbcf20d2cc5cc9b012f027e97f2f635807b5e`
- transport local factual receipt: `0663385911513ea2d2d1befd8debb71688fd488f9620372bce75284641fe6484`
- scientific train.py: `e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43`
- protected rehearsal: `7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`
- package long launch argv: `b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd`

All remaining packet pins are in `reviewed-pins.json`. The supplement references frozen commit `13c90714ba9cdfe5df788cae1ed5607c23afcca5` and manifest hash `93b9df5252e3a0df541f4151ff5b43fdb930b51395ed61edf492a53a95c1a1d9`; the manifest file itself is not in this packet and is not independently byte-authenticated here. Acceptance covers the displayed repin/staged-guardian/release-pin changes, not an execution of transport or proof of its external prerequisites.

Independent tests authored here: `verify.py` and `gates.py`. Both were run normally and under `python3 -O`, using explicit unconditional `check` failures. Results: `normal-results.json`, `optimized-results.json`, `gates-0.json`, `gates-1.json`; per-case raw event and worker logs remain in the corresponding directories. The gate test deliberately compiles extracted production assertion statements at optimize=0, matching the production driver's explicit rejection of optimized execution.

The exact protected `process = subprocess.Popen(command, cwd=sandbox, env=env)` statement was obtained from the protected harness AST and executed with a harmless bounded CPU child. The old behavior reproduced in both modes: harness exited while its new-session child survived and a real flock competitor acquired the lock. Repaired cases independently passed normal exit, harness crash exit 7, actual foreground timeout exit 124, caught SIGTERM, helper timeout plus driver exception, postflight failure, and synthetic ENOSPC during audit writes. The TERM-ignoring new-session grandchild required owned-pidfd KILL and was reaped before release. Competitors remained excluded throughout active descendants, phase gap and postflight; unrelated sibling processes survived. ENOSPC and postflight failure retained the original lock after quiescence. Parked test guardians were disposed only after independent confirmation that their bounded child was absent; that test disposal is not an operational remedy.

Static review confirms SIGINT/SIGHUP/SIGQUIT share the tested SIGTERM handler and cleanup path. Helpers are launched only after subreaper installation, use bounded synchronous subprocess.run calls, and are included in final child-quiescence checks; the helper-timeout test independently confirmed orphan adoption and cleanup. New sessions do not bypass kernel parent-child ownership. PID/start-time checks plus pidfds avoid numeric-PID signaling of unrelated jobs. ECHILD provides the actual lineage-empty criterion. Cleanup exceptions and unknown postflight state park while retaining the inherited open description; audit-write failures are swallowed into a fail-closed audit flag rather than interrupting containment. Driver finally invokes cleanup even when terminal/log receipt writes raise. Initial no-job refusal does not perform the external postflight idle wait and no new idle-check exemption was introduced.

Source and harness byte hashes match the specification. AST equality independently establishes unchanged TRAIN_COMMAND/EVAL_COMMAND/TRAIN_ENV/EVAL_ENV between old and new drivers. Long argv matches the disabled binding; protected accounting/evaluator code is not edited by the diff. prepare.py is not provided as bytes here: its unchanged required hash remains enforced in the driver; no fresh remote authentication is claimed. Real local flock tests rejected an unlocked descriptor and wrong inode, accepted the held original identity. Extracted actual disk gates rejected one byte below and accepted exact boundaries at 13.5 GiB training and 10.5 GiB evaluation. Final production acquisition explicitly pins device 66305/inode 832589, again before each phase. Failed training cannot reach eval because its exit/budget/checkpoint assertions precede the eval path and optimized driver execution is rejected.

Scope limit: guardian SIGKILL, fatal native guardian failure, and host death are outside the guarantee. The guardian is not enclosed in a timeout and must not be externally killed while its descendants exist. CPU mechanics do not prove native launch readiness or competition quality. The receipt's pending gates and disabled authorization fields remain controlling.

No source edits were made. An import-generated bytecode cache from the first test was moved into this output directory; subsequent imports disabled bytecode writes. No packet source bytes changed.
