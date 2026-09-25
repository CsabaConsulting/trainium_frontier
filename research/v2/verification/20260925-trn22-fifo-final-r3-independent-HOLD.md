# Independent trn22 FIFO final-r3 transfer review — HOLD

Reviewed on 2026-09-25 by a Verifier separate from the Implementer. Exact
subject: `d571ea61b9fd2c3c7a21f254e044338ff873edba`, pushed at
`origin/codex/20260925-trn22-r4-transfer-lock-final-request-r3`. The
Implementer checkout was clean and inspected read-only. This receipt and its
synthetic test are on a separate reviewer branch. Repository instructions
were reread in the required order. The r1/r2 packages, disabled `5740a0e0`
subject, their historical reviews/HOLDs and the five required instruction
files are unchanged.

**Verdict: HOLD for final one-shot transfer readiness.** The r3 safety repair
passes the bounded offline checks below, but there is no evidence that its
post-final-label host checks can reliably finish inside the mandatory one-
second label-to-`O_EXCL` window on trn22. A fixed independent final PASS file
was deliberately **not** issued. No Supervisor assignment, marker, journal,
SSH, host contact, transfer, checkout, native run, evaluation, cleanup or AWS
action occurred. The request's `dispatch_authorized: true` remains intent
only, not authority.

## Static safety finding and operational blocker

The new remote source rechecks checkpoint and `/dev/neuron0` holders,
`neuron-ls`, the held cooperative-lock identity, all fixed STOP paths and
the process inventory **after** the final label scan. It repeats the STOP
path sweep after the process walk. The final scan-to-open monotonic bound
remains one second. A persistent STOP or Neuron holder appearing during the
slow scan, or a post-scan checkpoint holder, matched process, lock replacement
or delayed probe, prevents target creation in synthetic tests. Transport
comparison rejects absent, busy, stale, reordered or mismatched post-scan
receipts. Local marker/Popen STOP repairs, source-FD binding, exact authority
chain, distinct verify and failed-write read-only postflight remain present.

The same r3 remote sequence at
`stage_r4_remote_lock_final_r3.py:740-784` serially runs **two `fuser`
subprocesses, one `neuron-ls` subprocess, no-follow lock/ACL identity,
`/proc` process inventory, and two complete STOP-path scans** after the
final label scan. Each subprocess independently permits a 20-second timeout;
even much smaller ordinary delays can exhaust the aggregate one-second
limit. The exact code correctly fails closed before `O_EXCL` if that happens.
However, `operator_transfer_lock_final_r3.py:283-284` consumes the fixed
`O_EXCL` marker and journal **before** the SSH probe/write sequence, and the
request forbids retry or same-path replacement. Thus a timing miss can
irreversibly consume the one attempt without staging the archive. That is an
operational gate for an exact one-shot final request, not a safety PASS.

The archived trn22 read-only lock feasibility record has full-probe SSH
durations of 6.374 and 6.368 seconds (plus a 4.647-second lock diagnostic),
but **does not break out** label-scan versus post-scan probe time; it cannot
prove the new one-second substage succeeds or fails. No host-side timing
distribution, worst-case bound or margin for the exact r3 sequence exists.
The independent local fixture shows that three merely 350-ms probe delays
consume the real one-second window and safely prevent synthetic `O_EXCL`.
Those delays are a sensitivity test, **not** a measured trn22 latency claim.
No final transfer-ready PASS can be inferred from 15 fast mocked tests.

For a repair, keep the expensive holder/Neuron observations before a final
label scan while holding the existing exact `device0.lock`, then place a
bounded quick STOP/process/lock recheck and `O_EXCL` in one short critical
section. This ordering alone would **not** catch an uncooperative Neuron
holder arising during the scan: retain a demonstrably fast post-scan
Neuron/holder observation or add a shared interlock that such activity obeys.
Do not silently drop a required gate for speed. The held flock atomically
excludes only cooperating launchers; same-UID/root and noncooperating manual
activity remain outside the accepted proof. No finite ordering of unsynced
snapshots eliminates the tiny race after its last check; the repair should
bound and state that residual instead of repeatedly moving a slow check to
the end. A separately authorized read-only timing study of the exact
post-scan sequence, with repeatability and margin, could also resolve this
feasibility HOLD without transfer or marker creation.

A narrow Operator timing assignment could use a separately reviewed,
read-only diagnostic that authenticates the exact r3 remote SHA, trn22 host
key/identity and idle preflight, then opens the existing `device0.lock`
no-follow/read-only and acquires/releases its nonblocking flock, as in the
prior lock feasibility check. While holding it, each of a fixed small number
of trials (for example, 20) would call the exact r3 final-label observation
and `_post_final_label_safety()` helpers **without calling `write()` or
creating/opening the archive**. Monotonic timings should cover the entire
scan-completion-to-end-of-post-scan sequence and each fuser, `neuron-ls`,
lock/ACL, `/proc` and STOP component; record raw outputs, result and all
failures, then release the lock and perform a full read-only postflight after
the trials. A single under-one-second sample is inadequate: the Supervisor
must set a conservative worst-observed-time margin before interpreting the
receipt, and a separate Verifier must check it. The diagnostic must have no
marker, journal, transfer stdin, target `O_EXCL`, retry or native operation;
its own exact source and SSH argv need independent review before assignment.
Even a clean timing study is feasibility evidence, not transfer authority.

## Exact bytes and offline evidence

| Evidence | SHA-256 |
| --- | --- |
| Final-r3 request | `92c0adc015da75e3e6825c8003b1dbaaf1d3a8997a7ccee833507fcbbce7648f` |
| Final-r3 core | `a0d38c7e4c6e2ea93b369c469558ed156b0a9187b4040af8ca567eda3453914a` |
| Final-r3 wrapper | `ae8eb66fabafa33f8732a8f29f14a86e9d3e713abc57ea6316c0c0d2d7f24fae` |
| Final-r3 transport | `a12b8b202e18a8e809355c1fe0499ac7b2b68a94c5766ddb383f331f07b0c6c1` |
| Final-r3 remote | `fcf6f59e19e168a986de155338457d7b209a7bc2131674eea0841f7112ea0d5a` |
| Final-r3 runbook / supplied test | `3a4b04ccf493068172ee80ab12ebb0fcfc877c4136e2de634462ff85c1029845` / `eee0aa3111b7737dbfdb1767d27e5e3344ae71b0ba26fea016b5f369025f6c1e` |
| Historical r2 / r1 HOLD receipts | `b2c68151d8dc9ace6d08d8a0bc4d1828133bd2342eb240cbfe8ba25f5fbeed6b` / `8d1bf8caa6fa0643b067c9242cbeed1e8adb756592bbedc274fefcc328ca04ad` |
| Historical disabled-static review / remote | `67c3e430bd14623b4f11a305f9a445ad6d3276e6e63833a693d004aa94988f6d` / `cdf06f700722e144dcb304e517a7e000fe0c6b2bcb210e4d2bb38f92757d5e27` |
| Source-stage contract / review | `79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69` / `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e` |
| 104,154,875-byte capsule / local known-hosts | `767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a` / `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3` |
| Verifier synthetic test | `3a42f05061e8dc7f99c29627a3a8e0f62be2345367d56b5904117c23f21d0442` |

The package files and origin commit were independently rehashed. The local
source checker verified the exact same-FD capsule bytes/mode; local ED25519
known-hosts/fingerprint verification passed without reading private-key
contents or contacting the host. The fixed final-r3 review, Supervisor
assignment, marker and journal were absent.

Supplied r3 tests passed **15/15** normally and **15/15** under `python -O`.
Verifier tests passed **5/5** in both modes, including an eight-case
activity/latency synthetic matrix, authority mutations, no-follow evidence and
post-scan receipt tampering. Historical unchanged suites passed 14/14, 4/4,
11/11, 9/9, 13/13 and 9/9. Optimized standalone core, transport, remote and
alternate-operation entrypoints refused dispatch. `git diff --check` passed.

The safety checks above are static and synthetic. They are not a fresh host
preflight, an operational timing proof, a transfer result or native-launch
approval. This exact final-r3 request remains **DISPATCH HOLD**.
