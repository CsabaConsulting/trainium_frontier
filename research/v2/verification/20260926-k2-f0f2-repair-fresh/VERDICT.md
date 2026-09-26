# Independent repair review

Verdict: ACCEPT_FOUR_REPAIRS_CPU_MECHANICS_ONLY.

The four concrete findings in the preserved original refusal are resolved to the
extent established by source review and the independent local tests below. This
is not a blanket packet acceptance, host release, launch authorization, native
correctness result, full K2 qualification, or speed/BPB evidence. BINDING=None
remains mandatory pending a separate immutable host-specific review/release.
The old REFUSE_IMPLEMENTATION_MECHANICS_REVIEW verdict remains unchanged for its
original packet; neither its file nor any old evidence was edited.

Reviewed only frozen packet /home/poojith-devan/k2-f0f2-repair-review-EHvK9J,
manifest SHA256 4bfe88c615b53ff6143ec13739fbd676cdc0541b4522c2c6600dc8951ea49d2e.
Every manifest entry passed before and after the review. Reviewed source hashes
are retained in raw-results.json: pins. Key repaired pins:

- host_driver.py: 9a1e5ab9fc4d5ec039a743da520c011dfa2a9cb6fa682fc26fbf191f74bc643f
- coordinator.py: 3d0fe301b55a84f64722bdea644ce1a020f8af9c4f5cd52f9dc49cdea5846912
- monitor.py: 36343be8359c241724d1f07427cf8fee5f238e9150a7a85e31cab46ee01cdb34
- consumer_nki.py: 9425390c96053a1b24d4addf8b61b1a87c4993f3551f08c7cdcefcf142d84dcd
- oracle.py: 93a6309e9fec0d8680905adf963ac49b3fcb3aeca56cfc942b70dae16a7709d0

## Independently executed evidence

checks.py was newly written for this review; no author test was imported or
rerun as independent evidence. Command: /home/poojith-devan/venv/bin/python -B
/home/poojith-devan/k2-f0f2-repair-independent-tFbwTc/checks.py. It exited zero.
Raw evidence: raw-results.json, job-late/logs/terminal.json, checks.py.

1. Actual host_driver.main with host-only checks mocked, real local flock and
   injected ENOSPC or existing terminal.json: the process remains alive and the
   lock unavailable, before and after SIGTERM. Each test process was then killed
   and reaped by this reviewer. These are actual main/finally/protection_hold
   paths, not a replacement finalizer. They do not test native util-linux flock.
2. Actual bounded_run with real forked TERM-ignoring helpers: a timeout case and
   zero-exit parent with detached child both fail; actual pidfd containment and
   kernel ECHILD are recorded. A 1.2-second configured helper budget completed
   within the independent 1.5-second observation bound. The production own
   deadline check remains tighter. No test-owned descendants remain.
3. Actual main receives a late new-session TERM-ignoring child inside the
   postflight callback, after the earlier termination. Final census finds it,
   TERM/KILL containment removes it and final kernel ECHILD is recorded. Main
   returns failure from the deliberately failed preflight, not diagnostic success.
4. Four real child interpreters connect over Unix sockets. Actual SO_PEERCRED,
   /proc rank variables, argv, executable, ancestry and starttime are checked
   through actual authenticate_worker. Four distinct processes are accepted;
   every duplicate process, false rank environment and launcher-as-worker probe
   rejects. Test paths/interpreter replace host paths only as function arguments.
   The coordinator diff places this call before client receipts/scheduling and
   checks all four admitted identities live. This review did not independently
   rerun the entire mocked coordinator.main one-PID regression; its supplied
   regression remains author evidence, while the actual admission path is freshly
   exercised here with kernel peers.
5. Own Fraction exact-sum / binary-search finite-FP32-lattice nearest-even
   reference matches all 1024 F0 four-addition recurrences, and eight F2 classes
   on rows 0,127,128,4095,4096,8191 (1072 total). The reference does not call the
   candidate add32 implementation and uses no NumPy addition reference. F2
   timeline values/IDs come from the frozen scalar recipes, whose source was
   checked against SELECTED_SPEC. This is not full-shape native comparison.

supplement.py also exited zero: the actual disabled entrypoint refuses; all
seven resource/gap negative probes reject one-unit threshold violations; the
inclusive threshold control succeeds. supplement-results.json holds raw results.

## Unchanged science and limitations

Compared frozen baseline SOURCE_PINS to repaired pins: the only changed existing
Python sources are coordinator.py, host_driver.py and monitor.py. New Python
files are peer_fixture.py and test_repairs.py. Consumer, integer oracle,
fixtures, generation, worker, evidence, artifacts, common and binding pins are
unchanged. The coordinator numerical/routing bodies are unchanged; peer admission
is the repair. Host diffs preserve scientific pin, native environment, resource
thresholds, initial floors, work/termination deadlines and 1200-second total.
Monitor adds final quiescence; resource guards are unchanged. Source review
confirms the terminal release-safe guard now encloses evidence failures, tracker
initialization precedes helpers and final recensus follows postflight.

No native SDK imports, host contact, full-size allocation, launch, model edit,
source edit or external state change occurred. Only reviewer-created local
processes were signaled. The supplied normal/optimized test receipts were read
as author evidence, not relabeled independent checks. Independent checks here
ran normal Python only. No claim of universal signal-race resistance, SIGKILL
survival, hardware invocation proof or arbitrary same-UID adversary resistance.

Frozen-SDK compatibility, real four-rank native execution, complete returned
bytes, full-size time/storage feasibility, ordinary control exactness and native
containment remain unobserved. The broader selected-spec consumer-bypass and
native qualification requirements are not newly certified by this bounded repair
review. No binding/adoption/launch gate is changed by this scoped verdict.
