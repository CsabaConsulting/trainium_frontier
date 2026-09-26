# Fresh independent disabled host-packet review

Verdict: **REFUSE** for the pinned host-driver mechanics. Native/host authority remains **HOLD, NOT EXECUTED**.

Fresh independent Verifier session, requested as Astra; independently attested runtime model identity is unavailable. Date 2026-09-26. Input was exclusively the frozen 20-file packet `/home/poojith-devan/clean36-native-trn21-packet-fresh-O2iY7F`, plus canonical operating-contract reading. No implementer branch or test source was read. Factual prior helper refusal, repair acceptance and unchanged guardian review were considered evidence, not substituted for tests. No packet/canonical repairs, host access, staging, native imports, full model/full-width tensors, training, data, evaluation or authorization changes occurred.

## Finding

**P1 — artifact observation fails open on directory scan errors.** `packet/host_driver.py:206` calls `os.walk(root, followlinks=False)` without an `onerror` callback. Python's walk suppresses a directory `scandir` OSError and skips its contents. Consequently the watchdog and final postflight can accept incomplete artifact/cache/rank counts instead of treating an observation failure as a failed gate.

The independently authored `observation_failure.py` creates only a 17-byte hidden receipt `receipts/.rank0.interrupted.tmp`, injects PermissionError specifically when the real walk scans that directory, and calls the actual `sizes` and `artifact_gate`. Both normal and optimized Python return zero artifact bytes, zero cache bytes and four zero rank counts; the artifact gate accepts. This is a reproduced observation defect, not an assertion that a native filesystem failed or that a cap was actually exceeded. An inaccessible subtree could conceal arbitrary partial receipt/cache size. The same helper is used by runtime monitoring and final resource postflight, so those consumers do not detect this uncertainty. It contradicts the packet's explicit rule that observation failure fails acceptance and requests owned cleanup.

Required repair direction: make directory traversal errors propagate into the existing failure/cleanup path, and independently test failed root and nested directory observations. No repair was made here and no threshold waiver is proposed.

## Evidence that passed within scope

`reviewer_tests.py` passed 15 grouped checks in normal and optimized Python, using explicit checks that survive `-O`. All 20 packet hashes and the frozen support manifest matched. Driver constants agree exactly with the disabled binding's child/timeout/outer argv, environment, six support pins and original trn21 lock identity 66305:596823. Helper937e2f23, foundation65e7, guardiane461ec5c and input0cc05699 are unchanged. The staging tree preserves sibling foundation resolution. Disabled entry exits2 without a RELEASE file. Runtime identities and transport/release metadata remain intentionally unbound; those are future gates, not invented source defects.

Independent arithmetic confirms 100,859,904 bytes/rank for six retained FP32 original outputs and 16,777,215 bytes for three CAP//3 receipt names. Exact 13.5GiB initial disk, 10.5GiB runtime disk, 64GiB initial RAM, 16MiB rank receipt and 256MiB artifact boundaries were checked, including one-byte failures. Independently counted visible/hidden partial files and caches agree with `sizes` when traversal succeeds; symlinks reject. These logical-size checks do not prove allocated disk or native resource sufficiency.

Real local flock tests reject unheld and wrong-inode descriptors. Nine independently constructed process scenarios cover leader exit with a TERM-ignoring setsid descendant, no-job pre-spawn release, watchdog deadline, disk floor, total artifact cap, hidden per-rank cap, audit-write failure, postflight failure and unknown quiescence. The watchdog deadline was shortened to0.5s and guardian grace to0.1s only in reviewer process globals; production900+15 values were separately checked unchanged. Oversized files are sparse fixture files, not large tensor allocations. Actual owned pidfd TERM/KILL and waitpid ECHILD events are archived. The escaped descendant was killed and reaped, and leader success with surviving descendants returned125. A real unrelated sibling survived every scenario.

Persistent disk/artifact/hidden-receipt faults, audit failure, postflight failure and injected unknown quiescence retained the actual fixture lock. Competitors failed while these guardians were parked. Parked fixtures were disposed only after checking their kernel child list was empty and the recorded descendant PID absent. This fixture disposal is not an operational remedy. Deadline cleanup released only after ECHILD and a successful test postflight. No-job release was separately exercised. These tests use actual local kernel locks/children but do not freshly execute the exact remote outer command/inherited-FD transport or native ps/neuron-ls/fuser commands.

Source inspection confirms preflight fresh controller receipt/STOP checks, runtime and17-file authentication, ps plus neuron-ls plus fuser under the original retained lock, private cache creation,900-second own-child timeout and watchdog,15-second grace, and postflight input/device/resource checks. The guardian retains ownership across session escape and checks ECHILD; audit failure does not interrupt attempted cleanup. Static constants/source checks do not authenticate a current host or its resources.

Tiny CPU scalar tests use only three output channels for B1/B2/B3 × T256/T257, independently pack float/int bytes with `struct`, compare all six original output/input SHA256 values, and check30 calls. Signed-zero byte inequality and input mutation reject. This is a new limited exactness regression; the earlier broader scientific/helper acceptance is only the pinned factual review. No new broad loader/model/native numerical acceptance is claimed.

## Artifacts and identities

Input SHA256SUMS: `9d966a95dedd5bb4d1bc68514544db2b05e92e8b35d896c08499fbbef034363b`.

Driver: `132fa83ae654a3d7921ef8b907f759c6c69d2d3be11035170b56d1d98ff9c54c`.

Reviewer test source: `86b349460a7e3b221ed96116788852e75905e9459a2c1736a2f8319057212109`.

Observation reproduction source: `832ef75c767e9ded209b5b6d9a37fcc100e2b9dbc9dc148fbfa12ffcf8de3818`.

Normal and optimized grouped-test logs each: `60731953f65808250ce41cc77568b91b08aeca8a00edf6889f97cada79ab4bbd`.

Normal and optimized scan-failure logs each: `18aa7bb357bb7a88c7cd0307fa1e684a1f49d6d3dacfec6dc998ace6bc8f2232`.

Commands: `/home/poojith-devan/venv/bin/python [-O] -B reviewer_tests.py` and `/home/poojith-devan/venv/bin/python [-O] -B observation_failure.py`. Raw process events/logs and summary files are beneath `normal/` and `optimized/`. `initial-normal/` preserves an initial successful run before strengthening persistent resource postflight assertions; it is not the final suite. `SHA256SUMS` inventories review artifacts, excluding itself. No author tests were executed.

This refusal concerns the disabled packet's artifact-observation mechanics. It grants no launch, transport, native result, resource feasibility, quality, adoption, or submission acceptance.
