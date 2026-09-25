# Independent P offline decoder mechanics: HOLD on one narrow defect

Reviewed frozen proposal-first `3bd0ba36` and packet `8429c1a2033e46936a40a7be4195419724971e48`. This is a local stdlib review of one offline decoder packet, not model/source qualification or staging/launch permission. No host, SDK, actual trace, tensors, provider call, or toolkit installation was used.

## Blocking finding and exact remedy

In `decoder_support.py:286`, a zero-exit leader is marked `PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN` before `stop_owned()` performs its final process census (`:294`). If both pipes close and a descendant is forked between the last loop census and leader exit, cleanup discovers and terminates that unexpected descendant, but never changes the success status. This violates the packet's explicit fail-closed rule for unexpected decoder descendants.

Independent real-process reproducer: close stdout/stderr; sleep 40 ms; fork; parent exits zero; child starts a new session and sleeps. With the actual 100 ms monitoring interval, all three normal and all three optimized runs reported success while their termination evidence recorded the detached child being sent TERM. The wrapper did restore quiescence in every run. This is a success-classification defect, **not evidence of an escaped live process**.

Smallest repair: if authenticated cleanup observations contain any nonleader process, retain those identities and force `REFUSED_OR_INCOMPLETE` with an explicit reason, even when the leader exited zero. Keep current termination/reaping, limits and one-shot policy. No new control framework or additional scientific gate is needed. The Implementer received the exact reproducer; the frozen original and its six failures remain preserved here.

## Independently exercised evidence

`audit.py` reads exact Git objects, never a mutable candidate import. It passes 310 explicit checks under both normal and optimized Python, where the original defect is an expected reproduced finding rather than a PASS waiver. `original-normal.json` and `original-optimized.json` retain actual commands, process identities, signals, resource observations, errors and timing.

- All manifest/runtime pins and the complete installed inventory match: 86 files, 80 NEFFs, 4,310,546,731 original bytes. The own tiny fixture preserves all 86 relative names with small synthetic contents. Every copied digest matches, every destination has an independent single-link inode, and every original's recorded metadata is unchanged. Assembled files/directories have read-only modes; extra-file, path-escape and symlink cases refuse.
- An own actual child reports one allowed CPU, the requested small test AS/CPU/file-size hard limits, and `GOMAXPROCS=1`.
- An own two-stream negative writes 1,700 bytes to each channel against a 2,048-byte combined retained limit. It refuses, retains exactly 2,048 bytes total and terminates the child.
- All own fixture descendants are quiescent after each case. No large profile or model allocation is involved.
- The Implementer's separate 16-test suite was also replayed under normal and optimized Python: all 16 pass in each. That existing suite did not detect the above late-descendant reporting race.

Reproduce original:

```text
python3 -I -B research/v2/verification/20260926-p-offline-decoder-independent/audit.py
python3 -I -B -O research/v2/verification/20260926-p-offline-decoder-independent/audit.py
```

The audit also supports `--commit <frozen-successor> --expect-fixed` to replay the identical negative after a narrow repair; this does not authorize or automatically execute a decoder.

## Other reviewed mechanics and limits

The fixed argv targets the **whole** copied rank0 `native` directory, retaining the session files and all 80 nested NEFFs. There is no arbitrary single-NEFF selection, flattening, new capture, database/UI, install, fallback or retry. Installed `2.32.0.498-b1f3998` help supports the selected directory and summary-json options; it does not prove actual ingestion or nested model matching. Archived inventory sizes are not prior hashes of original contents. Runtime copying hashes source and destination; successful postflight rehashes both and the binary.

The original shared guard and dispatcher pins remain unchanged. Exact hostname/root/support checks precede work. The authenticated original flock FD is retained throughout assembly, decoder and final idle checks and is passed to both version/decoder children. Fresh `ps` and `neuron-ls` checks run under that same lock before work and again afterward. Busy/error/unknown conditions refuse. Operator must still reserve an exclusive slot after the current full job **and attached evaluation**, authenticate staging, and establish final process/device/lock quiescence. Disabled JSON is not a runtime release token.

Ceilings remain 600 seconds overall (580-second work deadline), decoder 300 seconds plus 10-second TERM grace, 300 total CPU-seconds, one CPU and `GOMAXPROCS=1`, decoder AS 16 GiB, per-file 64 MiB, combined retained version+decoder stdout/stderr 16 MiB, and sampled generated scratch 256 MiB. The parent/version/final-idle reserves make the decoder CPU allowance stricter than the original ceiling. Floors remain 24 GiB available RAM and 12 GiB free disk. Insufficient bounded postflight time refuses rather than fabricating idle evidence.

These are process resource limits and sampled scratch checks, not a cgroup aggregate-memory guarantee or disk quota. Read-only assembled file modes and driver read-only source opens are not a hostile-executable filesystem sandbox. The fixed installed decoder is the trust boundary; no immunity from arbitrary same-user chmod/absolute-path writes is claimed. The code never writes original inputs, and only successful authenticated posthash evidence may support unchanged-input claims. Failure may leave posthash unknown and must preserve that limitation.

Original support hashes:

| Artifact | SHA256 |
| --- | --- |
| host_driver.py | a98fcf48c5ba445610177c7c32c5de45b030e16e6f990bfdd1bea6d8512c7b56 |
| decoder_support.py | 4ab61ecec601e99bebbe0baa499e0b5df2d7b5148f7fd5d9c3b1b5634a50bf27 |
| OFFLINE_BINDING.disabled.json | 84d05cc8d2f349cc02dcb88c9d07d938d359fe03b8a32ffefd38839ee196948c |
| MANIFEST.sha256.json | da6006875e172e77aff0b9aeac6e708ddf0c38211762b699872b00a20be09080 |

Read the accepted `research/v2/analysis/20260926-p-offline-decoder-plan/PLAN.md`, the complete frozen proposal/driver/helper/binding, and installed capability receipts at `research/v2/operator/20260926-trn21-profile-recursive6ef7/offline-cli-capability/`. Inventory SHA is `25b5b94bca799741cfc7e8eed35cc37cb578074aef1d338d55f1d3a352274454`; normalized help SHA is `e35c5342947390367013768b477c2a54d9035bbf039425e69d463f21564c2c12`.

Even after a mechanics repair, actual model matching/coverage remains unknown until decoded output is independently interpreted. No all-rank attribution, whole-step speed, K1/K2 benefit, internal matmul/SDPA dtype, MFU, quality or adoption conclusion follows from this review or an eventual decoder exit zero.
