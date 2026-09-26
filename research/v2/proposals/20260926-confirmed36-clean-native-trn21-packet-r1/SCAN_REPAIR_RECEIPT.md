# Author scan-observation repair receipt — disabled, pending fresh review

Date: 2026-09-26. Scope: reporting-only repair of disabled host packet
0d4c9a3c870761a971162988fd1157b6e935914c. The prior independent REFUSE is
preserved verbatim as evidence/HOST_PACKET_PRIOR_REFUSAL.md, SHA256
3363b898a0042d0634d1132ad31471e985c8dd3055ac4a331354c1ca08b0259c.
That report found that os.walk silently skipped failed directory observations.

The only runtime-code change is an onerror callback in host_driver.sizes that
re-raises the original traversal error. Failed root or nested scans therefore
do not produce acceptable incomplete totals. The existing monitor requests
guardian cancellation; persistent final observation failure reaches the
unchanged guardian's cleanup and retained-lock path. No new cleanup mechanism,
threshold, timeout, numerical waiver or authority is introduced.

Author commands, both exit 0:

```
/usr/bin/python3 -B research/v2/proposals/20260926-confirmed36-clean-native-trn21-packet-r1/test_packet.py
/usr/bin/python3 -O -B research/v2/proposals/20260926-confirmed36-clean-native-trn21-packet-r1/test_packet.py
```

Each passes 13 groups, including the previous packet checks and seven real
local process scenarios. The new checks use a 17-byte hidden interrupted
receipt, actual chmod-denied root and nested directories under UID 1000,
faithfully injected scandir PermissionError with the real os.walk, missing
root and non-directory root errors, original error identity, and monitor
SIGTERM cancellation requests. Assertions remain active in optimized Python.

New scanroot and scannested process cases activate the precise scan failure
after two separate-session TERM-ignoring leaves are alive. They use the actual
unchanged guardian with local fixture locks and shortened fixture-only timing.
Each case observes a busy competing lock while leaves live, owned pidfd KILL,
waitpid ECHILD, the causal scan error, and persistent postflight scan failure
parking with the lock held without release_ready. The test parent checks no
live recorded leaves and an empty kernel child list before disposing only its
own quiescent parked CPU fixture. This is not a host operational remedy.
The production 900-second child, 15-second TERM limit and original trn21 lock
66305:596823 remain unchanged.

Complete command outputs:

- evidence/author-scan-repair-normal.log:
  7e1442758aa61fd3eef56e01297056b3962297b445c65ca1fd3651624cd35afc
- evidence/author-scan-repair-optimized.log:
  0457779de280c727872f156adb5967e01309e15f498b96a285d15fbb4834791f

Recomputed packet identities:

- host_driver.py, 18586 bytes:
  4a062c97107e3afd9f0061824074e9f151644e204d2957dd28ddfd6b372e98f7
- MANIFEST.json:
  905eb5fb73281c4058c52f6d16f4824dd1146929fa80b64a7b9d8d7d41d61f8f
- NATIVE_BINDING.disabled.json, unchanged:
  c29fe99533c21cae045a4cd7b5116f40808feda65bdab14e5ed38e1ab0b11760

Only the driver's manifest entry changes. The disabled binding has no stored
driver hash; its future release manifest pin remains null. Its exact bytes,
all false authority fields, runtime nulls, argv, environment, limits and lock
identity are preserved. The helper937e2f23, foundation65e7eb1a, 17-input
manifest0cc05699, guardiane461ec5c, idle guardadafb54a, dispatchc71474bc,
selected proposal and science/packet specifications are unchanged.

No SSH, host access, stage, reservation, native import, full-model tensors,
training, evaluation, protected-artifact deletion, AWS action or shared state
change was performed. Prior refusal and author evidence remain historical,
not overwritten. These are Implementer tests, not independent acceptance.
Fresh independent repair review and all existing machine, resource, transport,
runtime-identity and sole-Operator release gates remain open. No RELEASE.json
exists; no native or quality pass and no launch authorization is claimed.
