# Author receipt-persistence repair — independent review still required

Bounded Implementer work on 26 September 2026, from exact implementation
commit ff94bae1ca5cb7531c5895c4b8f661fe4069d12a in the preserved clean worktree
/home/poojith-devan/clean36-native-impl. The fresh REFUSE report at
/home/poojith-devan/clean36-independent-review-m8vrKn/REPORT.md was authenticated
as SHA256 8c7d16096079fa140e07e15979ce96c6cf82c9c601f29c825a26c98e85744218.
The refused probe was 73c84e13ce8f9b92b1c74f4b4b3f5063be03cf73919cfc8e738174042c23c96d.
That refusal remains historical evidence; this author receipt does not replace it.

The only executable repair is proposal-local probe.py. It now publishes a
rankN.partial.json before work and updates it at saved-state, arm, case, and
call boundaries. A case enters its arm before execution; each call enters the
case before execution. Before a model call the durable snapshot identifies the
arm, shape, call, CPU and device input hashes, prior outputs and completed calls.
Output capture and completion after purity checks are distinct boundaries.
Failures preserve the active case and completed records. No signal handler or
Python finally execution is required to retain the previous snapshot on TERM.

Each replacement streams bounded JSON to a private same-directory temporary
file, flushes and fsyncs it, atomically replaces the partial, then fsyncs the
directory. Initial rank reservation and final publication use exclusive hard
links; existing final/partial/temporary evidence prevents reuse. Final receipts
are rankN.json with receipt_state FINAL; they are never overwritten. The partial
is kept until final publication and temporary cleanup succeed. An interrupted or
failed final publication can leave both names; the offline validator refuses
that set. Cleanup unlinks are not additionally fsynced: crash-resurrected cleanup
names cause conservative refusal, while the final data/link was already synced.

The bound includes the newline: each file is at most CAP//3 = 5,592,405 bytes.
At most a partial, one temporary and one final name coexist, totaling at most
16,777,215 bytes per rank, below the unchanged 16 MiB cap. A killed write can
leave its bounded private temporary file. Serialization/cap/write failures keep
the last intact partial, fail closed and do not substitute a tiny success record.
Retrieval must include partial and hidden temporary evidence after an abnormal
exit; this local helper does not implement host retrieval or a new launcher.

Causal failures retain the original FP32 subtraction/max error, unchanged 0.01
threshold, cut and prefix range, exact prefix-byte observation, complete prefix
input/output hashes, relative RMS including the zero-reference treatment, and
maximum-error index. Nonfinite causal errors remain failures and are represented
as strings in JSON. Case/error records keep every completed call. Extra metrics
run on existing CPU copies; scientific computation and the gate are unchanged.
Small unequal-prefix differences within 0.01 still pass this causal criterion.

The six shapes, 30 calls per source, seed 42, scientific source/checkpoint pins,
foundation, ordinary loaders, no-hook checks, evaluator, native initialization,
and disabled host binding are unchanged. Partial receipts remain UNRESOLVED;
LOCAL_COMPLETE is assigned only after successful work and process-group cleanup.
The offline validator also requires explicit complete case/call states and a
FINAL receipt marker. These are receipt-completion checks, not new science gates.

## Author CPU checks

Runtime: /home/poojith-devan/venv/bin/python with the existing CPU torch.
Both commands were run from the worktree, with no native import, full model,
full-width output/mask, host access, data read, evaluator, training, or AWS action:

```
/home/poojith-devan/venv/bin/python -B research/v2/proposals/20260926-confirmed36-clean-native-consistency/test_probe.py
/home/poojith-devan/venv/bin/python -O -B research/v2/proposals/20260926-confirmed36-clean-native-consistency/test_probe.py
```

Both passed all 22 tests: normal 4.871 s, optimized 6.365 s. The logs here are
the complete final command outputs. Tests include actual SIGTERM from inside a
tiny CPU model call and immediately before atomic replacement, completed-arm
and call preservation, exact future-leak magnitude/hash/range retention, causal
tolerance and overflow behavior, replace/cap/nonfinite/directory-sync failures,
exclusive final receipts, symlink/reuse rejection, cleanup failure, and rejecting
partial markers even when renamed. The subprocess cases execute real main and
persistence with a tiny CPU replacement body; they do not execute native run.
The existing mutation, byte, mask, payload and structural tests remain.

The first local ten-test replay failed only because the old metadata-only
positive fixture lacked the newly explicit COMPLETE fields. That fixture was
updated; no gate was relaxed. Final 22-test results above supersede the earlier
intermediate 18/19/21-test author runs. Default invocation still reports DISABLED
without model imports, and git diff --check passes.

## Exact hashes

- probe.py: 937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af
- test_probe.py: 2c2830ae3f3c6124d43588995efba51d594fcec249b9b9e81181d5f2bd7011b8
- author-repair-normal.log: 0ff5d59d0eb8c4a0454d4bf593509799eee5cc8a933b06b220b9058cc1589b9a
- author-repair-optimized.log: ee4e7807bab12c68201cb790bbe41f258688c9fb4c2cfa7e67b39b23e7f126e4
- Unchanged REVIEW_SPEC.md: 1b5a311e04904872dbdf614037a2def9ef45a5362c173d555c058e66eb7f599d
- Unchanged EXPERIMENT_PROPOSAL.json: 76f90b5db0cbf9d8d903e71ad6ce7d3bf05e484056a10f3a6011579f2fec5cf8
- Unchanged BINDING.disabled.json: fb9806e681aa79d554aee3a914391c088c4e34207955819c895967307c7842cd
- Unchanged CPU foundation: 65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8

Fresh independent source/CPU review remains required. Native behavior, actual
four-rank receipts, graph/resource cost, host binding/guardian integration,
timeout retrieval, and descendant quiescence are untested by this repair.
No native pass, host reservation, staging/release, BPB gain or submission
readiness is claimed. All binding authorization fields remain false.
