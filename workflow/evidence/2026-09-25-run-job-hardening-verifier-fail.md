# Independent verifier: `run_job.py` hardening

Verdict: **FAIL — production pilot blocked.** The legacy dense launcher is unaffected.

Reviewed implementation: `b83fe4b630259251e4bffe80a23d87598814a0a3`  
Base: `8b9fc841ed817eac263c7087c3bce7c9f008bda1`

The supplied end-to-end suite passes, and the patch materially improves source, metric, archive,
ledger, and checkpoint binding. Three independently reproduced lifecycle/evaluation defects still
permit a leaked workload or an invalid evaluation to become a successful record.

## Blocking findings

### R1-B1 — an exception after `Popen` leaks the workload and releases the host lease

`launch_and_wait()` starts the child at `workflow/tools/run_job.py:360`, then creates the runtime
directory and writes two launch receipts at lines 362–367. Those post-launch operations are not in a
cleanup `try/finally`. A clean tracked `runtime-train/` directory is not rejected by preflight. It
makes `runtime.mkdir()` raise after the workload has started; the exception is not a `JobError`, no
stop/quiescence runs, and `host_lease()` unwinds while the tagged process is still alive.

Independent reproduction precreated `runtime-train/`, invoked `launch_and_wait()` with a 120-second
sleeping child, caught `FileExistsError`, and observed:

```text
{'exception': 'FileExistsError', 'live_tagged_after_exception': [254801]}
```

The verifier killed and reaped that test child. Fix by rejecting every reserved stage path before
launch and, more importantly, placing every operation after successful `Popen` under a bounded
stop/kill/quiesce `finally` on any exception. Add deterministic mkdir/write/fsync failure tests and
require zero tagged survivors before the host lease can be released.

### R1-B2 — eval and causality outer timeouts are accepted as success

`launch_and_wait()` returns `None` on timeout. Training handles that explicitly, but evaluation and
causality use `if erc:` and `if crc:` at lines 739 and 752. `None` is false. A process can print one
valid JSON object and then hang past the outer timeout; the runner stops it, parses the earlier JSON,
archives/prunes, appends the ledger, and emits a normal verdict.

The independent end-to-end fixture printed exact public-20M JSON, slept past a 2-second outer eval
timeout, and produced:

```text
rc=0
record_status=evaluated
verdict=NOMINATE
eval_intervened=True
anomalies=['eval: tagged descendant outlived leader and containment stopped it']
```

An outer timeout is a failed gate, not an approved straggler anomaly. Handle `rc is None` explicitly
for both eval and causality after bounded cleanup and before parsing output. Add output-before-hang
leader and descendant timeout regressions for both phases; none may enter the ledger.

### R1-B3 — the evaluated source/checkpoint argv is not exact

`build_eval_cmd()` appends host-config `eval_args`, then validates only one exact `--seq-len`,
`--eval-tokens`, and `--timeout-seconds`. It does not require exactly one expected `--train-py` and
`--checkpoint`, and it does not close the remaining argument schema. A mistaken or drifted host config
can append alternate values; organizer argparse uses the later values. The candidate checkpoint is
then unchanged, so the post-eval checkpoint identity check does not detect that another model was
scored.

Independent reproduction was accepted and returned:

```text
checkpoint_values=['/repo/out/L/final.pt', '/tmp/unrelated.pt']
train_py_values=['pinned/train.py', '/tmp/unrelated.py']
```

Bind the complete eval argv to the verified G1/job contract. At minimum require exactly one
`--train-py` equal to the pinned tracked train file and one `--checkpoint` equal to this run's
`final.pt`, reject duplicate/unknown evaluation options, and add alternate/duplicate source and
checkpoint attacks.

## Checks rerun

All checks ran locally; there was no host, Neuron, AWS, training, cleanup, or submission action.

```text
python -m py_compile workflow/tools/run_job.py workflow/tools/check_causality.py workflow/tools/fake/*.py  PASS
bash -n workflow/tools/test_run_job.sh                                                        PASS
git diff --check 8b9fc841..b83fe4b6                                                         PASS
bash workflow/tools/test_run_job.sh                                                          PASS
```

The supplied suite ended with `PASS rows=7 no_leaked_processes`, but it contains no post-`Popen`
receipt/setup failure test, no eval/causality output-before-timeout test, and no duplicate alternate
eval source/checkpoint test. The three additional verifier attacks above all reproduced.

`workflow/tools/decide.py` remains unchanged at
`a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66`.

This receipt does not review or authorize a host pilot. A fresh independent review is required after
all three blockers and their negative tests are committed.
