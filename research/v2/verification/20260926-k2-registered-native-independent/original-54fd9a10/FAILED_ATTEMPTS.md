Two failed reviewer harness attempts are preserved without altering frozen inputs.

`checker-normal-attempt1.log` / `review_checker_attempt1.py`: the orchestration
mock returned code 0 with no group lifecycle events. The reviewer assertion failed:
`RuntimeError: cleanup after initialized outcome`. The mock supplied argv only to
the executed AST namespace, but the stdlib argparse module reads real sys.argv.
Consequently the checker correctly stayed disabled and never initialized a group.

`checker-normal-attempt2.log` / `review_checker_attempt2.py`: added pre-classification
output, confirming `code: 0`, `error: null`, `events: []`. The same reviewer assertion
failed. No candidate source change occurred.

The successful `review_checker.py` additionally patches real sys.argv inside the
mock execution context. `checker-normal-attempt3.log` and
`checker-optimized-attempt1.log` then pass all seven orchestration cases. Exact
failed sources remain available for reproduction; the errors were not erased.

The validator review exits 1 in both modes intentionally because it found four
false accepts in the frozen validator. Those are substantive review failures,
not a broken harness. Positive controls and 26 of 30 expected classifications
pass. The complete synthetic counterexamples and raw outputs are retained.

The first read-only inventory command also tried `sha256sum MANIFEST.json` and
reported `sha256sum: MANIFEST.json: No such file or directory`. The actual supplied
filename is `MANIFEST.sha256.json`; its authenticated hash is recorded throughout.
This filename typo did not affect any input or test result.
