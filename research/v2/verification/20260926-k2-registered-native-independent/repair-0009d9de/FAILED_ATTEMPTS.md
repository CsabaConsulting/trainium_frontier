# Preserved failed attempts

`normal-attempt1.log` contains the complete first execution, including its traceback. The reviewer's own signed-zero test searched for a nonexistent fixture named `zero`; the frozen fixture is named `signed_zero`. This caused `StopIteration` before that test could call the validator. The exact initial source is retained as `reassess_attempt1.py`.

Only the review test's fixture selector was corrected. No packet source, reference fixture, validator, input, or expected result was changed. The completed runs use `reassess.py`. Earlier successful observations in the failed attempt are not substituted for the completed normal and optimized runs.

All expected rejection exceptions are retained in the raw logs, including the missing runtime field's fail-closed `KeyError`. No failed attempt or candidate error was discarded.
