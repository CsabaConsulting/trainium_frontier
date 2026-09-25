# Independent trn22 final-r3 timing diagnostic review — HOLD

Reviewed subject: pushed Implementer commit
`4fd59da6d5de7db7e943b086bd1770c94409ec4f` on
`origin/codex/20260925-trn22-fifo-final-r3-timing-diagnostic-impl`.
The diagnostic pins final-r3 source commit
`d571ea61b9fd2c3c7a21f254e044338ff873edba`.
This Verifier used a separate worktree and branch. The repository contract and
required handoff files were read in order; their bytes matched the source
checkout. No EC2/SSH contact, enabled request, assignment, host write,
training, evaluation, cleanup, or submission was performed.

**Verdict: HOLD for this exact disabled package.** The supplied offline suite
cannot import, so the required normal and `-O` synthetic gate fails before any
test runs. This receipt does not authorize an Operator timing attempt or clear
the independent final-r3 transfer HOLD.

## Blocking evidence

Both `python3 -B -m unittest -v test_timing_readonly.py` and
`python3 -B -O -m unittest -v test_timing_readonly.py`, run from the package
directory with Python 3.14.4, exit nonzero at
`test_timing_readonly.py:114`:

```text
SyntaxError: closing parenthesis ')' does not match opening parenthesis '[' on line 112
```

The malformed `self.assertEqual` list prevents all eight supplied tests from
running. This is an exact-byte test-gate failure; the Verifier did not change
the Implementer package. A separate Implementer must repair it and provide a
new pinned package for independent review.

One additional evidence-retention issue merits review before enablement:
`operator_timing_readonly.py` limits the timing SSH process to 600 seconds,
while the remote bootstrap prints its JSON only after the entire 20-trial
`run()` returns. If the cumulative read-only work reaches that timeout, the
local exclusive receipt records the timeout and any SSH stdout/stderr, but
completed in-process trial/substage timings are unavailable because no partial
records have been emitted. The one attempt is still consumed. The exact
one-shot raw-evidence objective needs a bounded worst-case time argument or
durable partial timing output before this is treated as operationally ready.

## Bounded static and independent offline checks

The checked-in contract says `dispatch_authorized: false`; the exact enabled
request, independent PASS review, Supervisor assignment and fixed raw-result
file are absent. The collector calls `_authorize()` before `collect_once()`;
its bootstrap permits only `probe` and `timing`, uses `python3 -B`, and passes
`/dev/null` stdin to SSH. The timing code calls the pinned frozen
`_final_label_observation()` and `_post_final_label_safety()`; it does not call
the frozen `write()` or `verify()` functions. Its lock open is read-only,
nonblocking and no-follow; it uses an advisory flock and records a clock-only
`would_open` timestamp. These are static observations, not a successful full
synthetic or host gate.

Independent offline negative checks in
`test_20260925_trn22_fifo_final_r3_timing_disabled.py` passed **3/3** both
normally and under `-O`. They proved that a missing enabled request prevents
all subprocess calls and local result creation, that `write`/`verify` and
malformed operation shapes are rejected before transport access, and that a
synthetic timing-SSH failure retains a local failed raw receipt and invokes a
read-only postflight probe. `git diff --check` passed. These checks do not
repair or substitute for the malformed supplied suite.

| Exact subject | SHA-256 |
| --- | --- |
| Disabled contract | `e7c745294e9d7ed2655b70fd71511ada82499cb04f1fa01aa414db67b5d1a5d3` |
| Timing runbook | `cf0aab2d2ae628b08985d2c169ba345fdec2544bccf1328dae6aa1c5ce18931d` |
| Collector | `e15469f7d7cb71e247977603cf7c79d7004d57e98cd2d799fa45f7e2dc3bb43c` |
| Remote timing diagnostic | `452113ab31188bf84211f0d59c9ae78b8c27f96b8e080a673b349051a79750b9` |
| Supplied malformed test | `21fdfab28010f796b26eb9ca504b16fcca22fcf91789c413eb07dfe75f04fe19` |
| Frozen final-r3 remote / transport | `fcf6f59e19e168a986de155338457d7b209a7bc2131674eea0841f7112ea0d5a` / `a12b8b202e18a8e809355c1fe0499ac7b2b68a94c5766ddb383f331f07b0c6c1` |
| Prior independent final-r3 HOLD receipt | `34377be826b5cad75759082f94dcb7bd1539f5f93c388b4a22cedac529d87fdd` |
| This Verifier's negative tests | `bf4d58f46b7b3542a2a7802014f7389cd590f7edd75bea4746d19c8bfb735e31` |

The underlying final-r3 transfer remains DISPATCH HOLD. A new timing package
must pass fresh independent review before any distinct enabled read-only
request, review-bound Supervisor assignment, or host contact.
