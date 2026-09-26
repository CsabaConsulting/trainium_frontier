# Author implementation — local CPU only

Selected by `41872565`, implemented on `codex/clock36-batch-port` in an
isolated worktree. This is not independent acceptance or launch authority.
K2 and submission qualification retain host priority. No host was contacted.

`train.py` SHA256:
`eeb3c403b86fbc8344864c96e746d9b4f02f8012e68cf7f33812d7549eea5f9c`.
The retained `parent.py` is byte-identical to
`7419927c7417b5666a984ec9c2b849feadabaaa9:workflow/proposals/research-v2/train.py`,
SHA256 `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`.
Removing exactly five lines restores those parent bytes. The five-line patch
ports only the global declaration, integer option, positive guard and assignment
from `4e361d15`; default131072 and the existing modulus check remain unchanged.
All code and receipts here are proposal-local. Canonical training/submission,
prepare/data/evaluator, selector, launchers and state were not edited.

## Reproduction and scope

Use the existing `/home/poojith-devan/venv/bin/python` (torch2.13.0+cpu):

```
OBSERVE=1 /home/poojith-devan/venv/bin/python research/v2/proposals/20260926-clock36-batch262144/run_checks.py
/home/poojith-devan/venv/bin/python research/v2/proposals/20260926-clock36-batch262144/audit_receipts.py
```

The runner uses existing G2 CPU shims, synthetic data, a fake clock and four
gloo ranks. The actual model is eight layers at width128 (4,523,316 parameters),
with 2x128 synthetic inputs per rank, so actual synthetic tokens/update are
1024 and2048, not the production131072 and262144. Production arithmetic is
checked separately by executing the source's existing modulus/accumulation
and token-count AST. The frozen113256724-parameter model was never allocated.
No downloaded data or package installation was needed.

`receipts/` retains the initial uninstrumented run and
`observed-receipts/` contains the final detailed run. Commands and exit codes
are recorded. Each set runs normal and optimized Python; PYTHONOPTIMIZE is
also propagated to gloo children. The initial runner was extended during its
first execution to select the optional observer for the subsequent run;
its hash file is a final file snapshot, not proof of initial runner bytes.
The final observed run uses the frozen supplied runner and observer.

`observe_tiny.py` derives from the existing initial-cache observer, removing
the autograd.grad/retain_graph probe entirely. It adds read-only parameter
gradient hooks and records each micro backward contribution, differentiable
cache identity, gradients before/after clipping, optimizer state and parameters
after each update. Hooks return None. There is no graph retention, detachment,
extra cache refresh or altered source numerical behavior.

The parser suite checks default/explicit131072/262144; zero, negative and
noninteger rejection; invalid196608 at frozen pack32; ramp-off arithmetic;
and a four-rank float64 reference for the actual extracted micro-loop mean
gradient. `test_cache.py` uses the source's real Linear cache and micro loop
with two distinct BF16 microbatches, comparing their exact accumulated gradient
against the sum of independent half-gradient references. It then performs
one tiny clip/update. This component arithmetic check is not a whole-model
independent gradient oracle.

The unchanged G2 comparator checks three-step parent/default losses, ownership
plan and all65 final parameter digests. The additional receipt audit checks
exact parent/default per-step gradients, updates and optimizer state; six
actual backward contributions per parameter in the batch arm; differentiable
caches; one observed clip/update per optimizer step; and distinct nonzero
second-micro gradients. No backward-through-freed-cache error was observed;
the test preserves any such error and exits nonzero if it appears.

## Results

All12 three-step tiny CPU arms (six plain, six observed) exited0 and reported
PASS. Both observed parent/default comparisons are bit-identical across all65
parameters and each recorded update, gradient and optimizer state. Both parser
and component-normalization suites pass normal/-O. The receipt audit passes
194 checks in each Python mode. Batch262144 records six forward/backward calls
and three updates per rank, with both micro contributions present. Invalid
196608 is rejected; removing the five lines recovers the pinned parent exactly.

## Remaining gates

These are author tiny-CPU results only. Fresh independent review and its own
test, exact-size CPU/ordinary-checkpoint gates, native cold start, matched
timing and final BPB are pending. The existing G2 checkpoint smoke uses its
synthetic protocol and is not ordinary production checkpoint qualification.
No timing/speed/BPB/adoption/submission claim follows. The original proposal
and its authority bits remain unchanged, and no host reservation or staging
packet was created.
