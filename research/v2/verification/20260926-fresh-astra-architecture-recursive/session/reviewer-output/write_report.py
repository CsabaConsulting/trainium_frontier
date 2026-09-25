"""Assemble the review decision from preserved evidence; no additional tests."""
import collections
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

root=Path('/work')
art=root/'artifacts'
hashes=json.loads((art/'input_hashes.json').read_text())
test=json.loads((art/'exactness_summary.json').read_text())
receipt=json.loads((art/'receipt_audit.json').read_text())
source=json.loads((art/'source_audit.json').read_text())
manifest=json.loads(Path('/packet/INPUT_MANIFEST.json').read_text())
assert test['status']=='PASS' and test['failed_checks']==0
assert all(x['passed'] for x in receipt['checks']+source['checks'])
input_rows='\n'.join(f"| `{Path(x['path']).relative_to('/packet')}` | {x['bytes']} | `{x['sha256']}` |" for x in hashes if 'path' in x)
source_rows='\n'.join(f"| {name} | `{manifest[key]}` | `{manifest['source_sha256'][name]}` |" for name,key in [('baseline','baseline_commit'),('candidate','candidate_commit'),('schedule_reference','schedule_reference_commit')])
scope='Source and bounded/reduced-CPU suitability of candidate fe0969fd311d654e143dc6113614416c8970a424 for the next exact full-size CPU qualification only.'
pending=[
    'Exact full-size 8x1024, 113256724-parameter / 65-tensor CPU qualification of candidate default, explicit OFF and ON, including production group/ownership layout, actual tensor finiteness and reload/logit evidence.',
    'Production sequence 1024 / pack 32 / total batch 131072 / four LNC2 ranks and full gradient-history validation; supplied receipts use only 64 actual synthetic tokens per update.',
    'Native Neuron/NKI compilation, dynamic beta2 transport/recompile behavior, device arithmetic, host runtime and cold startup.',
    'Real timing/startup allowance/checkpoint accounting and 1800-second budget behavior.',
    'Real-data training, evaluator/causality/quality evidence, adoption and submission review.'
]
findings=[
    dict(id='F1',severity='verified',location='source/candidate.py:199, 3666, 3742',finding='Module and CLI default OFF; BooleanOptionalAction supports explicit OFF/ON. Exact extracted CLI parsing and assignment passed.'),
    dict(id='F2',severity='verified',location='source/candidate.py:3308, 3376, 3435, 4100',finding='ON schedule functions match supplied reference exactly; OFF arithmetic matches baseline for tested identical step/time inputs. Warmup, actual binary64 thresholds, optimizer-kind selection, batch scaling and flat precedence verified.'),
    dict(id='F3',severity='verified',location='source/candidate.py:2644, 2973, 3014',finding='ON compiled-body route receives a changing FP32 tensor containing rounded host binary64 1-beta2; OFF receives legacy float unless the old beta2 ramp is active. Eager path consumes current group beta2.'),
    dict(id='F4',severity='verified',location='source/candidate.py:3221, 3226, 3228, 3242, 3271',finding='Toggle is saved and restored; missing old-checkpoint toggle resets to captured False even after prior ON. Serialization exercised only with tiny test fixture; load_for_eval source unchanged.'),
    dict(id='F5',severity='verified',location='source/candidate.py:1933, 2196, 4023, 4137, 4158, 4931',finding='Whole 11-region diff inspected. Model/forward/evaluator, optimizer math, group construction, cache refresh, timer resets and token accounting unchanged. Main AST matches baseline after removing only approved schedule statements.'),
    dict(id='F6',severity='nonblocking_documentation',location='source/candidate.py:2658, 2387, 3438',finding='Existing beta2-transport docstrings still describe MUON_BETA2_FINAL > 0 as the tensor-path condition and do not mention RECURSIVE_SCHEDULE. Executable condition at 2973 is correct. No source modified.')
]
decision=dict(decision='PASS',scope=scope,default_until_evidence='REFUSE',
    completed_at_utc=datetime.now(timezone.utc).isoformat(),
    source_hashes=manifest['source_sha256'],source_commits={k:manifest[k] for k in ['baseline_commit','candidate_commit','schedule_reference_commit']},
    input_hashes=[x for x in hashes if 'path' in x],findings=findings,
    own_test=dict(status=test['status'],checks=test['checks'],failed_checks=test['failed_checks'],runtime_seconds=test['runtime_seconds'],
                  environment=test['environment'],code='/work/exactness_test.py',raw_cases='/work/artifacts/exactness_cases.json',summary='/work/artifacts/exactness_summary.json',
                  negative_controls='31 intentionally incorrect alternatives/nonidentity controls detected as expected',tolerances=test['tolerance_policy']),
    historical_failures=[dict(run='run01',type='reviewer_harness_extraction',exit_code=1,detail='StopIteration: searched for ast.For, actual training loop is ast.While. Corrected extractor only; preserved code/traceback/empty cases.',path='/work/artifacts/run01')],
    receipt_audit=dict(checks=len(receipt['checks']),failed_checks=0,default_and_off_losses_equal=True,default_and_off_65_rank0_digests_equal=True,on_changed_rank0_digests=65,
                       limitations=receipt['limitations'],path='/work/artifacts/receipt_audit.json'),
    source_audit=dict(checks=len(source['checks']),failed_checks=0,path='/work/artifacts/source_audit.json'),
    pending_gates=pending,authorization=dict(next_full_size_cpu_qualification_suitability=True,full_size_gate_complete=False,host=False,native=False,cold=False,quality=False,adoption=False,submission=False),
    report='/work/REPORT.md',commands='/work/COMMANDS.txt')
(root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')

report=f'''**PASS — limited to source and reduced-CPU suitability for the next exact full-size CPU qualification.**

Candidate `{manifest['candidate_commit']}` meets that narrow review scope. I began at REFUSE, reconstructed and authenticated the supplied sources, inspected the whole change and relevant execution context, compared receipt fields rather than accepting their PASS labels, and wrote and ran an independent bounded numerical test. No blocking candidate defect was found in this scope. This decision does **not** complete or authorize host/native/cold/quality/adoption/submission gates, and no full-size qualification was run.

The evidence supports default/explicit-OFF arithmetic identity at identical step/time inputs, faithful ON schedule arithmetic relative to the frozen reference, and the reviewed toggle/load plumbing. It does not prove identical production trajectories under different real wall clocks, compiler behavior, or full-model quality.

**Source and input identity.** Full-context diffs reconstructed 4,948 baseline lines, 4,980 candidate lines and 4,890 schedule-reference lines. All reconstructed source hashes and all seven manifest-listed file hashes/byte counts matched. The manifest itself was also hashed. Commit IDs below are supplied labels; no repository/history was available to authenticate commit reachability.

| Source | Supplied commit | Reconstructed SHA256 |
|---|---|---|
{source_rows}

| Packet input | Bytes | SHA256 |
|---|---:|---|
{input_rows}

The full compact candidate diff is [candidate.diff](/work/artifacts/candidate.diff); the detailed authentication record is [input_hashes.json](/work/artifacts/input_hashes.json). Original packet files were not changed. Reconstructed source is under `/work/source`.

**Findings and reviewed locations.**

- Default and CLI handling are correct: [module default](/work/source/candidate.py:199), [BooleanOptionalAction](/work/source/candidate.py:3666), [assignment](/work/source/candidate.py:3742). Default, explicit OFF, explicit ON, and ON followed by OFF were parsed using the actual extracted action and assignment.
- [LR](/work/source/candidate.py:3308), [momentum](/work/source/candidate.py:3376) and [beta2](/work/source/candidate.py:3435) functions are structurally identical to the supplied schedule reference and numerically agree in the tested cases. ON retains step warmup, clamps schedule progress, uses Muon/Adam warmdowns 0.95/0.65 and floor 0.05, applies the reference momentum cooldown over the step ramp, and raises beta2 from 0.95 to 0.97. The actual momentum comparison is `p > 1.0 - 0.95`, where the binary64 threshold is `0.050000000000000044`. At step 1, the exact threshold retains `0.8503333333333333`; its next representable successor produces `0.95`. This inherited precedence/discontinuity was preserved and tested, not smoothed into a different schedule.
- [Group updates](/work/source/candidate.py:4100) preserve initial LR and batch-scale multiplication order. Muon gets its schedule; non-Muon groups, including RMSProp tables, get the reference's Adam schedule. `no_lr_decay` takes precedence with the existing step-warmup multiplier. OFF uses the same multiplier for all kinds. [Group construction](/work/source/candidate.py:2196) and [ownership copying of live hyperparameters](/work/source/candidate.py:4714) are unchanged.
- [Compiled beta2 selection](/work/source/candidate.py:2973) adds only the authorized schedule condition. The existing [scalar staging](/work/source/candidate.py:2644) forms `1.0 - beta2` in host binary64 then rounds to FP32. ON passes a changing tensor; ordinary OFF keeps the Python float; the old explicit beta2 ramp still stages a tensor. [Eager beta2 use](/work/source/candidate.py:3014) reads the current group value. Tiny CPU trajectories passed byte comparisons for FP32 and BF16 computation, both eager and the compiled-body dispatch route. The latter executed `_muon_math` directly on CPU; no compiler was invoked.
- [Toggle registration/default capture](/work/source/candidate.py:3221), [restoration](/work/source/candidate.py:3228), [serialization](/work/source/candidate.py:3242) and [evaluation loading](/work/source/candidate.py:3271) preserve the model/load semantics. Actual save/load and restoration were tested with six test-only tensor elements: saved OFF/ON survives, missing old metadata resets to False after ON, unknown keys are ignored, and malformed toggle type raises the existing error. Full-model eval loading was inspected, not executed. Added checkpoint metadata means identical model digests do not imply identical checkpoint files.
- The entire diff contains 11 changed regions. Model construction, forward/loss/evaluator code, optimizer mathematics, base LRs, architecture, data paths, [initial cache refresh](/work/source/candidate.py:4062), [timer initialization](/work/source/candidate.py:4023), [startup reset](/work/source/candidate.py:4137), and [token accounting](/work/source/candidate.py:4158) are unchanged. The complete main-function AST equals baseline after removing only the approved schedule additions and restoring the LR variable name. This corroborates the manual diff review; it is not the numerical test.
- Nonblocking documentation issue: beta2-path docstrings at [line 2658](/work/source/candidate.py:2658), [line 2387](/work/source/candidate.py:2387) and [line 3438](/work/source/candidate.py:3438) still describe the old `MUON_BETA2_FINAL > 0` condition without the new toggle. Executable routing is correct. No source was edited.

The static parameter census matches 113,256,724 parameters / 65 tensors for the specified 8x1024 configuration, and 4,523,316 / 65 for 8x128. It assumes the packet configuration and padded vocabulary 8192; this is arithmetic consistency, not full-model instantiation evidence. Production tokens/update calculate as `1 × 32 × 1024 × 4 = 131072`. All 12 source-audit checks passed; details and source locations are in [source_audit.json](/work/artifacts/source_audit.json).

**Independent bounded test.** [exactness_test.py](/work/exactness_test.py) was written for this review. It extracts exact real functions, optimizer methods and the contiguous schedule/group-update statements without importing the full training module. A test-only class shell provides optimizer state/cache dictionaries; `print0` is suppressed and native NKI is disabled. No mathematical body is replaced. A separate Decimal oracle, precision 60, computes piecewise analytical expectations without calling candidate/reference schedule functions. Frozen baseline/reference execution supplies strict exactness comparisons; the independent oracle supplies a separate numerical correctness check.

The successful run completed **6,263 checks, zero failures**, in **1.950247551 seconds inside the script** (3.961109821 seconds process wall time). Python **3.14.4**, PyTorch **2.13.0+cpu**, CPU only, one intra-op and one inter-op thread; BLAS/OpenMP thread limits were all one. No training model or worker was instantiated. Only one tiny optimizer fixture was live at a time; largest parameter was 3x2. Dispatcher tracking covered fixture tensors and temporary tensor results, conservatively counting shared-storage views repeatedly: **522 bytes peak**, maximum single storage 24 bytes, far below 64 MiB. This is tensor-storage tracking, not total process RSS or internal library workspace measurement.

Inputs cover warmup 0/7, steps 0/1/6/7/150/299/300/777, negative/zero/late progress, adjacent representable values around both warmdown starts, progress above one, nonpositive/tiny/1800-second budgets, legacy OFF knobs, all three group kinds, flat/nonflat groups, and batch scales 1 and sqrt(2). Tiny six-update optimizer fixtures cover tall/wide/square and transposed matrices, FP32 and BF16 Newton-Schulz math, eager and CPU compiled-body dispatch, ordinary OFF and the old beta2 ramp. Initial parameter/gradient formulas are recorded in `trajectory()`; raw records preserve all compared parameter bytes, moment bytes, scalar operands and outputs. Thirty-one negative controls detect wrong floor/kind/linear momentum/ramp precedence/literal threshold/early beta2 ramp/flat precedence/stale beta2 or unintended ON=OFF identity.

Tolerance is **zero** for frozen-source binary64 comparisons and parameter/state byte equality. The independent high-precision schedule/group oracle uses **absolute 5e-16, relative 0**, reflecting differences in algebraic evaluation/rounding; this latter check is not labeled bit exact. All inputs, expected/actual outputs, contexts and tolerances are in [exactness_cases.json](/work/artifacts/exactness_cases.json); extraction line ranges/hashes are in [extractions.json](/work/artifacts/extractions.json). The final failure list is empty in [exactness_failures.json](/work/artifacts/exactness_failures.json).

One failed harness attempt is preserved in [run01](/work/artifacts/run01/exactness.stdout): the extractor searched for a `for` loop and raised `StopIteration` before numerical execution; the source uses a `while` loop. I corrected only the extractor to accept both. No candidate defect, tolerance relaxation, expected-value change or discarded numerical failure was involved. Final code/outputs are also archived in `artifacts/run02/`. Its stderr contains the library diagnostic `Error in cpuinfo: failed to parse processor information from /proc/cpuinfo`; execution completed with exit 0. Both logs and exit codes are retained.

**Supplied raw receipts.** The file named `tiny-g2-reference.json` identifies the baseline hash, not the separate schedule-reference source. All four receipts have the expected source hash, four rank IDs, three finite losses/rank, zero recorded process exit code and no timeout. All ranks report 65/65 changed parameters, no unchanged parameters and empty cross-rank mismatch lists. The 65 rank-0 digest keys are identical across cases.

| Comparison with baseline receipt | Exact recorded losses | Rank-0 parameter digests | Plans |
|---|---|---|---|
| Candidate default | 12/12 equal | 65/65 equal | Equal hash and summaries |
| Candidate explicit OFF | 12/12 equal | 65/65 equal | Equal hash and summaries |
| Candidate ON | First two updates equal on each rank; third differs | 65/65 differ, expected | Equal hash and summaries |

Common plan hash: `83b06c658067902fe9e255779ce792e5117006563c932b16bc1f4bba6f6595d5`. Owner element counts are `[1097728, 1097728, 1081376, 1246484]`, totaling 4,523,316. Each rank-0 receipt records checkpoint existence, successful deterministic reload and logits shape `[1,16,8192]`. All **116 field-audit checks** passed; complete loss arrays, changed-digest names, flags, shims and comparisons are in [receipt_audit.json](/work/artifacts/receipt_audit.json). Copies of supplied raw JSON are in `/work/artifacts/supplied_raw`.

Limits matter: receipt-byte authentication is not independent proof of execution provenance. Only rank 0 supplies final digests, each 16 hex characters; no raw parameter tensors, complete ownership plan, checkpoint bytes or logits were supplied. Thus I independently verified finite losses and consistency/equality of recorded fields, but cannot independently verify all parameter finiteness, the recorded cross-rank mismatch computation, reload determinism, or recompute the plan hash. These are reduced diagnostics, not completed full-size G2 evidence. Synthetic 1x16 microbatches across four ranks mean **64 actual tokens/update** despite retained production-style counters. Fake timing, test prepare/data, CPU execution and no native compile invalidate speed/cold/real-data conclusions. The log phrase “COMPILED body” means the CPU dispatch route here, not a native compiler result.

**Pending gates and authorization limit.**

{chr(10).join('- '+p for p in pending)}

The passing result supports taking this frozen candidate to the next separately bounded exact full-size CPU qualification. None of the pending gates is waived or claimed complete. The next full-size run was not launched in this restricted session.

Exact executed test commands, exit statuses, runtime and replay instructions are in [COMMANDS.txt](/work/COMMANDS.txt); `bash /work/run_tests.sh` reproduces the checks sequentially. [decision.json](/work/decision.json) records the machine-readable verdict, scope, hashes, findings and pending gates. [artifact_manifest.json](/work/artifact_manifest.json) hashes the final scripts, report and raw evidence.
'''
(root/'REPORT.md').write_text(report)
inventory=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name!='artifact_manifest.json':
        data=p.read_bytes()
        inventory.append(dict(path=str(p.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(root/'artifact_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(json.dumps(dict(decision='PASS',scope=scope,artifacts=len(inventory),report_bytes=len(report.encode()),checks=test['checks']),indent=2))
