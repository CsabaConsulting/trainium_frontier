#!/usr/bin/env python3
"""Assemble the independent report from preserved evidence, then hash artifacts."""
import datetime, hashlib, json, pathlib, shutil, sys
W=pathlib.Path('/work');P=pathlib.Path('/packet')
def read(path): return json.loads(path.read_text())
def write(path,obj):
    with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
inputs=read(P/'INPUT_MANIFEST.json')
normal=read(W/'raw/attempt-001/summary.json')
optimized=read(W/'raw/attempt-002-optimized/summary.json')
boundaries=read(W/'raw/attempt-003-boundaries/summary.json')
for summary in (normal,optimized,boundaries): require(summary['status']=='PASS','incomplete CPU evidence')
require(not list((W/'raw').rglob('FAILED.json')),'unexpected failed test attempt: disposition must be reconsidered')
for name,expected in inputs['candidate_source_sha256'].items():
    require(hashlib.sha256((W/'source'/name).read_bytes()).hexdigest()==expected,'candidate source modified')
for name in ('attempt-002-optimized','attempt-003-boundaries'):
    receipt=read(W/'raw'/(name+'.execution.json'))
    require(receipt['exit_code']==0 and receipt['runner_error'] is None,'test did not complete')
# Preserve the unmodified packet as audit inputs, not additional evidence.
snapshot=W/'packet_snapshot';snapshot.mkdir()
for path in sorted(P.rglob('*')):
    if path.is_file():
        dest=snapshot/path.relative_to(P);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
# First attempt preceded the generic runner. Record only observed facts.
script=(W/'scripts/review_test.py').read_bytes()
(W/'raw/attempt-001.script.py').write_bytes(script)
write(W/'raw/attempt-001.execution.json',dict(command=['python3','-B','/work/scripts/review_test.py','--output','/work/raw/attempt-001'],environment_overrides={x:'1' for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE')},exit_code=0,exit_code_evidence='observed exec session 24246 completion; retrospective receipt, no reconstructed start timestamp',elapsed_seconds=normal['elapsed_seconds'],script_sha256=hashlib.sha256(script).hexdigest(),sequential=True))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
limits=[
    'CPU tests and source inspection only; no NKI import, installed API inventory, native compilation, device execution or actual distributed teardown.',
    'Process-group initialization, device construction and synchronize API lookup precede the protected body. Device and synchronize failures after init have zero cleanup calls and no rank receipt in full-main stubs; initialization failure also has no cleanup.',
    'Protected final reporting attempts destroy once; destroy can raise or hang. The two finalizer timing/description assignments precede its inner try. No blanket cleanup guarantee exists.',
    'Exclusive file creation prevents overwrite; preencoding prevents serialization-caused partial files. Writes are not atomic: I/O failure can leave partial output, and file/stdout errors can suppress the final receipt or original error. These are failures, never success.',
    'No cross-rank aggregation, actual barrier behavior, timeout behavior or launcher cleanup exercised. Logical-empty input has direct-call coverage only in intended native checker, not autograd.',
    'Synthetic metadata producer validates routing; standalone kernel is not a general metadata validator. Native shape/dtype guards are Python asserts; CPU optimized tests do not prove native guards survive optimization.',
    'Finite bounded fixtures only for recurrence; arbitrary overflow/nonfinite input support is not certified. FP32 reference conversion can raise on overflow. BF16 conversion overflow cases were representation tests only.',
    'Kernel/reference candidate hashes authenticated. Their byte equality to a separate baseline cannot be independently reconstructed from /dev/null added-file diffs; baseline identity is supplied context, not an independently inspected repository.',
]
gates=[
    'Separately bind and authorize a bounded native feasibility/correctness experiment; this report authorizes no hardware or launcher action.',
    'Inventory required NKI/backend APIs and establish compilation legality, dynamic-loop/allocation behavior, indirect-access skip semantics, active-mask broadcast, BF16 conversion, FP32 rounding, signed zero and subnormals on device.',
    'Require all four LNC2 rank receipts, strict shape/dtype/byte comparisons at atol=0 and rtol=0, direct/repeated calls, eager and compiled enclosing custom-autograd paths, CPU dense oracle and present finite gradients for applicable cases.',
    'Treat missing/unsupported APIs, mismatches, missing/nonfinite gradients, timeout, partial/absent receipts and cleanup failures as refusal; provide bounded outer supervision for pre-guard setup failures, hangs and cross-rank failures.',
    'Establish performance, routing/transfer cost, whole-step integration, optimizer/model correctness, quality, adoption and submission independently; none is supported here.',
]
decision=dict(decision='PASS',scope='Corrected diagnostic/component source suitability for a separately bound native feasibility/correctness experiment only.',decision_basis='Independent complete supplied source review and executed bounded CPU evidence; historical status labels were not used as proof.',recorded_utc=now,candidate_commit_as_supplied=inputs['candidate_commit'],baseline_commit_as_supplied=inputs['baseline_commit'],candidate_source_sha256=inputs['candidate_source_sha256'],baseline_checker_sha256=inputs['baseline_checker_sha256'],source_modified=False,atol=0,rtol=0,acceptance='identical shape, dtype and bytes; signed zero included; null metric never means zero error',evidence=dict(normal=normal,optimized=optimized,boundaries=boundaries,unexpected_test_failures=0,expected_faults_preserved=True,provenance='raw/provenance.json',historical_arithmetic_audit='raw/attempt-001/historical_audit.json'),known_limitations=limits,remaining_gates=gates,native_legality='UNPROVEN',native_numerical_correctness='UNPROVEN',performance='UNMEASURED',production_integration='UNPROVEN',quality='UNPROVEN',adoption=False,submission_authorized=False,launch_authorized=False,hardware_authorized=False)
write(W/'decision.json',decision)
report='''**PASS — limited diagnostic/component source suitability only.**

Independent review of the supplied packet supports candidate `18b9365aabb63fa85e1720fdefc11177ca8e7853` as corrected diagnostic/component source for a separately bound native feasibility/correctness experiment. The decision remained REFUSE until the independent CPU runs completed. This PASS does not authorize hardware, launch, production integration, adoption or submission. Native legality and numerical correctness, performance and quality remain unproved. No candidate source was changed.

I read the complete checker diff and all supplied kernel/reference source, reconstructed both checker versions from the full-context hunk, and authenticated all seven manifest entries plus the four reconstructed source files. This authenticates the supplied bytes against the packet manifest; it does not independently authenticate a Git commit or external provenance. The support-file diffs are additions from `/dev/null`, so the claimed equality to the baseline support files cannot be independently reconstructed. The complete baseline checker was available and authenticated.

| Reconstructed file | SHA256 |
| --- | --- |
| candidate check_native.py | 89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850 |
| candidate consumer_nki.py | 57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843 |
| candidate reference.py | 14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f |
| baseline check_native.py | 17e2d2ad417407c12327e1446aed857f273d445946839357102e93dea46d4435 |

The entire successor change is diagnostic reporting: importing `math`, encoding nonfinite maximum errors as JSON null with a descriptive string, and protecting final reporting with a `finally` that calls cleanup. The byte acceptance expression at [checker line 81](/work/source/check_native.py:81) is unchanged: matching shape, dtype and bytes are all required. Finite metrics are still computed in FP64. Shape mismatch remains failure and omits a misleading broadcast error metric; equal-valued dtype mismatch remains failure. Signed-zero byte mismatch remains failure even though its numerical maximum error is zero. Null is an unrepresentable/nonfinite metric, never evidence of zero error or acceptance.

The [kernel](/work/source/consumer_nki.py:16) fixes V256/C128/T256 and accepts BF16 or FP32 upstream storage with FP32 accumulators/output. Counts, starts, positions and group maxima encode stable per-row occurrence lists. The two 128-row groups have separate maximum-count loop bounds. Each active iteration gathers one original token and adds it once; inactive lanes select their prior accumulator bits, with initialized sentinel addresses for both skipped indirect reads. Pointer/count updates continue on exhausted lanes, but the active selection must prevent those addresses from being used. Every output row, including unused positive-zero rows, is stored once. This supports the intended ordered recurrence at source level. It does not establish the native meaning or legality of `.ap`, register operations, dynamic ranges, mask broadcasting, DMA skip modes, allocation or device rounding.

The [reference](/work/source/reference.py:34) builds synthetic metadata only. Its validation checks dimensions, nonnegative integer counts, prefix offsets, exact stable routes, complete token permutation, padding sentinel and per-group maxima. The synthetic producer rejects invalid IDs and token overflow. It is not a general untrusted-metadata parser: starts/positions/maxima are primarily checked by equality rather than complete strict type checks. The standalone consumer/kernel relies on supplied valid inputs. `dense` checks finite FP32 inputs; arbitrary overflowing recurrences are outside this evidence and its FP32 pack conversion can raise. Kernel shape/dtype checks use Python asserts; the optimized CPU run does not prove their behavior under a native compiler.

I wrote [review_test.py](/work/scripts/review_test.py) independently. Its numerical oracle decodes finite binary32 values into signed integer multiples of 2^-149, adds exactly, and rounds to binary32 with ties to even after every original occurrence. It does not call the supplied FP32 conversion, dense implementation or consumer to construct expected results. Routing expectations use an independent stable sort by row/token. The supplied functions are compared against complete expected bytes. A separate scalar cross-check uses real CPU Torch addition, including cancellation, subnormals, signed zero, rounding boundaries and finite-input overflow to infinity.

| Executed evidence | Result |
| --- | --- |
| Integer FP32 oracle against real CPU additions | 4,217 exact pairs |
| Full fixed-shape recurrences | 32 exact cases: 20 supplied fixture/dtype combinations and 12 independent cases |
| Ordered-accumulation negative controls | Four grouped controls and two reversed-order controls differed as required |
| Metadata rejection controls | 15 rejected malformed inputs |
| Exact checker comparison functions | 17 real CPU tensor cases |
| Extracted case/error/final reporting | 24 cases, including expected I/O/serialization failures and rank filenames 0–3 |
| Python `-O` run | Same arithmetic, comparison, metadata and reporting suite passed |
| Independent BF16 conversion test | 2,059 exact representations versus CPU Torch and supplied converter |
| Whole checker `main`, setup dependencies stubbed | 14 failure-boundary scenarios matched expected outcomes |

The independent recurrence cases add unequal exhaustion in both groups, rows 0/127/128/129/255, interleaved duplicates, unused rows, empty input, signed zeros and four seeded broad-exponent FP32 cases. Every row's visit sequence was checked against original occurrence order. Active/inactive gather totals, group rounds and all 256 output stores were checked as CPU-model facts, not measured hardware traffic. Full expected, dense-actual and consumer-actual arrays are preserved at atol=0/rtol=0; every tested recurrence had zero numerical error and identical bytes.

For supplied cross-tile cancellation, sequential row 3 is +0; the forbidden grouped order gives 1 for the large case and 2^-24 for the unit case, in both input dtypes. The independent boundary-row reversal changes +0 (`00000000`) to 1 (`3f800000`). The raw wrong-result arrays are retained. Thus the suite would reject meaningful arithmetic regrouping instead of accepting numerical proximity.

The original historical receipts were treated as claims to inspect. All 20 case/dtype keys, actual output hashes, exactness flags, loop bounds, gather/skip counts, row-3 values and grouped-control values in each receipt agree with this fresh execution. Historical timestamps, actual command execution, memory peaks and production-file-change claims are not independently authenticated; their original test implementation was absent. The old PASS labels were not approval of this successor and were not used to determine this decision.

The exact `tensor_bytes`/`compare` functions were extracted by AST from authenticated source and executed with tiny real CPU tensors. Original case status/print/failure statements, exception handler, finalizer and return statements were likewise executed with completed CPU check records. [extraction.json](/work/raw/attempt-001/extraction.json) identifies the source lines and substitutions. No native API or kernel was emulated. A test cleanup sink counted attempts; it is not a real process group.

| Supplied scalar actual / expected bits | exact | finite | max_abs | Nonfinite description | Case / final outcome |
| --- | --- | --- | --- | --- | --- |
| 00000000 / 00000000 | true | true | 0 | absent | EXACT / synthetic harness success |
| 3f800000 / 00000000 | false | true | 1 | absent | FAIL_EXACTNESS / REFUSE_OR_UNSUPPORTED |
| 80000000 / 00000000 | false | true | 0 | absent | FAIL_EXACTNESS / REFUSE_OR_UNSUPPORTED |
| 7f800000 / 00000000 | false | false | null | inf | FAIL_EXACTNESS / REFUSE_OR_UNSUPPORTED |
| ff800000 / 00000000 | false | false | null | inf | FAIL_EXACTNESS / REFUSE_OR_UNSUPPORTED |
| 7fc00000 / 00000000 | false | false | null | nan | FAIL_EXACTNESS / REFUSE_OR_UNSUPPORTED |

Every scalar record and normal failure receipt was encoded with `allow_nan=False` and parsed with a decoder that rejects nonstandard numeric constants. Failed comparisons produced readable case JSON, final REFUSE receipts, error tracebacks and return code 1. A missing-gradient record also failed. Additional checks cover one-ULP/subnormal mismatches, both signed-zero directions, exact negative zero, empty tensors, noncontiguous tensors, BF16 equality, dtype and shape mismatches, and opposite maximum finite FP32 values whose FP64 error metric remains finite. Expected tensors in the diagnostic contract are finite; this is not a general comparator certification for arbitrary nonfinite expected data.

[Finalization](/work/source/check_native.py:187) encodes before exclusive file creation. Injected nonfinite/unserializable receipt fields raised before creating a rank file. Existing rank files remained byte-unchanged. Absent directories produced stdout receipts without inventing files. Directory query, file open, file write and stdout failures all attempted cleanup exactly once; cleanup's own injected exception propagated. Normal final file content matched stdout JSON. The native-looking success strings in a few synthetic harness artifacts describe tested source control flow only and are not native receipts.

The [whole-main boundary test](/work/scripts/boundary_test.py) used the unmodified checker with explicit import/setup stubs. It exercised missing arguments, support-hash rejection, environment binding, failed native import, invalid rank, failed initialization, failed device construction, missing synchronize API, existing directory/file, barrier failure, unsupported kernel construction, absent rank-1 output directory and cleanup failure. No stub could successfully construct or execute a kernel. Guarded failures stayed failures; missing/unsupported APIs were not converted into success.

**The cleanup guarantee is limited.** Initialization at line 70, device construction at line 71 and synchronize validation at lines 72–74 occur before the main protected body at line 112. Full-main stub executions confirmed that successful initialization followed by device or synchronize failure causes zero destroy calls and no rank receipt. Initialization failure also has no cleanup handler. Earlier argument/hash/environment/import/rank failures have no final rank receipt. These are unchanged setup limitations, not a claim that native teardown was exercised.

Within final reporting, cleanup is attempted but can itself raise or hang. Timing/description assignments at lines 184–185 are outside the inner reporting protection. Exclusive creation prevents overwrites, and preencoding prevents serialization-caused partial files; writes are not atomic and real I/O failure may leave a partial file. File/stdout errors can prevent the final receipt and skip the later original-error print. The injected write test used a failing stream, not a real partial-disk-write experiment. The diagnostic deliberately has no final barrier; no cross-rank aggregation or timeout enforcement is in this component. Incomplete receipts and cleanup failures remain refusal conditions for any future run.

The intended native checker covers direct and repeated calls and eager/compiled enclosing custom-autograd paths against the CPU dense oracle for nonempty fixtures. Logical-empty input explicitly skips autograd. This review did not execute any of those native paths, NKI imports, installed API inventory, actual distributed calls, native compiler or hardware launcher. No routing producer, production model, optimizer, data/evaluator or training path was available or reviewed.

All execution was sequential with numerical thread settings fixed to one. Python was 3.14.4; CPU Torch was 2.13.0+cpu. Dispatch-based tracking used weak tensor references and deduplicated underlying storage identities, sampling operation inputs/outputs. The maximum observed live tensor storage was 50,604 bytes; the separate BF16 run held 20,590 bytes. A stricter 8 MiB guard was enforced in the main suite, below the required 64 MiB. Python lists, library RSS and opaque internal library workspace are not claimed as tensor-storage measurements; all numerical arrays and operations were explicitly bounded. Standard Torch emitted a cpuinfo parse warning on stderr; it is preserved and did not affect results. There were no unexpected failed test attempts. Every intentional failure, input, captured output and exception was preserved. No networks, providers, concurrent reviewers, background jobs, model/training runs or external data were used.

Before any native result could be accepted, remaining gates are:

1. Separately bind and authorize a bounded four-rank LNC2 feasibility/correctness experiment, including required API inventory and a supervised timeout/cleanup policy that covers the setup gaps. This report supplies no hardware authorization.
2. Establish native compilation legality and actual indirect-access, mask, dynamic-loop, storage, BF16 conversion, FP32 rounding, signed-zero and subnormal behavior.
3. Obtain all required per-rank receipts for direct/repeated calls and applicable eager/compiled enclosing autograd, dense oracle and present finite gradients, with strict shape/dtype/bytes at atol=0/rtol=0. Missing APIs, mismatches, nonfinite/missing gradients, timeout or incomplete receipts must refuse.
4. Independently establish routing cost, component and whole-step performance, production integration, quality, adoption and submission suitability. None follows from this CPU/source PASS.

[decision.json](/work/decision.json) is the machine-readable disposition. [commands.sh](/work/commands.sh) records the sequential reproduction commands; runner receipts retain exact executable paths, limits and exit status. [raw](/work/raw) contains all normal, optimized and boundary evidence; [packet_snapshot](/work/packet_snapshot) preserves the original packet. [ARTIFACTS.md](/work/ARTIFACTS.md) documents binary formats. [artifact-manifest.json](/work/artifact-manifest.json) hashes every other regular artifact in `/work`; its own file is excluded to avoid self-reference.
'''
(W/'REPORT.md').write_text(report)
(W/'ARTIFACTS.md').write_text('''All artifact paths are relative to `/work` unless absolute.

- `packet_snapshot/`: byte copies of the complete supplied packet. `source/`: authenticated full candidate files and baseline checker reconstructed from diffs. Candidate copies are unchanged.
- `scripts/`: independent test, preparation, synchronous runner and finalization scripts. `commands.sh`: sequential commands. `raw/*.script.py`: exact snapshots for each test run. `raw/*.execution.json`: commands, environment overrides and observed exit status; attempt 001 is explicitly retrospective.
- `raw/provenance.json`: packet byte counts/digests and reconstructed source digests.
- `raw/attempt-001/` and `raw/attempt-002-optimized/`: identical suite, normal and optimized Python. `summary.json` reports completed tests, versions and live tensor storage. All tolerance records use atol=0, rtol=0 and exact bytes.
- `recurrence/*/input.json`: IDs, complete metadata, source dtype, logical length and physical dimensions. `input.f32le`: logical-length x 128 row-major binary32 values after input dtype conversion. Empty input has zero bytes; physical upstream storage in the actual checker would still be 256 x 128 zeros. `expected.f32le`, `consumer_actual.f32le`, `dense_actual.f32le`: complete 256 x 128 row-major little-endian binary32 matrices. `negative_*.f32le`: complete wrong-order results. Each `result.json` records tolerances, errors, hashes and control outcomes; `trace.json` preserves all visits.
- `oracle_selftest/inputs.u32le`: 4,217 interleaved pairs of little-endian uint32 IEEE754 bit patterns. `expected.u32le` and `actual.u32le`: one result pattern per pair.
- `comparison/*.json`: actual/expected bit patterns or raw byte hex, independently expected records, actual checker records and strict JSON strings. Numeric bit spellings in scalar inputs use conventional eight-digit hexadecimal, not a byte-stream ordering. The actual tensor byte hashes use little-endian storage.
- `metadata_negative_controls.json`: all malformed inputs and actual rejection errors. `historical_audit.json`: arithmetic fields checked in historical receipts and claims not independently verified.
- `reporting/*/`: completed input checks, output text, error text/tracebacks and actual rank files where creation succeeded. Some rank files deliberately contain the preexisting marker used for exclusivity tests and are not JSON receipts. Invalid/unserializable field injections are specified exactly in the preserved script. `mode` identifies the injected fault. Test cleanup is a counter only.
- `raw/attempt-003-boundaries/`: 2,059 BF16 conversion input/expected/CPU/reference uint32 patterns and full-main setup test folders. Conversion outputs are expanded to binary32 for exact representation comparison. Setup imports and distributed operations are explicit stubs; they are not native executions. `result.json`, stdout/stderr and exceptions retain each input scenario and outcome.
- `raw/*.stdout` and `raw/*.stderr`: complete process output, including the CPU-library cpuinfo warning. Expected injected exceptions are successful negative tests, not concealed failed attempts. No unexpected attempt failed.

Artifact hashes cover all files except the hash manifest itself. Generated receipt timestamps/durations may differ on reproduction; arithmetic bytes and logical results should match. Reproduction commands expect a fresh `/work` because tests and receipts deliberately use exclusive artifact creation; preserve existing evidence instead of overwriting it. Native imports are never required by these tests.
''')
# Check strict JSON for machine-readable result files (deliberate preexisting-marker
# rank files are excluded by parsing the named result artifacts only).
def bad(value): raise ValueError('nonstandard JSON constant '+value)
for path in [W/'decision.json']+list((W/'raw').rglob('summary.json'))+list((W/'raw').rglob('result.json')):
    json.loads(path.read_text(),parse_constant=bad)
# Manifest is written last, including this script, reports, commands and all raw data.
entries=[]
for path in sorted(W.rglob('*')):
    if path.is_file() and path.name!='artifact-manifest.json':
        data=path.read_bytes()
        entries.append(dict(path=str(path.relative_to(W)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
write(W/'artifact-manifest.json',dict(algorithm='SHA256',excluded=['artifact-manifest.json'],entries=entries))
for item in entries:
    data=(W/item['path']).read_bytes()
    require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'manifest verification failed')
print(json.dumps(dict(decision='PASS',scope=decision['scope'],artifact_files_verified=len(entries),bytes=sum(x['bytes'] for x in entries),recorded_utc=now,report='/work/REPORT.md',manifest='/work/artifact-manifest.json')))
