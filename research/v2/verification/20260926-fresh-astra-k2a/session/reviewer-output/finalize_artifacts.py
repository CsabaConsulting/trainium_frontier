"""Write machine-readable review decision and verify/package retained artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/work')
def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

tests = json.loads((ROOT / 'reference_results.json').read_text())
hashes = json.loads((ROOT / 'input_hashes.json').read_text())
receipts = json.loads((ROOT / 'receipt_audit.json').read_text())
for entry in hashes['inputs']:
    data = Path(entry['path']).read_bytes()
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
for name, entry in hashes['reconstructed_sources'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry['sha256']
assert tests['harness_error'] is None
assert tests['summary'] == {'checks': 714, 'failures': 6, 'actual_cpu_path_failures': 0}
counts = {}
for check in tests['checks']:
    group = counts.setdefault(check['name'].split('/')[0], dict(passed=0, failed=0))
    group['passed' if check['pass_exact'] else 'failed'] += 1
commands = {
    'scope': 'Evidence-producing executions and initial instruction-file discovery; shell reads are summarized in REVIEW.md and retained tool transcript.',
    'executions': [
        dict(command='ls -la /work /packet /AGENTS.md /work/AGENTS.md', exit_code=2,
             interpretation='Read-only discovery: /AGENTS.md and /work/AGENTS.md do not exist; not a test failure.'),
        dict(command='/opt/cpu/bin/python /work/audit_packet.py > /work/packet_audit.stdout 2> /work/packet_audit.stderr',
             exit_code=0, stdout='packet_audit.stdout', stderr='packet_audit.stderr'),
        dict(command='env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /opt/cpu/bin/python /work/reference_exactness_test.py > /work/reference_test.stdout 2> /work/reference_test.stderr',
             exit_code=1, stdout='reference_test.stdout', stderr='reference_test.stderr',
             interpretation='Completed with six retained exactness failures in CPU semantic emulation; no harness exception.',
             tool_session_id=83698, repeats=0),
        dict(command='/opt/cpu/bin/python /work/finalize_artifacts.py', exit_code=0,
             interpretation='Initial packaging execution completed; regenerated after correcting report line references. This command does not rerun the test.'),
    ],
}
write('commands.json', commands)
result = dict(
    schema='independent-k2-source-cpu-review-v1', verdict='REFUSE',
    scope='Supplied packet only; bounded CPU exactness and static source review; authorized SAME-family substitute, no cross-family claim.',
    candidate_commit_metadata='25b731f7638672aa8d03a7cd15b6a02b8f5776ea',
    base_commit_metadata='a4aa4351789115a2d0aa947d9ca00ff689f9bd0b',
    ancestry_independently_verified=False,
    input_hashes_verified=True, input_files=11,
    source_hashes=hashes['reconstructed_sources'],
    test_command=commands['executions'][2]['command'], test_exit_code=1,
    test_summary=tests['summary'], check_groups=counts, test_metadata=tests['metadata'],
    findings=[dict(
        id='R1', severity='blocking', title='Per-tile duplicate coalescing changes FP32 accumulation order and fails dense CPU exactness',
        source_locations=['candidate.py:1945', 'candidate.py:1970', 'candidate.py:1977', 'candidate.py:2008'],
        evidence_kind='Independent CPU semantic emulation plus actual dense/candidate CPU backward; not native execution',
        exact_bf16_counterexample=dict(vocabulary=256, channels=8, ids_shape=[2,128], row=3,
                                      flattened_positions=[0,128,129], upstream=[16777216,1,-16777216],
                                      dense_cpu=0, actual_candidate_cpu=0, emulated_tile_grouping=1),
        correction_required='Preserve the agreed dense accumulation semantics and pass retained counterexamples at zero tolerance; an approximate contract requires explicit user approval.',
    )],
    nonblocking_observations=[dict(id='O1', title='Training scatter flag is absent from saved checkpoint toggles and metadata',
                                  evaluation_effect='None found: evaluation bypasses scatter regardless of live flag; full load_for_eval not executed.')],
    positive_evidence=[
        'Complete reconstructed base/candidate sources match lane hashes; every listed input matches packet manifest.',
        'All 696 non-emulation checks pass, including actual candidate CPU backward, BF16 cast behavior and FP32 repeated accumulation.',
        'Three-step tiny extracted lookup/AdamW comparisons match base for candidate default/OFF/ON at every recorded output/gradient/parameter/moment.',
        'Supplied G2 recorded losses and all 22 rank-0 final digest strings match across base/default/OFF/ON; reduced shimmed CPU scope only.',
        'No assigned architecture/schedule/data/optimizer/evaluation/timer changes in supplied source diff.',
    ],
    raw_failures=tests['failures'], native_executed=False, full_model_executed=False,
    production_distributed_executed=False, performance_verdict='UNMEASURED', quality_verdict='UNESTABLISHED',
    pending_scope=[
        'Resolution of R1 for exact CPU semantic equivalence.',
        'Native NKI/compiler/DMA legality and correctness, production shapes, compiled autograd and two-program operation.',
        'Production world4 collectives and full-model correctness/evaluation.',
        'Cold startup and >=5% faster whole step versus base, under official accounting.',
        'Applicable quality gates; separate Operator assignment required for native work.',
    ],
    authority=dict(merge_permission=False, hardware_launch_permission=False),
    accounting=dict(budget_seconds=1800, startup_allowance_cap_seconds=900, startup_excess_charged=True,
                    fullchip=True, lnc=2, world_size=4, seed=42, native_assignment='sole Operator'),
    artifacts=dict(report='REVIEW.md', test_source='reference_exactness_test.py', raw_stdout='reference_test.stdout',
                   raw_stderr='reference_test.stderr', test_results='reference_results.json', tensors='reference_tensors.pt',
                   input_hashes='input_hashes.json', receipts='receipt_audit.json', commands='commands.json',
                   manifest='ARTIFACT_MANIFEST.json'),
)
write('result.json', result)
files = []
for path in sorted(ROOT.iterdir()):
    if path.is_file() and path.name not in ('ARTIFACT_MANIFEST.json', 'finalize.stdout', 'finalize.stderr'):
        data = path.read_bytes()
        files.append(dict(path=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
write('ARTIFACT_MANIFEST.json', dict(schema='artifact-manifest-v1', files=files,
      exclusion='Manifest excludes itself and transient finalization stdout/stderr to avoid self-reference and hashing live output.'))
print(json.dumps(dict(verdict=result['verdict'], test_summary=tests['summary'],
                      packet_unchanged=True, artifact_count=len(files)), sort_keys=True))
