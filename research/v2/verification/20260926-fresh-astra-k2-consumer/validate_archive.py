#!/usr/bin/env python3
"""Authenticate archived independent evidence; no tensors, model call, or rerun."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'session/reviewer-output'


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


archive = json.loads((ROOT / 'ARCHIVE_MANIFEST.json').read_text())
for item in archive['files']:
    path = ROOT / item['path']
    require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'archive drift: ' + item['path'])
manifest = json.loads((WORK / 'artifact_manifest.json').read_text())
for item in manifest['work_artifacts']:
    path = WORK / item['path']
    require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'reviewer artifact drift: ' + item['path'])
for item in manifest['packet_inputs']:
    path = ROOT / 'packet' / Path(item['path']).relative_to('/packet')
    require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'reviewer input drift')
events = [json.loads(line) for line in (ROOT / 'session/events.jsonl').read_text().splitlines()]
threads = [row['thread_id'] for row in events if row['type'] == 'thread.started']
turns = [row for row in events if row['type'] == 'turn.completed']
terminal = json.loads((ROOT / 'session/terminal.json').read_text())
require(len(threads) == len(turns) == 1 and terminal['exit_code'] == 0, 'session incomplete or reused')
require(terminal['prior_history'] is False and terminal['repository_mounted'] is False, 'operation isolation metadata')
require(terminal['model_requested'] == 'gpt-6-astra', 'requested model changed')
decision = json.loads((WORK / 'decision.json').read_text())
cpu = json.loads((WORK / 'cpu_results.json').read_text())
failure = json.loads((WORK / 'checker_failure_results.json').read_text())
require(decision['decision'] == 'REFUSE' and decision['native_executed'] is False, 'review verdict altered')
require(cpu['pass'] is True and len(cpu['cases']) == 42 and cpu['native_executed'] is False, 'independent CPU scope')
require(failure['defect_reproduced'] is True and failure['native_executed'] is False and failure['native_main_executed'] is False, 'failure reproduction scope')
bad = [case for case in failure['cases'] if not case['finite']]
require(len(bad) == 3 and all(not case['exact'] and not case['receipt_valid_json'] for case in bad), 'three nonfinite refusals retained')
for case in bad:
    file = WORK / 'checker_failure_artifacts' / (case['name'] + '.rank0.json')
    require(file.is_file(), 'missing partial receipt')
    try:
        json.loads(file.read_text())
    except json.JSONDecodeError:
        pass
    else:
        raise RuntimeError('failure receipt no longer reproduces malformed output')
result = dict(status='PASS_ARCHIVE_AUTHENTICATION_REVIEW_VERDICT_REFUSE', archive_files=len(archive['files']),
              reviewer_manifest_files=len(manifest['work_artifacts']), session_id=threads[0],
              model_requested=terminal['model_requested'], provider_resolved_model_separately_reported=False,
              started_utc=terminal['started_utc'], completed_utc=terminal['completed_utc'],
              elapsed_seconds=terminal['elapsed_seconds'], usage=turns[0]['usage'],
              input_manifest_sha256=terminal['manifest_sha256'], decision_sha256=digest(WORK / 'decision.json'),
              report_sha256=digest(WORK / 'report.md'), own_reference_test_sha256=digest(WORK / 'independent_exactness.py'),
              own_reference_results_sha256=digest(WORK / 'cpu_results.json'),
              failure_probe_sha256=digest(WORK / 'checker_failure_probe.py'),
              failure_results_sha256=digest(WORK / 'checker_failure_results.json'),
              independent_cpu_cases=42, native_executed=False, cpu_slot_released=True)
(ROOT / 'OPERATION_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
