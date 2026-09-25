"""Check copied evidence identity without rerunning or interpreting tensor tests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
checks = 0


def check(condition, message):
    global checks
    if not condition:
        raise SystemExit(message)
    checks += 1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


packet = ROOT / 'packet'
check(digest(packet / 'INPUT_MANIFEST.json') ==
      '96a204756ce2f465f6cf9dbee24dab4997703107a9c5494696faa933cb4354c5',
      'input manifest changed')
for entry in json.loads((packet / 'INPUT_MANIFEST.json').read_text())['files']:
    p = packet / entry['path']
    check(p.stat().st_size == entry['bytes'], f'input size mismatch: {p}')
    check(digest(p) == entry['sha256'], f'input hash mismatch: {p}')

out = ROOT / 'session2/reviewer-output'
for entry in json.loads((out / 'artifact_manifest.json').read_text())['files']:
    name = Path(entry['path']).name
    p = out / name
    if name == 'reference_exactness.tensors.pt':
        p = Path('/tmp/trainium-fresh-architecture-review-r2-20260926.Hx9821/output') / name
    check(p.stat().st_size == entry['bytes'], f'output size mismatch: {p}')
    check(digest(p) == entry['sha256'], f'output hash mismatch: {p}')

terminal = json.loads((ROOT / 'session2/terminal.json').read_text())
check(terminal['exit_code'] == 0, 'review operation failed')
check(terminal['prior_history'] is False, 'prior context reported')
check(terminal['repository_mounted'] is False, 'repository mounted')
check(terminal['model_requested'] == 'gpt-6-astra', 'wrong selected model')
result = json.loads((out / 'result.json').read_text())
check(result['verdict'] == 'PASS', 'unexpected reviewer disposition')
check(result['reviewer']['cross_family_review'] is False, 'misclassified family')
events = [json.loads(line) for line in (ROOT / 'session2/events.jsonl').read_text().splitlines()]
check(events[0]['thread_id'] == '01a0d9fa-c5f1-7522-b5ed-4b7905cc1000', 'wrong session')
check(any(e['type'] == 'turn.completed' for e in events), 'turn did not complete')
check(any(e.get('item', {}).get('type') == 'command_execution' and
          e['item'].get('exit_code') == 0 and
          '/work/reference_exactness.py > /work/reference_exactness.raw.log' in e['item'].get('command', '')
          for e in events), 'own final test did not execute successfully')
check(not list(ROOT.rglob('*.pt')), 'tensor artifact must not be committed')
print(json.dumps({'status': 'PASS', 'stdlib_archive_checks': checks,
                  'numerical_test_rerun_by_operator': False}, sort_keys=True))
