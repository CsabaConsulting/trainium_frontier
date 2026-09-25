"""Identity/structure audit only; does not rerun the independent tensor tests."""
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
      'e0e0575ae753731408c07feb27867ff47b38c63ac3cf46942012569d0057fe1e',
      'input manifest changed')
for entry in json.loads((packet / 'INPUT_MANIFEST.json').read_text())['files']:
    p = packet / entry['path']
    check(p.stat().st_size == entry['bytes'], f'input size mismatch: {p}')
    check(digest(p) == entry['sha256'], f'input hash mismatch: {p}')

out = ROOT / 'session/reviewer-output'
external = {Path(e['path']).name: e for e in
            json.loads((ROOT / 'BINARY_RETENTION.json').read_text())['files']}
for entry in json.loads((out / 'artifact_manifest.json').read_text())['files']:
    name = Path(entry['path']).name
    p = out / name
    if name in external:
        check(entry['sha256'] == external[name]['sha256'], 'retained binary hash pin')
        p = Path(external[name]['path'])
    check(p.stat().st_size == entry['bytes'], f'output size mismatch: {p}')
    check(digest(p) == entry['sha256'], f'output hash mismatch: {p}')

terminal = json.loads((ROOT / 'session/terminal.json').read_text())
check(terminal['exit_code'] == 0, 'review operation failed')
check(terminal['prior_history'] is False, 'prior context reported')
check(terminal['repository_mounted'] is False, 'repository mounted')
check(terminal['model_requested'] == 'gpt-6-astra', 'wrong selected model')
result = json.loads((out / 'result.json').read_text())
check(result['verdict'] == 'PASS', 'unexpected reviewer disposition')
own = json.loads((out / 'own_test_results.json').read_text())
check(own['status'] == 'PASS' and len(own['results']) == 6, 'own test group mismatch')
check(all(e['status'] == 'PASS' for e in own['results']), 'own test failed')
events = [json.loads(line) for line in (ROOT / 'session/events.jsonl').read_text().splitlines()]
check(events[0]['thread_id'] == '01a0da02-1e9c-7771-8c4e-22560f61836f', 'wrong session')
check(any(e['type'] == 'turn.completed' for e in events), 'turn did not complete')
check(any(e.get('item', {}).get('type') == 'command_execution' and
          e['item'].get('exit_code') == 0 and
          '/work/test_reference_exactness.py > /work/own_test.stdout.txt' in e['item'].get('command', '')
          for e in events), 'own test did not execute successfully')
check(not list(ROOT.rglob('*.pt')), 'binary tensors must not be committed')
print(json.dumps({'status': 'PASS', 'stdlib_archive_checks': checks,
                  'numerical_test_rerun_by_operator': False}, sort_keys=True))
