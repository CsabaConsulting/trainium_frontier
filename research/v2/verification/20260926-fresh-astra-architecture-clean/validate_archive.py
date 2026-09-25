"""Authenticate review artifacts and fresh-session provenance, not model correctness."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    archive = json.loads((HERE / 'ARCHIVE_MANIFEST.json').read_text())
    for item in archive['files']:
        path = HERE / item['path']
        require(path.is_file() and not path.is_symlink(), 'non-regular archive file')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'archive hash mismatch: ' + item['path'])
    require(len(archive['files']) == archive['count'], 'archive count')
    require(sum(row['bytes'] for row in archive['files']) == archive['total_bytes'], 'archive bytes')
    prepared = json.loads((HERE / 'PREPARED.json').read_text())
    packet = HERE / 'packet'
    require(sha(packet / 'INPUT_MANIFEST.json') == prepared['input_manifest_sha256'], 'input manifest pin')
    inputs = json.loads((packet / 'INPUT_MANIFEST.json').read_text())
    for row in inputs['included']:
        require(sha(packet / row['path']) == row['sha256'] and (packet / row['path']).stat().st_size == row['bytes'], 'input hash mismatch')
    require({str(p.relative_to(packet)) for p in packet.rglob('*') if p.is_file()} == {'INPUT_MANIFEST.json'} | {x['path'] for x in inputs['included']}, 'extra input file')
    require(inputs['prior_history'] is False and inputs['shared_branch'] is False, 'fresh input provenance')
    # Full-context diff reconstruction authenticates the three intended source files.
    lines = (packet / 'vehicle-to-candidate.diff').read_text().splitlines(keepends=True)
    old, new = [], []
    for line in lines:
        if line.startswith(('--- ', '+++ ', '@@')):
            continue
        if line.startswith((' ', '-')):
            old.append(line[1:])
        if line.startswith((' ', '+')):
            new.append(line[1:])
    reference = ''.join(line[1:] for line in (packet / 'scored-reference.diff').read_text().splitlines(keepends=True) if line.startswith('+') and not line.startswith('+++ '))
    for key, data in [('vehicle', ''.join(old)), ('candidate', ''.join(new)), ('scored_reference', reference)]:
        require(hashlib.sha256(data.encode()).hexdigest() == inputs['source_sha256'][key], 'reconstructed source hash: ' + key)
    launch = json.loads((HERE / 'session/launch.json').read_text())
    terminal = json.loads((HERE / 'session/terminal.json').read_text())
    preflight = json.loads((HERE / 'session/preflight.json').read_text())
    require(preflight['exit'] == 0, 'isolation preflight')
    require(launch['manifest_sha256'] == prepared['input_manifest_sha256'] == terminal['manifest_sha256'], 'session input binding')
    require(launch['model_requested'] == 'gpt-6-astra' and launch['prior_history'] is False and launch['repository_mounted'] is False, 'session model/isolation')
    command = launch['command']
    require(all(flag in command for flag in ('--ephemeral', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check', '--unshare-all')), 'fresh isolated CLI flags')
    require('resume' not in command and 'fork' not in command and command[command.index('-m') + 1] == 'gpt-6-astra', 'new session not resumed/forked')
    require(terminal['exit_code'] == 0 and not terminal.get('wall_cap_triggered'), 'completed within finite session')
    events = [json.loads(line) for line in (HERE / 'session/events.jsonl').read_text().splitlines() if line.strip()]
    threads = [r['thread_id'] for r in events if r.get('type') == 'thread.started']
    turns = [r for r in events if r.get('type') == 'turn.completed']
    require(len(threads) == len(turns) == 1 and not any(r.get('type') == 'turn.failed' for r in events), 'one new completed turn')
    require(len([r for r in events if r.get('type') == 'turn.started']) == 1, 'one user turn')
    own_manifest = json.loads((HERE / 'session/reviewer-output/artifact_manifest.json').read_text())
    print(json.dumps({'archive_integrity': 'PASS', 'files': archive['count'], 'bytes': archive['total_bytes'],
                      'thread_id': threads[0], 'model_requested': launch['model_requested'],
                      'usage': turns[0].get('usage'), 'input_manifest_sha256': prepared['input_manifest_sha256'],
                      'reviewer_artifact_manifest_type': type(own_manifest).__name__,
                      'model_correctness_is_not_decided_by_this_validator': True}, indent=2))

if __name__ == '__main__':
    main()
