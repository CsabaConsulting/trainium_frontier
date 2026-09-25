"""Copy an explicit completed-review artifact allowlist; no secret/provider files."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent

def main():
    prepared = json.loads((HERE / 'PREPARED.json').read_text())
    packet, run = Path(prepared['packet']), Path(prepared['run'])
    if not (run / 'terminal.json').is_file():
        raise RuntimeError('Do not archive incomplete session as terminal')
    if (HERE / 'ARCHIVE_MANIFEST.json').exists():
        raise RuntimeError('Archive already frozen')
    inputs = json.loads((packet / 'INPUT_MANIFEST.json').read_text())
    if hashlib.sha256((packet / 'INPUT_MANIFEST.json').read_bytes()).hexdigest() != prepared['input_manifest_sha256']:
        raise RuntimeError('Input manifest changed')
    for item in inputs['included']:
        data = (packet / item['path']).read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('Input changed')
    copies = [(p, HERE / 'packet' / p.relative_to(packet)) for p in packet.rglob('*') if p.is_file()]
    for name in ('operate.py', 'preflight.json', 'prompt-input.json', 'prompt-input.stderr', 'launch.json', 'events.jsonl', 'stderr.log', 'terminal.json'):
        copies.append((run / name, HERE / 'session' / name))
    copies += [(p, HERE / 'session/reviewer-output' / p.relative_to(run / 'output')) for p in (run / 'output').rglob('*') if p.is_file()]
    entries = []
    for src, dst in copies:
        if src.is_symlink() or not src.is_file():
            raise RuntimeError('Non-regular artifact: ' + str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        data = dst.read_bytes()
        entries.append(dict(path=str(dst.relative_to(HERE)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(files=sorted(entries, key=lambda row: row['path']), total_bytes=sum(row['bytes'] for row in entries), count=len(entries))
    (HERE / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'files': manifest['count'], 'bytes': manifest['total_bytes'], 'archive_manifest_sha256': hashlib.sha256((HERE / 'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()
