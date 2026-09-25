"""Archive only the explicit non-secret review inputs and generated evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch-only', action='store_true')
    args = parser.parse_args()
    prepared = json.loads((HERE / 'PREPARED.json').read_text())
    packet, run = Path(prepared['packet']), Path(prepared['run'])
    manifest_name = 'LAUNCH_ARCHIVE_MANIFEST.json' if args.launch_only else 'ARCHIVE_MANIFEST.json'
    if (HERE / manifest_name).exists():
        raise RuntimeError('Archive already frozen')
    if not args.launch_only and not (run / 'terminal.json').is_file():
        raise RuntimeError('Do not archive an incomplete session as terminal')
    inputs = json.loads((packet / 'INPUT_MANIFEST.json').read_text())
    if hashlib.sha256((packet / 'INPUT_MANIFEST.json').read_bytes()).hexdigest() != prepared['input_manifest_sha256']:
        raise RuntimeError('Input manifest changed')
    for item in inputs['included']:
        data = (packet / item['path']).read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('Input changed')
    copies = [(p, HERE / 'packet' / p.relative_to(packet)) for p in packet.rglob('*') if p.is_file()]
    names = ['operate.py', 'preflight.json', 'prompt-input.json', 'prompt-input.stderr', 'launch.json']
    if not args.launch_only:
        names += ['events.jsonl', 'stderr.log', 'terminal.json']
    copies += [(run / n, HERE / 'session' / n) for n in names]
    if not args.launch_only:
        copies += [(p, HERE / 'session/reviewer-output' / p.relative_to(run / 'output')) for p in (run / 'output').rglob('*') if p.is_file()]
    entries = []
    for src, dst in copies:
        if src.is_symlink() or not src.is_file():
            raise RuntimeError('Non-regular artifact: ' + str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        data = dst.read_bytes()
        entries.append(dict(path=str(dst.relative_to(HERE)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    result = dict(files=sorted(entries, key=lambda r: r['path']), total_bytes=sum(r['bytes'] for r in entries), count=len(entries), terminal=not args.launch_only)
    path = HERE / manifest_name
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'files':result['count'], 'bytes':result['total_bytes'], 'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'terminal':result['terminal']}))

if __name__ == '__main__':
    main()
