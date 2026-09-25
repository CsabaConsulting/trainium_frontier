"""Archive explicit non-secret inputs/outputs; no source judgment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch-only',action='store_true');args=parser.parse_args()
    prepared=json.loads((HERE/'PREPARED.json').read_text());packet=Path(prepared['packet']);run=Path(prepared['run'])
    name='LAUNCH_ARCHIVE_MANIFEST.json' if args.launch_only else 'ARCHIVE_MANIFEST.json'
    if (HERE/name).exists():raise RuntimeError('Archive already frozen')
    if not args.launch_only and not (run/'terminal.json').is_file():raise RuntimeError('Not terminal')
    inputs=json.loads((packet/'INPUT_MANIFEST.json').read_text())
    if hashlib.sha256((packet/'INPUT_MANIFEST.json').read_bytes()).hexdigest()!=prepared['input_manifest_sha256']:raise RuntimeError('Manifest changed')
    for item in inputs['included']:
        raw=(packet/item['path']).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise RuntimeError('Input changed')
    copies=[(p,HERE/'packet'/p.relative_to(packet)) for p in packet.rglob('*') if p.is_file()]
    names=['operate.py','preflight.json','prompt-input.json','prompt-input.stderr','launch.json']
    if not args.launch_only:names+=['events.jsonl','stderr.log','terminal.json']
    copies.extend((run/n,HERE/'session'/n) for n in names)
    if not args.launch_only:copies.extend((p,HERE/'session/reviewer-output'/p.relative_to(run/'output')) for p in (run/'output').rglob('*') if p.is_file())
    entries=[]
    for src,dst in copies:
        if src.is_symlink() or not src.is_file():raise RuntimeError('Nonregular artifact')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);raw=dst.read_bytes()
        entries.append({'path':str(dst.relative_to(HERE)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    result={'files':sorted(entries,key=lambda r:r['path']),'count':len(entries),'total_bytes':sum(x['bytes'] for x in entries),'terminal':not args.launch_only}
    path=HERE/name;path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'files':result['count'],'bytes':result['total_bytes'],'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'terminal':result['terminal']}))
if __name__=='__main__':main()
