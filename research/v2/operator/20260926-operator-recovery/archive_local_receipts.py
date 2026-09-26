"""Copy only the stopped monitor's new raw receipts without JSON reserialization."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator/research/v2/operator/20260926-adopted-batch-handoff')
DEST = HERE / 'recovered'
DEST.mkdir(exist_ok=True)
rows = []
for host in ('trn21', 'trn22'):
    for source in sorted(OLD.glob(host + '-status-*.json')):
        tag = source.stem.rsplit('-', 1)[1]
        if not ('105200' <= tag <= '105800'):
            continue
        raw = source.read_bytes()
        parsed = json.loads(raw)
        assert parsed['returncode'] == 0 and not parsed['stderr']
        assert parsed['remote']['read_only'] is True
        target = DEST / source.name
        with target.open('xb') as stream:
            stream.write(raw)
        assert target.read_bytes() == raw == source.read_bytes()
        rows.append({'source': str(source), 'archive': str(target.relative_to(HERE)),
                     'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
assert len(rows) == 18
manifest = {'schema': 'operator-recovery-raw-copy-v1',
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'raw_json_reserialized': False, 'old_worktree_modified': False,
            'host_contact_performed': False, 'files': rows}
out = HERE / 'recovered-manifest.json'
with out.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')
print(json.dumps({'copied': len(rows), 'manifest_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
