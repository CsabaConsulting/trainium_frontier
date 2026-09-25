"""Freeze input-only review for the distinct grid2 consumer; reuse isolation unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REF = '9ce2e6a037212aaa76bd85b9b4ab7d311a481757'
BASE = 'd0f0d7fbdbe07ddee39680ee1a2b79fdfac388ca'
SOURCE = 'research/v2/lanes/k2-symmetric-consumer/consumer_nki.py'
OLD_SOURCE = 'research/v2/lanes/k2-resident-consumer/consumer_nki.py'
INPUT = 'research/v2/proposals/20260926-k2-symmetric-consumer/review-input/'
RAW_REF = '7c40e4f5'
RAW_PATH = 'research/v2/operator/20260926-trn22-k2-namespace-native/completed/results-bundle.json'

def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    if (HERE / 'PREPARED.json').exists():
        raise RuntimeError('No duplicate preparation')
    before, after = blob(BASE, OLD_SOURCE), blob(REF, SOURCE)
    if sha(before) != '4939265bf73b3d5f42276e72e48f022df338226a40a9be5b79ed63bde6a4f5b8':
        raise RuntimeError('Baseline source drift')
    if sha(after) != '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae':
        raise RuntimeError('Candidate source drift')
    diff = blob(REF, INPUT + 'kernel-full-context.diff')
    old, new = [], []
    for line in diff.decode().splitlines(True):
        if line.startswith(('diff ', 'index ', '--- ', '+++ ', '@@')):
            continue
        if line.startswith((' ', '-')):
            old.append(line[1:])
        if line.startswith((' ', '+')):
            new.append(line[1:])
    if ''.join(old).encode() != before or ''.join(new).encode() != after:
        raise RuntimeError('Full-context diff reconstruction failed')
    packet = Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-symmetric-packet-20260926.'))
    run = Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-symmetric-review-20260926.'))
    (run / 'output').mkdir()
    (packet / 'kernel-full-context.diff').write_bytes(diff)
    spec = blob(REF, INPUT + 'lane-spec.md').decode()
    spec += '\n## Fresh independent-session constraints\n\nYou receive only the full-context source diff, this neutral contract and original raw gate receipts. No repository, prior session, implementation notes or prior reviewer conclusions are available. Do not use network, AWS, hosts, credentials, external data or weights. Do not start native or model training. Default REFUSE until the evidence needed for this limited source/CPU gate is complete. Reconstruct the exact candidate from the diff and independently write and RUN at least one tiny CPU reference test of an exactness claim: ordered FP32 accumulation, exact feature ownership or symmetric per-program control schedules. Use stdlib and tiny synthetic inputs; no SDK/model import, large allocations or installation. CPU emulation cannot prove NKI lowering, native roundoff, speed or full training. Do not call a native compiler result proved by source. Return REPORT.md plus your independently written test and raw outputs in /work; explicitly identify limits and any precise blocking defect. No code edits to the candidate.\n'
    (packet / 'lane-spec.md').write_text(spec)
    for name in ('rank0.json', 'terminal.json', 'compiler.log'):
        raw = blob(REF, INPUT + 'original-native/' + name)
        dest = packet / 'original-native' / name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(raw)
    included = [{'path': str(p.relative_to(packet)), 'bytes': p.stat().st_size,
                 'sha256': sha(p.read_bytes())} for p in sorted(packet.rglob('*')) if p.is_file()]
    manifest = {'candidate_commit': REF, 'baseline_commit': BASE, 'source_path': SOURCE,
                'source_sha256': {'baseline': sha(before), 'candidate': sha(after)},
                'original_native_commit': RAW_REF, 'included': included,
                'prior_history': False, 'shared_branch': False,
                'excluded': ['repository', 'implementation notes/tests/conclusions',
                             'prior reviewer conclusions', 'unrelated receipts']}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    (packet / 'INPUT_MANIFEST.json').write_bytes(encoded)
    pin = sha(encoded)
    template = (HERE.parent / '20260926-fresh-astra-k2-namespace/session/operate.py').read_text()
    old_packet = '/tmp/trainium-fresh-k2-namespace-packet-20260926.1fj0qggt'
    old_pin = '0126e05a79e02313c742b0d94ee7ee74e58b7ea04765041854f5b4ee6d706029'
    if template.count(old_packet) != 1 or template.count(old_pin) != 1:
        raise RuntimeError('Reviewed operation template drift')
    operation = template.replace(old_packet, str(packet)).replace(old_pin, pin)
    (run / 'operate.py').write_text(operation)
    prepared = {'packet': str(packet), 'run': str(run), 'input_manifest_sha256': pin,
                'candidate_commit': REF, 'candidate_sha256': sha(after),
                'model_requested': 'gpt-6-astra', 'operation_changes': ['packet path', 'manifest pin'],
                'fresh_session_started': False, 'task_file_count': len(included),
                'task_bytes': sum(row['bytes'] for row in included)}
    (HERE / 'PREPARED.json').write_text(json.dumps(prepared, indent=2, sort_keys=True) + '\n')
    print(json.dumps(prepared))

if __name__ == '__main__':
    main()
