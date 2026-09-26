"""New evidence, unchanged kernel: prepare an isolated source-only reassessment."""
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
DOC_REF = 'd51ed9121569c9242aca72c3dfee50233e98ed56'
DOC_ROOT = 'research/v2/analysis/20260926-k2-symmetric-api-evidence/'

def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    if (HERE / 'PREPARED.json').exists():
        raise RuntimeError('No duplicate preparation')
    before, after = blob(BASE, OLD_SOURCE), blob(REF, SOURCE)
    if sha(before) != '4939265bf73b3d5f42276e72e48f022df338226a40a9be5b79ed63bde6a4f5b8' or sha(after) != '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae':
        raise RuntimeError('Source drift')
    diff = blob(REF, INPUT + 'kernel-full-context.diff')
    old, new = [], []
    for line in diff.decode().splitlines(True):
        if line.startswith(('diff ', 'index ', '--- ', '+++ ', '@@')): continue
        if line.startswith((' ', '-')): old.append(line[1:])
        if line.startswith((' ', '+')): new.append(line[1:])
    if ''.join(old).encode() != before or ''.join(new).encode() != after:
        raise RuntimeError('Full-context reconstruction failed')
    docs = blob(DOC_REF, DOC_ROOT + 'SOURCES.json')
    if sha(docs) != '8b3ba9f60448c00681390be5e1569c7bc4e729e6a8129858efb81820d13af5e2':
        raise RuntimeError('Documentation archive drift')
    packet = Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-api-packet-20260926.'))
    run = Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-api-review-20260926.'))
    (run / 'output').mkdir()
    def put(name, data):
        p = packet / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    put('kernel-full-context.diff', diff)
    spec = blob(REF, INPUT + 'lane-spec.md').decode()
    spec += '\n## Fresh independent-session constraints\n\nYou receive only a full-context diff, this neutral lane contract, original raw gate receipts, versioned primary API documentation and exact unchanged gate support sources. No repository, prior session, implementation notes, prior tests or prior reviewer conclusions are available. No network, AWS, hosts, credentials, external data, weights, native execution or model training. Default REFUSE until evidence for this limited source/CPU gate is complete. Reconstruct the candidate and independently write and RUN a bounded stdlib CPU reference exactness test. Independently assess documented API use; do not treat documentation or source emulation as proof of installed compiler lowering, roundoff, performance or training. The support checker is evidence of intended native checks, not proof they passed. Use tiny synthetic inputs, no SDK import, installation or large allocations. Return REPORT.md plus your independently written test and raw outputs in /work, identifying limits and precise blockers. Do not edit the candidate.\n'
    put('lane-spec.md', spec.encode())
    for name in ('rank0.json', 'terminal.json', 'compiler.log'):
        put('original-native/' + name, blob(REF, INPUT + 'original-native/' + name))
    put('api/SOURCES.json', docs)
    put('api/EXCERPT_HASHES.json', blob(DOC_REF, DOC_ROOT + 'EXCERPT_HASHES.json'))
    put('api/API_ADDENDUM.md', (HERE / 'API_ADDENDUM.md').read_bytes())
    for name, pin in [('check_native.py', '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850'), ('reference.py', '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f')]:
        data = blob(REF, 'research/v2/lanes/k2-symmetric-consumer/' + name)
        if sha(data) != pin: raise RuntimeError('Gate support drift')
        put('gate-support/' + name, data)
    included = [{'path': str(p.relative_to(packet)), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in sorted(packet.rglob('*')) if p.is_file()]
    manifest = {'candidate_commit': REF, 'baseline_commit': BASE, 'source_path': SOURCE, 'source_sha256': {'baseline': sha(before), 'candidate': sha(after)}, 'documentation_commit': DOC_REF, 'included': included, 'prior_history': False, 'shared_branch': False, 'excluded': ['repository', 'implementer notes or conclusions', 'prior reviewer conclusions', 'prior CPU review outputs']}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    put('INPUT_MANIFEST.json', encoded)
    pin = sha(encoded)
    template = (HERE.parent / '20260926-fresh-astra-k2-namespace/session/operate.py').read_text()
    old_packet = '/tmp/trainium-fresh-k2-namespace-packet-20260926.1fj0qggt'
    old_pin = '0126e05a79e02313c742b0d94ee7ee74e58b7ea04765041854f5b4ee6d706029'
    if template.count(old_packet) != 1 or template.count(old_pin) != 1: raise RuntimeError('Reviewed isolation template drift')
    (run / 'operate.py').write_text(template.replace(old_packet, str(packet)).replace(old_pin, pin))
    result = {'packet': str(packet), 'run': str(run), 'input_manifest_sha256': pin, 'candidate_commit': REF, 'candidate_sha256': sha(after), 'model_requested': 'gpt-6-astra', 'operation_changes': ['packet path', 'manifest pin'], 'fresh_session_started': False, 'task_file_count': len(included), 'task_bytes': sum(r['bytes'] for r in included), 'new_evidence': ['official versioned API excerpts and offset semantics', 'exact unchanged native checker and reference'], 'previous_refusal_not_overridden': True}
    (HERE / 'PREPARED.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
