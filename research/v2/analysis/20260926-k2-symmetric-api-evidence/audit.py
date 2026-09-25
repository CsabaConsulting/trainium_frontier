"""Authenticate archived documentation excerpts and exact support; no imports/SDK/network."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMIT = '9ce2e6a037212aaa76bd85b9b4ab7d311a481757'
PREFIX = 'research/v2/lanes/k2-symmetric-consumer/'
PINS = {
    'consumer_nki.py': '2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae',
    'check_native.py': '89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850',
    'reference.py': '14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f',
}
checks = 0

def require(ok, why):
    global checks
    checks += 1
    if not ok:
        raise RuntimeError(why)

def sha(data):
    return hashlib.sha256(data).hexdigest()

sources = json.loads((ROOT / 'SOURCES.json').read_text())
require(sources['scope'] == 'PRIMARY_SOURCE_DOCUMENTATION_ONLY_NOT_SOURCE_OR_NATIVE_ACCEPTANCE', 'scope')
require(len(sources['sources']) == 11, 'source count')
require(len({s['id'] for s in sources['sources']}) == 11, 'unique sources')
excerpts = {}
for s in sources['sources']:
    require(s['url'].startswith('https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/'), 'primary/versioned URL')
    require(sum(len(q.split()) for q in s['excerpts']) <= 25, 'bounded per-page quotation')
    require(s['web_text_lines'] and s['section'] and s['title'], 'source locator')
    excerpts[s['id']] = [{'utf8_bytes': len(q.encode()), 'sha256': sha(q.encode())} for q in s['excerpts']]
support = {}
for name, pin in PINS.items():
    data = (ROOT / 'support' / name).read_bytes()
    git_data = subprocess.check_output(['git', 'show', COMMIT + ':' + PREFIX + name], cwd=ROOT)
    require(sha(data) == pin, name + ' SHA256')
    require(data == git_data, name + ' exact Git source')
    support[name] = {'sha256': pin, 'bytes': len(data), 'commit': COMMIT, 'path': PREFIX + name}
if (ROOT / 'EXCERPT_HASHES.json').exists():
    require(json.loads((ROOT / 'EXCERPT_HASHES.json').read_text())['excerpt_hashes'] == excerpts,
            'archived excerpt hashes')
if (ROOT / 'MANIFEST.json').exists():
    for name, entry in json.loads((ROOT / 'MANIFEST.json').read_text())['files'].items():
        data = (ROOT / name).read_bytes()
        require(sha(data) == entry['sha256'] and len(data) == entry['bytes'], 'manifest:' + name)
print(json.dumps({'status': 'ARCHIVE_AUTHENTICATED_NOT_GATE_ACCEPTANCE', 'checks': checks,
                  'excerpt_hash_scope': 'UTF-8 rendered excerpt text excluding newline; NOT entire webpage',
                  'excerpt_hashes': excerpts, 'support': support,
                  'native_executed': False, 'source_gate_changed': False}, sort_keys=True, indent=2))
