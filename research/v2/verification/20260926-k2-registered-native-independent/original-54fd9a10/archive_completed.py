"""Post-verdict exact local archival through apply_patch; no candidate writes."""
import hashlib
from pathlib import Path
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parent
DEST = Path('/home/poojith-devan/Desktop/Trainium/research/v2/verification/20260926-k2-registered-native-independent/original-54fd9a10')
manifest = SOURCE / 'OUTPUT_SHA256.txt'
files = []
for line in manifest.read_text().splitlines():
    expected, name = line.split('  ', 1)
    if Path(name).name != name or name in ('.', '..'):
        raise RuntimeError('unsafe artifact name')
    files.append((name, expected))
files.append(('OUTPUT_SHA256.txt', hashlib.sha256(manifest.read_bytes()).hexdigest()))
if DEST.exists():
    raise RuntimeError('Archive destination already exists; refusing overwrite')
apply_patch = shutil.which('apply_patch')
truncate = shutil.which('truncate')
if not apply_patch or not truncate:
    raise RuntimeError('Required local archival tools unavailable')
for name, expected in files:
    raw = (SOURCE / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('source changed: ' + name)
    text = raw.decode('utf-8')
    body = text[:-1] if text.endswith('\n') else text
    patch = '*** Begin Patch\n*** Add File: ' + str(DEST / name) + '\n'
    patch += '\n'.join('+' + line for line in body.split('\n')) + '\n*** End Patch\n'
    result = subprocess.run([apply_patch], input=patch, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    archived = (DEST / name).read_bytes()
    # apply_patch adds an LF to a new text file. Preserve generator-emitted JSON
    # without a final LF by a narrowly checked mechanical formatting correction.
    if not raw.endswith(b'\n') and archived == raw + b'\n':
        subprocess.run([truncate, '-s', str(len(raw)), str(DEST / name)], check=True)
    if (DEST / name).read_bytes() != raw:
        raise RuntimeError('archive differs: ' + name)
    print(name + ': BYTE_IDENTICAL ' + expected, flush=True)
print('ARCHIVED_EXACT_FILES=' + str(len(files)), flush=True)
