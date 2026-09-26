"""Archive this exact candidate's generated outputs; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SOURCE_SHA = 'f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e'
parser = argparse.ArgumentParser()
parser.add_argument('--receipt', type=Path, required=True)
args = parser.parse_args()
receipt_path = args.receipt.resolve()
assert receipt_path.parent == HERE, 'RECEIPT_OUTSIDE_LANE'
receipt = json.loads(receipt_path.read_text())
assert receipt['status'] == 'FULL_BACKWARD_COMPILE_PASS'
assert receipt['source_sha256'] == SOURCE_SHA
assert hashlib.sha256((HERE / 'train.py').read_bytes()).hexdigest() == SOURCE_SHA
assert receipt['production_default_options'] is True
scratch = Path(receipt['artifacts_dir']).resolve()
assert scratch.parent == HERE / 'scratch', 'ARTIFACTS_OUTSIDE_LANE'
artifacts = HERE / 'artifacts'
artifacts.mkdir(exist_ok=True)
mapping = {'k2.neff': 'k2.neff', 'log-neuron-cc.txt': 'backend.log',
           'nc00/sg00/instruction_stats.txt': 'core0-instruction-stats.txt',
           'nc01/sg00/instruction_stats.txt': 'core1-instruction-stats.txt',
           'kernel_info.json': 'kernel-info.json', 'mempressure.txt': 'mempressure.txt'}
manifest = {'source_sha256': SOURCE_SHA,
            'compile_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            'compile_status': receipt['status'], 'artifacts': {}}
for source_name, dest_name in mapping.items():
    source = scratch / source_name
    destination = artifacts / dest_name
    if destination.exists():
        assert destination.read_bytes() == source.read_bytes(), 'ARCHIVE_CONFLICT'
    else:
        shutil.copyfile(source, destination)
    raw = destination.read_bytes()
    manifest['artifacts'][dest_name] = {'bytes': len(raw),
                                      'sha256': hashlib.sha256(raw).hexdigest()}
assert manifest['artifacts']['k2.neff']['sha256'] == receipt['neff_sha256']
manifest_path = HERE / 'artifacts-manifest.json'
serialized = json.dumps(manifest, indent=2) + '\n'
if manifest_path.exists():
    assert manifest_path.read_text() == serialized, 'MANIFEST_CONFLICT'
else:
    manifest_path.write_text(serialized)
print(serialized, end='')
