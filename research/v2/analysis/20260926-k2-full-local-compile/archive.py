"""Archive generated compiler outputs and exact hashes, without model edits."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
receipt_path = HERE / 'compile-production-result-1790416102882879459.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['status'] == 'FULL_BACKWARD_COMPILE_PASS'
scratch = Path(receipt['artifacts_dir'])
artifacts = HERE / 'artifacts'
artifacts.mkdir(exist_ok=True)
mapping = {'k2.neff': 'k2.neff', 'log-neuron-cc.txt': 'backend.log',
           'nc00/sg00/instruction_stats.txt': 'core0-instruction-stats.txt',
           'nc01/sg00/instruction_stats.txt': 'core1-instruction-stats.txt',
           'kernel_info.json': 'kernel-info.json', 'mempressure.txt': 'mempressure.txt'}
manifest = {'source_sha256': receipt['source_sha256'],
            'compile_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            'compile_status': receipt['status'], 'artifacts': {}}
for source_name, dest_name in mapping.items():
    source = scratch / source_name
    destination = artifacts / dest_name
    if destination.exists():
        assert destination.read_bytes() == source.read_bytes(), 'archive conflict'
    else:
        shutil.copyfile(source, destination)
    raw = destination.read_bytes()
    manifest['artifacts'][dest_name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
assert manifest['artifacts']['k2.neff']['sha256'] == receipt['neff_sha256']
(HERE / 'artifacts-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
