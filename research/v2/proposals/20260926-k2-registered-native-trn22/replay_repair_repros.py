#!/usr/bin/env python3
"""Read-only replay of authenticated synthetic reviewer counterexamples.

Never edits the review directory or calls its test runner. No runtime imports.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import types
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PINS = {
    'synthetic-baseline.json': '1d025908661b2f5b371f47397f96af114e5eee3149b4cb05b443d23a106dcf6d',
    'counterexample-contradictory_candidate_forward.json': 'cab7d4cc81910870ea3e1cc5557a55b7586f8745a40cec2607a46c4c8ad02776',
    'counterexample-contradictory_upstream_count.json': '44aea413ebc70dd121342941d95449ae1f324f382367dfb7b9320ce786efb218',
    'counterexample-missing_runtime_versions.json': '3ccfd61275b5338af7ee3b6ed3388894563ec7e9001666fdd3fa5b064796b9a9',
    'counterexample-malformed_old_mismatch.json': '3a075f736cbca04923c9bdddcbc80cf91c333300dcb67c61b5d520adc1892c68',
}


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewer-dir', required=True, type=Path)
    args = p.parse_args()
    binding = json.loads((HERE / 'NATIVE_BINDING.disabled.json').read_text())
    spec = importlib.util.spec_from_file_location('repaired_validator', HERE / 'validate_receipts.py')
    new = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new)
    raw = subprocess.check_output(['git', 'show', '54fd9a10:research/v2/proposals/20260926-k2-registered-native-trn22/validate_receipts.py'], cwd=HERE)
    require(hashlib.sha256(raw).hexdigest() == '0ed1d17a9e389c87cb6cfe63d80ca54986dc2167a0bb9e8c49c01a19d34dd402', 'ORIGINAL_PIN')
    old = types.ModuleType('original_refused_validator')
    old.__file__ = str(HERE / 'validate_receipts.py')
    exec(compile(raw, old.__file__, 'exec'), old.__dict__)
    results = []
    for name, pin in PINS.items():
        raw = (args.reviewer_dir / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == pin, 'REPRO_PIN:' + name)
        material = json.loads(raw)
        require(material['SYNTHETIC_ONLY'] is True, 'NOT_SYNTHETIC')
        documents = material['documents']
        class MemoryPath(PurePosixPath):
            def read_text(self):
                return json.dumps(documents[str(self.relative_to('/synthetic-only'))], allow_nan=False)
        observations = {}
        for label, module in [('original_54fd9a10', old), ('repaired', new)]:
            try:
                with patch.object(module, 'Path', MemoryPath):
                    module.validate('/synthetic-only', binding['label'], binding['environment'],
                                    binding['support_sha256'], binding['checker_args'])
                observations[label] = dict(accepted=True)
            except Exception as exc:
                observations[label] = dict(accepted=False, error=type(exc).__name__ + ': ' + str(exc))
        require(observations['original_54fd9a10']['accepted'], 'Original reproduction changed:' + name)
        require(observations['repaired']['accepted'] == name.startswith('synthetic-baseline'), 'Repair outcome:' + name)
        results.append(dict(file=name, sha256=pin, observations=observations))
    require('torch' not in sys.modules and 'nki' not in sys.modules, 'NATIVE_IMPORT_FORBIDDEN')
    print(json.dumps(dict(status='OWN_READONLY_REPLAY_COMPLETED_NOT_INDEPENDENT_REVIEW',
                         original_gate='REFUSED_PRESERVED', optimized=not __debug__, results=results,
                         native_executed=False, launch_authorized=False), indent=2))


if __name__ == '__main__':
    main()
