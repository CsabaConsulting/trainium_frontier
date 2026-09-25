#!/usr/bin/env python3
"""Exercise candidate OFF paths while rejecting any Torch/NKI import attempt."""
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

root = Path('/work/reconstructed/research/v2/lanes/k2-resident-consumer')
attempts = []
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('nki', 'torch', 'torch_neuronx'):
        attempts.append(name)
        raise RuntimeError('FORBIDDEN_NATIVE_OR_TORCH_IMPORT: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded

def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

consumer = load('consumer_nki')
checker = load('check_native')
# OFF must not even hash/read support files through the enabled branch.
def forbidden_digest(path):
    raise RuntimeError('SUPPORT_DIGEST_CALLED_WHILE_OFF')
checker.digest = forbidden_digest
results = []
for keywords in ({}, {'enabled': False}):
    try:
        consumer.build_kernel(**keywords)
    except RuntimeError as exc:
        message = str(exc)
        ok = message == 'EXPERIMENT_DISABLED: no native module imported'
    else:
        message, ok = 'unexpected return', False
    results.append({'kind': 'factory', 'kwargs': keywords, 'error': message, 'pass': ok})
for args in ([], ['--timing-repeats', '5', '--out-dir', '/work/off_must_not_exist',
                  '--expected-kernel-sha256', 'incorrect', '--expected-reference-sha256', 'incorrect']):
    sys.argv = ['check_native.py'] + args
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = checker.main()
    result = json.loads(stdout.getvalue())
    ok = code == 0 and result == {'status': 'DISABLED', 'native_imported': False, 'native_executed': False}
    results.append({'kind': 'checker', 'args': args, 'exit_code': code, 'stdout': stdout.getvalue(), 'pass': ok})
report = {'optimization': sys.flags.optimize, 'checks': results, 'forbidden_import_attempts': attempts,
          'off_directory_created': Path('/work/off_must_not_exist').exists(),
          'native_executed': False}
report['pass'] = all(x['pass'] for x in results) and not attempts and not report['off_directory_created']
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['pass'] else 1)
