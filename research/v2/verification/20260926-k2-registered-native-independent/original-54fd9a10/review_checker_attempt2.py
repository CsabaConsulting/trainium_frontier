"""Checker control flow and comparison mocks; no Tensor/native modules imported."""
import argparse
import ast
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
PACKET = OUT.parent
spec = importlib.util.spec_from_file_location('own_common', OUT / 'review_validator.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
require, emit = common.require, common.emit
common.authenticate()
checker = common.load('frozen_checker_mock_review', PACKET / 'check_native.py')
reference = common.load('frozen_reference_mock_review', PACKET / 'reference.py')
binding = json.loads((PACKET / 'NATIVE_BINDING.disabled.json').read_text())
tree = ast.parse((PACKET / 'check_native.py').read_bytes())
main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')

# Verify unchanged helpers against the minus side of the supplied complete diff.
diff = (PACKET / 'checker-vs-fp32-first.diff').read_text()
old_lines = []
inside = False
for line in diff.splitlines():
    if line.startswith('@@'):
        inside = True
    elif inside and (line.startswith(' ') or (line.startswith('-') and not line.startswith('---'))):
        old_lines.append(line[1:])
# The patch may omit unchanged leading/trailing source; helper source functions
# are checked by matching their unmodified bodies in the explicit diff context.
for name in ('make_wrapper', 'tensor_record', 'compare', 'observe_route'):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    segment = ast.get_source_segment((PACKET / 'check_native.py').read_text(), node)
    old_text = '\n'.join(old_lines)
    require(segment in old_text, 'helper absent/changed in frozen old diff context: ' + name)
emit('helper_source_identity', unchanged=['make_wrapper', 'tensor_record', 'compare', 'observe_route'])

with patch.object(sys, 'argv', ['check_native.py']), contextlib.redirect_stdout(io.StringIO()) as default_output:
    require(checker.main() == 0, 'disabled return')
require(json.loads(default_output.getvalue())['status'] == 'DISABLED', 'disabled evidence')
require(not any(n == 'torch' or n == 'nki' or n.startswith(('torch.', 'nki.', 'torch_neuronx')) for n in sys.modules), 'disabled native import')
emit('disabled_checker', status='PASS', output=json.loads(default_output.getvalue()))


def orchestration_trial(fault=None):
    events, calls, registrations, streams = [], [], [], {}
    class ReceiptStream(io.StringIO):
        def __exit__(self, *args):
            return False
    class MemoryPath(PurePosixPath):
        def mkdir(self, **kwargs):
            require(kwargs == dict(parents=True, exist_ok=False), 'unique output creation')
            events.append('output_created')
        def is_dir(self):
            return True
        def open(self, mode):
            require(mode == 'x', 'exclusive receipt')
            events.append('receipt_open_attempt')
            if fault == 'receipt_io_failure':
                raise OSError('MOCK_RECEIPT_IO_FAILURE')
            handle = ReceiptStream()
            streams[str(self)] = handle
            return handle
    def compile_fn(function, **kwargs):
        require(kwargs == dict(backend='neuron', fullgraph=True, dynamic=False), 'native compile intent')
        events.append('compile_created')
        return object()
    def build_forward(kernel, dtype, **kwargs):
        require(kwargs == dict(enabled=True), 'explicit adapter activation')
        registrations.append(dtype)
        return object()
    def init(group):
        require(group == 'neuron', 'native group intent')
        events.append('init_group')
        if fault == 'init_failure':
            raise RuntimeError('MOCK_INIT_FAILURE')
    dist = SimpleNamespace(init_process_group=init, destroy_process_group=lambda:events.append('destroy_group'),
                           barrier=lambda:events.append('barrier'))
    fake_torch = SimpleNamespace(__version__='MOCK_TORCH_NOT_RUNTIME_EVIDENCE',
        float32='fp32', bfloat16='bf16', compile=compile_fn, distributed=dist,
        nn=SimpleNamespace(functional=SimpleNamespace()),
        set_num_threads=lambda n:require(n == 1, 'one thread'),
        set_num_interop_threads=lambda n:require(n == 1, 'one interop'),
        manual_seed=lambda n:require(n == 42, 'seed42'), device=lambda name:'MOCK_DEVICE',
        neuron=SimpleNamespace(synchronize=lambda:None))
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # No named native or Tensor module is loaded; these are plain namespaces.
        if name in ('torch', 'torch.distributed', 'torch.nn.functional'):
            return dist if name == 'torch.distributed' and fromlist else fake_torch
        if name == 'torch_neuronx':
            return SimpleNamespace()
        if name == 'consumer_nki':
            return SimpleNamespace(build_kernel=lambda **kwargs:object())
        if name == 'reference':
            return reference
        if name == 'adapter':
            return SimpleNamespace(build_registered_forward=build_forward)
        raise RuntimeError('Unexpected import from checker main: ' + name)
    def run_case(name, dtype, kernel, device, sync, functions, *, changed_tangent=False):
        calls.append(dict(name=name, dtype=dtype, changed=changed_tangent,
                          functions_id=id(functions), compiled_id=id(functions['registered_compiled'])))
        if fault == 'suite_exception' and len(calls) == 3:
            raise RuntimeError('MOCK_SUITE_EXCEPTION')
        exact = not ((fault == 'sentinel_failure' and len(calls) == 1) or
                     (fault == 'required_case_failure' and len(calls) == 3))
        return dict(name=name, dtype=dtype, changed_tangent=changed_tangent,
                    status='EXACT' if exact else 'MISMATCH_OR_ERROR',
                    checks=dict(mock=dict(exact=exact)), routes=[dict(status='EXACT' if exact else 'MISMATCH', inputs_unchanged=True)],
                    original_routes=[dict(status='ERROR' if fault == 'old_error' else 'MISMATCH', error='MOCK_OLD_FAILURE')])
    ns = dict(__builtins__=dict(vars(builtins), __import__=fake_import), __file__=str(PACKET / 'check_native.py'),
              argparse=argparse, Path=MemoryPath, HERE=PACKET, time=time, json=json, traceback=traceback,
              sys=SimpleNamespace(argv=['check_native.py', *binding['checker_args']]),
              os=SimpleNamespace(environ=dict(WORLD_SIZE='4', NEURON_LOGICAL_NC_CONFIG='2', RANK='0')),
              importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda name:'MOCK_NKI_NOT_RUNTIME_EVIDENCE')),
              digest=lambda path:hashlib.sha256((PACKET / path.name).read_bytes()).hexdigest(),
              KERNEL_SHA=checker.KERNEL_SHA, REFERENCE_SHA=checker.REFERENCE_SHA, ADAPTER_SHA=checker.ADAPTER_SHA,
              run_case=run_case, case_exact=checker.case_exact,
              make_wrapper=lambda *args:(object(), object()))
    exec(compile(ast.Module(body=[main_node], type_ignores=[]), str(PACKET / 'check_native.py'), 'exec'), ns)
    error, code = None, None
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        try:
            code = ns['main']()
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    emit('mock_outcome_before_classification', fault=fault, code=code, error=error, events=events,
         captured_stdout=stdout.getvalue() if error is not None else None)
    if fault not in ('init_failure',):
        require('destroy_group' in events, 'cleanup after initialized outcome')
    else:
        require('destroy_group' not in events, 'no uninitialized group cleanup')
    require('receipt_open_attempt' in events, 'final receipt attempted')
    if fault == 'sentinel_failure':
        require(len(calls) == 1 and code == 1 and registrations == ['fp32'], 'sentinel gates full suite')
    elif fault in (None, 'old_error', 'required_case_failure', 'receipt_io_failure'):
        require(len(calls) == 23 and registrations == ['fp32', 'bf16'], 'full suite / once per dtype')
        expected = [('unique', 'fp32', False)]
        for dtype in ('bf16', 'fp32'):
            expected += [(name, dtype, False) for name in reference.CASE_NAMES] + [('unique', dtype, True)]
        require([(c['name'], c['dtype'], c['changed']) for c in calls] == expected, 'exact sentinel full20 replays order')
        for dtype in ('bf16', 'fp32'):
            require(len({c['functions_id'] for c in calls if c['dtype'] == dtype}) == 1 and
                    len({c['compiled_id'] for c in calls if c['dtype'] == dtype}) == 1, 'runtime callable reuse')
        if fault == 'required_case_failure':
            require(code == 1, 'required mismatch cannot complete')
        elif fault == 'receipt_io_failure':
            require(error is not None, 'I/O failure must fail')
        else:
            require(code == 0, 'old failure does not suppress candidates')
    elif fault in ('suite_exception', 'init_failure'):
        require(code == 1, 'exception must fail')
    with (OUT / ('mock-checker-' + str(fault or 'normal') + ('-optimized' if not __debug__ else '') + '.log')).open('x') as handle:
        handle.write(stdout.getvalue())
    emit('mock_checker_orchestration', fault=fault, code=code, error=error, calls=calls,
         registrations=registrations, events=events, real_native_import=False)


for fault in (None, 'sentinel_failure', 'required_case_failure', 'old_error', 'suite_exception', 'init_failure', 'receipt_io_failure'):
    orchestration_trial(fault)

# Directly exercise the frozen comparison/observer helpers with synthetic scalar objects.
class Scalar:
    def __init__(self, value):
        self.value = value
        self.record = dict(shape=[1], dtype='torch.float32', finite=math.isfinite(value),
                           nonzero=int(value != 0), sha256=hashlib.sha256(repr(value).encode()).hexdigest())
    def detach(self): return self
    def cpu(self): return self
    def double(self): return self
    def __sub__(self, other): return Scalar(self.value - other.value)
    def abs(self): return Scalar(abs(self.value))
    def max(self): return self.value

with patch.object(checker, 'tensor_record', lambda value:copy.deepcopy(value.record)):
    expected = Scalar(1.0)
    for name, value in [('nan', float('nan')), ('positive_inf', float('inf')), ('negative_inf', -float('inf'))]:
        result = checker.compare(Scalar(value), expected)
        require(result['exact'] is False and result['max_abs'] is None and 'max_abs_nonfinite' in result, 'safe nonfinite compare')
        json.dumps(result, allow_nan=False)
        emit('safe_nonfinite_comparison', name=name, result=result)
    missing = checker.compare(None, expected)
    require(missing['missing'] is True and missing['exact'] is False, 'missing compare')
    signed = checker.compare(Scalar(-0.0), Scalar(0.0))
    require(signed['exact'] is False and signed['max_abs'] == 0, 'signed-zero bytes')
    inputs = {str(i):Scalar(float(i)) for i in range(7)}
    observed = checker.observe_route('synthetic_error', inputs,
        lambda:(_ for _ in ()).throw(RuntimeError('MOCK_EXEC_ERROR')), {'gradient':expected}, lambda:None)
    require(observed['status'] == 'ERROR' and 'MOCK_EXEC_ERROR' in observed['error'] and observed['inputs_unchanged'], 'preserve route error and post hashes')
    emit('observer_error', result=observed)

require(not any(n == 'torch' or n == 'nki' or n.startswith(('torch.', 'nki.', 'torch_neuronx')) for n in sys.modules), 'real runtime imported')
emit('checker_summary', status='PASS_MOCKED_CHECKER_ONLY', orchestration_trials=7,
     original_comparisons_per_rank='18 nonempty * 8 + 2 empty * 3 = 150',
     native_executed=False, native_qualification=False, original_source_bytes_unchanged=True)
