"""Tiny mocked inheritance/restoration and exact scope tests; no native/tools/tensors."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import types

HERE = Path(__file__).resolve().parent
OLD = 'research/v2/proposals/20260926-p-offline-decoder-trn21/'
checks = 0

def check(ok, why):
    global checks
    checks += 1
    if not ok:
        raise RuntimeError(why)

def original(name):
    return subprocess.check_output(['git', 'show', 'b529468f:' + OLD + name], cwd=HERE)

driver = (HERE / 'host_driver.py').read_text()
tree = ast.parse(driver)
wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'readiness_guard')
pins = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PINS' for t in n.targets)))
plan = json.loads((HERE / 'EXPERIMENT_PROPOSAL.json').read_text())
parent = json.loads(original('EXPERIMENT_PROPOSAL.json'))
check(plan['limits'] == parent['limits'], 'ALL resource ceilings unchanged')
check(plan['inputs'] == parent['inputs'], 'ALL86 input paths/sizes unchanged')
check(plan['source_root'] == parent['source_root'], 'same protected profile root')
check(plan['core_argv'] == [v.replace(parent['root'], plan['root']) for v in parent['core_argv']], 'only unique session path rebound')
check(not any(plan[k] for k in ('stage_authorized', 'launch_authorized', 'host_reserved')), 'disabled')
check((HERE / 'decoder_support.py').read_bytes() == original('decoder_support.py'), 'unchanged full containment helper')
check(hashlib.sha256((HERE / 'EXPERIMENT_PROPOSAL.json').read_bytes()).hexdigest() == pins['EXPERIMENT_PROPOSAL.json'], 'proposal pin')
old = original('host_driver.py').decode().replace(parent['root'], plan['root']).replace(hashlib.sha256(original('EXPERIMENT_PROPOSAL.json')).hexdigest(), pins['EXPERIMENT_PROPOSAL.json'])
old = old.replace("lock_fd = guard.require_idle_under_inherited_lock(LOCK, ROOT / 'host_driver.py', record)", 'lock_fd = readiness_guard(guard, limits, record)')
old = old.replace("post_fd = guard.require_idle_under_inherited_lock(LOCK, ROOT / 'host_driver.py',", 'post_fd = readiness_guard(guard, limits,')
without = ast.Module(body=[n for n in tree.body if n is not wrapper], type_ignores=[])
check(ast.dump(without, include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False), 'ENTIRE old driver AST only two wrappers/root/proposal pin')
check(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'readiness_guard' for n in ast.walk(tree)) == 2, 'exactly pre AND post guarded calls')

low, high = 536870912, 17179869184
class FakeResource:
    RLIMIT_AS = 9
    def __init__(self, state=(low, high), fault=None):
        self.state, self.fault, self.calls = state, fault, []
    def getrlimit(self, kind):
        check(kind == self.RLIMIT_AS, 'only AS queried')
        return self.state
    def setrlimit(self, kind, value):
        check(kind == self.RLIMIT_AS, 'no CPU/file/other limit mutation')
        self.calls.append(value)
        if self.fault == 'raise' and value == (high, high):
            raise OSError('injected raise-limit failure')
        if self.fault == 'restore' and value == (low, high):
            raise OSError('injected restoration failure')
        self.state = value

for mode in ('success', 'guard_error', 'interrupt', 'receipt_error', 'raise', 'restore', 'unexpected_parent'):
    resource = FakeResource(state=(low-1, high) if mode == 'unexpected_parent' else (low, high), fault=mode)
    calls, receipts = [], []
    def record(name, value):
        check(resource.state == (low, high), 'receipt/parent work executes after restoration')
        receipts.append((name, value))
        if mode == 'receipt_error':
            raise RuntimeError('injected receipt failure')
    def guard_call(lock, driver_path, callback):
        calls.append((lock, str(driver_path)))
        check(resource.state == (high, high), 'external readiness children inherit exactly16GiB')
        if mode == 'guard_error':
            raise RuntimeError('NEURON_LS_COMMAND_FAILED')
        if mode == 'interrupt':
            raise KeyboardInterrupt('injected signal')
        return 19
    ns = dict(resource=resource, LOCK='/original/lock', ROOT=Path('/new/r2'))
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<exact-wrapper>', 'exec'), ns)
    error, result = None, None
    try:
        result = ns['readiness_guard'](types.SimpleNamespace(require_idle_under_inherited_lock=guard_call), plan['limits'], record)
    except BaseException as exc:
        error = exc
    check((error is None) == (mode == 'success'), 'no failure becomes success:' + mode)
    if mode == 'success':
        check(result == 19 and calls == [('/original/lock', '/new/r2/host_driver.py')], 'same authenticated FD/path')
    if mode not in ('restore', 'unexpected_parent'):
        check(resource.state == (low, high), 'finally restores on success/failure:' + mode)
        check(resource.calls[-1] == (low, high), 'last operation restores:' + mode)
    if mode in ('raise', 'unexpected_parent'):
        check(not calls, 'no guard after failed precondition/elevation')
    if mode == 'guard_error':
        check(str(error) == 'NEURON_LS_COMMAND_FAILED' and receipts[-1][1]['guard_error'].endswith(str(error)), 'original refusal propagates and records')
    if mode == 'unexpected_parent':
        check(not resource.calls, 'unexpected starting limits not modified')
    if mode == 'restore':
        check(error is not None and not receipts, 'restoration failure cannot claim restored evidence')

# Preserve the old inheritance failure as a mocked negative, not an installed Go test.
old_resource = FakeResource()
def go_readiness(*args):
    if old_resource.state[0] == low:
        raise RuntimeError('fatal error: failed to reserve page summary memory')
    return 3
failed = False
try:
    go_readiness('/original/lock', Path('/old/r1/host_driver.py'), lambda *x: None)
except RuntimeError as exc:
    failed = str(exc) == 'fatal error: failed to reserve page summary memory'
check(failed and old_resource.state == (low, high), 'old unwrapped inheritance reproducer (mock only)')
ns = dict(resource=old_resource, LOCK='/original/lock', ROOT=Path('/new/r2'))
exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<exact-wrapper>', 'exec'), ns)
check(ns['readiness_guard'](types.SimpleNamespace(require_idle_under_inherited_lock=go_readiness), plan['limits'], lambda *x: None) == 3, 'same mocked readiness succeeds under exact16GiB')
check(old_resource.state == (low, high), 'same reproducer restores512MiB')
if (HERE / 'OFFLINE_BINDING.disabled.json').exists():
    binding = json.loads((HERE / 'OFFLINE_BINDING.disabled.json').read_text())
    check(not any(binding[k] for k in ('stage_authorized', 'launch_authorized', 'host_reserved')), 'binding remains disabled')
    check(binding['root'] == plan['root'] and binding['label'] == plan['label'], 'new identity coherent')
    check(binding['core_argv'] == plan['core_argv'] and binding['bounds'] == plan['limits'], 'argv/ALL limits bound')
    for name, pin in binding['runtime_sha256'].items():
        data = (HERE / name).read_bytes() if (HERE / name).exists() else subprocess.check_output(['git','show','b529468f:research/v2/operator/20260926-common-host-idle-guard/' + name], cwd=HERE)
        check(hashlib.sha256(data).hexdigest() == pin, 'bound runtime SHA:' + name)
    check(binding['test_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'test pin')
print(json.dumps({'status':'LOCAL_IMPLEMENTER_TESTS_PASS_NOT_INDEPENDENT_APPROVAL','checks':checks,'modes':7,'native_executed':False,'go16GiB_compatibility':'UNPROVEN'}, sort_keys=True))
