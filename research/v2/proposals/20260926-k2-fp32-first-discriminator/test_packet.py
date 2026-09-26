#!/usr/bin/env python3
"""Author stdlib packet tests; synthetic receipts, no native effects."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import posixpath
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CHECKS = []


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)
    CHECKS.append(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


b = json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())
check(all(b[k] is False for k in ('stage_authorized', 'launch_authorized', 'source_adoption', 'operator_assigned', 'host_reserved')), 'disabled')
check(b['host'] == 'trn22' and b['expected_hostname'] == 'ip-172-31-32-253', 'host')
check(b['timeout_seconds'] == 600 and b['term_grace_seconds'] == 60, 'cap')
check(b['original_lock']['inode'] == 832589 and b['original_lock']['device'] == 66305, 'original_lock')
for name, pin in b['support_sha256'].items():
    check(sha((HERE/name).read_bytes()) == pin, 'pin_' + name)
check(sha((HERE/'host_driver.py').read_bytes()) == b['driver_sha256'], 'driver_pin')
check(shlex.split((HERE/'bound-command.txt').read_text()) == b['command'], 'command_text')
check('--timing-repeats' not in b['command'] and b['command'][9:] == b['checker_args'], 'no_timings')
check(b['command'][:8] == ['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s','/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4'], 'four_rank_watchdog')
tree = ast.parse((HERE/'host_driver.py').read_text())
constants = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        value = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
        constants[node.targets[0].id] = ast.literal_eval(value)
for var, key in [('ROOT','root'),('LABEL','label'),('PINS','support_sha256'),('COMMAND','command'),('CHECKER_ARGS','checker_args'),('ENV','environment')]:
    check(constants[var] == b[key], 'constant_' + var)
parent_prefix = '710ea6772d5525d97d990ecdf7f835f7f65d8380:research/v2/proposals/20260926-k2-symmetric-native-trn21/'
def original(name):
    return subprocess.check_output(['git','show',parent_prefix+name],cwd=REPO).decode()
for name in ['host_driver.py','rank_entry.py']:
    old, new = ast.parse(original(name)), ast.parse((HERE/name).read_text())
    old.body = [n for n in old.body if not isinstance(n, ast.Assign)]
    new.body = [n for n in new.body if not isinstance(n, ast.Assign)]
    check(ast.dump(old) == ast.dump(new), 'unchanged_body_' + name)
for name in ['idle_guard.py','dispatch_driver.py']:
    check((HERE/name).read_bytes() == (REPO/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes(), 'unchanged_shared_' + name)

spec=importlib.util.spec_from_file_location('validator', HERE/'validate_receipts.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
case=json.loads((HERE/'CPU-normal.json').read_text())['results'][0]['case']
files={}
for rank in range(4):
    files[f'/fixture/receipts/rank{rank}.json']=dict(status='FP32_FIRST_DIAGNOSTIC_OBSERVED_EXACT',returncode=0,
        scope='FP32_FIRST_SOURCE_NEUTRAL_DIAGNOSTIC_ONLY',rank=rank,world_size=4,label=b['label'],seed=42,
        shape=dict(V=256,T=256,C=128),argv=b['checker_args'],
        checker_sha256=b['support_sha256']['check_native.py'],kernel_sha256=b['support_sha256']['consumer_nki.py'],reference_sha256=b['support_sha256']['reference.py'],
        tolerance=dict(atol=0,rtol=0,criterion='identical bytes'),route_order=list(v.ROUTES),first_custom_compiled_fixture='unique/fp32',native_qualification=False,
        source_adoption=False,accepted_speedup=False,public_20m=None,whole_step_speedup='UNMEASURED',timings=[],
        original_twenty_case_gate='FAILED_EXACTNESS_PARKED_UNCHANGED',no_training=True,no_eval=True,diagnostic_elapsed_seconds=1.0,case=copy.deepcopy(case))
    files[f'/fixture/rank-environment-r{rank}.json']=dict(rank=rank,local_rank=rank,world_size=4,environment=copy.deepcopy(b['environment']),checker_sha256=b['support_sha256']['check_native.py'],checker_argv=b['checker_args'])
    files[f'/fixture/rank-wrapper-exit-r{rank}.json']=dict(rank=rank,returncode=0,error=None)


def validate(fixture):
    class VirtualPath:
        def __init__(self,path):self.path=str(path)
        def __truediv__(self,other):return VirtualPath(self.path+'/'+str(other))
        def read_text(self):return json.dumps(fixture[self.path])
    with patch.object(v,'Path',VirtualPath):
        return v.validate('/fixture',b['label'],b['environment'],b['support_sha256'],b['checker_args'])


check(validate(files)['native_qualification'] is False, 'complete_not_native_qualification')
key='/fixture/receipts/rank2.json'
mutations=[
 ('missing_rank',lambda f:f.pop('/fixture/receipts/rank3.json')),
 ('wrong_status',lambda f:f[key].update(status='TINY_CONSUMER_EXACT_ONLY')),
 ('boolean_rc',lambda f:f[key].update(returncode=False)),
 ('wrong_hash',lambda f:f[key].update(kernel_sha256='0'*64)),
 ('wrong_host_env',lambda f:f['/fixture/rank-environment-r2.json']['environment'].update(NEURON_LOGICAL_NC_CONFIG='1')),
 ('bad_exit',lambda f:f['/fixture/rank-wrapper-exit-r2.json'].update(returncode=1)),
 ('ordinary_missing',lambda f:f[key]['case']['routes'].pop()),
 ('route_reordered',lambda f:f[key]['case']['routes'].reverse()),
 ('bool_error',lambda f:f[key]['case']['routes'][0]['checks']['gradient'].update(max_abs=False)),
 ('relaxed_gate',lambda f:f[key]['tolerance'].update(atol=.01)),
 ('speed_claim',lambda f:f[key].update(accepted_speedup=True)),
 ('native_claim',lambda f:f[key].update(native_qualification=True)),
 ('original_failure_hidden',lambda f:f[key].update(original_twenty_case_gate='PASS')),
 ('zero_gradient',lambda f:f[key]['case']['routes'][2]['checks']['gradient']['actual'].update(nonzero=0,sha256='0'*64)),
 ('missing_gradient',lambda f:f[key]['case']['routes'][2]['checks'].pop('gradient')),
 ('saved_input_drift',lambda f:f[key]['case']['routes'][2]['inputs_after']['counts'].update(sha256='0'*64)),
 ('wrong_fixture',lambda f:f[key]['case'].update(case='cross_tile_large')),
 ('nonfinite',lambda f:f[key]['case']['routes'][0]['checks']['gradient']['actual'].update(finite=False)),
 ('captured_exception',lambda f:f[key]['case']['routes'][2].update(error='bad')),
 ('nan_elapsed',lambda f:f[key].update(diagnostic_elapsed_seconds=float('nan'))),
]
for name, mutation in mutations:
    changed=copy.deepcopy(files);mutation(changed)
    try:validate(changed)
    except (RuntimeError,KeyError,TypeError,ValueError):CHECKS.append('refuse_'+name)
    else:raise RuntimeError('false completion '+name)

# Reuse the proven mocked driver-effect fixture without invoking its old scientific tests.
old_tests=ast.parse(original('test_binding.py'))
fixture_fn=next(n for n in old_tests.body if isinstance(n,ast.FunctionDef) and n.name=='fake_driver_case')
exec(compile(ast.Module(body=[fixture_fn],type_ignores=[]),'<pinned-old-driver-fixture>','exec'))
for scenario in ('success','wrong_host','stop','disk','hash','existing_cache','guard_refusal','child_nonzero','incomplete'):
    fake_driver_case(b,scenario)
print(json.dumps(dict(status='IMPLEMENTER_STDLIB_PACKET_PASS_NOT_INDEPENDENT_REVIEW',count=len(CHECKS),checks=CHECKS,
    effects='All driver filesystem/process/guard effects mocked; no host/native/tensors',
    launch_authorized=False),indent=2))
