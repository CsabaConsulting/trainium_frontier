"""Local mocked binding checks. Never execute a host command."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import posixpath
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CHECKS = []


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)
    CHECKS.append(reason)


def source(name):
    return subprocess.check_output(['git','show','765b332d:research/v2/proposals/20260926-k2-registered-native-trn22/'+name],cwd=REPO).decode()


b = json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())
check(all(b[k] is False for k in ('stage_authorized','launch_authorized','host_reserved','operator_assigned','source_adoption')), 'disabled')
check((b['host'],b['label'],b['original_lock']['device'],b['original_lock']['inode']) == ('trn21','fs-20260926-ordinary-fp32-subnormal-trn21-d1',66305,596823), 'exact_binding')
check(b['command'][:8] == ['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','600s','/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4'], 'bounded_world4')
check(b['command'][8] == b['root']+'/rank_entry.py' and b['command'][9:] == b['checker_args'], 'exact_argv')
for name, pin in {**b['support_sha256'],'host_driver.py':b['driver_sha256']}.items():
    check(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == pin, 'pin_'+name)
for name in ('idle_guard.py','dispatch_driver.py'):
    check((HERE/name).read_text() == source(name), 'unchanged_'+name)
for name in ('host_driver.py','rank_entry.py'):
    old = ast.parse(source(name).replace('check_native.py','ordinary_subnormal_probe.py'))
    new = ast.parse((HERE/name).read_text())
    # Exact function bodies; module constants and description are intentionally rebound.
    check([ast.dump(n) for n in old.body if isinstance(n,(ast.FunctionDef,ast.If,ast.Import,ast.ImportFrom))] ==
          [ast.dump(n) for n in new.body if isinstance(n,(ast.FunctionDef,ast.If,ast.Import,ast.ImportFrom))], 'unchanged_body_'+name)
tree = ast.parse((HERE/'host_driver.py').read_text())
constants = {}
for n in tree.body:
    if isinstance(n,ast.Assign):
        constants[n.targets[0].id] = ast.literal_eval(n.value.args[0] if isinstance(n.value,ast.Call) else n.value)
for var,key in [('ROOT','root'),('LABEL','label'),('COMMAND','command'),('CHECKER_ARGS','checker_args'),('PINS','support_sha256'),('ENV','environment')]:
    check(constants[var] == b[key], 'constant_'+var)
old_tests = ast.parse(subprocess.check_output(['git','show','710ea677:research/v2/proposals/20260926-k2-symmetric-native-trn21/test_binding.py'],cwd=REPO).decode())
fixture = next(n for n in old_tests.body if isinstance(n,ast.FunctionDef) and n.name == 'fake_driver_case')
exec(compile(ast.Module(body=[fixture],type_ignores=[]),'<pinned-existing-mocked-driver-test>','exec'))
for scenario in ('success','wrong_host','stop','disk','hash','existing_cache','guard_refusal','child_nonzero','incomplete'):
    fake_driver_case(b,scenario)
print(json.dumps(dict(status='SELF_CHECK_PASS_MOCKED_NOT_NATIVE',checks=CHECKS,count=len(CHECKS),launch_authorized=False),indent=2))
