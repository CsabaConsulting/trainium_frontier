"""Independent stdlib-only frozen binding/raw-evidence checks; never executes the driver."""
import ast
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

REPO = Path('/home/poojith-devan/Desktop/Trainium')
PACKET = 'research/v2/proposals/20260926-base-cache-full-trn21/'
REF = '7da2559f03cfda04af3a7858e1b2d4037f8b676b'
rows = []

def check(ok, why):
    if not ok:
        raise RuntimeError(why)
    rows.append(why)

def raw(path, ref=REF):
    return subprocess.check_output(['git','show',ref + ':' + path], cwd=REPO)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def get(name):
    return json.loads(raw(PACKET + name))

b, p = get('NATIVE_BINDING.disabled.json'), get('EXPERIMENT_PROPOSAL.json')
manifest = get('MANIFEST.sha256.json')
for name, pin in manifest['files'].items():
    check(sha(raw(PACKET + name)) == pin, 'packet pin:' + name)
source = raw(b['source_git_path'], b['source_commit'])
check(sha(source) == b['source_sha256'] == p['source_sha256'], 'exact bcbd source')
baseline = raw(b['source_git_path'], 'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b')
without = b''.join(line for line in source.splitlines(keepends=True) if b"parser.add_argument('--initial-cache-refresh'" not in line)
without = without.replace(b'        if step == 0 and args.initial_cache_refresh:\n            for m in cached_linears:\n                m.refresh_weight_cache(COMPUTE_DTYPE)\n', b'')
check(without == baseline, 'only exact four-line source addition vs6ef7')
driver = raw(PACKET + 'host_driver.py')
prior = raw(b['derivation']['parent_driver_path'], b['derivation']['parent_commit'])
check(sha(prior) == b['derivation']['parent_driver_sha256'], 'frozen predecessor pin')
old_binding = json.loads(raw(b['derivation']['parent_driver_path'].replace('host_driver.py','NATIVE_BINDING.disabled.json'), b['derivation']['parent_commit']))
mapped = prior.decode().replace(old_binding['label'], b['label']).replace(old_binding['source_sha256'], b['source_sha256']).replace('Exact architecture Recursive stack then fresh public20M under one lifetime lock.', 'Exact base initial-cache quality observation then fresh public20M under one lifetime lock.')
check(mapped.encode() == driver, 'ENTIRE driver literal equality')
check(sha(driver) == b['driver_sha256'], 'driver SHA')
for key in ('command','evaluation_command','training_environment','evaluation_environment','outer_command','budgets','evaluation','device_lock'):
    mapped_obj = json.loads(json.dumps(old_binding[key]).replace(old_binding['label'], b['label']))
    check(mapped_obj == b[key], 'unchanged bound mechanical field:' + key)
for key in ('stage_authorized','launch_authorized','host_reserved','source_adoption','submission_authorized'):
    check(b[key] is False and p[key] is False, 'disabled:' + key)
check('-O' not in b['outer_command'] and '-O' not in b['dispatch_command'], 'normal Python assertions required')
ledger = [json.loads(line) for line in (REPO/'research/v2/ledger.jsonl').read_text().splitlines() if line.strip()]
earlier = next(r for r in ledger if r['label'] == 'fs-20260925-l3-timer-dense-trn21-r1-full')
later = next(r for r in ledger if r['label'] == 'fs-20260925-l3-base-repeat-trn21-r2-full')
check(earlier['kind'] == 'schedule_arm' and earlier['public_20m'] == .9900560897366009, 'earlier schedule_arm remains earlier')
check(later['kind'] == 'control' and later['public_20m'] == .9899528600562257 and later['host'] == 'trn21', 'genuine later same-host control retained')
check(later['argv'] == earlier['argv'] and later['train_py_sha256'] == earlier['train_py_sha256'], 'same science control vs earlier arm')
check(b['training_command'] == ['/home/ubuntu/workspace/native_venv/bin/torchrun','--standalone','--nproc_per_node=4','train.py'] + earlier['argv'] + ['--initial-cache-refresh'], 'science adds initial-cache only;Recursive alreadyON')
check(b['training_command'] == p['argv'] == shlex.split(raw(PACKET+'package/launch-command.txt').decode()), 'exact launch command/proposal')
for name, pin in b['package_support_sha256'].items():
    check(sha(raw(PACKET+'package/'+name)) == pin, 'package:' + name)
guard_path = 'research/v2/operator/20260926-common-host-idle-guard/'
guard = raw(guard_path+'idle_guard.py')
check(sha(guard) == b['idle_guard_sha256'], 'unchanged common guard')
check(sha(raw(guard_path+'dispatch_driver.py')) == b['dispatch_driver_sha256'], 'unchanged sole dispatcher')
check(sha((REPO/'prepare.py').read_bytes()) == b['prepare_sha256'], 'unchanged evaluator')
harness = raw('research/scratch/submission_rehearsal.py', '00f2ca7f79696e65ce52458f074dd13b871da90c')
check(sha(harness) == b['harness_sha256'], 'unchanged official-clock observer')
check(b'"--batch-size", "1", "--seq-len", "1024"' in harness and b'"--timeout-seconds", "1200"' in harness, 'fresh fixed evaluator configuration')
check(b['evaluation_command'][-2:] == ['--eval-tokens','20971520'], 'exact public20M')
check(driver.count(b'require_idle_under_inherited_lock(') == 2 and b'pass_fds=(lock_fd,)' in driver, 'train/eval freshguard and inheritedFD')

# Authenticate actual cold bytes, not the packet summary alone.
c = b['cold_gate']
cold_data = (REPO/c['archive']).read_bytes()
check(sha(cold_data) == c['receipt_sha256'], 'actual cold receipt pin')
cold = json.loads(cold_data)
check(sha((REPO/c['launch_archive']).read_bytes()) == c['launch_sha256'], 'actual cold launch pin')
check(cold['returncode'] == 0 and cold['step_boundaries'] == 5 and len(cold['fresh_empty_sdk_caches']) == 6, 'actual cold exit/five/six caches')
fifth = cold['observations'][4]['observed_seconds']
check(fifth == 493.3377903739456 and fifth <= 600 and c['host'] == 'trn22', 'cold is trn22 only;target satisfied')
check(math.isclose(cold['wall_seconds']-min(fifth,900),cold['charged_through_process_exit'],abs_tol=1e-10) and cold['charged_through_process_exit_within_budget'] is True, 'actual cold accounting')

# Authenticate all88 raw CPU artifacts, then independently compare recorded numeric fields.
cpu = 'research/v2/operator/20260926-trn22-base-cache-cpu/results-complete/'
cpu_ref = '2718bc5c2de31a73ef01ffaad753eb4da0b9fe45'
cpu_get = lambda n: json.loads(raw(cpu+n,cpu_ref))
cm = cpu_get('RETRIEVAL_MANIFEST.json')
check(len(cm['files']) == 88, 'complete CPU artifact count')
for name, info in cm['files'].items():
    data = raw(cpu+name,cpu_ref)
    check(len(data) == info['bytes'] and sha(data) == info['sha256'], 'CPU raw:' + name)
ref_reports = [cpu_get('reference-6ef7/rank%d.json'%r) for r in range(4)]
ref_cp = cpu_get('reference-6ef7/checkpoint-identity.json')
for arm in ('candidate-default','candidate-off','candidate-on'):
    for rank, reference in enumerate(ref_reports):
        got = cpu_get(arm+'/rank%d.json'%rank)
        check(got['status'] == 'PASS', arm+' raw rank/pass'+str(rank))
        if rank == 0:
            check(got['checkpoint_check']['ok'], arm+' rank0 checkpoint reload')
            check(len(got['final_digests']) == 62 and got['final_digests'] == reference['final_digests'], arm+' rank0 all62 digests identical')
        check(len(got['losses']) == 3 and all(math.isfinite(v) for v in got['losses']) and got['n_params'] == got['n_changed'] == 62, arm+' finite3/all62'+str(rank))
        check(all(got[k] == reference[k] for k in ('losses','plan_sha256','plan_summary')), arm+' own numeric equality'+str(rank))
    cp = cpu_get(arm+'/checkpoint-identity.json')
    check((cp['sha256'],cp['bytes']) == (ref_cp['sha256'],ref_cp['bytes']), arm+' raw checkpoint identity')

# Execute unchanged guard's pure check function with independently selected negative outputs.
ns = {}
exec(compile(guard, '<pinned-idle-guard>', 'exec'), ns)
device = {'neuron_device':0,'instance_type':'trn2.3xlarge','nc_count':4,'logical_neuroncore_config':2,'neuroncore_ids':[0,1,2,3],'neuron_processes':[]}
for mode in ('idle','go_error','unknown_schema','busy'):
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        if command[0].endswith('/ps'):
            return SimpleNamespace(returncode=0,stderr='',stdout='11 10 python driver\n10 1 flock outer\n'+('12 1 torchrun torchrun\n' if mode=='busy' else ''))
        return SimpleNamespace(returncode=2 if mode=='go_error' else 0,stderr='fatal error' if mode=='go_error' else '',stdout=json.dumps([] if mode=='unknown_schema' else [device]))
    error=None
    try: ns['check_idle'](run,11,10,{})
    except Exception as exc: error=type(exc).__name__+':'+str(exc)
    check((error is None) == (mode=='idle'), 'own guard negative:' + mode)
    check(len(calls)==2, 'both idle checks run:'+mode)

# Execute exact driver budget predicate with malformed/failing receipts, assertions kept even under -O test run.
tree = ast.parse(driver)
gate = next(n for n in ast.walk(tree) if isinstance(n,ast.Assert) and isinstance(n.msg,ast.Constant) and n.msg.value=='TRAIN_BUDGET_FAILED')
for rc, within, allowed in ((0,True,True),(1,True,False),(0,False,False),(0,1,False),(0,None,False)):
    accepted=True
    try: exec(compile(ast.Module(body=[gate],type_ignores=[]),'<exact-budget>','exec',optimize=0),{'rehearsal':{'returncode':rc,'charged_through_process_exit_within_budget':within}})
    except AssertionError: accepted=False
    check(accepted == allowed, 'exact failclosed budget predicate:'+repr((rc,within)))
print(json.dumps({'verdict':'PASS_DISABLED_BINDING_MECHANICS_ONLY','checks':len(rows),'checks_detail':rows,'raw_cpu_artifacts':88,'cold_trn22_fifth':fifth,'earlier_schedule_arm_bpb':earlier['public_20m'],'later_genuine_control_bpb':later['public_20m'],'source_sha256':b['source_sha256'],'driver_sha256':b['driver_sha256'],'launch_authorized':False,'host_or_tensor_execution':False},indent=2,sort_keys=True))
