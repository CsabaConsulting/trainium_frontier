"""Operator mechanical checks only; no host calls or launch authority."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
SOURCE = '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'
TEMPLATES = {
    'trn21': ('20260926-trn21-arch3e79-replicate', 'b4647403e8338cea1bc9d16b7077315d9f3705e29c4bb364a869ee30f398fe6c'),
    'trn22': ('20260926-trn22-arch3e79-full', 'c8957e5dbd67bbf017510d6110bcd0e0ab09d2183db098d18654a54f5498e7e0'),
}
count = 0

def check(value, message):
    global count
    if not value:
        raise RuntimeError(message)
    count += 1

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

reports = {}
for host, (folder, old_driver_sha) in TEMPLATES.items():
    packet = ROOT / host
    old = ROOT.parent / folder
    b = json.loads((packet/'NATIVE_BINDING.disabled.json').read_text())
    q = json.loads((packet/'EXPERIMENT_PROPOSAL.json').read_text())
    reference = json.loads((old/'NATIVE_BINDING.disabled.json').read_text())
    old_label = reference['label']
    label = 'fs-20260926-arch3e79-base-control-'+host+'-r1-full'
    check(b['label'] == q['label'] == label, host+': label')
    check(b['host'] == q['host'] == host, host+': host')
    check(b['stage_authorized'] is False and b['launch_authorized'] is False and q['launch_authorized'] is False, host+': disabled')
    check(b['kind'] == q['kind'] == 'control', host+': control')
    check(b['base_sha256'] == q['base_sha256'] == SOURCE, host+': base')
    check(b['source_sha256'] == q['source_sha256'] == SOURCE, host+': source')
    check(b['result_row_identity'] == q['result_row_identity'] == {'label':label,'host':host,'kind':'control','base_sha256':SOURCE,'source_sha256':SOURCE}, host+': row identity')
    check(q['primary_comparator'] is None and q['secondary_comparator'] is None, host+': not relabelled comparator')
    check('same_host_control' not in b and 'cross_host_original' not in b, host+': no old control alias')
    check(b['historical_context']['original_architecture']['kind'] == 'architecture_arm', host+': old row unchanged')
    check(b['only_science_delta'] == q['only_science_delta'] == {}, host+': no science delta')
    check(digest(old/'host_driver.py') == old_driver_sha, host+': authentic template driver')
    driver = (packet/'host_driver.py').read_text()
    check(driver == (old/'host_driver.py').read_text().replace(old_label,label), host+': only label paths in driver')
    check(digest(packet/'host_driver.py') == b['driver_sha256'] == q['driver_sha256'], host+': driver pin')
    check(b['training_command'] == q['argv'] == reference['training_command'], host+': exact science argv')
    check('--recursive-schedule' not in q['argv'], host+': no Recursive')
    for flag, value in [('--matrix-lr','0.015'),('--seed','42'),('--max-train-seconds','1800'),('--num-steps','1000000'),('--seq-len','1024'),('--attention-layers','0,2,4,6,7')]:
        check(q['argv'][q['argv'].index(flag)+1] == value, host+': '+flag)
    check('--nproc_per_node=4' in q['argv'] and '--initial-cache-refresh' in q['argv'], host+': ranks and priming')
    for name in ('root','package','sandbox','eval_sandbox','driver_path','command','evaluation_command','outer_command','training_environment','evaluation_environment'):
        check(json.dumps(b[name],sort_keys=True) == json.dumps(reference[name],sort_keys=True).replace(old_label,label), host+': binding '+name)
    for name in ('source_commit','source_git_path','source_sha256','prepare_path','prepare_sha256','harness_path','harness_sha256','idle_guard_sha256','dispatch_driver_sha256','device_lock','budgets','evaluation','cold_gate'):
        check(b[name] == reference[name], host+': preserved '+name)
    values = {}
    for node in ast.parse(driver).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            value = node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='Path':
                value = value.args[0]
            values[node.targets[0].id] = ast.literal_eval(value)
    for key,name in [('ROOT','root'),('PREP','prepare_path'),('HARNESS','harness_path'),('LABEL','label'),('SOURCE_SHA','source_sha256'),('PREP_SHA','prepare_sha256'),('HARNESS_SHA','harness_sha256'),('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment')]:
        check(values[key] == b[name], host+': driver constant '+key)
    source = subprocess.check_output(['git','show',b['source_commit']+':'+b['source_git_path']],cwd=REPO)
    check(hashlib.sha256(source).hexdigest() == SOURCE, host+': source git blob')
    guard = ROOT.parent/'20260926-common-host-idle-guard'
    check(digest(guard/'idle_guard.py') == b['idle_guard_sha256'], host+': guard')
    check(digest(guard/'dispatch_driver.py') == b['dispatch_driver_sha256'], host+': dispatch')
    check(b['no_auto_next_job'] is True and b['old_queue']=='PAUSED_AND_EXITED; no re-arm',host+': no second launcher')
    reports[host] = {name:digest(packet/name) for name in ('NATIVE_BINDING.disabled.json','EXPERIMENT_PROPOSAL.json','host_driver.py')}
print(json.dumps({'status':'OPERATOR_MECHANICAL_PASS','checks':count,'not_independent_review':True,'no_host_actions':True,'launch_authorized':False,'packets':reports},sort_keys=True))
