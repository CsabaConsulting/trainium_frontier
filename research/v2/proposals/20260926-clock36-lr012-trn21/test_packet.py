"""Implementer stdlib identity/refusal tests, never imports the effectful driver."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
B=json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())
P=json.loads((HERE/'EXPERIMENT_PROPOSAL.json').read_text())
DRIVER=(HERE/'host_driver.py').read_text()
checks=[]


def check(ok,name):
    if not ok:raise RuntimeError(name)
    checks.append(name)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)


def contract(b,q):
    if any(b[k] is not False for k in ('stage_authorized','launch_authorized','native_launch_authorized','source_changed','submission_authorized','host_reserved','operator_assigned')):raise ValueError('AUTHORITY')
    if b['source_sha256']!=q['source_sha256'] or b['source_commit']!=q['source_commit']:raise ValueError('SOURCE')
    expected=list(q['training_command']);expected[expected.index('--matrix-lr')+1]='0.012'
    if b['training_command']!=expected:raise ValueError('EXACT_LR_ONLY')
    if b['original_lock']!=q['original_lock'] or b['expected_hostname']!=q['expected_hostname']:raise ValueError('HOST_LOCK')
    for key in ('command','evaluation_command','training_environment','evaluation_environment','budgets','evaluation'):
        if b[key]!=json.loads(json.dumps(q[key]).replace(q['label'],b['label'])):raise ValueError('OPERATIONAL:'+key)
    if b['formal_control'] is not None or not b['selector_status'].startswith('INSUFFICIENT'):raise ValueError('FORMAL_CONTROL')


def main():
    parent=B['derivation']['parent_commit'];path='research/v2/proposals/20260926-startup-adjusted-full-trn21-replication/'
    q=json.loads(blob(parent,path+'NATIVE_BINDING.disabled.json'));old=blob(parent,path+'host_driver.py')
    check(sha(old)==q['driver_sha256']==B['derivation']['parent_driver_sha256'],'parent_driver_pin')
    expected=old.decode().replace(q['label'],B['label']).replace('Exact startup-clock cross-host replication then fresh public20M under one lifetime lock.','Exact clock36 matrix-LR arm then fresh public20M under one lifetime lock.')
    check(DRIVER==expected,'whole_driver_only_label_and_docstring')
    check(sha(DRIVER.encode())==B['driver_sha256'],'driver_pin')
    source=blob(B['source_commit'],B['source_git_path'])
    check(sha(source)==P['source_sha256']==q['source_sha256']==B['source_sha256']=='36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb','exact_unchanged_source')
    check(sha((HERE/'EXPERIMENT_PROPOSAL.json').read_bytes())==B['proposal_sha256'],'proposal_preserved')
    check((HERE/'EXPERIMENT_PROPOSAL.json').read_bytes()==blob(B['proposal_first_commit'],str((HERE/'EXPERIMENT_PROPOSAL.json').relative_to(REPO))),'proposal_first_commit')
    check(B['training_command'][4:]==P['argv'],'proposal_argv')
    changes=[(i,a,b) for i,(a,b) in enumerate(zip(q['training_command'],B['training_command'])) if a!=b]
    check(len(q['training_command'])==len(B['training_command']) and changes==[(q['training_command'].index('--matrix-lr')+1,'0.015','0.012')],'exactly_one_scientific_value_change')
    check('--recursive-schedule' not in B['training_command'] and B['training_command'].count('--startup-adjusted-schedule')==1,'no_recursive_stack')
    contract(B,q);check(True,'positive_binding_contract')
    for key in ('stage_authorized','launch_authorized','submission_authorized'):check(P[key] is False,'proposal_disabled_'+key)
    check(shlex.split((HERE/'package/launch-command.txt').read_text())==B['training_command'],'launch_command')
    prov=json.loads((HERE/'package/provenance.json').read_text())
    check(prov['train_sha256']==B['source_sha256'] and prov['source_commit']==B['source_commit'] and prov['launch_authorized'] is False,'provenance')
    check(shlex.split(prov['launch_command'])==B['training_command'],'provenance_argv')
    for name,pin in B['package_support_sha256'].items():check(sha((HERE/'package'/name).read_bytes())==pin,'package_pin_'+name)
    for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
        check(sha((REPO/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes())==B[key]==q[key],'shared_'+name)
    check(sha((REPO/'prepare.py').read_bytes())==B['prepare_sha256']==q['prepare_sha256'],'prepare_pin')
    check(sha(blob('00f2ca7f79696e65ce52458f074dd13b871da90c','research/scratch/submission_rehearsal.py'))==B['harness_sha256']==q['harness_sha256'],'harness_pin')
    constants={}
    for node in ast.parse(DRIVER).body:
        if isinstance(node,ast.Assign):
            value=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
            constants[node.targets[0].id]=ast.literal_eval(value)
    for var,key in [('ROOT','root'),('LABEL','label'),('SOURCE_SHA','source_sha256'),('PREP','prepare_path'),('HARNESS','harness_path'),('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment')]:check(constants[var]==B[key],'driver_binding_'+var)
    check(B['outer_command']==json.loads(json.dumps(q['outer_command']).replace(q['label'],B['label'])) and '-O' not in B['outer_command'],'normal_outer_original_lock')
    check(B['dispatch_command']==json.loads(json.dumps(q['dispatch_command']).replace(q['label'],B['label'])),'single_existing_dispatch')
    check(B['original_lock']['inode']==596823 and B['original_lock']['device']==66305 and B['expected_hostname']=='ip-172-31-34-85','original_trn21_identity')
    check(DRIVER.count('require_idle_under_inherited_lock(')==2 and 'pass_fds=(lock_fd,)' in DRIVER and "assert eval_fd==lock_fd,'LOCK_FD_CHANGED'" in DRIVER,'two_idle_checks_same_lifetime_fd')
    check("assert rehearsal['returncode']==0 and rehearsal['charged_through_process_exit_within_budget'] is True" in DRIVER and "assert (ROOT/'train/out/final.pt').is_file()" in DRIVER,'budget_checkpoint_before_eval')
    check(B['command'][4]=='2730s' and B['evaluation_command'][4]=='1800s' and B['command'][3]==B['evaluation_command'][3]=='--kill-after=60s','outer_watchdogs')
    check(B['budgets']['source_seconds']==1800 and B['budgets']['startup_cap_seconds']==900,'official_budget')
    check(B['evaluation']['tokens']==20971520 and B['evaluation']['seq_len']==1024 and B['evaluation']['batch_size']==1 and B['evaluation']['fresh_process'],'fresh20m_B1_context1024')
    cache_keys=['TORCH_NEURONX_NEFF_CACHE_DIR','TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','TORCH_NEURONX_HLO_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','NEURON_COMPILE_CACHE_DIR','TORCH_NEURONX_DEBUG_DIR','NEURON_COMPILE_CACHE_URL']
    for phase in ('training_environment','evaluation_environment'):
        check(all(B[phase][k]==q[phase][k] for k in cache_keys),'unchanged_cache_paths_'+phase)
        check(B[phase]['NEURON_COMPILE_CACHE_URL']==B[phase]['NEURON_COMPILE_CACHE_DIR'],'compiler_url_dir_'+phase)
    check(B['cold_startup_proven'] is False and B['cold_gate']==q['cold_gate'],'no_new_cold_claim')
    rows=[json.loads(line) for line in blob('3902c289','research/v2/ledger.jsonl').decode().splitlines()]
    row=next(r for r in rows if r['label']==P['control']['run_label'])
    check(row['kind']==P['control']['immutable_ledger_kind']=='architecture_startup_clock_arm' and row['base_sha256']==P['control']['immutable_ledger_base_sha256'],'parent_role_not_relabelled')
    check(row['public_20m']==P['control']['public_val_bpb']==.9882310747242088 and row['argv']==q['training_command'][4:],'parent_scientific_comparator')
    check(not any(r.get('kind')=='control' and r.get('base_sha256')==B['source_sha256'] for r in rows),'no_base36_control_at_proposal_snapshot')
    check(B['scientific_parent_comparator']==P['control'] and B['formal_control'] is None,'formal_insufficient_not_fabricated')
    # Reuse the actual inherited preflight function against mocked effects;
    # force optimize0 because the real bound host interpreter is normal Python.
    old_test=ast.parse(blob('7172b012c0e10e8f429c8b48755c1b03495a63c6','research/v2/proposals/20260926-architecture-recursive-full-trn21/test_binding.py'))
    method=next(n for n in ast.walk(old_test) if isinstance(n,ast.FunctionDef) and n.name=='test_real_preflight_negative_fixtures')
    ns={'ast':ast,'DRIVER':DRIVER,'B':B,'SimpleNamespace':SimpleNamespace}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'pinned-real-preflight-fixture','exec'),ns)
    ns['test_real_preflight_negative_fixtures'](unittest.TestCase())
    checks.extend(['mock_preflight_positive','refuse_STOP','refuse_disk','refuse_source','refuse_prepare','refuse_harness','refuse_missing_cache'])
    mutations={'authority':lambda b:b.update(launch_authorized=True),'source':lambda b:b.update(source_sha256='0'*64),
               'lr':lambda b:b['training_command'].__setitem__(b['training_command'].index('--matrix-lr')+1,'0.013'),
               'extra_flag':lambda b:b['training_command'].append('--recursive-schedule'),'lock':lambda b:b['original_lock'].update(inode=1),
               'cache':lambda b:b['training_environment'].update(NEURON_COMPILE_CACHE_URL='/changed'),
               'eval':lambda b:b['evaluation'].update(tokens=2097152),'control':lambda b:b.update(formal_control=P['control'])}
    for name,mutate in mutations.items():
        bad=copy.deepcopy(B);mutate(bad)
        try:contract(bad,q)
        except ValueError:check(True,'negative_'+name)
        else:raise RuntimeError('NEGATIVE_ACCEPTED:'+name)
    print(json.dumps({'status':'IMPLEMENTER_STDLIB_PREPARATION_ONLY','checks':len(checks),'names':checks,'source_changed':False,'host_or_tensor_actions':False,'launch_authorized':False,'formal_selection':'INSUFFICIENT_UNTIL_GENUINE_BASE36_CONTROL'},indent=2))


if __name__=='__main__':main()
