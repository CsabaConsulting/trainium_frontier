#!/usr/bin/env python3
"""Author stdlib checks only: immutable pins, replication argv and mocked refusals."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
B = json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())
P = json.loads((HERE/'EXPERIMENT_PROPOSAL.json').read_text())
checks = []


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)
    checks.append(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(ref, path):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)


parent = B['derivation']['parent_commit']
parent_path = 'research/v2/proposals/20260926-architecture-recursive-full-trn21/'
Q = json.loads(blob(parent, parent_path+'NATIVE_BINDING.disabled.json'))
DRIVER = (HERE/'host_driver.py').read_text()
old = blob(parent, parent_path+'host_driver.py')
check(sha(old) == B['derivation']['parent_driver_sha256'] == Q['driver_sha256'], 'parent_driver_authenticated')
expected = old.decode().replace(Q['label'], B['label']).replace(Q['source_sha256'], B['source_sha256']).replace(
    'Exact architecture Recursive stack then fresh public20M under one lifetime lock.',
    'Exact startup-clock cross-host replication then fresh public20M under one lifetime lock.')
check(expected == DRIVER, 'entire_driver_identity_after_three_literal_changes')
check(sha(DRIVER.encode()) == B['driver_sha256'], 'driver_pin')
for obj in (B,P):
    check(all(obj[k] is False for k in ('stage_authorized','launch_authorized','source_adoption','submission_authorized','host_reserved','operator_assigned')), 'all_permissions_disabled')
check(B['host'] == P['host'] == 'trn21' and B['expected_hostname'] == 'ip-172-31-34-85', 'host')
check(B['label'] == P['label'] == 'fs-20260926-arch36a8-startup-clock-trn21-r2-full', 'exact_label')
check(B['original_lock']['inode'] == 596823 and B['original_lock']['device'] == 66305, 'original_lock')
check(B['source_commit'] == P['source_commit'] == '7419927c7417b5666a984ec9c2b849feadabaaa9', 'source_commit')
source=blob(B['source_commit'],B['source_git_path'])
check(sha(source) == B['source_sha256'] == P['source_sha256'] == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb', 'unchanged_source_bytes')
check(sha((REPO/'prepare.py').read_bytes()) == B['prepare_sha256'] == Q['prepare_sha256'], 'prepare_pin')
check(B['harness_sha256'] == Q['harness_sha256'] == '7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b', 'harness_pin')
for n,k in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
    check(sha((REPO/'research/v2/operator/20260926-common-host-idle-guard'/n).read_bytes()) == B[k] == Q[k], 'shared_'+n)
trn22 = json.loads((REPO/'research/v2/proposals/20260926-startup-adjusted-full-trn22/EXPERIMENT_PROPOSAL.json').read_text())
check(B['training_command'] == P['argv'] == trn22['argv'], 'exact_completed_trn22_scientific_argv')
check(B['training_command'] == [('--startup-adjusted-schedule' if x == '--recursive-schedule' else x) for x in Q['training_command']], 'only_flag_contrast_vs75b9')
check('--recursive-schedule' not in B['training_command'] and B['training_command'].count('--startup-adjusted-schedule') == 1, 'no_recursive')
command=(HERE/'package/launch-command.txt').read_text()
check(shlex.split(command) == B['training_command'], 'package_command')
check(command == (REPO/'research/v2/proposals/20260926-startup-adjusted-full-trn22/package/launch-command.txt').read_text(), 'command_bytes_equal_trn22')
prov=json.loads((HERE/'package/provenance.json').read_text())
check(prov['source_commit'] == B['source_commit'] and prov['train_sha256'] == B['source_sha256'], 'provenance_source')
check(shlex.split(prov['launch_command']) == B['training_command'] and prov['launch_authorized'] is False, 'provenance_command_disabled')
for n,s in B['package_support_sha256'].items():check(sha((HERE/'package'/n).read_bytes()) == s, 'package_pin_'+n)
for key in ('command','evaluation_command','training_environment','evaluation_environment','outer_command','prepare_path','harness_path','device_lock','budgets','evaluation'):
    expected=json.loads(json.dumps(Q[key]).replace(Q['label'],B['label']))
    check(B[key] == expected, 'unchanged_operational_'+key)
constants={}
for node in ast.parse(DRIVER).body:
    if isinstance(node,ast.Assign):
        val=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
        constants[node.targets[0].id]=ast.literal_eval(val)
for var,key in [('ROOT','root'),('LABEL','label'),('SOURCE_SHA','source_sha256'),('PREP','prepare_path'),('HARNESS','harness_path'),('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment')]:
    check(constants[var] == B[key], 'driver_constant_'+var)
check('-O' not in B['outer_command'] and '-O' not in B['dispatch_command'], 'normal_driver_interpreter')
check(B['command'][4] == '2730s' and B['evaluation_command'][4] == '1800s' and B['command'][3] == B['evaluation_command'][3] == '--kill-after=60s', 'watchdogs')
check(B['budgets']['source_seconds'] == 1800 and B['budgets']['startup_cap_seconds'] == 900, 'official_budget')
check(B['evaluation']['fresh_process'] and B['evaluation']['tokens'] == 20971520 and B['evaluation']['batch_size'] == 1 and B['evaluation']['seq_len'] == 1024, 'fresh20m_B1_context1024')
check(DRIVER.count('require_idle_under_inherited_lock(') == 2 and DRIVER.count('pass_fds=(lock_fd,)') == 1 and "assert eval_fd==lock_fd,'LOCK_FD_CHANGED'" in DRIVER, 'same_lock_two_fresh_guards')
check("assert rehearsal['returncode']==0 and rehearsal['charged_through_process_exit_within_budget'] is True" in DRIVER and "assert (ROOT/'train/out/final.pt').is_file()" in DRIVER, 'budget_checkpoint_before_eval')

# Authenticate prior accepted cold evidence; do not relabel it trn21/full cold.
gate=B['cold_gate']; bundle_raw=(REPO/gate['archive']).read_bytes()
check(sha(bundle_raw) == gate['raw_bundle_sha256'], 'cold_bundle_pin')
bundle=json.loads(bundle_raw)
for name,record in bundle['files'].items():
    raw=record['text'].encode();check(sha(raw) == record['sha256'] and len(raw) == record['bytes'], 'cold_raw_'+name)
report=json.loads(bundle['files'][gate['receipt_key']]['text'])
check(sha(bundle['files'][gate['receipt_key']]['text'].encode()) == gate['receipt_sha256'], 'cold_receipt_pin')
check(report['returncode'] == 0 and report['step_boundaries'] == 5 and report['observations'][-1]['observed_seconds'] == gate['observed_uncapped_fifth_seconds'] == 845.6273377500474, 'cold_actual_fifth')
check(gate['host'] == 'trn22' and B['cold_startup_proven'] is False and len(report['fresh_empty_sdk_caches']) == 6, 'not_trn21_cold')
check(report['charged_through_process_exit_within_budget'] and gate['charged_through_exit_seconds'] == 4.469341197051108, 'cold_official_accounting')
cpu=B['source_gate']
check(cpu['source_sha256'] == B['source_sha256'] and sha((REPO/cpu['full_parameter_cpu_raw']).read_bytes()) == cpu['full_parameter_cpu_raw_sha256'], 'prior_accepted_fullCPU_pin_only')
rows={json.loads(line)['label']:(json.loads(line),sha((line+'\n').encode())) for line in blob('1c879c2b','research/v2/ledger.jsonl').decode().splitlines()}
for key in ['primary_formal_control','replication_trigger']:
    comp=P[key];row,pin=rows[comp['label']]
    check(pin == comp['raw_row_sha256'] and row['public_20m'] == comp['public20m_bpb'] and row['train_py_sha256'] == comp['source_sha256'], 'exact_historical_'+key)
check(rows[P['primary_formal_control']['label']][0]['kind'] == 'control' and P['primary_formal_control']['public20m_bpb'] == .9899528600562257, 'genuine_control_not_relabelled')
check(rows[P['additional_comparators'][0]['label']][0]['kind'] == 'schedule_arm' and P['additional_comparators'][0]['public20m_bpb'] == .9900560897366009, 'earlier_schedule_arm_separate')
check(P['replication_trigger']['public20m_bpb'] == .9883821509811542, 'authenticated_trigger_score')

# Pinned predecessor's preflight-effect fixture, executed with optimize=0 to
# match the mandatory normal host command, even when this test uses -O.
old_test=ast.parse(blob(parent,parent_path+'test_binding.py'))
method=next(n for n in ast.walk(old_test) if isinstance(n,ast.FunctionDef) and n.name=='test_real_preflight_negative_fixtures')
exec(compile(ast.Module(body=[method],type_ignores=[]),'<pinned-preflight-fixture>','exec'))
test_real_preflight_negative_fixtures(unittest.TestCase())
checks.extend(['mock_preflight_success','refuse_STOP','refuse_disk','refuse_source_pin','refuse_prepare_pin','refuse_harness_pin','refuse_missing_cache'])
print(json.dumps(dict(status='IMPLEMENTER_STDLIB_PREPARATION_PASS_NOT_INDEPENDENT_REVIEW',count=len(checks),checks=checks,
    effects='Local immutable Git/files plus fake preflight only; no hosts/tensors/driver execution',
    native_launch_authorized=False,source_changed=False,cold_trn21_proven=False),indent=2))
