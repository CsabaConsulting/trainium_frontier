#!/usr/bin/env python3
"""Offline final authentication and query-only unchanged selector. No ledger writes."""
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import runpy
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
ROOT = 'research/v2/operator/20260926-trn22-arch3e79-full'
REF = 'e5d28dc9'
IMPORT = '7132a648'
BASE = 'c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586'
ARM = 'L2-unet8x1024-3e79-primed'
SELECTOR_SHA = 'fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac'
checks = []


def check(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(path, ref=REF):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)


def raw(name):
    return git(ROOT + '/completed-eval/' + name)


def record(name):
    return json.loads(raw(name))


def tagged(log, prefix):
    found = [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    check(len(found) == 1, 'unique ' + prefix)
    return found[0]


def read_eval(log):
    for match in re.finditer(r'\{', log):
        try:
            value, _ = json.JSONDecoder().raw_decode(log[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and 'total_nats' in value and 'val_bpb' in value:
            return value
    raise RuntimeError('missing evaluation')


def utc(log, prefix):
    return datetime.datetime.fromisoformat(re.search('^' + prefix + '=(.+)$', log, re.M).group(1))


# Independently re-execute the complete prior training/binding/accounting audit.
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    runpy.run_path(str(HERE / 'audit_training.py'), run_name='__main__')
training = json.loads(captured.getvalue())
check(training['status'] == 'PASS_TRAINING_AUTHENTICATION_ONLY' and training['checks'] == 65,
      '65 training/binding/guard/source/actual-budget checks rerun')
binding = json.loads(git(ROOT + '/NATIVE_BINDING.disabled.json', '83db5fe3'))
manifest = record('retrieval-manifest.json')
file_hashes = {}
for name, expected in manifest['metadata']['files'].items():
    data = raw(name)
    check(len(data) == expected['bytes'] and sha(data) == expected['sha256'], 'final artifact ' + name)
    check(data == git(ROOT + '/completed-eval/' + name, IMPORT), 'canonical byte identity ' + name)
    file_hashes[name] = sha(data)
check(len(file_hashes) == len(manifest['retrieved']) == 10, 'all ten retrieved artifacts')
check(all(row['hash_verified'] is True and row['returncode'] == 0 and row['sha256'] == file_hashes[row['path']]
          for row in manifest['retrieved']), 'transport success agrees with own hashes')
check(manifest['metadata']['root'] == binding['root'] and manifest['metadata']['no_checkpoint_copy'] is True,
      'exact remote scope; checkpoint not downloaded')
for name in ('checkpoint.json', 'eval-under-lock-idle.json', 'fresh-public20m-launch.json'):
    check(raw(name) == git(ROOT + '/completed-training/' + name, '8d9ee4b0'), 'unchanged handoff ' + name)
terminal = record('terminal.json')
outer = record('outer-flock-exit.json')
check(terminal['label'] == binding['label'] and terminal['status'] == 'COMPLETE_PENDING_INDEPENDENT_REVIEW'
      and terminal['eval_exit_code'] == outer['returncode'] == 0 and raw('eval-exit-code.txt').strip() == b'0', 'zero exits and complete terminal')
check(terminal['no_automatic_next_job'] is True and terminal['submission_authorized'] is False, 'no automatic next/adoption/submission')
check(outer['command'][:3] == ['/usr/bin/flock', '-n', binding['device_lock']]
      and outer['command'][-1] == binding['root'] + '/host_driver.py', 'same bound outer lifetime flock command')
log = raw('fresh-public20m.log').decode()
isolated = tagged(log, 'ISOLATED_SUBMISSION ')
command = tagged(log, 'REHEARSAL_COMMAND ')
eval_rehearsal = record('eval/rehearsal.json')
check(tagged(log, 'SUBMISSION_REHEARSAL ') == eval_rehearsal and eval_rehearsal['returncode'] == 0,
      'eval process observer receipt matches log')
check(isolated['mode'] == 'eval' and isolated['sandbox'] == binding['eval_sandbox']
      and isolated['files'] == ['prepare.py', 'train.py']
      and isolated['train_sha256'] == binding['source_sha256']
      and isolated['prepare_sha256'] == binding['prepare_sha256'], 'fresh exact-source minimal eval sandbox')
check(shlex.split(isolated['launch_command']) == binding['training_command'], 'eval source training-config command unchanged')
expected_eval = [binding['evaluation_command'][5], 'prepare.py', 'eval-public', '--train-py', 'train.py',
                 '--checkpoint', binding['sandbox'] + '/out/final.pt', '--device', 'neuron',
                 '--batch-size', '1', '--seq-len', '1024', '--eval-tokens', '20971520', '--timeout-seconds', '1200']
check(command == expected_eval, 'effective fresh20M context1024 exact argv')
evaluation = read_eval(log)
check(evaluation['num_tokens'] == 20971520 and evaluation['num_bytes'] == 83462186
      and evaluation['total_nats'] == 57061344.0, 'complete fresh20M raw totals')
bpb = evaluation['total_nats'] / (evaluation['num_bytes'] * math.log(2))
check(math.isfinite(bpb) and bpb == evaluation['val_bpb'] == 0.9863403052402583, 'independent BPB arithmetic')
check(evaluation['eval_over_timeout'] is False and evaluation['eval_seconds'] < evaluation['eval_timeout'] == 1200,
      'full evaluation within unchanged timeout')
parse = datetime.datetime.fromisoformat
start, end = utc(log, 'FRESH-PUBLIC20M_START_UTC'), utc(log, 'FRESH-PUBLIC20M_END_UTC')
check(parse(training['training_end_utc']) < parse(training['fresh20M_launch_utc']) <= start < end
      <= parse(terminal['utc']) < parse(outer['ended_utc']), 'train/fresh-eval/terminal/outer chronology')
check(evaluation['eval_seconds'] <= eval_rehearsal['wall_seconds'] <= (end - start).total_seconds(), 'evaluator versus whole-process time distinguished')
quiescence = record('terminal-quiescence.json')
state = quiescence['data']
check(quiescence['rc'] == state['ps_rc'] == state['neuron_ls_rc'] == state['lock_competitor_rc'] == 0
      and not state['process_details'] and not state['compiler_capture'] and not state['job_lines']
      and not state['old_queue_command_lines'] and not any(item['exists'] for item in state['old_queue_pid_checks']),
      'post-exit process/compiler/retired-queue absence and free lock')
devices = json.loads(state['neuron_ls'])
check(devices and all(not device['neuron_processes'] for device in devices), 'post-exit no Neuron holder')
check([state['lock_device'], state['lock_inode']] == training['lock_device_inode'] == [66305, 832589], 'original common lock inode unchanged')
check(parse(outer['ended_utc']) < parse(state['utc']), 'quiescence after full outer exit')
check(state['logs']['fresh-public20m.log']['sha256'] == sha(raw('fresh-public20m.log')), 'second observation agrees on raw eval log hash')
checkpoint = record('checkpoint.json')
check(checkpoint == training['checkpoint'], 'exact trained checkpoint receipt carried into eval archive')

selector_path, ledger_path = REPO / 'workflow/tools/decide.py', REPO / 'research/v2/ledger.jsonl'
selector_bytes, ledger_bytes = selector_path.read_bytes(), ledger_path.read_bytes()
check(sha(selector_bytes) == SELECTOR_SHA, 'selector bytes unchanged')
spec = importlib.util.spec_from_file_location('unchanged_decide', selector_path)
decide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decide)
rows = decide.load(str(ledger_path))
check(not any(row['label'] == binding['label'] for row in rows), 'candidate is query-only, not already in ledger')
controls = decide.controls_for(rows, 'trn22', BASE)
check(len(controls) == 1 and controls[0]['label'] == binding['same_host_control']['label'], 'exact matched c5c/trn22 control, no Recursive substitution')
control = controls[0]
control_raw = (REPO / control['eval_log']).read_bytes()
check(sha(control_raw) == control['eval_log_sha256'], 'matched control raw eval authentication')
control_eval = read_eval(control_raw.decode())
check(control_eval['total_nats'] / (control_eval['num_bytes'] * math.log(2)) == control['public_20m'] == .991513958330512,
      'matched control raw BPB arithmetic')
check(control['steps'] == 3633 and training['steps'] == 3629, 'matched steps reported separately')
candidate = dict(label=binding['label'], arm_id=ARM, kind='architecture_arm', host='trn22',
                 timestamp_utc=terminal['utc'], base_sha256=BASE, train_py_sha256=binding['source_sha256'],
                 source_commit=binding['source_commit'], status='evaluated', git_dirty=False,
                 git_dirty_scope='Exact frozen committed science bytes; not a global host-tree cleanliness claim',
                 argv=binding['training_command'][4:], env=binding['training_environment'],
                 training_budget_seconds=1800, over_budget=False, steps=training['steps'],
                 training_seconds=training['source_estimates_separate']['training_seconds'],
                 charged_seconds_est=training['source_estimates_separate']['charged_seconds_est'],
                 charged_through_exit_seconds=training['charged_through_exit_seconds'],
                 charged_through_exit_evidence_scope='Pinned official observer monotonic process wall minus min(uncapped fifth boundary,900); includes through process exit',
                 process_through_exit_wall_seconds=training['process_wall_seconds'],
                 startup_measured_seconds=training['uncapped_fifth_seconds'],
                 startup_allowance_seconds=training['startup_allowance_seconds'],
                 startup_allowance_est=training['source_estimates_separate']['startup_allowance_est'],
                 cold_startup_proven=False,
                 cold_scope='This full run reused exact-source prior cold-five cache roots. Separate earlier cold-five evidence is not reclassified as a cold full replay.',
                 causality_passed=None, eval_tokens=evaluation['num_tokens'], eval_seq_len=1024,
                 public_20m=bpb, eval_seconds=evaluation['eval_seconds'], eval_over_timeout=False,
                 checkpoint_sha256=checkpoint['sha256'], checkpoint_bytes=checkpoint['bytes'],
                 checkpoint_path=binding['sandbox'] + '/out/final.pt',
                 checkpoint_evidence_scope='Retained on host; same pre-eval receipt authenticated after eval, not a new post-eval checkpoint-byte rehash or download',
                 eval_log=ROOT + '/completed-eval/fresh-public20m.log', eval_log_sha256=sha(raw('fresh-public20m.log')),
                 training_log=ROOT + '/completed-training/training.log',
                 training_log_sha256=training['retrieved_artifact_sha256']['training.log'],
                 training_archive_commit=training['raw_commit'], final_archive_commit=subprocess.check_output(['git','rev-parse',REF],cwd=REPO,text=True).strip(),
                 run_purpose='quality', submission_authorized=False)
check(decide.invalid_reason(candidate) is None, 'selector-valid research result')
verdict = decide.decide_arm(rows + [candidate], ARM)
delta = bpb - control['public_20m']
check(verdict['verdict'] == 'NOMINATE' and verdict['per_host'][0]['delta'] == round(delta, 7)
      and verdict['per_host'][0]['controls'] == [control['label']], 'unchanged selector NOMINATE on matched control')
check(delta <= -.005 and delta > -.010 and bpb >= .975, 'separate human provisional threshold met, no big-win hold')
check(selector_path.read_bytes() == selector_bytes and ledger_path.read_bytes() == ledger_bytes, 'no selector or ledger mutation')
result = dict(status='PASS_RESULT_AUTHENTICATION', own_training_checks=training['checks'], final_checks=len(checks),
              checks=checks, candidate_query_only=candidate, source_sha256=binding['source_sha256'],
              control_label=control['label'], control_bpb=control['public_20m'], delta_unrounded=delta,
              control_steps=control['steps'], candidate_steps=training['steps'],
              selector_sha256=SELECTOR_SHA, ledger_sha256=sha(ledger_bytes), historical_rows=len(rows),
              selector_result=verdict, human_provisional_delta_threshold=-.005, human_threshold_met=delta <= -.005,
              independent_adoption=False, cross_host_confirmation='PENDING', full_run_cold=False,
              native_causality_observed=None, checkpoint_post_eval_bytes_rehashed=False,
              evaluation=evaluation, evaluation_process_wall_seconds=eval_rehearsal['wall_seconds'],
              train_accounting=training, final_artifact_sha256=file_hashes)
(HERE / 'FINAL_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
(HERE / 'candidate-ledger-row.query-only.json').write_text(json.dumps(candidate, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], training_checks=training['checks'], final_checks=len(checks),
                     candidate_bpb=bpb, control_bpb=control['public_20m'], delta=delta,
                     selector_verdict=verdict['verdict'], human_threshold_met=delta <= -.005,
                     ledger_sha256=sha(ledger_bytes)), indent=2))
