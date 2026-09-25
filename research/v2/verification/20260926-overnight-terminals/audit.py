"""Supervisor raw-evidence authentication; stdlib only, no host/source mutation."""
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path('/home/poojith-devan/Desktop/Trainium')
names = []
def check(ok, name):
    if not ok:
        raise RuntimeError(name)
    names.append(name)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def read(path):
    return json.loads((ROOT / path).read_text())
def git(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)
def tagged(log, prefix):
    hits = [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    check(len(hits) == 1, 'one tagged record ' + prefix)
    return hits[0]
def bundle(path, ref):
    raw = (ROOT / path).read_bytes()
    check(raw == git(ref, path), 'immutable archive ' + path)
    value = json.loads(raw)
    for name, item in value['files'].items():
        data = item['text'].encode()
        check(len(data) == item['bytes'] and sha(data) == item['sha256'], 'embedded bytes ' + name)
    return value, lambda name: json.loads(value['files'][name]['text'])
def guard(value, inode):
    check(value['lock']['verified_existing_exclusive_lock'] and value['lock']['inode'] == inode
          and value['lock']['path'] == '/home/ubuntu/.cache/trainium_freshstart/device0.lock', 'original lifetime lock')
    check(value['ps']['returncode'] == value['neuron_ls']['returncode'] == 0
          and not value['ps']['busy'] and not value['neuron_ls']['process_pids'], 'fresh ps and neuron-ls idle')

# Cold five: the observer's uncapped fifth boundary, not a capped estimate.
coldpath = 'research/v2/operator/20260926-trn21-architecture-recursive-cold-five'
cold, cf = bundle(coldpath + '/completed/results-bundle.json', '00179410')
binding = read(coldpath + '/CT_BINDING.disabled.json')
check(sha((ROOT / coldpath / 'CT_BINDING.disabled.json').read_bytes()) == '9b1b0c0185167d670157c4abdb0b5cb7b4af33f21ecdc7af854d668dfae06185', 'cold binding pin')
pre, launch, receipt = cf('preflight.json'), cf('training-launch.json'), cf('train/rehearsal.json')
check(pre['source_sha256'] == launch['source_sha256'] == '75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f', 'cold source pin')
guard(cf('under-lock-idle.json'), 596823)
log = cold['files']['training.log']['text']
isolation, command = tagged(log, 'ISOLATED_SUBMISSION '), tagged(log, 'REHEARSAL_COMMAND ')
check(isolation['train_sha256'] == launch['source_sha256'] and isolation['prepare_sha256'] == pre['prepare_sha256'], 'cold effective source/prepare')
check(command == shlex.split(isolation['launch_command']) and '--recursive-schedule' in command
      and command[command.index('--num-steps') + 1] == '5' and command[command.index('--max-train-seconds') + 1] == '1800'
      and '--nproc_per_node=4' in command and command[command.index('--seed') + 1] == '42', 'cold exact scientific shape/budget')
check(len(receipt['fresh_empty_sdk_caches']) == 6 and receipt['fresh_empty_sdk_caches'] == pre['cold_cache_dirs_absent_before_stage']
      and isolation['fresh_empty_sdk_caches'] == receipt['fresh_empty_sdk_caches'] and '--fresh-caches' in launch['command'], 'genuine six fresh caches')
check(receipt['step_boundaries'] == 5 and [x['step'] for x in receipt['observations']] == list(range(5))
      and list(map(int, cold['files']['train/steps.jsonl']['text'].splitlines())) == list(range(5)), 'five complete observed steps')
times = [x['observed_seconds'] for x in receipt['observations']]
check(times == sorted(times) and all(math.isfinite(x) for x in times), 'finite monotonic actual boundaries')
fifth = times[4]
check(fifth == receipt['startup_allowance_observed'] == 831.4215557309799 and fifth <= 870, 'actual cold fifth under target and cap')
check(receipt['wall_seconds'] - min(fifth, 900) == receipt['charged_through_process_exit'] < 1800, 'unchanged cold budget arithmetic')
check(receipt['returncode'] == cf('terminal.json')['returncode'] == cf('outer-flock-exit.json')['returncode'] == 0, 'cold complete exits')
check(receipt['checkpoint_bytes'] == 453048658 and receipt['checkpoint_sha256'] == 'dc56b0b18487d644ddcdb2add32bd234e3619a57aa23f0b4b0fd67640e07f6be', 'cold checkpoint exists and pinned')
check(cold['lock_inode'] == 596823 and cold['lock_competitor_rc'] == cold['neuron_ls_rc'] == 0 and not cold['job_lines'] and not cold['workers'], 'cold terminal quiescence')
check(cf('terminal.json')['no_automatic_full'] and cf('terminal.json')['no_eval'], 'cold only not BPB or full release')

# LR quality: authenticate both closed transports and recompute BPB ourselves.
lrpath = 'research/v2/operator/20260926-trn22-arch3e79-lr012'
b = read(lrpath + '/NATIVE_BINDING.disabled.json')
for phase, ref in [('completed-training', '3bfcf886'), ('completed-eval', 'd4c4d6f8')]:
    retrieval = read(lrpath + '/' + phase + '/retrieval.json')
    for name, entry in retrieval['metadata']['files'].items():
        data = (ROOT / lrpath / phase / name).read_bytes()
        check(data == git(ref, lrpath + '/' + phase + '/' + name) and sha(data) == entry['sha256']
              and len(data) == entry['bytes'], 'LR closed archive ' + phase + '/' + name)
tr = read(lrpath + '/completed-training/train/rehearsal.json')
check(tr['step_boundaries'] == 3629 and [x['step'] for x in tr['observations']] == list(range(3629)), 'LR complete steps')
charge = tr['wall_seconds'] - min(tr['observations'][4]['observed_seconds'], 900)
check(charge == tr['charged_through_process_exit'] < 1800 and tr['charged_through_process_exit_within_budget'] and tr['returncode'] == 0, 'LR actual official accounting')
tlog = (ROOT / lrpath / 'completed-training/training.log').read_text()
science = tagged(tlog, 'REHEARSAL_COMMAND ')
check(science == b['training_command'] and re.search(r'^over_budget:\s*False$', tlog, re.M), 'LR exact science and no overbudget')
rows = [json.loads(line) for line in (ROOT / 'research/v2/ledger.jsonl').read_text().splitlines() if line.strip()]
parent = next(x for x in rows if x['label'] == b['same_host_control']['label'])
parent_argv = parent['argv']
check(science[4:] == ['0.012' if i and parent_argv[i-1] == '--matrix-lr' else x for i,x in enumerate(parent_argv)], 'only LR changed against recorded parent argv')
elog = (ROOT / lrpath / 'completed-eval/fresh-public20m.log').read_text()
iso = tagged(elog, 'ISOLATED_SUBMISSION ')
check(iso['train_sha256'] == b['source_sha256'] and iso['prepare_sha256'] == b['prepare_sha256'], 'LR eval same source and protected prepare')
evalcmd = tagged(elog, 'REHEARSAL_COMMAND ')
check(evalcmd[1:3] == ['prepare.py', 'eval-public'] and evalcmd[evalcmd.index('--seq-len')+1] == '1024'
      and evalcmd[evalcmd.index('--eval-tokens')+1] == '20971520' and evalcmd[evalcmd.index('--batch-size')+1] == '1', 'fresh public20M context1024 batch1')
evaluation = None
for match in re.finditer(r'\{', elog):
    try:
        v, _ = json.JSONDecoder().raw_decode(elog[match.start():])
    except ValueError:
        continue
    if isinstance(v, dict) and 'total_nats' in v and 'val_bpb' in v:
        evaluation = v
        break
check(evaluation is not None, 'actual eval totals present')
bpb = evaluation['total_nats'] / (evaluation['num_bytes'] * math.log(2))
check(evaluation['total_nats'] == 57178204 and evaluation['num_bytes'] == 83462186 and evaluation['num_tokens'] == 20971520
      and bpb == evaluation['val_bpb'] == .9883603019664198 and evaluation['eval_over_timeout'] is False, 'own LR raw BPB and complete eval')
post = read(lrpath + '/completed-eval/post-eval-checkpoint.json')
check(post['post_eval_rehash'] and post['sha256'] == tr['checkpoint_sha256'] and post['bytes'] == tr['checkpoint_bytes'] == 453048594, 'LR retained same checkpoint after eval')
check(read(lrpath + '/completed-eval/terminal.json')['eval_exit_code'] == read(lrpath + '/completed-eval/outer-flock-exit.json')['returncode'] == 0, 'LR terminal exits')
guard(read(lrpath + '/completed-eval/eval-under-lock-idle.json'), 832589)
check(parent['kind'] != 'control' and parent['public_20m'] == .9863403052402583 and bpb > parent['public_20m'], 'negative LR result; no fabricated control eligibility')

# Retained-weight inference is a bounded CPU diagnostic, not a changed evaluator.
cpupath = 'research/v2/operator/20260926-trn22-clean-checkpoint-cpu'
cpu, cr = bundle(cpupath + '/completed/results-bundle.json', '74e2b06b')
r = cr('receipts/rank0.json')
guard(cr('under-lock-idle.json'), 832589)
check(r['backend'] == 'cpu' and r['rank'] == 0 and r['world_size'] == 1, 'CPU only one rank')
check(r['reference_sha256'] == parent['train_py_sha256'] and r['candidate_sha256'] == '3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f'
      and r['checkpoint_sha256'] == '068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25', 'unchanged exact trained checkpoint and loaders')
order = ['A','B','AB','BA','AC','AA','suffix_AB','AB_repeat','A_repeat']
expected_names = {s+'_'+c for s in ('reference','candidate') for c in ('B2_vs_independent_rows','permutation','partner_content_A','duplicate_AA','repeat_AB','repeat_A','synthetic_suffix_causality','state_invariance')} | {'source_parity_'+c for c in order}
check(r['call_order'] == order and len(r['cases']) == 25 and {c['name'] for c in r['cases']} == expected_names, 'complete exact diagnostic matrix')
for case in r['cases']:
    check(case['passed'] is True, 'CPU case ' + case['name'])
    if 'actual' in case:
        check(case['actual'] == case['expected'] and case['actual']['finite'] and case['exact_bytes'] is True
              and case['byte_mismatch_count'] == case['element_mismatch_count'] == case['max_abs'] == 0, 'actual byte parity '+case['name'])
check(r['outputs']['reference'] == r['outputs']['candidate'], 'all nine independent output digest records match')
check(len(r['checkpoint_state']) == 65, 'all65 checkpoint state tensors observed')
checkpoint_hashes = cr('checkpoint-readonly.json')
check(checkpoint_hashes['before'] == checkpoint_hashes['after'] == r['checkpoint_sha256'] and checkpoint_hashes['bytes'] == 453048594, 'checkpoint unchanged after diagnostic')
check(cr('terminal.json')['returncode'] == cr('outer-flock-exit.json')['returncode'] == 0 and cr('terminal.json')['driver_elapsed_to_terminal_seconds'] < 600, 'CPU bounded terminal exit')
check(r['source_review'] == 'REFUSE_UNCHANGED' and r['upload'] == 'HOLD' and r['quality_evidence'] is False, 'review refusal and upload hold preserved')

print(json.dumps({'status':'AUTHENTICATED_COLD_GATE_AND_LR_LOSS','checks':len(names),'check_names':names,
    'cold':{'source_sha256':launch['source_sha256'],'fifth_seconds':fifth,'process_wall_seconds':receipt['wall_seconds'],
            'charged_seconds':receipt['charged_through_process_exit'],'cold_gate_pass':True,'quality_measured':False,'steps':5},
    'lr':{'label':b['label'],'public20m':bpb,'delta_vs_same_host_parent':bpb-parent['public_20m'],'steps':3629,
          'startup_seconds':tr['observations'][4]['observed_seconds'],'charged_seconds':charge,
          'checkpoint':post,'adopt':False,'formal_control_eligible':False},
    'checkpoint_cpu':{'cases':25,'same_backend_exact':True,'scope':r['coverage_limit'],'native_tested':False,'source_review':'REFUSE_UNCHANGED'},
    'submission_eligible':False},indent=2))
