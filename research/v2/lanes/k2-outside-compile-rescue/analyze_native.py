"""Report the precommitted native201 net-work screen; never launches/adopts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCE = '949f262c092c141f715bdd71532c42e0bcf502d926a1ee20bc885a2255062daa'


def net_screen(observations, reference):
    if len(observations) < 200: raise ValueError('LESS_THAN200_BOUNDARIES')
    by_step = {o['step']: o['observed_seconds'] for o in observations}
    if len(by_step) != len(observations): raise ValueError('DUPLICATE_BOUNDARY')
    values = [o['observed_seconds'] for o in observations]
    if not all(math.isfinite(v) and v >= 0 for v in values): raise ValueError('NONFINITE_TIME')
    if not all(b > a for a, b in zip(values, values[1:])): raise ValueError('NONMONOTONIC_TIME')
    if observations[4]['step'] != 5: raise ValueError('WRONG_FIFTH_BOUNDARY')
    startup = observations[4]['observed_seconds']
    seconds = by_step[200] - by_step[9]
    rate = 191 * 262144 / seconds
    charge = max(0, startup - 900)
    net = max(0, 1800 - charge) * rate
    base = reference['zero_charge_base_upper_net_tokens']
    return {'cold_startup_seconds': startup, 'charged_startup_seconds': charge,
            'steady_window_boundaries': [9, 200], 'steady_window_seconds': seconds,
            'steady_tok_s': rate, 'net_tokens': net, 'base_upper_net_tokens': base,
            'net_token_ratio': net / base, 'net_screen_pass': net >= 1.03 * base,
            'baseline_cold_measured': False,
            'scope': 'Cold candidate vs conservative zero-charge matched base; screen only, not BPB/adoption'}


def report(path):
    raw = path.read_bytes()
    bundle = json.loads(raw)
    files = bundle['files']
    for name, item in files.items():
        data = item['text'].encode()
        if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
            raise ValueError('RAW_FILE_DRIFT:' + name)
    obj = lambda name: json.loads(files[name]['text'])
    binding = obj('NATIVE_BINDING.json')
    if binding['source_sha256'] != SOURCE or files['package/train.py']['sha256'] != SOURCE:
        raise ValueError('WRONG_SOURCE')
    if binding['host'] != 'trn21' or binding['steps'] != 201: raise ValueError('WRONG_TIMING_JOB')
    terminal = obj('terminal.json')
    if terminal['status'] != 'TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS' or terminal['train_exit_code'] != 0:
        raise ValueError('NATIVE_TIMING_FAILED:' + json.dumps(terminal))
    tr = obj('train/rehearsal.json')
    if tr['returncode'] != 0 or tr['charged_through_process_exit_within_budget'] is not True:
        raise ValueError('NATIVE_BUDGET_OR_EXIT_FAILED')
    created = obj('operator-preflight.json')['fresh_cache_absent_then_created_empty']
    if len(created) != 6 or not all(x['created_empty'] for x in created): raise ValueError('NO_COLD_PROOF')
    if {x['path'] for x in created} != set(binding['fresh_cache_directories']): raise ValueError('WRONG_COLD_PATHS')
    reference_raw = (HERE / 'NATIVE_TIMING_REFERENCE.json').read_bytes()
    if hashlib.sha256(reference_raw).hexdigest() != binding['timing_reference_sha256']:
        raise ValueError('TIMING_REFERENCE_CHANGED')
    reference = next(r for r in json.loads(reference_raw)['rows'] if r['host'] == binding['host'])
    screen = net_screen(tr['observations'], reference)
    if abs(tr['startup_allowance_observed'] - min(900, screen['cold_startup_seconds'])) > 1e-8:
        raise ValueError('STARTUP_ACCOUNTING_MISMATCH')
    candidate_losses = {}
    for line in files['training.log']['text'].splitlines():
        m = re.match(r'step (\d+) \| loss ([\d.eE+-]+)', line)
        if m and int(m[1]) <= 200:
            value = float(m[2])
            if not math.isfinite(value): raise ValueError('NONFINITE_LOSS')
            candidate_losses[int(m[1])] = value
    loss_pairs = []
    for r in reference['smoothed_loss_samples']:
        step = r['step']
        if step not in candidate_losses: raise ValueError('LOSS_SAMPLE_MISSING:' + str(step))
        loss_pairs.append({'step': step, 'base': r['smoothed_logged_loss'],
                           'candidate': candidate_losses[step],
                           'delta': candidate_losses[step] - r['smoothed_logged_loss']})
    cp = bundle['checkpoint_after_terminal']
    if not cp or not cp['stable'] or cp['sha256'] != tr['checkpoint_sha256']:
        raise ValueError('CHECKPOINT_MISSING_OR_CHANGED')
    post = bundle['postflight']
    if not (post['all_observed_pids_absent'] and post['neuron_device_idle'] and post['lock_competitor_rc'] == 0):
        raise ValueError('JOB_NOT_QUIESCENT')
    if post['busy_job_rows'] or post['related_rows'] or post['fuser_rc'] != 1: raise ValueError('BUSY_POSTFLIGHT')
    return dict(screen, host=binding['host'], label=binding['label'], source_sha256=SOURCE,
                archive_sha256=hashlib.sha256(raw).hexdigest(), steps=tr['step_boundaries'],
                charged_through_exit=tr['charged_through_process_exit'], loss_pairs=loss_pairs,
                loss_scope='Reported same-step smoothed losses; time-based LR can differ. No bit-exact/native correctness PASS inferred.',
                status='NET_SCREEN_PASS_PENDING_LOSS_REVIEW' if screen['net_screen_pass'] else 'NET_SCREEN_FAIL',
                full_bpb=None, submission_authorized=False)


def self_test():
    ref = {'zero_charge_base_upper_net_tokens': 1800 * 281291.26872485835}
    def obs(startup, rate):
        return [{'step': i, 'observed_seconds': startup - (5-i)*.1 if i <= 5 else startup+(i-5)*262144/rate} for i in range(1,202)]
    a = net_screen(obs(859, 290000), ref)
    if not a['net_screen_pass'] or a['charged_startup_seconds'] != 0: raise ValueError('ZERO_CHARGE_TEST')
    b = net_screen(obs(1200, 290000), ref)
    if b['net_screen_pass'] or b['charged_startup_seconds'] != 300: raise ValueError('CHARGE_TEST')
    if net_screen(obs(2701, 400000), ref)['net_tokens'] != 0: raise ValueError('EXHAUSTED_BUDGET_TEST')
    cases = [[], obs(859,290000)[:199], obs(859,290000)[:-1]+[obs(859,290000)[0]]]
    for bad in cases:
        try: net_screen(bad, ref)
        except ValueError: pass
        else: raise ValueError('INVALID_BOUNDARIES_ACCEPTED')
    print(json.dumps({'status': 'SYNTHETIC_CALCULATION_TEST_PASS', 'native_result': False, 'cases': 6}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', nargs='?', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: self_test()
    elif args.bundle: print(json.dumps(report(args.bundle), indent=2))
    else: parser.error('bundle or --self-test required')
