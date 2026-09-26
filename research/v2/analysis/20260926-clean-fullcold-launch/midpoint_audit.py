"""Authenticate a live midpoint and read-only binding facts, not final results."""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'research/v2/operator'
P = BASE / '20260926-dual-status-0523'
COUNT = 0

def check(value, name):
    global COUNT
    if not value:
        raise RuntimeError(name)
    COUNT += 1

hosts = {}
for host, pin in (
    ('trn21', '052080a1de4924cc7f21c8c34c54699cc6b10c8d8e287e1e8b8e7425ad9396f1'),
    ('trn22', '4a6f40535f608bb95c4f32748f99a86fb4dbb96f731fc207f395ef2cf9a14c40'),
):
    raw = (P / (host + '.json')).read_bytes()
    check(hashlib.sha256(raw).hexdigest() == pin, host + ':raw')
    record = json.loads(raw)
    check(record['returncode'] == 0 and record['stderr'] == '', host + ':transport')
    hosts[host] = record['remote']

r = hosts['trn22']
check((r['lock']['device'], r['lock']['inode'], r['lock']['held']) == (66305, 832589, True), 'original_held_lock')
check(r['ps_rc'] == 0 and r['neuron_ls_rc'] == 0, 'live_observation_commands')
check(len(r['files']) == 8, 'eight_complete_captures')
for name, f in r['files'].items():
    b = f['text'].encode()
    check(f['capture'] == 'complete_at_observation' and len(b) == f['captured_bytes'] == f['bytes_at_stat'] and hashlib.sha256(b).hexdigest() == f['sha256_captured'], name)
prior = json.loads((BASE / '20260926-trn22-confirmed36-clean-cold-full-r2/trn22.json').read_text())['remote']
old = {p['pid']: p for p in prior['processes']}
now = {p['pid']: p for p in r['processes']}
for pid in (3281907, 3281908, 3281909, 3281925, 3281926, 3281940, 3281955, 3281956, 3281957, 3281958):
    p = now[pid]
    check(p['exists'] and p['identity_stable'], 'live:' + str(pid))
    check(all(p[k] == old[pid][k] for k in ('starttime', 'ppid', 'pgid', 'sid', 'argv', 'cwd', 'environment_whitelist')), 'identity_env:' + str(pid))
check({p['pid'] for p in r['neuron_ls'][0]['neuron_processes']} == {3281955, 3281956, 3281957, 3281958}, 'four_actual_device_workers')
markers = [int(x) for x in r['files']['train/steps.jsonl']['text'].splitlines()]
check(markers == list(range(1855)), 'sequential_completed_markers')
log = r['files']['training.log']['text']
matches = re.findall(r'^step (\d+) \| loss ([0-9.]+).*?tok/s ([0-9,]+).*?remaining_time (\d+)s', log, re.M)
check(all(math.isfinite(float(m[1])) for m in matches), 'finite_printed_losses')
check(tuple(matches[-1]) == ('01850', '2.9475', '265,783', '890'), 'latest_printed_step')
check(not re.search(r'Traceback|RuntimeError|AssertionError|FATAL', log), 'no_observed_training_error')
check('terminal.json' not in r['files'] and 'fresh-public20m.log' not in r['files'], 'no_observed_final_result')
check(r['free_bytes'] > 13.5 * 1024**3, 'current_trn22_disk_reserve')

t = hosts['trn21']
check((t['lock']['device'], t['lock']['inode'], t['lock']['held_for_readonly_checks']) == (66305, 596823, True), 'trn21_original_readonly_lock')
check(t['job_rows'] == [] and t['device_idle'], 'trn21_idle_at_observation')
check(t['checks']['ps']['returncode'] == t['checks']['neuron_ls']['returncode'] == 0, 'trn21_ps_neuron')
check(t['project_clean'] and t['checks']['project_status']['stdout'] == '', 'trn21_clean')
head = t['checks']['project_head']['stdout'].strip()
check(head == 'e65a3097a5b00f6178188a935bb2acb9a5042c6c', 'trn21_project_head')
check(t['source']['matches_expected'] and t['source']['stable'] and t['source']['before'] == t['source']['after'], 'trn21_stable_source')
check(t['source']['sha256'] == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb', 'trn21_exact36')
check(len(t['path_existence']) == 2 and all(not p['exists'] for p in t['path_existence'].values()), 'k2_targets_absent')
check(t['memory']['MemAvailable_bytes'] > 64 * 1024**3, 'trn21_available_ram')
check(t['free_bytes'] < 13.5 * 1024**3 < 16 * 1024**3, 'trn21_disk_still_fails')
check(t['no_host_writes'] and t['no_reservation'] and t['no_launch'], 'trn21_observation_only')
print(json.dumps({'verdict': 'ACCEPT_LIVE_MIDPOINT_AND_READONLY_BINDING_FACTS', 'checks_passed': COUNT, 'observed_trn22_utc': r['utc'], 'completed_markers': len(markers), 'printed_step': 1850, 'tokens_per_second': 265783, 'source_remaining_seconds': 890, 'training_complete': False, 'public20m': None, 'observed_trn21_utc': t['utc'], 'trn21_project_head': head, 'trn21_free_bytes': t['free_bytes'], 'trn21_memavailable_bytes': t['memory']['MemAvailable_bytes'], 'no_new_launch_authority': True, 'submission_ready': False}, sort_keys=True))
