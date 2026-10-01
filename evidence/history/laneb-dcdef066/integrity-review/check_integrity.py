#!/usr/bin/env python3
"""Independent, read-only stdlib reference; no supplied checker is imported."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path('/home/poojith-devan/Desktop/Trainium')
PACKAGE = ROOT / 'research/v2/submission-prep/20260930-r136-laneb-two-receipt-audit'
MANIFEST = '2618fe90bd2580c76e08eca28c599401f1428c1ddcd224aa082713b6aad78ace'
SOURCE = 'dcdef066d8d0827f4ecdf0a80f8517e5348af28eea7acc06e03df4db49d4e90d'
COMMAND = b'NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0\n'
FULL_PINS = ['05a764083999229bb92dc7a9c3b177480dcdb3d0f526cdc78bdb884c73645bac',
             'defb95ea5df8ac81778e4f0c637ab44846b779ceda006b3b6b16de748e962b38']
ALLOWED_PACKAGE_CONTENTS = {
    'train.py', 'launch-command.txt', 'SOURCE.diff', 'AUDIT_BRIEF.md',
    'CHECKLIST.json', 'DIFF_AND_MANIFEST.md', 'REFERENCES.json',
    'cold-evidence.json', 'source-cpu-evidence.json', 'ancestor-evidence.json',
    'SOURCE_INVENTORY.json', 'LEGACY_FULL21_ADAPTER.diff', 'LEGACY_PARENT_ADAPTER.diff',
}
CHECKS = []


def check(name, ok):
    if not ok:
        raise AssertionError(name)
    CHECKS.append(name)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def pinned(reference):
    path = Path(reference['path'])
    check('canonical reference ' + path.name, path.is_absolute() and path.is_relative_to(ROOT) and not path.is_symlink())
    data = path.read_bytes()
    check('reference pin ' + path.name, digest(data) == reference['sha256'] and len(data) == reference['bytes'])
    return json.loads(data)


def raw_files(archive, index=None):
    result = {}
    for name, record in archive['files'].items():
        raw = record['text'].encode()
        check('member ' + name, digest(raw) == record['sha256'] and len(raw) == record['bytes'])
        if 'full_file_sha256' in record:
            check('complete member ' + name, record['full_file_sha256'] == digest(raw) and record['full_file_bytes'] == len(raw))
        if index is not None:
            check('indexed member ' + name, index[name] == {'sha256': digest(raw), 'bytes': len(raw)})
        result[name] = raw
    if index is not None:
        check('member index exhaustive', set(result) == set(index))
    return result


def json_objects(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def apply_diff(parent, patch):
    """Own unified-diff replay, validating every removed/context source line."""
    original = parent.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    output, cursor, i = [], 0, 2
    while i < len(lines):
        hunk = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n', lines[i])
        assert hunk, lines[i]
        position = int(hunk[1]) - 1
        assert position >= cursor
        output.extend(original[cursor:position])
        cursor, i = position, i + 1
        old_count = new_count = 0
        while i < len(lines) and not lines[i].startswith('@@'):
            sign, value = lines[i][0], lines[i][1:]
            assert sign in ' +-'
            if sign in ' -':
                assert original[cursor] == value
                cursor += 1
                old_count += 1
            if sign in ' +':
                output.append(value)
                new_count += 1
            i += 1
        assert old_count == int(hunk[2] or 1) and new_count == int(hunk[4] or 1)
    output.extend(original[cursor:])
    return ''.join(output)


def full_receipt(summary, expected_source, expected_archive):
    check('raw FULL pin independently specified', summary['raw_archive']['sha256'] == expected_archive)
    archive, packet = pinned(summary['raw_archive']), pinned(summary['actual_packet'])
    files = raw_files(archive, summary['all_raw_members'])
    job = packet['binding']
    packet_raw = {}
    for name, item in packet['files'].items():
        raw = base64.b64decode(item['b64'], validate=True)
        check('packet member hash ' + name, digest(raw) == item['sha256'])
        # Not all retained package members were archived. Do not invent them.
        if name in files:
            check('packet archive identity ' + name, files[name] == raw)
        packet_raw[name] = raw
    check('source bytes in actual archive', digest(files['package/train.py']) == expected_source)
    if expected_source == SOURCE:
        check('local package equals executed source', files['package/train.py'] == (PACKAGE / 'train.py').read_bytes())
    check('command bytes actual', files['package/launch-command.txt'] == COMMAND)
    selname = summary['actual_selection_member']
    check('real selection member', selname in files and files[selname] == packet_raw[selname] and
          digest(files[selname]) == summary['actual_selection_member_sha256'])
    selection = json.loads(files[selname])
    check('real selection fields', selection['host'] == job['host'] and selection['source_sha256'] == expected_source)
    check('no synthesized R98 selection', selname == 'R98_ROOT_SELECTION.json' or 'R98_ROOT_SELECTION.json' not in files)
    j = lambda name: json.loads(files[name])
    pre = j('operator-preflight.json')
    caches = pre['fresh_cache_absent_then_created_empty']
    check('six empty private cache receipts', len(caches) == 6 and all(x['created_empty'] is True for x in caches) and
          len({x['path'] for x in caches}) == 6 and len({(x['device'], x['inode']) for x in caches}) == 6)
    check('cache records faithfully copied', caches == summary['measured_cold_and_budget']['six_cache_creation_records'])
    check('eval cache disclosed reused', pre['eval_cache_mode'] == 'EXISTING_SHARED_RESEARCH_CACHE_PRESERVED')
    train = j('train/rehearsal.json')
    obs = train['observations']
    check('recorded boundary sequence', [x['step'] for x in obs] == list(range(len(obs))) and
          all(a['observed_seconds'] < b['observed_seconds'] for a, b in zip(obs, obs[1:])))
    cold = obs[4]['observed_seconds']
    charged = train['wall_seconds'] - min(cold, 900.0)
    check('own measured budget arithmetic', cold == train['startup_allowance_observed'] <= 900 and
          abs(charged - train['charged_through_process_exit']) < 1e-9 and 0 < charged <= 1800 and
          train['charged_through_process_exit_within_budget'] is True and train['returncode'] == 0)
    measured = summary['measured_cold_and_budget']
    check('budget summary agrees raw', measured['raw_cold_seconds'] == cold and
          measured['charged_through_process_exit_seconds'] == train['charged_through_process_exit'])
    before, after = j('checkpoint.json'), j('checkpoint-after-postflight.json')
    collection = archive['checkpoint_after_terminal']
    check('stable checkpoint receipt chain', before['sha256'] == after['sha256'] == collection['sha256'] == train['checkpoint_sha256'] and
          before['bytes'] == after['bytes'] == collection['bytes'] == train['checkpoint_bytes'] and collection['stable'] is True)
    check('successful terminal exits', j('terminal.json')['eval_exit_code'] == j('outer-flock-exit.json')['returncode'] == 0 and
          files['run-exit-code.txt'].strip() == files['eval-exit-code.txt'].strip() == b'0')
    check('postflight idle receipt', archive['postflight']['all_observed_pids_absent'] is True and
          archive['postflight']['neuron_device_idle'] is True and not archive['postflight']['busy_job_rows'])
    env = j('training-launch.json')['environment']
    check('seed full chip budget', env['NEURON_COMPETITION_R1_SEED'] == '42' and env['NEURON_LOGICAL_NC_CONFIG'] == '2' and
          env['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800')
    git = j('source-git-prelaunch.json')
    check('clean executed snapshot', all(git[k]['git_dirty'] is False and git[k]['files']['train.py'] == expected_source
          for k in ('before_stage', 'before_dispatch')))
    elog = files['fresh-public20m.log'].decode()
    argv = json.loads(next(line[len('REHEARSAL_COMMAND '):] for line in elog.splitlines() if line.startswith('REHEARSAL_COMMAND ')))
    check('fresh evaluator arguments', argv[2] == 'eval-public' and argv[argv.index('--seq-len') + 1] == '1024' and
          argv[argv.index('--batch-size') + 1] == '1' and argv[argv.index('--eval-tokens') + 1] == '20971520')
    score = next(x for x in json_objects(elog) if 'val_bpb' in x)
    calculated = score['total_nats'] / (math.log(2) * score['num_bytes'])
    check('own BPB arithmetic and 20M', math.isfinite(score['val_bpb']) and abs(calculated - score['val_bpb']) < 1e-12 and
          score['num_tokens'] == 20971520 and score['eval_over_timeout'] is False)
    check('score summary agrees raw', score['val_bpb'] == summary['verifier_proof']['val_bpb'])
    tlog = files['training.log'].decode()
    steps = int(re.search(r'^num_steps:\s+(\d+)$', tlog, re.M)[1])
    check('steps and rank consistency', steps == len(obs) == train['step_boundaries'] and
          re.search(r'^world_size:\s+4$', tlog, re.M) and 'pending_tokens=0' in tlog and
          'D2_TABLE_RANK_CONSISTENCY world_size=4 max_abs_diff_vs_rank0=0 ' in tlog)
    check('unknown storage fields preserved', all(summary['disk_and_io'][k] is None for k in
          ('total_bytes_written', 'cumulative_cache_bytes_written', 'cumulative_checkpoint_bytes_written',
           'cumulative_log_bytes_written', 'no_large_leftovers')) and
          summary['checkpoint']['currently_retained_on_host'] is None and summary['checkpoint']['backup_exists'] is None)
    return {'host': job['host'], 'archive_sha256': expected_archive, 'authenticated_members': len(files),
            'selection_member': selname, 'val_bpb': score['val_bpb'], 'raw_cold_seconds': cold,
            'charged_seconds_own_arithmetic': charged, 'steps': steps, 'eval_tokens': score['num_tokens'],
            'context': 1024, 'checkpoint_bytes': before['bytes'], 'checkpoint_sha256': before['sha256']}


def main():
    check('frozen manifest', digest((PACKAGE / 'MANIFEST.json').read_bytes()) == MANIFEST)
    manifest = read_json(PACKAGE / 'MANIFEST.json')
    for name in sorted(ALLOWED_PACKAGE_CONTENTS):
        check('scoped package pin ' + name, digest((PACKAGE / name).read_bytes()) == manifest['files'][name])
    check('no package acceptance', manifest['package_accepted'] is False and manifest['launch_authorized'] is False and
          manifest['submission_authorized'] is False and manifest['export_authorized'] is False)
    source = (PACKAGE / 'train.py').read_text()
    check('exact candidate and command', digest(source.encode()) == SOURCE and (PACKAGE / 'launch-command.txt').read_bytes() == COMMAND)
    parent = (ROOT / 'research/v2/proposals/20260929-r109-ce2-flat-zbundle/train.py').read_text()
    check('parent exact source', digest(parent.encode()) == 'e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310')
    check('own unified diff reconstructs every byte', apply_diff(parent, (PACKAGE / 'SOURCE.diff').read_text()) == source)
    ast.parse(source)
    check('source contains no training attribute', not any(isinstance(n, ast.Attribute) and n.attr == 'training' for n in ast.walk(ast.parse(source))))
    cold = read_json(PACKAGE / 'cold-evidence.json')
    receipts = [full_receipt(r, SOURCE, pin) for r, pin in zip(cold['receipts'], FULL_PINS)]
    mean = sum(r['val_bpb'] for r in receipts) / 2
    check('two receipt mean', len(cold['receipts']) == 2 and mean == cold['mean_public20m'])
    ancestor = read_json(PACKAGE / 'ancestor-evidence.json')
    ancestor_result = full_receipt(ancestor, 'e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310',
                                  '088637f6f4db8ce993f5cc52e80ccc95f5e3441aacb5cc06ce88e59af2be1a0f')
    for filename, actual in [('LEGACY_FULL21_ADAPTER.diff', 'R113_ROOT_SELECTION.json'),
                             ('LEGACY_PARENT_ADAPTER.diff', 'R100_ROOT_SELECTION.json')]:
        patch = (PACKAGE / filename).read_text().splitlines()
        removed = [x[1:] for x in patch if x.startswith('-') and not x.startswith('---')]
        added = [x[1:] for x in patch if x.startswith('+') and not x.startswith('+++')]
        check('single lookup legacy adapter ' + actual, len(removed) == len(added) == 1 and
              removed[0].replace('R98_ROOT_SELECTION.json', actual) == added[0])
    cpu = read_json(PACKAGE / 'source-cpu-evidence.json')
    cpu_results = []
    for kind, groups in [('author', 10), ('own', 8)]:
        transport = pinned(cpu[kind + '_archive'])
        check('CPU transport success ' + kind, transport['returncode'] == 0)
        files = raw_files(json.loads(transport['stdout']), cpu[kind + '_raw_members'])
        result, terminal = json.loads(files['result.json']), json.loads(files['reference-terminal.json'])
        check('CPU raw result and terminal ' + kind, result['source_sha256'] == SOURCE and result['numerical_pass'] is True and
              result['CPU_eager_kernel_surrogates_only'] is True and result['native_kernel_executed'] is False and
              terminal['raw_worker_exit_code'] == terminal['exit_code'] == 0 and
              terminal['result_sha256'] == digest(files['result.json']))
        count = result['check_groups'] if kind == 'author' else len(result['checks'])
        check('CPU group count ' + kind, count == groups)
        cpu_results.append({'kind': kind, 'groups': count, 'metrics': len(result['metrics']), 'result_sha256': digest(files['result.json']),
                            'status': result['status'], 'native_executed': result['native_kernel_executed']})
    prior_review = pinned(cpu['review_reference'])
    check('existing sourceCPU authority preserved', prior_review['verdict'] == 'ACCEPT_SOURCE_CPU_FOR_RESEARCH_FULL' and
          prior_review['source_sha256'] == SOURCE and prior_review['independent_result_sha256'] == cpu_results[1]['result_sha256'])
    checklist = read_json(PACKAGE / 'CHECKLIST.json')
    check('all eight honest statuses', [r['item'] for r in checklist['items']] == list(range(1, 9)) and
          [r['status'] for r in checklist['items']] == ['FAIL', 'FAIL', 'FAIL', 'PASS', 'FAIL', 'FAIL', 'FAIL', 'FAIL'])
    check('unknowns and nonacceptance preserved', all(checklist[k] is None for k in
          ('claude_audit', 'last_claude_audited_ancestor', 'fresh_independent_package_review')) and
          checklist['unknown_is_fail_pending_evidence'] is True and checklist['package_accepted'] is False)
    brief = (PACKAGE / 'AUDIT_BRIEF.md').read_text()
    check('brief reflects numeric receipts', all(str(x) in brief for r in receipts for x in
          (r['val_bpb'], r['raw_cold_seconds'])) and 'Mean' + str(mean) in brief)
    print(json.dumps({'verdict': 'PACKAGE_INTEGRITY_ONLY', 'checks_passed': len(CHECKS), 'manifest_sha256': MANIFEST,
          'test_sha256': digest(Path(__file__).read_bytes()), 'source_sha256': SOURCE,
          'scoped_package_members_hashed': len(ALLOWED_PACKAGE_CONTENTS),
          'package_members_not_read': sorted(set(manifest['files']) - ALLOWED_PACKAGE_CONTENTS),
          'full_receipts': receipts, 'mean_public20m': mean, 'research_ancestor': ancestor_result,
          'existing_cpu_receipts': cpu_results, 'fresh_session': False,
          'fresh_spawn_failure': 'collab spawn failed: agent thread limit reached',
          'sourceCPU_acceptance': 'existing c5ebb332 review only; no new source acceptance',
          'claude_compliance_acceptance': None, 'submission_ready': False, 'host_calls': 0,
          'torch_or_model_execution': False, 'package_changed': False}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
