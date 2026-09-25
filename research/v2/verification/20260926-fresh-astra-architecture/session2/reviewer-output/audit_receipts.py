"""Independently check internal consistency and exact recorded OFF/ON parity.

No gate harness or checkpoint tensors are supplied: these are receipt checks,
not recomputed production losses/digests or a rerun of full-size G2.
"""
import hashlib
import json
import math
import re
import struct
from pathlib import Path

ROOT = Path('/packet/raw')
OUT = Path('/work')
CHECKS = 0

def check(value, message):
    global CHECKS
    CHECKS += 1
    assert value, message

def read(path):
    return json.loads(path.read_text())

def float_bits(value):
    return struct.pack('>d', value).hex()

def main():
    expected_hashes = read(OUT / 'input_audit.json')['sources']
    gates, summaries = {}, {}
    for mode in ('reference', 'off', 'on'):
        root = ROOT / mode
        gate = read(root / 'g2.json')
        gates[mode] = gate
        expected_source = expected_hashes['original' if mode == 'reference' else 'candidate']['sha256']
        check(gate['train_py_sha256'] == expected_source, mode + ' source identity')
        check(gate['schema'] == 'g2-preflight-v1', mode + ' schema')
        check(gate['steps'] == 3 and gate['nproc'] == 4 and gate['micro_batch'] == [2, 256], mode + ' shape')
        check(gate['torchrun_rc'] == 0 and gate['timed_out'] is False, mode + ' completion')
        check(gate['verdict'] == 'PASS', mode + ' verdict label consistency')
        check(gate['prepare']['stub'] is False, mode + ' non-stub prepare claim')
        check(len(gate['ranks']) == 4, mode + ' four rank records')
        check(gate['torch'] == '2.11.0+cpu', mode + ' reported CPU torch')
        appended = gate['appended_argv']
        check(appended[:6] == ['--device-type', 'cpu', '--no-compile', '--num-steps', '3', '--out-dir'], mode + ' override scope')
        rank_summary = []
        for rank in range(4):
            r = read(root / f'rank{rank}.json')
            check(r == gate['ranks'][rank], mode + f' raw rank{rank} equals aggregate')
            check(r['rank'] == rank and r['status'] == 'PASS', mode + ' rank identity/status')
            check(len(r['losses']) == 3 and all(math.isfinite(x) and x < 100 for x in r['losses']), mode + ' three sane losses')
            check(r['n_changed'] == r['n_params'] == 65, mode + ' all parameters changed claim')
            for key in ('cross_rank_mismatches', 'unchanged_params', 'unchanged_tables', 'tables'):
                check(r[key] == [], mode + '/' + key)
            check(r['rmsprop_trace'] == {'calls': 0, 'checks': [], 'optimizer_steps': 3}, mode + ' RMS tracing vacuous')
            check(r['plan_sha256'] == gate['plan_sha256'], mode + ' plan digest')
            check(r['shims'] == gate['shims'] and r['appended_argv'] == appended, mode + ' rank shims/overrides')
            check(r['torch'] == gate['torch'], mode + ' rank torch')
            rank_summary.append({'rank': rank, 'losses': r['losses'], 'loss_float64_bits': list(map(float_bits, r['losses'])),
                                 'n_params': r['n_params'], 'n_changed': r['n_changed']})
        r0 = gate['ranks'][0]
        check(len(r0['final_digests']) == 65, mode + ' 65 final rank0 digests')
        check(all(re.fullmatch('[0-9a-f]{16}', v) for v in r0['final_digests'].values()), mode + ' 64-bit digest format')
        cp = r0['checkpoint_check']
        check(cp == {'deterministic': True, 'logits_shape': [1, 256, 8192], 'ok': True, 'path_exists': True}, mode + ' checkpoint receipt')
        log = (root / 'torchrun.log').read_text()
        lines = log.splitlines()
        check(lines[-len(gate['torchrun_log_tail']):] == gate['torchrun_log_tail'], mode + ' raw log tail')
        check(not re.search(r'Traceback|\bFAIL\b|ChildFailedError', log), mode + ' no recorded failure')
        step_lines = re.findall(r'^step (\d+) \| loss ([0-9.]+) \|', log, re.M)
        check(len(step_lines) == 3, mode + ' three logged steps')
        smooth = 0.0
        for i, ((step, rounded), raw_loss) in enumerate(zip(step_lines, r0['losses'])):
            smooth = 0.9 * smooth + 0.1 * raw_loss
            check(int(step) == i and rounded == f'{smooth / (1 - 0.9 ** (i + 1)):.4f}', mode + ' logged smoothed losses agree')
        owner_lines = [json.loads(line.split(' ', 1)[1]) for line in lines if line.startswith('SUBMISSION_OPTIMIZER_INTEGRATION ')]
        check(sorted(r['rank'] for r in owner_lines) == list(range(4)), mode + ' logged four optimizer owners')
        for owner in owner_lines:
            check(owner.pop('owner_partitioning') is True, mode + ' owner enabled')
            rank = owner.pop('rank')
            check(owner == gate['ranks'][rank]['plan_summary'], mode + ' owner log equals rank plan')
        for text in ('device=cpu dtype=torch.bfloat16 world_size=4',
                     'training target=3 steps max_seconds=1800 official_budget=1800s',
                     'micro_tokens=131,072 grad_accum_steps=1 pack_factor=32',
                     'compiled:          False', 'compile_muon:      True',
                     'budget_seconds:    1800', 'startup_allowance_est: 0.0',
                     "architecture: unet_skip_pairs=4 attention_layers=(0, 2, 4, 6, 7) value_embedding_layers=('7',) ngram_tables=()"):
            check(text in log, mode + ' log scope: ' + text)
        check(sum('muon step: COMPILED body' in line for line in lines) == 3, mode + ' Muon compile messages')
        if mode != 'reference':
            request, launch, terminal = (read(root / (name + '.json')) for name in ('request', 'launch', 'terminal'))
            for record in (request, launch, terminal):
                check(record['cpu_only'] is True and record['source_sha256'] == expected_source, mode + ' metadata identity')
            check(request['argv'] == gate['argv'], mode + ' request argv')
            check(launch['command'] == request['command'], mode + ' launch command')
            check(request['command'][request['command'].index('--') + 1:] == gate['argv'], mode + ' command argv')
            check(terminal['exit_code'] == 0 and terminal['verdict'] == 'PASS', mode + ' terminal completion')
            check(request['prepare_sha256'] == gate['prepare']['sha256'], mode + ' prepare identity')
            check(request['steps'] == 3 and request['nproc'] == 4 and request['micro_rows'] == 2 and request['micro_tokens'] == 256, mode + ' request dimensions')
            check(request['timeout_seconds'] == 3600, mode + ' supplied gate timeout')
            # These are supplied launch-receipt fields; no process environment is inspected.
            check(all(launch['environment'][k] == '1' for k in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), mode + ' recorded thread limits')
            check(launch['environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800', mode + ' recorded budget')
            check(launch['environment']['NEURON_COMPETITION_R1_SEED'] == '42', mode + ' recorded seed')
        summaries[mode] = {'source_sha256': expected_source, 'ranks': rank_summary,
                           'final_digest_count_rank0': len(r0['final_digests']),
                           'checkpoint_receipt': cp, 'plan_sha256': gate['plan_sha256'],
                           'elapsed_s_receipt': gate['elapsed_s'], 'raw_log_checks': 'PASS'}
    reference = gates['reference']
    for mode, flag in (('off', '--no-initial-cache-refresh'), ('on', '--initial-cache-refresh')):
        gate = gates[mode]
        check(gate['argv'] == reference['argv'] + [flag], mode + ' only intended flag differs')
        for key in ('prepare', 'shims', 'plan_sha256', 'steps', 'nproc', 'micro_batch', 'torch', 'python'):
            check(gate[key] == reference[key], mode + ' same ' + key)
        for rank in range(4):
            a, b = reference['ranks'][rank], gate['ranks'][rank]
            check({k: v for k, v in a.items() if k != 'appended_argv'} == {k: v for k, v in b.items() if k != 'appended_argv'}, mode + ' all recorded rank fields parity')
            check(list(map(float_bits, a['losses'])) == list(map(float_bits, b['losses'])), mode + ' loss float encodings')
        check(gate['ranks'][0]['final_digests'] == reference['ranks'][0]['final_digests'], mode + ' all 65 final digests')
    check(read(ROOT / 'off/request.json')['g2_tool_sha256'] == read(ROOT / 'on/request.json')['g2_tool_sha256'], 'same candidate harness hash claim')
    result = {'status': 'PASS', 'assertions': CHECKS, 'runs': summaries,
              'off_vs_reference': {'loss_values_equal': 12, 'final_rank0_digests_equal': 65},
              'on_vs_reference': {'loss_values_equal': 12, 'final_rank0_digests_equal': 65},
              'shims': reference['shims'], 'prepare_sha256_recorded': reference['prepare']['sha256'],
              'g2_tool_sha256_recorded': read(ROOT / 'off/request.json')['g2_tool_sha256'],
              'limitations': ['Receipt consistency, not independent execution of supplied full-model G2.',
                              'No gate harness source, prepare.py, tensors, checkpoint, or per-step parameter digests supplied.',
                              'Only rank0 has final parameter digest values; other ranks report empty mismatch lists.',
                              '64-bit final digests cannot establish byte equality of every intermediate full-model tensor.',
                              'Synthetic data/tokenizer, BF16 override, CE no-op, fake clock, 2x256 slice, three steps.',
                              'Whole-model compile disabled; Muon compile remains enabled per logs.',
                              'RMSprop trace has zero calls; no table updates exercised.',
                              'No native compilation, startup, hardware performance, scored quality, or submission evidence.']}
    (OUT / 'receipt_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
