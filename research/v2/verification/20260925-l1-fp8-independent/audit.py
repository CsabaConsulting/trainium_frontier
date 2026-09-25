#!/usr/bin/env python3
"""Independent bounded receipt, protected-AST, and layout-algebra checks.

Does not import the training file, run a model, or execute NKI. The layout
witness tests the source's index mapping only, not compiler support.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LANE = ROOT / 'research/v2/lanes/fp8-projections'
TRAIN = 'workflow/proposals/research-v2/train.py'
BASE = 'c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586'
CANDIDATE = 'dea47910547572c704d8d5cc0e3dd058282fc8d862c913939bfc5e40830713b4'


def read_receipt(path, expected):
    receipt = json.loads(path.read_text())
    assert receipt['verdict'] == 'PASS' and receipt['torchrun_rc'] == 0
    assert receipt['train_py_sha256'] == expected
    assert receipt['nproc'] == 4 and receipt['micro_batch'] == [1, 16]
    assert sorted(row['rank'] for row in receipt['ranks']) == list(range(4))
    for row in receipt['ranks']:
        assert row['status'] == 'PASS' and len(row['losses']) == 3
        assert all(math.isfinite(loss) for loss in row['losses'])
        assert row['n_params'] == row['n_changed'] == 22
        assert row['cross_rank_mismatches'] == []
        assert row['plan_sha256'] == receipt['plan_sha256']
    rank0 = next(row for row in receipt['ranks'] if row['rank'] == 0)
    assert rank0['checkpoint_check']['ok'] and rank0['checkpoint_check']['deterministic']
    return receipt


def equal_training(a, b):
    assert a['argv'] == b['argv'] and a['plan_sha256'] == b['plan_sha256']
    left = {row['rank']: row for row in a['ranks']}
    right = {row['rank']: row for row in b['ranks']}
    for rank in range(4):
        assert left[rank]['losses'] == right[rank]['losses']
        assert left[rank]['plan_summary'] == right[rank]['plan_summary']
    assert left[0]['final_digests'] == right[0]['final_digests']


def main():
    torch.set_num_threads(1)
    source = (ROOT / TRAIN).read_bytes()
    base = subprocess.check_output(['git', 'show', '8b6d6cd9^:' + TRAIN], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == CANDIDATE
    assert hashlib.sha256(base).hexdigest() == BASE
    left, right = ast.parse(base), ast.parse(source)
    funcs = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    bf, cf = funcs(left), funcs(right)
    protected = ['load_for_eval', 'save_checkpoint', 'restore_eval_toggles',
                 'make_dataloader_requeue', 'lr_multiplier', 'clip_and_sync_gradients']
    for name in protected:
        assert ast.dump(bf[name]) == ast.dump(cf[name]), name
    loops = lambda node: [ast.dump(item) for item in ast.walk(node) if isinstance(item, ast.While)]
    assert loops(bf['main']) == loops(cf['main'])
    for path in ('prepare.py', 'workflow/tools/preflight_dryrun.py'):
        old = subprocess.check_output(['git', 'show', '8b6d6cd9^:' + path], cwd=ROOT)
        assert (ROOT / path).read_bytes() == old, path

    control = read_receipt(LANE / 'g2-base-c5c-small.json', BASE)
    supplied_off = read_receipt(LANE / 'g2-off-r3-small.json', CANDIDATE)
    supplied_on = read_receipt(LANE / 'g2-on-r3-small.json', CANDIDATE)
    rerun_off = read_receipt(HERE / 'g2-off-rerun-small.json', CANDIDATE)
    rerun_on = read_receipt(HERE / 'g2-on-rerun-small.json', CANDIDATE)
    equal_training(control, supplied_off)
    equal_training(control, rerun_off)
    equal_training(supplied_on, rerun_on)
    assert rerun_on['argv'] == rerun_off['argv'] + ['--fp8-proj']

    # Distinct token, K-block, K-partition, and output coordinates expose
    # incorrect transpose/order in the native source's [128,K/128,N] layout.
    tokens, channels, outputs = 128, 512, 256
    x = ((torch.arange(tokens * channels).reshape(tokens, channels) * 7) % 13 - 6).float()
    w = ((torch.arange(outputs * channels).reshape(outputs, channels) * 3) % 11 - 5).float()
    xp = x.reshape(tokens, channels // 128, 128).permute(2, 1, 0)
    wp = w.reshape(outputs, channels // 128, 128).permute(2, 1, 0)
    contracted = torch.zeros(tokens, outputs)
    for pair in range(channels // 256):
        contracted += torch.einsum('pit,pio->to', xp[:, 2*pair:2*pair+2], wp[:, 2*pair:2*pair+2])
    assert torch.equal(contracted, x @ w.T)
    token_scales = (torch.arange(tokens).float() + 1).reshape(-1, 1) / 128
    output_scales = (torch.arange(outputs).float() + 1).reshape(1, -1) / 256
    assert torch.equal((contracted * output_scales) * token_scales,
                       ((x @ w.T) * output_scales) * token_scales)
    result = dict(verdict='PASS_BOUNDED_INDEPENDENT_CPU_AND_STATIC',
        candidate_sha256=CANDIDATE, base_sha256=BASE,
        protected_function_asts_unchanged=protected, main_while_loop_ast_unchanged=True,
        prepare_and_g2_harness_bytes_unchanged=True,
        corrected_c5c_receipt_vs_supplied_off_identical=True,
        corrected_c5c_receipt_vs_independent_off_rerun_identical=True,
        supplied_on_vs_independent_on_rerun_identical=True,
        compared_parameters=22, ranks=4, optimizer_steps=3,
        effective_cpu_microbatch=[1, 16], reduced_model=[2, 256],
        native_index_mapping_algebra_witness=True,
        native_kernel_executed=False, exact_size_g2=False, speed_claim=False,
        base_control_rerun_in_this_review=False)
    print(json.dumps(result, indent=2))
    (HERE / 'independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
