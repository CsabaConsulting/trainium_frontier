"""Packet-local completion checks, not correctness self-approval or a launcher."""
import json
import math
from pathlib import Path
import re

CASE_NAMES = ('cross_tile_large', 'cross_tile_unit', 'unique', 'all_same',
              'counts_253_1_1_1', 'boundary_rows', 'signed_zero', 'bf16_subnormal',
              'fp32_subnormal', 'empty')
DTYPES = ('torch.bfloat16', 'torch.float32')
COMMON = ('upstream_roundtrip', 'direct_consumer', 'direct_repeated_call')
AUTOGRAD = ('dense_CPU_oracle', 'eager_autograd_forward', 'eager_autograd_backward',
            'compiled_enclosing_autograd_forward', 'compiled_enclosing_autograd_backward')


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate(root, label, environment, hashes, checker_args):
    root = Path(root)
    summaries = []
    for rank in range(4):
        receipt = json.loads((root / 'receipts' / ('rank' + str(rank) + '.json')).read_text())
        env = json.loads((root / ('rank-environment-r' + str(rank) + '.json')).read_text())
        exited = json.loads((root / ('rank-wrapper-exit-r' + str(rank) + '.json')).read_text())
        require(receipt['status'] == 'TINY_CONSUMER_EXACT_ONLY', 'CHECKER_NOT_COMPLETE')
        require(receipt['scope'] == 'TINY_CONSUMER_FALSIFIER_ONLY', 'WRONG_SCOPE')
        require(receipt['rank'] == rank and receipt['world_size'] == 4 and receipt['label'] == label, 'RECEIPT_IDENTITY')
        require(receipt['shape'] == {'V': 256, 'C': 128, 'T': 256} and receipt['seed'] == 42, 'SHAPE_SEED')
        require(receipt['argv'] == checker_args, 'CHECKER_ARGV')
        require(receipt['kernel_sha256'] == hashes['consumer_nki.py'] and
                receipt['checker_sha256'] == hashes['check_native.py'] and
                receipt['reference_sha256'] == hashes['reference.py'], 'CHECKER_SUPPORT_SHA')
        require(receipt['tolerance'] == {'atol': 0, 'rtol': 0, 'criterion': 'identical bytes'}, 'EXACTNESS_CRITERION')
        require(receipt['accepted_speedup'] is False and receipt['whole_step_speedup'] == 'UNMEASURED', 'INVALID_SPEED_CLAIM')
        require(finite(receipt['diagnostic_elapsed_seconds']), 'INVALID_ELAPSED')
        require(env['rank'] == env['local_rank'] == rank and env['world_size'] == 4, 'ENV_RANK')
        require(env['environment'] == environment, 'RANK_ENVIRONMENT_MISMATCH')
        require(env['checker_sha256'] == hashes['check_native.py'] and env['checker_argv'] == checker_args, 'ENV_SOURCE_ARGV')
        require(exited['rank'] == rank and type(exited['returncode']) is int and exited['returncode'] == 0 and exited['error'] is None, 'RANK_WRAPPER_EXIT')
        cases = receipt['cases']
        require(len(cases) == 20, 'CASE_COUNT')
        expected_cases = {(name, dtype) for name in CASE_NAMES for dtype in DTYPES}
        require({(case['name'], case['dtype']) for case in cases} == expected_cases, 'CASE_SET')
        comparisons = 0
        for case in cases:
            require(case['status'] == 'EXACT' and finite(case['direct_first_call_seconds']), 'CASE_INCOMPLETE')
            names = COMMON + (AUTOGRAD if case['name'] != 'empty' else ())
            require(len(case['checks']) == len(names) and {x['name'] for x in case['checks']} == set(names), 'COMPARISON_COVERAGE')
            for check in case['checks']:
                require(check['exact'] is True and check['finite'] is True, 'NONEXACT_OR_NONFINITE')
                require(check['shape'] == [256, 128], 'COMPARISON_SHAPE')
                expected_dtype = case['dtype'] if check['name'] in ('upstream_roundtrip', 'eager_autograd_forward', 'compiled_enclosing_autograd_forward') else 'torch.float32'
                require(check['actual_dtype'] == expected_dtype, 'COMPARISON_DTYPE')
                require(type(check['actual_sha256']) is str and re.fullmatch('[0-9a-f]{64}', check['actual_sha256']) is not None, 'INVALID_TENSOR_SHA')
                require(check['actual_sha256'] == check['expected_sha256'] and
                        finite(check['max_abs']) and check['max_abs'] == 0, 'CONTRADICTORY_EXACTNESS')
                comparisons += 1
        times = receipt['timings']
        require(len(times) == 4 and {(x['case'], x['dtype']) for x in times} ==
                {(name, dtype) for name in ('unique', 'all_same') for dtype in DTYPES}, 'TIMING_COVERAGE')
        for timing in times:
            require(len(timing['seconds']) == 3 and all(finite(x) and x > 0 for x in timing['seconds']), 'INVALID_TIMING')
            require(timing['scope'] == 'consumer_only_excludes_routing_and_transfer', 'TIMING_SCOPE')
        summaries.append(dict(rank=rank, complete_cases=len(cases), exact_comparisons=comparisons,
                              timing_samples=sum(len(x['seconds']) for x in times)))
    return dict(status='TINY_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_ARTIFACT_REVIEW',
                ranks=summaries, accepted_speedup=False, source_adoption=False,
                actual_model_correctness='NOT_TESTED', production_routing='NOT_IMPLEMENTED',
                full_run_authorized=False)
