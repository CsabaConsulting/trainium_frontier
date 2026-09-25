"""Offline prospective screen only; cannot launch, adopt or grant authority."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

CONTROL_MEDIAN = 268831.0
THRESHOLD = CONTROL_MEDIAN * 1.05
WINDOW = list(range(10, 101, 10))
ROW = re.compile(r'^step (\d+) \| loss (\S+) \| lrm (\S+) \| dt (\S+)s \| tok/s ([\d,]+)', re.M)


def analyze(log, receipt):
    if 'device=neuron dtype=torch.bfloat16 world_size=4' not in log:
        raise ValueError('NATIVE_FOUR_RANK_LOG_REQUIRED')
    if 'micro_tokens=131,072 grad_accum_steps=2 pack_factor=32' not in log:
        raise ValueError('BATCH262144_ACCUMULATION_NOT_PROVED')
    if (receipt.get('returncode') != 0 or receipt.get('mode') != 'train'
            or receipt.get('charged_through_process_exit_within_budget') is not True
            or receipt.get('checkpoint_bytes', 0) <= 0):
        raise ValueError('TERMINAL_CHECKPOINT_BUDGET_GATE_FAILED')
    observations = receipt.get('observations', [])
    if len(observations) != 105 or [o.get('step') for o in observations] != list(range(105)):
        raise ValueError('REQUIRE_105_SEQUENTIAL_BOUNDARIES')
    times = [o['observed_seconds'] for o in observations]
    if not all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in times):
        raise ValueError('NONFINITE_BOUNDARY')
    if any(b < a for a, b in zip(times, times[1:])) or times[104] <= times[4]:
        raise ValueError('INVALID_OBSERVED_INTERVAL')
    rows = {}
    for match in ROW.finditer(log):
        step = int(match[1])
        values = [float(match[2]), float(match[3]), float(match[4])]
        if not all(math.isfinite(v) for v in values) or values[0] > 100:
            raise ValueError('NONFINITE_OR_BAD_LOGGED_LOSS')
        if step in rows:
            raise ValueError('DUPLICATE_PRINTED_STEP')
        rows[step] = {'step': step, 'loss': values[0], 'lrm': values[1],
                      'rounded_dt_seconds': values[2], 'tok_s': int(match[5].replace(',', ''))}
    if not all(step in rows for step in WINDOW):
        raise ValueError('MISSING_PREDECLARED_STEADY_SAMPLES')
    selected = [rows[step] for step in WINDOW]
    if any(r['tok_s'] <= 0 or r['lrm'] != 1.0 for r in selected):
        raise ValueError('NONPOSITIVE_SPEED_OR_DIFFERENT_LR_WINDOW')
    median = statistics.median(r['tok_s'] for r in selected)
    interval = times[104] - times[4]
    aggregate = 100 * 262144 / interval
    return {'status': 'SPEED_SCREEN_PASS_PENDING_REVIEW' if median >= THRESHOLD else 'SPEED_SCREEN_BELOW_THRESHOLD',
            'control_median_tok_s': CONTROL_MEDIAN, 'threshold_tok_s': THRESHOLD,
            'candidate_median_tok_s': median, 'median_relative_gain': median / CONTROL_MEDIAN - 1,
            'candidate_observed_100step_seconds': interval,
            'candidate_observed_100step_tok_s': aggregate,
            'aggregate_comparison': 'Sanity diagnostic only; no speed veto against historical printed median. A matched control aggregate is unavailable.',
            'aggregate_vs_printed_candidate_ratio': aggregate / median,
            'sanity_flags': ['AGGREGATE_AND_PRINTED_CANDIDATE_TIMING_DIVERGE_REVIEW_STALLS'] if aggregate < .8 * median else [],
            'observed_uncapped_startup_seconds': times[4], 'cold_proven': False,
            'window': selected, 'quality_evidence': False, 'automatic_full_run': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-log', required=True, type=Path)
    parser.add_argument('--rehearsal', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.candidate_log.read_text(), json.loads(args.rehearsal.read_text())), indent=2, allow_nan=False))
