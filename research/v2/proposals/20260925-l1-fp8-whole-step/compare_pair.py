#!/usr/bin/env python3
"""Read two existing whole-step receipts; never launch or modify a run.

Exit 0: timing PASS; 1: timing FAIL; 2: polling INCONCLUSIVE; 3: invalid input.
Every result still requires external source/argv/terminal/budget/correctness review.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

SOURCE_SHA = '921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858'
THRESHOLD = 0.03


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite number')
    return float(value)


def close(actual, expected):
    if not math.isclose(number(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('receipt summary disagrees with raw timing')


def validate(row, arm):
    if row['schema'] != 'l1-whole-step-observation-v1' or row['arm'] != arm:
        raise ValueError('wrong receipt schema or arm order')
    if row['train_sha256'] != SOURCE_SHA:
        raise ValueError('wrong exact source')
    if row['warmup'] != 20 or row['measured_steps'] != 80:
        raise ValueError('require fixed 20 warmup plus 80 measured steps')
    for key in ('host', 'label', 'train_py'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError('missing identity field: ' + key)
    events = row['events']
    if len(events) != 100 or [e['step'] for e in events] != list(range(100)):
        raise ValueError('require exactly 100 contiguous boundaries')
    times = [number(e['monotonic']) for e in events]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('nonmonotonic boundary stream')
    if any(e['observed_batch'] != 1 for e in events[19:]):
        raise ValueError('coalesced measured boundary')
    intervals = [times[i] - times[i-1] for i in range(20, 100)]
    if min(intervals) <= 0:
        raise ValueError('nonpositive complete-step interval')
    mean = (times[99] - times[19]) / 80
    close(row['measured_seconds'], mean * 80)
    close(row['seconds_per_step'], mean)
    poll = number(row['poll_seconds'])
    gap = number(row['max_observed_poll_gap_seconds'])
    if poll != 0.005 or gap < poll:
        raise ValueError('wrong polling policy or invalid poll bound')
    error = 2 * gap / 80
    close(row['timing_uncertainty_seconds_per_step'], error)
    if not 0 <= error < mean:
        raise ValueError('invalid endpoint uncertainty')
    ordered = sorted(intervals)

    def quantile(p):
        h = 79 * p
        lo = int(h)
        return ordered[lo] + (h-lo) * (ordered[min(lo+1, 79)] - ordered[lo])

    distribution = dict(label=row['label'], n=80, mean=mean,
        median=statistics.median(intervals), population_stddev=statistics.pstdev(intervals),
        min=min(intervals), p10=quantile(.1), p90=quantile(.9), p95=quantile(.95),
        max=max(intervals), block10_means=[statistics.mean(intervals[i:i+10]) for i in range(0, 80, 10)])
    return mean, error, distribution


def compare(off, on):
    off_mean, off_error, off_dist = validate(off, 'off')
    on_mean, on_error, on_dist = validate(on, 'on')
    if off['host'] != on['host'] or off['train_py'] != on['train_py']:
        raise ValueError('host or absolute source path differs')
    if off['label'] == on['label']:
        raise ValueError('distinct labels required')
    if off['events'][-1]['monotonic'] >= on['events'][0]['monotonic']:
        raise ValueError('boundary chronology is not OFF then ON')
    point = 1 - on_mean / off_mean
    lower = 1 - (on_mean + on_error) / (off_mean - off_error)
    upper = 1 - (on_mean - on_error) / (off_mean + off_error)
    verdict = 'PASS' if lower >= THRESHOLD else ('FAIL' if upper < THRESHOLD else 'INCONCLUSIVE')
    return dict(timing_verdict=verdict, threshold=THRESHOLD,
        step_time_reduction=point, conservative_step_time_reduction=lower,
        optimistic_step_time_reduction=upper,
        distributions_seconds=[off_dist, on_dist],
        repeat_policy='Only concrete uncertainty; never average a valid sub-3-percent failure into promotion.',
        uncertainty_scope='Endpoint polling only, not a statistical confidence interval.',
        external_argv_environment_cache_chronology_budget_terminal_and_native_correctness_review_required=True,
        bp_benefit_proven=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('off_receipt')
    parser.add_argument('on_receipt')
    args = parser.parse_args()
    try:
        result = compare(json.loads(Path(args.off_receipt).read_text()),
                         json.loads(Path(args.on_receipt).read_text()))
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        print(json.dumps(dict(timing_verdict='INVALID', reason=str(exc))))
        return 3
    print(json.dumps(result, indent=2))
    return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['timing_verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
