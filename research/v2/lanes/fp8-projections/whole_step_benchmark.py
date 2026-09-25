#!/usr/bin/env python3
"""Observe unchanged production step boundaries and compare OFF/ON/ON/OFF.

No launcher, host contact, training-loop patch, or evaluator modification.
The distinct Operator starts an approved, uniquely labelled production run
and this observer under the existing containment/preflight protocol.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time


def observe(args):
    path, output = Path(args.step_log), Path(args.out)
    if path.exists() or output.exists():
        raise ValueError('fresh absent step-log and receipt paths are required')
    if not 20 <= args.warmup < args.steps or args.steps - args.warmup < 60:
        raise ValueError('require >=20 warmup and >=60 measured steps')
    if not 0.001 <= args.poll_seconds <= 0.01:
        raise ValueError('poll interval must be between 1 and 10 ms')
    source = Path(args.train_py).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    started = time.monotonic()
    previous_poll, max_poll_gap = started, args.poll_seconds
    events = []
    while len(events) < args.steps:
        now = time.monotonic()
        max_poll_gap = max(max_poll_gap, now - previous_poll)
        previous_poll = now
        if now - started > args.timeout:
            raise TimeoutError('step observer timed out; preserve log and diagnose')
        if path.exists():
            text = path.read_text()
            complete = text[:text.rfind('\n') + 1]
            steps = [int(line) for line in complete.splitlines()]
            if steps != list(range(len(steps))):
                raise ValueError('non-contiguous, duplicated, or unexpected step boundaries')
            if len(steps) < len(events):
                raise ValueError('step log was truncated')
            observed = time.monotonic()
            max_poll_gap = max(max_poll_gap, observed - now)
            for step in steps[len(events):]:
                events.append(dict(step=step, monotonic=observed, observed_batch=len(steps) - len(events)))
        time.sleep(args.poll_seconds)
    left, right = events[args.warmup - 1], events[args.steps - 1]
    intervals = args.steps - args.warmup
    measured_seconds = right['monotonic'] - left['monotonic']
    # A burst means the observer was stalled. Reject it in the measured range.
    if any(e['observed_batch'] != 1 for e in events[args.warmup - 1:args.steps]):
        raise ValueError('observer coalesced boundaries; timing receipt unusable')
    record = dict(schema='l1-whole-step-observation-v1', arm=args.arm, label=args.label,
        host=platform.node(), train_py=str(source), train_sha256=digest,
        warmup=args.warmup, measured_steps=intervals, measured_seconds=measured_seconds,
        seconds_per_step=measured_seconds / intervals, poll_seconds=args.poll_seconds,
        max_observed_poll_gap_seconds=max_poll_gap,
        timing_uncertainty_seconds_per_step=2 * max_poll_gap / intervals,
        events=events, scope='complete production step boundary intervals',
        requires_operator_terminal_and_checkpoint_review=True)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'events'}, indent=2))


def compare(args):
    rows = [json.loads(Path(path).read_text()) for path in args.receipts]
    if [r['arm'] for r in rows] != ['off', 'on', 'on', 'off']:
        raise ValueError('provide the chronological OFF ON ON OFF receipts')
    for key in ('schema', 'host', 'train_sha256', 'warmup', 'measured_steps'):
        if len({r[key] for r in rows}) != 1:
            raise ValueError('incomparable ' + key)
    if len({r['label'] for r in rows}) != 4:
        raise ValueError('four unique run labels required')
    for row in rows:
        speed, error = row['seconds_per_step'], row['timing_uncertainty_seconds_per_step']
        if not math.isfinite(speed) or not math.isfinite(error) or not 0 <= error < speed:
            raise ValueError('invalid timing observation')
    pairs = [(rows[0], rows[1]), (rows[3], rows[2])]
    gains = [1 - on['seconds_per_step'] / off['seconds_per_step'] for off, on in pairs]
    lower = [1 - (on['seconds_per_step'] + on['timing_uncertainty_seconds_per_step']) /
                 (off['seconds_per_step'] - off['timing_uncertainty_seconds_per_step'])
             for off, on in pairs]
    result = dict(timing_verdict='PASS' if min(lower) >= 0.03 else 'FAIL',
                  pair_speedup_fractions=gains, conservative_pair_speedup_fractions=lower,
                  mean_pair_speedup=statistics.mean(gains), threshold=0.03,
                  gate='each paired complete-step improvement >=3%, including observation uncertainty',
                  native_numerics_terminal_checkpoint_causality_and_exact_argv_review_required=True,
                  bp_benefit_proven=False)
    print(json.dumps(result, indent=2))
    return 0 if result['timing_verdict'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    p = modes.add_parser('observe')
    p.add_argument('--train-py', required=True)
    p.add_argument('--step-log', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--arm', choices=['off', 'on'], required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--poll-seconds', type=float, default=0.005)
    p.add_argument('--timeout', type=float, default=2400)
    p = modes.add_parser('compare')
    p.add_argument('receipts', nargs=4)
    args = parser.parse_args()
    return (observe if args.mode == 'observe' else compare)(args)


if __name__ == '__main__':
    raise SystemExit(main())
