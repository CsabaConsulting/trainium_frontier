#!/usr/bin/env python3
"""Synthetic tests for timing decision math; no model, host or timing claim."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def main():
    path = Path(__file__).with_name('whole_step_benchmark.py')
    spec = importlib.util.spec_from_file_location('benchmark_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    verdicts = []
    with tempfile.TemporaryDirectory(prefix='l1-benchmark-math-') as tmp:
        receipts = [str(Path(tmp) / f'{i}.json') for i in range(4)]
        for gain, expected in [(0.04, 'PASS'), (0.02, 'FAIL'), (0.03, 'FAIL')]:
            for i, (arm, speed) in enumerate(zip(['off', 'on', 'on', 'off'],
                                                [0.5, 0.5 * (1-gain), 0.5 * (1-gain), 0.5])):
                record = dict(schema='l1-whole-step-observation-v1', host='synthetic', train_sha256='synthetic',
                    warmup=20, measured_steps=80, label=f'synthetic-{i}', arm=arm,
                    seconds_per_step=speed, timing_uncertainty_seconds_per_step=0.000125)
                Path(receipts[i]).write_text(json.dumps(record))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = mod.compare(SimpleNamespace(receipts=receipts))
            result = json.loads(output.getvalue())
            assert result['timing_verdict'] == expected
            assert rc == (0 if expected == 'PASS' else 1)
            verdicts.append(dict(injected_gain=gain, verdict=expected))
        row = json.loads(Path(receipts[1]).read_text())
        row['host'] = 'different'
        Path(receipts[1]).write_text(json.dumps(row))
        try:
            mod.compare(SimpleNamespace(receipts=receipts))
        except ValueError as exc:
            assert str(exc) == 'incomparable host'
        else:
            raise AssertionError('different hosts accepted')
    print(json.dumps(dict(verdict='PASS_SYNTHETIC_ONLY', cases=verdicts,
                          host_mismatch_rejected=True), indent=2))


if __name__ == '__main__':
    main()
