#!/usr/bin/env python3
"""CPU domain study for a joint softcap-exponential/derivative BF16 table.

All finite BF16 codes are enumerated. Active exponent intervals use FP32 table
entries; tiny inputs use a low-degree polynomial and large inputs use saturation.
No training data or learned coefficients are used. CPU division/log are ideally
rounded FP32 in this study; device gather, reduction, reciprocal/log accuracy,
full backward, table loads and timing remain separate gates.
"""
import argparse
import json
import math
from pathlib import Path
import random
import struct


def f32(value):
    return struct.unpack('!f', struct.pack('!f', value))[0]


def value_from_code(code):
    return struct.unpack('!f', struct.pack('!I', code << 16))[0]


def bf16_code(value):
    bits = struct.unpack('!I', struct.pack('!f', value))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def exact_pair(value):
    tangent = math.tanh(value / 15.0)
    return math.exp(15.0 * tangent), (1.0 - tangent) * (1.0 + tangent)


def make_table(lower_exponent):
    first = (lower_exponent + 127) << 7
    end = (8 + 127) << 7  # magnitude256; tanh is saturated to FP32 accuracy
    exps, derivatives = [], []
    for sign in (0, 1):
        for code in range(first, end):
            exponential, derivative = exact_pair(value_from_code(code | (sign << 15)))
            exps.append(f32(exponential))
            derivatives.append(f32(derivative))
    return dict(first=first, end=end, per_sign=end - first,
                exps=exps, derivatives=derivatives)


def approximate_pair(code, table, degree):
    magnitude, sign = code & 0x7fff, code >> 15
    value = value_from_code(code)
    if table['first'] <= magnitude < table['end']:
        index = magnitude - table['first'] + sign * table['per_sign']
        return table['exps'][index], table['derivatives'][index]
    if magnitude >= table['end']:
        return f32(math.exp(-15.0 if sign else 15.0)), 0.0
    if degree == 1:
        return f32(1.0 + value), 1.0
    exponential = f32(1.0 + f32(value * f32(1.0 + f32(.5 * value))))
    derivative = f32(1.0 - f32(f32(value * value) * f32(1.0 / 225.0)))
    return exponential, derivative


def sum_f32(values):
    total = 0.0
    for value in values:
        total = f32(total + value)
    return total


def blocked_sum(values):
    return sum_f32([sum_f32(values[i:i + 512]) for i in range(0, len(values), 512)])


def stable_log1p(ratio):
    if ratio > .0625:
        return f32(math.log(f32(1.0 + ratio)))
    # Alternating degree6 polynomial; the omitted term is bounded by r**7/7.
    poly = f32(-1.0 / 6.0)
    for coefficient in (1.0 / 5.0, -1.0 / 4.0, 1.0 / 3.0, -.5, 1.0):
        poly = f32(coefficient + f32(ratio * poly))
    return f32(ratio * poly)


def row_gate(name, codes, target, table, degree):
    pairs = [approximate_pair(code, table, degree) for code in codes]
    exact = [exact_pair(value_from_code(code)) for code in codes]
    exponentials = [pair[0] for pair in pairs]
    others = exponentials.copy()
    others[target] = 0.0
    other_sum = blocked_sum(others)
    ratio = f32(other_sum / exponentials[target])
    loss = stable_log1p(ratio)
    total = f32(other_sum + exponentials[target])
    inverse = f32(1.0 / total)
    expected_ratio = math.fsum(pair[0] / exact[target][0]
                               for i, pair in enumerate(exact) if i != target)
    expected_loss = math.log1p(expected_ratio)
    expected_total = math.fsum(pair[0] for pair in exact)
    errors = []
    for i, ((exponential, derivative), (truth_exp, truth_derivative)) in enumerate(zip(pairs, exact)):
        if i == target:
            probability_gradient = f32(-ratio / f32(1.0 + ratio))
            truth = -expected_ratio / (1.0 + expected_ratio) * truth_derivative
        else:
            probability_gradient = f32(exponential * inverse)
            truth = truth_exp / expected_total * truth_derivative
        errors.append(abs(f32(probability_gradient * derivative) - truth))
    record = dict(name=name, tokens_in_vocabulary=len(codes), loss=loss,
                  fp64_loss=expected_loss, loss_abs_error=abs(loss - expected_loss),
                  gradient_max_abs_error=max(errors), ratio=ratio,
                  uses_target_exclusion_and_stable_log1p=True)
    record['passed'] = record['loss_abs_error'] <= 5e-5 and record['gradient_max_abs_error'] <= 2e-6
    if expected_loss < 1e-6:
        record['passed'] &= record['loss_abs_error'] <= 1e-12
    assert record['passed'], record
    return record


def study(lower_exponent, degree):
    table = make_table(lower_exponent)
    maxima = {'exponential_relative': 0.0, 'derivative_absolute': 0.0}
    worst = {}
    count = 0
    for code in range(65536):
        value = value_from_code(code)
        if not math.isfinite(value):
            continue
        actual, derivative = approximate_pair(code, table, degree)
        expected, truth_derivative = exact_pair(value)
        errors = dict(exponential_relative=abs(actual - expected) / expected,
                      derivative_absolute=abs(derivative - truth_derivative))
        for field, error in errors.items():
            if error > maxima[field]:
                maxima[field] = error
                worst[field] = dict(code=code, value=value, error=error)
        count += 1
    assert maxima['exponential_relative'] <= 2e-7, maxima
    assert maxima['derivative_absolute'] <= 1e-7, maxima
    rows = []
    cases = [('uniform_zero', [0.0] * 8192, 0), ('uniform_point7', [.7] * 8192, 8191)]
    for name, low, high in [('confident', 0., 500.), ('extreme_confident', -500., 500.),
                             ('confident_30', -30., 30.), ('confident_wrong', 0., -500.)]:
        values = [low] * 8192
        values[4095] = high
        if name == 'confident_wrong':
            values[4096] = 500.
        cases.append((name, values, 4095))
    generator = random.Random(9197)
    for scale in (.001, .1, 1., 3., 6., 15., 100.):
        cases.append((f'random_scale_{scale}', [generator.gauss(0., scale) for _ in range(8192)], 117))
    for name, values, target in cases:
        rows.append(row_gate(name, [bf16_code(x) for x in values], target, table, degree))
    return dict(lower_exponent=lower_exponent, small_exponential_degree=degree,
                active_magnitude_interval=[2.0**lower_exponent, 256.0], finite_bf16_codes=count,
                entries_per_function=len(table['exps']), fp32_fields_per_entry=2,
                replicated_table_bytes=128 * len(table['exps']) * 2 * 4,
                maxima=maxima, worst=worst, rows=rows, passed=True,
                device_accuracy_or_speed_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [study(lower, degree) for lower, degree in [(-8, 2), (-12, 1), (-16, 1)]]
    report = dict(contract='finite BF16 logits; softcap15; FP32 table and tail arithmetic',
                  exp_relative_bound=2e-7, derivative_absolute_bound=1e-7,
                  cpu_only=True, device_division_log_gather_and_speed_not_measured=True,
                  results=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for result in results:
        print(json.dumps({k: result[k] for k in ['lower_exponent', 'small_exponential_degree',
              'finite_bf16_codes', 'replicated_table_bytes', 'maxima', 'passed']}), flush=True)


if __name__ == '__main__':
    main()
