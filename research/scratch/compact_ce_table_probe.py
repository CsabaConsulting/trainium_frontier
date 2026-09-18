#!/usr/bin/env python3
"""Device gate for the compact joint-function table, before loss integration."""
import argparse
import json

import torch

from compact_ce_table_study import make_table
from compact_ce_table_nki import build_kernel
from mlp_fused_bf16_probe import timing


def emit(kind, record):
    print(kind + ' ' + json.dumps(record), flush=True)


def expected(x, table):
    codes = x.view(torch.int16).int() & 65535
    magnitude = codes & 32767
    indices = (magnitude.clamp(table['first'], table['end'] - 1) - table['first']
               + (codes >> 15) * table['per_sign'])
    e = torch.tensor(table['exps'])[indices]
    d = torch.tensor(table['derivatives'])[indices]
    tiny = magnitude < table['first']
    return torch.where(tiny, 1.0 + x.float(), e), torch.where(tiny, 1.0, d)


def truth(x):
    tangent = (x.double() / 15.0).tanh()
    return (15.0 * tangent).exp(), (1.0 - tangent) * (1.0 + tangent)


def errors(actual, reference):
    return dict(exponential_relative=float(((actual[0].double() - reference[0]).abs()
                                            / reference[0]).max()),
                derivative_absolute=float((actual[1].double() - reference[1]).abs().max()))


def gate(name, x, table, device_table, kernel, lanes, full_reference=True):
    xd = x.to('neuron')
    outputs = tuple(v.cpu() for v in kernel[lanes](xd, device_table, True, lanes))
    repeat = tuple(v.cpu() for v in kernel[lanes](xd, device_table, True, lanes))
    target = expected(x, table)
    exact = [torch.equal(a, b) for a, b in zip(outputs, target)]
    repeated = [torch.equal(a, b) for a, b in zip(outputs, repeat)]
    finite = all(bool(torch.isfinite(v).all()) for v in outputs)
    # Exhaustive code checks precede the large tensor. The real tensor uses
    # all-coordinate table/repeat checks and stratified FP64 truth samples.
    ref_x = x if full_reference else x[::127, ::61]
    compared = outputs if full_reference else tuple(v[::127, ::61] for v in outputs)
    reference = truth(ref_x)
    measured = errors(compared, reference)
    passed = (all(exact) and all(repeated) and finite
              and measured['exponential_relative'] <= 2e-7
              and measured['derivative_absolute'] <= 1e-7
              and torch.equal(xd.cpu(), x))
    record = dict(name=name, shape=list(x.shape), lanes=lanes, finite=finite,
                  table_and_tail_exact=exact, repeat_exact=repeated,
                  fp64_all_coordinates=full_reference, errors=measured,
                  input_unchanged=torch.equal(xd.cpu(), x), passed=bool(passed))
    emit('COMPACT_TABLE_GATE', record)
    assert passed, record
    baseline = tuple(v.cpu() for v in kernel[lanes](xd, device_table, False, lanes))
    compared_baseline = baseline if full_reference else tuple(v[::127, ::61] for v in baseline)
    emit('COMPACT_TABLE_SCALAR_DIAGNOSTIC', dict(name=name,
         errors=errors(compared_baseline, reference),
         finite=all(bool(torch.isfinite(v).all()) for v in baseline),
         baseline_is_existing_scalar_math=True, quality_gate=False))
    # Change a boundary row and an interior row, leaving every other row exact.
    changed = x.clone()
    changed[127] = -changed[127] + 3.0
    changed[128] = -changed[128] - 3.0
    changed_outputs = tuple(v.cpu() for v in kernel[lanes](changed.to('neuron'), device_table, True, lanes))
    keep = torch.ones(x.shape[0], dtype=torch.bool)
    keep[127:129] = False
    independent = all(torch.equal(a[keep], b[keep]) for a, b in zip(outputs, changed_outputs))
    changed_target = expected(changed, table)
    changed_exact = all(torch.equal(a, b) for a, b in zip(changed_outputs, changed_target))
    emit('COMPACT_TABLE_INDEPENDENCE', dict(name=name, passed=independent and changed_exact,
                                          unchanged_rows_exact=independent,
                                          changed_outputs_exact=changed_exact))
    assert independent and changed_exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-shape', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    table = make_table(-12)
    assert len(table['exps']) == 5120
    cpu_table = torch.tensor([table['exps'], table['derivatives']], dtype=torch.float32)
    cpu_table = cpu_table.unsqueeze(0).repeat(128, 1, 1)
    device_table = cpu_table.to('neuron')
    kernel = wrap_nki(build_kernel())
    emit('COMPACT_TABLE_CONTRACT', dict(finite_bf16_codes=65280,
         exp_relative_bound=2e-7, derivative_absolute_bound=1e-7,
         replicated_table_bytes=cpu_table.numel() * cpu_table.element_size(),
         no_fit_to_training_or_validation_data=True, softcap=15,
         loss_reduction_log_reciprocal_and_autograd_not_tested=True))
    codes = torch.arange(65536, dtype=torch.int32)
    codes[(codes & 32640) == 32640] = 0  # replace all NaN/Inf encodings only
    exhaustive = codes.to(torch.uint16).view(torch.bfloat16).repeat(4).reshape(512, 512)
    gate('all_finite_codes', exhaustive, table, device_table, kernel, 2)
    generator = torch.Generator().manual_seed(9474)
    for rows, lanes in ((256, 1), (256, 2), (512, 2), (1024, 2)):
        x = (torch.randn(rows, 512, generator=generator) * 6.0).bfloat16()
        gate(f'random_rows{rows}_lanes{lanes}', x, table, device_table, kernel, lanes)
    assert torch.equal(device_table.cpu(), cpu_table)
    if args.real_shape:
        x = (torch.randn(16384, 8192, generator=generator) * 6.0).bfloat16()
        gate('real_logits_shape', x, table, device_table, kernel, 2, False)
        xd = x.to('neuron')
        arms = {
            'scalar_tanh_exp_derivative': lambda: kernel[2](xd, device_table, False, 2),
            'compact_joint_table': lambda: kernel[2](xd, device_table, True, 2),
        }
        measured = timing(arms, args.steps, torch.neuron.synchronize)
        emit('COMPACT_TABLE_BENCH', dict(shape=list(x.shape), **measured,
             table_load_and_index_gather_tail_and_output_costs_included=True,
             units='ms per joint primitive call', whole_loss_speed_claim=False))
    emit('COMPACT_TABLE_RESULT', dict(passed=True, real_shape=args.real_shape,
                                    training_promotion=False))


if __name__ == '__main__':
    main()
