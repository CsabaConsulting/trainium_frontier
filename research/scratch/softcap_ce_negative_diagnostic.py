#!/usr/bin/env python3
"""Preserve and isolate tiny negative losses already present in native CE.

Diagnostic only: do not alter the original head/loss gate or production math.
Expose the target affine argument, target exponential, denominator and log,
plus runtime exp(0)/log(1) controls to distinguish rounding and approximation.
"""
import json

import torch

from softcap_ce_parallel_nki import build_kernels


def build_diagnostic():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def diagnostic(z, target, controls):
        tokens, vocab = z.shape
        assert tokens % 256 == 0 and tokens >= 512 and vocab % 512 == 0
        assert z.dtype == target.dtype == controls.dtype == nl.float32
        assert target.shape == (tokens, 1) and controls.shape == (tokens, 2)
        result = nl.ndarray((tokens, 11), dtype=nl.float32, buffer=nl.shared_hbm)
        for tile in nl.affine_range(tokens // 256):
            rows = nl.ds((2 * tile + nl.program_id(0)) * 128, 128)
            logits = nl.ndarray((128, vocab), dtype=nl.float32, buffer=nl.sbuf)
            capped = nl.ndarray((128, vocab), dtype=nl.float32, buffer=nl.sbuf)
            labels = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            indices = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            target_unit = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            target_value = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            negative_target = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=logits, src=z[rows, :])
            nisa.dma_copy(dst=labels, src=target[rows, :])
            nisa.activation(dst=capped, data=logits, op=nl.tanh, scale=1.0 / 15.0)
            nisa.tensor_copy(dst=indices, src=labels)
            nisa.nc_n_gather(dst=target_unit, data=capped, indices=indices)
            nisa.tensor_scalar(dst=target_value, data=target_unit, op0=nl.multiply, operand0=15.0)
            nisa.tensor_scalar(dst=negative_target, data=target_unit, op0=nl.multiply, operand0=-15.0)
            partial = nl.ndarray((128, vocab // 512), dtype=nl.float32, buffer=nl.sbuf)
            for vi in nl.affine_range(vocab // 512):
                columns = nl.ds(vi * 512, 512)
                nisa.activation(dst=logits[:, columns], data=capped[:, columns], op=nl.exp,
                                scale=15.0, bias=negative_target, reduce_op=nl.add,
                                reduce_res=partial[:, nl.ds(vi, 1)],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            target_exp = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            denominator = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            logarithm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            lse = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_n_gather(dst=target_exp, data=logits, indices=indices)
            nisa.tensor_reduce(dst=denominator, data=partial, op=nl.add, axis=(1,))
            nisa.activation(dst=logarithm, data=denominator, op=nl.log)
            nisa.tensor_tensor(dst=lse, data1=target_value, data2=logarithm, op=nl.add)
            affine = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            separate_argument = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            separate_exp = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            zero = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            one = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            exp_zero = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            log_one = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=affine, data=target_unit, op=nl.copy,
                            scale=15.0, bias=negative_target)
            nisa.tensor_tensor(dst=separate_argument, data1=target_value,
                               data2=negative_target, op=nl.add)
            nisa.activation(dst=separate_exp, data=separate_argument, op=nl.exp)
            nisa.dma_copy(dst=zero, src=controls[rows, 0:1])
            nisa.dma_copy(dst=one, src=controls[rows, 1:2])
            nisa.activation(dst=exp_zero, data=zero, op=nl.exp)
            nisa.activation(dst=log_one, data=one, op=nl.log)
            stats = nl.ndarray((128, 11), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=stats[:, 0:1], src=target_unit)
            nisa.tensor_copy(dst=stats[:, 1:2], src=target_value)
            nisa.tensor_copy(dst=stats[:, 2:3], src=negative_target)
            nisa.tensor_copy(dst=stats[:, 3:4], src=affine)
            nisa.tensor_copy(dst=stats[:, 4:5], src=target_exp)
            nisa.tensor_copy(dst=stats[:, 5:6], src=denominator)
            nisa.tensor_copy(dst=stats[:, 6:7], src=logarithm)
            nisa.tensor_copy(dst=stats[:, 7:8], src=lse)
            nisa.tensor_copy(dst=stats[:, 8:9], src=exp_zero)
            nisa.tensor_copy(dst=stats[:, 9:10], src=log_one)
            nisa.tensor_copy(dst=stats[:, 10:11], src=separate_exp)
            nisa.dma_copy(dst=result[rows, :], src=stats)
        return result

    return diagnostic


def main():
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    diagnostic = wrap_nki(build_diagnostic())
    original, _ = (wrap_nki(kernel) for kernel in build_kernels())
    names = ['target_unit', 'target_value', 'negative_target', 'fused_affine_argument',
             'target_exp', 'denominator', 'loss', 'lse', 'runtime_exp_zero',
             'runtime_log_one', 'separately_centered_exp']
    negative_cases = 0
    with torch.no_grad():
        for vocab in (1024, 8192):
            for strength in (0.0, 8.0, 30.0, 80.0, 300.0):
                z = torch.full((512, vocab), -strength)
                z[:, 0] = strength
                target = torch.zeros(512, 1)
                controls = torch.stack((torch.zeros(512), torch.ones(512)), dim=1)
                inputs = tuple(t.to('neuron') for t in (z, target, controls))
                reference = tuple(t.cpu() for t in original[2](inputs[0], inputs[1], 2, True))
                stats = diagnostic[2](*inputs).cpu()
                assert bool(torch.isfinite(stats).all())
                exact = [torch.equal(stats[:, 6:7], reference[0]),
                         torch.equal(stats[:, 7:8], reference[1])]
                record = dict(vocab=vocab, strength=strength,
                              exact_original_loss_lse=exact,
                              all_rows_identical=bool((stats == stats[0]).all()),
                              stages=dict(zip(names, stats[0].tolist())),
                              stage_bits={name: int(value) for name, value in
                                          zip(names, stats[0].view(torch.int32).tolist())})
                u, bias = stats[0, 0].double(), stats[0, 2].double()
                record['fp64_affine_from_actual_fp32_operands'] = float(15.0 * u + bias)
                record['fp64_exp_of_actual_affine'] = float((15.0 * u + bias).exp())
                record['fp64_log_of_actual_denominator'] = float(stats[0, 5].double().log())
                capped = (z[0].double() / 15.0).tanh() * 15.0
                record['independent_fp64_loss'] = float((capped - capped[0]).exp().sum().log())
                record['negative_original_loss_rows'] = int((reference[0] < 0).sum())
                negative_cases += int(record['negative_original_loss_rows'] > 0)
                print('CE_NEGATIVE_STAGE_DIAGNOSTIC ' + json.dumps(record), flush=True)
                assert all(exact), 'Instrumentation changed the reference CE result'
                assert torch.equal(inputs[0].cpu(), z) and torch.equal(inputs[2].cpu(), controls)
    print('CE_NEGATIVE_DIAGNOSTIC_RESULT ' + json.dumps(dict(
        cases=10, negative_original_cases=negative_cases,
        original_failed_head_gate_retained=True,
        diagnostic_only=True, kernel_math_or_gate_changed=False)), flush=True)


if __name__ == '__main__':
    main()
