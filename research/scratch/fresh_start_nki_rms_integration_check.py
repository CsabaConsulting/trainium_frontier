#!/usr/bin/env python3
"""Exercise the actual opt-in table optimizer on native Neuron, against CPU math.

Use an idle host and an explicit NEURON_RT_VISIBLE_CORES. This checks mutation,
runtime LR/bias changes, two table shapes, zero/None/sparse/dense gradients and
two beta values. It is not a model-quality or whole-run speed measurement.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch_neuronx  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def run(beta):
    torch.manual_seed(829)
    eps = 1e-10
    shapes = ((4096, 128), (8192, 128))
    params = [torch.nn.Parameter((torch.randn(s) * 0.01).bfloat16().to("neuron"))
              for s in shapes]
    optimizer = T.MuonAdamW([dict(kind="rmsprop", params=params, lr=0.003,
                                betas=(0.0, beta), eps=eps, weight_decay=0.0)])
    moments = [torch.zeros(s) for s in shapes]
    steps = [0, 0]
    max_state_error = max_weight_error = 0.0
    for step in range(20):
        lr = (0.003, 0.00071, 0.01, 0.0)[step % 4]
        optimizer.param_groups[0]["lr"] = lr
        previous = [p.detach().cpu().clone() for p in params]
        gradients = []
        for index, p in enumerate(params):
            if index == 1 and step % 5 == 0:
                gradients.append(None)
                p.grad = None
                continue
            g = (torch.randn(p.shape) * 0.003).bfloat16()
            if step % 5 == 1:
                g.zero_()
            elif step % 3 != 0:
                g[torch.rand(p.shape) < 0.98] = 0
            if step == 7:
                g[0, :4] = torch.tensor([64.0, -64.0, 1e-6, -1e-6])
            gradients.append(g)
            p.grad = g.to("neuron")
        optimizer.step()
        torch.neuron.synchronize()
        for index, (p, g) in enumerate(zip(params, gradients)):
            state = optimizer.state[p]
            actual_p = p.detach().cpu()
            if g is None:
                assert torch.equal(actual_p, previous[index])
                assert state.get("step", 0) == steps[index]
                if steps[index]:
                    torch.testing.assert_close(state["exp_avg_sq"].cpu(), moments[index],
                                               rtol=3e-5, atol=1e-14)
                continue
            steps[index] += 1
            gf = g.float()
            moments[index] = (beta * moments[index].double()
                              + (1.0 - beta) * gf.square().double()).float()
            denom = (moments[index] / (1.0 - beta ** steps[index])).sqrt() + eps
            expected_p = (previous[index].float() - lr * gf / denom).bfloat16()
            actual_v = state["exp_avg_sq"].cpu()
            assert state["step"] == steps[index] and actual_v.dtype == torch.float32
            assert p.grad.dtype == torch.bfloat16 and torch.equal(p.grad.cpu(), g)
            torch.testing.assert_close(actual_v, moments[index], rtol=3e-5, atol=1e-14)
            torch.testing.assert_close(actual_p, expected_p, rtol=0.009, atol=2e-7)
            assert torch.isfinite(actual_v).all() and torch.isfinite(actual_p).all()
            if not g.any() or lr == 0:
                assert torch.equal(actual_p, previous[index])
            max_state_error = max(max_state_error, float((actual_v - moments[index]).abs().max()))
            max_weight_error = max(max_weight_error, float((actual_p.float() - expected_p.float()).abs().max()))
        print(json.dumps(dict(check="nki_rms_integration_step", beta=beta, step=step,
                              lr=lr, parameter_steps=steps)), flush=True)
    # Cached CPU/device scalar storage must remain a tiny runtime input, not a
    # table-shaped FP32 gradient allocation. Its contents must follow the last LR.
    for p in params:
        state = optimizer.state[p]
        assert state["nki_rms_scalars_cpu"].shape == (128, 2)
        assert state["nki_rms_scalars_device"].shape == (128, 2)
    print(json.dumps(dict(check="nki_rms_integration", status="PASS", beta=beta,
                          shapes=shapes, max_state_abs_error=max_state_error,
                          max_one_step_weight_abs_error=max_weight_error,
                          bit_exact_equivalence_claim=False)), flush=True)


def main():
    torch.set_num_threads(2)
    assert not T.USE_NKI_NGRAM_RMS and not T.NGRAM_RMS_FP32
    T.USE_NKI_NGRAM_RMS = True
    T.NGRAM_RMS_FP32 = True
    T.USE_FOREACH_OPTIM = True
    T.USE_TENSOR_LR_SCALARS = True
    assert T._HAS_NKI and T.current_toggles()["USE_NKI_NGRAM_RMS"]
    for beta in (0.999, 0.9):
        run(beta)
    print("PASS: actual NKI table optimizer integration", flush=True)


if __name__ == "__main__":
    main()
