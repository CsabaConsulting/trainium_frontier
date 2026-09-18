#!/usr/bin/env python3
"""Gate four shared embedding tables through compiled autograd and accumulation."""
import argparse
import copy
import json

import torch

from embedding_compact_autograd_check import Probe
from embedding_segmented_dma_autograd import install
from embedding_segmented_probe import check, gradient_for, indices_for, reference


def run(device, tokens, channels, vocabulary):
    torch.manual_seed(831)
    baseline = Probe(vocabulary, channels).to(device)
    candidate = copy.deepcopy(baseline)
    before_rng = torch.get_rng_state().clone()
    state = install(candidate, device)
    assert torch.equal(before_rng, torch.get_rng_state())
    if device == "neuron":
        control = torch.compile(baseline, backend="neuron", fullgraph=True, dynamic=False)
        optimized = torch.compile(candidate, backend="neuron", fullgraph=True, dynamic=False)
    else:
        control, optimized = baseline, candidate
    first = indices_for("training" if tokens == 8192 else "zipf", tokens, vocabulary, 731)
    cases = [("initial", first, "uniform"),
             ("changed_ids", (first + 17) % vocabulary, "uniform"),
             ("uniform", indices_for("uniform", tokens, vocabulary, 734), "uniform"),
             ("one_id", indices_for("one_id", tokens, vocabulary, 731), "uniform"),
             ("cancellation", indices_for("cancellation", tokens, vocabulary, 731), "cancellation"),
             ("zero_after_nonzero", first, "zero"),
             ("unsupported_shape", first[:124], "uniform"),
             ("return_supported", first, "uniform")]
    for number, (name, ids, kind) in enumerate(cases):
        count = ids.numel()
        shape = (4, count // 4)
        x = ids.reshape(shape).to(device)
        gradient = gradient_for(kind, count, channels, 840 + number)
        baseline.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        state.prepare(x)
        assert state.enabled == (name != "unsupported_shape")
        expected_output, output = control(x), optimized(x)
        torch.testing.assert_close(output.cpu(), expected_output.cpu(), rtol=0, atol=0)
        gd = gradient.reshape(*shape, channels).to(device)
        expected_output.backward(gd)
        output.backward(gd)
        max_difference = 0.0
        for i, ((key, p), (other_key, q)) in enumerate(zip(
                candidate.named_parameters(), baseline.named_parameters())):
            assert key == other_key and p.grad.dtype == q.grad.dtype == torch.float32
            g = gradient * (1.0 if i == 0 else 0.5 ** i)
            if device == "neuron":
                g = g.bfloat16().float()
            expected, bound, counts = reference(g, ids, vocabulary)
            check(p.grad, expected, bound, counts, f"dma_autograd:{name}:{key}:{tokens}:{channels}")
            check(q.grad, expected, bound, counts, f"native:{name}:{key}:{tokens}:{channels}")
            difference = (p.grad.cpu() - q.grad.cpu()).abs()
            max_difference = max(max_difference, float(difference.max()))
            assert torch.all(difference.double() <= 2 * bound)
        print("DMA_EMBED_AUTOGRAD_GATE " + json.dumps(dict(
            case=name, tokens=count, channels=channels, vocabulary=vocabulary,
            segmented_enabled=state.enabled, forward_exact=True, passed=True,
            max_abs_gradient_difference=max_difference)), flush=True)
    # Two outstanding forwards save their own metadata even when prepare()
    # changes the shared state before either backward is invoked.
    baseline.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    outputs, controls, grads = [], [], []
    for j in range(2):
        xx = ((first + 29 * j) % vocabulary).reshape(4, tokens // 4).to(device)
        state.prepare(xx)
        outputs.append(optimized(xx))
        controls.append(control(xx))
        grads.append(gradient_for("uniform", tokens, channels, 950+j).reshape(
            4, tokens // 4, channels).to(device))
    for y, z, g in zip(outputs, controls, grads):
        y.backward(g)
        z.backward(g)
    for p, q in zip(candidate.parameters(), baseline.parameters()):
        torch.testing.assert_close(p.grad.cpu(), q.grad.cpu(), rtol=2e-5, atol=2e-6)
    candidate.eval()
    baseline.eval()
    with torch.no_grad():
        xx = first[:128].reshape(2, 64).to(device)
        torch.testing.assert_close(candidate(xx).cpu(), baseline(xx).cpu(), rtol=0, atol=0)
    restored = Probe(vocabulary, channels).to(device)
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(xx).cpu(), candidate(xx).cpu(), rtol=0, atol=0)
    assert state.fallback_calls == 1 and state.segmented_calls == len(cases) + 1
    print("DMA_EMBED_AUTOGRAD_STATE " + json.dumps(dict(
        **state.statistics(), outstanding_metadata_and_accumulation_passed=True,
        eval_exact=True, strict_native_checkpoint_load_passed=True)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "neuron"), default="neuron")
    ap.add_argument("--real-shape", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.device == "neuron":
        import torch_neuronx  # noqa:F401
    run(args.device, 256, 256, 256)
    if args.real_shape:
        run(args.device, 8192, 1024, 8192)
    print("PASS: shared DMA embedding autograd, accumulation, fallback and checkpoint gates", flush=True)


if __name__ == "__main__":
    main()
