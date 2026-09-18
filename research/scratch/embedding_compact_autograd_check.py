#!/usr/bin/env python3
"""Validate shared metadata through actual compiled forward/backward calls.

Exercises changed IDs, repeated IDs, zeros, native overflow fallback and return
to compaction, then checks ordinary evaluation and checkpoint compatibility.
No training speed or model-quality claim.
"""
import argparse
import copy
import json

import torch

from embedding_compact_autograd import install
from embedding_segmented_probe import check, gradient_for, indices_for, reference


class Probe(torch.nn.Module):
    def __init__(self, vocabulary, channels):
        super().__init__()
        self.transformer = torch.nn.ModuleDict({"wte": torch.nn.Embedding(vocabulary, channels)})
        self.value_embeds = torch.nn.ModuleDict({str(i): torch.nn.Embedding(vocabulary, channels)
                                                for i in range(3)})

    def forward(self, ids):
        result = self.transformer.wte(ids)
        for i, table in enumerate(self.value_embeds.values()):
            result = result + table(ids) * (0.5 ** (i + 1))
        return result


def run(device, tokens, channels, vocabulary, bucket):
    torch.manual_seed(831)
    baseline = Probe(vocabulary, channels).to(device)
    candidate = copy.deepcopy(baseline)
    before_rng = torch.get_rng_state().clone()
    state = install(candidate, device, bucket)
    assert torch.equal(before_rng, torch.get_rng_state())
    if device == "neuron":
        control = torch.compile(baseline, backend="neuron", fullgraph=True, dynamic=False)
        optimized = torch.compile(candidate, backend="neuron", fullgraph=True, dynamic=False)
    else:
        control, optimized = baseline, candidate
    # This is the exact real input layout of four packed2048-token sequences.
    shape = (4, tokens // 4)
    compact_kind = "training" if tokens == 8192 else "zipf"
    first_ids = indices_for(compact_kind, tokens, vocabulary, 731)
    cases = [
        ("compact", first_ids, "uniform"),
        ("changed_ids", (first_ids + 17) % vocabulary, "uniform"),
        ("native_overflow", indices_for("uniform", tokens, vocabulary, 734), "uniform"),
        ("return_compact", first_ids, "uniform"),
        ("one_id", indices_for("one_id", tokens, vocabulary, 731), "uniform"),
        ("zero", first_ids, "zero"),
    ]
    if tokens > bucket and vocabulary > bucket:
        cases.extend([
            ("bucket_boundary", torch.arange(tokens).int() % bucket, "uniform"),
            ("boundary_overflow", torch.arange(tokens).int() % (bucket + 1), "uniform"),
        ])
    for number, (name, ids, grad_kind) in enumerate(cases):
        x = ids.reshape(shape).to(device)
        g = gradient_for(grad_kind, tokens, channels, 840 + number)
        baseline.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        state.prepare(x)
        if name in ("native_overflow", "boundary_overflow"):
            assert not state.enabled
        else:
            assert state.enabled
        expected_output, output = control(x), optimized(x)
        torch.testing.assert_close(output.cpu(), expected_output.cpu(), rtol=0, atol=0)
        gd = g.reshape(*shape, channels).to(device)
        expected_output.backward(gd)
        output.backward(gd)
        max_difference = 0.0
        for i, ((key, p), (other_key, q)) in enumerate(zip(
                candidate.named_parameters(), baseline.named_parameters())):
            assert key == other_key
            coefficient = 1.0 if i == 0 else 0.5 ** i
            effective_g = g * coefficient
            if device == "neuron":
                effective_g = effective_g.bfloat16().float()
            expected, bound, counts = reference(effective_g, ids, vocabulary)
            check(p.grad, expected, bound, counts, f"autograd:{name}:{key}:{tokens}:{channels}")
            check(q.grad, expected, bound, counts, f"native:{name}:{key}:{tokens}:{channels}")
            difference = (p.grad.cpu() - q.grad.cpu()).abs()
            max_difference = max(max_difference, float(difference.max()))
            assert torch.all(difference.double() <= 2 * bound)
        print("EMBED_AUTOGRAD_GATE " + json.dumps(dict(
            case=name, tokens=tokens, channels=channels, vocabulary=vocabulary,
            compact_enabled=state.enabled, max_abs_gradient_difference=max_difference,
            forward_exact=True, passed=True)), flush=True)
    # Eval bypasses training metadata and accepts a different batch/context.
    candidate.eval()
    baseline.eval()
    with torch.no_grad():
        xx = first_ids[:128].reshape(2, 64).to(device)
        torch.testing.assert_close(candidate(xx).cpu(), baseline(xx).cpu(), rtol=0, atol=0)
    restored = Probe(vocabulary, channels).to(device)
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(xx).cpu(), candidate(xx).cpu(), rtol=0, atol=0)
    expected_fallback = sum(name in ("native_overflow", "boundary_overflow") for name, _, _ in cases)
    assert state.compact_calls == len(cases) - expected_fallback
    assert state.fallback_calls == expected_fallback
    print("EMBED_AUTOGRAD_STATE " + json.dumps(state.statistics()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "neuron"), default="neuron")
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--bucket", type=int, default=4096)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.device == "neuron":
        import torch_neuronx  # noqa: F401
    run(args.device, 256, 256, 256, 128)
    if args.real_shape:
        run(args.device, 8192, 1024, 8192, args.bucket)
    print("PASS: shared compact embedding autograd, fallback and checkpoint gates", flush=True)


if __name__ == "__main__":
    main()
