#!/usr/bin/env python3
"""Gate mixed routing, all master gradients, cache refresh and native restore."""
import copy

import torch
import torch_neuronx  # noqa: F401

from mlp_fused_bf16_probe import compare
from mlp_fused_integration import install as install_native
from mlp_fused_integration_check import Probe, refresh, T
from nki_library_partial_mlp import install
from nki_library_probe_common import emit, exact


class SixLayerProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.ModuleDict({'h': torch.nn.ModuleList(
            [Probe(256).transformer.h[0] for _ in range(6)])})

    def forward(self, value):
        # Independent slices expose a wrong layer selection without hiding it
        # behind a sum. Full-model composition is a separate finite phase run.
        return torch.stack([block.mlp(value[index])
                            for index, block in enumerate(self.transformer.h)])


def run(selected):
    torch._dynamo.reset()
    torch.manual_seed(195021)
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = True
    T.USE_GATED_MLP = T.USE_COALESCED_SWIGLU = T.USE_NKI_RELU2 = False
    T.USE_PARALLEL_BLOCK = False
    T.RELU2_TAU = 0.0
    baseline = SixLayerProbe().to('neuron')
    candidate = copy.deepcopy(baseline)
    install_native(baseline, 'neuron', T, forward_backend='native')
    contract = install(candidate, 'neuron', T, selected)
    assert contract['library_layer_indices'] == list(selected)
    assert contract['modules'] == len(selected)
    native = torch.compile(baseline, backend='neuron', fullgraph=True, dynamic=False)
    optimized = torch.compile(candidate, backend='neuron', fullgraph=True, dynamic=False)
    x = torch.randn(6, 4, 64, 256).bfloat16().to('neuron')
    gradient = (torch.randn(x.shape) * .01).bfloat16().to('neuron')
    changed_caches = 0
    for cached in (False, True):
        previous = None
        for step in range(2):
            baseline.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            if cached:
                refresh(baseline)
                refresh(candidate)
                current = [b.mlp.c_fc._cached_weight.detach().cpu().clone()
                           for b in candidate.transformer.h]
                if previous is not None:
                    assert all(not torch.equal(a, b) for a, b in zip(previous, current))
                    changed_caches += 6
                previous = current
            for micro in range(2):
                value = x + micro * .03125
                bx = value.detach().clone().requires_grad_()
                cx = value.detach().clone().requires_grad_()
                expected, output = native(bx), optimized(cx)
                assert type(expected.grad_fn).__name__ == 'CompiledFunctionBackward'
                assert type(output.grad_fn).__name__ == 'CompiledFunctionBackward'
                name = f'partial:{selected}:{cached}:{step}:{micro}'
                compare(name + ':output', output, expected)
                expected.backward(gradient)
                output.backward(gradient)
                compare(name + ':dx', cx.grad, bx.grad)
            for (name, p), (other, q) in zip(candidate.named_parameters(), baseline.named_parameters()):
                assert name == other and p.dtype == q.dtype == torch.float32
                assert p.grad is not None and p.grad.dtype == q.grad.dtype == torch.float32
                compare(f'partial:{selected}:{cached}:{step}:{name}', p.grad, q.grad)
            with torch.no_grad():
                for p, q in zip(candidate.parameters(), baseline.parameters()):
                    q.add_(q.grad, alpha=-.001)
                    p.copy_(q)
    assert changed_caches == 6
    refresh(candidate)
    with torch.no_grad():
        original = optimized(x)
        changed = x.clone()
        changed[1:] *= -3
        exact(f'partial:{selected}:other_layers', optimized(changed)[0], original[0])
        changed = x.clone()
        changed[:, 1:] *= -2
        exact(f'partial:{selected}:other_batch', optimized(changed)[:, :1], original[:, :1])
        exact(f'partial:{selected}:repeat', optimized(x), original)
    candidate.eval()
    baseline.eval()
    refresh(candidate)
    refresh(baseline)
    restored = SixLayerProbe().to('neuron')
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval()
    refresh(restored)
    with torch.no_grad():
        short = x[:, :2, :31]
        exact(f'partial:{selected}:native_eval', candidate(short), baseline(short))
        exact(f'partial:{selected}:strict_restore', restored(short), candidate(short))
    emit('NKILIB_PARTIAL_MLP_GATE', dict(passed=True, selected=list(selected),
        total_layers=6, channels=256, tokens_per_layer=256,
        original_native_coordinate_and_rms_bounds=True,
        all_layers_compiled_forward_and_backward=True,
        all_fp32_master_gradients_and_two_microbatch_accumulation=True,
        cached_and_uncached=True, changed_caches=changed_caches,
        exact_layer_batch_independence_and_repeat=True,
        exact_native_evaluation_and_checkpoint=True, contract=contract,
        model_performance_or_quality_claim=False))


if __name__ == '__main__':
    torch.set_num_threads(4)
    for selected in ((), (0,), (5,), tuple(range(6))):
        run(selected)
    emit('NKILIB_PARTIAL_MLP_RESULT', dict(passed=True, cases=4,
        selected_layers=[[], [0], [5], list(range(6))],
        all_original_native_bounds_retained=True,
        model_performance_or_quality_claim=False))
