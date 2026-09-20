"""Gate actual cached/uncached Linear integration of the library training MLP."""
import argparse
import copy

import torch
import torch_neuronx  # noqa: F401

from mlp_fused_bf16_probe import compare
from mlp_fused_integration_check import Probe, refresh, T
from nki_library_mlp_integration import install
from nki_library_probe_common import emit, exact


def backward_node_names(output):
    pending, seen, names = [output.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        pending.extend(child for child, _ in node.next_functions)
    return names


def run(tokens, channels, separate_region=False):
    torch._dynamo.reset()
    torch.manual_seed(172021 + channels)
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = True
    T.USE_GATED_MLP = T.USE_COALESCED_SWIGLU = T.USE_NKI_RELU2 = False
    T.USE_PARALLEL_BLOCK = False
    T.RELU2_TAU = 0.0
    baseline = Probe(channels).to('neuron')
    candidate = copy.deepcopy(baseline)
    contract = install(candidate, 'neuron', T, separate_region=separate_region)
    native = torch.compile(baseline, backend='neuron', fullgraph=True, dynamic=False)
    optimized = torch.compile(candidate, backend='neuron', fullgraph=not separate_region, dynamic=False)
    x = torch.randn(4, tokens // 4, channels).bfloat16().to('neuron')
    gradient = (torch.randn(x.shape) * .01).bfloat16().to('neuron')
    refresh_changes = 0
    for cached in (False, True):
        last_cache = None
        for step in range(2):
            baseline.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            if cached:
                refresh(baseline)
                refresh(candidate)
                current = candidate.transformer.h[0].mlp.c_fc._cached_weight.detach().cpu()
                if last_cache is not None:
                    assert not torch.equal(last_cache, current)
                    refresh_changes += 1
                last_cache = current.clone()
            for micro in range(2):
                value = x + micro * .03125
                bx = value.detach().clone().requires_grad_()
                cx = value.detach().clone().requires_grad_()
                expected, output = native(bx), optimized(cx)
                assert type(expected.grad_fn).__name__ == 'CompiledFunctionBackward'
                if separate_region:
                    assert 'CompiledFunctionBackward' in backward_node_names(output)
                else:
                    assert type(output.grad_fn).__name__ == 'CompiledFunctionBackward'
                compare(f'library_integration:{channels}:{cached}:{step}:{micro}:output', output, expected)
                expected.backward(gradient)
                output.backward(gradient)
                compare(f'library_integration:{channels}:{cached}:{step}:{micro}:dx', cx.grad, bx.grad)
            for (name, p), (other, q) in zip(candidate.named_parameters(), baseline.named_parameters()):
                assert name == other and p.dtype == q.dtype == torch.float32
                assert p.grad is not None and p.grad.dtype == q.grad.dtype == torch.float32
                compare(f'library_integration:{channels}:{cached}:{step}:{name}:accumulated', p.grad, q.grad)
            with torch.no_grad():
                for p, q in zip(candidate.parameters(), baseline.parameters()):
                    q.add_(q.grad, alpha=-.001)
                    p.copy_(q)
    assert refresh_changes == 1
    # This is a training call: exact row independence cannot be satisfied by
    # the native evaluation path. Batches and suffix rows are both perturbed.
    refresh(candidate)
    with torch.no_grad():
        original = optimized(x)
        altered = x.clone()
        altered[1:] *= -3
        other = optimized(altered)
        exact(f'library_integration:{channels}:training_other_batch', other[:1], original[:1])
        altered = x.clone()
        altered[:, tokens // 8:] *= -2
        future = optimized(altered)
        exact(f'library_integration:{channels}:training_future_rows',
              future[:, :tokens // 8], original[:, :tokens // 8])
        exact(f'library_integration:{channels}:training_repeat', optimized(x), original)
    candidate.eval()
    baseline.eval()
    refresh(candidate)
    refresh(baseline)
    restored = Probe(channels).to('neuron')
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval()
    refresh(restored)
    with torch.no_grad():
        short = x[:2, :31]
        exact(f'library_integration:{channels}:native_eval', candidate(short), baseline(short))
        exact(f'library_integration:{channels}:strict_restore', restored(short), candidate(short))
    emit('NKILIB_MLP_INTEGRATION_GATE', dict(passed=True, tokens=tokens, channels=channels,
        cached_and_uncached=True, reference_updates=4, microbatches=2, fp32_master_gradients=True,
        changed_cache_values=True, compiled_execution_asserted=True,
        compiled_backward_graph_walk=separate_region,
        training_row_and_batch_independence=True, native_eval_and_checkpoint_exact=True,
        separate_compiled_region=separate_region, contract=contract))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-width', action='store_true')
    parser.add_argument('--separate-region', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(256, 256, args.separate_region)
    if args.real_width:
        run(2048, 1024, args.separate_region)
    emit('NKILIB_MLP_INTEGRATION_RESULT', dict(passed=True, cases=2 if args.real_width else 1,
        separate_compiled_region=args.separate_region, full_model_or_quality_claim=False))


if __name__ == '__main__':
    main()
