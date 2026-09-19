#!/usr/bin/env python3
"""Gate the real attention adapter, master/cache lifecycle and causal behavior.

Independent projection FP64 gates are a prerequisite. This uses the campaign's
existing, unchanged paired-native BF16 bounds for the complete nonlinear
attention path; exact equality remains required for causality, repeatability,
native evaluation and strict checkpoint restoration. No tolerance is fitted to
candidate results. No whole-model speed or quality claim follows.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from mlp_fused_bf16_probe import compare
from serial_qkv_integration import install


def require_component(root, layout):
    root = Path(root).resolve()
    current = Path(__file__).resolve().parents[2]
    label = 'fs-w151-serial-qkv-' + layout + '-component-r1'
    post_path = root / 'research/run-logs' / (label + '.postflight.json')
    log_path = root / 'research/run-logs' / (label + '.log')
    post = json.loads(post_path.read_text())
    assert post['source_commit'] == '85addadd2ade1d0eb37fa9a5294b598bfe2ba9b4'
    assert post['command_returncode'] == 0 and post['host_clean']
    assert not post['failures'] and not post['guard_error'] and not post['interrupted_signals']
    digests = {}
    for name in ('serial_qkv_nki.py', 'serial_qkv_probe.py'):
        relative = Path('research/scratch') / name
        a, b = [(p / relative).read_bytes() for p in (root, current)]
        assert a == b, f'Primitive or its gate changed after the component prerequisite: {name}'
        digests[name] = hashlib.sha256(a).hexdigest()
    records = {}
    for line in log_path.read_text().splitlines():
        if line.startswith('SERIAL_QKV_'):
            marker, payload = line.split(' ', 1)
            records.setdefault(marker, []).append(json.loads(payload))
    result, = records['SERIAL_QKV_RESULT']
    assert result['layout'] == layout and result['cases'] == 9 and result['passed']
    assert result['real_tokens'] == 32768 and result['real_channels'] == 1024
    assert result['independent_fp64_gates'] and result['lifecycle_passed']
    cases = records['SERIAL_QKV_CASE']
    assert len(cases) == 9 and all(c['passed'] for c in cases)
    assert {c['kind'] for c in cases} == {
        'random', 'zero_input', 'zero_weight', 'zero_gradient', 'tiny_gradient',
        'outliers', 'changed_weights', 'cancellation', 'real_random'}
    print('SERIAL_QKV_COMPONENT_PREREQUISITE ' + json.dumps(dict(
        passed=True, source_commit=post['source_commit'], label=label,
        component_log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
        postflight_sha256=hashlib.sha256(post_path.read_bytes()).hexdigest(),
        current_primitive_and_gate_sha256=digests)), flush=True)


class Probe(torch.nn.Module):
    def __init__(self, channels, length=128):
        super().__init__()
        config = T.GPTConfig(sequence_len=length, vocab_size=8192, n_layer=1,
                             n_head=channels // 128, n_embd=channels, mlp_ratio=4)
        block = torch.nn.Module()
        block.attn = T.CausalSelfAttention(config, 0)
        self.transformer = torch.nn.ModuleDict({'h': torch.nn.ModuleList([block])})

    def forward(self, q, k, v, ve, cos, sin):
        return self.transformer.h[0].attn(q, k, v, ve, cos, sin, None)


def refresh(model):
    for module in model.modules():
        if isinstance(module, T.Linear): module.refresh_weight_cache(torch.bfloat16)


def exact(name, actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    passed = bool(torch.isfinite(a).all()) and torch.equal(a, b)
    print('SERIAL_QKV_ATTENTION_EXACT ' + json.dumps(dict(name=name, passed=passed)), flush=True)
    assert passed, name


def run(channels, layout):
    torch.manual_seed(151031 + channels)
    baseline = Probe(channels).to('neuron')
    candidate = copy.deepcopy(baseline)
    contract = install(candidate, 'neuron', T, layout)
    native = torch.compile(baseline, backend='neuron', fullgraph=True, dynamic=False)
    optimized = torch.compile(candidate, backend='neuron', fullgraph=True, dynamic=False)
    host_inputs = [torch.nn.functional.rms_norm(torch.randn(4, 128, channels), (channels,)).bfloat16()
                   for _ in range(3)]
    host_inputs.append((torch.randn(4, 128, channels) * .1).bfloat16())
    angles = torch.outer(torch.arange(128).float(),
                         100000.0**(-torch.arange(0, 128, 2).float() / 128))
    cos, sin = [v.reshape(1, 128, 1, 64).to('neuron') for v in (angles.cos(), angles.sin())]
    gradient = (torch.randn(4, 128, channels) * .01).bfloat16().to('neuron')
    for cached in (False, True):
        for step in range(2):
            baseline.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
            if cached:
                refresh(baseline); refresh(candidate)
            for micro in range(2):
                inputs = [x + micro * .03125 for x in host_inputs]
                bx = [x.to('neuron').detach().clone().requires_grad_() for x in inputs]
                cx = [x.to('neuron').detach().clone().requires_grad_() for x in inputs]
                expected, output = native(*bx, cos, sin), optimized(*cx, cos, sin)
                assert type(expected.grad_fn).__name__ == type(output.grad_fn).__name__ == 'CompiledFunctionBackward'
                tag = f'qkv_attention:{channels}:{cached}:{step}:{micro}'
                compare(tag + ':output', output, expected)
                expected.backward(gradient); output.backward(gradient)
                for index, (actual, truth) in enumerate(zip(cx, bx)):
                    compare(tag + f':input_gradient{index}', actual.grad, truth.grad)
            for (name, actual), (other_name, truth) in zip(candidate.named_parameters(), baseline.named_parameters()):
                assert name == other_name and actual.dtype == truth.dtype == torch.float32
                if truth.grad is None:
                    assert actual.grad is None
                else:
                    assert actual.grad is not None and actual.grad.dtype == torch.float32
                    compare(f'qkv_attention:{channels}:{cached}:{step}:accumulated:{name}', actual.grad, truth.grad)
            # Same reference update isolates refreshed-cache behavior from
            # accumulated numerical trajectory differences across optimizers.
            with torch.no_grad():
                for actual, truth in zip(candidate.parameters(), baseline.parameters()):
                    if truth.grad is not None: truth.add_(truth.grad, alpha=-.002)
                    actual.copy_(truth)
    refresh(candidate); refresh(baseline)
    with torch.no_grad():
        inputs = [x.to('neuron') for x in host_inputs]
        first = optimized(*inputs, cos, sin)
        exact(f'{channels}:training_repeat', optimized(*inputs, cos, sin), first)
        changed = [x.clone() for x in inputs]
        for value in changed: value[0, 64:].add_(.5)
        result = optimized(*changed, cos, sin)
        exact(f'{channels}:future_independence', result[0, :64], first[0, :64])
        exact(f'{channels}:other_batch_independence', result[1:], first[1:])
        # Three batches require projection-row padding, while keeping the
        # original attention length and the RoPE fast-path geometry unchanged.
        short = [x[:3].contiguous() for x in inputs]
        compare(f'{channels}:padded_training_rows', optimized(*short, cos, sin), native(*short, cos, sin))
    candidate.eval(); baseline.eval()
    refresh(candidate); refresh(baseline)
    with torch.no_grad():
        exact(f'{channels}:native_evaluation', candidate(*inputs, cos, sin), baseline(*inputs, cos, sin))
    restored = Probe(channels).to('neuron')
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval(); refresh(restored)
    with torch.no_grad():
        exact(f'{channels}:strict_native_restore', restored(*inputs, cos, sin), candidate(*inputs, cos, sin))
    print('SERIAL_QKV_ATTENTION_GATE ' + json.dumps(dict(
        channels=channels, layout=layout, passed=True, compiled_forward_backward=True,
        cached_and_uncached=True, original_separate_parameters=True,
        accumulation_and_refresh=True, distinct_input_gradients=True,
        causal_and_batch_independence_exact=True, padded_projection_rows=True,
        native_evaluation_and_restore_exact=True, contract=contract)), flush=True)


def real_attention_timing(layout, steps):
    """Matched real-shape forward/backward, charging cache refresh and grads."""
    batch, length, channels = 32, 1024, 1024
    torch.manual_seed(152032)
    baseline = Probe(channels, length).to('neuron')
    candidate = copy.deepcopy(baseline)
    install(candidate, 'neuron', T, layout)
    native = torch.compile(baseline, backend='neuron', fullgraph=True, dynamic=False)
    optimized = torch.compile(candidate, backend='neuron', fullgraph=True, dynamic=False)
    values = [torch.nn.functional.rms_norm(torch.randn(batch, length, channels), (channels,)).bfloat16()
              for _ in range(3)]
    values.append((torch.randn(batch, length, channels) * .1).bfloat16())
    bx = [x.to('neuron').requires_grad_() for x in values]
    cx = [x.to('neuron').detach().clone().requires_grad_() for x in values]
    angles = torch.outer(torch.arange(length).float(),
                         100000.0**(-torch.arange(0, 128, 2).float() / 128))
    cos, sin = [v.reshape(1, length, 1, 64).to('neuron') for v in (angles.cos(), angles.sin())]
    gradient = (torch.randn(batch, length, channels) * .01).bfloat16().to('neuron')

    def update(model, compiled, inputs):
        model.zero_grad(set_to_none=True)
        for x in inputs: x.grad = None
        refresh(model)
        output = compiled(*inputs, cos, sin)
        output.backward(gradient)
        return output

    expected = update(baseline, native, bx)
    actual = update(candidate, optimized, cx)
    compare('qkv_real_attention:output', actual, expected)
    for index, (a, b) in enumerate(zip(cx, bx)):
        compare(f'qkv_real_attention:input_gradient{index}', a.grad, b.grad)
    for (name, a), (other, b) in zip(candidate.named_parameters(), baseline.named_parameters()):
        assert name == other
        if b.grad is None: assert a.grad is None
        else:
            assert a.grad.dtype == b.grad.dtype == torch.float32
            compare('qkv_real_attention:parameter_gradient:' + name, a.grad, b.grad)
    calls = {'native': (baseline, native, bx), 'candidate': (candidate, optimized, cx)}
    for args in calls.values():
        for _ in range(3): update(*args)
    torch.neuron.synchronize()
    samples = {name: [] for name in calls}
    for block in range(5):
        for name in (('native', 'candidate') if block % 2 == 0 else ('candidate', 'native')):
            torch.neuron.synchronize()
            start = time.perf_counter()
            for _ in range(steps): update(*calls[name])
            torch.neuron.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000 / steps)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    print('SERIAL_QKV_ATTENTION_TIMING ' + json.dumps(dict(
        layout=layout, batch=batch, length=length, channels=channels,
        passed=True, real_shape_output_input_and_master_gradients_gated=True,
        samples_ms=samples, median_ms=medians, steps_per_block=steps,
        improvement_pct=100 * (medians['native'] - medians['candidate']) / medians['native'],
        complete_attention_forward_backward=True, weight_cache_refresh_timed=True,
        compiled_native_control=True, all_layout_and_dispatch_costs_timed=True,
        whole_model_or_quality_claim=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=('tokens', 'channels'), required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--component-root', required=True)
    args = parser.parse_args()
    assert args.steps > 0
    require_component(args.component_root, args.layout)
    import torch_neuronx  # noqa: F401
    torch.set_num_threads(4)
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = T.USE_NKI_ROPE_NORM = T.COMPILE_SDPA_DIRECT = True
    T.USE_FUSED_QKV = T.USE_PARALLEL_BLOCK = T.USE_NGRAM_VE = T.USE_HEAD_GATE = T.USE_NKI_ATTN = False
    T.USE_VE = True
    assert T._HAS_NKI
    run(256, args.layout)
    torch._dynamo.reset()
    print('SERIAL_QKV_ATTENTION_GUARD_RESET ' + json.dumps(dict(
        completed_width=256, next_width=1024, shared_device_cache_purged=False)), flush=True)
    run(1024, args.layout)
    torch._dynamo.reset()
    print('SERIAL_QKV_ATTENTION_GUARD_RESET ' + json.dumps(dict(
        next_shape=[32, 1024, 1024], shared_device_cache_purged=False)), flush=True)
    real_attention_timing(args.layout, args.steps)
    print('SERIAL_QKV_INTEGRATION_RESULT ' + json.dumps(dict(
        layout=args.layout, widths=[256, 1024], passed=True,
        paired_native_bounds_unchanged=True, native_rope_geometry_preserved=True,
        real_shape_attention_timing_complete=True,
        whole_model_or_quality_claim=False)), flush=True)


if __name__ == '__main__':
    main()
