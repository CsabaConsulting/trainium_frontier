"""Test-only observations layered on the existing G2 driver; no native launcher."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('g2', ROOT / 'workflow/tools/preflight_dryrun.py')
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
# Keep its launcher/worker/aggregation unchanged except adding these observations.
# Its subprocess returns through this file so the same hooks exist on each rank.
g2.__file__ = __file__


def observe(mod):
    import torch
    torch.set_num_threads(1)
    rank = int(os.environ['RANK'])
    evidence = {'scope': 'tiny CPU only; no Neuron compilation', 'steps': [],
                'forwards': [], 'cache_refreshes': [], 'mlp_custom_backward_calls': 0}
    captured = {}

    def digest(t):
        t = t.detach().cpu().contiguous()
        raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
        return {'shape': list(t.shape), 'dtype': str(t.dtype),
                'sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest()}

    def tree(value):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise ValueError('nonfinite observed tensor')
            return digest(value)
        if isinstance(value, dict):
            return {str(k): tree(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [tree(v) for v in value]
        return value

    def params(model, gradients=False):
        return {n: (None if p.grad is None else tree(p.grad)) if gradients else tree(p)
                for n, p in model.named_parameters()}

    original_refresh = mod.Linear.refresh_weight_cache
    def refresh(self, dtype):
        evidence['cache_refreshes'].append({'completed_steps': len(evidence['steps']),
                                           'forward_calls': len(evidence['forwards']),
                                           'grad_enabled': torch.is_grad_enabled()})
        return original_refresh(self, dtype)
    mod.Linear.refresh_weight_cache = refresh

    original_forward = mod.GPT.forward
    def forward(self, idx, targets=None, *args, **kwargs):
        if self.training and targets is not None:
            captured['model'] = self
            cache_records = {}
            for name, linear in self.named_modules():
                if not isinstance(linear, mod.Linear):
                    continue
                cached = linear._cached_weight
                if cached is None:
                    cache_records[name] = None
                    continue
                if not torch.equal(cached, linear.weight.detach().to(cached.dtype)):
                    raise ValueError('stale or numerically different cache: ' + name)
                connected = None
                if not evidence['forwards']:
                    # autograd.grad leaves Parameter.grad unmodified. Retain the
                    # cast graph for the real ensuing training backward.
                    before = linear.weight.grad
                    grad, = torch.autograd.grad(cached.sum(), linear.weight, retain_graph=True)
                    connected = bool(torch.equal(grad, torch.ones_like(linear.weight)))
                    if not connected or linear.weight.grad is not before:
                        raise ValueError('cache detached from FP32 master: ' + name)
                cache_records[name] = {'value': tree(cached), 'connected_first_forward': connected,
                                       'requires_grad': cached.requires_grad,
                                       'grad_fn': type(cached.grad_fn).__name__,
                                       'master_dtype': str(linear.weight.dtype)}
            out = original_forward(self, idx, targets, *args, **kwargs)
            # This proves the selected custom MLP autograd function is present;
            # the loss graph's backward runs in the unchanged main loop.
            seen, todo, custom = set(), [out.grad_fn], 0
            while todo:
                node = todo.pop()
                if node is None or node in seen:
                    continue
                seen.add(node)
                custom += type(node).__name__ == 'FusedMLPBackward'
                todo.extend(next_node for next_node, _ in node.next_functions)
            evidence['mlp_custom_backward_calls'] += custom
            evidence['forwards'].append({'input_shape': list(idx.shape), 'loss': float(out.detach()),
                                         'caches': cache_records, 'custom_mlp_nodes': custom})
            return out
        return original_forward(self, idx, targets, *args, **kwargs)
    mod.GPT.forward = forward

    original_clip = mod.clip_and_sync_gradients
    def clip(model, *args, **kwargs):
        captured['before_clip'] = params(model, gradients=True)
        result = original_clip(model, *args, **kwargs)
        captured['after_clip'] = params(model, gradients=True)
        return result
    mod.clip_and_sync_gradients = clip

    original_step = mod.OwnedOptimizer.step
    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        evidence['steps'].append({'gradients_before_clip': captured['before_clip'],
                                  'gradients_after_clip': captured['after_clip'],
                                  'parameters_after_update': params(captured['model']),
                                  'optimizer_state': tree(self.state_dict())})
        return result
    mod.OwnedOptimizer.step = step

    original_save = mod.save_checkpoint
    def save(path, *args, **kwargs):
        result = original_save(path, *args, **kwargs)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        evidence['checkpoint'] = tree(payload)
        evidence['checkpoint_file_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return result
    mod.save_checkpoint = save

    original_cleanup = mod.cleanup_runtime
    def cleanup():
        (Path(os.environ['G2_WORKDIR']) / f'cache-observe-rank{rank}.json').write_text(
            json.dumps(evidence, sort_keys=True, indent=2))
        return original_cleanup()
    mod.cleanup_runtime = cleanup


original_spec = g2.importlib.util.spec_from_file_location
def instrumented_spec(name, *args, **kwargs):
    result = original_spec(name, *args, **kwargs)
    if name == 'train_under_test':
        original_exec = result.loader.exec_module
        def load(mod):
            original_exec(mod)
            observe(mod)
        result.loader.exec_module = load
    return result
g2.importlib.util.spec_from_file_location = instrumented_spec

original_aggregate = g2.aggregate
def aggregate(ns, train_py, prepare_dir, train_argv, ranks, *args):
    result = original_aggregate(ns, train_py, prepare_dir, train_argv, ranks, *args)
    for rank in ranks:
        if not rank.get('appended_argv'):
            continue
        work = Path(rank['appended_argv'][-1]).parent
        path = work / f"cache-observe-rank{rank['rank']}.json"
        rank['cache_observations'] = json.loads(path.read_text()) if path.exists() else None
    result['observation_driver_sha256'] = g2.sha256_file(Path(__file__))
    return result
g2.aggregate = aggregate

if __name__ == '__main__':
    sys.exit(g2.main())
