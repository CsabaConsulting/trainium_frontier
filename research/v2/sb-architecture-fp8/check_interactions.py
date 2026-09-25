"""Bounded S-B self-checks. No native execution, real data, or launch authority."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import torch


SOURCE = 'workflow/proposals/research-v2/train.py'
REFS = {
    'architecture': ('1e21c3d797eed764f108c41546b270065ad5650c', SOURCE,
                     '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5'),
    'fp8': ('50a901374ae6f76b35b8556d32fd72b1ece28fc6', SOURCE,
            '921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858'),
    'dense': ('b18fb3586494c21f10008a9c63fd2056d1e6c00a', SOURCE,
              'c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586'),
}


def load(name, source, path):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(source, str(path), 'exec'), mod.__dict__)
    return mod


def configure(mod, architecture, fp8):
    mod.USE_UNET_SKIP = architecture
    mod.USE_PARTIAL_ATTN_8 = architecture
    mod.ATTENTION_LAYERS = '0,2,4,6,7'
    mod.USE_FP8_PROJ = fp8
    mod.USE_NGRAM_VE = False
    mod.USE_QK_SHIFT = False
    mod.USE_BF16_NORM_OUTPUT = True
    mod.COMPUTE_DTYPE = torch.bfloat16
    mod.compute_dtype_for = lambda device: torch.bfloat16
    mod.EVAL_MIN_SEQ_LEN = 32  # Only bounds the unchanged evaluation mask allocation.


def model_for(mod, architecture, fp8):
    configure(mod, architecture, fp8)
    torch.manual_seed(42)
    cfg = mod.GPTConfig(sequence_len=32, vocab_size=64,
                        n_layer=8 if architecture else 6, n_head=2, n_embd=128)
    model = mod.GPT(cfg)
    model.init_weights()
    return model, model.setup_optimizer()


def refresh(mod, model):
    for layer in model.modules():
        if isinstance(layer, mod.Linear):
            layer.refresh_weight_cache(torch.bfloat16)


def update(mod, model, optimizer, token, target):
    optimizer.zero_grad(set_to_none=True)
    loss = model(token, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    optimizer.step()
    refresh(mod, model)
    return float(loss.detach()), gradients


def identical(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), 'tensor mismatch'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-py', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    path = Path(args.train_py).resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = path.read_text()
    texts = {'combined': source}
    mods = {'combined': load('sb_combined', source, path)}
    for name, (commit, rel, expected) in REFS.items():
        raw = subprocess.check_output(['git', 'show', f'{commit}:{rel}'])
        assert hashlib.sha256(raw).hexdigest() == expected
        texts[name] = raw.decode()
        mods[name] = load(f'sb_{name}', texts[name], path)
    checks = {}
    nodes = {name: {n.name: n for n in ast.parse(text).body
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
             for name, text in texts.items()}
    dump = lambda n: ast.dump(n, include_attributes=False)
    changed = {'Linear', 'CausalSelfAttention', 'MLP', 'main', '_install_submission_training'}
    for name, node in nodes['architecture'].items():
        if name not in changed:
            assert dump(node) == dump(nodes['combined'][name]), name
    for name in ['_fp8_row_reference', '_build_fp8_projection_kernels',
                 '_ensure_fp8_projection_kernels', '_FP8Projection', 'Linear',
                 'CausalSelfAttention', 'MLP', '_install_submission_training']:
        assert dump(nodes['combined'][name]) == dump(nodes['fp8'][name]), name
    for name in ['architecture', 'fp8']:
        loops = [n for n in ast.walk(nodes[name]['main']) if isinstance(n, ast.While)
                 and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                 and n.test.left.id == 'step']
        combined = [n for n in ast.walk(nodes['combined']['main']) if isinstance(n, ast.While)
                    and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                    and n.test.left.id == 'step']
        assert len(loops) == len(combined) == 1 and dump(loops[0]) == dump(combined[0])
    assert 'USE_RECURSIVE_SCHEDULE' not in source
    assert mods['combined'].TOTAL_BATCH_SIZE == mods['architecture'].TOTAL_BATCH_SIZE == 131072
    checks['exact_component_AST_and_unchanged_loop_loader_eval_optimizer_batch'] = True

    # Test the public preset without allocating its full-size model.
    class BeforeRuntime(Exception):
        pass
    m = mods['combined']
    old_runtime, old_argv = m.init_runtime, sys.argv
    def stop(*a, **k):
        raise BeforeRuntime
    m.init_runtime = stop
    for flags, expected in [([], (False, False)),
                            (['--unet-8x1024', '--no-fp8-proj'], (True, False)),
                            (['--fp8-proj'], (False, True)),
                            (['--unet-8x1024', '--fp8-proj'], (True, True))]:
        # Fresh defaults before every parser invocation.
        m.USE_UNET_SKIP = m.USE_PARTIAL_ATTN_8 = m.USE_FP8_PROJ = False
        m.DEPTH, m.N_EMBD = 6, 1024
        sys.argv = [str(path), '--no-ngram-ve'] + flags
        try:
            m.main()
            raise AssertionError('runtime must be stopped')
        except BeforeRuntime:
            assert (m.USE_UNET_SKIP, m.USE_FP8_PROJ) == expected
            assert m.build_config().n_layer == (8 if expected[0] else 6)
            assert m.build_config().n_embd == 1024
    m.init_runtime, sys.argv = old_runtime, old_argv
    checks['four_independent_CLI_modes_and_default_OFF_preset_without_allocation'] = True

    generator = torch.Generator().manual_seed(42)
    token = torch.randint(0, 64, (2, 16), generator=generator)
    target = torch.randint(0, 64, (2, 16), generator=generator)
    for name, architecture, fp8 in [('dense', False, False),
                                     ('architecture', True, False), ('fp8', False, True)]:
        a, oa = model_for(m, architecture, fp8)
        ref = mods[name]
        b, ob = model_for(ref, architecture, fp8)
        identical(a.state_dict(), b.state_dict())
        losses = []
        for _ in range(3):
            la, ga = update(m, a, oa, token, target)
            lb, gb = update(ref, b, ob, token, target)
            assert la == lb
            identical(ga, gb)
            identical(a.state_dict(), b.state_dict())
            losses.append(la)
        checks[f'{name}_three_step_exact_losses_gradients_and_weights'] = losses
        del a, b, oa, ob

    model, optimizer = model_for(m, True, True)
    marked = [name for name, layer in model.named_modules()
              if isinstance(layer, m.Linear) and layer.fp8_projection]
    assert len(marked) == 36 and not model.lm_head.fp8_projection
    assert [i for i, b in enumerate(model.transformer.h) if b.attn is not None] == [0, 2, 4, 6, 7]
    assert list(model.value_embeds) == ['7'] and not model.ngram_embeds
    assert all(p.shape == (1,) and p.item() == 0 for p in model.unet_skip_gates)
    losses = [update(m, model, optimizer, token, target)[0] for _ in range(3)]
    assert any(p.item() != 0 for p in model.unet_skip_gates)
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters())
    assert all(layer._cached_weight is None for layer in model.modules()
               if isinstance(layer, m.Linear) and layer.fp8_projection)
    checks['combined_36_projections_no_unused_tables_FP32_masters_learned_scalar_skips'] = True
    checks['combined_three_finite_optimizer_steps'] = losses

    model.eval()
    with torch.no_grad():
        expected = model(token)
        changed = token.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 64
        assert torch.equal(expected[:, :8], model(changed)[:, :8])
        assert torch.equal(expected, torch.cat([model(row[None]) for row in token]))
        assert torch.equal(expected, model(token))
        assert not torch.equal(expected, model((token + 3) % 64))
    checks['combined_causality_determinism_batch_independence_input_sensitive'] = True
    with tempfile.TemporaryDirectory(prefix='sb-combined-tiny-') as tmp:
        checkpoint = Path(tmp) / 'final.pt'
        m.save_checkpoint(checkpoint, model, {'step': 3, 'synthetic': True})
        m.USE_UNET_SKIP = m.USE_PARTIAL_ATTN_8 = m.USE_FP8_PROJ = False
        m.ATTENTION_LAYERS = '1,3,5,7'
        restored = m.load_for_eval(str(checkpoint), torch.device('cpu'))
        assert m.USE_UNET_SKIP and m.USE_PARTIAL_ATTN_8 and m.USE_FP8_PROJ
        assert m.ATTENTION_LAYERS == '0,2,4,6,7'
        assert not restored.training and all(not p.requires_grad for p in restored.parameters())
        identical(model.state_dict(), restored.state_dict())
        with torch.no_grad():
            assert torch.equal(expected, restored(token))
            assert torch.equal(expected[:, :8], restored(changed)[:, :8])
            assert torch.equal(expected, torch.cat([restored(row[None]) for row in token]))
        payload = torch.load(checkpoint, weights_only=True)
        assert payload['toggles']['USE_FP8_PROJ'] and payload['toggles']['USE_UNET_SKIP']
        del payload['model_state']['unet_skip_gates.0']
        broken = Path(tmp) / 'missing-skip.pt'
        torch.save(payload, broken)
        try:
            m.load_for_eval(str(broken), torch.device('cpu'))
        except RuntimeError as exc:
            assert 'Missing key' in str(exc)
        else:
            raise AssertionError('strict loading must reject missing skip weights')
        for key in ['USE_FP8_PROJ', 'USE_UNET_SKIP', 'USE_PARTIAL_ATTN_8', 'ATTENTION_LAYERS']:
            payload['toggles'].pop(key)
        m.restore_eval_toggles(payload)
        assert not m.USE_FP8_PROJ and not m.USE_UNET_SKIP and not m.USE_PARTIAL_ATTN_8
    checks['combined_weights_only_strict_reload_exact_trained_toggles_and_logits'] = True
    checks['legacy_toggles_default_OFF_and_missing_skip_rejected'] = True
    result = dict(verdict='PASS_IMPLEMENTER_SELF_CHECK_ONLY', seed=42,
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  torch=torch.__version__, max_shape='8x128,vocab64,seq32; two models at once',
                  threads=1, checks=checks, native_executed=False, exact_size_g2=False,
                  performance_claim=False, bpb_measured=False,
                  limitations=['CPU FP8 numerical reference only', 'Test-only BF16 dtype and eval mask32',
                               'Synthetic tokens only', 'Independent review pending'])
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
