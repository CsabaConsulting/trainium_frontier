"""Bounded L2 self-checks: synthetic tokens, <=8x128 models, no host/data access.

Run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 and this folder's test_support plus
workflow/tools/stub_prepare on PYTHONPATH. This is not exact-size G2 or approval.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch


def load(path):
    spec = importlib.util.spec_from_file_location('l2_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-py', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m = load(args.train_py)
    checks = {}

    # Exercise actual argparse/main up to init_runtime, before model allocation.
    class StopBeforeRuntime(Exception):
        pass

    native_runtime = m.init_runtime
    def no_runtime(*args, **kwargs):
        raise StopBeforeRuntime
    m.init_runtime = no_runtime
    saved_argv = sys.argv
    sys.argv = [args.train_py, '--unet-8x1024', '--no-ngram-ve', '--n-embd', '128', '--depth', '2']
    try:
        m.main()
        raise AssertionError('main should stop before runtime')
    except StopBeforeRuntime:
        cfg = m.build_config()
        assert (cfg.n_layer, cfg.n_embd) == (8, 1024)
        assert m.USE_UNET_SKIP and m.USE_PARTIAL_ATTN_8
        checks['preset_forces_8x1024_before_runtime'] = True
    finally:
        sys.argv = saved_argv
        m.init_runtime = native_runtime

    m.COMPUTE_DTYPE = torch.bfloat16
    m.compute_dtype_for = lambda device: torch.bfloat16
    m.USE_BF16_NORM_OUTPUT = True
    m.USE_NGRAM_VE = False
    m.USE_QK_SHIFT = False
    m.EVAL_MIN_SEQ_LEN = 32  # Test-only bound; production eval helper is unchanged.
    cfg = m.GPTConfig(sequence_len=32, vocab_size=64, n_layer=8, n_head=2, n_embd=128)
    torch.manual_seed(42)
    with_skip = m.GPT(cfg)
    with_skip.init_weights()
    assert [i for i, b in enumerate(with_skip.transformer.h) if b.attn is not None] == [0, 2, 4, 6, 7]
    assert list(with_skip.value_embeds) == ['7']
    assert len(with_skip.unet_skip_gates) == 4
    assert all(p.shape == (1,) and p.item() == 0 for p in with_skip.unet_skip_gates)
    for i in [1, 3, 5]:
        block = with_skip.transformer.h[i]
        assert block.attn is None and block.local_conv is None and block.qk_shift_beta is None
    opt = with_skip.setup_optimizer()
    for gate in with_skip.unet_skip_gates:
        groups = [g for g in opt.param_groups if any(gate is p for p in g['params'])]
        assert len(groups) == 1 and groups[0]['lr'] == m.SCALAR_LR
        assert groups[0]['kind'] == 'adamw' and groups[0]['weight_decay'] == 0
    checks['attention_scalar_optimizer_and_no_unused_ve'] = True

    m.USE_UNET_SKIP = False
    torch.manual_seed(42)
    no_skip = m.GPT(cfg)
    no_skip.init_weights()
    assert all(torch.equal(p, with_skip.state_dict()[k]) for k, p in no_skip.state_dict().items())
    tokens = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        assert torch.equal(with_skip(tokens), no_skip(tokens))
    checks['zero_init_skip_exact_logits_and_common_weights'] = True

    m.USE_UNET_SKIP = True
    with_skip.train()
    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = with_skip(tokens, tokens.roll(-1, dims=1))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in with_skip.parameters() if p.requires_grad)
        opt.step()
        for layer in with_skip.modules():
            if isinstance(layer, m.Linear):
                layer.refresh_weight_cache(m.COMPUTE_DTYPE)
        losses.append(loss.item())
    assert any(p.item() != 0 for p in with_skip.unet_skip_gates)
    checks['three_finite_steps_all_grads_and_learned_skips'] = losses

    with tempfile.TemporaryDirectory(prefix='l2-small-check-') as scratch:
        path = Path(scratch) / 'final.pt'
        m.save_checkpoint(path, with_skip, {'step': 3, 'synthetic': True})
        with_skip.eval()
        with torch.no_grad():
            reference = with_skip(tokens[:1])
        m.USE_UNET_SKIP = m.USE_PARTIAL_ATTN_8 = False
        m.ATTENTION_LAYERS = '1,3,5,7'
        ev = m.load_for_eval(str(path), torch.device('cpu'))
        assert m.USE_UNET_SKIP and m.USE_PARTIAL_ATTN_8 and m.ATTENTION_LAYERS == '0,2,4,6,7'
        assert not ev.training and all(not p.requires_grad for p in ev.parameters())
        with torch.no_grad():
            logits = ev(tokens)
            assert torch.equal(logits[:1], reference)
            assert torch.equal(logits, ev(tokens))
            assert torch.equal(logits, torch.cat([ev(row[None]) for row in tokens]))
            changed = tokens.clone()
            changed[:, 8:] = (changed[:, 8:] + 1) % 64
            assert torch.equal(logits[:, :8], ev(changed)[:, :8])
        checks['strict_reload_logit_parity_determinism_batch_independence_causality'] = True
        payload = m.torch_load_weights(path, 'cpu')
        del payload['model_state']['unet_skip_gates.0']
        broken = Path(scratch) / 'missing-skip.pt'
        torch.save(payload, broken)
        try:
            m.load_for_eval(str(broken), torch.device('cpu'))
        except RuntimeError as exc:
            assert 'Missing key' in str(exc)
        else:
            raise AssertionError('missing skip checkpoint key must fail strict load')
        checks['missing_skip_key_rejected'] = True

    for value in ['', '0,,2', '0,0', '-1,7', '0,8', 'x,2']:
        m.ATTENTION_LAYERS = value
        try:
            m.attention_layer_set(8)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid list accepted: {value!r}')
    m.ATTENTION_LAYERS = '0,2,4,6,7'
    for depth in [1, 3, 5, 7, 9]:
        try:
            m.GPT(m.GPTConfig(sequence_len=16, vocab_size=64, n_layer=depth, n_head=2, n_embd=128))
        except ValueError as exc:
            assert 'even depth' in str(exc)
        else:
            raise AssertionError(f'odd depth {depth} accepted')
    checks['malformed_attention_and_odd_depth_rejected_before_allocation'] = True

    m.USE_NGRAM_VE = True
    m.NGRAM_VE_TABLE_MULT = 1
    m.NGRAM_VE_DIM = 16
    m.NGRAM_VE_LAYERS = '1,-1'
    try:
        m.GPT(cfg)
    except ValueError as exc:
        assert 'MLP-only' in str(exc)
    else:
        raise AssertionError('explicit table request on removed attention must fail')
    for selection in ['', '-1']:
        m.NGRAM_VE_LAYERS = selection
        model = m.GPT(cfg)
        assert list(model.value_embeds) == ['7']
        assert list(model.ngram_embeds) == ['b7', 't7']
    m.ATTENTION_LAYERS = '0,1,3,5,7'
    m.NGRAM_VE_LAYERS = '1,-1'
    model = m.GPT(cfg)
    assert list(model.value_embeds) == ['1', '3', '5', '7']
    assert list(model.ngram_embeds) == ['b1', 't1', 'b7', 't7']
    checks['table_filtering_explicit_conflict_and_custom_attention'] = True

    result = {'verdict': 'PASS_SELF_CHECK_ONLY', 'source_sha256': hashlib.sha256(Path(args.train_py).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'seed': 42, 'max_shape': '8x128, vocab64, sequence32',
              'eval_min_seq_len_test_override': 32, 'checks': checks,
              'limitations': ['Not exact 8x1024 G2', 'No Neuron or NKI execution', 'No real data or BPB', 'Not independent review']}
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
