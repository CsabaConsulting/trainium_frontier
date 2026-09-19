#!/usr/bin/env python3
"""Exact caller-gradient and native-evaluation gates for cached-head integration."""
import argparse
import copy
import gc
import io
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from head_loss_fused_integration import install
from mlp_fused_integration import install as install_mlp
from softcap_ce_parallel_integration import install as install_ce


def emit(kind, value):
    print(kind + " " + json.dumps(value), flush=True)


def refresh(model):
    for module in model.modules():
        if isinstance(module, T.Linear):
            module.refresh_weight_cache(torch.bfloat16)


def equal(name, first, second):
    a, b = first.detach().cpu(), second.detach().cpu()
    passed = torch.equal(a, b) and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    record = dict(name=name, shape=list(a.shape), dtype=str(a.dtype), exact=torch.equal(a, b),
                  max_abs=float((a.float() - b.float()).abs().max()), passed=passed)
    emit("CACHED_HEAD_MODEL_EXACT", record)
    assert passed, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-width", action="store_true")
    args = ap.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 32
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = T.USE_NKI_LOCAL_CONV = True
    T.USE_NGRAM_VE = T.USE_OUT_BIGRAM = T.USE_ENGRAM = False
    # Match the current training path. The repository's other attention
    # wrapper is compiler-disabled and cannot be traced by this fullgraph gate.
    T.COMPILE_SDPA_DIRECT = True
    T.USE_NKI_ATTN = T.USE_BF16_SDPA_INPUT = T.USE_QK_SHIFT = False
    T.QK_SHIFT_BETA = 0.0
    T.QK_SHIFT_FREEZE = True
    T.COMPUTE_DTYPE = torch.bfloat16
    install_ce(T)
    torch.manual_seed(10491)
    generator = torch.Generator().manual_seed(20931)
    for channels in ((256, 1024) if args.real_width else (256,)):
        config = T.GPTConfig(sequence_len=256, vocab_size=8192, n_layer=2,
                             n_head=channels // 128, n_embd=channels, mlp_ratio=4)
        baseline = T.GPT(config)
        baseline.init_weights()
        # Nonzero head and output branches make the body-gradient gate
        # meaningful at initialization, rather than checking all-zero paths.
        with torch.no_grad():
            baseline.lm_head.weight.copy_(torch.randn(baseline.lm_head.weight.shape,
                                                      generator=generator) * .03)
            for block in baseline.transformer.h:
                for layer in (block.attn.c_proj, block.mlp.c_proj):
                    layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * .02)
        candidate = copy.deepcopy(baseline)
        baseline.to("neuron"); candidate.to("neuron")
        for model in (baseline, candidate):
            install_mlp(model, "neuron", T, forward_backend="native")
        before_rng = torch.get_rng_state().clone()
        contract = install(candidate, T)
        assert torch.equal(before_rng, torch.get_rng_state())
        before_keys = tuple(baseline.state_dict())
        assert before_keys == tuple(candidate.state_dict())
        native, fused = (torch.compile(model, backend="neuron", fullgraph=True, dynamic=False)
                         for model in (baseline, candidate))
        ids = torch.randint(8192, (2, 256), generator=generator, dtype=torch.int32).to("neuron")
        targets = torch.randint(8192, (2, 256), generator=generator, dtype=torch.int64).to("neuron")
        for step in range(2):
            for model in (baseline, candidate):
                model.train(); model.zero_grad(set_to_none=True); refresh(model)
            for micro in range(2):
                tokens = (ids + micro * 7) % 8192
                values = []
                for function in (native, fused):
                    value = function(tokens, targets)
                    assert type(value.grad_fn).__name__ == "CompiledFunctionBackward"
                    value.backward()
                    values.append(value)
                equal(f"{channels}:{step}:{micro}:loss", *values)
            pairs = zip(baseline.named_parameters(), candidate.named_parameters())
            checked, absent = 0, []
            for (name, p), (other, q) in pairs:
                assert name == other and p.dtype == q.dtype == torch.float32
                if p.grad is None or q.grad is None:
                    assert p.grad is None and q.grad is None, name
                    absent.append(name)
                    continue
                assert p.grad.dtype == q.grad.dtype == torch.float32
                equal(f"{channels}:{step}:master_gradient:{name}", p.grad, q.grad)
                checked += 1
            assert checked > 10
            emit("CACHED_HEAD_MODEL_GRADIENTS", dict(channels=channels, step=step,
                exact_master_gradients=checked, absent_in_both=absent, passed=True))
            with torch.no_grad():
                for p, q in zip(baseline.parameters(), candidate.parameters()):
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-.001)
                        q.add_(q.grad, alpha=-.001)
                    equal(f"{channels}:{step}:updated_parameter", p, q)
        baseline.eval(); candidate.eval(); refresh(baseline); refresh(candidate)
        with torch.no_grad():
            short = ids[:, :64]
            a = baseline(short)
            b = candidate(short)
            equal(f"{channels}:native_eval", a, b)
            changed = short.clone(); changed[:, 32:] = (changed[:, 32:] + 1) % 8192
            equal(f"{channels}:eval_future_independence", b[:, :32], candidate(changed)[:, :32])
            changed = short.clone(); changed[0] = (changed[0] + 1) % 8192
            equal(f"{channels}:eval_batch_independence", b[1:], candidate(changed)[1:])
            equal(f"{channels}:eval_repeat", b, candidate(short))
            stream = io.BytesIO()
            torch.save({k: v.detach().cpu() for k, v in candidate.state_dict().items()}, stream)
            stream.seek(0)
            restored = T.GPT(config)
            restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
            restored.to("neuron").eval(); refresh(restored)
            equal(f"{channels}:native_checkpoint_roundtrip", b, restored(short))
            contract["restore"]()
            equal(f"{channels}:explicit_adapter_restore", b, candidate(short))
        emit("CACHED_HEAD_MODEL_RESULT", dict(channels=channels, passed=True,
            two_microbatch_accumulation=True, two_master_updates=True,
            cache_refresh=True, exact_native_gradients=True, original_eval=True,
            checkpoint_roundtrip=True, causality_batch_repeat=True,
            protected_train_source_unchanged=True, whole_step_or_quality_promotion=False))
        del baseline, candidate, native, fused, restored, stream
        gc.collect()


if __name__ == "__main__":
    main()
