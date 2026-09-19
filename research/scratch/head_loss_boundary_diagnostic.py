#!/usr/bin/env python3
"""Distinguish a head-kernel error from a native compiled-boundary effect.

This is a diagnostic, not a relaxed replacement for the exact GPT gate. The
original failed gate remains authoritative. Compare native/cached heads under
both allocation scopes on identical fixed model states and one ordinary
backward. Require exact native/cached agreement within each scope, and report
every monolithic/split gradient difference without calling it a numerical pass.
"""
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
import head_loss_fused_autograd as head_ops
from head_loss_fused_integration import install
from head_loss_fused_model_probe import compiled_nodes, refresh
from mlp_fused_integration import install as install_mlp
from softcap_ce_parallel_integration import install as install_ce


def emit(kind, value):
    print(kind + " " + json.dumps(value), flush=True)


def compare(name, left, right, require_exact):
    assert left.keys() == right.keys()
    records = []
    for key in left:
        a, b = left[key], right[key]
        if a is None or b is None:
            assert a is None and b is None, key
            continue
        assert a.dtype == b.dtype == torch.float32
        finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        delta = a.double() - b.double()
        norm = float(a.double().square().sum().sqrt())
        record = dict(comparison=name, field=key, shape=list(a.shape),
                      finite=finite, exact=torch.equal(a, b),
                      differing_elements=int((a != b).sum()),
                      max_abs=float(delta.abs().max()),
                      relative_l2=float(delta.square().sum().sqrt()) / max(norm, 1e-300))
        emit("HEAD_BOUNDARY_DIFFERENCE", record)
        assert finite
        if require_exact:
            assert record["exact"], record
        records.append(record)
    return dict(name=name, exact=all(row["exact"] for row in records),
                differing_fields=[row["field"] for row in records if not row["exact"]])


def native_head_factory(save_softcap=False, fp32_carrier=False):
    assert not save_softcap and not fp32_carrier

    def loss(x, weight, targets, reduction="mean"):
        return T.softcap_cross_entropy(torch.nn.functional.linear(x, weight).float(),
                                       targets, reduction)
    return loss


def main():
    import torch_neuronx
    torch.set_num_threads(4)
    torch._dynamo.config.recompile_limit = 32
    T.USE_WEIGHT_CACHE = T.USE_BF16_NORM_OUTPUT = T.USE_NKI_LOCAL_CONV = True
    T.USE_NGRAM_VE = T.USE_OUT_BIGRAM = T.USE_ENGRAM = False
    T.COMPILE_SDPA_DIRECT = True
    T.USE_NKI_ATTN = T.USE_BF16_SDPA_INPUT = T.USE_QK_SHIFT = False
    T.QK_SHIFT_BETA = 0.0
    T.QK_SHIFT_FREEZE = True
    T.COMPUTE_DTYPE = torch.bfloat16
    install_ce(T)
    torch.manual_seed(10491)
    generator = torch.Generator().manual_seed(20931)
    channels = 256
    config = T.GPTConfig(sequence_len=256, vocab_size=8192, n_layer=2,
                         n_head=channels // 128, n_embd=channels, mlp_ratio=4)
    initial = T.GPT(config)
    initial.init_weights()
    with torch.no_grad():
        initial.lm_head.weight.copy_(torch.randn(initial.lm_head.weight.shape,
                                                generator=generator) * .03)
        for block in initial.transformer.h:
            for layer in (block.attn.c_proj, block.mlp.c_proj):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * .02)
    ids = torch.randint(8192, (2, 256), generator=generator, dtype=torch.int32).to("neuron")
    targets = torch.randint(8192, (2, 256), generator=generator, dtype=torch.int64).to("neuron")
    results = {}
    for name in ("native_monolithic", "cached_monolithic", "native_split", "cached_split"):
        model = copy.deepcopy(initial).to("neuron")
        install_mlp(model, "neuron", T, forward_backend="native")
        split = name.endswith("_split")
        if name == "native_split":
            factory = head_ops.loss_factory
            try:
                head_ops.loss_factory = native_head_factory
                install(model, T, separate_region=True)
            finally:
                head_ops.loss_factory = factory
        elif name.startswith("cached"):
            install(model, T, separate_region=split)
        model.train(); model.zero_grad(set_to_none=True); refresh(model)
        optimized = torch.compile(model, backend="neuron", fullgraph=not split, dynamic=False)
        loss = optimized(ids, targets)
        count = compiled_nodes(loss).count("CompiledFunctionBackward")
        assert count >= (2 if split else 1)
        loss.backward()
        snapshot = {key: (parameter.grad.detach().cpu().clone()
                          if parameter.grad is not None else None)
                    for key, parameter in model.named_parameters()}
        snapshot["loss"] = loss.detach().cpu()
        results[name] = snapshot
        emit("HEAD_BOUNDARY_CASE", dict(name=name, compiled_backward_nodes=count,
                                        loss=float(snapshot["loss"])))
    controls = [compare("native_vs_cached_monolithic", results["native_monolithic"],
                        results["cached_monolithic"], True),
                compare("native_vs_cached_split", results["native_split"],
                        results["cached_split"], True)]
    cross = compare("native_monolithic_vs_native_split", results["native_monolithic"],
                     results["native_split"], False)
    emit("HEAD_BOUNDARY_DIAGNOSTIC_RESULT", dict(
        same_scope_native_cached_controls=controls, native_boundary_effect=cross,
        exact_original_model_gate_replaced=False, speed_or_quality_promotion=False))


if __name__ == "__main__":
    main()
