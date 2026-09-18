"""CPU-only real-attention equivalence and Dynamo graph-partition diagnostic.

    python research/scratch/fresh_start_attention_graph_check.py

Never selects or initializes a Neuron device. Checks the real CausalSelfAttention
module in FP32 and BF16, input/parameter gradients, explicit-mask invariance,
custom-wrapper precedence, and direct-path fullgraph capture with backend='eager'.
Reports CPU graph counts; these do not establish a Neuron speedup or prove which
native backend kernel executes. This is the gate before hardware profiling.
"""

import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def make_attention(dtype, scale=0.0):
    torch.manual_seed(20260918)
    config = T.GPTConfig(sequence_len=128, vocab_size=8192, n_layer=2,
                         n_head=2, n_embd=256, attn_scale=scale)
    model = T.CausalSelfAttention(config)
    x = torch.randn(2, 128, 256, dtype=dtype)
    cos = torch.ones(1, 128, 1, 64, dtype=dtype)
    sin = torch.zeros_like(cos)
    grad = torch.randn_like(x)
    return model, x, cos, sin, grad


def evaluate(model, x, cos, sin, output_grad, direct, custom, mask):
    T.COMPILE_SDPA_DIRECT = direct
    T.USE_NKI_ATTN = custom
    inputs = [x.detach().clone().requires_grad_(True) for _ in range(3)]
    out = model(*inputs, None, cos, sin, mask)
    grads = torch.autograd.grad(out, [*inputs, *model.parameters()], output_grad)
    return out.detach(), [g.detach() for g in grads]


def graph_record(model, x, cos, sin, direct, custom, mask=None):
    T.COMPILE_SDPA_DIRECT = direct
    T.USE_NKI_ATTN = custom
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model)(x, x, x, None, cos, sin, mask)
    return {"direct": direct, "custom": custom, "masked": mask is not None,
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": [str(reason.reason) for reason in explanation.break_reasons]}


def main():
    torch.set_num_threads(1)
    assert T.COMPILE_SDPA_DIRECT is False
    assert T.current_toggles()["COMPILE_SDPA_DIRECT"] is False
    assert "USE_NKI_ATTN" in T.current_toggles()
    # A small real attention module, with optional VE/gates absent so this isolates
    # the requested SDPA route and never allocates the current huge n-gram tables.
    overrides = {"USE_VE": False, "USE_NGRAM_VE": False, "USE_HEAD_GATE": False,
                 "USE_FUSED_QKV": False, "USE_WEIGHT_CACHE": False}
    original = {name: getattr(T, name) for name in overrides}
    original.update(COMPILE_SDPA_DIRECT=T.COMPILE_SDPA_DIRECT, USE_NKI_ATTN=T.USE_NKI_ATTN)
    try:
        for name, value in overrides.items():
            setattr(T, name, value)
        numerical_cases = 0
        for dtype in (torch.float32, torch.bfloat16):
            for scale in (0.0, 0.05):
                model, x, cos, sin, grad = make_attention(dtype, scale)
                causal = torch.triu(torch.full((128, 128), float("-inf")), diagonal=1)
                for mask in (None, causal):
                    for custom in (False, True):
                        old_out, old_grads = evaluate(model, x, cos, sin, grad,
                                                     False, custom, mask)
                        new_out, new_grads = evaluate(model, x, cos, sin, grad,
                                                     True, custom, mask)
                        assert torch.equal(old_out, new_out), (dtype, scale, mask is not None, custom)
                        assert all(torch.equal(old, new) for old, new in zip(old_grads, new_grads))
                        numerical_cases += 1

        model, x, cos, sin, _ = make_attention(torch.float32)
        records = [graph_record(model, x, cos, sin, direct, custom)
                   for direct, custom in ((False, False), (True, False), (True, True))]
        historical, direct, custom = records
        assert historical["graph_break_count"] >= 1, historical
        assert direct["graph_count"] == 1 and direct["graph_break_count"] == 0, direct
        assert custom["graph_break_count"] >= 1, custom

        # Explicit masks already call SDPA directly and must not gain graph breaks.
        causal = torch.triu(torch.full((128, 128), float("-inf")), diagonal=1)
        masked = [graph_record(model, x, cos, sin, flag, False, causal) for flag in (False, True)]
        assert masked[0]["graph_count"] == masked[1]["graph_count"] == 1
        assert masked[0]["graph_break_count"] == masked[1]["graph_break_count"] == 0

        T.COMPILE_SDPA_DIRECT = True
        T.USE_NKI_ATTN = False
        compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
        got = compiled(x, x, x, None, cos, sin, None)
        expected = model(x, x, x, None, cos, sin, None)
        assert torch.equal(got, expected)
        print(json.dumps({"status": "PASS", "numerical_cases": numerical_cases,
                          "outputs_and_gradients_bit_exact": True,
                          "causal_graphs": records, "explicit_mask_graphs": masked,
                          "direct_fullgraph_eager_backend": "PASS",
                          "device": "cpu", "hardware_speed_claim": False}), flush=True)
    finally:
        for name, value in original.items():
            setattr(T, name, value)
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
