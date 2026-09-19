"""Opt-in serial-attention projection adapter retaining the native attention body.

All original Linear Parameters/caches, normalization inputs and optimizer groups
remain separate. The original CausalSelfAttention.forward consumes the three
contiguous preprojected tensors through its existing qkv argument. No parallel
block, fused Parameter, shared normalization derivative or new state is created.
Hardware lifecycle/attention gates are required before whole-model use.
"""
import types

import torch
import torch.nn.functional as F

from serial_qkv_nki import build_kernel, make_autograd


def make_forward(native_forward, use_weight_cache, layout):
    from torch_neuronx import wrap_nki
    kernel = wrap_nki(build_kernel())
    token_major = layout == 'tokens'

    def call(xq, xk, xv, wq, wk, wv):
        rows = xq.shape[0]
        tile = 128 if token_major else (512 if rows % 1024 == 0 else 256 if rows % 512 == 0 else 128)
        return kernel[2](xq, xk, xv, wq, wk, wv, 2, tile, token_major)

    apply = make_autograd(call)

    def weight(linear, dtype):
        cache = linear._cached_weight
        if use_weight_cache and cache is not None and cache.dtype == dtype:
            return cache
        return linear.weight.to(dtype=dtype)

    def forward(self, x_q, x_k, x_v, ve, cos, sin, attn_mask,
                ve_bi=None, ve_tri=None, qkv=None):
        if not self.training or qkv is not None:
            return native_forward(self, x_q, x_k, x_v, ve, cos, sin, attn_mask,
                                  ve_bi, ve_tri, qkv)
        if (x_q.dtype != torch.bfloat16 or x_k.dtype != x_q.dtype or x_v.dtype != x_q.dtype
                or x_q.shape != x_k.shape or x_q.shape != x_v.shape):
            raise ValueError('Serial QKV requires three equally shaped, already normalized BF16 streams')
        batch, length, channels = x_q.shape
        q_rows, k_rows, v_rows = (x.reshape(-1, channels).contiguous() for x in (x_q, x_k, x_v))
        rows = batch * length
        padding = (-rows) % 256
        if padding:
            q_rows, k_rows, v_rows = (F.pad(x, (0, 0, 0, padding)) for x in (q_rows, k_rows, v_rows))
        outputs = apply(q_rows, k_rows, v_rows, weight(self.c_q, x_q.dtype),
                        weight(self.c_k, x_k.dtype), weight(self.c_v, x_v.dtype))
        projections = tuple(y[:rows].reshape(batch, length, self.n_head, self.head_dim)
                            for y in outputs)
        assert all(y.is_contiguous() for y in projections)
        # Preserve the supplied pre-convolution/normalized streams and pass
        # projection outputs into the unchanged native attention implementation.
        return native_forward(self, x_q, x_k, x_v, ve, cos, sin, attn_mask,
                              ve_bi, ve_tri, projections)

    return forward


def install(model, device, train_module, layout):
    if layout not in ('tokens', 'channels') or not str(device).startswith('neuron'):
        raise ValueError('Serial QKV requires a declared Neuron layout')
    if train_module.USE_FUSED_QKV or train_module.USE_PARALLEL_BLOCK:
        raise ValueError('The preserved-Parameter experiment requires the ordinary serial block')
    if not train_module.USE_BF16_NORM_OUTPUT:
        raise ValueError('The current kernel requires the separately gated BF16 normalization boundary')
    keys = tuple(model.state_dict())
    parameter_ids = tuple(id(p) for p in model.parameters())
    rng = torch.get_rng_state().clone()
    forwards = {}
    layers = []
    for index, block in enumerate(model.transformer.h):
        attn = block.attn
        if (attn.coalesced != 'none' or attn.n_kv_heads != attn.n_head
                or attn.n_embd not in (128, 256, 512, 1024)):
            raise ValueError('Only independently gated square dense projections are supported')
        for linear in (attn.c_q, attn.c_k, attn.c_v):
            assert not linear.weight_transposed and linear.bias is None
            assert tuple(linear.weight.shape) == (attn.n_embd, attn.n_embd)
        if type(attn) not in forwards:
            forwards[type(attn)] = make_forward(type(attn).forward, train_module.USE_WEIGHT_CACHE, layout)
        attn.forward = types.MethodType(forwards[type(attn)], attn)
        layers.append(index)
    assert tuple(model.state_dict()) == keys
    assert tuple(id(p) for p in model.parameters()) == parameter_ids
    assert torch.equal(rng, torch.get_rng_state())
    return dict(layout=layout, layers=layers, separate_parameter_objects=True,
                original_parameter_shapes_and_checkpoint_keys=True,
                original_weight_refresh=True, distinct_normalized_inputs=True,
                contiguous_qkv_outputs=True, native_attention_body=True,
                native_evaluation=True, short_inputs_pad_projection_rows_only=True,
                whole_model_or_quality_claim=False)
