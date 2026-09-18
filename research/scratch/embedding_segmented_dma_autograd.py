"""Scratch shared-metadata integration of the gated three-kernel backward.

Ordinary embedding forward/parameters/evaluation are preserved. Preparation is
explicit and must run inside every timed microstep: input IDs D2H, stable CPU
sort and three metadata uploads. All four ordinary tables reuse those tensors.
Unsupported token counts use the original native embedding backward.
"""
import torch
import torch.nn.functional as F

from embedding_segmented_dma import build_kernels, dma_metadata
from embedding_segmented_nki import cpu_scan


def make_function(device):
    if str(device).startswith("neuron"):
        from torch_neuronx import wrap_nki
        permute, scan, expand = (wrap_nki(k) for k in build_kernels())

        def reduce(grad, permutation, continuation, mapping, vocabulary):
            ordered = permute[2](grad, permutation, 2, True)
            scanned = scan[2](ordered, continuation, 2, True)
            return expand[2](scanned, mapping, vocabulary, 2)
    else:
        def reduce(grad, permutation, continuation, mapping, vocabulary):
            # CPU autograd has no Neuron BF16 matmul lowering. Preserve its
            # FP32 inputs here; device tests separately bound the BF16 policy.
            return cpu_scan(grad, (permutation.reshape(1, -1), continuation,
                                  mapping[:, 0].reshape(1, -1),
                                  mapping[:, 1].reshape(1, -1)))

    class SegmentedEmbeddingFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, ids, permutation, continuation, mapping):
            ctx.save_for_backward(permutation, continuation, mapping)
            ctx.vocabulary, ctx.channels = weight.shape
            return F.embedding(ids, weight)

        @staticmethod
        def backward(ctx, grad):
            permutation, continuation, mapping = ctx.saved_tensors
            result = reduce(grad.reshape(-1, ctx.channels).contiguous(),
                            permutation, continuation, mapping, ctx.vocabulary)
            return result, None, None, None, None

    return SegmentedEmbeddingFunction


class MetadataState:
    def __init__(self, vocabulary, device):
        assert vocabulary % 128 == 0
        self.vocabulary = vocabulary
        self.device = device
        self.function = make_function(device)
        self.enabled = False
        self.permutation = self.continuation = self.mapping = None
        self.calls = self.segmented_calls = self.fallback_calls = 0

    def prepare(self, ids):
        tokens = ids.numel()
        self.calls += 1
        self.enabled = 0 < tokens <= 8192 and tokens % 128 == 0
        if self.enabled:
            cpu_ids = ids.detach().cpu().reshape(-1)
            tensors = dma_metadata(cpu_ids, self.vocabulary)
            self.permutation, self.continuation, self.mapping = (
                t.to(self.device) for t in tensors)
            self.segmented_calls += 1
        else:
            self.fallback_calls += 1

    def statistics(self):
        return dict(calls=self.calls, segmented_calls=self.segmented_calls,
                    fallback_calls=self.fallback_calls, vocabulary=self.vocabulary,
                    id_d2h_and_metadata_creation_upload_in_timed_microstep=True,
                    backward="indirect_dma_bf16_permutation_fp32_scan_dense_expansion")


class SegmentedEmbedding(torch.nn.Module):
    def __init__(self, original, state):
        super().__init__()
        assert original.padding_idx is None and original.max_norm is None
        assert not original.scale_grad_by_freq and not original.sparse
        assert original.num_embeddings == state.vocabulary
        assert original.embedding_dim % 256 == 0
        assert original.weight.dtype == torch.float32
        self.weight = original.weight
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.state = state

    def forward(self, ids):
        if self.training and self.state.enabled:
            return self.state.function.apply(self.weight, ids, self.state.permutation,
                                             self.state.continuation, self.state.mapping)
        return F.embedding(ids, self.weight)


def install(model, device):
    keys = tuple(model.state_dict())
    parameter_ids = tuple(id(p) for p in model.parameters())
    state = MetadataState(model.transformer.wte.num_embeddings, device)
    model.transformer.wte = SegmentedEmbedding(model.transformer.wte, state)
    for key, embedding in tuple(model.value_embeds.items()):
        model.value_embeds[key] = SegmentedEmbedding(embedding, state)
    assert tuple(model.state_dict()) == keys
    assert tuple(id(p) for p in model.parameters()) == parameter_ids
    return state
