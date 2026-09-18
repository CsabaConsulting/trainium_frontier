"""Scratch training integration for the gated compact embedding backward.

Forward lookup, parameters and state-dict names remain ordinary embeddings.
Per-microbatch metadata is shared by WTE and unigram value embeddings. Preparing
it includes copying the input IDs to CPU, constructing the map and uploading it;
call prepare inside the measured microstep. Overflow uses native backward.
"""
import torch
import torch.nn.functional as F

from embedding_compact_probe import build_gather, compact_metadata, dense_mapping


def make_function(device):
    if str(device).startswith("neuron"):
        from torch_neuronx import wrap_nki
        wrapped = wrap_nki(build_gather())

        def expand(small, mapping, vocabulary):
            return wrapped[2](small, mapping, vocabulary, 2)
    else:
        def expand(small, mapping, vocabulary):
            return small[mapping[:, 0].long()] * mapping[:, 1:2]

    class CompactEmbeddingFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, ids, inverse, mapping, bucket):
            ctx.save_for_backward(inverse, mapping)
            ctx.vocabulary, ctx.channels = weight.shape
            ctx.bucket = bucket
            return F.embedding(ids, weight)

        @staticmethod
        def backward(ctx, grad):
            inverse, mapping = ctx.saved_tensors
            small = torch.ops.aten.embedding_dense_backward.default(
                grad.reshape(-1, ctx.channels).contiguous(), inverse, ctx.bucket, -1, False)
            result = expand(small, mapping, ctx.vocabulary)
            return result, None, None, None, None

    return CompactEmbeddingFunction


class MetadataState:
    def __init__(self, vocabulary, device, bucket=4096):
        assert 128 <= bucket <= vocabulary and bucket % 128 == 0
        self.vocabulary = vocabulary
        self.bucket = bucket
        self.device = device
        self.function = make_function(device)
        self.enabled = False
        self.inverse = self.mapping = None
        self.calls = self.compact_calls = self.fallback_calls = self.max_unique = 0

    def prepare(self, ids):
        # No loader modification or hidden free preprocessing. The D2H transfer
        # is deliberately inside each measured training microstep.
        cpu_ids = ids.detach().cpu().reshape(-1)
        inv, rows, unique = compact_metadata(cpu_ids, self.vocabulary, self.bucket)
        self.enabled = unique <= self.bucket
        self.calls += 1
        self.max_unique = max(self.max_unique, unique)
        if self.enabled:
            assert len(rows) == self.bucket
            mapping = dense_mapping(rows, unique, self.vocabulary)
            self.inverse, self.mapping = inv.to(self.device), mapping.to(self.device)
            self.compact_calls += 1
        else:
            # Native full-vocabulary gradient; no truncation, dropped IDs or
            # out-of-range compact rows. First branch compilation is charged.
            self.fallback_calls += 1

    def statistics(self):
        return dict(calls=self.calls, compact_calls=self.compact_calls,
                    fallback_calls=self.fallback_calls, max_unique=self.max_unique,
                    fixed_compact_bucket=self.bucket, vocabulary=self.vocabulary,
                    id_d2h_and_metadata_creation_upload_in_timed_microstep=True)


class CompactEmbedding(torch.nn.Module):
    def __init__(self, original, state):
        super().__init__()
        assert original.padding_idx is None and original.max_norm is None
        assert not original.scale_grad_by_freq and not original.sparse
        assert original.num_embeddings == state.vocabulary
        assert original.weight.dtype == torch.float32
        self.weight = original.weight
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        # Ordinary Python state is shared, not a registered parameter/buffer.
        # The checkpoint contains exactly the original embedding weight names.
        self.state = state

    def forward(self, ids):
        if self.training and self.state.enabled:
            return self.state.function.apply(self.weight, ids, self.state.inverse,
                                             self.state.mapping, self.state.bucket)
        return F.embedding(ids, self.weight)


def install(model, device, bucket=4096):
    keys = tuple(model.state_dict())
    parameter_ids = tuple(id(p) for p in model.parameters())
    state = MetadataState(model.transformer.wte.num_embeddings, device, bucket)
    model.transformer.wte = CompactEmbedding(model.transformer.wte, state)
    for key, embedding in tuple(model.value_embeds.items()):
        model.value_embeds[key] = CompactEmbedding(embedding, state)
    assert tuple(model.state_dict()) == keys
    assert tuple(id(p) for p in model.parameters()) == parameter_ids
    return state
