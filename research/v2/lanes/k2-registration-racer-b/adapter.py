"""Disabled standalone K2 registration experiment; never imported by training.

The two injected dependencies exist solely for bounded CPU contract tests.
Without injection the kernel is the exact sibling build_kernel(enabled=True).
This file does not establish native correctness or permit a native launch.
"""


def _register(kernel, output_dtype, nki_op, name):
    import torch
    import torch.nn.functional as F

    @nki_op(name, mutates_args=())
    def embedding(weight: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor,
                  starts: torch.Tensor, positions: torch.Tensor,
                  maximums: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, weight).to(output_dtype)

    def setup_context(ctx, inputs, output):
        weight, ids, counts, starts, positions, maximums = inputs
        ctx.save_for_backward(counts, starts, positions, maximums)

    def backward(ctx, gradient):
        counts, starts, positions, maximums = ctx.saved_tensors
        result = kernel(gradient.contiguous(), counts, starts, positions, maximums)
        return result, None, None, None, None, None

    embedding.register_autograd(backward, setup_context=setup_context)
    return embedding


def original_failure_control(kernel, output_dtype):
    """Original DiagnosticEmbedding class verbatim; separately named control."""
    import torch
    import torch.nn.functional as F

    class DiagnosticEmbedding(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, ids, counts, starts, positions, maximums):
            ctx.save_for_backward(counts, starts, positions, maximums)
            return F.embedding(ids, weight).to(output_dtype)

        @staticmethod
        def backward(ctx, gradient):
            counts, starts, positions, maximums = ctx.saved_tensors
            result = kernel(gradient.contiguous(), counts, starts, positions, maximums)
            return result, None, None, None, None, None

    def enclosing(weight, ids, counts, starts, positions, maximums):
        return DiagnosticEmbedding.apply(weight, ids, counts, starts, positions, maximums)

    return enclosing


def build(*, enabled=False, dtype="fp32", test_kernel_factory=None,
          test_nki_op=None):
    """Explicit opt-in only; disabled calls import neither Torch nor NKI.

    Return the registered candidate and the named original failure control.
    Build once per dtype per process: diagnostic names are fixed, no cache.
    CPU substitutes require both test dependencies, never partial injection.
    """
    if not enabled:
        raise RuntimeError("EXPERIMENT_DISABLED: no Torch/NKI module imported")
    if dtype not in ("fp32", "bf16"):
        raise ValueError("Only fp32 and bf16 are in the proposal")
    if (test_kernel_factory is None) != (test_nki_op is None):
        raise ValueError("CPU test dependencies must be supplied together")
    import torch

    if test_kernel_factory is None:
        import importlib.util
        from pathlib import Path
        from torch_neuronx import nki_op

        source = Path(__file__).resolve().parent.parent / "k2-symmetric-consumer" / "consumer_nki.py"
        spec = importlib.util.spec_from_file_location("k2_racer_b_immutable_consumer", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = module.build_kernel(enabled=True)
    else:
        nki_op = test_nki_op
        kernel = test_kernel_factory(enabled=True)
    output_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
    candidate = _register(kernel, output_dtype, nki_op,
                          "k2_registration_racer_b::embedding_" + dtype)
    return candidate, original_failure_control(kernel, output_dtype)


if __name__ == "__main__":
    build()
