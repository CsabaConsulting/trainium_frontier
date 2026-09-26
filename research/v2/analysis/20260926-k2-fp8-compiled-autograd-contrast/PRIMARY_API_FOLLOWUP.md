# Registration route: documented hypothesis, not diagnosed cause

Read-only primary-source research, 2026-09-26 00:34 UTC.

AWS's versioned native PyTorch/Neuron guide describes `nki_op` registration
and explicit `register_autograd` for training. Its appendix labels the older
`torch.autograd.Function` pattern less recommended. This makes explicit
operator/autograd registration a plausible next integration experiment,
not evidence that our current wrapper must fail or that registration fixes it.
[AWS Neuron 2.31 native PyTorch guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/frameworks/torch/pytorch-native-overview.html#neuron-kernel-interface-nki-integration)

PyTorch's Triton integration tutorial likewise prefers registered autograd
for compile composability and requires custom kernels called by backward
to be represented through appropriate operators. That is a general design
principle, not a portable Triton-on-Neuron implementation. The viewed tutorial
is current2.14 documentation; our native Torch2.11 environment must be checked
against its installed API rather than assumed compatible.
[PyTorch training-registration guidance](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html#adding-training-support)

Current source-neutral FP32-first diagnostic remains byte-frozen. Do not
change its wrapper, relax exactness, change SDK installations, or infer the
two earlier K2/FP8 failures share a root cause. If its ordinary compiled
control passes while custom still fails, a separately proposed registered-op
wrapper can keep kernel/math identical and test only integration. First read
already archived installed-API evidence, then CPU reference/negative tests,
fresh independent review, and one bounded native gate. Full20-case coverage
and whole-step timing remain required before production integration. This
note authorizes neither host work nor a new patch.
