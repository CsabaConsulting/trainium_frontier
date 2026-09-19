"""Scratch BF16 head/loss boundary; unchanged FP32 loss arithmetic and prediction.

The only GPT training-body edit removes the redundant .float() on already-BF16
head logits. An AST assertion permits exactly that known assignment. Prediction,
evaluation mode, CPU calls, output-bigram models and unsupported losses retain
the original forward. The protected source file and checkpoint state are intact.
"""
import ast
import inspect
import linecache
import textwrap

import torch


def install(train_module):
    if getattr(train_module, '_fresh_start_bf16_ce_installed', False):
        raise RuntimeError('BF16 CE boundary is already installed')
    if train_module.LOSS_BYTE_WEIGHT or train_module.LOGIT_SOFTCAP != 15.0:
        raise ValueError('BF16 CE boundary currently gates unweighted softcap15 only')
    if not getattr(train_module, '_fresh_start_adaptive_ce_installed', False):
        from softcap_ce_parallel_integration import install as install_adaptive
        install_adaptive(train_module)

    original_loss = train_module.softcap_cross_entropy
    original_forward = train_module.GPT.forward
    source = textwrap.dedent(inspect.getsource(original_forward))
    module = ast.parse(source)
    expected = ast.parse('logits = self.lm_head(x)[..., :self.config.vocab_size].float()').body[0]
    matches = [node for node in ast.walk(module)
               if isinstance(node, ast.Assign)
               and ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    if len(matches) != 1:
        raise RuntimeError('Protected GPT head assignment changed; inspect before adapting it')
    assignment = matches[0]
    assignment.value = assignment.value.func.value
    module.body[0].name = '_fresh_start_bf16_training_forward'
    ast.fix_missing_locations(module)
    rewritten = ast.unparse(module) + '\n'
    filename = inspect.getsourcefile(original_forward) + '::<scratch_bf16_head>'
    linecache.cache[filename] = (len(rewritten), None, rewritten.splitlines(True), filename)
    namespace = {}
    exec(compile(rewritten, filename, 'exec'), train_module.__dict__, namespace)
    training_forward = namespace['_fresh_start_bf16_training_forward']

    def loss(logits, targets, reduction):
        rows = logits.numel() // logits.shape[-1]
        if (logits.device.type == 'neuron' and logits.dtype == torch.bfloat16
                and logits.is_contiguous() and rows % 128 == 0
                and logits.shape[-1] % 512 == 0 and reduction in ('mean', 'none')):
            z = logits.reshape(rows, logits.shape[-1])
            target = targets.reshape(-1, 1).float()
            per_token = train_module._SoftcapCrossEntropyNKI.apply(z, target)
            return per_token.mean() if reduction == 'mean' else per_token.view_as(targets)
        return original_loss(logits.float(), targets, reduction)

    def forward(self, idx, targets=None, loss_reduction='mean'):
        if (not self.training or targets is None or idx.device.type != 'neuron'
                or self.out_bigram is not None or loss_reduction not in ('mean', 'none')):
            return original_forward(self, idx, targets, loss_reduction)
        return training_forward(self, idx, targets, loss_reduction)

    train_module.softcap_cross_entropy = loss
    train_module.GPT.forward = forward
    train_module._fresh_start_bf16_ce_installed = True
    return dict(original_forward=original_forward, original_loss=original_loss,
                training_forward=training_forward,
                source_edit='remove exactly one .float() from the known head-logits assignment')
