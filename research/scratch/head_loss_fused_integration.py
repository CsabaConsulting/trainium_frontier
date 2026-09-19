"""Install a restricted compiled-training head/CE branch on one GPT instance.

The original forward body is retained with one guarded return immediately
before the head projection. Prediction, evaluation and unsupported row counts
continue through that original body. No parameters, scorer or source files
change. Installation returns the original bound method for explicit restore.
"""
import ast
import inspect
import textwrap
import types

import torch


@torch.compiler.disable(recursive=False)
def call_separate_region(compiled_function, *inputs):
    """Keep head scratch allocation outside the compiled transformer body."""
    return compiled_function(*inputs)


def install(model, train_module, separate_region=False, save_softcap=False,
            fp32_carrier=False, include_final_norm=False, native_head=False):
    if native_head and not separate_region:
        raise ValueError("Native head adapter is a separate-scope experimental control")
    if fp32_carrier and not separate_region:
        raise ValueError("FP32 carrier is restricted to the explicit head/body boundary")
    if include_final_norm and (not separate_region or fp32_carrier):
        raise ValueError("Final norm belongs to a separate region with the original head interface")
    if getattr(model, "_fresh_start_cached_head_installed", False):
        raise RuntimeError("Cached head adapter already installed")
    if (not train_module.USE_BF16_NORM_OUTPUT or not train_module.USE_WEIGHT_CACHE
            or train_module.LOSS_BYTE_WEIGHT or train_module.LOGIT_SOFTCAP != 15.0
            or model.out_bigram is not None or model.lm_head.weight_transposed
            or model.config.vocab_size != model.lm_head.out_features
            or model.config.vocab_size % 512 or model.config.n_embd not in (256, 1024)):
        raise ValueError("Cached head requires gated BF16 widths, unweighted softcap15 and no output bias")
    if not getattr(train_module, "_fresh_start_adaptive_ce_installed", False):
        raise ValueError("Install the exact adaptive CE reference before the cached head")
    from head_loss_fused_autograd import loss_factory
    if native_head:
        def fused_loss(x, weight, targets, reduction):
            if fp32_carrier:
                x = x.bfloat16()
            return train_module.softcap_cross_entropy(
                torch.nn.functional.linear(x, weight).float(), targets, reduction)
    else:
        fused_loss = loss_factory(save_softcap=save_softcap, fp32_carrier=fp32_carrier)
    if include_final_norm:
        head_loss = fused_loss

        def fused_loss(x, weight, targets, reduction):
            return head_loss(train_module.norm(x), weight, targets, reduction)
    if separate_region:
        compiled_loss = torch.compile(fused_loss, backend="neuron",
                                      fullgraph=True, dynamic=False)

        def fused_loss(*inputs):
            return call_separate_region(compiled_loss, *inputs)
    original = model.forward
    before_ids = {name: id(p) for name, p in model.named_parameters()}
    before_keys = tuple(model.state_dict())
    tree = ast.parse(textwrap.dedent(inspect.getsource(train_module.GPT.forward)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef) and function.name == "forward"
    positions = [i for i, stmt in enumerate(function.body)
                 if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                 and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "logits"]
    assert len(positions) == 1
    position = positions[0]
    assert ast.unparse(function.body[position].value) == \
        "self.lm_head(x)[..., :self.config.vocab_size].float()"
    if include_final_norm:
        position -= 1
        assert ast.unparse(function.body[position]) == "x = norm(x)"

    def model_loss(current, x, targets, reduction):
        weight = current.lm_head._cached_weight
        head_dtype = torch.bfloat16 if include_final_norm else x.dtype
        if weight is None or weight.dtype != head_dtype:
            weight = current.lm_head.weight.to(head_dtype)
        if fp32_carrier:
            x = x.float()
        return fused_loss(x.reshape(-1, x.shape[-1]), weight, targets, reduction)

    guard_source = '''
if (targets is not None and self.training and torch.compiler.is_compiling()
        and x.device.type == "neuron" and x.dtype == torch.bfloat16
        and x.numel() // x.shape[-1] >= 512
        and (x.numel() // x.shape[-1]) % 256 == 0
        and loss_reduction in ("mean", "none")):
    return _fresh_start_cached_head_loss(self, x, targets, loss_reduction)
'''
    if include_final_norm:
        guard_source = guard_source.replace("x.dtype == torch.bfloat16", "x.dtype == torch.float32")
    guard = ast.parse(guard_source).body[0]
    function.body.insert(position, guard)
    ast.fix_missing_locations(tree)
    namespace = dict(vars(train_module))
    namespace["_fresh_start_cached_head_loss"] = model_loss
    exec(compile(tree, __file__ + ":generated_GPT_forward", "exec"), namespace)
    model.forward = types.MethodType(namespace["forward"], model)
    model._fresh_start_cached_head_installed = True
    assert before_ids == {name: id(p) for name, p in model.named_parameters()}
    assert before_keys == tuple(model.state_dict())

    def restore():
        model.forward = original
        delattr(model, "_fresh_start_cached_head_installed")

    return dict(restore=restore, original_forward=original,
                source_edit=("one guarded training return before final norm" if include_final_norm
                             else "one guarded training return before original head projection"),
                transpose_every_forward=True, parameter_identity_unchanged=True,
                separate_compiled_region=bool(separate_region),
                saved_softcap=bool(save_softcap),
                fp32_carrier=bool(fp32_carrier),
                include_final_norm=bool(include_final_norm),
                native_head_control=bool(native_head),
                eager_and_evaluation_use_original=True)
