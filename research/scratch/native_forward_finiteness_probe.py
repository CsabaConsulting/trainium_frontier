#!/usr/bin/env python3
"""Localize a first-forward native NaN; this is not a timing or quality run.

Optional hooks retain intermediate tensors and therefore alter allocation and
scheduling. A finite instrumented graph does not vindicate an uninstrumented
graph. Program-count changes are diagnostic until separately gated in training.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def stats(name, tensor):
    host = tensor.detach().float().cpu()
    finite = torch.isfinite(host)
    good = host[finite]
    return dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                stride=list(tensor.stride()), finite=bool(finite.all()),
                nonfinite=int((~finite).sum()),
                min=None if not good.numel() else float(good.min()),
                max=None if not good.numel() else float(good.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conv", choices=["native", "integrated", "scratch"], default="integrated")
    ap.add_argument("--programs", type=int, choices=[1, 2], default=2)
    ap.add_argument("--precision", choices=["baseline", "norm-bf16", "sdpa-bf16"], default="baseline")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    T.DEPTH, T.N_EMBD, T.PACK_FACTOR = 6, 1024, 4
    T.USE_NGRAM_VE, T.USE_QK_SHIFT = False, False
    T.COMPILE_SDPA_DIRECT = True
    T.USE_NKI_LOCAL_CONV = args.conv == "integrated"
    T.USE_BF16_NORM_OUTPUT = args.precision == "norm-bf16"
    T.USE_BF16_SDPA_INPUT = args.precision == "sdpa-bf16"
    T.USE_COMPILE_MUON, T.USE_NKI_MUON = True, False
    if args.conv == "scratch":
        if args.programs != 2:
            ap.error("one-program isolation uses the integrated kernels")
        from causal_conv_nki2 import nki_causal_conv
        T.CausalConv.forward = nki_causal_conv
    elif args.conv == "integrated" and args.programs == 1:
        class OneProgramConv(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                wt = weight[:, 0, :].T.contiguous().to(x.dtype)
                xx = x.contiguous()
                y = T._conv2_forward_native[1](xx, wt, 1)
                ctx.save_for_backward(xx, wt)
                ctx.lanes, ctx.weight_dtype = 1, weight.dtype
                return y

            @staticmethod
            def backward(ctx, grad):
                # Calling another autograd.Function.backward directly makes
                # Dynamo expect that class's FakeBackwardCFunction. Spell the
                # same pure tensor body here for the diagnostic context.
                xx, wt = ctx.saved_tensors
                dx, partials = T._conv2_backward_native[1](xx, grad.contiguous(), wt, 1)
                dw = partials.sum(dim=(0, 1)).squeeze(-1).to(wt.dtype)
                return dx, dw.T.unsqueeze(1).to(ctx.weight_dtype)

        def single_program(self, x):
            assert self.kernel_size == 2
            return OneProgramConv.apply(x, self.weight)
        T.CausalConv.forward = single_program
    elif args.conv == "native" and args.programs != 2:
        ap.error("--programs applies only to the NKI kernel")

    _, rank, _, world_size, device = T.init_runtime("neuron", seed=42)
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)
    model = T.GPT(T.build_config())
    model.init_weights()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.to(device)
    captured = []
    handles = []
    if args.capture:
        def hook_for(name):
            def hook(module, inputs, output):
                for i, value in enumerate(inputs):
                    if isinstance(value, torch.Tensor):
                        captured.append((f"{name}.input{i}", value.detach()))
                if isinstance(output, torch.Tensor):
                    captured.append((name + ".output", output.detach()))
            return hook
        for name, module in model.named_modules():
            if isinstance(module, (T.Block, T.CausalConv, T.CausalSelfAttention,
                                   T.MLP, T.Linear, torch.nn.Embedding)):
                handles.append(module.register_forward_hook(hook_for(name)))
    tokenizer = T.ensure_tokenizer(build_if_missing=False)
    loader = T.make_dataloader(tokenizer, 1, T.PACK_FACTOR * T.SEQ_LEN, "train", device)
    x, y, _ = next(loader)
    x, y = x.reshape(4, T.SEQ_LEN), y.reshape(4, T.SEQ_LEN)
    fn = model if args.eager else torch.compile(model, backend="neuron", dynamic=False)
    print("FORWARD_PROBE_CONFIG " + json.dumps(dict(
        **vars(args), rank=rank, world_size=world_size, model=T.build_config().__dict__,
        toggles=T.current_toggles(), input_shape=list(x.shape), timing_valid=False)), flush=True)
    passed = True
    try:
        for i in range(args.repeats):
            captured.clear()
            loss = fn(x, y)
            T.synchronize(device)
            record = stats("loss", loss)
            passed = passed and record["finite"]
            print("FORWARD_PROBE_JSON " + json.dumps(dict(repeat=i, **record)), flush=True)
            if i == 0:
                for name, value in captured:
                    print("FORWARD_INTERMEDIATE_JSON " + json.dumps(stats(name, value)), flush=True)
            del loss
        print("FORWARD_PROBE_RESULT " + json.dumps(dict(finite=passed, timing_valid=False)), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        T.cleanup_runtime()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
