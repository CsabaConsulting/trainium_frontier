#!/usr/bin/env python3
"""Unprofiled per-phase step timing for train.py. Imports train.py; never modifies it.

Splits a real steady-state step into:
  zero_grad | grad_accum x microstep | sync_gradients | set_lr | optimizer.step | weight_cache_refresh
so the fixed-overhead-vs-compute split can be read directly, at any world_size/LNC.
"""

import argparse
import json
import os
import sys
import time

REPO = os.environ.get("PROF_REPO", "/home/ubuntu/trainium_frontier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--local-conv", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--per-micro", action=argparse.BooleanOptionalAction, default=False,
                    help="synchronize after each microstep (adds sync cost, gives per-micro times)")
    ap.add_argument("--foreach-optim", action=argparse.BooleanOptionalAction, default=False,
                    help="MuonAdamW batched-dispatch path (train.py USE_FOREACH_OPTIM): "
                         "torch._foreach_* over each param group instead of a Python "
                         "per-parameter loop. Bit-identical update math.")
    ap.add_argument("--compile-optim", action=argparse.BooleanOptionalAction, default=False,
                    help="ALSO torch.compile(optimizer.step) separately from the model. "
                         "Approach 2 of the optimizer-dispatch investigation. Expected to "
                         "misbehave on this stack: MuonAdamW's bias corrections are Python "
                         "floats derived from an integer step counter that changes every "
                         "step, which is the same graph-constant/recompile trap "
                         "USE_TENSOR_LR_SCALARS was introduced to fix for the lr. Measured "
                         "rather than assumed.")
    ap.add_argument("--conv-mode", choices=["normal", "zeros"], default="normal",
                    help="zeros: monkeypatch CausalConv.forward to return zeros_like(x). "
                         "Mathematically identical because the weight is provably all-zero, "
                         "but skips pad+conv1d+2 transposes. train.py is NOT modified.")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    os.chdir(REPO)

    import torch
    import train as T

    T.USE_LOCAL_CONV = args.local_conv
    T.USE_TENSOR_LR_SCALARS = True
    T.USE_FOREACH_OPTIM = args.foreach_optim
    T.USE_NKI_ROPE_NORM = True
    T.USE_NKI_SOFTCAP_CE = True

    if args.conv_mode == "zeros":
        def _zero_forward(self, x):
            return torch.zeros_like(x)
        T.CausalConv.forward = _zero_forward
        print("[pt] MONKEYPATCH: CausalConv.forward -> zeros_like(x)", flush=True)

    device_type = T.detect_device_type("")
    ddp, rank, local_rank, world_size, device = T.init_runtime(device_type)
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)

    config = T.build_config()
    model = T.GPT(config)
    model.init_weights()
    model.to(device)
    if args.compile:
        model = torch.compile(model, backend="neuron" if device_type == "neuron" else "inductor",
                              dynamic=False)
    optimizer = model.setup_optimizer()
    if args.compile_optim:
        _raw_step = optimizer.step
        optimizer.step = torch.compile(
            _raw_step, backend="neuron" if device_type == "neuron" else "inductor",
            dynamic=False)
        print("[pt] optimizer.step wrapped in torch.compile", flush=True)

    micro_tokens = T.DEVICE_BATCH_SIZE * T.PACK_FACTOR * T.SEQ_LEN * world_size
    grad_accum = T.TOTAL_BATCH_SIZE // micro_tokens
    if rank == 0:
        print(f"[pt] world_size={world_size} LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} "
              f"micro_tokens={micro_tokens:,} grad_accum={grad_accum} "
              f"local_conv={args.local_conv} conv_mode={args.conv_mode} compile={args.compile} "
              f"foreach_optim={T.USE_FOREACH_OPTIM} "
              f"weight_cache={T.USE_WEIGHT_CACHE}", flush=True)

    rows = T.DEVICE_BATCH_SIZE * T.PACK_FACTOR
    gen = torch.Generator().manual_seed(1234 + rank)
    cpu_x = torch.randint(0, config.vocab_size, (rows, T.SEQ_LEN), dtype=torch.int32, generator=gen)
    cpu_y = torch.randint(0, config.vocab_size, (rows, T.SEQ_LEN), dtype=torch.long, generator=gen)
    x = cpu_x.to(device)
    y = cpu_y.to(device)

    linears = [m for m in model.modules() if isinstance(m, T.Linear)]

    def micro():
        loss = model(x, y)
        (loss / grad_accum).backward()

    def set_lr(step):
        lrm = 1.0 - 0.001 * step
        wd_progress = min(1.0, step / 1000.0)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = T.muon_momentum(step)
                group["weight_decay"] = (T.WEIGHT_DECAY * (1.0 - wd_progress)
                                         if T.WD_DECAY_TO_ZERO else T.WEIGHT_DECAY)

    def refresh_cache():
        if T.USE_WEIGHT_CACHE:
            for m in linears:
                m.refresh_weight_cache(T.COMPUTE_DTYPE)

    for s in range(args.warmup_steps):
        t = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            micro()
        T.sync_gradients(model)
        set_lr(s)
        optimizer.step()
        refresh_cache()
        T.synchronize(device)
        if rank == 0:
            print(f"[pt] warmup {s} dt={time.monotonic() - t:.3f}s", flush=True)

    out = []
    for s in range(args.steps):
        base = args.warmup_steps + 1 + s
        rec = {}
        T.synchronize(device)
        t_step = time.monotonic()

        t = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        T.synchronize(device)
        rec["zero_grad_ms"] = (time.monotonic() - t) * 1e3

        t = time.monotonic()
        micros = []
        for i in range(grad_accum):
            tm = time.monotonic()
            micro()
            if args.per_micro:
                T.synchronize(device)
                micros.append((time.monotonic() - tm) * 1e3)
        T.synchronize(device)
        rec["micro_total_ms"] = (time.monotonic() - t) * 1e3
        rec["micro_mean_ms"] = rec["micro_total_ms"] / grad_accum
        if micros:
            rec["micro_first_ms"] = micros[0]
            rec["micro_rest_mean_ms"] = sum(micros[1:]) / max(len(micros) - 1, 1)

        t = time.monotonic()
        T.sync_gradients(model)
        T.synchronize(device)
        rec["sync_grad_ms"] = (time.monotonic() - t) * 1e3

        t = time.monotonic()
        set_lr(base)
        T.synchronize(device)
        rec["set_lr_ms"] = (time.monotonic() - t) * 1e3

        t = time.monotonic()
        optimizer.step()
        T.synchronize(device)
        rec["opt_step_ms"] = (time.monotonic() - t) * 1e3

        t = time.monotonic()
        refresh_cache()
        T.synchronize(device)
        rec["wcache_ms"] = (time.monotonic() - t) * 1e3

        rec["step_total_ms"] = (time.monotonic() - t_step) * 1e3
        rec["tail_ms"] = (rec["zero_grad_ms"] + rec["sync_grad_ms"] + rec["set_lr_ms"]
                          + rec["opt_step_ms"] + rec["wcache_ms"])
        rec["tail_pct"] = 100.0 * rec["tail_ms"] / rec["step_total_ms"]
        out.append(rec)
        if rank == 0:
            print(f"[pt] STEP {s} total={rec['step_total_ms']:.1f}ms "
                  f"micro_total={rec['micro_total_ms']:.1f} (mean {rec['micro_mean_ms']:.1f}) "
                  f"zero_grad={rec['zero_grad_ms']:.1f} sync_grad={rec['sync_grad_ms']:.1f} "
                  f"set_lr={rec['set_lr_ms']:.1f} opt_step={rec['opt_step_ms']:.1f} "
                  f"wcache={rec['wcache_ms']:.1f} => TAIL={rec['tail_ms']:.1f}ms "
                  f"({rec['tail_pct']:.1f}%)", flush=True)

    if rank == 0:
        print("PHASE_JSON " + json.dumps(dict(
            world_size=world_size, lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            grad_accum=grad_accum, micro_tokens=micro_tokens,
            local_conv=args.local_conv, conv_mode=args.conv_mode,
            compile=args.compile, foreach_optim=args.foreach_optim,
            compile_optim=args.compile_optim, steps=out)))
    if ddp:
        try:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
