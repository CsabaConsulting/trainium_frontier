#!/usr/bin/env python3
"""External profile driver for train.py -- imports train.py as a module and
never modifies it. Reproduces main()'s model/optimizer/step construction
exactly (synthetic batches, same shapes), then captures a device profile of a
chosen window with torch_neuronx.profiling.

Windows:
  micro : one grad-accum microstep (fwd + bwd)   -- comparable to the prior profile
  step  : one full optimizer step (all microsteps + sync_gradients + optimizer.step())
"""

import argparse
import json
import os
import sys
import time

REPO = os.environ.get("PROF_REPO", "/home/ubuntu/trainium_frontier")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="profile_output_dir for NRT traces")
    ap.add_argument("--window", choices=["micro", "microacc", "step", "opt"], default="micro")
    ap.add_argument("--warmup-steps", type=int, default=2,
                    help="full optimizer steps to run before the profiled window")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--local-conv", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tensor-lr-scalars", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--foreach-optim", action=argparse.BooleanOptionalAction, default=False,
                    help="MuonAdamW batched-dispatch path (train.py USE_FOREACH_OPTIM)")
    ap.add_argument("--nki-rope-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nki-softcap-ce", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-events-per-nc", type=int, default=0,
                    help="0 = leave at runtime default")
    ap.add_argument("--capture-nc", type=str, default="0")
    ap.add_argument("--timing-steps", type=int, default=3,
                    help="extra un-profiled steps timed for wall-clock dt after capture")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()


def main():
    args = build_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    import torch
    from torch.profiler import ProfilerActivity, profile
    from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

    import train as T

    # Apply the flags exactly the way main() does (module globals read at
    # construction time by Block.__init__ / the optimizer).
    T.USE_LOCAL_CONV = args.local_conv
    T.USE_TENSOR_LR_SCALARS = args.tensor_lr_scalars
    T.USE_FOREACH_OPTIM = args.foreach_optim
    T.USE_NKI_ROPE_NORM = args.nki_rope_norm
    T.USE_NKI_SOFTCAP_CE = args.nki_softcap_ce

    device_type = T.detect_device_type("")
    ddp, rank, local_rank, world_size, device = T.init_runtime(device_type)
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)
    print(f"[drv] device={device} dtype={T.COMPUTE_DTYPE} world_size={world_size} "
          f"device_type={device_type}", flush=True)

    config = T.build_config()
    model = T.GPT(config)
    model.init_weights()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[drv] num_params={n_params:,} config={config}", flush=True)

    if args.compile:
        backend = "neuron" if device_type == "neuron" else "inductor"
        model = torch.compile(model, backend=backend, dynamic=False)
        print(f"[drv] torch.compile enabled (backend={backend})", flush=True)

    world_tokens_per_micro = T.DEVICE_BATCH_SIZE * T.PACK_FACTOR * T.SEQ_LEN * world_size
    grad_accum = T.TOTAL_BATCH_SIZE // world_tokens_per_micro
    print(f"[drv] micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum}", flush=True)

    rows = T.DEVICE_BATCH_SIZE * T.PACK_FACTOR
    gen = torch.Generator().manual_seed(1234)
    cpu_x = torch.randint(0, config.vocab_size, (rows, T.SEQ_LEN), dtype=torch.int32, generator=gen)
    cpu_y = torch.randint(0, config.vocab_size, (rows, T.SEQ_LEN), dtype=torch.long, generator=gen)

    optimizer = model.setup_optimizer()

    def micro(x, y):
        loss = model(x, y)
        (loss / grad_accum).backward()
        return loss

    def set_lr(step):
        # Mimic the real cooldown: a DIFFERENT lr every step, which is exactly
        # what USE_TENSOR_LR_SCALARS was introduced to make free.
        lrm = 1.0 - 0.001 * step
        wd_progress = min(1.0, step / 1000.0)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = T.muon_momentum(step)
                group["weight_decay"] = (T.WEIGHT_DECAY * (1.0 - wd_progress)
                                         if T.WD_DECAY_TO_ZERO else T.WEIGHT_DECAY)

    def fetch():
        return cpu_x.to(device), cpu_y.to(device)

    def full_step(step, n_micro=None):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(n_micro if n_micro is not None else grad_accum):
            x, y = fetch()
            micro(x, y)
        T.sync_gradients(model)
        set_lr(step)
        optimizer.step()
        if T.USE_WEIGHT_CACHE:
            for m in model.modules():
                if isinstance(m, T.Linear):
                    m.refresh_weight_cache(T.COMPUTE_DTYPE)

    nc_kwargs = dict(
        modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
        capture_enabled_for_nc=args.capture_nc,
        profile_output_dir=args.out,
        precache_node_info=True,
    )
    if args.max_events_per_nc:
        nc_kwargs["max_events_per_nc"] = args.max_events_per_nc
    nconf = NeuronConfig(**nc_kwargs)
    nprof = NeuronProfiler(nconf)

    # --- warmup (compile + precache), outside the profiler -------------------
    t_warm = time.monotonic()
    for s in range(args.warmup_steps):
        ts = time.monotonic()
        full_step(s)
        T.synchronize(device)
        print(f"[drv] warmup step {s} dt={time.monotonic() - ts:.3f}s", flush=True)
    print(f"[drv] warmup total {time.monotonic() - t_warm:.1f}s", flush=True)

    # --- profiled window -----------------------------------------------------
    T.synchronize(device)
    session_dir = None
    t0 = time.monotonic()
    with profile(activities=[ProfilerActivity.CPU], experimental_config=nconf) as prof:
        if args.window == "micro":
            optimizer.zero_grad(set_to_none=True)
            x, y = fetch()
            micro(x, y)
            T.synchronize(device)
        elif args.window == "microacc":
            # steady-state microstep: grads are ALREADY populated (the warmup
            # step left them there and we deliberately do not zero_grad), so
            # backward accumulates -- this is what 31 of the 32 microsteps in a
            # real grad-accum step actually do.
            x, y = fetch()
            micro(x, y)
            T.synchronize(device)
        elif args.window == "opt":
            # grads are already populated by the last warmup step; profile only
            # the gradient-sync + optimizer.step() tail of a real training step.
            T.sync_gradients(model)
            set_lr(args.warmup_steps)
            optimizer.step()
            if T.USE_WEIGHT_CACHE:
                for m in model.modules():
                    if isinstance(m, T.Linear):
                        m.refresh_weight_cache(T.COMPUTE_DTYPE)
            T.synchronize(device)
        else:
            full_step(args.warmup_steps)
            T.synchronize(device)
        window_wall = time.monotonic() - t0
        try:
            session_dir = nconf.get_session_dir()
        except Exception as exc:  # noqa: BLE001
            print(f"[drv] get_session_dir failed: {exc}", flush=True)
    print(f"[drv] profiled window ({args.window}) wall={window_wall * 1e3:.2f} ms", flush=True)
    try:
        nprof.export_trace(prof, filename="trace.json")
    except Exception as exc:  # noqa: BLE001
        print(f"[drv] export_trace failed: {exc}", flush=True)
    print(f"[drv] session_dir={session_dir}", flush=True)

    # --- clean wall-clock dt, no profiler attached ---------------------------
    dts = []
    for i in range(args.timing_steps):
        T.synchronize(device)
        ts = time.monotonic()
        full_step(args.warmup_steps + 1 + i)
        T.synchronize(device)
        dts.append(time.monotonic() - ts)
        print(f"[drv] timed step {i} dt={dts[-1]:.4f}s", flush=True)

    # micro-level wall clock too (one fwd+bwd, no optimizer)
    micro_dts = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        T.synchronize(device)
        ts = time.monotonic()
        x, y = fetch()
        micro(x, y)
        T.synchronize(device)
        micro_dts.append(time.monotonic() - ts)
    print(f"[drv] micro dts (s): {['%.4f' % d for d in micro_dts]}", flush=True)

    info = {
        "window": args.window,
        "window_wall_ms": window_wall * 1e3,
        "session_dir": session_dir,
        "grad_accum_steps": grad_accum,
        "world_size": world_size,
        "num_params": n_params,
        "compile": args.compile,
        "local_conv": args.local_conv,
        "tensor_lr_scalars": args.tensor_lr_scalars,
        "nki_rope_norm": args.nki_rope_norm,
        "nki_softcap_ce": args.nki_softcap_ce,
        "step_dts_s": dts,
        "micro_dts_s": micro_dts,
        "seq_len": T.SEQ_LEN,
        "depth": T.DEPTH,
        "n_embd": config.n_embd,
        "total_batch_size": T.TOTAL_BATCH_SIZE,
        "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG", ""),
    }
    print("[drv] INFO " + json.dumps(info), flush=True)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(info, fh, indent=2)
    T.cleanup_runtime()


if __name__ == "__main__":
    main()
