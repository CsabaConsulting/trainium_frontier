#!/usr/bin/env python3
"""Unprofiled per-phase step timing for train.py. Imports train.py; never modifies it.

Splits a real steady-state step into:
  zero_grad | grad_accum x microstep | sync_gradients | set_lr | optimizer.step | weight_cache_refresh
so the fixed-overhead-vs-compute split can be read directly, at any world_size/LNC.
"""

import argparse
import json
import math
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
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="T1.1 re-profile: include torch.nn.utils.clip_grad_norm_(model.parameters(), "
                         "this) in the timed window, matching train.py's real per-step order "
                         "(clip -> sync_gradients). 0 disables it, matching GRAD_CLIP<=0.")
    ap.add_argument("--clip-foreach", action=argparse.BooleanOptionalAction, default=False,
                    help="T1.3: pass foreach=True to clip_grad_norm_ explicitly.")
    ap.add_argument("--grad-bucket", action=argparse.BooleanOptionalAction, default=False,
                    help="T1.2: flat fp32 gradient buffer + one all_reduce/clip instead of "
                         "per-tensor (mirrors train.py's --grad-bucket).")
    ap.add_argument("--control-collective", action=argparse.BooleanOptionalAction, default=True,
                    help="T1.1 re-profile: also time train.py's real per-step control tensor "
                         "(NaN/stop-flag all_reduce + two .item() calls) so this harness's "
                         "step_total_ms is apples-to-apples with a real run's dt, and the only "
                         "remaining gap is the dataloader fetch() cost (see --dataloader).")
    ap.add_argument("--dataloader", choices=["synthetic", "real"], default="synthetic",
                    help="synthetic (default): reuse one cached (x,y) batch forever, no fetch() "
                         "call at all -- the correctness-irrelevant timing probe from T1.1's "
                         "'direct dataloader-exposure test'. real: build the actual train "
                         "dataloader and call fetch() every microstep exactly like train.py's "
                         "main loop, so (real step_total_ms - synthetic step_total_ms) is the "
                         "true exposed host dataloader cost, including any overlap with device "
                         "compute (fetch() runs between backward() and the next forward(), so "
                         "whatever the device queue can hide, it does -- this measures only what "
                         "is NOT hidden).")
    ap.add_argument("--pack-factor", type=int, default=None,
                    help="T1.5: override train.PACK_FACTOR, i.e. the micro-batch ROW COUNT "
                         "(rows = DEVICE_BATCH_SIZE * PACK_FACTOR). grad_accum is recomputed from "
                         "TOTAL_BATCH_SIZE so total tokens/step stays fixed. Only meaningful with "
                         "--dataloader synthetic; the real loader path also reshapes by PACK_FACTOR.")
    ap.add_argument("--device-batch-size", type=int, default=None,
                    help="T1.5: override train.DEVICE_BATCH_SIZE. Combined with --pack-factor this "
                         "separates 'same device shape, different loader code path' (e.g. "
                         "--device-batch-size 2 --pack-factor 2 gives the same [4,2048] micro-batch "
                         "as --device-batch-size 1 --pack-factor 4).")
    ap.add_argument("--wcache-mode", choices=["loop", "foreach"], default="loop",
                    help="T1.10: loop = train.py's real per-Linear `weight.to(dtype)` (one device op "
                         "each, ~40 launches/step). foreach = one torch._foreach_copy_ into a set of "
                         "preallocated bf16 buffers. Measures whether the launch count is actually "
                         "costing anything before train.py is changed at all.")
    ap.add_argument("--time-loss-item", action=argparse.BooleanOptionalAction, default=False,
                    help="T1.9: time `float(loss.item())` as its own phase, separately from the "
                         "control collective, so the real cost of reading the loss every step "
                         "(right after an explicit synchronize, i.e. when the device is already "
                         "idle) is a measured number rather than an assumption.")
    ap.add_argument("--micro-only", type=int, default=0,
                    help="T1.5: instead of full steps, time N individual forward+backward "
                         "microsteps (with a synchronize around each) and report ms and ms/token. "
                         "Isolates pure device compute from the optimizer/collective tail, which is "
                         "what the PACK_FACTOR anomaly needs to be judged on.")
    ap.add_argument("--ngram-ve", action=argparse.BooleanOptionalAction, default=False,
                    help="T3.1: hashed bigram/trigram value embeddings (train.USE_NGRAM_VE). "
                         "The 15-step smoke test measured dt 1.62s against baseline 0.90s, so "
                         "this exists to attribute that 0.72s to a phase before tuning anything.")
    ap.add_argument("--ngram-ve-trigram", action=argparse.BooleanOptionalAction, default=True,
                    help="T3.1: trigram tables on the first/last VE layer (5 tables vs 3).")
    ap.add_argument("--ngram-ve-table-mult", type=int, default=8,
                    help="T3.1: rows per table = this x padded_vocab_size. Varying ONLY this "
                         "holds the per-token gather/injection work fixed and scales every "
                         "param-count-proportional cost (allreduce bytes, dense embedding-grad "
                         "scatter, optimizer traffic), which is what separates the two.")
    ap.add_argument("--ngram-ve-clip-exclude", action=argparse.BooleanOptionalAction, default=True,
                    help="T3.1: keep the table grads out of the global clip norm, like train.py.")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    os.chdir(REPO)

    import torch
    import train as T

    T.USE_LOCAL_CONV = args.local_conv
    if args.pack_factor is not None:
        T.PACK_FACTOR = args.pack_factor
    if args.device_batch_size is not None:
        T.DEVICE_BATCH_SIZE = args.device_batch_size
    T.USE_TENSOR_LR_SCALARS = True
    T.USE_FOREACH_OPTIM = args.foreach_optim
    T.USE_NKI_ROPE_NORM = True
    T.USE_NKI_SOFTCAP_CE = True
    T.USE_NGRAM_VE = args.ngram_ve
    T.NGRAM_VE_TRIGRAM = args.ngram_ve_trigram
    T.NGRAM_VE_TABLE_MULT = args.ngram_ve_table_mult
    T.NGRAM_VE_CLIP_EXCLUDE = args.ngram_ve_clip_exclude

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
    orig_model = model
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
    grad_accum = max(1, T.TOTAL_BATCH_SIZE // micro_tokens)
    if rank == 0:
        print(f"[pt] world_size={world_size} LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} "
              f"micro_tokens={micro_tokens:,} grad_accum={grad_accum} "
              f"local_conv={args.local_conv} conv_mode={args.conv_mode} compile={args.compile} "
              f"foreach_optim={T.USE_FOREACH_OPTIM} "
              f"weight_cache={T.USE_WEIGHT_CACHE} "
              f"ngram_ve={T.USE_NGRAM_VE} ngram_tables={len(orig_model.ngram_embeds)} "
              f"ngram_rows={orig_model.ngram_rows} "
              f"params={sum(p.numel() for p in orig_model.parameters()):,}", flush=True)

    rows = T.DEVICE_BATCH_SIZE * T.PACK_FACTOR
    gen = torch.Generator().manual_seed(1234 + rank)
    cpu_x = torch.randint(0, config.vocab_size, (rows, T.SEQ_LEN), dtype=torch.int32, generator=gen)
    cpu_y = torch.randint(0, config.vocab_size, (rows, T.SEQ_LEN), dtype=torch.long, generator=gen)
    x = cpu_x.to(device)
    y = cpu_y.to(device)

    train_loader = None
    if args.dataloader == "real":
        tokenizer = T.ensure_tokenizer(build_if_missing=False)
        loader_seq_len = T.PACK_FACTOR * T.SEQ_LEN
        train_loader = T.make_dataloader(tokenizer, T.DEVICE_BATCH_SIZE, loader_seq_len, "train", device)

        def fetch():
            xb, yb, state = next(train_loader)
            if T.PACK_FACTOR > 1:
                xb = xb.reshape(T.DEVICE_BATCH_SIZE * T.PACK_FACTOR, T.SEQ_LEN)
                yb = yb.reshape(T.DEVICE_BATCH_SIZE * T.PACK_FACTOR, T.SEQ_LEN)
            return xb, yb

        x, y = fetch()  # prime, exactly like train.py's `x, y, loader_state = fetch()` before the loop

    linears = [m for m in model.modules() if isinstance(m, T.Linear)]
    clip_main, clip_ngram = T.split_clip_params(orig_model)

    grad_flat_buffer = None
    if args.grad_bucket:
        grad_flat_buffer, _gb_params = T.build_flat_grad_buffer(model)
        if rank == 0:
            print(f"[pt] grad_bucket: {len(_gb_params)} params, {grad_flat_buffer.numel():,} floats", flush=True)

    last_loss = None

    def micro():
        nonlocal x, y, last_loss
        loss = model(x, y)
        last_loss = loss.detach()
        (loss / grad_accum).backward()
        if args.dataloader == "real":
            x, y = fetch()

    def clip_and_sync():
        if grad_flat_buffer is not None:
            if args.grad_clip > 0:
                T.clip_grad_norm_flat_(grad_flat_buffer, args.grad_clip)
            T.sync_gradients_flat(grad_flat_buffer)
        else:
            if args.grad_clip > 0:
                # Mirrors train.py's real per-step order, including T3.1's separate clip for
                # the n-gram tables (they must not enter the main global norm).
                for plist in (clip_main, clip_ngram):
                    if plist:
                        torch.nn.utils.clip_grad_norm_(
                            plist, args.grad_clip, foreach=True if args.clip_foreach else None)
            T.sync_gradients(model)

    def control_collective():
        # Mirrors train.py's real per-step control tensor: the loss .item() call,
        # a NaN/stop-flag all_reduce, and its two .item() calls -- all real host
        # syncs on every step, missing from earlier profiles. Synthetic here
        # (stop_after_step is always False) -- only the COST is under test, not
        # the stop/NaN logic itself.
        train_loss = float(last_loss.item()) if last_loss is not None else 0.0
        control = torch.tensor(
            [1 if math.isnan(train_loss) else 0, 0], device=device, dtype=torch.int32)
        if ddp:
            import torch.distributed as dist
            dist.all_reduce(control, op=dist.ReduceOp.MAX)
        _ = int(control[0].item())
        _ = int(control[1].item())

    def set_lr(step):
        lrm = 1.0 - 0.001 * step
        wd_progress = min(1.0, step / 1000.0)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = T.muon_momentum(step)
                group["weight_decay"] = (T.WEIGHT_DECAY * (1.0 - wd_progress)
                                         if T.WD_DECAY_TO_ZERO else T.WEIGHT_DECAY)

    # T1.10: preallocated bf16 destinations for the batched refresh. Built once; the loop-mode
    # path never touches them, so the two modes are strictly comparable.
    wcache_dsts = [torch.empty_like(m.weight, dtype=T.COMPUTE_DTYPE) for m in linears]
    wcache_srcs = [m.weight for m in linears]

    def refresh_cache():
        if not T.USE_WEIGHT_CACHE:
            return
        if args.wcache_mode == "foreach":
            # One dispatch for all ~40 casts instead of one per Linear. _foreach_copy_ does the
            # dtype conversion as part of the copy, exactly like `weight.to(bf16)` does.
            torch._foreach_copy_(wcache_dsts, wcache_srcs)
            for m, d in zip(linears, wcache_dsts):
                m._cached_weight = d
        else:
            for m in linears:
                m.refresh_weight_cache(T.COMPUTE_DTYPE)

    def zero_grads():
        if grad_flat_buffer is not None:
            grad_flat_buffer.zero_()
        else:
            optimizer.zero_grad(set_to_none=True)

    if args.micro_only:
        # T1.5: pure forward+backward at this micro shape, nothing else in the window.
        micro_rows = T.DEVICE_BATCH_SIZE * T.PACK_FACTOR
        tok_per_micro = micro_rows * T.SEQ_LEN          # per rank
        for s in range(args.warmup_steps):
            zero_grads()
            micro()
            T.synchronize(device)
            if rank == 0:
                print(f"[pt] micro-warmup {s}", flush=True)
        times = []
        for s in range(args.micro_only):
            zero_grads()
            T.synchronize(device)
            t = time.monotonic()
            micro()
            T.synchronize(device)
            ms = (time.monotonic() - t) * 1e3
            times.append(ms)
            if rank == 0:
                print(f"[pt] MICRO {s} ms={ms:.2f} us_per_token={ms * 1e3 / tok_per_micro:.4f}",
                      flush=True)
        if rank == 0:
            srt = sorted(times)
            med = srt[len(srt) // 2]
            print("MICRO_JSON " + json.dumps(dict(
                world_size=world_size, lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
                device_batch_size=T.DEVICE_BATCH_SIZE, pack_factor=T.PACK_FACTOR,
                micro_rows=micro_rows, seq_len=T.SEQ_LEN, tokens_per_micro_per_rank=tok_per_micro,
                tokens_per_micro_global=tok_per_micro * world_size,
                compile=args.compile, dataloader=args.dataloader,
                ms=times, ms_median=med, ms_min=min(times),
                us_per_token_median=med * 1e3 / tok_per_micro,
                us_per_token_min=min(times) * 1e3 / tok_per_micro)), flush=True)
        if ddp:
            try:
                import torch.distributed as dist
                dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
        return

    for s in range(args.warmup_steps):
        t = time.monotonic()
        zero_grads()
        for _ in range(grad_accum):
            micro()
        clip_and_sync()
        control_collective()
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
        zero_grads()
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
        if grad_flat_buffer is not None:
            if args.grad_clip > 0:
                T.clip_grad_norm_flat_(grad_flat_buffer, args.grad_clip)
        else:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                               foreach=True if args.clip_foreach else None)
        T.synchronize(device)
        rec["clip_ms"] = (time.monotonic() - t) * 1e3

        t = time.monotonic()
        if grad_flat_buffer is not None:
            T.sync_gradients_flat(grad_flat_buffer)
        else:
            T.sync_gradients(model)
        T.synchronize(device)
        rec["sync_grad_ms"] = (time.monotonic() - t) * 1e3

        if args.time_loss_item:
            # T1.9: exactly train.py's `train_loss = float(last_loss.item())`, timed alone.
            t = time.monotonic()
            _ = float(last_loss.item()) if last_loss is not None else 0.0
            rec["loss_item_ms"] = (time.monotonic() - t) * 1e3

        t = time.monotonic()
        control_collective()
        T.synchronize(device)
        rec["control_ms"] = (time.monotonic() - t) * 1e3

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
        rec["tail_ms"] = (rec["zero_grad_ms"] + rec["clip_ms"] + rec["sync_grad_ms"]
                          + rec.get("loss_item_ms", 0.0) + rec["control_ms"] + rec["set_lr_ms"]
                          + rec["opt_step_ms"] + rec["wcache_ms"])
        rec["tail_pct"] = 100.0 * rec["tail_ms"] / rec["step_total_ms"]
        out.append(rec)
        if rank == 0:
            print(f"[pt] STEP {s} total={rec['step_total_ms']:.1f}ms "
                  f"micro_total={rec['micro_total_ms']:.1f} (mean {rec['micro_mean_ms']:.1f}) "
                  f"zero_grad={rec['zero_grad_ms']:.1f} clip={rec['clip_ms']:.1f} "
                  f"sync_grad={rec['sync_grad_ms']:.1f} control={rec['control_ms']:.1f} "
                  f"set_lr={rec['set_lr_ms']:.1f} opt_step={rec['opt_step_ms']:.1f} "
                  f"loss_item={rec.get('loss_item_ms', 0.0):.1f} "
                  f"wcache={rec['wcache_ms']:.1f} => TAIL={rec['tail_ms']:.1f}ms "
                  f"({rec['tail_pct']:.1f}%)", flush=True)

    if rank == 0:
        print("PHASE_JSON " + json.dumps(dict(
            world_size=world_size, lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            grad_accum=grad_accum, micro_tokens=micro_tokens,
            local_conv=args.local_conv, conv_mode=args.conv_mode,
            compile=args.compile, foreach_optim=args.foreach_optim,
            compile_optim=args.compile_optim, grad_bucket=args.grad_bucket,
            clip_foreach=args.clip_foreach, dataloader=args.dataloader,
            wcache_mode=args.wcache_mode, time_loss_item=args.time_loss_item,
            device_batch_size=T.DEVICE_BATCH_SIZE, pack_factor=T.PACK_FACTOR, steps=out)))
    if ddp:
        try:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
