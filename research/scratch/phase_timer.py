#!/usr/bin/env python3
"""Unprofiled per-phase step timing for train.py. Imports train.py; never modifies it.

Splits a real steady-state step into:
  zero_grad | grad_accum x microstep | sync_gradients | set_lr | optimizer.step | weight_cache_refresh
so the fixed-overhead-vs-compute split can be read directly, at any world_size/LNC.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = os.environ.get("PROF_REPO", "/home/ubuntu/trainium_frontier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trace-dir", type=str, default=None,
                    help="After unprofiled timing, capture one extra complete step with the "
                         "installed native torch_neuronx profiler. Separate directory per rank; "
                         "not a scored training run and profiled latency is not a speed result.")
    ap.add_argument("--trace-mode", choices=("runtime", "device"), default="runtime",
                    help="runtime: CPU and Neuron Runtime events; device additionally records "
                         "device instructions. Device traces can be substantially larger.")
    ap.add_argument("--trace-target", choices=("step", "microstep"), default="step",
                    help="step captures the complete optimizer step; microstep captures one "
                         "forward/backward pair with real warmed-up inputs. Both captures run "
                         "after the unprofiled measurements and are diagnostic only.")
    ap.add_argument("--trace-rank", type=int, default=None,
                    help="Record only this rank while every rank executes the same capture "
                         "workload. Defaults to all ranks; limits large device trace artifacts.")
    ap.add_argument("--dump-neffs-dir", type=str, default=None,
                    help="Export actual native compiled graphs from the in-process cache "
                         "after timing. Device captures also export these beside their traces; "
                         "the installed profiler alone may omit pre-warmed graph binaries.")
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
    ap.add_argument("--ngram-ve-dim", type=int, default=0,
                    help="T3.1: table WIDTH (0 = n_embd). >0 adds a per-table up-projection. The "
                         "measured cost scales with rows x width in both the forward gather and the "
                         "dense embedding backward, so this is the lever with the most headroom.")
    ap.add_argument("--ngram-ve-index-gather", action=argparse.BooleanOptionalAction, default=True,
                    help="T3.1: spell the row gather as weight[idx] instead of nn.Embedding(idx). "
                         "Numerically identical; embedding_dense_backward costs time proportional to "
                         "ROW COUNT on this stack (19.4ms at 65536 rows) while advanced indexing's "
                         "backward stays at 1.8-5.2ms. --no-... reproduces the old, slow lowering.")
    ap.add_argument("--ngram-ve-clip-exclude", action=argparse.BooleanOptionalAction, default=True,
                    help="T3.1: keep the table grads out of the global clip norm, like train.py.")
    ap.add_argument("--ngram-ve-freeze", action="store_true",
                    help="T3.1 DIAGNOSTIC ONLY (not a trainable configuration): requires_grad_(False) "
                         "on the tables, so the gather still runs in the forward but no "
                         "embedding_dense_backward runs at all. (full - frozen) is the backward "
                         "cost of the tables, which is the number that decides whether a narrower "
                         "table + up-projection is worth building.")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="set any train.py module global before the model is built, e.g. "
                         "--set USE_PARALLEL_BLOCK=True --set PARALLEL_BLOCK_COALESCE=vfc. "
                         "Repeatable. The value is coerced to the type of the existing global, so a "
                         "typo'd NAME or an uncoercible VALUE is an error rather than a silent "
                         "no-op. Applied AFTER the explicit flags above, so it wins.")
    args = ap.parse_args()
    if args.trace_dir and args.micro_only:
        ap.error("--trace-dir captures after full-step timing; do not combine it with --micro-only")

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
    T.NGRAM_VE_DIM = args.ngram_ve_dim
    T.NGRAM_VE_INDEX_GATHER = args.ngram_ve_index_gather
    T.NGRAM_VE_CLIP_EXCLUDE = args.ngram_ve_clip_exclude

    for spec in args.set:
        name, _, raw = spec.partition("=")
        if not _:
            raise SystemExit(f"[pt] --set needs NAME=VALUE, got {spec!r}")
        if not hasattr(T, name):
            raise SystemExit(f"[pt] --set: train.py has no global named {name!r}")
        cur = getattr(T, name)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, str):
            val = raw
        else:
            raise SystemExit(f"[pt] --set: {name} is a {type(cur).__name__}, not settable from CLI")
        setattr(T, name, val)
        print(f"[pt] set {name} = {val!r} (was {cur!r})", flush=True)

    if args.conv_mode == "zeros":
        def _zero_forward(self, x):
            return torch.zeros_like(x)
        T.CausalConv.forward = _zero_forward
        print("[pt] MONKEYPATCH: CausalConv.forward -> zeros_like(x)", flush=True)

    device_type = T.detect_device_type("")
    ddp, rank, local_rank, world_size, device = T.init_runtime(device_type)
    T.COMPUTE_DTYPE = T.compute_dtype_for(device)

    trace_config = None
    if args.trace_rank is not None and not 0 <= args.trace_rank < world_size:
        raise ValueError("--trace-rank must be a rank in this process group")
    if args.trace_dir and (args.trace_rank is None or args.trace_rank == rank):
        if device_type != "neuron":
            raise ValueError("--trace-dir uses the native Neuron profiler")
        from torch_neuronx.profiling import NeuronConfig, ProfileMode
        trace_path = Path(args.trace_dir).resolve() / f"rank{rank}"
        trace_path.mkdir(parents=True, exist_ok=False)
        modes = [ProfileMode.RUNTIME]
        if args.trace_mode == "device":
            modes.append(ProfileMode.DEVICE)
        # Enable metadata caching before compilation/warmup. The measured step
        # loop below runs with no active profiler; capture is an additional step.
        trace_config = NeuronConfig(modes=modes, profile_output_dir=str(trace_path),
                                    max_events_per_nc=1000000, precache_node_info=True)

    if T.CLIP_AFTER_REDUCE:
        raise ValueError("phase_timer splits clip then sync; use train.py for clip-after-reduce timing")
    torch.manual_seed(args.seed)
    config = T.build_config()
    model = T.GPT(config)
    model.init_weights()
    model.to(device)
    if args.ngram_ve_freeze:
        for p in model.ngram_embeds.parameters():
            p.requires_grad_(False)
        print(f"[pt] DIAGNOSTIC: froze {len(list(model.ngram_embeds.parameters()))} n-gram tables "
              f"(forward gather only, no embedding_dense_backward)", flush=True)
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
        print("PROFILE_CONFIG " + json.dumps(dict(
            seed=args.seed, model=asdict(config), toggles=T.current_toggles(),
            compiler_flags=os.environ.get("NEURON_CC_FLAGS"))), flush=True)
        print(f"[pt] world_size={world_size} LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} "
              f"micro_tokens={micro_tokens:,} grad_accum={grad_accum} "
              f"local_conv={args.local_conv} conv_mode={args.conv_mode} compile={args.compile} "
              f"foreach_optim={T.USE_FOREACH_OPTIM} "
              f"weight_cache={T.USE_WEIGHT_CACHE} "
              f"ngram_ve={T.USE_NGRAM_VE} ngram_tables={len(orig_model.ngram_embeds)} "
              f"ngram_rows={orig_model.ngram_rows} ngram_dim={orig_model.ngram_dim} "
              f"ngram_index_gather={T.NGRAM_VE_INDEX_GATHER} "
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
            group["lr"] = group["initial_lr"] * (1.0 if group.get("no_lr_decay") else lrm)
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
                # Same two groups as warmup and train.py. The previous timed path
                # accidentally included the enormous table grads in the dense norm.
                for plist in (clip_main, clip_ngram):
                    if plist:
                        torch.nn.utils.clip_grad_norm_(
                            plist, args.grad_clip, foreach=True if args.clip_foreach else None)
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
    if args.trace_dir:
        from torch.profiler import ProfilerActivity, profile, record_function
        from torch_neuronx.profiling import NeuronProfiler
        T.synchronize(device)
        if ddp:
            import torch.distributed as dist
            dist.barrier()
        if args.trace_target == "microstep":
            # Start with the same gradient state as the first microbatch of a
            # real step. Keep allocation/zeroing outside the device capture.
            zero_grads()
            T.synchronize(device)
        profiler_context = (profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                                    experimental_config=trace_config)
                            if trace_config is not None else nullcontext())
        with profiler_context as prof:
            if ddp:
                # Profiler startup can be slow. Align all ranks after startup
                # so unrecorded ranks still provide the usual chip contention.
                dist.barrier()
                T.synchronize(device)
            if args.trace_target == "microstep":
                with record_function("fresh_start_single_forward_backward_microstep"):
                    micro()
                    T.synchronize(device)
            else:
                with record_function("fresh_start_complete_training_step"):
                    with record_function("zero_grad"):
                        zero_grads()
                    for i in range(grad_accum):
                        with record_function(f"forward_backward_microstep_{i}"):
                            micro()
                    with record_function("clip_then_all_reduce_gradients"):
                        clip_and_sync()
                    with record_function("training_control_collective"):
                        control_collective()
                    with record_function("set_optimizer_scalars"):
                        set_lr(args.warmup_steps + args.steps + 1)
                    with record_function("optimizer_step"):
                        optimizer.step()
                    with record_function("refresh_weight_cache"):
                        refresh_cache()
                    T.synchronize(device)
        if trace_config is not None:
            NeuronProfiler(trace_config).export_trace(prof)
            # Export only once: Kineto does not support repeated exports.
            print("PROFILE_ARTIFACT " + json.dumps(dict(
                rank=rank, mode=args.trace_mode, directory=str(trace_path),
                target=args.trace_target, profiled_steps=int(args.trace_target == "step"),
                profiled_microsteps=grad_accum if args.trace_target == "step" else 1,
                profiled_latency_is_performance_evidence=False)), flush=True)
    if trace_config is not None or args.dump_neffs_dir:
        if device_type != "neuron":
            raise ValueError("NEFF export requires native Neuron")
        import torch_neuronx
        from torch_neuronx.neuron_dynamo_backend.cache_artifact import ArtifactType
        from torch_neuronx.neuron_dynamo_backend.compile import get_compiled_cache_keys
        export_path = (Path(args.dump_neffs_dir).resolve() / f"rank{rank}"
                       if args.dump_neffs_dir else trace_path / "neffs")
        export_path.mkdir(parents=True, exist_ok=True)
        exported = []
        missing = []
        for key in sorted(get_compiled_cache_keys(ArtifactType.NEURON_NEFF)):
            info = torch_neuronx._C.get_neff_info(key)
            if info is None:
                missing.append(key)
                continue
            persistent_key, data = info
            destination = export_path / f"{persistent_key}.neff"
            # A capture may already have copied this exact binary. Never replace
            # a distinct artifact silently.
            if destination.exists():
                assert destination.read_bytes() == data, str(destination)
            else:
                destination.write_bytes(data)
            exported.append(dict(cache_key=key, persistent_cache_key=persistent_key,
                                 file=destination.name, bytes=len(data),
                                 sha256=hashlib.sha256(data).hexdigest()))
        manifest = dict(rank=rank, exported=exported, missing=missing,
                        source="installed native in-process compilation cache")
        (export_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("NEFF_ARTIFACT " + json.dumps(dict(rank=rank, directory=str(export_path),
                                                exported=len(exported), missing=missing)), flush=True)
    if ddp:
        try:
            import torch.distributed as dist
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
