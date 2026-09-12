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
    ap.add_argument("--window", choices=["micro", "microacc", "fwd", "step", "opt"], default="micro")
    ap.add_argument("--warmup-steps", type=int, default=2,
                    help="full optimizer steps to run before the profiled window")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--local-conv", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tensor-lr-scalars", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--foreach-optim", action=argparse.BooleanOptionalAction, default=False,
                    help="MuonAdamW batched-dispatch path (train.py USE_FOREACH_OPTIM)")
    ap.add_argument("--nki-rope-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nki-softcap-ce", action=argparse.BooleanOptionalAction, default=True)
    # T3.1 overhead hunt (2026-09-12): the on/off arms of the n-gram value embeddings, so
    # the ~15ms/table residual can be localized in the REAL compiled graph. The fragment
    # rebuilt in isolation (research/scratch/ngram_frag_bench.py) only costs ~5.5ms/table,
    # so whatever is left is a whole-graph scheduling/fusion effect and only visible here.
    ap.add_argument("--ngram-ve", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ngram-ve-trigram", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ngram-ve-table-mult", type=int, default=32)
    ap.add_argument("--ngram-ve-dim", type=int, default=128)
    ap.add_argument("--ngram-ve-index-gather", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ngram-ablate", type=str, default="real",
                    choices=["real", "copy", "nogate", "slice0", "gatefull", "noinject",
                             "injectpost"],
                    help="Which PIECE of the n-gram injection to remove, to attribute the "
                         "+47ms forward regression. Implemented by re-exec'ing the REAL "
                         "CausalSelfAttention.forward source (inspect.getsource) with exact "
                         "string substitutions, so `copy` is a literal drift control and any "
                         "train.py edit that breaks a substitution fails loudly instead of "
                         "silently measuring the wrong thing. real=no patch at all.")
    ap.add_argument("--ngram-only-layer", type=int, default=-1,
                    help=">=0: give a bigram table to EXACTLY this layer. Distinguishes a "
                         "per-gather-COUNT threshold (every single-layer arm equally cheap) "
                         "from a layer-SPECIFIC cost (one arm expensive).")
    ap.add_argument("--ngram-merge-gather", action=argparse.BooleanOptionalAction, default=False,
                    help="Do ONE dynamic gather from a single [n_tables*rows, dim] table "
                         "instead of one per table, splitting the result afterwards. Same "
                         "bytes, same output, same hash -- collapses N dynamic-DMA ops into "
                         "1. Timing arm for the >=2-gather step change.")
    ap.add_argument("--ngram-max-tables", type=int, default=0,
                    help="0 = all VE layers. N>0 restricts bigram tables to layers <N, to "
                         "check whether the forward cost is linear in table count.")
    ap.add_argument("--ngram-gather", choices=["index", "static"], default="index",
                    help="static: replace `tbl[rows]` with a CONTIGUOUS slice `tbl[:b*t]` "
                         "reshaped to the same shape. Same bytes out of HBM, same output "
                         "shape, same downstream graph -- but static addresses instead of "
                         "8192 dynamic 256-byte descriptors. Isolates dynamic-DMA cost. "
                         "NOT numerically meaningful; a timing arm only.")
    ap.add_argument("--ngram-proj", choices=["matmul", "repeat"], default="matmul",
                    help="repeat: replace each table's Linear(dim, n_embd) up-projection "
                         "with `out.repeat(1, 1, n_embd // dim)`. Same output shape and "
                         "dtype, no matmul, no extra params in the graph. Prices the "
                         "projection separately from the gather. Timing arm only.")
    ap.add_argument("--check-ngram-grads", action=argparse.BooleanOptionalAction, default=False,
                    help="After the timed micro, print each n-gram table's grad abs-sum. A "
                         "zero/None grad means the compiler DCE'd the gather and the arm is "
                         "not measuring what its name claims.")
    ap.add_argument("--ngram-ve-hash", choices=["prime", "mask"], default="prime",
                    help="prime: train.py's `% largest_prime_below(rows)`. mask: `& (rows-1)`, "
                         "monkeypatched in here so the two lowerings are A/B-able without "
                         "editing train.py. Quality-neutral -- see ngram_hash_quality.py.")
    ap.add_argument("--max-events-per-nc", type=int, default=0,
                    help="0 = leave at runtime default")
    ap.add_argument("--capture-nc", type=str, default="0")
    ap.add_argument("--timing-steps", type=int, default=3,
                    help="extra un-profiled steps timed for wall-clock dt after capture")
    ap.add_argument("--capture", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-capture skips the NeuronProfiler entirely and only reports "
                         "the clean un-profiled step/micro dt. Used for ablation sweeps where "
                         "the number that matters is dt, not the trace (and where a 1.4 GB "
                         "NTFF per arm is pure cost).")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()


# --- n-gram injection ablations -------------------------------------------------------
# Rebuilt from train.py's OWN source rather than hand-copied: a hand copy silently rots
# the moment attention changes, and the whole point of these arms is that the ONLY
# difference between them is the piece named in the mode.
GBI_SLICE = "x_v[..., VE_GATE_CHANNELS:2 * VE_GATE_CHANNELS]"
GBI_GATE = "        g_bi = 2 * torch.sigmoid(self.ve_gate_bi(" + GBI_SLICE + "))\n"
GBI_INJECT = ("        v = v + g_bi.unsqueeze(-1) * "
              "ve_bi.view(b, t, self.n_head, self.head_dim)\n")

# 1e-30 (not 0.0) so no constant-folding pass can prove the term dead and delete the
# gather with it: the arm exists precisely to keep the gather + its scatter-add backward
# in the graph while removing the injection into v.
_EPS_KEEPALIVE = "1e-30"

GBI_RETURN = "    return self.c_proj(y), v_raw\n"

ABLATIONS = {
    # (needle, replacement) pairs applied to the extracted forward source.
    "copy": [],
    "nogate": [(GBI_GATE + GBI_INJECT,
                "        v = v + ve_bi.view(b, t, self.n_head, self.head_dim)\n")],
    # Suspect: an offset-0 innermost slice is a free view, a 32-wide slice at offset 32
    # out of 640 is a strided gather of 8192 rows x 64 bytes. Same shapes, same FLOPs,
    # same parameter count -- only the offset changes.
    "slice0": [(GBI_SLICE, "x_v[..., :VE_GATE_CHANNELS]")],
    # No slice at all: gate reads the whole residual. Costs a wider matmul (640x5 vs
    # 32x5, i.e. +20 MFLOP/layer, negligible) but removes every strided read.
    "gatefull": [(GBI_GATE,
                  "        g_bi = 2 * torch.sigmoid(F.linear(x_v, self._gatefull_w))\n")],
    # Keep the gather, the projection, the gate, the elementwise multiply AND the
    # materialization of a full [b,t,n_head,head_dim] product -- but add it to the block
    # OUTPUT instead of to V. Same tensors, same bytes, same FLOPs; the only thing that
    # changes is whether the sum enters the c_v -> rope -> SDPA chain. This is the arm
    # that separates "the injection costs" from "injecting into V specifically costs".
    "injectpost": [(GBI_INJECT,
                    "        _ve_post = (g_bi.unsqueeze(-1) * ve_bi.view("
                    "b, t, self.n_head, self.head_dim)).reshape(b, t, c)\n"),
                   (GBI_RETURN,
                    "    return self.c_proj(y) + (_ve_post if ve_bi is not None else 0), v_raw\n")],
    "noinject": [(GBI_INJECT,
                  "        v = v + (g_bi.float().mean() + ve_bi.float().mean()"
                  ").to(v.dtype) * " + _EPS_KEEPALIVE + "\n")],
}


def _apply_ablation(T, mode: str):
    import inspect
    import textwrap

    import torch
    import torch.nn as nn

    src = textwrap.dedent(inspect.getsource(T.CausalSelfAttention.forward))
    for needle, repl in ABLATIONS[mode]:
        assert src.count(needle) == 1, (
            f"ablation {mode!r}: expected exactly 1 occurrence of {needle!r} in "
            f"CausalSelfAttention.forward, found {src.count(needle)}. train.py changed; "
            f"update ABLATIONS before trusting any number from this run.")
        src = src.replace(needle, repl)

    # Execute in train.py's LIVE module dict (not a copy) so every name the body
    # references (USE_FUSED_QKV, VE_GATE_CHANNELS, rope_norm, F, torch,
    # _cross_head_compose, ...) resolves to the exact object the real forward would have
    # used, including any global set after this patch is applied.
    assert "forward" not in T.__dict__, "train.py has a module-level `forward`; exec would clobber it"
    exec(compile(src, f"<ablate:{mode}>", "exec"), T.__dict__)
    T.CausalSelfAttention.forward = T.__dict__.pop("forward")

    if mode == "gatefull":
        _orig_init = T.CausalSelfAttention.__init__

        def _init(self, config, layer_idx=0, _o=_orig_init):
            _o(self, config, layer_idx)
            if self.ve_gate_bi is not None:
                # Same fan-in-agnostic zero init the real gate gets from init_weights(),
                # so this arm is still a no-op at step 0 (2*sigmoid(0) == 1 either way,
                # matching the real gate's own zero-init behaviour).
                self._gatefull_w = nn.Parameter(
                    torch.zeros(self.n_head, config.n_embd))

        T.CausalSelfAttention.__init__ = _init
    print(f"[drv] MONKEYPATCH: CausalSelfAttention.forward -> ablate={mode} "
          f"({len(ABLATIONS[mode])} substitution(s))", flush=True)


def _install_merged_gather(T, model, device):
    """One dynamic gather for all n-gram tables instead of one per table.

    The >=2-table step change (1 table +4.9ms, 2 tables +49.8ms, 3 tables +52.6ms, all
    independent of row count) says the cost is per dynamic-gather OP, not per byte. If
    that is what it is, folding N gathers into one read of a [N*rows, dim] table -- same
    bytes moved, same output values, same hash -- should recover almost all of it.

    Implemented by hijacking `_ngram_hash_indices` (called exactly once per forward, in
    GPT.forward, before any `_ngram_lookup`) to do the single gather and stash the per-key
    slices, and reducing `_ngram_lookup` to a dict read plus the table's own projection.

    Timing arm only: grads land on the merged table rather than on the originals, so the
    optimizer sees fewer live params. micro dt is fwd+bwd with no optimizer, which is the
    number this arm exists to produce.
    """
    import torch
    import torch.nn as nn

    keys = sorted(model.ngram_embeds.keys())
    rows, dim = model.ngram_rows, model.ngram_embeds[keys[0]].weight.shape[1]
    merged = nn.Parameter(torch.cat([model.ngram_embeds[k].weight.detach()
                                     for k in keys], dim=0).clone())
    model.register_parameter("_merged_ngram", merged)
    model._merged_keys = keys
    orig_hash = T.GPT._ngram_hash_indices

    def _hash_and_gather(self, idx, _keys=keys, _rows=rows, _oh=orig_hash):
        # Returns GATHERED ROWS keyed the same way the index dict was. GPT.forward only
        # ever does `self._ngram_lookup(key, ngram_idx[key])`, so handing the already
        # gathered tensor through that same slot keeps this side-effect-free -- no module
        # attribute mutation for dynamo to break the graph on.
        out = _oh(self, idx)
        b, t = idx.shape
        # Offset each key's rows into its own block of the merged table, then ONE gather.
        flat = torch.cat([out[k].reshape(-1) + i * _rows for i, k in enumerate(_keys)], dim=0)
        big = self._merged_ngram[flat]
        return {k: big[i * b * t:(i + 1) * b * t].view(b, t, -1)
                for i, k in enumerate(_keys)}

    def _lookup_merged(self, key, gathered):
        out = gathered.to(T.COMPUTE_DTYPE)
        if key in self.ngram_proj:
            out = self.ngram_proj[key](out)
        return out

    T.GPT._ngram_hash_indices = _hash_and_gather
    T.GPT._ngram_lookup = _lookup_merged
    model.to(device)
    print(f"[drv] MONKEYPATCH: merged gather over {len(keys)} tables "
          f"-> one [{len(keys) * rows}, {dim}] read", flush=True)


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
    T.USE_NGRAM_VE = args.ngram_ve
    T.NGRAM_VE_TRIGRAM = args.ngram_ve_trigram
    T.NGRAM_VE_TABLE_MULT = args.ngram_ve_table_mult
    T.NGRAM_VE_DIM = args.ngram_ve_dim
    T.NGRAM_VE_INDEX_GATHER = args.ngram_ve_index_gather
    if args.ngram_ve_hash == "mask":
        def _mask_hash(self, idx):
            cur = idx.to(torch.int32)
            pad = torch.full_like(cur[:, :1], self.ngram_pad_id)
            prev = torch.cat((pad, cur[:, :-1]), dim=1)
            need_tri = any(k.startswith("t") for k in self.ngram_embeds)
            prev2 = torch.cat((pad, pad, cur[:, :-2]), dim=1) if need_tri else None
            m = self.ngram_rows - 1
            assert self.ngram_rows & m == 0, self.ngram_rows
            out = {}
            for key in self.ngram_embeds:
                layer = int(key[1:])
                if key[0] == "b":
                    p0, p1 = T.ngram_bigram_primes(layer)
                    h = torch.bitwise_xor(prev * p0, cur * p1)
                else:
                    p0, p1, p2 = T.ngram_trigram_primes(layer)
                    h = torch.bitwise_xor(torch.bitwise_xor(prev2 * p0, prev * p1), cur * p2)
                out[key] = torch.bitwise_and(h, m)
            return out

        T.GPT._ngram_hash_indices = _mask_hash
        print("[drv] MONKEYPATCH: _ngram_hash_indices -> power-of-two mask", flush=True)

    if args.ngram_only_layer >= 0:
        _orig_has_bi_ol = T.has_ngram_bigram

        def _only_has_bi(layer_idx, n_layer, _o=_orig_has_bi_ol, _L=args.ngram_only_layer):
            return _o(layer_idx, n_layer) and layer_idx == _L

        T.has_ngram_bigram = _only_has_bi
        print(f"[drv] MONKEYPATCH: bigram table on layer {args.ngram_only_layer} only",
              flush=True)

    if args.ngram_max_tables > 0:
        _orig_has_bi = T.has_ngram_bigram

        def _capped_has_bi(layer_idx, n_layer, _o=_orig_has_bi, _n=args.ngram_max_tables):
            return _o(layer_idx, n_layer) and layer_idx < _n

        T.has_ngram_bigram = _capped_has_bi
        print(f"[drv] MONKEYPATCH: bigram tables capped to layers <{args.ngram_max_tables}",
              flush=True)

    if args.ngram_gather != "index" or args.ngram_proj != "matmul":
        _n_embd_for_lookup = T.build_config().n_embd

        def _lookup(self, key, rows, _g=args.ngram_gather, _p=args.ngram_proj,
                    _ne=_n_embd_for_lookup):
            tbl = self.ngram_embeds[key].weight
            if _g == "static":
                n = rows.numel()
                out = tbl[:n].view(*rows.shape, tbl.shape[1])
            else:
                out = tbl[rows]
            out = out.to(T.COMPUTE_DTYPE)
            if key in self.ngram_proj:
                if _p == "repeat":
                    out = out.repeat(1, 1, _ne // out.shape[-1])
                else:
                    out = self.ngram_proj[key](out)
            return out

        T.GPT._ngram_lookup = _lookup
        print(f"[drv] MONKEYPATCH: _ngram_lookup gather={args.ngram_gather} "
              f"proj={args.ngram_proj}", flush=True)

    if args.ngram_ablate != "real":
        _apply_ablation(T, args.ngram_ablate)

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
    print(f"[drv] ngram_ve={T.USE_NGRAM_VE} tables={len(model.ngram_embeds)} "
          f"rows={model.ngram_rows} dim={model.ngram_dim} mod={model.ngram_mod} "
          f"hash={args.ngram_ve_hash} index_gather={T.NGRAM_VE_INDEX_GATHER}", flush=True)

    if args.ngram_merge_gather and len(model.ngram_embeds):
        _install_merged_gather(T, model, device)

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
    nconf = NeuronConfig(**nc_kwargs) if args.capture else None
    nprof = NeuronProfiler(nconf) if args.capture else None

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
    import contextlib
    _ctx = (profile(activities=[ProfilerActivity.CPU], experimental_config=nconf)
            if args.capture else contextlib.nullcontext())
    with _ctx as prof:
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
        elif args.window == "fwd":
            # FORWARD ONLY. The 2026-09-12 T3.1 overhead hunt measured the n-gram cost
            # as +47.0ms in the forward NEFF and only +5.6ms in the backward one, so a
            # forward-only capture is both the relevant window and a far smaller NTFF.
            # NOT under no_grad: that would compile a different (inference) graph.
            optimizer.zero_grad(set_to_none=True)
            x, y = fetch()
            _ = model(x, y)
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
        if args.capture:
            try:
                session_dir = nconf.get_session_dir()
            except Exception as exc:  # noqa: BLE001
                print(f"[drv] get_session_dir failed: {exc}", flush=True)
    print(f"[drv] profiled window ({args.window}) wall={window_wall * 1e3:.2f} ms", flush=True)
    if args.capture:
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

    if args.check_ngram_grads:
        # The last micro left grads populated. A None or exactly-zero table grad means the
        # compiler proved the gather dead and removed it -- in which case the arm's dt is
        # not measuring the thing the arm is named after.
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        for key, emb in base.ngram_embeds.items():
            g = emb.weight.grad
            print(f"[drv] grad[{key}] {'None' if g is None else '%.6e' % g.float().abs().sum().item()}",
                  flush=True)

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
        "ngram_ve": args.ngram_ve,
        "ngram_ve_trigram": args.ngram_ve_trigram,
        "ngram_ve_table_mult": args.ngram_ve_table_mult,
        "ngram_ve_dim": args.ngram_ve_dim,
        "ngram_ve_hash": args.ngram_ve_hash,
        "ngram_ablate": args.ngram_ablate,
        "ngram_gather": args.ngram_gather,
        "ngram_proj": args.ngram_proj,
        "ngram_max_tables": args.ngram_max_tables,
        "ngram_only_layer": args.ngram_only_layer,
        "ngram_merge_gather": args.ngram_merge_gather,
        "capture": args.capture,
    }
    print("[drv] INFO " + json.dumps(info), flush=True)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(info, fh, indent=2)
    T.cleanup_runtime()


if __name__ == "__main__":
    main()
