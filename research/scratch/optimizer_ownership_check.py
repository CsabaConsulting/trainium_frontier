#!/usr/bin/env python3
"""Require exact replicated/owned optimizer equivalence before timing ownership."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from optimizer_ownership import install_optimizer


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def template_groups(real):
    if real:
        T.DEPTH, T.N_EMBD = 6, 1024
        T.USE_NGRAM_VE = T.USE_QK_SHIFT = False
        T.QK_SHIFT_BETA, T.QK_SHIFT_FREEZE = 0.0, True
        T.USE_LOCAL_CONV = T.TRAIN_LOCAL_CONV = True
        with torch.device("meta"):
            model = T.GPT(T.build_config())
            groups = model.setup_optimizer().param_groups
        assert sum(p.numel() for g in groups for p in g["params"]) == 117453580
        return groups
    groups = [dict(kind="adamw", params=[torch.empty(64, 32, device="meta") for _ in range(5)],
                   lr=.02, betas=(.8, .96), eps=1e-10, weight_decay=.01),
              dict(kind="adamw", params=[torch.empty(1, device="meta") for _ in range(12)],
                   lr=.002, betas=(.9, .95), eps=1e-10, weight_decay=0.0)]
    for shape, count in (((8, 32), 3), ((32, 32), 8), ((32, 64), 6), ((64, 32), 6)):
        groups.append(dict(kind="muon", params=[torch.empty(shape, device="meta") for _ in range(count)],
                           lr=.02, momentum=.95, beta2=.9, ns_steps=5,
                           weight_decay=.2, transposed=False))
    return groups


def make_optimizer(groups, initial, device, owned, pad_singleton_muon=False,
                   adamw_math_mode=None):
    parameters = [torch.nn.Parameter(x.clone().to(device)) for x in initial]
    copied, offset = [], 0
    for group in groups:
        size = len(group["params"])
        copied.append({**group, "params": parameters[offset:offset + size]})
        offset += size
    optimizer = T.MuonAdamW(copied)
    if adamw_math_mode is not None:
        from adamw_fused_integration import install_optimizer as install_adamw
        install_adamw(optimizer, T, math_mode=adamw_math_mode)
    adapter = install_optimizer(optimizer, T, pad_singleton_muon=pad_singleton_muon) if owned else None
    return parameters, optimizer, adapter


def compare(reference_params, reference, parameters, candidate, adapter, device, tag):
    failures = []
    digest = hashlib.sha256()

    def equal(name, actual, expected):
        if not isinstance(actual, torch.Tensor):
            if actual != expected:
                failures.append(dict(name=name, actual=actual, expected=expected))
            return
        a, b = actual.detach().cpu(), expected.detach().cpu()
        if not torch.equal(a, b) or not bool(torch.isfinite(a).all()):
            error = (a.double() - b.double()).abs()
            failures.append(dict(name=name, shape=list(a.shape), finite=bool(torch.isfinite(a).all()),
                                 different=int((a != b).sum()), max_abs=float(error.max()),
                                 relative_rms=float(error.norm() / b.double().norm().clamp_min(1e-30))))

    for index, (p, expected) in enumerate(zip(parameters, reference_params)):
        current = p.detach().cpu()
        equal(f"parameter:{index}", current, expected)
        digest.update(current.numpy().tobytes())
    for group_index, (full, owned, positions) in enumerate(zip(reference.param_groups, adapter.groups, adapter.positions)):
        if not positions:
            continue
        if full["kind"] == "adamw":
            for local, position in enumerate(positions):
                state = candidate.state.get(owned["params"][local], {})
                expected = reference.state.get(full["params"][position], {})
                assert set(state) == set(expected)
                for key in state:
                    equal(f"adamw:{group_index}:{position}:{key}", state[key], expected[key])
        else:
            state = candidate.state[owned["params"][0]]
            expected = reference.state[full["params"][0]]
            for key in ("momentum_buffer", "second_momentum_buffer"):
                actual = state[key].detach().cpu()
                equal(f"muon:{group_index}:{key}", actual[:len(positions)],
                      expected[key].detach().cpu()[positions])
                if group_index in adapter.padding:
                    assert torch.equal(actual[len(positions):], torch.zeros_like(actual[len(positions):]))
            if group_index in adapter.padding:
                dummy = adapter.padding[group_index]
                assert torch.equal(dummy.detach().cpu(), torch.zeros_like(dummy, device="cpu"))
                assert torch.equal(dummy.grad.cpu(), torch.zeros_like(dummy, device="cpu"))
    # A digest collective checks every weight byte without an unsupported CPU
    # object collective on the Neuron backend. No reduction can hide disagreement.
    local_digest = torch.tensor(list(digest.digest()), dtype=torch.float32).to(device)
    all_digests = torch.empty(32 * adapter.world_size, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(all_digests, local_digest)
    T.synchronize(device)
    all_digests = all_digests.cpu().view(adapter.world_size, 32)
    rank_exact = all(torch.equal(row, all_digests[0]) for row in all_digests)
    assert [id(p) for p in parameters] == adapter.parameter_ids
    print("OPTIMIZER_OWNERSHIP_EQUIVALENCE " + json.dumps(dict(
        tag=tag, rank=adapter.rank, exact_reference=not failures,
        failure_count=len(failures), failures=failures[:8],
        all_rank_weight_bytes_exact=rank_exact, weight_sha256=digest.hexdigest())), flush=True)
    assert rank_exact and not failures, (tag, failures[:8])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "neuron"), required=True)
    parser.add_argument("--real-shape", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--pad-singleton-muon", action="store_true")
    parser.add_argument("--adamw-math-mode", type=int, choices=(0, 2), default=None,
                        help="Compare ownership against replicated identical NKI AdamW math. "
                             "Its independent arithmetic gates remain required.")
    args = parser.parse_args()
    if args.adamw_math_mode is not None and (args.device != "neuron" or not args.real_shape):
        parser.error("NKI AdamW composition requires Neuron and the actual dense1024 tables")
    if args.device == "neuron":
        import torch_neuronx
    torch.set_num_threads(4)
    T.USE_FOREACH_OPTIM = T.USE_TENSOR_LR_SCALARS = T.USE_COMPILE_MUON = True
    T.USE_NKI_MUON = T.USE_CAUTIOUS_WD = False
    T.COMPUTE_DTYPE = torch.bfloat16
    _, rank, _, world, device = T.init_runtime(args.device)
    assert world == 4
    try:
        groups = template_groups(args.real_shape)
        generator = torch.Generator().manual_seed(9407)
        initial = [torch.randn(p.shape, generator=generator) * .03 for g in groups for p in g["params"]]
        reference_params, reference, _ = make_optimizer(
            groups, initial, device, False, adamw_math_mode=args.adamw_math_mode)
        parameters, candidate, adapter = make_optimizer(
            groups, initial, device, True, args.pad_singleton_muon, args.adamw_math_mode)
        del initial
        print("OPTIMIZER_OWNERSHIP_PLAN " + json.dumps(dict(
            rank=rank, pad_singleton_muon=args.pad_singleton_muon,
            adamw_math_mode=args.adamw_math_mode,
            reference_math="native" if args.adamw_math_mode is None else "identical replicated NKI AdamW",
            padded_groups=list(adapter.padding), **adapter.summary)), flush=True)
        # Exercise the actual Linear cache after owner updates and all-gather.
        cache_parameter = next(p for p in parameters if p.ndim == 2)
        linear = T.Linear(cache_parameter.shape[1], cache_parameter.shape[0], bias=False)
        linear.weight = cache_parameter
        projection = (torch.compile(linear, backend="neuron", fullgraph=True, dynamic=False)
                      if args.device == "neuron" else linear)
        x = torch.randn(2, cache_parameter.shape[1], generator=generator).to(device).bfloat16()
        last_gradients = None
        expected_reference_calls = expected_owned_calls = 0
        for iteration in range(1, 5):
            gradients = [torch.randn(p.shape, generator=generator) * .003 for p in parameters]
            if iteration == 2:
                for index in range(0, len(gradients), 5):
                    gradients[index] = None
            if iteration == 3:
                for index in range(1, len(gradients), 4):
                    gradients[index].zero_()
            for opt, ps in ((reference, reference_params), (candidate, parameters)):
                for index, group in enumerate(opt.param_groups):
                    group["lr"] = 0.0 if iteration == 4 else groups[index]["lr"] / iteration
                    if group["kind"] == "muon":
                        group["momentum"] = .85 + iteration * .02
                    else:
                        group["betas"] = (.8 if iteration < 3 else .9, .96)
                for p, gradient in zip(ps, gradients):
                    p.grad = None if gradient is None else gradient.to(device)
                opt.step(update_adamw=iteration != 3)
            T.synchronize(device)
            if args.adamw_math_mode is not None:
                active_tables = [entry for entry, gradient in zip(adapter.entries, gradients)
                                 if entry["kind"] == "adamw" and entry["shape"] == [8192, 1024]
                                 and gradient is not None and iteration != 3]
                expected_reference_calls += len(active_tables)
                expected_owned_calls += sum(entry["owner"] == rank for entry in active_tables)
                assert getattr(reference, "_fresh_start_adamw_calls", 0) == expected_reference_calls
                assert getattr(candidate, "_fresh_start_adamw_calls", 0) == expected_owned_calls
            compare(reference_params, reference, parameters, candidate, adapter, device, f"iteration{iteration}")
            for p, gradient in zip(parameters, gradients):
                assert (p.grad is None) if gradient is None else torch.equal(p.grad.cpu(), gradient)
            with torch.no_grad():
                linear.refresh_weight_cache(torch.bfloat16)
                assert torch.equal(linear._cached_weight.cpu(), cache_parameter.detach().cpu().bfloat16())
                torch.testing.assert_close(projection(x).cpu(),
                                           torch.nn.functional.linear(x, cache_parameter.bfloat16()).cpu(),
                                           rtol=0, atol=0)
            last_gradients = gradients
        # Save only persistent local optimizer state in the explicit ownership
        # schema, with all model weights. A fresh instance must resume exactly.
        payload = dict(weights=[p.detach().cpu().clone() for p in parameters],
                       optimizer=cpu_tree(candidate.state_dict()))
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        restored_params, restored, restored_adapter = make_optimizer(
            groups, payload["weights"], device, True, args.pad_singleton_muon, args.adamw_math_mode)
        restored.load_state_dict(payload["optimizer"])
        invalid = {**payload["optimizer"], "rank": (rank + 1) % world}
        try:
            restored.load_state_dict(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong-owner checkpoint accepted")
        if args.adamw_math_mode is not None:
            invalid_math = {**payload["optimizer"], "adamw_math_mode": None}
            try:
                restored.load_state_dict(invalid_math)
            except ValueError:
                pass
            else:
                raise AssertionError("different AdamW arithmetic checkpoint accepted")
        del payload, stream
        for opt, ps in ((reference, reference_params), (candidate, parameters), (restored, restored_params)):
            for group in opt.param_groups:
                group["lr"] = .007
            for p, gradient in zip(ps, last_gradients):
                p.grad = None if gradient is None else gradient.to(device)
            opt.step()
        T.synchronize(device)
        compare(reference_params, reference, parameters, candidate, adapter, device, "continued")
        compare(reference_params, reference, restored_params, restored, restored_adapter, device, "restored")
        del restored_params, restored, restored_adapter
        print("PASS: exact owned/replicated updates, states, cross-rank weights, "
              "None/zero gradients, skipped AdamW, schedules, cache and state restore", flush=True)
        if args.adamw_math_mode is not None:
            assert expected_reference_calls > 0
            print("OPTIMIZER_OWNERSHIP_ADAMW_COMPOSITION " + json.dumps(dict(
                rank=rank, math_mode=args.adamw_math_mode,
                checked_reference_calls=expected_reference_calls,
                checked_owned_calls=expected_owned_calls,
                replicated_same_math_exact=True, cross_rank_weights_exact=True,
                wrong_math_checkpoint_rejected=True,
                independent_arithmetic_authority=("wave66 reciprocal-refinement integration gate"
                    if args.adamw_math_mode == 2 else "wave60 original FP32 integration gate"),
                speed_or_quality_promotion=False)), flush=True)
        if args.device == "neuron" and args.steps:
            arms = {"replicated_optimizer": reference.step, "owned_optimizer_with_exchange": candidate.step}
            samples = {name: [] for name in arms}
            with torch.no_grad():
                for call in arms.values():
                    for _ in range(3):
                        call()
                        T.synchronize(device)
                for block in range(3):
                    order = list(arms) if block % 2 == 0 else list(reversed(arms))
                    for name in order:
                        dist.barrier()
                        T.synchronize(device)
                        begin = time.perf_counter()
                        for _ in range(args.steps):
                            arms[name]()
                            T.synchronize(device)
                        samples[name].append((time.perf_counter() - begin) * 1000 / args.steps)
            compare(reference_params, reference, parameters, candidate, adapter, device, "after_timing")
            print("OPTIMIZER_OWNERSHIP_BENCH " + json.dumps(dict(
                rank=rank, samples_ms=samples,
                median_ms={name: statistics.median(values) for name, values in samples.items()},
                full_optimizer_including_exchange=True, whole_model=False)), flush=True)
        dist.barrier()
    finally:
        T.cleanup_runtime()


if __name__ == "__main__":
    main()
