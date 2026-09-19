#!/usr/bin/env python3
"""Measure actual model optimizer groups before planning a new width's owners.

The model inventory comes from a recorded phase configuration. A measurement
runs one actual group or a prefix of it in a fresh process, including the native
stack/copy/update work. It does not change the ownership planner or establish
replicated/owned numerical equivalence. Those gates remain separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def inventory(config_log, width):
    records = [json.loads(line.split(" ", 1)[1])
               for line in config_log.read_text().splitlines()
               if line.startswith("PROFILE_CONFIG ")]
    assert len(records) == 1, "Require one exact recorded phase configuration"
    record = records[0]
    for name, value in record["toggles"].items():
        assert hasattr(T, name), name
        setattr(T, name, value)
    T.N_EMBD = width
    T.USE_FOREACH_OPTIM = T.USE_TENSOR_LR_SCALARS = T.USE_COMPILE_MUON = True
    T.USE_NKI_MUON = False
    T.COMPUTE_DTYPE = torch.bfloat16
    config = T.build_config()
    assert config.n_embd == width and width % config.n_head == 0
    with torch.device("meta"):
        model = T.GPT(config)
        groups = model.setup_optimizer().param_groups
    names = {id(p): name for name, p in model.named_parameters()}
    seen = set()
    rows = []
    for index, group in enumerate(groups):
        assert group["kind"] in ("adamw", "muon")
        parameters = []
        for p in group["params"]:
            assert id(p) not in seen and p.dtype == torch.float32
            seen.add(id(p))
            parameters.append(dict(name=names[id(p)], shape=list(p.shape), elements=p.numel()))
        rows.append(dict(index=index, kind=group["kind"], count=len(parameters),
                         elements=sum(p["elements"] for p in parameters),
                         hyperparameters={k: v for k, v in group.items() if k != "params"},
                         parameters=parameters))
    assert seen == {id(p) for p in model.parameters() if p.requires_grad}
    return groups, dict(width=width, config=vars(config), groups=rows,
                        parameters=sum(r["elements"] for r in rows),
                        input_config_log=str(config_log),
                        input_config_sha256=hashlib.sha256(config_log.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-log", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--group-index", type=int)
    parser.add_argument("--count", type=int, help="Use the first count real parameters of this group")
    parser.add_argument("--pad-singleton-muon", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--steps-per-block", type=int, default=10)
    args = parser.parse_args()
    if args.width < 128 or args.width % 128:
        parser.error("width must be a positive multiple of128")
    if min(args.warmup_steps, args.blocks, args.steps_per_block) < 1:
        parser.error("warmups, blocks and steps must be positive")
    torch.set_num_threads(4)
    groups, record = inventory(args.config_log, args.width)
    print("OPTIMIZER_ACTUAL_INVENTORY " + json.dumps(record), flush=True)
    if args.inventory_only:
        return
    if args.group_index is None or not 0 <= args.group_index < len(groups):
        parser.error("a valid group index is required for device measurement")
    source = groups[args.group_index]
    count = len(source["params"]) if args.count is None else args.count
    if not 1 <= count <= len(source["params"]):
        parser.error("count must select a nonempty prefix of the actual group")
    if args.pad_singleton_muon and (source["kind"] != "muon" or count != 1):
        parser.error("singleton padding applies only to one real Muon parameter")
    import torch_neuronx  # noqa: F401
    device = torch.device("neuron:0")
    generator = torch.Generator().manual_seed(9407)
    parameters = []
    for template in source["params"][:count]:
        p = torch.nn.Parameter((torch.randn(template.shape, generator=generator) * .03).to(device))
        p.grad = (torch.randn(template.shape, generator=generator) * .003).to(device)
        parameters.append(p)
    if args.pad_singleton_muon:
        dummy = torch.nn.Parameter(torch.zeros_like(parameters[0]))
        dummy.grad = torch.zeros_like(dummy)
        parameters.append(dummy)
    group = {**source, "params": parameters}
    optimizer = T.MuonAdamW([group])
    group = optimizer.param_groups[0]
    step = optimizer._muon_step if group["kind"] == "muon" else optimizer._adamw_step_foreach
    T.synchronize(device)
    with torch.no_grad():
        for _ in range(args.warmup_steps):
            step(group)
            T.synchronize(device)
        samples = []
        for _ in range(args.blocks):
            start = time.perf_counter()
            for _ in range(args.steps_per_block):
                step(group)
            T.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000 / args.steps_per_block)
    finite_tensors = 0
    for tensor in parameters:
        assert bool(torch.isfinite(tensor.detach().cpu()).all())
        finite_tensors += 1
    for fields in optimizer.state.values():
        for tensor in fields.values():
            if isinstance(tensor, torch.Tensor):
                assert bool(torch.isfinite(tensor.detach().cpu()).all())
                finite_tensors += 1
    if args.pad_singleton_muon:
        assert torch.count_nonzero(dummy.detach().cpu()).item() == 0
        assert torch.count_nonzero(dummy.grad.cpu()).item() == 0
    print("OPTIMIZER_GROUP_COST " + json.dumps(dict(
        width=args.width, group_index=args.group_index, kind=group["kind"],
        actual_group_count=len(source["params"]), real_parameter_count=count,
        measured_parameter_count=len(parameters), pad_singleton_muon=args.pad_singleton_muon,
        shapes=[list(p.shape) for p in source["params"][:count]],
        warmup_steps=args.warmup_steps, blocks=args.blocks, steps_per_block=args.steps_per_block,
        milliseconds_per_step=samples, median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples), min_ms=min(samples), max_ms=max(samples),
        finite_parameter_and_state_tensors=finite_tensors,
        compiler_flags=os.environ.get("NEURON_CC_FLAGS"),
        scope="Native group dispatch, stacking/copies and updates; no owner exchange or full model.",
        input_policy="Seed9407 normal FP32 parameters*.03 and fixed normal gradients*.003; synthetic cost probe.",
        input_config_sha256=record["input_config_sha256"],
        ownership_plan_modified=False, numerical_ownership_gate_required=True)), flush=True)


if __name__ == "__main__":
    main()
