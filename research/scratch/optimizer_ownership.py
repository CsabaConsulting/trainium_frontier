"""Scratch whole-matrix optimizer ownership with exact FP32 weight exchange.

Keep the caller's complete param_groups for scheduling, clipping, reduction and
zero_grad. Only step dispatch uses persistent owner subsets. Muon's grouped
states are indexed by the first *owned* matrix and retain their original axes.
"""
from types import MethodType

import torch
import torch.distributed as dist


def make_plan(groups, world_size):
    muon_ms = {(1024, 1024): 12.54413 / 24,
               (1024, 4096): 9.76316 / 6,
               (4096, 1024): 8.98251 / 6,
               (8, 32): .23355 / 3,
               # Source144 actual-width groups, native dispatch/stack/copy/update:
               # fs-w144-width1152-optimizer-group-costs-r1.summary.json.
               # Amortized per-matrix estimates from representative owner counts;
               # actual grouped cost is not perfectly linear. D1024 is unchanged.
               (1152, 1152): 5.493693903554231 / 6,
               (1152, 4608): 5.106646800413728 / 2,
               (4608, 1152): 5.528476898325607 / 2,
               (9, 32): .6609485950320959 / 2}
    entries, parameters, seen = [], [], set()
    for group_index, group in enumerate(groups):
        if group["kind"] not in ("adamw", "muon"):
            raise ValueError("ownership probe is restricted to AdamW and Muon")
        if "depth_mults" in group:
            raise ValueError("per-depth optimizer multipliers require a separate ownership gate")
        for position, p in enumerate(group["params"]):
            if id(p) in seen or p.dtype != torch.float32 or not p.is_contiguous():
                raise ValueError("ownership requires unique contiguous FP32 Parameters")
            seen.add(id(p))
            cost = (muon_ms.get(tuple(p.shape), .01) if group["kind"] == "muon"
                    else 4.00487945 * p.numel() / (8192 * 1024) + .01)
            entries.append(dict(index=len(entries), group=group_index, position=position,
                                kind=group["kind"], shape=list(p.shape),
                                elements=p.numel(), estimated_ms=cost))
            parameters.append(p)
    if not parameters:
        raise ValueError("empty optimizer")
    counts, costs = [0] * world_size, [0.0] * world_size
    for entry in sorted(entries, key=lambda e: (-e["estimated_ms"], -e["elements"], e["index"])):
        owner = min(range(world_size), key=lambda r: (costs[r], counts[r], r))
        entry.update(owner=owner, offset=counts[owner])
        counts[owner] += entry["elements"]
        costs[owner] += entry["estimated_ms"]
    stride = (max(counts) + 127) // 128 * 128
    return entries, parameters, dict(elements_by_owner=counts,
                                    estimated_ms_by_owner=costs,
                                    shard_elements=stride,
                                    total_padded_bytes=stride * world_size * 4)


class OwnedOptimizer:
    def __init__(self, optimizer, pad_singleton_muon=False):
        if not dist.is_initialized():
            raise ValueError("optimizer ownership requires an initialized process group")
        if optimizer.state:
            raise ValueError("install ownership before the first update; use its checkpoint loader to resume")
        self.optimizer = optimizer
        self.pad_singleton_muon = bool(pad_singleton_muon)
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.entries, self.parameters, self.summary = make_plan(optimizer.param_groups, self.world_size)
        self.parameter_ids = [id(p) for p in self.parameters]
        self.groups, self.positions, self.padding = [], [], {}
        for group_index, source in enumerate(optimizer.param_groups):
            positions = [e["position"] for e in self.entries
                         if e["group"] == group_index and e["owner"] == self.rank]
            self.positions.append(positions)
            self.groups.append({**source, "params": [source["params"][i] for i in positions]})
            if self.pad_singleton_muon and source["kind"] == "muon" and len(positions) == 1:
                # Native batch-one tall Muon differs from batches two/six on
                # this stack. The original math is per matrix; a zero dummy
                # keeps the real matrix on the already matching batched path.
                # It is never exchanged, trained by the model or serialized as
                # a model Parameter. Its grouped moment rows remain explicit.
                dummy = torch.nn.Parameter(torch.zeros_like(self.groups[-1]["params"][0]))
                dummy.grad = torch.zeros_like(dummy)
                self.groups[-1]["params"].append(dummy)
                self.padding[group_index] = dummy
        device = self.parameters[0].device
        if any(p.device != device for p in self.parameters):
            raise ValueError("all Parameters must share a device")
        stride = self.summary["shard_elements"]
        self.send = torch.zeros(stride, dtype=torch.float32, device=device)
        self.received = torch.empty(stride * self.world_size, dtype=torch.float32, device=device)
        self.owned, self.send_views, self.receive_views = [], [], []
        for entry, p in zip(self.entries, self.parameters):
            self.receive_views.append(self.received.narrow(
                0, entry["owner"] * stride + entry["offset"], entry["elements"]).view_as(p))
            if entry["owner"] == self.rank:
                self.owned.append(p)
                self.send_views.append(self.send.narrow(0, entry["offset"], entry["elements"]).view_as(p))
        self.raw_state_dict = optimizer.state_dict
        self.raw_load_state_dict = optimizer.load_state_dict
        self.calls = 0

    @torch.no_grad()
    def step(self, update_adamw=True):
        opt = self.optimizer
        for source, owned in zip(opt.param_groups, self.groups):
            # Same dict object each step preserves the optimizer's scalar-cache
            # identity. Read schedules from the caller's complete groups.
            owned.update({key: value for key, value in source.items() if key != "params"})
            if not owned["params"]:
                continue
            if owned["kind"] == "adamw":
                if update_adamw:
                    opt._adamw_step_foreach(owned)
            else:
                # Retain the native Muon semantics: None gradients become zero
                # rows, but momentum/decay can still update that matrix.
                opt._muon_step(owned)
        if self.owned:
            torch._foreach_copy_(self.send_views, self.owned)
        dist.all_gather_into_tensor(self.received, self.send)
        torch._foreach_copy_(self.parameters, self.receive_views)
        self.calls += 1

    def state_dict(self):
        # Explicit rank-local schema prevents an incomplete ordinary optimizer
        # state from masquerading as a portable replicated state_dict. Full
        # model weights are saved separately by the existing checkpoint writer.
        state = self.raw_state_dict()
        state["state"] = {key: {name: value for name, value in fields.items()
                                 if name not in ("stacked_params", "stacked_grads")}
                          for key, fields in state["state"].items()}
        return dict(schema="fresh-start-owned-optimizer-v1", rank=self.rank,
                    world_size=self.world_size, entries=self.entries,
                    pad_singleton_muon=self.pad_singleton_muon,
                    adamw_math_mode=getattr(self.optimizer, "_fresh_start_adamw_math_mode", None),
                    optimizer=state)

    def load_state_dict(self, payload):
        if (payload.get("schema") != "fresh-start-owned-optimizer-v1"
                or payload.get("rank") != self.rank
                or payload.get("world_size") != self.world_size
                or payload.get("pad_singleton_muon") != self.pad_singleton_muon
                or payload.get("adamw_math_mode") != getattr(
                    self.optimizer, "_fresh_start_adamw_math_mode", None)
                or payload.get("entries") != self.entries):
            raise ValueError("optimizer checkpoint ownership/schema mismatch")
        self.raw_load_state_dict(payload["optimizer"])
        self.optimizer._scalar_cache.clear()


def install_optimizer(optimizer, train_module, pad_singleton_muon=False):
    if not train_module.USE_FOREACH_OPTIM or not train_module.USE_TENSOR_LR_SCALARS:
        raise ValueError("ownership gate requires the current foreach/runtime-scalar policy")
    if hasattr(optimizer, "_fresh_start_owned"):
        raise ValueError("ownership already installed")
    adapter = OwnedOptimizer(optimizer, pad_singleton_muon=pad_singleton_muon)
    optimizer._fresh_start_owned = adapter
    optimizer.step = MethodType(lambda self, update_adamw=True: adapter.step(update_adamw), optimizer)
    optimizer.state_dict = MethodType(lambda self: adapter.state_dict(), optimizer)
    optimizer.load_state_dict = MethodType(lambda self, payload: adapter.load_state_dict(payload), optimizer)
    return adapter
