"""Gate-only five-step observer for the A1 hybrid-ownership equivalence package.

This module is never imported by the production candidate.  It records compact hashes and
metadata only; tensor values are never printed or written.  Every report file is O_EXCL and each
rank owns exactly one file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

import torch


REPORT_SCHEMA = "a1-native-five-equivalence-v1"
TABLE_NAMES = ("ngram_embeds.b5.weight", "ngram_embeds.t5.weight")
ARM_LABELS = {
    "G0": "fs-20260924-v2-a1-native5-g0-hybrid-observation-off-trn21-r1",
    "G1": "fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn21-r3",
    "G2": "fs-20260924-v2-a1-native5-g2-ordinary-observation-on-trn21-r1",
}


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().to("cpu").contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    finite = bool(torch.isfinite(cpu).all().item())
    if not finite:
        raise ValueError("native gate encountered a nonfinite tensor")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": cpu.numel(),
        "finite": finite,
        "norm": float(cpu.float().norm().item()),
    }


def _model_digest(model: torch.nn.Module) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().to("cpu").contiguous()
        if not bool(torch.isfinite(cpu).all().item()):
            raise ValueError(f"nonfinite persistent model state {name}")
        raw = cpu.view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(cpu.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(raw)
        tensor_count += 1
    return {"sha256": digest.hexdigest(), "tensor_count": tensor_count}


class NativeGateObserver:
    def __init__(self, *, arm: str, detailed: bool, expected_mode: str, rank: int,
                 world_size: int, report_dir: str | Path, dist_module,
                 label: str, host: str, config: dict[str, Any]):
        if arm not in ("G0", "G1", "G2"):
            raise ValueError(f"unknown native gate arm {arm!r}")
        if expected_mode not in ("hybrid", "ordinary"):
            raise ValueError(f"unknown optimizer mode {expected_mode!r}")
        if (arm, detailed, expected_mode) not in (
            ("G0", False, "hybrid"), ("G1", True, "hybrid"),
            ("G2", True, "ordinary"),
        ):
            raise ValueError("native gate arm/observation/mode contract mismatch")
        if world_size != 4 or rank not in range(world_size):
            raise ValueError("native gate requires world_size=4 and a valid rank")
        if host != "trn21" or label != ARM_LABELS[arm]:
            raise ValueError("native gate host/label contract mismatch")
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        if report_dir.is_symlink() or not report_dir.is_dir():
            raise ValueError("native gate report directory must be a real directory")
        path = report_dir / f"rank-{rank}.jsonl"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(path, flags, 0o600)
        self.path = path
        self.arm = arm
        self.detailed = detailed
        self.expected_mode = expected_mode
        self.rank = rank
        self.world_size = world_size
        self.dist = dist_module
        self.model = None
        self.optimizer = None
        self.ownership = None
        self.names: dict[int, str] = {}
        self.loss_steps: list[int] = []
        self.pre_steps: list[int] = []
        self.post_steps: list[int] = []
        self.collectives: list[dict[str, Any]] = []
        self.rms_calls = 0
        self._install_collective_spy()
        self.emit("header", schema=REPORT_SCHEMA, arm=arm, detailed=detailed,
                  optimizer_mode=expected_mode, rank=rank, world_size=world_size,
                  label=label, host=host, config=config)

    def emit(self, kind: str, **record: Any) -> None:
        payload = json.dumps({"kind": kind, **record}, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        os.write(self._fd, payload)
        os.fsync(self._fd)

    def _install_collective_spy(self) -> None:
        original_all_reduce = self.dist.all_reduce
        original_all_gather = self.dist.all_gather_into_tensor

        def all_reduce(tensor, *args, **kwargs):
            op = kwargs.get("op", args[0] if args else None)
            self.collectives.append({
                "index": len(self.collectives), "op": "all_reduce", "reduce_op": str(op),
                "shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(),
            })
            return original_all_reduce(tensor, *args, **kwargs)

        def all_gather_into_tensor(output, input_tensor, *args, **kwargs):
            self.collectives.append({
                "index": len(self.collectives), "op": "all_gather_into_tensor",
                "output_shape": list(output.shape), "input_shape": list(input_tensor.shape),
                "output_dtype": str(output.dtype), "input_dtype": str(input_tensor.dtype),
                "output_numel": output.numel(), "input_numel": input_tensor.numel(),
            })
            return original_all_gather(output, input_tensor, *args, **kwargs)

        self.dist.all_reduce = all_reduce
        self.dist.all_gather_into_tensor = all_gather_into_tensor

    def bind(self, model: torch.nn.Module, optimizer, ownership=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.ownership = ownership
        raw_rmsprop_step = optimizer._rmsprop_step

        def observed_rmsprop_step(group):
            self.rms_calls += 1
            return raw_rmsprop_step(group)

        optimizer._rmsprop_step = observed_rmsprop_step
        self.names = {id(parameter): name for name, parameter in model.named_parameters()}
        missing = [name for name in TABLE_NAMES if name not in set(self.names.values())]
        if missing:
            raise ValueError(f"native gate missing exact A1 tables: {missing}")
        group_inventory = []
        for index, group in enumerate(optimizer.param_groups):
            names = [self.names.get(id(parameter), "<gate-padding>") for parameter in group["params"]]
            group_inventory.append({"index": index, "kind": group.get("kind"),
                                    "parameter_names": names})
        if self.expected_mode == "hybrid":
            if ownership is None:
                raise ValueError("hybrid gate is missing ownership adapter")
            if any(entry["kind"] == "rmsprop" for entry in ownership.entries):
                raise ValueError("RMSProp table entered owner planning")
            rms_groups = [index for index, group in enumerate(optimizer.param_groups)
                          if group.get("kind") == "rmsprop"]
            if rms_groups != list(ownership.replicated_rmsprop_group_identity):
                raise ValueError("replicated RMSProp inventory mismatch")
            ownership_record = {
                "entries": ownership.entries,
                "summary": ownership.summary,
                "replicated_rmsprop_groups": rms_groups,
            }
        else:
            if ownership is not None:
                raise ValueError("ordinary gate unexpectedly installed ownership")
            ownership_record = None
        self.emit("inventory", groups=group_inventory, ownership=ownership_record,
                  model_state=_model_digest(model))

    def loss(self, step: int, loss: torch.Tensor) -> None:
        if step != len(self.loss_steps):
            raise ValueError("loss step sequence drift")
        cpu = loss.detach().float().to("cpu").reshape(())
        if not bool(torch.isfinite(cpu).item()):
            raise ValueError("nonfinite native gate loss")
        raw = struct.pack("<f", float(cpu.item()))
        self.loss_steps.append(step)
        self.emit("loss", step=step, value=float(cpu.item()), scalar_sha256=hashlib.sha256(raw).hexdigest())

    def gradients(self, step: int, stage: str, model: torch.nn.Module) -> None:
        if not self.detailed:
            return
        named = dict(model.named_parameters())
        tables = {}
        for name in TABLE_NAMES:
            grad = named[name].grad
            if grad is None:
                raise ValueError(f"missing gradient for {name} at {stage} step {step}")
            record = _tensor_record(grad)
            if not record["finite"]:
                raise ValueError(f"nonfinite gradient for {name} at {stage} step {step}")
            tables[name] = record
        self.emit("gradients", step=step, stage=stage, tables=tables)

    def clip_scalars(self, step: int, main_norm: torch.Tensor | None,
                     table_norm: torch.Tensor | None, max_norm: float) -> None:
        if not self.detailed:
            return
        values = []
        for value in (main_norm, table_norm):
            values.append(None if value is None else float(value.detach().float().to("cpu").item()))
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("nonfinite clip norm")
        encoded = b"".join(struct.pack("<d", -1.0 if value is None else value) for value in values)
        self.emit("clip", step=step, main_norm=values[0], table_norm=values[1],
                  max_norm=max_norm, scalar_bytes_sha256=hashlib.sha256(encoded).hexdigest())

    def _logical_optimizer_state(self) -> list[dict[str, Any]]:
        records = []
        for group_index, source in enumerate(self.optimizer.param_groups):
            kind = source.get("kind")
            local = source["params"] if self.ownership is None or kind == "rmsprop" else self.ownership.groups[group_index]["params"]
            if kind == "muon" and local:
                anchor = local[0]
                state = self.optimizer.state.get(anchor, {})
                for position, parameter in enumerate(local):
                    name = self.names.get(id(parameter))
                    if name is None:
                        continue
                    fields = {}
                    for field in ("momentum_buffer", "second_momentum_buffer"):
                        if field in state:
                            fields[field] = _tensor_record(state[field][position])
                    records.append({"name": name, "kind": kind, "fields": fields})
                continue
            for parameter in local:
                name = self.names.get(id(parameter))
                if name is None:
                    continue
                state = self.optimizer.state.get(parameter, {})
                fields = {}
                for field, value in sorted(state.items()):
                    if field in ("stacked_params", "stacked_grads", "nki_rms_scalars_cpu",
                                 "nki_rms_scalars_device"):
                        continue
                    fields[field] = (_tensor_record(value) if isinstance(value, torch.Tensor)
                                     else value)
                if kind == "rmsprop" and state:
                    step = state.get("step")
                    if type(step) is not int or step < 1:
                        raise ValueError(f"bad RMS state step for {name}")
                    fields["bias2"] = 1.0 - source["betas"][1] ** step
                records.append({"name": name, "kind": kind, "fields": fields})
        return sorted(records, key=lambda record: record["name"])

    def update(self, step: int, phase: str) -> None:
        expected = self.pre_steps if phase == "pre" else self.post_steps
        if phase not in ("pre", "post") or step != len(expected):
            raise ValueError("native gate update sequence drift")
        expected.append(step)
        named = dict(self.model.named_parameters())
        tables = {name: _tensor_record(named[name]) for name in TABLE_NAMES}
        if not all(record["finite"] for record in tables.values()):
            raise ValueError("nonfinite table parameter")
        rms_calls = self.rms_calls
        if (self.ownership is not None
                and rms_calls != self.ownership.replicated_rmsprop_calls):
            raise ValueError("hybrid RMS call counters disagree")
        state = self._logical_optimizer_state()
        if phase == "post":
            table_state = {record["name"]: record for record in state if record["name"] in TABLE_NAMES}
            if set(table_state) != set(TABLE_NAMES):
                raise ValueError("missing table optimizer state")
            for record in table_state.values():
                if record["fields"].get("step") != step + 1:
                    raise ValueError("duplicate or missed RMS update")
            if rms_calls != step + 1:
                raise ValueError("RMS call count mismatch")
        self.emit("update", step=step, phase=phase, tables=tables,
                  model_state=_model_digest(self.model), optimizer_state=state,
                  rms_call_count=rms_calls)

    def finish(self, expected_steps: int=5) -> None:
        if self.loss_steps != list(range(expected_steps)):
            raise ValueError("native gate did not record exactly five losses")
        if self.pre_steps != list(range(expected_steps)) or self.post_steps != list(range(expected_steps)):
            raise ValueError("native gate did not record exactly five updates")
        if not self.collectives:
            raise ValueError("native gate observed no collectives")
        sequence_bytes = json.dumps(self.collectives, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")
        self.emit("collectives", count=len(self.collectives), sequence=self.collectives,
                  sequence_sha256=hashlib.sha256(sequence_bytes).hexdigest())
        self.emit("complete", expected_steps=expected_steps,
                  final_model_state=_model_digest(self.model))
        os.fsync(self._fd)
        os.close(self._fd)
        self._fd = -1
        os.chmod(self.path, 0o400, follow_symlinks=False)
        directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
