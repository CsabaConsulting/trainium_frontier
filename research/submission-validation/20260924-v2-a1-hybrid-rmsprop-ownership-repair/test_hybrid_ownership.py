#!/usr/bin/env python3
"""CPU/static gates for the inline replicated-RMSProp/owned-AdamW-Muon repair."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT / "submissions/dense6x1024_public0992851/train.py"
PARENT = "6cb04c550d497e09b2e0ffa530b5ec4488c229b4"
OLD_SHA = "35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
A1_FLAGS = (
    "--n-embd 1024 --ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 "
    "--ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop "
    "--ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift "
    "--compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 "
    "--clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015"
)


class FakeDist:
    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.gathered = None
        self.collective_calls = 0

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def all_gather_into_tensor(self, received, send):
        self.collective_calls += 1
        if self.world_size == 1:
            received.copy_(send)
        else:
            if self.gathered is None or self.gathered.shape != received.shape:
                raise AssertionError("fake-world gather fixture missing")
            received.copy_(self.gathered)


DIST = FakeDist()


def load_adapter():
    source = CANDIDATE.read_text()
    start = source.index("def make_plan(groups, world_size):")
    end = source.index("\ndef build_kernels():", start)
    namespace = {
        "torch": torch,
        "dist": DIST,
        "json": json,
        "MethodType": types.MethodType,
    }
    exec(compile(source[start:end], str(CANDIDATE), "exec"), namespace)
    return namespace


def load_full_candidate():
    pyarrow = types.ModuleType("pyarrow")
    parquet = types.ModuleType("pyarrow.parquet")
    pyarrow.parquet = parquet
    prepare = types.ModuleType("prepare")
    constants = {
        "EVAL_TIMEOUT_SECONDS": 1200, "TOKENIZER_VOCAB_SIZE": 65536,
        "TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS": 900, "TRAIN_STARTUP_STEPS_EXCLUDED": 5,
        "TRAIN_TIME_BUDGET_SECONDS": 1800, "Tokenizer": type("Tokenizer", (), {}),
    }
    for name, value in constants.items():
        setattr(prepare, name, value)
    for name in ("_document_batches", "_row_groups_for_split", "causality_check", "ensure_tokenizer",
                 "evaluate_bpb", "get_dist_info", "get_token_bytes", "make_dataloader", "print0",
                 "record_step_boundary"):
        setattr(prepare, name, lambda *args, **kwargs: None)
    saved = {name: sys.modules.get(name) for name in ("pyarrow", "pyarrow.parquet", "prepare")}
    module_name = "hybrid_ownership_full_candidate"
    try:
        sys.modules["pyarrow"] = pyarrow
        sys.modules["pyarrow.parquet"] = parquet
        sys.modules["prepare"] = prepare
        spec = importlib.util.spec_from_file_location(module_name, CANDIDATE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        module.dist = DIST
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


ADAPTER = load_adapter()
make_plan = ADAPTER["make_plan"]
OwnedOptimizer = ADAPTER["OwnedOptimizer"]


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return copy.deepcopy(value)


class ToyOptimizer:
    """Small deterministic optimizer exposing the candidate dispatch surface."""

    def __init__(self, groups):
        self.param_groups = groups
        self.state = {}
        self._scalar_cache = {}

    def _state(self, parameter, key):
        state = self.state.setdefault(parameter, {"step": 0})
        if key not in state:
            state[key] = torch.zeros_like(parameter, dtype=torch.float32)
        return state

    @torch.no_grad()
    def _adamw_step_foreach(self, group):
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self._state(parameter, "exp_avg")
            state.setdefault("exp_avg_sq", torch.zeros_like(parameter, dtype=torch.float32))
            state["step"] += 1
            grad = parameter.grad.float()
            state["exp_avg"].lerp_(grad, 1 - beta1)
            state["exp_avg_sq"].lerp_(grad.square(), 1 - beta2)
            update = (state["exp_avg"] / (1 - beta1 ** state["step"])) / (
                (state["exp_avg_sq"] / (1 - beta2 ** state["step"])).sqrt() + group["eps"]
            )
            parameter.mul_(1 - group["lr"] * group["weight_decay"])
            parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])

    @torch.no_grad()
    def _muon_step(self, group):
        if not group["params"]:
            return
        parameter = group["params"][0]
        state = self.state.setdefault(parameter, {})
        count = len(group["params"])
        if not state:
            state["momentum_buffer"] = torch.zeros(count, *parameter.shape, dtype=parameter.dtype)
            state["second_momentum_buffer"] = torch.zeros(count, parameter.shape[-2], 1)
        for index, item in enumerate(group["params"]):
            grad = torch.zeros_like(item) if item.grad is None else item.grad
            state["momentum_buffer"][index].lerp_(grad, 1 - group["momentum"])
            item.add_(state["momentum_buffer"][index], alpha=-group["lr"])

    @torch.no_grad()
    def _rmsprop_step(self, group):
        beta2 = group["betas"][1]
        if group["weight_decay"] != 0:
            raise AssertionError("toy RMSProp keeps the candidate's zero-WD contract")
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state.setdefault(parameter, {})
            if not state:
                state["step"] = 0
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            grad = parameter.grad.float()
            state["exp_avg_sq"].lerp_(grad.to(parameter.dtype).square(), 1 - beta2)
            update = grad / ((state["exp_avg_sq"].float() / (1 - beta2 ** state["step"])).sqrt() + group["eps"])
            parameter.sub_((update * group["lr"]).to(parameter.dtype))

    def step(self, update_adamw=True):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                if update_adamw:
                    self._adamw_step_foreach(group)
            elif group["kind"] == "muon":
                self._muon_step(group)
            elif group["kind"] == "rmsprop":
                if update_adamw:
                    self._rmsprop_step(group)
            else:
                raise ValueError("unknown toy optimizer kind")

    def state_dict(self):
        parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        return {
            "state": {indices[id(parameter)]: clone_value(state) for parameter, state in self.state.items()},
            "param_groups": [
                {**{key: clone_value(value) for key, value in group.items() if key != "params"},
                 "params": [indices[id(parameter)] for parameter in group["params"]]}
                for group in self.param_groups
            ],
        }

    def load_state_dict(self, payload):
        parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        if len(payload["param_groups"]) != len(self.param_groups):
            raise ValueError("group count mismatch")
        self.state = {parameters[index]: clone_value(state) for index, state in payload["state"].items()}
        for target, source in zip(self.param_groups, payload["param_groups"]):
            if source["params"] != list(range(source["params"][0], source["params"][0] + len(target["params"]))):
                raise ValueError("parameter order mismatch")
            target.update({key: clone_value(value) for key, value in source.items() if key != "params"})


def group_templates():
    return [
        {"kind": "adamw", "lr": 0.02, "betas": (0.8, 0.96), "eps": 1e-8, "weight_decay": 0.01},
        {"kind": "muon", "lr": 0.015, "momentum": 0.9, "beta2": 0.9, "ns_steps": 5,
         "weight_decay": 0.0, "transposed": False},
        {"kind": "rmsprop", "lr": 0.03, "betas": (0.0, 0.95), "eps": 1e-8,
         "weight_decay": 0.0, "no_lr_decay": True},
    ]


def make_toy(initial):
    parameters = [torch.nn.Parameter(value.clone()) for value in initial]
    groups = group_templates()
    groups[0]["params"] = parameters[:2]
    groups[1]["params"] = parameters[2:3]
    groups[2]["params"] = parameters[3:]
    return parameters, ToyOptimizer(groups)


def set_grads(parameters, step):
    for index, parameter in enumerate(parameters):
        base = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        gradient = (base + 1 + 3 * step + index) * (0.001 + index * 0.0002)
        parameter.grad = gradient.to(parameter.dtype)


def assert_tree_equal(test, actual, expected, path="root"):
    test.assertEqual(type(actual), type(expected), path)
    if isinstance(actual, torch.Tensor):
        test.assertTrue(torch.equal(actual, expected), path)
    elif isinstance(actual, dict):
        test.assertEqual(set(actual), set(expected), path)
        for key in actual:
            assert_tree_equal(test, actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        test.assertEqual(len(actual), len(expected), path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_tree_equal(test, left, right, f"{path}[{index}]")
    else:
        test.assertEqual(actual, expected, path)


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


class PlanGates(unittest.TestCase):
    def setUp(self):
        DIST.rank, DIST.world_size, DIST.gathered, DIST.collective_calls = 0, 1, None, 0

    def test_rmsprop_is_replicated_and_never_owner_planned(self):
        adam = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
        rms = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        entries, parameters, summary = make_plan([
            {"kind": "adamw", "params": [adam]}, {"kind": "rmsprop", "params": [rms]},
        ], 4)
        self.assertEqual(parameters, [adam])
        self.assertEqual({entry["kind"] for entry in entries}, {"adamw"})
        self.assertEqual(summary["replicated_rmsprop_groups"], [{
            "group": 1, "parameters": [{"position": 0, "shape": [8], "elements": 8,
                                         "dtype": "torch.bfloat16", "contiguous": True,
                                         "requires_grad": True, "device_type": "cpu",
                                         "device_index": None}],
        }])

    def test_unknown_duplicate_and_bf16_owned_groups_fail_closed(self):
        fp32 = torch.nn.Parameter(torch.zeros(4))
        bf16 = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "recognizes only"):
            make_plan([{"kind": "sgd", "params": [fp32]}], 1)
        for kind in ("adamw", "muon"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "FP32"):
                make_plan([{"kind": kind, "params": [bf16]}], 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            make_plan([{"kind": "rmsprop", "params": [fp32]},
                       {"kind": "adamw", "params": [fp32]}], 1)
        for world in (0, -1, 1.0):
            with self.subTest(world=world), self.assertRaisesRegex(ValueError, "positive integer"):
                make_plan([{"kind": "adamw", "params": [fp32]}], world)


class WorldOneEquivalence(unittest.TestCase):
    def setUp(self):
        DIST.rank, DIST.world_size, DIST.gathered, DIST.collective_calls = 0, 1, None, 0
        generator = torch.Generator().manual_seed(42)
        self.initial = [torch.randn((2, 3), generator=generator), torch.randn(5, generator=generator),
                        torch.randn((2, 2), generator=generator),
                        torch.randn((3, 4), generator=generator).to(torch.bfloat16)]

    def test_multistep_skip_and_save_load_resume_are_bit_exact(self):
        reference_parameters, reference = make_toy(self.initial)
        candidate_parameters, candidate = make_toy(self.initial)
        adapter = OwnedOptimizer(candidate)
        skipped_rms = None
        for step, update_adamw in enumerate((True, False, True, True)):
            set_grads(reference_parameters, step)
            set_grads(candidate_parameters, step)
            if not update_adamw:
                skipped_rms = candidate_parameters[-1].detach().clone()
                skipped_adam = candidate_parameters[0].detach().clone()
            reference.step(update_adamw=update_adamw)
            adapter.step(update_adamw=update_adamw)
            if not update_adamw:
                self.assertTrue(torch.equal(candidate_parameters[-1], skipped_rms))
                self.assertTrue(torch.equal(candidate_parameters[0], skipped_adam))
            for actual, expected in zip(candidate_parameters, reference_parameters):
                self.assertTrue(torch.equal(actual, expected))
            assert_tree_equal(self, candidate.state_dict(), reference.state_dict())
        self.assertEqual(adapter.replicated_rmsprop_calls, 3)
        payload = {"weights": [parameter.detach().clone() for parameter in candidate_parameters],
                   "optimizer": clone_value(adapter.state_dict())}
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        restored_parameters, restored = make_toy(payload["weights"])
        restored_adapter = OwnedOptimizer(restored)
        restored_adapter.load_state_dict(payload["optimizer"])
        set_grads(reference_parameters, 9)
        set_grads(candidate_parameters, 9)
        set_grads(restored_parameters, 9)
        reference.step()
        adapter.step()
        restored_adapter.step()
        for actual, expected, resumed in zip(candidate_parameters, reference_parameters, restored_parameters):
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(resumed, expected))
        assert_tree_equal(self, restored.state_dict(), candidate.state_dict())
        bad = clone_value(payload["optimizer"])
        bad["replicated_rmsprop_groups"] = []
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            restored_adapter.load_state_dict(bad)

    def test_actual_muon_adamw_rmsprop_path_matches_owned_adapter(self):
        module = load_full_candidate()
        module.USE_FOREACH_OPTIM = True
        module.USE_TENSOR_LR_SCALARS = False
        module.NGRAM_RMS_FP32 = False
        module.USE_NKI_NGRAM_RMS = False

        def build():
            adam = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 8).reshape(2, 4))
            rms = torch.nn.Parameter(torch.linspace(-0.4, 0.4, 16).reshape(4, 4).to(torch.bfloat16))
            groups = [
                {"kind": "adamw", "params": [adam], "lr": 0.02, "betas": (0.8, 0.96),
                 "eps": 1e-10, "weight_decay": 0.01},
                {"kind": "rmsprop", "params": [rms], "lr": 0.03, "betas": (0.0, 0.999),
                 "eps": 1e-10, "weight_decay": 0.0, "no_lr_decay": True},
            ]
            return [adam, rms], module.MuonAdamW(groups)

        reference_parameters, reference = build()
        candidate_parameters, candidate = build()
        adapter = module.OwnedOptimizer(candidate)
        for step, update in enumerate((True, False, True)):
            set_grads(reference_parameters, step)
            set_grads(candidate_parameters, step)
            reference.step(update_adamw=update)
            adapter.step(update_adamw=update)
            for actual, expected in zip(candidate_parameters, reference_parameters):
                self.assertTrue(torch.equal(actual, expected))
            assert_tree_equal(self, candidate.state_dict(), reference.state_dict())
        self.assertEqual(adapter.replicated_rmsprop_calls, 2)
        saved = clone_value(adapter.state_dict())
        restored_parameters, restored = build()
        for target, source in zip(restored_parameters, candidate_parameters):
            target.data.copy_(source)
        restored_adapter = module.OwnedOptimizer(restored)
        restored_adapter.load_state_dict(saved)
        set_grads(reference_parameters, 7)
        set_grads(candidate_parameters, 7)
        set_grads(restored_parameters, 7)
        reference.step()
        adapter.step()
        restored_adapter.step()
        for actual, expected, resumed in zip(candidate_parameters, reference_parameters, restored_parameters):
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(resumed, expected))

    def test_checkpoint_rejects_param_group_kind_drift_before_mutation(self):
        parameters, optimizer = make_toy(self.initial)
        adapter = OwnedOptimizer(optimizer)
        payload = clone_value(adapter.state_dict())
        expected_kinds = [group["kind"] for group in optimizer.param_groups]
        for group_index, bad_kind in ((2, "sgd"), (2, "adamw"), (0, "rmsprop")):
            with self.subTest(group=group_index, kind=bad_kind):
                restored_parameters, restored = make_toy(self.initial)
                restored_adapter = OwnedOptimizer(restored)
                attacked = clone_value(payload)
                attacked["optimizer"]["param_groups"][group_index]["kind"] = bad_kind
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    restored_adapter.load_state_dict(attacked)
                self.assertEqual([group["kind"] for group in restored.param_groups], expected_kinds)

    def test_checkpoint_group_and_parameter_attacks_reject_atomically(self):
        parameters, optimizer = make_toy(self.initial)
        adapter = OwnedOptimizer(optimizer)
        set_grads(parameters, 0)
        adapter.step()
        payload = clone_value(adapter.state_dict())
        attacks = []
        changed_kind_and_schema = clone_value(payload)
        changed_kind_and_schema["optimizer"]["param_groups"][2]["kind"] = "adamw"
        changed_kind_and_schema["group_schema"][2]["kind"] = "adamw"
        attacks.append(changed_kind_and_schema)
        reordered_refs = clone_value(payload)
        reordered_refs["optimizer"]["param_groups"][0]["params"].reverse()
        attacks.append(reordered_refs)
        reordered_refs_and_schema = clone_value(payload)
        reordered_refs_and_schema["optimizer"]["param_groups"][0]["params"].reverse()
        reordered_refs_and_schema["group_schema"][0]["parameters"].reverse()
        attacks.append(reordered_refs_and_schema)
        changed_parameter_schema = clone_value(payload)
        changed_parameter_schema["group_schema"][2]["parameters"][0]["dtype"] = "torch.float32"
        attacks.append(changed_parameter_schema)
        unknown_state_parameter = clone_value(payload)
        unknown_state_parameter["optimizer"]["state"][999] = {"step": 1}
        attacks.append(unknown_state_parameter)
        for index, attacked in enumerate(attacks):
            with self.subTest(attack=index):
                before_parameters = [parameter.detach().clone() for parameter in parameters]
                before_state = clone_value(optimizer.state)
                before_calls = (adapter.calls, adapter.replicated_rmsprop_calls, DIST.collective_calls)
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    adapter.load_state_dict(attacked)
                for actual, expected in zip(parameters, before_parameters):
                    self.assertTrue(torch.equal(actual, expected))
                assert_tree_equal(self, optimizer.state, before_state)
                self.assertEqual((adapter.calls, adapter.replicated_rmsprop_calls, DIST.collective_calls),
                                 before_calls)

    def test_checkpoint_tensor_schema_fields_reject_before_raw_load(self):
        parameters, optimizer = make_toy(self.initial)
        adapter = OwnedOptimizer(optimizer)
        payload = clone_value(adapter.state_dict())
        attacks = (
            ("shape", [99]),
            ("elements", 99),
            ("dtype", "torch.float64"),
            ("contiguous", False),
            ("requires_grad", False),
            ("device_type", "meta"),
            ("device_index", 0),
        )
        for field, value in attacks:
            with self.subTest(field=field):
                attacked = clone_value(payload)
                attacked["group_schema"][0]["parameters"][0][field] = value
                raw_calls = []
                adapter.raw_load_state_dict = lambda _state: raw_calls.append(True)
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    adapter.load_state_dict(attacked)
                self.assertEqual(raw_calls, [])

    def test_checkpoint_state_schema_attacks_reject_before_raw_load(self):
        parameters, optimizer = make_toy(self.initial)
        adapter = OwnedOptimizer(optimizer)
        set_grads(parameters, 0)
        adapter.step()
        payload = clone_value(adapter.state_dict())
        attacks = []
        missing_parameter_state = clone_value(payload)
        missing_parameter_state["optimizer"]["state"].pop(3)
        attacks.append(missing_parameter_state)
        missing_state_field = clone_value(payload)
        missing_state_field["optimizer"]["state"][0].pop("exp_avg")
        attacks.append(missing_state_field)
        extra_state_field = clone_value(payload)
        extra_state_field["optimizer"]["state"][0]["unexpected"] = torch.ones(1)
        attacks.append(extra_state_field)
        wrong_state_shape = clone_value(payload)
        wrong_state_shape["optimizer"]["state"][0]["exp_avg"] = torch.zeros(99)
        attacks.append(wrong_state_shape)
        wrong_state_dtype = clone_value(payload)
        wrong_state_dtype["optimizer"]["state"][0]["exp_avg"] = wrong_state_dtype["optimizer"]["state"][0]["exp_avg"].double()
        attacks.append(wrong_state_dtype)
        noncontiguous_state = clone_value(payload)
        noncontiguous_state["optimizer"]["state"][2]["momentum_buffer"] = torch.zeros(2, 2).t()
        attacks.append(noncontiguous_state)
        wrong_step_type = clone_value(payload)
        wrong_step_type["optimizer"]["state"][0]["step"] = "one"
        attacks.append(wrong_step_type)
        swapped_state = clone_value(payload)
        swapped_state["optimizer"]["state"][0] = clone_value(swapped_state["optimizer"]["state"][1])
        attacks.append(swapped_state)
        empty_state = clone_value(payload)
        empty_state["optimizer"]["state"] = {}
        attacks.append(empty_state)
        for index, attacked in enumerate(attacks):
            with self.subTest(attack=index):
                restored_parameters, restored = make_toy(self.initial)
                restored_adapter = OwnedOptimizer(restored)
                raw_calls = []
                raw_loader = restored_adapter.raw_load_state_dict

                def recorded_raw_loader(state):
                    raw_calls.append(True)
                    return raw_loader(state)

                restored_adapter.raw_load_state_dict = recorded_raw_loader
                before_parameters = [parameter.detach().clone() for parameter in restored_parameters]
                before_state = clone_value(restored.state)
                before_groups = clone_value(restored.state_dict()["param_groups"])
                before_calls = (restored_adapter.calls, restored_adapter.replicated_rmsprop_calls,
                                DIST.collective_calls)
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    restored_adapter.load_state_dict(attacked)
                self.assertEqual(raw_calls, [])
                assert_tree_equal(self, restored.state, before_state)
                assert_tree_equal(self, restored.state_dict()["param_groups"], before_groups)
                for actual, expected in zip(restored_parameters, before_parameters):
                    self.assertTrue(torch.equal(actual, expected))
                self.assertEqual((restored_adapter.calls,
                                  restored_adapter.replicated_rmsprop_calls,
                                  DIST.collective_calls), before_calls)

    def test_pristine_state_is_an_explicit_distinct_phase(self):
        _parameters, optimizer = make_toy(self.initial)
        adapter = OwnedOptimizer(optimizer)
        payload = clone_value(adapter.state_dict())
        self.assertEqual(payload["state_phase"], "pristine")
        self.assertEqual(payload["optimizer_calls"], 0)
        self.assertEqual(payload["replicated_rmsprop_calls"], 0)
        self.assertEqual(payload["state_manifest"], [])
        self.assertEqual(payload["optimizer"]["state"], {})
        _restored_parameters, restored = make_toy(self.initial)
        restored_adapter = OwnedOptimizer(restored)
        restored_adapter.load_state_dict(payload)
        self.assertEqual(restored.state, {})
        self.assertEqual((restored_adapter.calls, restored_adapter.replicated_rmsprop_calls),
                         (0, 0))
        for field, value in (("state_phase", "initialized"), ("optimizer_calls", 1),
                             ("replicated_rmsprop_calls", 1)):
            with self.subTest(field=field):
                attacked = clone_value(payload)
                attacked[field] = value
                raw_calls = []
                restored_adapter.raw_load_state_dict = lambda _state: raw_calls.append(True)
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    restored_adapter.load_state_dict(attacked)
                self.assertEqual(raw_calls, [])

    def test_initialized_manifest_preserves_legitimate_lazy_parameter_state(self):
        parameters, optimizer = make_toy(self.initial)
        adapter = OwnedOptimizer(optimizer)
        parameters[0].grad = torch.ones_like(parameters[0])
        adapter.step()
        payload = clone_value(adapter.state_dict())
        self.assertEqual(payload["state_phase"], "initialized")
        self.assertEqual(payload["state_manifest"], [0, 2])
        restored_parameters, restored = make_toy(self.initial)
        restored_adapter = OwnedOptimizer(restored)
        restored_adapter.load_state_dict(payload)
        self.assertEqual(set(restored.state_dict()["state"]), {0, 2})
        set_grads(restored_parameters, 1)
        restored_adapter.step()
        self.assertEqual(set(restored.state_dict()["state"]), {0, 1, 2, 3})

    def test_post_plan_kind_drift_rejects_before_updates_or_collective(self):
        parameters, optimizer = make_toy(self.initial)
        adapter = OwnedOptimizer(optimizer)
        set_grads(parameters, 0)
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.param_groups[2]["kind"] = "sgd"
        with self.assertRaisesRegex(ValueError, "kind changed after planning"):
            adapter.step()
        self.assertEqual(adapter.calls, 0)
        for actual, expected in zip(parameters, before):
            self.assertTrue(torch.equal(actual, expected))

    def test_every_live_plan_drift_rejects_before_mutation_or_collective(self):
        def cases(adapter, optimizer):
            replacement = torch.nn.Parameter(optimizer.param_groups[0]["params"][0].detach().clone())
            return [
                lambda: optimizer.param_groups[0]["params"].reverse(),
                lambda: optimizer.param_groups[0]["params"].__setitem__(0, replacement),
                lambda: adapter.groups[0].__setitem__("kind", "muon"),
                lambda: adapter.groups[0]["params"].reverse(),
                lambda: adapter.entries[0].__setitem__("owner", 99),
                lambda: adapter.summary["replicated_rmsprop_groups"].clear(),
                lambda: adapter.parameters.reverse(),
                lambda: adapter.owned.clear(),
            ]

        for case_index in range(8):
            with self.subTest(case=case_index):
                parameters, optimizer = make_toy(self.initial)
                adapter = OwnedOptimizer(optimizer)
                set_grads(parameters, 0)
                adapter.step()
                set_grads(parameters, 1)
                before_parameters = [parameter.detach().clone() for parameter in parameters]
                before_state = clone_value(optimizer.state)
                before_calls = (adapter.calls, adapter.replicated_rmsprop_calls, DIST.collective_calls)
                cases(adapter, optimizer)[case_index]()
                with self.assertRaisesRegex(ValueError, "changed after planning|plan drifted"):
                    adapter.step()
                for actual, expected in zip(parameters, before_parameters):
                    self.assertTrue(torch.equal(actual, expected))
                assert_tree_equal(self, optimizer.state, before_state)
                self.assertEqual((adapter.calls, adapter.replicated_rmsprop_calls, DIST.collective_calls),
                                 before_calls)

    def test_complete_summary_drift_rejects_before_mutation_or_collective(self):
        for field in ("elements_by_owner", "estimated_ms_by_owner", "shard_elements",
                      "total_padded_bytes"):
            with self.subTest(field=field):
                parameters, optimizer = make_toy(self.initial)
                adapter = OwnedOptimizer(optimizer)
                set_grads(parameters, 0)
                if isinstance(adapter.summary[field], list):
                    adapter.summary[field][0] += 1
                else:
                    adapter.summary[field] += 1
                before_parameters = [parameter.detach().clone() for parameter in parameters]
                before_state = clone_value(optimizer.state)
                before_calls = (adapter.calls, adapter.replicated_rmsprop_calls,
                                DIST.collective_calls)
                with self.assertRaisesRegex(ValueError, "plan drifted"):
                    adapter.step()
                for actual, expected in zip(parameters, before_parameters):
                    self.assertTrue(torch.equal(actual, expected))
                assert_tree_equal(self, optimizer.state, before_state)
                self.assertEqual((adapter.calls, adapter.replicated_rmsprop_calls,
                                  DIST.collective_calls), before_calls)


class FakeWorldFour(unittest.TestCase):
    def test_replicated_rmsprop_params_and_state_match_on_all_ranks(self):
        DIST.world_size, DIST.collective_calls = 4, 0
        initial_adam = torch.tensor([1.0, -2.0, 3.0, -4.0])
        initial_rms = torch.linspace(-0.5, 0.5, 16).reshape(4, 4).to(torch.bfloat16)
        instances = []
        for rank in range(4):
            DIST.rank = rank
            adam = torch.nn.Parameter(initial_adam.clone())
            rms = torch.nn.Parameter(initial_rms.clone())
            groups = [
                {"kind": "adamw", "params": [adam], "lr": 0.0, "betas": (0.8, 0.96),
                 "eps": 1e-8, "weight_decay": 0.0},
                {"kind": "rmsprop", "params": [rms], "lr": 0.03, "betas": (0.0, 0.95),
                 "eps": 1e-8, "weight_decay": 0.0},
            ]
            optimizer = ToyOptimizer(groups)
            instances.append((adam, rms, optimizer, OwnedOptimizer(optimizer)))
        stride = instances[0][3].summary["shard_elements"]
        DIST.gathered = torch.zeros(stride * 4)
        owner_entry = instances[0][3].entries[0]
        DIST.gathered.narrow(0, owner_entry["owner"] * stride + owner_entry["offset"],
                             owner_entry["elements"]).copy_(initial_adam)
        for rank, (adam, rms, optimizer, adapter) in enumerate(instances):
            DIST.rank = rank
            adam.grad = torch.ones_like(adam)
            rms.grad = torch.linspace(0.01, 0.16, 16).reshape_as(rms).to(torch.bfloat16)
            adapter.step()
            self.assertEqual(adapter.replicated_rmsprop_calls, 1)
            self.assertNotIn(1, {entry["group"] for entry in adapter.entries})
        digests = []
        for _adam, rms, optimizer, _adapter in instances:
            state = optimizer.state[rms]["exp_avg_sq"]
            digests.append(hashlib.sha256(tensor_bytes(rms) + tensor_bytes(state)).hexdigest())
        self.assertEqual(len(set(digests)), 1)
        snapshots = [(rms.detach().clone(), clone_value(optimizer.state[rms]))
                     for _adam, rms, optimizer, _adapter in instances]
        for rank, (_adam, rms, optimizer, adapter) in enumerate(instances):
            DIST.rank = rank
            rms.grad = torch.ones_like(rms)
            adapter.step(update_adamw=False)
            self.assertEqual(adapter.replicated_rmsprop_calls, 1)
            self.assertTrue(torch.equal(rms, snapshots[rank][0]))
            assert_tree_equal(self, optimizer.state[rms], snapshots[rank][1])


class StaticContract(unittest.TestCase):
    def test_source_prepare_scope_and_proposal(self):
        import subprocess

        old = subprocess.check_output(["git", "show", f"{PARENT}:submissions/dense6x1024_public0992851/train.py"], cwd=ROOT)
        self.assertEqual(hashlib.sha256(old).hexdigest(), OLD_SHA)
        self.assertEqual(hashlib.sha256((ROOT / "prepare.py").read_bytes()).hexdigest(), PREPARE_SHA)
        source = CANDIDATE.read_text()
        parent_source = old.decode()
        self.assertEqual(source[:source.index("def make_plan(groups, world_size):")],
                         parent_source[:parent_source.index("def make_plan(groups, world_size):")])
        self.assertEqual(source[source.index("def _submission_optimizer_install"):],
                         parent_source[parent_source.index("def _submission_optimizer_install"):])
        proposal = json.loads((HERE / "EXPERIMENT_PROPOSAL.json").read_text())
        self.assertEqual(proposal["immutable_base"]["sha256_after_implementation"],
                         hashlib.sha256(CANDIDATE.read_bytes()).hexdigest())
        self.assertEqual(proposal["a1_science"]["launch_command"],
                         "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py " + A1_FLAGS)
        self.assertFalse(proposal["gates"]["native_5_step_authorized"])
        self.assertFalse(proposal["gates"]["full_a1_authorized"])
        self.assertFalse(proposal["lifecycle_contract"]["requests_created"])
        self.assertEqual(list(HERE.glob("*authorization*.json")), [])
        self.assertEqual(list(HERE.glob("*request*.json")), [])
        old_request = json.loads((ROOT / "research/submission-validation/20260924-v2-track-a-a1-r2/a1-trn21-r2-request.json").read_text())
        self.assertEqual(old_request["source_sha256"]["submissions/dense6x1024_public0992851/train.py"], OLD_SHA)
        self.assertNotEqual(old_request["source_sha256"]["submissions/dense6x1024_public0992851/train.py"],
                            hashlib.sha256(CANDIDATE.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
