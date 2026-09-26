#!/usr/bin/env python3
"""Independent, stdlib-only composition check; reads only the four review inputs.

Executes the exact added schedule branches and changed routing expressions from
source.diff, against separately written scalar references. Does not import the
candidate, author test, Torch, a compiler, SDK, data, or any host interface.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LANE = ROOT / "research/v2/lanes/sprint-batch-lr021-recursive-stack"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=3e-16),
            f"{label}: {actual!r} != {expected!r}")
    return abs(actual - expected)


def lr_reference(step, elapsed, budget, kind, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = max(0.0, min(1.0, elapsed / max(budget, 1e-6)))
    width = 0.95 if kind == "muon" else 0.65
    if p < 1.0 - width:
        return 1.0
    # Independent affine interpolation, with the same binary boundary.
    return 0.05 + 0.95 * (1.0 - p) / width


def momentum_reference(step, progress):
    p = max(0.0, min(1.0, progress))
    if p <= 1.0 - 0.95:
        return 0.85 + 0.10 * min(step / 300, 1.0)
    q = (p - (1.0 - 0.95)) / 0.95
    return 0.95 - 0.16 * q * q


def beta_reference(progress):
    p = max(0.0, min(1.0, progress))
    if p <= 1.0 - 0.95:
        return 0.95
    return 0.95 + 0.02 * ((p - (1.0 - 0.95)) / 0.95)


def main():
    names = ["source.diff", "SPEC.md", "CPU_RECEIPT.json", "MANIFEST.json"]
    data = {name: (LANE / name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}
    manifest = json.loads(data["MANIFEST.json"])
    receipt = json.loads(data["CPU_RECEIPT.json"])
    for name in names[:-1]:
        require(hashes[name] == manifest["files"][str((LANE / name).relative_to(ROOT))],
                f"manifest mismatch: {name}")
    diff = data["source.diff"].decode()
    hunks = diff.split("\n@@ ")[1:]
    require(len(hunks) == 6, "unexpected diff hunk count")
    added = lambda h: "\n".join(line[1:] for line in h.splitlines() if line.startswith("+"))
    lr_text = added(next(h for h in hunks if h.startswith("-1307,")))
    momentum_beta = added(next(h for h in hunks if h.startswith("-1321,")))
    momentum_text, beta_text = momentum_beta.split("\ndef muon_beta2", 1)
    beta_text = "def muon_beta2" + beta_text.split("\n    global RECURSIVE_SCHEDULE")[0]
    env = {"RECURSIVE_SCHEDULE": True, "WARMUP_STEPS": 20, "MUON_BETA2": 0.9}
    exec(lr_text + "\n    raise RuntimeError('OFF suffix not supplied by diff')\n", env)
    exec("def muon_momentum(step, progress=0.0):\n" + momentum_text + "\n", env)
    exec(beta_text + "\n", env)
    boundary = 1.0 - 0.95
    adam_boundary = 1.0 - 0.65
    points = [-1.0, 0.0, 0.05, boundary, math.nextafter(boundary, -math.inf),
              math.nextafter(boundary, math.inf), 0.25, adam_boundary,
              math.nextafter(adam_boundary, -math.inf),
              math.nextafter(adam_boundary, math.inf), 0.5, 0.955, 1.0, 2.0]
    schedule_cases = 0
    max_error = 0.0
    for warmup in (0, 1, 20):
        env["WARMUP_STEPS"] = warmup
        for step in (0, 1, 19, 20, 299, 300, 301, 10000):
            for p in points:
                for kind in ("muon", "adamw", "other"):
                    for budget in (1800.0, 0.0, -1.0):
                        elapsed = p * max(1e-6, budget)
                        a = env["lr_multiplier"](step, elapsed, budget, kind)
                        e = lr_reference(step, elapsed, budget, kind, warmup)
                        max_error = max(max_error, close(a, e, "LR"))
                        schedule_cases += 1
                max_error = max(max_error, close(env["muon_momentum"](step, p),
                                                momentum_reference(step, p), "momentum"))
                max_error = max(max_error, close(env["muon_beta2"](p), beta_reference(p), "beta2"))
                schedule_cases += 2
    # Endpoints have directly interpretable scalar values.
    close(env["lr_multiplier"](300, 1800, 1800, "muon"), 0.05, "Muon LR endpoint")
    close(env["lr_multiplier"](300, 1800, 1800, "adamw"), 0.05, "AdamW LR endpoint")
    close(env["muon_momentum"](300, 1.0), 0.79, "momentum endpoint")
    close(env["muon_beta2"](1.0), 0.97, "beta endpoint")
    routing_line = next(line[1:].strip() for line in diff.splitlines()
                        if line.startswith("+            group['lr'] ="))
    transport_line = next(line[1:].strip() for line in diff.splitlines()
                          if line.startswith("+            one_beta2 ="))
    route_cases = 0
    # Use actual receipt initial LRs, supplemented with the other-kind branch.
    groups = [g for check in receipt["checks"] for g in check["result"]["optimizer_groups"]]
    groups.append({"kind": "other", "initial_lr": 0.123})
    for enabled in (False, True):
        for group_record in groups:
            for flat in (False, True):
                for lrm, adam, flat_value in ((1.0, 1.0, 0.05), (0.5, 0.75, 1.0),
                                             (0.05, 0.05, 1.0), (0.123, 0.987, 0.456)):
                    group = dict(group_record, no_lr_decay=flat)
                    actual_adam = adam if enabled else lrm
                    local = dict(group=group, lrm=lrm, lrm_adam=actual_adam, lrm_flat=flat_value)
                    exec(routing_line, {}, local)
                    multiplier = lrm
                    if enabled and group["kind"] != "muon":
                        multiplier = adam
                    if flat:
                        multiplier = flat_value
                    expected = group["initial_lr"] * multiplier
                    require(group["lr"].hex() == expected.hex(), "routing bit identity failure")
                    route_cases += 1
        for beta in (0.0, 0.9, 0.95, 0.97, 1.0):
            tensor_marker = object()
            local = dict(RECURSIVE_SCHEDULE=enabled, group={"beta2": beta}, one_b2_t=tensor_marker)
            exec(transport_line, {}, local)
            require(local["one_beta2"] is tensor_marker if enabled else
                    local["one_beta2"].hex() == (1 - beta).hex(), "beta2 transport failure")
    require([(x["check"], x["result"]["total_batch_argument"], x["result"]["matrix_lr"],
              len(x["result"]["steps"])) for x in receipt["checks"]] ==
            [("parent_vs_candidate_OFF3", 131072, 0.015, 3),
             ("parent_vs_candidate_OFF3", 262144, 0.021, 3),
             ("donor_vs_candidate_ON3", 262144, 0.021, 3)], "receipt scope differs")
    print(json.dumps({"verdict": "PASS", "input_hashes": hashes,
                      "source_sha256_declared_not_opened": manifest["source_sha256"],
                      "schedule_scalar_comparisons": schedule_cases,
                      "schedule_abs_tolerance": 3e-16, "schedule_max_absolute_error": max_error,
                      "endpoint_comparisons": 4, "routing_bit_exact_comparisons": route_cases,
                      "beta2_transport_comparisons": 10, "full_source_read": False,
                      "torch_or_native_execution": False}, indent=2))


if __name__ == "__main__":
    main()
