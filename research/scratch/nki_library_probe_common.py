"""Provenance and independent numerical checks for the pinned library screens."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "research/vendor/nki_library_92d11f63"
NAMESPACE = "research.vendor.nki_library_92d11f63"


def emit(marker, value):
    print(marker + " " + json.dumps(value), flush=True)


def verify_sources():
    manifest = json.loads((VENDOR / "manifest.json").read_text())
    for name, row in manifest["files"].items():
        path = VENDOR / name
        assert path.stat().st_size == row["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], name
    assert hashlib.sha256((VENDOR / "__init__.py").read_bytes()).hexdigest() == manifest["generated_root_sha256"]
    bundled = importlib.import_module("nkilib")
    bundled_root = Path(bundled.__file__).parent
    assert "site-packages/nkilib" in str(bundled_root), bundled_root
    selected = manifest["installed_comparison"]["selected"]
    for name, row in selected.items():
        assert hashlib.sha256((bundled_root / name).read_bytes()).hexdigest() == row["installed_sha256"], name
    emit("NKILIB_PROVENANCE", dict(
        passed=True, upstream_commit=manifest["commit"], vendored_files=len(manifest["files"]),
        bundled_root=str(bundled_root), vendor_namespace=NAMESPACE,
        nki=importlib.metadata.version("nki"), compiler=importlib.metadata.version("neuronx-cc"),
        package_installation=False, bundled_package_replaced=False))


def exact(name, actual, expected):
    if isinstance(actual, torch.Tensor):
        actual, expected = (actual,), (expected,)
    passed = len(actual) == len(expected) and all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
        for a, b in zip(actual, expected))
    emit("NKILIB_EXACT", dict(name=name, passed=passed))
    assert passed, name


def bounded(name, actual, expected, bound, relative_limit=None):
    actual = actual.detach().cpu().double()
    expected, bound = expected.double(), bound.double()
    error = (actual - expected).abs()
    denominator = expected.norm().clamp_min(1e-30)
    relative = float(error.norm() / denominator)
    finite = bool(torch.isfinite(actual).all())
    failures = int((error > bound).sum())
    negative_failures = int(((expected * .5 - expected).abs() > bound).sum())
    meaningful = bool(expected.abs().max() > 1e-20)
    passed = finite and not failures and (relative_limit is None or relative < relative_limit)
    if meaningful:
        passed &= negative_failures > 0
    row = dict(name=name, shape=list(actual.shape), finite=finite, bound_failures=failures,
               relative_rms=relative, max_abs=float(error.max()),
               max_error_over_bound=float((error / bound.clamp_min(1e-300)).max()),
               half_output_negative_control_failures=negative_failures, passed=passed)
    emit("NKILIB_NUMERICAL", row)
    assert passed, row


def attention_cpu_check(name, actual, expected):
    # Predeclared BF16 attention screen: ~2 BF16 roundings of the reference RMS,
    # plus a relative term for large coordinates. This is not an equivalence
    # relaxation of any existing gate; old/new comparison additionally uses the
    # unchanged campaign compare() with its stricter 0.004 relative-RMS limit.
    expected = expected.double()
    rms = expected.square().mean().sqrt()
    bound = .03 * expected.abs() + .008 * rms + 1e-30
    bounded(name, actual, expected, bound, relative_limit=.01)
