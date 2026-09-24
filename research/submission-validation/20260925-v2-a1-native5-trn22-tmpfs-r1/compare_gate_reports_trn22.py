#!/usr/bin/env python3
"""Exact trn22 identity binding for the reviewed native-five comparator.

The comparator implementation and ownership arithmetic remain in the reviewed
module.  This adapter changes only the exact host and three fresh arm labels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "20260924-v2-a1-hybrid-native5-equivalence-gate" / "compare_gate_reports.py"
LABELS = {
    "G0": "fs-20260925-v2-a1-native5-g0-hybrid-observation-off-trn22-r1",
    "G1": "fs-20260925-v2-a1-native5-g1-hybrid-observation-on-trn22-r1",
    "G2": "fs-20260925-v2-a1-native5-g2-ordinary-observation-on-trn22-r1",
}


def _load_base():
    spec = importlib.util.spec_from_file_location("a1_native_five_exact_comparator", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load exact native-five comparator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPECTED_HOST = "trn22"
    module.LABELS = dict(LABELS)
    return module


BASE_COMPARATOR = _load_base()
compare = BASE_COMPARATOR.compare


def main() -> int:
    return BASE_COMPARATOR.main()


if __name__ == "__main__":
    raise SystemExit(main())
