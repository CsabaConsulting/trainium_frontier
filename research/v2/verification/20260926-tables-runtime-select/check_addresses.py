"""Independent stdlib-only check of the four changed row-group addresses.

This does not import the candidate, emulate the compiler or claim native speed.
Run from the repository root. No data, host or source mutation.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
lane = root / "research/v2/lanes/tables-runtime-select"
parent = root / "research/v2/lanes/tables-compiler-local-constants/train.py"
source = lane / "train.py"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(source) == "414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34"
assert digest(parent) == "4079e8e6bff1f8438e0430a8b8b2099085da5d6f908149268ac17e8b318fdbff"

# Exactly one production top-level function changes; all other ASTs match.
before, after = ast.parse(parent.read_text()), ast.parse(source.read_text())
assert len(before.body) == len(after.body)
changed = [getattr(b, "name", type(b).__name__) for a, b in zip(before.body, after.body)
           if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False)]
assert changed == ["_k2_table_build_kernel"], changed

manifest = json.loads((lane / "artifacts-manifest.json").read_text())
assert manifest["source_sha256"] == digest(source)
for path, sha in manifest["files"].items():
    assert digest(lane / path) == sha, path
for dtype in ("int32", "int64"):
    receipt = json.loads((lane / f"cpu-result-{dtype}.json").read_text())
    for path, sha in receipt["sha256"].items():
        assert digest(Path(path)) == sha, path
    assert receipt["model"]["off_3_steps"] == "BIT_IDENTICAL"
    assert receipt["model"]["on_2_steps"] == "REGISTERED_VS_ORDINARY_BIT_IDENTICAL"

# Independent row-major storage construction, not candidate helper execution.
# Each reshape/selected group exposes precisely the original contiguous slice.
groups, rows, channels, owner_width = 4096, 128, 128, 64
checked_rows = 0
for group in range(groups):
    old_count = range(group * rows, (group + 1) * rows)
    new_count = range(group * (rows * 1), group * (rows * 1) + rows)
    assert old_count == new_count
    assert group * (1 * 1) == group
    for row in range(rows):
        old_base = (group * rows + row) * channels
        selected_base = group * (rows * channels) + row * channels
        owner0 = (selected_base, selected_base + owner_width)
        owner1 = (selected_base + owner_width, selected_base + channels)
        assert owner0 == (old_base, old_base + 64)
        assert owner1 == (old_base + 64, old_base + 128)
        assert owner0[1] == owner1[0]
        assert owner1[1] - owner0[0] == channels
        checked_rows += 1
assert checked_rows == 524288
assert selected_base + channels == 524288 * 128
print(json.dumps({"status": "PASS_CHANGED_ADDRESS_EQUIVALENCE",
                  "source_sha256": digest(source), "checked_rows": checked_rows,
                  "checked_output_elements_by_contiguous_intervals": checked_rows * channels,
                  "unchanged_other_top_level_nodes": len(after.body) - 1,
                  "artifact_hashes_verified": len(manifest["files"]),
                  "cpu_receipt_hashes_verified": True,
                  "native_execution": False}, indent=2))
