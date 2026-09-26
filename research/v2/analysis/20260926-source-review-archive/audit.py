"""Authenticate retained independent CPU evidence; never grants host authority."""
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[4]
CHECKS = []

def check(value, label):
    if not value:
        raise RuntimeError(label)
    CHECKS.append(label)

def digest(path):
    check(stat.S_ISREG(path.lstat().st_mode), "regular:" + str(path.relative_to(ROOT)))
    return hashlib.sha256(path.read_bytes()).hexdigest()

clean = ROOT / "research/v2/verification/20260926-clean-native-repair-fresh"
pins = {
    "REPORT.md": "03a6321019e2214ff0d84f10f314ca6419d51b793b13f065cc115c8a4fc26b47",
    "reviewer_tests.py": "dad0a797c19403e5275167325232267beae98b7112813fed9d5ae7081bb04aa7",
    "normal.log": "d2c3a0eebc5543c45f541d717c4ef33c0999fb339442db5b5d6e340a1bae660e",
    "optimized.log": "431460d3b13261382ce3121e7978fc732cf9fb04d13cd116969c2a2c212aa588",
}
check({p.name for p in clean.iterdir()} == set(pins), "clean_exact_files")
for name, expected in pins.items():
    check(digest(clean / name) == expected, "clean_hash:" + name)
for name in ("normal.log", "optimized.log"):
    body = (clean / name).read_text()
    check("Ran 14 tests" in body and body.rstrip().endswith("OK"), "clean_14:" + name)

k2 = ROOT / "research/v2/verification/20260926-k2-bound-worker-fresh"
check(digest(k2 / "SHA256SUMS") == "4acb38ab3b2bd548024065b62be2217abcaef30deab718dfa64f35b506e16f00", "k2_manifest")
expected_files = {"SHA256SUMS"}
for line in (k2 / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    relative = Path(name)
    check(not relative.is_absolute() and ".." not in relative.parts and name not in expected_files, "safe_unique:" + name)
    expected_files.add(name)
    check(digest(k2 / relative) == expected, "k2_hash:" + name)
actual_files = set()
for p in k2.rglob("*"):
    mode = p.lstat().st_mode
    check(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "no_special:" + str(p.relative_to(k2)))
    if stat.S_ISREG(mode):
        actual_files.add(str(p.relative_to(k2)))
check(actual_files == expected_files, "k2_exact_archive")
summaries = []
for name in ("normal-reviewed.json", "optimized-reviewed.json"):
    result = json.loads((k2 / name).read_text())
    check(result["scalar"]["recurrences"] == 1144, name + ":recurrences")
    check(result["scalar"]["additional_finite_pairs"] == 1027, name + ":pairs")
    check(len(result["routes"]) == 13, name + ":route_cases")
    routes = {}
    for route in result["routes"]:
        label = route["mode"]
        check(label not in routes, name + ":unique:" + label)
        ordinals = list(range(7)) if label == "reference_native_seam" else list(range(4)) if label in ("stale_f0", "signed_zero") else []
        check(route["accepted_ordinals"] == ordinals, name + ":ordinals:" + label)
        check(bool(route["errors"]) == (label != "reference_native_seam"), name + ":errors:" + label)
        check("No admission/native result claimed" in route["caveat"], name + ":scope:" + label)
        routes[label] = {"accepted_ordinals": ordinals, "errors": route["errors"]}
    summaries.append(routes)
check(summaries[0] == summaries[1], "normal_optimized_route_equivalence")
print(json.dumps({"verdict": "AUTHENTICATED_SOURCE_CPU_REVIEW_ARCHIVES_ONLY", "checks_passed": len(CHECKS), "clean_files": 4, "k2_manifest_members": 1683, "k2_route_cases_per_mode": 13, "clean_tests_per_mode": 14, "clean_receipt_defects_closed": True, "k2_native_execution_proven": False, "host_release": False, "submission_ready": False}, sort_keys=True))
