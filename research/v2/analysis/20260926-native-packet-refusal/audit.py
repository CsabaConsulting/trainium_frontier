"""Authenticate an independent refusal archive, not a native qualification."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "research/v2/verification/20260926-clean-native-host-packet-fresh"
ORIGINAL = Path("/home/poojith-devan/clean36-host-review-OqqNW1")
EXPECTED_MANIFEST = "d316d2ea1296df6381641591651ccd4f275fa27a21123ade55ad303a6576add0"
EXPECTED_REPORT = "3363b898a0042d0634d1132ad31471e985c8dd3055ac4a331354c1ca08b0259c"
OMITTED = {
    f"{mode}/{relative}": size
    for mode in ("initial-normal", "normal", "optimized")
    for relative, size in (
        ("artifact/oversize.log", 268435457),
        ("hidden/receipts/.rank0.interrupted.tmp", 16777217),
    )
}
FIXTURE_LINKS = {
    f"{mode}/{name}": str(ORIGINAL / ("normal" if mode == "initial-normal" else mode) / target)
    for mode in ("initial-normal", "normal", "optimized")
    for name, target in (("sizes/bad", "sizes/.hidden"), ("inputlink", "input"))
}
count = 0


def check(value, message):
    global count
    if not value:
        raise RuntimeError(message)
    count += 1


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


check(digest(ARCHIVE / "SHA256SUMS") == EXPECTED_MANIFEST, "manifest pin")
check(digest(ORIGINAL / "SHA256SUMS") == EXPECTED_MANIFEST, "original manifest pin")
check(digest(ARCHIVE / "REVIEW.md") == EXPECTED_REPORT, "REFUSE report pin")
entries = {}
for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines():
    sha, relative = line.split("  ", 1)
    p = Path(relative)
    check(not p.is_absolute() and ".." not in p.parts, "safe manifest path")
    name = p.as_posix()
    check(name not in entries, "no duplicate manifest path")
    entries[name] = sha
    original = ORIGINAL / p
    check(stat.S_ISREG(original.lstat().st_mode), "regular original: " + name)
    check(digest(original) == sha, "original bytes: " + name)
    if name in OMITTED:
        check(original.stat().st_size == OMITTED[name], "synthetic sparse fixture size")
        check(not (ARCHIVE / p).exists(), "explicitly omitted synthetic fixture")
    else:
        target = ARCHIVE / p
        check(stat.S_ISREG(target.lstat().st_mode), "regular archived evidence: " + name)
        check(digest(target) == sha, "archived exact bytes: " + name)
check(set(OMITTED) <= set(entries), "all six omitted fixtures inventoried")
actual = set()
actual_links = set()
for p in ARCHIVE.rglob("*"):
    mode = p.lstat().st_mode
    name = p.relative_to(ARCHIVE).as_posix()
    if stat.S_ISLNK(mode):
        check(name in FIXTURE_LINKS, "only exact negative-test fixture links")
        check(os.readlink(p) == os.readlink(ORIGINAL / name) == FIXTURE_LINKS[name], "literal fixture link target")
        actual_links.add(name)
        continue
    check(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "no unexpected special archive members")
    if stat.S_ISREG(mode):
        actual.add(p.relative_to(ARCHIVE).as_posix())
check(actual == (set(entries) - set(OMITTED)) | {"SHA256SUMS"}, "complete selected archive, no extras")
check(actual_links == set(FIXTURE_LINKS), "all six synthetic rejection-test links preserved without following")
normal = json.loads((ARCHIVE / "normal/summary.json").read_text())
optimized = json.loads((ARCHIVE / "optimized/summary.json").read_text())
check(normal["mode"] == "normal" and optimized["mode"] == "optimized", "both Python modes")
check(normal["checks"] == optimized["checks"] and len(normal["checks"]) == 15, "15 grouped checks both modes")
faults = []
for mode in ("normal", "optimized"):
    fault = json.loads((ARCHIVE / f"scan-{mode}.log").read_text())
    check(fault["reproduced_defect"] is True and fault["actual_hidden_bytes"] == 17, "real independent fault fixture")
    check(fault["observation"]["artifact_bytes"] == 0, "failed scan falsely became zero")
    faults.append(fault)
check(faults[0] == faults[1], "same independent failure under optimized Python")
print(json.dumps({
    "verdict": "AUTHENTICATED_REFUSE_NEW_DISABLED_NATIVE_PACKET_ONLY",
    "checks_passed": count,
    "optimized": bool(sys.flags.optimize),
    "original_manifest_files": len(entries),
    "archived_files_including_manifest": len(actual),
    "preserved_literal_synthetic_fixture_symlinks": FIXTURE_LINKS,
    "omitted_reconstructible_synthetic_sparse_fixtures": OMITTED,
    "omitted_fixture_apparent_bytes": sum(OMITTED.values()),
    "report_sha256": EXPECTED_REPORT,
    "native_or_host_actions": False,
    "running_fullcold_affected": False,
    "launch_authorized": False,
}, sort_keys=True))
