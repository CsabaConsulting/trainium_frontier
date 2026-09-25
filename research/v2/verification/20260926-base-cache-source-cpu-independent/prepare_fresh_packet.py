#!/usr/bin/env python3
"""Mechanically package ONLY full-context diff, neutral spec and raw receipts.

Creates a new external packet, never launches a reviewer. No notes, tests,
source repository, previous session or implementation prose are included.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

BASE = "a4aa4351789115a2d0aa947d9ca00ff689f9bd0b"
CANDIDATE = "b635c8b0cf2a10e1bcdd983e86192672ed6ad56f"
SOURCE = "workflow/proposals/research-v2/train.py"
SPEC = """# Base initial-cache refresh: bounded independent source review

Start at REFUSE. Review only the supplied full-context source diff, this lane
specification, and raw CPU receipts. Do not consult earlier implementation or
review notes. The requested decision concerns suitability for a later exact
full-size CPU qualification, not native execution or scientific adoption.

The proposed causal factor is a Boolean CLI option that defaults OFF. When ON,
it requests differentiable casts of the existing Linear weights immediately
before the first training forward. Preserve FP32 master parameters, the assigned
BF16 compute path, gradients and optimizer semantics; preserve the default/OFF
execution and saved-model/evaluation behavior. Do not change training data,
microshape, LR/schedule, total batch, model architecture or budget accounting.
All preparation inside the timed training loop must remain charged by the
existing clock. The normal later cache-refresh behavior must be preserved.

Base source SHA256: 6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3
Candidate source SHA256: bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522

The historical receipts are reduced CPU observations: four ranks, three steps,
width 128, depth 6, physical synthetic inputs 2x128 per rank, synthetic prepare,
no compilation, and a deterministic fake clock. They are not full117M G2, real
data, native correctness, compile-time, throughput, cold-start or budget proof.
Full117M standard G2 has not run for this exact source. Raw observer fields can
be used as evidence only within their documented/structural meaning; do not
infer execution counts merely from field names.

Independently reconstruct both sources from the diff and authenticate their
hashes. Write and actually run your own small single-thread CPU reference for
default/OFF identity and ON BF16-cache/FP32-master forward and backward/update
equivalence. Check repeated steps, cache connectivity, and state/load behavior
where relevant. Do not import the full training module or allocate a full model;
use extracted definitions and very small weights. Standard CPU libraries are
available. Do not access other repositories, branches, notes, hosts, network,
credentials, or hardware; do not modify candidate source. Do not launch training.

Return REFUSE with concrete counterexample if evidence is insufficient or any
required behavior differs. Otherwise state the precise limited CPU-qualification
scope supported, and keep full-size, fresh-session policy and native gates
distinct. Preserve your own source/test/results and actual commands. No prior
review verdict is supplied in this packet.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=pathlib.Path, required=True)
    p.add_argument("--destination", type=pathlib.Path, required=True)
    args = p.parse_args()
    target = args.destination.resolve()
    if target.exists() or target.parent != pathlib.Path("/tmp") or not target.name.startswith("trainium-fresh-base-cache-packet-"):
        raise RuntimeError("require new non-repository /tmp/trainium-fresh-base-cache-packet-* target")
    target.mkdir(mode=0o700)
    (target / "raw").mkdir()
    diff = subprocess.check_output(["git", "diff", "--unified=100000", BASE, CANDIDATE, "--", SOURCE], cwd=args.repo)
    (target / "candidate.diff").write_bytes(diff)
    (target / "lane-spec.md").write_text(SPEC)
    for arm in ("base", "default", "off", "on"):
        shutil.copyfile(args.repo / "research/v2/lanes/initial-cache-refresh" / f"tiny-{arm}.json",
                        target / "raw" / f"tiny-{arm}.json")
    files = []
    for f in sorted(target.rglob("*")):
        if f.is_file():
            data = f.read_bytes()
            files.append({"path": str(f.relative_to(target)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema": "input-only-review-packet-v1", "base_commit": BASE, "candidate_commit": CANDIDATE,
                "included": files, "excluded": ["implementation notes", "implementation tests", "prior reviews",
                    "current verifier tests/results", "repository metadata", "session history", "credentials"],
                "reviewer_session_started": False, "native_authority": False}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (target / "INPUT_MANIFEST.json").write_bytes(encoded)
    print(json.dumps({"packet_path": str(target), "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                      "files": len(files), "total_payload_bytes": sum(f["bytes"] for f in files),
                      "reviewer_session_started": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
