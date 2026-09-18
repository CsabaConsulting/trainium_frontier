#!/usr/bin/env python3
"""Preserve eager NEFFs named by a native trace, alongside compiled exports.

The in-process Dynamo export does not cover warmed eager operations. A device
trace can require their binaries even when profiling did not copy them. Resolve
the trace's persistent keys against the installed native cache before another
job changes that cache. Record compiled logical-key aliases separately: their
trace persistent key can differ from the key in the compiled export manifest.
Only successful Explorer ingestion establishes complete UUID coverage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def preserve_trace_neffs(trace_root: Path, cache_root: Path, output: Path) -> dict:
    trace_root, cache_root, output = (p.resolve() for p in (trace_root, cache_root, output))
    output.mkdir(parents=True, exist_ok=True)
    compiled = {}
    compiled_manifest = output / "manifest.json"
    if not compiled_manifest.exists():
        compiled_manifest = trace_root / "neffs" / "manifest.json"
    if compiled_manifest.exists():
        for record in json.loads(compiled_manifest.read_text())["exported"]:
            compiled[record["cache_key"]] = record
    references: dict[str, set[str]] = {}
    traces = sorted(trace_root.rglob("trace.json"))
    if not traces:
        raise ValueError(f"No trace.json under {trace_root}")
    for trace in traces:
        for event in json.loads(trace.read_text())["traceEvents"]:
            args = event.get("args", {})
            key = args.get("persistent_cache_key")
            if key:
                if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}", key):
                    raise ValueError(f"Unexpected persistent cache key: {key!r}")
                references.setdefault(key, set()).add(args.get("neff_cache_key", ""))

    files, aliases, missing = [], [], []
    for key, logical_keys in sorted(references.items()):
        source = cache_root / key[:2] / key[2:4] / (key + ".neff")
        if source.is_file():
            data = source.read_bytes()
            destination = output / source.name
            if destination.exists():
                if destination.read_bytes() != data:
                    raise ValueError(f"Refusing to replace distinct NEFF: {destination}")
            else:
                with destination.open("xb") as stream:
                    stream.write(data)
            files.append(dict(persistent_cache_key=key, logical_keys=sorted(logical_keys),
                              source=str(source), file=destination.name, bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest()))
        else:
            available = [compiled[k] for k in sorted(logical_keys) if k in compiled]
            record = dict(trace_persistent_key=key, logical_keys=sorted(logical_keys))
            if available:
                # This is a provenance link, not proof that profiling executed
                # the same binary. Explorer must match the NTFF's actual UUIDs.
                aliases.append(dict(**record, existing_compiled_exports=available))
            else:
                missing.append(record)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    trace_root=str(trace_root), cache_root=str(cache_root),
                    source_traces=[str(p) for p in traces], files=files,
                    compiled_logical_key_aliases=aliases, unresolved=missing,
                    uuid_coverage_verified_by_this_helper=False,
                    next_gate="Explorer ingestion must confirm every required NEFF UUID")
    (output / "trace-neffs-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/neff_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preserve_trace_neffs(args.trace_root, args.cache_root, args.output)
    print(json.dumps(result))
    if result["unresolved"]:
        raise SystemExit("Some trace keys have neither a cached binary nor a compiled export")


if __name__ == "__main__":
    main()
