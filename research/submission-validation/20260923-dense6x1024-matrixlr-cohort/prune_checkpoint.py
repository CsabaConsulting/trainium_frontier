#!/usr/bin/env python3
"""Operator-only exact checkpoint prune after retrieved evidence is committed and pushed."""
import argparse, hashlib, json, os, pathlib

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--expected-bytes", type=int, required=True)
    ap.add_argument("--evidence-pushed-receipt", type=pathlib.Path, required=True)
    ap.add_argument("--receipt", type=pathlib.Path, required=True)
    args = ap.parse_args()
    pushed = json.loads(args.evidence_pushed_receipt.read_text())
    if pushed.get("retrieved_verified_committed_pushed") is not True:
        raise SystemExit("evidence push is not proven")
    checkpoint = args.checkpoint.resolve()
    if checkpoint.name != "final.pt" or checkpoint.parent.name != "out":
        raise SystemExit("checkpoint path is not exact sandbox out/final.pt")
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise SystemExit("checkpoint is missing/non-regular/symlink")
    if checkpoint.stat().st_size != args.expected_bytes or sha(checkpoint) != args.expected_sha256:
        raise SystemExit("checkpoint identity mismatch")
    if args.receipt.exists():
        raise SystemExit("prune receipt already exists")
    checkpoint.unlink()
    if checkpoint.exists():
        raise SystemExit("checkpoint deletion failed")
    receipt = {"exact_path": str(checkpoint), "sha256": args.expected_sha256,
               "bytes": args.expected_bytes, "deleted": True,
               "evidence_pushed_receipt": str(args.evidence_pushed_receipt.resolve()),
               "other_deletions": []}
    fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")

if __name__ == "__main__":
    main()
