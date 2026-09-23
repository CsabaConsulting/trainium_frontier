#!/usr/bin/env python3
"""Authenticated exact prune after evidence retrieval, commit and remote push."""
import argparse, hashlib, json, os, pathlib, re, subprocess

ORIGIN = "https://github.com/CsabaConsulting/trainium_frontier.git"
BRANCH = "experiment/20260923-dense6x1024-matrixlr-cohort"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("control", "candidate"), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--expected-bytes", type=int, required=True)
    ap.add_argument("--archive", type=pathlib.Path, required=True)
    ap.add_argument("--archive-sha256", required=True)
    ap.add_argument("--evidence-pushed-receipt", type=pathlib.Path, required=True)
    ap.add_argument("--evidence-pushed-receipt-sha256", required=True)
    ap.add_argument("--receipt", type=pathlib.Path, required=True)
    args = ap.parse_args()
    proof_path = args.evidence_pushed_receipt.resolve()
    if sha(proof_path) != args.evidence_pushed_receipt_sha256:
        raise SystemExit("evidence-push proof hash mismatch")
    proof = json.loads(proof_path.read_text())
    checkpoint, archive = args.checkpoint.resolve(), args.archive.resolve()
    exact = {"schema": "dense6x1024-cohort-evidence-push-v1", "arm": args.arm,
             "label": args.label, "archive_path": str(archive),
             "archive_sha256": args.archive_sha256, "checkpoint_path": str(checkpoint),
             "checkpoint_sha256": args.expected_sha256, "checkpoint_bytes": args.expected_bytes,
             "origin": ORIGIN, "branch": BRANCH, "retrieved_verified_committed_pushed": True}
    if any(proof.get(key) != value for key, value in exact.items()):
        raise SystemExit("evidence-push proof contract mismatch")
    evidence_commit = proof.get("evidence_commit")
    if not isinstance(evidence_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", evidence_commit):
        raise SystemExit("invalid evidence commit")
    remote = subprocess.run(["git", "ls-remote", ORIGIN, f"refs/heads/{BRANCH}"],
                            text=True, capture_output=True, check=True).stdout.split()
    if len(remote) != 2 or remote[0] != evidence_commit:
        raise SystemExit("evidence commit is not the verified remote branch head")
    if not archive.is_file() or archive.is_symlink() or sha(archive) != args.archive_sha256:
        raise SystemExit("archive identity mismatch")
    if checkpoint.name != "final.pt" or checkpoint.parent.name != "out":
        raise SystemExit("checkpoint path is not exact sandbox out/final.pt")
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise SystemExit("checkpoint is missing/non-regular/symlink")
    if checkpoint.stat().st_size != args.expected_bytes or sha(checkpoint) != args.expected_sha256:
        raise SystemExit("checkpoint identity mismatch")
    receipt_path = args.receipt.resolve(); receipt_path.parent.mkdir(parents=True, exist_ok=True)
    intent = {**exact, "evidence_commit": evidence_commit, "proof_path": str(proof_path),
              "proof_sha256": args.evidence_pushed_receipt_sha256,
              "state": "durable_intent_reserved", "deleted": False, "other_deletions": []}
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(intent, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(receipt_path.parent)
    checkpoint.unlink(); fsync_directory(checkpoint.parent)
    if checkpoint.exists(): raise SystemExit("checkpoint deletion failed")
    complete = {**intent, "state": "complete", "deleted": True}
    temporary = receipt_path.with_name(receipt_path.name + ".complete.tmp")
    temp_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(temp_fd, "w") as stream:
        json.dump(complete, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, receipt_path); fsync_directory(receipt_path.parent)

if __name__ == "__main__": main()
