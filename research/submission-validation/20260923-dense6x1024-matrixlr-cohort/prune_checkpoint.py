#!/usr/bin/env python3
"""Authenticated, no-follow exact checkpoint prune with durable journal paths."""
import argparse, hashlib, json, os, pathlib, re, stat, subprocess

ORIGIN = "https://github.com/CsabaConsulting/trainium_frontier.git"
BRANCH = "experiment/20260923-dense6x1024-matrixlr-cohort"
ARMS = {
    "control": ("fs-20260923-dense6x1024-9173-cold-repeat-r1",
                pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_dense9173_control")),
    "candidate": ("fs-20260923-dense6x1024-matrixlr025-r1",
                  pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_matrixlr025_candidate")),
}

def nofollow_bytes(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode): raise SystemExit(f"not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream: return stream.read(), info.st_size
    finally: os.close(fd)

def sha(path): return hashlib.sha256(nofollow_bytes(path)[0]).hexdigest()

def require_no_symlink_components(path, require_leaf):
    if not path.is_absolute(): raise SystemExit("path must be absolute")
    current = pathlib.Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        leaf = index == len(parts) - 1
        try: info = os.lstat(current)
        except FileNotFoundError:
            if leaf and not require_leaf: return
            raise SystemExit(f"missing path component: {current}")
        if stat.S_ISLNK(info.st_mode): raise SystemExit(f"symlink path component: {current}")

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)

def remote_head():
    output = subprocess.run(["git", "ls-remote", ORIGIN, f"refs/heads/{BRANCH}"],
                            text=True, capture_output=True, check=True).stdout.split()
    if len(output) != 2: raise SystemExit("remote branch lookup malformed")
    return output[0]

def execute(args, lookup_remote_head=remote_head):
    label, root = ARMS[args.arm]
    expected = {
        "label": label,
        "checkpoint": root / "out" / label / "train" / "out" / "final.pt",
        "archive": root / "out" / "archives" / f"{label}.tar.gz",
        "proof": root / "out" / label / "evidence-pushed-receipt.json",
        "receipt": root / "out" / label / "checkpoint-prune-receipt.json",
    }
    supplied = {"label": args.label, "checkpoint": args.checkpoint, "archive": args.archive,
                "proof": args.evidence_pushed_receipt, "receipt": args.receipt}
    if any(str(supplied[key]) != str(value) for key, value in expected.items()):
        raise SystemExit("lexical arm path/label contract mismatch")
    checkpoint, archive, proof_path, receipt_path = (expected[k] for k in
                                                      ("checkpoint", "archive", "proof", "receipt"))
    for path in (checkpoint, archive, proof_path): require_no_symlink_components(path, True)
    require_no_symlink_components(receipt_path, False)
    proof_bytes, _ = nofollow_bytes(proof_path)
    if hashlib.sha256(proof_bytes).hexdigest() != args.evidence_pushed_receipt_sha256:
        raise SystemExit("evidence-push proof hash mismatch")
    proof = json.loads(proof_bytes)
    exact = {"schema": "dense6x1024-cohort-evidence-push-v1", "arm": args.arm,
             "label": label, "archive_path": str(archive), "archive_sha256": args.archive_sha256,
             "checkpoint_path": str(checkpoint), "checkpoint_sha256": args.expected_sha256,
             "checkpoint_bytes": args.expected_bytes, "origin": ORIGIN, "branch": BRANCH,
             "retrieved_verified_committed_pushed": True}
    if set(proof) != set(exact) | {"evidence_commit"} or any(proof.get(k) != v for k, v in exact.items()):
        raise SystemExit("evidence-push proof contract mismatch")
    evidence_commit = proof["evidence_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", evidence_commit) or lookup_remote_head() != evidence_commit:
        raise SystemExit("evidence commit is not verified remote head")
    if sha(archive) != args.archive_sha256: raise SystemExit("archive identity mismatch")
    checkpoint_bytes, checkpoint_size = nofollow_bytes(checkpoint)
    if checkpoint_size != args.expected_bytes or hashlib.sha256(checkpoint_bytes).hexdigest() != args.expected_sha256:
        raise SystemExit("checkpoint identity mismatch")
    temporary = receipt_path.with_name(receipt_path.name + ".complete.tmp")
    require_no_symlink_components(temporary, False)
    intent = {**exact, "evidence_commit": evidence_commit,
              "proof_sha256": args.evidence_pushed_receipt_sha256,
              "state": "durable_intent_reserved", "deleted": False, "other_deletions": []}
    receipt_fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        temp_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except Exception:
        os.close(receipt_fd); raise
    with os.fdopen(receipt_fd, "w") as stream:
        json.dump(intent, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    complete = {**intent, "state": "complete", "deleted": True}
    with os.fdopen(temp_fd, "w") as stream:
        json.dump(complete, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(receipt_path.parent)
    os.unlink(checkpoint); fsync_directory(checkpoint.parent)
    os.replace(temporary, receipt_path); fsync_directory(receipt_path.parent)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--expected-sha256", required=True); ap.add_argument("--expected-bytes", type=int, required=True)
    ap.add_argument("--archive", type=pathlib.Path, required=True); ap.add_argument("--archive-sha256", required=True)
    ap.add_argument("--evidence-pushed-receipt", type=pathlib.Path, required=True)
    ap.add_argument("--evidence-pushed-receipt-sha256", required=True)
    ap.add_argument("--receipt", type=pathlib.Path, required=True)
    execute(ap.parse_args())

if __name__ == "__main__": main()
