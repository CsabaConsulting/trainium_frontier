#!/usr/bin/env python3
"""Fail-closed local arm driver; outer fresh-start guard remains mandatory."""
import argparse, hashlib, json, os, pathlib, subprocess, sys, tarfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
EVIDENCE = pathlib.Path(__file__).resolve().parent
EXPECTED = {
    "control": ("trn22", "submissions/dense6x1024_lean_review",
                "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"),
    "candidate": ("trn21", "submissions/dense6x1024_lean_review_matrixlr025",
                  "abdc8372808489a2bdf94885922aa12e7173b5a1dc4d91cab1bd874d2573add8"),
}
LAUNCH_SHA = "b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def require_tracked_clean(paths):
    relative = [str(path.relative_to(ROOT)) for path in paths]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, text=True, capture_output=True, check=True).stdout
    if dirty:
        raise SystemExit("protected cohort files are dirty")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=EXPECTED, required=True)
    ap.add_argument("--authorization-sha256", required=True)
    ap.add_argument("--sandbox-root", type=pathlib.Path, required=True)
    ap.add_argument("--archive", type=pathlib.Path, required=True)
    args = ap.parse_args()
    host, package_rel, train_sha = EXPECTED[args.arm]
    auth_path = EVIDENCE / f"{args.arm}-authorization.json"
    if sha(auth_path) != args.authorization_sha256:
        raise SystemExit("authorization hash mismatch")
    auth = json.loads(auth_path.read_text())
    exact = {"arm": args.arm, "host": host, "package": package_rel,
             "train_sha256": train_sha, "launch_command_sha256": LAUNCH_SHA}
    if any(auth.get(key) != value for key, value in exact.items()):
        raise SystemExit("authorization contract mismatch")
    if auth.get("state") != "human_approved_pending_operator" or auth.get("smoke_authorized"):
        raise SystemExit("authorization state/mode mismatch")
    if os.environ.get("TRAINIUM_COHORT_HOST") != host:
        raise SystemExit("TRAINIUM_COHORT_HOST does not match authorized host")
    package = ROOT / package_rel
    require_tracked_clean([auth_path, pathlib.Path(__file__).resolve(), package / "train.py",
                           package / "launch-command.txt", package / "provenance.json"])
    if sha(package / "train.py") != train_sha or sha(package / "launch-command.txt") != LAUNCH_SHA:
        raise SystemExit("package hash mismatch")
    if args.sandbox_root.exists() or args.archive.exists():
        raise SystemExit("sandbox/archive already exists")
    train_box = args.sandbox_root / "train"
    eval_box = args.sandbox_root / "eval"
    rehearsal = ROOT / "research/scratch/submission_rehearsal.py"
    subprocess.run([sys.executable, str(rehearsal), "--package", str(package),
                    "--sandbox", str(train_box), "--mode", "train", "--fresh-caches"],
                   cwd=ROOT, check=True)
    checkpoint = train_box / "out/final.pt"
    if not checkpoint.is_file():
        raise SystemExit("training produced no checkpoint")
    subprocess.run([sys.executable, str(rehearsal), "--package", str(package),
                    "--sandbox", str(eval_box), "--mode", "eval", "--checkpoint", str(checkpoint),
                    "--eval-tokens", "20971520", "--fresh-caches"], cwd=ROOT, check=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(args.sandbox_root, arcname=args.sandbox_root.name)
    if not args.archive.is_file() or args.archive.stat().st_size == 0:
        raise SystemExit("archive failed")
    print(json.dumps({"arm": args.arm, "host": host, "checkpoint_sha256": sha(checkpoint),
                      "archive": str(args.archive), "archive_sha256": sha(args.archive),
                      "checkpoint_pruned": False}, sort_keys=True))

if __name__ == "__main__":
    main()
