#!/usr/bin/env python3
"""Fail-closed local arm driver; outer fresh-start guard remains mandatory."""
import argparse, hashlib, json, math, os, pathlib, socket, subprocess, sys, tarfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
EVIDENCE = pathlib.Path(__file__).resolve().parent
EXPECTED = {
    "control": ("trn22", "ip-172-31-32-253", "fs-20260923-dense6x1024-9173-cold-repeat-r1", "submissions/dense6x1024_lean_review",
                "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"),
    "candidate": ("trn21", "ip-172-31-34-85", "fs-20260923-dense6x1024-matrixlr025-r1", "submissions/dense6x1024_lean_review_matrixlr025",
                  "abdc8372808489a2bdf94885922aa12e7173b5a1dc4d91cab1bd874d2573add8"),
    "control_r2": ("trn22", "ip-172-31-32-253", "fs-20260923-dense6x1024-9173-cold-repeat-r2", "submissions/dense6x1024_lean_review",
                   "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"),
    "candidate_r2": ("trn21", "ip-172-31-34-85", "fs-20260923-dense6x1024-matrixlr025-r2", "submissions/dense6x1024_lean_review_matrixlr025",
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

def extract_eval_metrics(text):
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"val_bpb", "num_tokens", "num_bytes"} <= value.keys():
            return value
    raise SystemExit("fresh eval emitted no metrics JSON")

def validate_eval_metrics(metrics):
    if (metrics.get("num_tokens") != 20971520 or metrics.get("num_bytes") != 83462186
            or not math.isfinite(float(metrics.get("val_bpb", math.nan)))):
        raise SystemExit("fresh eval metrics contract failed")

def scrub_environment(environment):
    clean = {key: value for key, value in environment.items()
             if not key.startswith("NEURON_COMPETITION_R1_")}
    blocked = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
               "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
               "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
               "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_ERROR_FILE")
    for key in blocked:
        clean.pop(key, None)
    clean.update({"PYTHONUNBUFFERED": "1", "NEURON_LOGICAL_NC_CONFIG": "2"})
    return clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=EXPECTED, required=True)
    ap.add_argument("--authorization-sha256", required=True)
    ap.add_argument("--sandbox-root", type=pathlib.Path, required=True)
    ap.add_argument("--archive", type=pathlib.Path, required=True)
    args = ap.parse_args()
    host, hostname, label, package_rel, train_sha = EXPECTED[args.arm]
    auth_path = EVIDENCE / f"{args.arm}-authorization.json"
    if sha(auth_path) != args.authorization_sha256:
        raise SystemExit("authorization hash mismatch")
    auth = json.loads(auth_path.read_text())
    exact = {"arm": args.arm, "host": host, "expected_hostname": hostname,
             "label": label, "package": package_rel,
             "train_sha256": train_sha, "launch_command_sha256": LAUNCH_SHA}
    if any(auth.get(key) != value for key, value in exact.items()):
        raise SystemExit("authorization contract mismatch")
    if auth.get("state") != "human_approved_pending_operator" or auth.get("smoke_authorized"):
        raise SystemExit("authorization state/mode mismatch")
    if socket.gethostname() != hostname:
        raise SystemExit("actual hostname does not match authorized host")
    package = ROOT / package_rel
    require_tracked_clean([auth_path, pathlib.Path(__file__).resolve(), package / "train.py",
                           package / "launch-command.txt", package / "provenance.json"])
    if sha(package / "train.py") != train_sha or sha(package / "launch-command.txt") != LAUNCH_SHA:
        raise SystemExit("package hash mismatch")
    source = (package / "train.py").read_text()
    if ("default=int(os.environ.get('NEURON_COMPETITION_R1_SEED', '42'))" not in source
            or "MAX_TRAIN_SECONDS = 1800" not in source):
        raise SystemExit("seed/time constants are not exact")
    safe_sandbox = (ROOT / "out" / label).resolve()
    safe_archive = (ROOT / "out" / "archives" / f"{label}.tar.gz").resolve()
    if args.sandbox_root.resolve() != safe_sandbox or args.archive.resolve() != safe_archive:
        raise SystemExit("sandbox/archive path is not the exact label-owned safe path")
    if args.sandbox_root.exists() or args.archive.exists():
        raise SystemExit("sandbox/archive already exists")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    train_box = args.sandbox_root / "train"
    eval_box = args.sandbox_root / "eval"
    rehearsal = ROOT / "research/scratch/submission_rehearsal.py"
    child_env = scrub_environment(os.environ)
    subprocess.run([sys.executable, str(rehearsal), "--package", str(package),
                    "--sandbox", str(train_box), "--mode", "train", "--fresh-caches"],
                   cwd=ROOT, env=child_env, check=True)
    checkpoint = train_box / "out/final.pt"
    if not checkpoint.is_file():
        raise SystemExit("training produced no checkpoint")
    checkpoint_sha, checkpoint_size = sha(checkpoint), checkpoint.stat().st_size
    evaluated = subprocess.run(
        [sys.executable, str(rehearsal), "--package", str(package), "--sandbox", str(eval_box),
         "--mode", "eval", "--checkpoint", str(checkpoint), "--eval-tokens", "20971520",
         "--fresh-caches"], cwd=ROOT, env=child_env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    eval_log = args.sandbox_root / "fresh-eval.log"
    eval_log.write_text(evaluated.stdout)
    metrics = extract_eval_metrics(evaluated.stdout)
    validate_eval_metrics(metrics)
    command_line = next((line for line in evaluated.stdout.splitlines()
                         if line.startswith("REHEARSAL_COMMAND ")), None)
    if command_line is None:
        raise SystemExit("fresh eval command receipt missing")
    eval_command = json.loads(command_line.removeprefix("REHEARSAL_COMMAND "))
    required_tail = ["--device", "neuron", "--batch-size", "1", "--seq-len", "1024",
                     "--eval-tokens", "20971520", "--timeout-seconds", "1200"]
    if not all(eval_command[index:index + len(required_tail)] == required_tail
               for index in range(len(eval_command))):
        raise SystemExit("fresh eval was not B1/world1/seq1024 with exact protected size")
    evidence = {"arm": args.arm, "host": host, "hostname": hostname, "label": label,
                "checkpoint_sha256": checkpoint_sha, "checkpoint_bytes": checkpoint_size,
                "metrics": metrics, "checkpoint_archive_excluded": True,
                "compile_caches_archive_excluded": True}
    evidence_path = args.sandbox_root / "cohort-result.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    include = [train_box / "rehearsal.json", train_box / "steps.jsonl",
               eval_box / "rehearsal.json", eval_log, evidence_path]
    missing = [str(path) for path in include if not path.is_file()]
    if missing:
        raise SystemExit(f"required archive evidence missing: {missing}")
    archive_manifest = {str(path.relative_to(args.sandbox_root)): sha(path) for path in include}
    manifest_path = args.sandbox_root / "archive-manifest.json"
    manifest_path.write_text(json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n")
    include.append(manifest_path)
    with tarfile.open(args.archive, "x:gz") as archive:
        for path in include:
            archive.add(path, arcname=str(path.relative_to(args.sandbox_root)))
    if not args.archive.is_file() or args.archive.stat().st_size == 0:
        raise SystemExit("archive failed")
    archive_sha = sha(args.archive)
    expected_members = set(archive_manifest) | {"archive-manifest.json"}
    with tarfile.open(args.archive, "r:gz") as archive:
        actual_members = set(archive.getnames())
        if actual_members != expected_members:
            raise SystemExit("archive member set mismatch")
        for name, expected_sha in archive_manifest.items():
            stream = archive.extractfile(name)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected_sha:
                raise SystemExit(f"archive member hash mismatch: {name}")
    if sha(checkpoint) != checkpoint_sha or checkpoint.stat().st_size != checkpoint_size:
        raise SystemExit("checkpoint changed before retrieval handoff")
    print(json.dumps({**evidence, "archive": str(args.archive), "archive_sha256": archive_sha,
                      "checkpoint_pruned": False, "ready_to_prune_after_retrieval_push": True,
                      "prune_contract": {"exact_path": str(checkpoint),
                                         "expected_sha256": checkpoint_sha,
                                         "expected_bytes": checkpoint_size}}, sort_keys=True))

if __name__ == "__main__":
    main()
