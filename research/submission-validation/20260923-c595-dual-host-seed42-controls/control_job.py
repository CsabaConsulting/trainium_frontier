#!/usr/bin/env python3
"""Fail-closed exact-c595 training control with two fresh singleton evaluations."""
import argparse, hashlib, json, math, os, pathlib, re, socket, subprocess, sys, tarfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
TRAIN_SHA = "c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
LAUNCH_SHA = "b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd"
COMPILER = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
ARMS = {
    "control_trn21": ("trn21", "ip-172-31-34-85", "fs-20260923-c595-control-trn21-r1"),
    "control_trn22": ("trn22", "ip-172-31-32-253", "fs-20260923-c595-control-trn22-r1"),
    "control_trn21_r2": ("trn21", "ip-172-31-34-85", "fs-20260924-c595-control-trn21-r2"),
    "control_trn22_r2": ("trn22", "ip-172-31-32-253", "fs-20260924-c595-control-trn22-r2"),
}
REFERENCE_LOSSES = {0:9.0112, 1:8.7816, 2:8.3625, 3:8.2986, 4:8.0851,
                    10:6.8196, 20:5.9603, 30:5.5119, 40:5.2152, 50:4.8776}
REQUIRED_LOSS_STEPS = (0, 1, 2, 3, 4, 10, 20, 30, 40)

def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()

def durable_create(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally: os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(directory)
    finally: os.close(directory)

def scrub_eval_environment(source):
    blocked = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
               "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "PYTHONPATH",
               "PYTHONHOME", "NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES"}
    result = {}
    for key, value in source.items():
        if key in blocked or key.startswith("TORCHELASTIC_") or key.startswith("NEURON_COMPETITION_R1_"):
            continue
        if key == "NEURON_CC_FLAGS":
            continue
        result[key] = value
    result.update(PATH="/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
                  NKI_VENV_PATH="/home/ubuntu/workspace/native_venv", PYTHONUNBUFFERED="1",
                  NEURON_LOGICAL_NC_CONFIG="2")
    return result

def parse_metrics(output, expected_tokens, expected_bytes, expected_timeout=1200):
    decoder, found = json.JSONDecoder(), []
    metric_keys = {"val_bpb", "num_tokens", "num_bytes", "total_nats", "eval_seconds",
                   "eval_timeout", "eval_over_timeout"}
    for offset, character in enumerate(output):
        if character != "{": continue
        try: value, _ = decoder.raw_decode(output[offset:])
        except json.JSONDecodeError: continue
        if isinstance(value, dict) and metric_keys.intersection(value):
            found.append(value)
    if len(found) != 1: raise SystemExit(f"expected exactly one evaluation metrics object, got {len(found)}")
    value = found[0]
    if set(value) != metric_keys: raise SystemExit("evaluation metrics schema mismatch")
    numeric = lambda item: isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
    if not numeric(value["val_bpb"]) or value["val_bpb"] <= 0: raise SystemExit("invalid BPB")
    if not numeric(value["total_nats"]) or value["total_nats"] <= 0: raise SystemExit("invalid total_nats")
    if not numeric(value["eval_seconds"]) or value["eval_seconds"] <= 0: raise SystemExit("invalid eval_seconds")
    if type(value["num_tokens"]) is not int or value["num_tokens"] != expected_tokens:
        raise SystemExit("invalid token count/type")
    if type(value["num_bytes"]) is not int or value["num_bytes"] != expected_bytes:
        raise SystemExit("invalid byte count/type")
    if type(value["eval_timeout"]) is not int or value["eval_timeout"] != expected_timeout:
        raise SystemExit("invalid eval timeout")
    if value["eval_over_timeout"] is not False: raise SystemExit("evaluation exceeded timeout")
    return value

def parse_losses(output):
    values = {}
    for match in re.finditer(r"^step (\d{5}) \| loss ([0-9]+(?:\.[0-9]+)?) \|", output, re.M):
        step, loss = int(match.group(1)), float(match.group(2))
        if step in values: raise SystemExit(f"duplicate rank-zero loss record at step {step}")
        values[step] = loss
    missing = [step for step in (*REQUIRED_LOSS_STEPS, 50) if step not in values]
    if missing: raise SystemExit(f"missing frozen source loss records: {missing}")
    failures = {step: abs(values[step] - REFERENCE_LOSSES[step]) for step in REQUIRED_LOSS_STEPS
                if abs(values[step] - REFERENCE_LOSSES[step]) > 0.0001}
    if failures: raise SystemExit(f"source-loss gate failed: {failures}")
    return {"required": {str(s): values[s] for s in REQUIRED_LOSS_STEPS},
            "step_50_separate": values[50], "reference": REFERENCE_LOSSES,
            "maximum_absolute_delta": max(abs(values[s]-REFERENCE_LOSSES[s]) for s in REQUIRED_LOSS_STEPS)}

def run_eval(argv, environment, log_path, expected_tokens, expected_bytes):
    completed = subprocess.run(argv, cwd=ROOT, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout)
    if completed.returncode: raise SystemExit(f"evaluation failed rc={completed.returncode}")
    metrics = parse_metrics(completed.stdout, expected_tokens, expected_bytes, 1200)
    return metrics

def strict_checkpoint_check(checkpoint, environment, log_path):
    source = """import json,sys,torch\nvalue=torch.load(sys.argv[1],map_location='cpu',weights_only=True)\ncount=0\nstack=[value]\nwhile stack:\n v=stack.pop()\n if isinstance(v,dict): stack.extend(v.values())\n elif isinstance(v,(list,tuple)): stack.extend(v)\n elif isinstance(v,torch.Tensor):\n  count+=1\n  if (v.is_floating_point() or v.is_complex()) and not torch.isfinite(v).all().item(): raise SystemExit('nonfinite checkpoint tensor')\nprint(json.dumps({'weights_only':True,'finite':True,'tensor_count':count}))\n"""
    completed=subprocess.run([PYTHON,"-c",source,str(checkpoint)],cwd=ROOT,env=environment,
                             text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout)
    if completed.returncode: raise SystemExit("strict finite checkpoint check failed")
    record=json.loads(completed.stdout.strip().splitlines()[-1])
    if not record.get("weights_only") or not record.get("finite") or record.get("tensor_count",0)<1:
        raise SystemExit("strict checkpoint receipt invalid")
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--sandbox-root", type=pathlib.Path, required=True)
    parser.add_argument("--archive", type=pathlib.Path, required=True)
    args = parser.parse_args()
    host, hostname, label = ARMS[args.arm]
    auth_path = HERE / f"{args.arm}-authorization.json"
    if sha(auth_path) != args.authorization_sha256: raise SystemExit("authorization hash mismatch")
    auth = json.loads(auth_path.read_text())
    expected_auth = {"arm":args.arm, "host":host, "expected_hostname":hostname, "label":label,
                     "package":"submissions/dense6x1024_direct_review", "train_sha256":TRAIN_SHA,
                     "launch_command_sha256":LAUNCH_SHA, "single_causal_change":"none"}
    if any(auth.get(k) != v for k,v in expected_auth.items()): raise SystemExit("authorization mismatch")
    if auth.get("state") != "human_approved_pending_operator" or auth.get("submission_authorized"):
        raise SystemExit("authorization state mismatch")
    if socket.gethostname() != hostname: raise SystemExit("wrong physical host")
    if os.environ.get("NEURON_CC_FLAGS") != COMPILER: raise SystemExit("wrong effective training compiler flags")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2": raise SystemExit("wrong LNC")
    package = ROOT / "submissions/dense6x1024_direct_review"
    if sha(package/"train.py") != TRAIN_SHA or sha(package/"launch-command.txt") != LAUNCH_SHA:
        raise SystemExit("protected package mismatch")
    source = (package/"train.py").read_text()
    if "default=int(os.environ.get('NEURON_COMPETITION_R1_SEED', '42'))" not in source or "MAX_TRAIN_SECONDS = 1800" not in source:
        raise SystemExit("seed/time source contract mismatch")
    exact_sandbox = ROOT/"out"/label
    exact_archive = ROOT/"out"/"archives"/f"{label}.tar.gz"
    if str(args.sandbox_root) != str(exact_sandbox) or str(args.archive) != str(exact_archive):
        raise SystemExit("nonexact owned output path")
    if exact_sandbox.exists() or exact_archive.exists(): raise SystemExit("owned output already exists")
    exact_archive.parent.mkdir(parents=True, exist_ok=True)
    train_box = exact_sandbox/"train"
    rehearsal = ROOT/"research/scratch/submission_rehearsal.py"
    train_env = scrub_eval_environment(os.environ)
    train_env["NEURON_CC_FLAGS"] = COMPILER
    command = [sys.executable, str(rehearsal), "--package", str(package), "--sandbox", str(train_box),
               "--mode", "train", "--fresh-caches"]
    trained = subprocess.run(command, cwd=ROOT, env=train_env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    exact_sandbox.mkdir(parents=True, exist_ok=True)
    training_log = exact_sandbox/"training.log"; training_log.write_text(trained.stdout)
    if trained.returncode: raise SystemExit(f"training failed rc={trained.returncode}")
    for marker in ("device=neuron dtype=torch.bfloat16 world_size=4",
                   "torch.compile enabled (backend=neuron)", "muon step: COMPILED"):
        if marker not in trained.stdout: raise SystemExit(f"accelerated-path marker missing: {marker}")
    loss_comparison = parse_losses(trained.stdout)
    checkpoint = train_box/"out/final.pt"
    if not checkpoint.is_file(): raise SystemExit("checkpoint missing")
    checkpoint_sha, checkpoint_bytes = sha(checkpoint), checkpoint.stat().st_size
    eval_env = scrub_eval_environment(os.environ)
    checkpoint_check = strict_checkpoint_check(checkpoint, eval_env, exact_sandbox/"checkpoint-strict-finite.log")
    primary_argv = [PYTHON, "prepare.py", "eval-public", "--train-py",
                    "submissions/dense6x1024_direct_review/train.py", "--checkpoint", str(checkpoint)]
    primary = run_eval(primary_argv, eval_env, exact_sandbox/"primary-public1024.log", 2097152, 8418678)
    secondary_argv = [PYTHON, "prepare.py", "eval-public", "--train-py",
                      "submissions/dense6x1024_direct_review/train.py", "--checkpoint", str(checkpoint),
                      "--device", "neuron", "--batch-size", "1", "--seq-len", "1024",
                      "--eval-tokens", "20971520", "--timeout-seconds", "1200"]
    secondary = run_eval(secondary_argv, eval_env, exact_sandbox/"secondary-public20m.log", 20971520, 83462186)
    rehearsal_record = json.loads((train_box/"rehearsal.json").read_text())
    steps = rehearsal_record["step_boundaries"]
    result = {"schema":"c595-dual-control-result-v1", "arm":args.arm, "host":host,
              "hostname":hostname, "label":label, "train_sha256":TRAIN_SHA, "seed":42,
              "world_size":4, "lnc":2, "compiler_flags":COMPILER,
              "checkpoint_sha256":checkpoint_sha, "checkpoint_bytes":checkpoint_bytes,
              "checkpoint_strict_finite":checkpoint_check,
              "checkpoint_archive_excluded":True, "compile_caches_archive_excluded":True,
              "primary":primary, "primary_argv":primary_argv, "secondary":secondary,
              "secondary_argv":secondary_argv, "loss_comparison":loss_comparison,
              "completed_steps":steps, "historical_step_band":{"minimum":3602,"maximum":3674,
              "inside":3602 <= steps <= 3674, "policy":"diagnostic_only"},
              "prior_component_top_passed":False, "prior_discriminator_top_passed":False,
              "prior_failures_preserved_not_relabeled":True, "ready_to_prune":False}
    result_path = exact_sandbox/"control-result.json"
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    include = [train_box/"rehearsal.json", train_box/"steps.jsonl", training_log,
               exact_sandbox/"checkpoint-strict-finite.log",
               exact_sandbox/"primary-public1024.log", exact_sandbox/"secondary-public20m.log", result_path]
    missing = [str(p) for p in include if not p.is_file()]
    if missing: raise SystemExit(f"required evidence missing: {missing}")
    manifest = {str(p.relative_to(exact_sandbox)):sha(p) for p in include}
    manifest_path = exact_sandbox/"archive-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    include.append(manifest_path)
    with tarfile.open(exact_archive, "x:gz") as archive:
        for path in include: archive.add(path, arcname=str(path.relative_to(exact_sandbox)))
    with tarfile.open(exact_archive, "r:gz") as archive:
        if set(archive.getnames()) != set(manifest)|{"archive-manifest.json"}: raise SystemExit("archive member mismatch")
        for name,want in manifest.items():
            stream=archive.extractfile(name)
            if stream is None or hashlib.sha256(stream.read()).hexdigest()!=want: raise SystemExit("archive hash mismatch")
    if sha(checkpoint)!=checkpoint_sha or checkpoint.stat().st_size!=checkpoint_bytes: raise SystemExit("checkpoint changed")
    archive_sha = sha(exact_archive)
    core_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    completion = {"schema":"c595-control-archive-completion-v1", "arm":args.arm,
                  "label":label, "core_commit":core_commit, "candidate_sha256":TRAIN_SHA,
                  "checkpoint_path":str(checkpoint), "checkpoint_sha256":checkpoint_sha,
                  "checkpoint_bytes":checkpoint_bytes, "archive_path":str(exact_archive),
                  "archive_sha256":archive_sha, "archive_bytes":exact_archive.stat().st_size,
                  "archive_members":{**manifest, "archive-manifest.json":sha(manifest_path)},
                  "ready_to_retrieve":True, "archive_immutable_after_sha256":True}
    completion_path = exact_sandbox/"archive-completion.json"
    durable_create(completion_path, completion)
    result.update(archive=str(exact_archive), archive_sha256=archive_sha,
                  archive_completion=str(completion_path),
                  archive_completion_sha256=sha(completion_path), ready_to_prune=False,
                  prune_contract={"exact_path":str(checkpoint),"expected_sha256":checkpoint_sha,
                                  "expected_bytes":checkpoint_bytes})
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
