#!/usr/bin/env python3
"""Run ONE approved job on THIS Trn2 host, end to end, with receipts. Operator-only.

  python workflow/tools/run_job.py --host-config research/v2/host.json --job research/v2/experiments/<label>/job.json

Stages (each writes a receipt into research/v2/experiments/<label>/; any failure stops the job):
  1 preflight  idle host (no tagged/torch processes, no /dev/neuron holders), disk >= floor + footprint,
               git clean at the job's commit, train file SHA-256 == job, NEURON_CC_FLAGS sane,
               G2 receipt present with verdict PASS for exactly these bytes and argv
  2 launch     contained.py-equivalent launch (env tag + new session [+ systemd scope])
  3 wait       until the training leader exits (timeout -> stop)
  4 quiesce    no tagged processes, no neuron holders, stable files; else stop -> quiesce -> breaker
  5 parse      the '---' summary block + step lines (steps, seconds, over_budget, startup, epoch-2 step)
  6 eval       fresh process, contained, organizer `prepare.py eval-public` with the host config's
               verified eval args (context 1024, 20,971,520 tokens); parse val_bpb
  7 record     append a schema'd record to research/v2/ledger.jsonl (flock)
  8 decide     decide.py verdict for the arm -> verdict.json
  9 archive    copy checkpoint to archive/sha256/<hash>.pt, verify, delete on host unless protected

Exit codes: 0 done; 3 containment failure (open breaker, alert human); 4 preflight refused;
5 training failed; 6 evaluation failed; 7 archive failed. Never relaunches; a retry is a new label.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contained  # noqa: E402
import decide  # noqa: E402

EVAL_TOKENS, EVAL_SEQ = decide.EVAL_TOKENS, decide.EVAL_SEQ


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(d: Path, name: str, obj) -> None:
    (d / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


class JobError(Exception):
    def __init__(self, code, stage, msg):
        super().__init__(msg)
        self.code, self.stage = code, stage


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()


def other_training_procs():
    pat = re.compile(r"(torchrun|train\.py|prepare\.py eval-public|run_experiment\.py)")
    me = os.getpid()
    hits = []
    for cmd_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(cmd_path.parent.name)
            cmd = cmd_path.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except (OSError, ValueError):
            continue
        if pid != me and pat.search(cmd) and "run_job.py" not in cmd and "preflight_dryrun.py" not in cmd:
            hits.append({"pid": pid, "cmd": cmd[:160]})
    return hits


def preflight(cfg, job, d: Path, repo: Path):
    checks = {}
    checks["tagged_processes"] = contained.describe(contained.tagged_pids(job["label"]))
    checks["other_training_processes"] = other_training_procs()
    checks["neuron_holders"] = contained.neuron_holders()
    st = shutil.disk_usage(repo)
    need = cfg.get("disk_floor_gib", 6.5) * 2**30 + job.get("footprint_gib", 1.2) * 2**30
    checks["disk_free_bytes"], checks["disk_needed_bytes"] = st.free, int(need)
    checks["git_head"] = git(repo, "rev-parse", "HEAD")
    checks["git_dirty"] = bool(git(repo, "status", "--porcelain", "--untracked-files=no"))
    train = repo / job["train_py"]
    checks["train_sha256"] = sha256(train) if train.exists() else None
    cc = os.environ.get("NEURON_CC_FLAGS")
    checks["neuron_cc_flags"] = cc
    g2p = repo / job["g2_receipt"]
    g2 = json.loads(g2p.read_text()) if g2p.exists() else {}
    checks["g2"] = {"path": str(g2p), "verdict": g2.get("verdict"), "sha256": g2.get("train_py_sha256"), "argv": g2.get("argv")}
    problems = []
    if checks["tagged_processes"] or checks["other_training_processes"]:
        problems.append("host not idle")
    if checks["neuron_holders"]:
        problems.append("/dev/neuron held")
    if st.free < need:
        problems.append("disk below floor + footprint")
    if job.get("git_commit") and checks["git_head"] != job["git_commit"]:
        problems.append("git HEAD differs from job commit")
    if checks["git_dirty"]:
        problems.append("tracked files modified")
    if checks["train_sha256"] != job["train_sha256"]:
        problems.append("train file SHA-256 differs from job")
    if cc is not None and "--optlevel=1" not in cc:
        problems.append("NEURON_CC_FLAGS set without --optlevel=1")
    if g2.get("verdict") != "PASS" or g2.get("train_py_sha256") != job["train_sha256"] or g2.get("argv") != job["argv"]:
        problems.append("no PASS G2 receipt for exactly these bytes and argv")
    if not cfg.get("eval_args_verified"):
        problems.append("host config eval_args not verified by G1 (set eval_args_verified after reading prepare.py)")
    checks["problems"], checks["ok"], checks["utc"] = problems, not problems, now()
    write(d, "preflight.json", checks)
    if problems:
        raise JobError(4, "preflight", "; ".join(problems))


def launch_and_wait(cfg, job, d: Path, repo: Path, label: str, cmd: list[str], log_name: str, timeout: int):
    env = dict(os.environ, **cfg.get("env", {}), **{contained.TAG: label})
    argv = list(cmd)
    scope = None
    if cfg.get("use_cgroup") and shutil.which("systemd-run"):
        scope = f"exp-{label}"
        argv = ["systemd-run", "--user", "--scope", f"--unit={scope}", "--collect", "--quiet"] + argv
    with open(d / log_name, "ab") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=str(repo))
    rec = {"label": label, "started_utc": now(), "leader_pid": proc.pid, "pgid": proc.pid, "cgroup_scope": scope,
           "argv": cmd, "env_keys": sorted(cfg.get("env", {}))}
    write(d, f"launch-{label}.json", rec)
    (d / "launch.json").write_text(json.dumps(rec))  # contained.stop reads this
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = None
    return rc, rec


def quiesce(label, d: Path, files, stop_first=False):
    """Prove quiescence; if anything tagged survives, stop it (TERM -> KILL) and prove again.
    Returns the receipt with `intervened=True` when containment had to stop a straggler (an anomaly the
    Supervisor must alert on); raises exit 3 (breaker) if anything survives the stop."""
    ns = argparse.Namespace(label=label, dir=str(d), grace=60, window=10, files=[str(f) for f in files])
    with contextlib.redirect_stdout(sys.stderr):
        if stop_first:
            contained.cmd_stop(ns)
        rc = contained.cmd_quiesce(ns)
    q = json.loads((d / "quiescence.json").read_text())
    q["intervened"] = stop_first
    (d / f"quiescence-{label}.json").write_text(json.dumps(q, indent=2))
    if rc != 0 and not stop_first:
        return quiesce(label, d, files, stop_first=True)
    if rc != 0:
        raise JobError(3, "quiesce", f"{label}: survivors or holders after stop; breaker")
    return q


SUMMARY_KEYS = {"training_seconds": float, "startup_allowance_est": float, "charged_seconds_est": float,
                "over_budget": lambda v: v.strip() == "True", "num_steps": int, "total_tokens_M": float,
                "world_size": int, "seq_len": int, "num_params_M": float}
STEP_RE = re.compile(r"^step (\d+) \| loss ([0-9.naninf]+) .*?\| dt ([0-9.]+)s .*?\| epoch (\d+)")


def parse_training_log(path: Path) -> dict:
    out, dts, epoch2 = {}, [], None
    in_summary = False
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line.strip())
        if m:
            step, dt, ep = int(m.group(1)), float(m.group(3)), int(m.group(4))
            if step >= 20:
                dts.append(dt)
            if ep >= 2 and epoch2 is None:
                epoch2 = step
            continue
        if line.strip() == "---":
            in_summary = True
            continue
        if in_summary and ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in SUMMARY_KEYS:
                try:
                    out[k] = SUMMARY_KEYS[k](v.strip())
                except ValueError:
                    pass
    dts.sort()
    out["median_step_s"] = dts[len(dts) // 2] if dts else None
    out["epoch_boundary_step"] = epoch2
    out["bad_loss"] = "bad loss detected" in path.read_text(errors="replace")
    return out


def append_ledger(ledger: Path, rec: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def archive(cfg, ckpt: Path, digest: str, protected: bool) -> str:
    root = Path(cfg["archive_dir"])
    dst = root / "sha256" / f"{digest}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_suffix(".partial")
        shutil.copyfile(ckpt, tmp)
        if sha256(tmp) != digest:
            tmp.unlink(missing_ok=True)
            raise JobError(7, "archive", "archived copy hash mismatch")
        tmp.rename(dst)
    elif sha256(dst) != digest:
        raise JobError(7, "archive", "existing archive object hash mismatch")
    if not protected:
        ckpt.unlink()
    return str(dst)


def run(cfg, job, repo: Path) -> dict:
    label = job["label"]
    d = repo / "research" / "v2" / "experiments" / label
    d.mkdir(parents=True, exist_ok=True)
    write(d, "job.json", job)
    preflight(cfg, job, d, repo)

    out_dir = Path(cfg.get("out_root", "out")) / label
    train_cmd = (cfg.get("train_cmd_override") or ["torchrun", "--standalone", "--nproc_per_node=4", job["train_py"]]) \
        + job["argv"] + ["--out-dir", str(out_dir)]
    rc, _ = launch_and_wait(cfg, job, d, repo, label, train_cmd, "run.log", job.get("timeout_s", 3600))
    ckpt = repo / out_dir / "final.pt"
    if rc is None:
        quiesce(label, d, [d / "run.log"], stop_first=True)
        raise JobError(5, "train", "training exceeded job timeout; stopped")
    q = quiesce(label, d, [p for p in (ckpt, d / "run.log") if p.exists()])
    summary = parse_training_log(d / "run.log")
    write(d, "train-summary.json", {**summary, "exit_code": rc, "quiescence_sha256": q.get("sha256", {})})
    if rc != 0 or summary.get("bad_loss") or not ckpt.exists():
        raise JobError(5, "train", f"training failed: rc={rc} bad_loss={summary.get('bad_loss')} ckpt={ckpt.exists()}")

    eval_label = f"{label}-eval"
    eval_cmd = cfg.get("eval_cmd_override") or (
        [cfg.get("python", "python"), "prepare.py", "eval-public", "--train-py", job["train_py"], "--checkpoint", str(out_dir / "final.pt")]
        + cfg["eval_args"])
    erc, _ = launch_and_wait(cfg, job, d, repo, eval_label, eval_cmd, "eval.log", job.get("eval_timeout_s", 2400))
    qe = quiesce(eval_label, d, [d / "eval.log"], stop_first=erc is None)
    text = (d / "eval.log").read_text(errors="replace")
    m = re.findall(cfg.get("eval_bpb_regex", r"val_bpb[^0-9]*([0-9]+\.[0-9]+)"), text)
    if erc != 0 or not m:
        raise JobError(6, "eval", f"evaluation failed: rc={erc} parsed={bool(m)}")
    causal = re.findall(r"causality_passed\W+(True|False)", text + (d / "run.log").read_text(errors="replace"))
    ev = {"public_20m": float(m[-1]), "eval_tokens": cfg.get("eval_tokens", EVAL_TOKENS),
          "eval_seq_len": cfg.get("eval_seq_len", EVAL_SEQ), "causality_passed": (causal[-1] == "True") if causal else None}
    write(d, "eval.json", ev)

    digest = sha256(ckpt)
    protected = job.get("kind") == "control" or label in cfg.get("protected_labels", [])
    uri = archive(cfg, ckpt, digest, protected)
    rec = {"schema": "trn-run-record-v2", "label": label, "arm_id": job["arm_id"], "kind": job["kind"], "host": cfg["host"],
           "timestamp_utc": now(), "base_sha256": job["base_sha256"], "train_py_sha256": job["train_sha256"],
           "git_commit": git(repo, "rev-parse", "HEAD"), "git_dirty": False, "argv": job["argv"],
           "env": {k: os.environ.get(k) for k in ("NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS", "NEURON_COMPILE_CACHE_DIR")} | cfg.get("env", {}),
           "status": "evaluated", "steps": summary.get("num_steps"), "training_seconds": summary.get("training_seconds"),
           "startup_s": summary.get("startup_allowance_est"), "over_budget": summary.get("over_budget"),
           "epoch_boundary_step": summary.get("epoch_boundary_step"), "median_step_s": summary.get("median_step_s"),
           "causality_passed": ev["causality_passed"], "eval_tokens": ev["eval_tokens"], "eval_seq_len": ev["eval_seq_len"],
           "public_20m": ev["public_20m"], "checkpoint_sha256": digest, "checkpoint_uri": uri,
           "anomalies": [f"{x}: a tagged descendant outlived the leader and was stopped by containment"
                         for x, qq in ((label, q), (eval_label, qe)) if qq.get("intervened")]}
    write(d, "record.json", rec)
    ledger = repo / "research" / "v2" / "ledger.jsonl"
    append_ledger(ledger, rec)
    verdict = decide.drift(decide.load(str(ledger))) if job["kind"] == "control" else decide.decide_arm(decide.load(str(ledger)), job["arm_id"])
    write(d, "verdict.json", verdict)
    return {"label": label, "public_20m": ev["public_20m"], "steps": rec["steps"], "verdict": verdict.get("verdict", "control"),
            "drift_alarms": verdict.get("drift_alarms"), "anomalies": rec["anomalies"], "checkpoint_uri": uri}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-config", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--repo", default=".")
    ns = ap.parse_args()
    cfg = json.loads(Path(ns.host_config).read_text())
    job = json.loads(Path(ns.job).read_text())
    repo = Path(ns.repo).resolve()
    try:
        res = run(cfg, job, repo)
        print(json.dumps(res, indent=2))
        return 0
    except JobError as e:
        d = repo / "research" / "v2" / "experiments" / job["label"]
        d.mkdir(parents=True, exist_ok=True)
        write(d, "failure.json", {"stage": e.stage, "code": e.code, "error": str(e), "utc": now()})
        print(json.dumps({"label": job["label"], "failed_stage": e.stage, "exit_code": e.code, "error": str(e)}, indent=2))
        return e.code


if __name__ == "__main__":
    sys.exit(main())
