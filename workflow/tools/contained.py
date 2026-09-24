#!/usr/bin/env python3
"""Process containment for training/eval jobs on a Trn2 host (run it ON the host).

Why: a direct training process can exit while a descendant survives and later mutates evidence.
Process groups alone do not catch a descendant that calls setsid(). This tool tags every process
it launches with an environment marker (TRN_EXP_LABEL), which all descendants inherit even after
setsid/double-fork, and scans /proc for it. If `systemd-run` is available it also puts the job in
its own transient scope (a cgroup, which unprivileged processes cannot leave).

  contained.py launch  --label L --dir D [--cgroup] -- <command...>   # detached; writes D/launch.json
  contained.py status  --label L                                      # live tagged processes
  contained.py wait    --label L --dir D [--timeout S]                # until the job leader exits
  contained.py stop    --label L --dir D [--grace 60]                 # TERM -> wait -> KILL -> verify
  contained.py quiesce --label L --dir D --files F [F ...]            # prove quiescence, then hash

`quiesce` writes D/quiescence.json and exits 0 only if: no process carries the label, no process
holds /dev/neuron*, and every listed file has identical size+mtime across a 10 s window. Hash the
evidence ONLY after it succeeds. Exit 3 = survivors or holders (open a breaker; never weaken this).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

TAG = "TRN_EXP_LABEL"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tagged_pids(label: str) -> list[int]:
    needle = f"{TAG}={label}".encode()
    me = os.getpid()
    out = []
    for env_path in glob.glob("/proc/[0-9]*/environ"):
        pid = int(env_path.split("/")[2])
        if pid == me:
            continue
        try:
            data = Path(env_path).read_bytes()
        except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
            continue
        if needle in data.split(b"\x00"):
            try:
                state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
            except OSError:
                continue
            if state != "Z":  # zombies hold no resources; their parent reaps them
                out.append(pid)
    return sorted(out)


def describe(pids: list[int]) -> list[dict]:
    rows = []
    for pid in pids:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")[:200]
            stat = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
            rows.append({"pid": pid, "ppid": int(stat[1]), "pgid": int(stat[2]), "sid": int(stat[3]), "cmd": cmd})
        except OSError:
            pass
    return rows


def neuron_holders() -> list[str]:
    devs = glob.glob("/dev/neuron*")
    if not devs:
        return []
    fuser = shutil.which("fuser")
    if fuser:
        p = subprocess.run([fuser] + devs, capture_output=True, text=True)
        pids = (p.stdout + " " + p.stderr).split()
        return [x for x in pids if x.rstrip("mce").isdigit()]
    holders = []
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd).startswith("/dev/neuron"):
                holders.append(fd.split("/")[2])
        except OSError:
            continue
    return sorted(set(holders))


def cmd_launch(ns, command):
    d = Path(ns.dir)
    d.mkdir(parents=True, exist_ok=True)
    if tagged_pids(ns.label):
        print(f"refusing: processes already tagged {ns.label}", file=sys.stderr)
        return 3
    env = dict(os.environ, **{TAG: ns.label})
    argv = list(command)
    cgroup = None
    if ns.cgroup and shutil.which("systemd-run"):
        cgroup = f"exp-{ns.label}"
        argv = ["systemd-run", "--user", "--scope", f"--unit={cgroup}", "--collect", "--quiet"] + argv
    log = open(d / "run.log", "ab")
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, cwd=ns.cwd or None)
    rec = {"label": ns.label, "started_utc": now(), "leader_pid": proc.pid, "pgid": proc.pid, "sid": proc.pid,
           "cgroup_scope": cgroup, "argv": list(command), "cwd": ns.cwd or os.getcwd(), "tag": f"{TAG}={ns.label}"}
    (d / "launch.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec))
    return 0


def cmd_status(ns):
    rows = describe(tagged_pids(ns.label))
    print(json.dumps({"label": ns.label, "live": rows, "neuron_holders": neuron_holders()}, indent=2))
    return 0 if rows else 1


def _leader(ns):
    return json.loads((Path(ns.dir) / "launch.json").read_text())["leader_pid"]


def cmd_wait(ns):
    pid = _leader(ns)
    t0 = time.time()
    while Path(f"/proc/{pid}").exists():
        try:
            st = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
            if st == "Z":
                break
        except OSError:
            break
        if ns.timeout and time.time() - t0 > ns.timeout:
            return 124
        time.sleep(2)
    return 0


def _signal_all(label, pgid, sig):
    for pid in tagged_pids(label):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    if pgid:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def cmd_stop(ns):
    d = Path(ns.dir)
    rec = json.loads((d / "launch.json").read_text()) if (d / "launch.json").exists() else {}
    pgid, scope = rec.get("pgid"), rec.get("cgroup_scope")
    steps = []
    if scope and shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGTERM", f"{scope}.scope"], capture_output=True)
    _signal_all(ns.label, pgid, signal.SIGTERM)
    steps.append({"t": now(), "action": "SIGTERM", "targets": tagged_pids(ns.label)})
    deadline = time.time() + ns.grace
    while tagged_pids(ns.label) and time.time() < deadline:
        time.sleep(1)
    if tagged_pids(ns.label):
        if scope and shutil.which("systemctl"):
            subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", f"{scope}.scope"], capture_output=True)
        _signal_all(ns.label, pgid, signal.SIGKILL)
        steps.append({"t": now(), "action": "SIGKILL", "targets": tagged_pids(ns.label)})
        deadline = time.time() + 120
        while tagged_pids(ns.label) and time.time() < deadline:
            time.sleep(1)
    survivors = describe(tagged_pids(ns.label))
    res = {"label": ns.label, "stopped_utc": now(), "steps": steps, "survivors": survivors}
    (d / "stop.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 3 if survivors else 0


def stat_sig(paths):
    sig = {}
    for p in paths:
        try:
            st = os.stat(p)
            sig[p] = [st.st_size, st.st_mtime_ns]
        except FileNotFoundError:
            sig[p] = None
    return sig


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_quiesce(ns):
    d = Path(ns.dir)
    live = describe(tagged_pids(ns.label))
    holders = neuron_holders()
    s1 = stat_sig(ns.files)
    time.sleep(ns.window)
    live2 = describe(tagged_pids(ns.label))
    holders2 = neuron_holders()
    s2 = stat_sig(ns.files)
    stable = s1 == s2 and all(v is not None for v in s2.values())
    ok = not live and not live2 and not holders and not holders2 and stable
    res = {"label": ns.label, "checked_utc": now(), "window_s": ns.window, "tagged_processes": live + live2,
           "neuron_holders": sorted(set(holders + holders2)), "files_stable": stable, "file_stat": s2,
           "quiescent": ok, "sha256": {p: sha256(p) for p in ns.files if Path(p).exists()} if ok else {}}
    (d / "quiescence.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if ok else 3


def main() -> int:
    argv = sys.argv[1:]
    command = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["launch", "status", "wait", "stop", "quiesce"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--cwd", default="")
    ap.add_argument("--cgroup", action="store_true")
    ap.add_argument("--grace", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--files", nargs="*", default=[])
    ns = ap.parse_args(argv)
    if ns.action == "launch":
        if not command:
            ap.error("launch needs -- <command>")
        return cmd_launch(ns, command)
    return {"status": cmd_status, "wait": cmd_wait, "stop": cmd_stop, "quiesce": cmd_quiesce}[ns.action](ns)


if __name__ == "__main__":
    sys.exit(main())
