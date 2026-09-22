#!/usr/bin/env python3
"""Remote detached launcher (copied to the Trn2 host by the operator; stdlib only, py3.8+).

Usage on the host:  <native_venv python> remote_launcher.py <launch.json>

launch.json: {"cwd": str, "argv": [str], "env": {str: str}, "stdout": str, "pidfile": str,
              "donefile": str, "expect_cwd_head": str}

Behaviour:
  * refuses to run if cwd is not a git checkout at expect_cwd_head (fail closed);
  * refuses if pidfile or donefile already exist (never relaunch a label);
  * double-forks so the training process survives the SSH session ending;
  * the grandchild writes {"returncode", "finished_at"} to donefile when the runner exits;
  * prints {"pid": <runner pid>} once and exits 0.

It never deletes anything and never composes a shell string.
"""
import json
import os
import subprocess
import sys
import time


def _fail(msg):
    sys.stdout.write(json.dumps({"error": msg}) + "\n")
    sys.stdout.flush()
    sys.exit(2)


def main():
    if len(sys.argv) != 2:
        _fail("usage: remote_launcher.py <launch.json>")
    with open(sys.argv[1]) as fh:
        spec = json.load(fh)
    for key in ("cwd", "argv", "env", "stdout", "pidfile", "donefile", "expect_cwd_head"):
        if key not in spec:
            _fail("launch spec missing %s" % key)
    cwd = spec["cwd"]
    if not os.path.isdir(os.path.join(cwd, ".git")):
        _fail("cwd is not a git checkout")
    head = subprocess.run(["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != spec["expect_cwd_head"]:
        _fail("cwd HEAD mismatch")
    porcelain = subprocess.run(["git", "-C", cwd, "status", "--porcelain"], capture_output=True, text=True)
    if porcelain.returncode != 0 or porcelain.stdout.strip():
        _fail("cwd is dirty")
    if os.path.exists(spec["pidfile"]) or os.path.exists(spec["donefile"]):
        _fail("pidfile/donefile already exist; refusing to relaunch")
    argv = spec["argv"]
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        _fail("argv must be a list of strings")
    if "--no-commit" not in argv:
        _fail("argv lacks --no-commit")

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in spec["env"].items()})
    if "--optlevel=1" not in env.get("NEURON_CC_FLAGS", "").split():
        _fail("NEURON_CC_FLAGS lacks --optlevel=1")

    os.makedirs(os.path.dirname(spec["stdout"]), exist_ok=True)

    # First fork: parent returns the runner pid to the SSH caller.
    r, w = os.pipe()
    pid = os.fork()
    if pid > 0:
        os.close(w)
        data = os.read(r, 64)
        os.close(r)
        os.waitpid(pid, 0)
        try:
            runner_pid = int(data.decode().strip())
        except ValueError:
            _fail("supervisor did not report a pid")
        sys.stdout.write(json.dumps({"pid": runner_pid}) + "\n")
        sys.stdout.flush()
        return
    # Child: detach, then supervise.
    os.close(r)
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)
    # Grandchild (supervisor): start the runner, report its pid, wait, write donefile.
    with open(spec["stdout"], "ab", buffering=0) as out:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        with open(spec["pidfile"], "w") as fh:
            fh.write("%d\n" % proc.pid)
        os.write(w, ("%d\n" % proc.pid).encode())
        os.close(w)
        rc = proc.wait()
    tmp = spec["donefile"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"returncode": rc, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh)
    os.replace(tmp, spec["donefile"])
    os._exit(0)


if __name__ == "__main__":
    main()
