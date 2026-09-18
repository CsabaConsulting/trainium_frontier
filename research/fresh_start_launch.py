#!/usr/bin/env python3
"""Launch one bounded, manifested job on an idle campaign host.

The JSON request supplies host (trn21/trn22), remote checkout, expected_commit,
label, purpose, command (argv, beginning with timeout), and explicit environment.
Set runner_owns_manifest for run_experiment.py: its log and launch manifest must
not be pre-created. This helper never stops an existing process or edits source.
Deployment and selection of an experiment remain separate decisions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess

from fresh_start_sync import HOSTS, ROOT, SSH


REMOTE_SCRIPT = r'''
import datetime, fcntl, hashlib, json, os, pathlib, shutil, socket, subprocess, sys
m = json.load(sys.stdin)
r = pathlib.Path(m['remote'])
assert r.is_dir(), 'checkout missing'
sys.path.insert(0, str(r / 'research'))
from fresh_start_host import HOST_DIRECTORY, HOST_LOCK, preflight
HOST_DIRECTORY.mkdir(parents=True, exist_ok=True)
host_lock = HOST_LOCK.open('a')
fcntl.flock(host_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
# preflight first cleans only descendants of the previously completed marked
# job, then checks all device holders. An early fuser assertion would prevent
# that authorized cleanup if a stale owned descendant still held the device.
free = shutil.disk_usage(r).free
assert free >= 8 * 1024**3, 'less than 8 GiB free'
source_paths = ['train.py', 'prepare.py', 'research/scratch', 'research/run_experiment.py',
                'research/fresh_start_host.py', 'research/fresh_start_guard.py',
                'research/fresh_start_launch.py', 'research/fresh_start_backup.py']
dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *source_paths], cwd=r)
new = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *source_paths], cwd=r)
assert not dirty.strip() and not new.strip(), 'experiment source is not committed'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r, text=True).strip()
assert commit == m['expected_commit'], 'unexpected source commit: ' + commit
fdir = r / 'research/run-logs'
manifest = fdir / (m['label'] + '.launch.json')
log = fdir / (m['label'] + '.log')
assert not manifest.exists() and not log.exists(), 'label already exists'
if m.get('runner_owns_manifest'):
    log = fdir / (m['label'] + '.driver.log')
assert not log.exists(), 'driver label already exists'
pre = preflight(r, m['label'])
paths = ['train.py', 'prepare.py', 'research/run_experiment.py',
         'research/fresh_start_host.py', 'research/fresh_start_guard.py',
         'research/fresh_start_backup.py'] + m.get('fingerprint_paths', [])
m.update(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         hostname=socket.gethostname(), git_commit=commit, source_dirty=False,
         preflight_free_bytes=free,
         source_sha256={p: hashlib.sha256((r / p).read_bytes()).hexdigest() for p in dict.fromkeys(paths)})
env = os.environ.copy()
for key in ('NEURON_RT_VISIBLE_CORES', 'NEURON_RT_NUM_CORES'):
    env.pop(key, None)
env.update(m['environment'])
if not m.get('runner_owns_manifest'):
    with manifest.open('x') as f:
        json.dump(m, f, indent=2)
job = fdir / (m['label'] + '.job.json')
with job.open('x') as f:
    json.dump(m, f, indent=2)
with log.open('x') as f:
    p = subprocess.Popen([sys.executable, str(r / 'research/fresh_start_guard.py'),
                          str(job), '--lock-fd', str(host_lock.fileno())],
                         cwd=r, env=env, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True, pass_fds=(host_lock.fileno(),))
print(json.dumps(dict(label=m['label'], pid=p.pid, commit=commit,
                     hostname=m['hostname'], log=str(log), started_at=m['started_at'],
                     protocol_version=2, preflight_passed=pre['passed'])))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert request["host"] in HOSTS
    assert re.fullmatch(r"fs-[A-Za-z0-9_.-]+", request["label"])
    assert re.fullmatch(r"[0-9a-f]{40}", request["expected_commit"])
    remote = Path(request["remote"])
    assert remote.parent == Path("/home/ubuntu") and remote.name.startswith("trainium_freshstart_")
    command = request["command"]
    assert isinstance(command, list) and command and all(isinstance(a, str) for a in command)
    assert command[0] == "timeout", "a bounded command is required"
    assert isinstance(request["environment"], dict)
    assert request["environment"].get("NEURON_LOGICAL_NC_CONFIG") in ("1", "2")
    assert "--optlevel=1" in request["environment"].get("NEURON_CC_FLAGS", "")
    receipt = ROOT / "research/run-logs" / (request["label"] + ".dispatch.json")
    with receipt.open("x") as output:
        # Preserve the intended action locally even if the remote host disappears.
        payload = dict(request=request, requested_at=datetime.now(timezone.utc).isoformat())
        json.dump(payload, output, indent=2)
        output.flush()
    completed = subprocess.run(
        [*SSH, HOSTS[request["host"]], "python3 -c " + shlex.quote(REMOTE_SCRIPT)],
        input=json.dumps(request), text=True, capture_output=True, timeout=90)
    payload.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    completed.check_returncode()


if __name__ == "__main__":
    main()
