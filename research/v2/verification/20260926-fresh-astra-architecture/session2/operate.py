#!/usr/bin/env python3
"""One fresh packet-only CLI review. No repository or credential files mounted."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
import threading
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
PACKET = BASE / 'packet'
OUTPUT = BASE / 'output'
CLI = '/usr/lib/chatgpt/resources/codex'
PROMPT = 'Read /packet/lane-spec.md. Independently review only the supplied packet and write/run your own bounded CPU reference exactness test, then leave the required report and raw test artifacts in /work. Default REFUSE until evidence is complete.'
SAFE = {'HOME': '/home/reviewer', 'CODEX_HOME': '/home/reviewer/.codex',
        'PATH': '/opt/cpu/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'PYTHONHASHSEED': '42', 'PYTHONNOUSERSITE': '1'}
DISABLED = ['multi_agent', 'multi_agent_v2', 'apps', 'plugins', 'hooks', 'memories',
            'chronicle', 'browser_use', 'browser_use_external', 'browser_use_full_cdp_access',
            'computer_use', 'in_app_browser', 'skill_search', 'workspace_dependencies',
            'shell_snapshot', 'unbounded_connection_retries']

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def manifest():
    files = []
    for path in sorted(PACKET.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Packet may not contain symlinks')
        if path.is_file() and path.name != 'INPUT_MANIFEST.json':
            data = path.read_bytes()
            files.append({'path': str(path.relative_to(PACKET)), 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()})
    result = {'scope': 'exact source diff, user lane specification, raw CPU gate receipts only',
              'model_requested': 'gpt-6-astra', 'files': files}
    write_json(PACKET / 'INPUT_MANIFEST.json', result)
    return hashlib.sha256((PACKET / 'INPUT_MANIFEST.json').read_bytes()).hexdigest()

def sandbox():
    return ['/usr/bin/bwrap', '--unshare-all', '--share-net', '--die-with-parent',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
            '--ro-bind', '/etc/ssl', '/etc/ssl', '--ro-bind', '/etc/resolv.conf', '/etc/resolv.conf',
            '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/home/reviewer/.codex',
            '--dir', '/proc/self', '--symlink', CLI, '/proc/self/exe',
            '--ro-bind', '/home/poojith-devan/venv', '/opt/cpu',
            '--ro-bind', str(PACKET), '/packet', '--bind', str(OUTPUT), '/work',
            '--chdir', '/work']

def config():
    args = ['-c', 'web_search="disabled"', '-c', 'project_doc_max_bytes=0',
            '-c', 'shell_environment_policy.inherit="none"',
            '-c', 'shell_environment_policy.set=' + '{' + ','.join(k+'='+json.dumps(v) for k,v in SAFE.items()) + '}',
            '-c', 'model_reasoning_effort="high"', '-c', 'approval_policy="never"']
    for feature in DISABLED:
        args += ['--disable', feature]
    return args + ['--enable', 'skip_host_skill_discovery']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'start'])
    args = parser.parse_args()
    packet_sha = manifest()
    if args.mode == 'preflight':
        probe = "import os,pathlib,socket; assert not pathlib.Path('/home/poojith-devan').exists(); assert not pathlib.Path('/proc/self/environ').exists(); assert not pathlib.Path('/work/.git').exists(); assert not pathlib.Path('/home/reviewer/.codex/auth.json').exists(); assert set(os.environ) <= " + repr(set(SAFE) | {'PWD', 'LC_CTYPE'}) + ", sorted(set(os.environ)); print('packet-only filesystem and clean environment verified')"
        run = subprocess.run(sandbox() + ['/opt/cpu/bin/python', '-c', probe], env=SAFE, capture_output=True, text=True, timeout=20)
        write_json(BASE/'preflight.json', {'start':now(), 'manifest_sha256':packet_sha, 'exit':run.returncode, 'stdout':run.stdout, 'stderr':run.stderr})
        if run.returncode:
            raise RuntimeError('Isolation preflight failed; see preflight.json')
        # No model request: inspect built-in + fresh user input composition.
        debug = subprocess.run(sandbox()+[CLI,'--no-daemon','debug','prompt-input']+config()+[PROMPT], env=SAFE, capture_output=True, text=True, timeout=30)
        (BASE/'prompt-input.json').write_text(debug.stdout)
        (BASE/'prompt-input.stderr').write_text(debug.stderr)
        if debug.returncode:
            raise RuntimeError('Fresh prompt preflight failed')
        print(json.dumps({'preflight':'PASS','manifest_sha256':packet_sha,'debug_exit':debug.returncode}))
        return
    if (BASE/'launch.json').exists():
        raise RuntimeError('No automatic retry or second session permitted')
    auth_path = Path('/home/poojith-devan/.codex/auth.json')
    info = auth_path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError('Auth owner/mode check failed')
    auth = json.loads(auth_path.read_text())
    key = auth.get('OPENAI_API_KEY')
    if not isinstance(key, str) or len(key) < 20 or auth.get('auth_mode') != 'apikey':
        raise RuntimeError('Expected existing API-key authentication unavailable')
    env = dict(SAFE, CODEX_API_KEY=key)
    command = sandbox()+[CLI,'--no-daemon','exec','-m','gpt-6-astra','--ephemeral',
              '--ignore-user-config','--ignore-rules','--skip-git-repo-check',
              '--json','--color','never','-C','/work','-o','/work/final-response.md',
              '--sandbox','danger-full-access']+config()+[PROMPT]
    receipt = {'started_utc': now(), 'model_requested':'gpt-6-astra', 'manifest_sha256':packet_sha,
               'cli_version':'0.155.0-alpha.16.4','command':command,
               'auth':'existing API-key loaded in memory; Codex process environment only',
               'prior_history':False,'repository_mounted':False,'wall_cap_seconds':1200,
               'tool_environment':'explicit safe allowlist; no inherited provider key',
               'proc':'no process filesystem; only fixed /proc/self/exe symlink',
               'containment':'external bubblewrap packet-only filesystem; redundant nested Codex sandbox disabled after ENOENT failure in prior distinct session',
               'network':'Codex provider network permitted; model tool network forbidden by lane specification; tools receive no credential variables'}
    started = time.monotonic()
    with (BASE/'events.jsonl').open('w') as stdout, (BASE/'stderr.log').open('w') as stderr:
        proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        receipt['operator_pid'] = os.getpid()
        receipt['child_pid'] = proc.pid
        write_json(BASE/'launch.json', receipt)
        print(json.dumps({'started_utc':receipt['started_utc'],'pid':proc.pid,'manifest_sha256':packet_sha}), flush=True)
        def pump(source, target):
            for line in source:
                target.write(line.replace(key, '[REDACTED]'))
                target.flush()
        streams = [threading.Thread(target=pump, args=(proc.stdout, stdout)), threading.Thread(target=pump, args=(proc.stderr, stderr))]
        for stream in streams:
            stream.start()
        try:
            proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            receipt['wall_cap_triggered'] = True
        for stream in streams:
            stream.join()
    receipt.update(completed_utc=now(), elapsed_seconds=time.monotonic()-started, exit_code=proc.returncode)
    write_json(BASE/'terminal.json', receipt)
    print(json.dumps({'completed_utc':receipt['completed_utc'],'exit_code':proc.returncode,'elapsed_seconds':receipt['elapsed_seconds']}))

if __name__ == '__main__':
    main()
