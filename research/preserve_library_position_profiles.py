#!/usr/bin/env python3
"""Preserve the four fixed source197 artifacts, then exit.

This local sidecar launches no experiment, changes no cron or Git history, and
deletes no remote data. It waits for the finite sequence to finish, rechecks
every frozen completion requirement, retrieves and hashes each artifact, and
uses the existing locked uploader to verify its backup in the user's fork.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fresh_start_backup import ROOT, atomic_json, desktop_keyring_environment, now
import fresh_start_artifact_backup as artifacts
from fresh_start_sequence import read_requests, require_completed
from fresh_start_sequence_start import require_committed_files


LABEL = 'fs-w197-profile-preservation-sidecar-r1'
SEQUENCE = 'fs-w197-nkilib-partial-profile-sequence-r2'
PLAN_SHA256 = '9ca7cdbf49b31f894ddfd54c99eb59f1a38864b912717a35f0c9c832a60da470'
ORIGINAL_SEQUENCE = 'fs-w197-nkilib-partial-profile-sequence-r1'
ORIGINAL_PLAN_SHA256 = '87f89c86678053c6604b6e502d9d5e5bb1131c29d31c24450dffb1c12bc65699'
SOURCE = '282935ca9b1e30f6c21c397143770b03034363f9'


def checked_plan():
    path = ROOT / 'research/run-logs' / (SEQUENCE + '.plan.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PLAN_SHA256
    plan = json.loads(path.read_text())
    assert len(read_requests(plan)) == 3
    original_path = ROOT / 'research/run-logs' / (ORIGINAL_SEQUENCE + '.plan.json')
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == ORIGINAL_PLAN_SHA256
    original_plan = json.loads(original_path.read_text())
    assert len(read_requests(original_plan)) == 4
    # The first raw capture was already fully clean. Its first ingestion failed
    # only the storage reserve and is retained separately; never waive that gate.
    # The continuation repeats ingestion with its own clean completion record.
    plan = dict(plan, steps=[original_plan['steps'][0], *plan['steps']])
    requests = read_requests(plan)
    assert len(requests) == 4
    for request in requests:
        assert request['host'] == 'trn22' and request['expected_commit'] == SOURCE
        assert request['remote'] == '/home/ubuntu/trainium_freshstart_20260918_wave195'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require_committed_files(ROOT, commit, [Path(__file__).resolve(), path, original_path,
        *[ROOT / step['request'] for step in plan['steps']]])
    return plan, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan, requests = checked_plan()
    if args.check_only:
        print(json.dumps(dict(passed=True, plan_sha256=PLAN_SHA256, artifacts=4,
                              read_only_check=True)), flush=True)
        return
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    directory = ROOT / 'out/fresh-start-sequences'
    directory.mkdir(parents=True, exist_ok=True)
    ownership = (directory / (LABEL + '.lock')).open('a')
    fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = ROOT / 'research/run-logs'
    state_path = logs / (LABEL + '.json')
    state = dict(label=LABEL, pid=os.getpid(), started_at=now(), status='waiting',
                 sequence=SEQUENCE, plan_sha256=PLAN_SHA256, helper_sha256=own_hash,
                 max_seconds=11000, completed=[], remote_deletion=False,
                 experiment_dispatch=False, cron_changed=False)
    with state_path.open('x') as stream:
        stream.write(json.dumps(state, indent=2) + '\n')
    deadline = time.monotonic() + 11000
    try:
        # Waiting for all four avoids profile transfers competing with the
        # remaining model timing. The first raw capture may already be local.
        while True:
            assert time.monotonic() < deadline, 'Preservation deadline'
            assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_hash
            current = json.loads((logs / (SEQUENCE + '.sequence.json')).read_text())
            assert current['plan_sha256'] == PLAN_SHA256
            assert current['status'] != 'stopped', current
            if current['status'] == 'complete':
                assert len(current['completed']) == 3
                break
            time.sleep(45)
        desktop_keyring_environment()
        for step, request in zip(plan['steps'], requests):
            label = request['label']
            assert time.monotonic() < deadline, 'Preservation deadline'
            checked_plan()
            post = json.loads((logs / (label + '.postflight.json')).read_text())
            raw = (logs / (label + '.log')).read_text()
            require_completed(post, raw, request, step['requirements'])
            suffix = '-parquet' if '-ingest-' in label else ''
            artifact_id = label + suffix
            destination = 'out/fresh-start-profiles/' + artifact_id
            state.update(status='preserving', current=artifact_id, updated_at=now())
            atomic_json(state_path, state)
            subprocess.run([sys.executable, str(ROOT / 'research/fresh_start_preserve_directory.py'),
                '--host', request['host'], '--remote', request['remote'], '--label', label,
                '--commit', SOURCE, '--relative', 'out/' + artifact_id,
                '--destination', destination], cwd=ROOT, check=True,
                timeout=min(1100, max(1, deadline - time.monotonic())))
            entry = dict(id=artifact_id, source=destination,
                         kind='derived_profile' if suffix else 'raw_profile',
                         preservation_receipt='research/run-logs/' + label + '.preservation.json')
            artifacts.DIRECTORY.mkdir(parents=True, exist_ok=True)
            with (artifacts.DIRECTORY / 'backup.lock').open('a') as cloud_lock:
                while True:
                    assert time.monotonic() < deadline, 'Cloud backup lock deadline'
                    try:
                        fcntl.flock(cloud_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        state.update(status='waiting_for_uploader', current=artifact_id, updated_at=now())
                        atomic_json(state_path, state)
                        time.sleep(45)
                config = json.loads(artifacts.CONFIG.read_text())
                assert config['repository'] == 'CsabaConsulting/trainium_frontier'
                state.update(status='uploading', current=artifact_id, updated_at=now())
                atomic_json(state_path, state)
                folder, manifest = artifacts.prepare(entry)
                result = artifacts.upload(config, artifacts.release(config), folder, manifest)
                assert result['all_github_digests_match']
            state['completed'].append(dict(artifact=artifact_id,
                verified_at=now(), all_local_and_cloud_hashes_verified=True,
                payload_sha256=result['payload_sha256']))
            atomic_json(state_path, state)
        state.update(status='complete', current=None, completed_at=now())
    except Exception as error:
        state.update(status='stopped', error=str(error), stopped_at=now())
        atomic_json(state_path, state)
        raise
    atomic_json(state_path, state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
