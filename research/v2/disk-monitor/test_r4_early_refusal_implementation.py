#!/usr/bin/env python3
"""Synthetic remote-launch refusals; no SSH, guard, or training is started."""

from __future__ import annotations

import builtins
import fcntl
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import fresh_start_host as host
import fresh_start_launch as launcher
import fresh_start_monitor_interlock as interlock
import test_paired_root_free_guard_monitor as pairfixtures


class R4EarlyRefusalImplementationTest(unittest.TestCase):
    def request(self, root, remote):
        return {
            'remote': str(remote),
            'label': 'fs-synthetic-r4-early-refusal',
            'expected_commit': 'a' * 40,
            'min_free_bytes': host.MIN_OPERATIONAL_FREE_BYTES,
            'environment': {},
            'monitor_interlock': {
                'pair_binding_path': str(root / 'pair-binding.json'),
                'no_spawn_path': str(root / 'no-spawn.json'),
                'interlock_path': str(root / 'interlock.json'),
            },
        }

    def execute_remote(self, request):
        original_path = sys.path[:]
        try:
            with patch.object(sys, 'stdin', io.StringIO(json.dumps(request))):
                exec(compile(launcher.REMOTE_SCRIPT, '<synthetic-remote-script>', 'exec'),
                     {'__name__': '__main__'})
        finally:
            sys.path[:] = original_path

    def assert_no_spawn_receipt(self, root, request, stage):
        path = root / 'no-spawn.json'
        self.assertTrue(path.exists())
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt['schema'], 'paired-monitor-no-spawn-v1')
        self.assertEqual(receipt['status'], 'NO_GUARD_SPAWNED')
        self.assertEqual(receipt['stage'], stage)
        self.assertEqual(receipt['label'], request['label'])
        self.assertEqual(receipt['expected_commit'], request['expected_commit'])
        self.assertEqual(receipt['pair_binding_path'],
                         request['monitor_interlock']['pair_binding_path'])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertTrue(path.read_bytes().endswith(b'\n'))

    def test_missing_checkout_writes_private_durable_no_spawn_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root, root / 'missing-checkout')
            with patch.object(subprocess, 'Popen',
                              side_effect=AssertionError('guard spawn reached')):
                with self.assertRaisesRegex(AssertionError, 'checkout missing'):
                    self.execute_remote(request)
            self.assert_no_spawn_receipt(root, request, 'before_checkout')

    def test_checkout_import_failure_writes_no_spawn_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / 'checkout'
            checkout.mkdir()
            request = self.request(root, checkout)
            original_import = builtins.__import__

            def fail_host_import(name, *args, **kwargs):
                if name == 'fresh_start_host':
                    raise ImportError('synthetic checkout import failure')
                return original_import(name, *args, **kwargs)

            with (patch('builtins.__import__', side_effect=fail_host_import),
                  patch.object(subprocess, 'Popen',
                               side_effect=AssertionError('guard spawn reached'))):
                with self.assertRaisesRegex(ImportError, 'checkout import failure'):
                    self.execute_remote(request)
            self.assert_no_spawn_receipt(root, request, 'checkout_import')

    def test_bootstrap_receipt_is_accepted_by_paired_monitor(self):
        case = pairfixtures.PairMonitorTest(
            methodName='test_no_spawn_receipt_ends_pre_guard_watch_without_signal')
        case.setUp()
        self.addCleanup(case.doCleanups)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root, root / 'missing-checkout')
            request['label'] = case.binding['label']
            request['expected_commit'] = case.binding['expected_commit']
            request['monitor_interlock']['pair_binding_path'] = str(case.pair_path)
            request['monitor_interlock']['no_spawn_path'] = case.pair['no_spawn_path']
            with patch.object(subprocess, 'Popen',
                              side_effect=AssertionError('guard spawn reached')):
                with self.assertRaisesRegex(AssertionError, 'checkout missing'):
                    self.execute_remote(request)
            receipt_path = Path(case.pair['no_spawn_path'])
            case.system.files[receipt_path] = {}
            self.assertTrue(case.monitor._no_spawn_receipt())

    def test_log_open_failure_before_popen_writes_no_spawn_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / 'checkout'
            logdir = checkout / 'research/run-logs'
            logdir.mkdir(parents=True)
            request = self.request(root, checkout)
            paths = [
                'train.py', 'prepare.py', 'research/run_experiment.py',
                'research/fresh_start_host.py', 'research/fresh_start_guard.py',
                'research/fresh_start_backup.py',
                'research/fresh_start_monitor_interlock.py',
            ]
            for relative in paths:
                target = checkout / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('synthetic source\n')
            hostdir = root / 'host-state'
            hostlock = hostdir / 'device0.lock'
            log_path = logdir / (request['label'] + '.log')
            original_open = Path.open

            def fail_log_open(path, mode='r', *args, **kwargs):
                if path == log_path and mode == 'x':
                    raise OSError('synthetic log open failure')
                return original_open(path, mode, *args, **kwargs)

            def git_output(argv, **_kwargs):
                if argv[:3] == ['git', 'rev-parse', 'HEAD']:
                    return request['expected_commit']
                return b''

            with (patch.object(host, 'HOST_DIRECTORY', hostdir),
                  patch.object(host, 'HOST_LOCK', hostlock),
                  patch.object(host, 'preflight', return_value={'passed': True}),
                  patch.object(interlock, 'require_live_pair', return_value={}),
                  patch.object(interlock, 'write_interlock_pass'),
                  patch.object(fcntl, 'flock'),
                  patch.object(shutil, 'disk_usage',
                               return_value=SimpleNamespace(free=11 * 1024**3)),
                  patch.object(subprocess, 'check_output', side_effect=git_output),
                  patch.object(subprocess, 'Popen',
                               side_effect=AssertionError('guard spawn reached')),
                  patch.object(Path, 'open', fail_log_open)):
                with self.assertRaisesRegex(OSError, 'log open failure'):
                    self.execute_remote(request)
            self.assert_no_spawn_receipt(root, request, 'before_guard_spawn')


if __name__ == '__main__':
    unittest.main()
