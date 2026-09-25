#!/usr/bin/env python3
"""Synthetic r4 guard/host safety tests; no real child or signal is created."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import fresh_start_guard as guard
import fresh_start_host as host
import root_free_guard_monitor as monitor


class FakePopenChild:
    def __init__(self, *, term_ignoring=False, kill_ignoring=False):
        self.pid = 24680
        self.returncode = None
        self.actions = []
        self.term_ignoring = term_ignoring
        self.kill_ignoring = kill_ignoring

    def poll(self):
        self.actions.append(('poll', None))
        return self.returncode

    def wait(self, timeout=None):
        self.actions.append(('wait', timeout))
        if timeout is None:
            self.returncode = 0
            return 0
        if timeout == 8 and self.term_ignoring:
            raise subprocess.TimeoutExpired(['synthetic-child'], timeout)
        if timeout == 3 and self.kill_ignoring:
            raise subprocess.TimeoutExpired(['synthetic-child'], timeout)
        self.returncode = -9 if timeout == 3 else -15
        return self.returncode

    def terminate(self):
        self.actions.append(('terminate', None))

    def kill(self):
        self.actions.append(('kill', None))


class R4HostCleanupImplementationTest(unittest.TestCase):
    def invoke_guard(self, child=None, *, pidfd_error=False,
                     process_receipt_error=False, marker_candidates=(),
                     recent_candidates=(), spawn_error=False):
        with tempfile.TemporaryDirectory() as directory:
            remote = Path(directory)
            label = 'fs-synthetic-r4-guard-host'
            request_path = remote / 'request.json'
            request_path.write_text(json.dumps({
                'label': label, 'remote': str(remote),
                'expected_commit': 'synthetic-source-commit',
                'command': ['synthetic-command'],
            }))
            writes, signaled = [], []

            def write_json(path, value, replace=False):
                if process_receipt_error and str(path).endswith('.process.json'):
                    raise OSError('synthetic process receipt write failure')
                writes.append((Path(path), value, replace))

            def pidfd_open(_pid, _flags):
                if pidfd_error:
                    raise OSError('synthetic pidfd open failure')
                return 555

            def pidfd_signal(pidfd, sig, _info, _flags):
                signaled.append((pidfd, sig))

            popen_behavior = OSError('synthetic spawn failure') if spawn_error else child
            with (patch.object(guard.sys, 'argv',
                               ['fresh_start_guard.py', str(request_path), '--lock-fd', '77']),
                  patch.object(guard.subprocess, 'Popen',
                               side_effect=popen_behavior if spawn_error else None,
                               return_value=None if spawn_error else child),
                  patch.object(guard.os, 'pidfd_open', side_effect=pidfd_open),
                  patch.object(guard.os, 'close'),
                  patch.object(guard.signal, 'signal'),
                  patch.object(guard.signal, 'pidfd_send_signal', side_effect=pidfd_signal),
                  patch.object(guard.time, 'sleep'),
                  patch.object(guard, 'snapshot', return_value={}),
                  patch.object(guard, 'health_failures', return_value=[]),
                  patch.object(guard, 'write_json', side_effect=write_json),
                  patch.object(guard, 'process_record', return_value={
                      'pid': 24680, 'session': 24680, 'process_group': 24680,
                      'start_ticks': 88002,
                  }),
                  patch.object(host, 'owned_processes', return_value=list(marker_candidates)),
                  patch.object(host, 'session_processes', return_value=[]),
                  patch.object(host, 'recent_processes', return_value=list(recent_candidates)),
                  patch.object(host.os, 'kill', side_effect=AssertionError('numeric signal')),
                  patch('builtins.print')):
                result = guard.main()
            postflight = next(value for path, value, _ in writes
                              if path.name.endswith('.postflight.json'))
            return result, postflight, signaled

    def test_guard_shutdown_marker_spoof_is_inventory_only_hold(self):
        spoof = {'pid': 45679, 'start_ticks': 88005, 'session': 45679,
                 'state': 'S', 'argv': ['unrelated']}
        child = FakePopenChild()
        result, postflight, signaled = self.invoke_guard(
            child, marker_candidates=[spoof])
        self.assertEqual(result, 1)
        self.assertFalse(postflight['host_clean'])
        self.assertEqual(postflight['cleanup']['actions'], [])
        self.assertEqual(postflight['cleanup']['remaining'], [spoof])
        self.assertEqual(signaled, [])
        self.assertNotIn(('terminate', None), child.actions)
        self.assertNotIn(('kill', None), child.actions)

    def test_guard_postflight_markerless_changed_session_orphan_is_hold(self):
        orphan = {'pid': 45680, 'start_ticks': 88003, 'session': 45680,
                  'state': 'S', 'argv': ['/usr/bin/sleep', '900']}
        child = FakePopenChild()
        result, postflight, signaled = self.invoke_guard(
            child, recent_candidates=[orphan])
        self.assertEqual(result, 1)
        self.assertFalse(postflight['host_clean'])
        self.assertEqual(postflight['cleanup']['remaining'], [orphan])
        self.assertEqual(postflight['child_start_ticks'], 88002)
        self.assertEqual(signaled, [])

    def test_spawn_failure_never_signals_marker_spoof(self):
        spoof = {'pid': 45679, 'start_ticks': 88005, 'session': 45679,
                 'state': 'S', 'argv': ['unrelated']}
        result, postflight, signaled = self.invoke_guard(
            spawn_error=True, marker_candidates=[spoof])
        self.assertEqual(result, 1)
        self.assertFalse(postflight['host_clean'])
        self.assertEqual(postflight['cleanup']['actions'], [])
        self.assertEqual(postflight['cleanup']['remaining'], [spoof])
        self.assertEqual(signaled, [])

    def test_receipt_write_failure_stops_exact_pidfd_child_and_holds(self):
        child = FakePopenChild(term_ignoring=True)
        result, postflight, signaled = self.invoke_guard(
            child, process_receipt_error=True)
        self.assertEqual(result, 1)
        self.assertFalse(postflight['host_clean'])
        self.assertIn('process receipt write failure', postflight['guard_error'])
        self.assertEqual(signaled, [(555, signal.SIGTERM), (555, signal.SIGKILL)])
        self.assertEqual(postflight['exception_containment']['signals'],
                         ['SIGTERM', 'SIGKILL'])
        self.assertTrue(postflight['exception_containment']['quiescent'])
        self.assertNotIn(('terminate', None), child.actions)
        self.assertNotIn(('kill', None), child.actions)

    def test_pidfd_open_failure_uses_only_unreaped_popen_child_and_holds(self):
        child = FakePopenChild(term_ignoring=True)
        result, postflight, signaled = self.invoke_guard(child, pidfd_error=True)
        self.assertEqual(result, 1)
        self.assertFalse(postflight['host_clean'])
        self.assertIn('pidfd open failure', postflight['guard_error'])
        self.assertEqual(signaled, [])
        self.assertIn(('terminate', None), child.actions)
        self.assertIn(('kill', None), child.actions)
        self.assertEqual(postflight['exception_containment']['fallback'],
                         'unreaped_popen_direct_child')
        self.assertTrue(postflight['exception_containment']['quiescent'])

    def test_term_kill_ignoring_unreceipted_child_is_explicit_unresolved_hold(self):
        child = FakePopenChild(term_ignoring=True, kill_ignoring=True)
        result, postflight, signaled = self.invoke_guard(
            child, process_receipt_error=True)
        self.assertEqual(result, 1)
        self.assertFalse(postflight['host_clean'])
        self.assertFalse(postflight['exception_containment']['quiescent'])
        self.assertEqual(signaled, [(555, signal.SIGTERM), (555, signal.SIGKILL)])
        self.assertIn('unreceipted direct child did not quiesce after bounded stop',
                      postflight['failures'])

    def test_preflight_previous_marker_spoof_holds_without_signaling(self):
        spoof = {'pid': 45679, 'start_ticks': 88005, 'session': 45679,
                 'state': 'S', 'argv': ['unrelated']}
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'previous-state.json'
            state.write_text(json.dumps({'label': 'fs-old', 'child_session': 24680,
                                         'completed_at': 'synthetic-time'}))
            with (patch.object(host, 'HOST_STATE', state),
                  patch.object(host, 'owned_processes', return_value=[spoof]),
                  patch.object(host, 'session_processes', return_value=[]),
                  patch.object(host, 'snapshot', return_value={}),
                  patch.object(host, 'health_failures', return_value=[]),
                  patch.object(host, 'write_json'),
                  patch.object(host.os, 'kill', side_effect=AssertionError('numeric signal'))):
                with self.assertRaisesRegex(RuntimeError, 'previous run has unproven'):
                    host.preflight(Path(directory), 'fs-new')

    def test_zombie_only_session_inventory_does_not_hold(self):
        class ProcRoot:
            name = '24681'

            def stat(self):
                return SimpleNamespace(st_uid=os.getuid())

            def __truediv__(self, child):
                self_child = child
                return SimpleNamespace(read_text=lambda: (
                    '24681 (zombie) Z 1 24680 24680 0 0 0 0' if self_child == 'stat' else ''))

        class ProcDirectory:
            def iterdir(self):
                return [ProcRoot()]

        with (patch.object(monitor, 'Path', return_value=ProcDirectory()),
              patch.object(monitor.RealSystem, 'proc',
                           side_effect=AssertionError('zombie should be skipped'))):
            self.assertEqual(monitor.RealSystem.session_processes(24680), [])
        with (patch.object(host, 'Path', return_value=ProcDirectory()),
              patch.object(host, 'process_record', return_value={
                  'pid': 24681, 'start_ticks': 88001, 'session': 24680,
                  'state': 'Z', 'argv': [],
              })):
            self.assertEqual(host.session_processes(24680), [])

    def test_markerless_changed_session_process_is_in_birth_inventories(self):
        orphan_pid = 45680
        child_birth = 88002
        orphan_birth = 88003
        fields = ['S', '1', str(orphan_pid), str(orphan_pid)] + ['0'] * 15
        fields.append(str(orphan_birth))

        class ProcRoot:
            name = str(orphan_pid)

            def stat(self):
                return SimpleNamespace(st_uid=os.getuid())

            def __truediv__(self, child):
                self_child = child
                return SimpleNamespace(read_text=lambda: (
                    f'{orphan_pid} (sleep) ' + ' '.join(fields)
                    if self_child == 'stat' else ''))

        class ProcDirectory:
            def iterdir(self):
                return [ProcRoot()]

        proc = monitor.ProcInfo(
            orphan_pid, os.getuid(), orphan_birth, 1,
            orphan_pid, orphan_pid, ('/usr/bin/sleep', '900'), '/usr/bin/sleep')
        with (patch.object(monitor, 'Path', return_value=ProcDirectory()),
              patch.object(monitor.RealSystem, 'proc', return_value=proc)):
            self.assertEqual(monitor.RealSystem.recent_processes(child_birth), [proc])
        host_record = {'pid': orphan_pid, 'start_ticks': orphan_birth,
                       'session': orphan_pid, 'state': 'S',
                       'argv': ['/usr/bin/sleep', '900']}
        with (patch.object(host, 'Path', return_value=ProcDirectory()),
              patch.object(host, 'process_record', return_value=host_record)):
            self.assertEqual(host.recent_processes(child_birth), [host_record])


if __name__ == '__main__':
    unittest.main()
