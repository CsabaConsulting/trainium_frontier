"""Local-only factual connector receipt; SSH and remote work are prohibited."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
CONNECTOR = Path('/home/poojith-devan/Desktop/Trainium/research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py')
CONNECTOR_SHA = '7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899'
REMOTE_ARGV = ['/usr/bin/python3', '-I', '-B', '-']
TRANSPORT_SHA = 'fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    require(sha(CONNECTOR.read_bytes()) == CONNECTOR_SHA, 'CONNECTOR_PIN')
    spec = importlib.util.spec_from_file_location('pinned_connection_receipt', CONNECTOR)
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    original_run = subprocess.run
    calls = []

    def local_public_key_check_only(argv, *args, **kwargs):
        # The unchanged connector uses only these local public host-key checks.
        # Refuse SSH (or any other child), rather than relying on caller intent.
        check_entry = len(argv) == 5 and argv[:2] == ['ssh-keygen', '-F'] and argv[3] == '-f'
        check_fingerprint = argv == ['ssh-keygen', '-l', '-E', 'sha256', '-f', '-']
        require(check_entry or check_fingerprint, 'REMOTE_OR_UNEXPECTED_CHILD_FORBIDDEN')
        calls.append('public-host-entry' if check_entry else 'public-host-fingerprint')
        return original_run(argv, *args, **kwargs)

    subprocess.run = local_public_key_check_only
    try:
        argv, display, identity, transport_sha = connector.load_transport('trn21')
    finally:
        subprocess.run = original_run
    require(transport_sha == TRANSPORT_SHA, 'NESTED_CONNECTOR_PIN')
    require(set(identity) == {'known_hosts_sha256', 'host_fingerprint', 'host_key_algorithm'}, 'PUBLIC_IDENTITY_ONLY')
    remote_command = 'exec ' + shlex.join(REMOTE_ARGV)
    argv[-1] = remote_command
    display[-1] = remote_command
    require(shlex.split(remote_command) == ['exec', *REMOTE_ARGV], 'REMOTE_ARGV_ROUNDTRIP')
    credential_index = argv.index('-i') + 1
    require(display[credential_index] == '<private-key-path-redacted>', 'IDENTITY_PATH_REDACTED')
    require(all(a == b for i, (a, b) in enumerate(zip(argv, display)) if i != credential_index), 'ONLY_CREDENTIAL_PATH_REDACTED')
    require(len(argv) == len(display), 'ARGV_LENGTH')
    options = [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == '-o']
    receipt = {
        'schema': 'clean-native-trn21-local-connection-facts-v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host': 'trn21', 'expected_hostname': 'ip-172-31-34-85',
        'endpoint': argv[-2],
        'connector_source_path': str(CONNECTOR),
        'connector_source_sha256': CONNECTOR_SHA,
        'nested_connector_source_sha256': transport_sha,
        'public_host_identity': identity,
        'safe_actual_ssh_argv': display,
        'actual_ssh_argv_json_sha256': sha(json.dumps(argv, separators=(',', ':')).encode()),
        'ssh_options': options,
        'remote_argv': REMOTE_ARGV,
        'remote_shell_command': remote_command,
        'credential_argument_index': credential_index,
        'credential_resolution': 'Only the unchanged pinned connector supplies the omitted identity-file pathname at execution time; key contents are never read or archived by this receipt preparation.',
        'local_public_key_checks': calls,
        'ssh_invocations': 0, 'remote_calls': 0,
        'credential_contents_read': False, 'release_created': False,
        'stage_authorized': False, 'launch_authorized': False,
        'retrieval_authorized': False, 'independent_acceptance': False,
        'scope': 'Local factual preparation only. Fresh connector authentication, independent connection review, exact Supervisor assignment, controller and host gates remain required before any future use.'
    }
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
    require(connector.SENSITIVE.search(raw) is None, 'SENSITIVE_RECEIPT_REFUSAL')
    # Explicit negative mechanics test: the local boundary may not execute SSH.
    try:
        local_public_key_check_only(['ssh', 'example.invalid'])
    except RuntimeError as error:
        require(str(error) == 'REMOTE_OR_UNEXPECTED_CHILD_FORBIDDEN', 'WRONG_NEGATIVE_TEST')
    else:
        raise RuntimeError('SSH_NEGATIVE_TEST_FAILED')
    sys.stdout.buffer.write(raw)


if __name__ == '__main__':
    main()
