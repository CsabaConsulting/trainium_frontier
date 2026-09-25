"""Final r2 trn21 transport; validates the late science-label gate.

This module only builds argv and validates offline receipts. It never opens
SSH itself. The original static-reviewed transport stays byte-identical.
"""

import datetime
import hashlib
import importlib.util
from pathlib import Path
import shlex

import stage_r4_transport as base


HERE = Path(__file__).resolve().parent
REMOTE = HERE / "stage_r4_remote_final_r2.py"
REMOTE_SHA256 = "6b1cfe6b0b0d178a48bc881c389a2b2c3949736d7e9457c2aa4fa840581803c7"
SOURCE = base.SOURCE
SOURCE_BYTES = base.SOURCE_BYTES
SOURCE_SHA256 = base.SOURCE_SHA256
TARGET = base.TARGET
ENDPOINT = base.ENDPOINT
IDENTITY_FILE = base.IDENTITY_FILE
KNOWN_HOSTS = base.KNOWN_HOSTS
KNOWN_HOSTS_SHA256 = base.KNOWN_HOSTS_SHA256
HOST_FINGERPRINT = base.HOST_FINGERPRINT
PRE_FREE = base.PRE_FREE
POST_FREE = base.POST_FREE
MIN_INODES = base.MIN_INODES
checked_local_source = base.checked_local_source
checked_ssh_identity = base.checked_ssh_identity
validate_probe = base.validate_probe
compare_postflight = base.compare_postflight
argv_sha256 = base.argv_sha256


def reviewed_remote_source():
    data = REMOTE.read_bytes()
    if base.sha256(data) != REMOTE_SHA256:
        raise RuntimeError("final remote source hash drift")
    return data.decode("utf-8")


def remote_constants():
    reviewed_remote_source()
    spec = importlib.util.spec_from_file_location("trn21_r4_remote_final_exact", REMOTE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ssh_argv(operation, *, preflight=None):
    if operation not in ("probe", "write", "verify"):
        raise ValueError("only fixed probe/write/verify operations exist")
    if operation == "write":
        if preflight is None:
            raise RuntimeError("writer requires a validated fresh preflight identity")
        validate_probe(preflight, before_write=True)
        pin = remote_constants()._identity_digest(preflight["identity_after"])
    elif preflight is not None:
        raise RuntimeError("preflight identity may only bind the single write")
    else:
        pin = None
    checked_ssh_identity()
    source = reviewed_remote_source()
    words = ["exec", "python3", "-c", source, operation]
    if pin is not None:
        words.append(pin)
    command = " ".join(shlex.quote(word) for word in words)
    if shlex.split(command) != words:
        raise RuntimeError("remote command quoting drift")
    return ["ssh", "-F", "/dev/null", "-T", "-i", IDENTITY_FILE,
            "-o", "UserKnownHostsFile=" + str(KNOWN_HOSTS),
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "HostKeyAlgorithms=ssh-ed25519",
            "-o", "UpdateHostKeys=no", "-o", "VerifyHostKeyDNS=no",
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=20", ENDPOINT, command]


def compare_preflight_to_write(pre, write):
    """Authenticate both sides of the final label scan before O_EXCL."""
    base.compare_preflight_to_write(pre, write)
    first, final = write.get("immediate_gate"), write.get("final_creation_gate")
    if not isinstance(first, dict) or not isinstance(final, dict):
        raise RuntimeError("final pre-O_EXCL STOP/process/Neuron gate is missing")
    label = final.get("science_label_scan")
    if (not isinstance(label, dict) or
            set(label) != {"rc", "matches", "match_count", "truncated",
                           "stderr_empty", "stdout_sha256", "started_utc",
                           "completed_utc"} or
            label.get("rc") != 0 or label.get("matches") != [] or
            label.get("match_count") != 0 or
            label.get("truncated") is not False or
            label.get("stderr_empty") is not True or
            label.get("stdout_sha256") != hashlib.sha256(b"").hexdigest()):
        raise RuntimeError("final pre-O_EXCL science label scan evidence is missing or dirty")
    pre_label = final.get("pre_label_gate")
    if not isinstance(pre_label, dict):
        raise RuntimeError("final pre-O_EXCL label scan lacks its preceding activity gate")
    try:
        first_at = datetime.datetime.fromisoformat(first["utc"])
        pre_label_at = datetime.datetime.fromisoformat(pre_label["utc"])
        label_started_at = datetime.datetime.fromisoformat(label["started_utc"])
        label_completed_at = datetime.datetime.fromisoformat(label["completed_utc"])
        final_at = datetime.datetime.fromisoformat(final["utc"])
        write_at = datetime.datetime.fromisoformat(write["utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("final pre-O_EXCL label/gate time evidence is malformed") from error
    if (any(item.tzinfo is None for item in (first_at, pre_label_at,
            label_started_at, label_completed_at, final_at, write_at)) or
            not (first_at < pre_label_at <= label_started_at <=
                 label_completed_at <= final_at < write_at)):
        raise RuntimeError("final pre-O_EXCL label scan/gate order is not proven")
    # The disabled validator authenticates each entire STOP/process/target/
    # holder/lock/Neuron record, both before and after the potentially slow
    # find. The last one is the actual final creation gate.
    for gate in (pre_label, final):
        gate_view = dict(write)
        gate_view["immediate_gate"] = gate
        base.compare_preflight_to_write(pre, gate_view)
    return {"status": "PREFLIGHT_TO_FINAL_CREATION_GATE_PASS",
            "preflight_identity_sha256": write["preflight_identity_sha256"],
            "first_gate_utc": first["utc"],
            "pre_label_gate_utc": pre_label["utc"],
            "label_started_utc": label["started_utc"],
            "label_completed_utc": label["completed_utc"],
            "label_stdout_sha256": label["stdout_sha256"],
            "final_gate_utc": final["utc"]}


def compare_sessions(pre, write, verify):
    compare_preflight_to_write(pre, write)
    return base.compare_sessions(pre, write, verify)


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: this module only builds/reviews commands; it never opens SSH")
