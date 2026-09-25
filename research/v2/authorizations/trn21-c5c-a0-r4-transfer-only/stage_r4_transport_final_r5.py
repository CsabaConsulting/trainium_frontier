"""Disabled final-r5 trn21 transport and offline receipt validators.

This module only builds argv and validates offline receipts. It never opens
SSH itself. The held final-r4 transport and remote stay byte-identical.
Final-r5 adds the missing terminal free-inode floor to every full probe.
"""

import datetime
import hashlib
import importlib.util
from pathlib import Path
import shlex

import stage_r4_transport as base


HERE = Path(__file__).resolve().parent
REMOTE = HERE / "stage_r4_remote_final_r4.py"
REMOTE_SHA256 = "ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b"
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
argv_sha256 = base.argv_sha256


def reviewed_remote_source():
    data = REMOTE.read_bytes()
    if base.sha256(data) != REMOTE_SHA256:
        raise RuntimeError("final remote source hash drift")
    return data.decode("utf-8")


def remote_constants():
    reviewed_remote_source()
    spec = importlib.util.spec_from_file_location("trn21_r4_remote_final_r5_reused_exact", REMOTE)
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
        remote = remote_constants()
        pin = remote._identity_digest(preflight["identity_after"])
        lock_pin = remote._lock_identity_digest(preflight["host_lock_identity"])
    elif preflight is not None:
        raise RuntimeError("preflight identity may only bind the single write")
    else:
        pin = None
        lock_pin = None
    checked_ssh_identity()
    source = reviewed_remote_source()
    words = ["exec", "python3", "-c", source, operation]
    if pin is not None:
        words.extend((pin, lock_pin))
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


def _time(value, purpose):
    try:
        instant = datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(purpose + " UTC is missing or malformed") from error
    if instant.tzinfo is None:
        raise RuntimeError(purpose + " UTC lacks an offset")
    return instant


def _require_lock_identity(identity, remote):
    if (not isinstance(identity, dict) or
            set(identity) != {"path", "ubuntu", "cache", "host_directory", "file",
                              "trust"} or
            identity.get("path") != remote.LOCK_PATH):
        raise RuntimeError("existing host lock identity is missing or misbound")
    parent = identity["ubuntu"]
    if not isinstance(parent, dict) or parent.get("device") != 66305:
        raise RuntimeError("existing host lock parent is not the pinned root device")
    modes = {}
    for key in ("ubuntu", "cache", "host_directory"):
        item = identity[key]
        if (not isinstance(item, dict) or
                set(item) != {"type", "device", "inode", "uid", "gid", "mode"} or
                item["type"] != "directory" or item["uid"] != remote.UID or
                item["device"] != parent["device"] or
                type(item["inode"]) is not int or item["inode"] <= 0):
            raise RuntimeError("existing host lock directory identity is unsafe")
        try:
            modes[key] = int(item["mode"], 8)
        except (TypeError, ValueError) as error:
            raise RuntimeError("existing host lock directory mode is malformed") from error
    item = identity["file"]
    if (not isinstance(item, dict) or
            set(item) != {"exists", "type", "device", "inode", "uid", "gid",
                          "mode", "nlink", "bytes"} or
            item["exists"] is not True or item["type"] != "regular" or
            item["device"] != parent["device"] or item["uid"] != remote.UID or
            item["nlink"] != 1 or type(item["inode"]) is not int or
            item["inode"] <= 0 or type(item["bytes"]) is not int or
            item["bytes"] < 0):
        raise RuntimeError("existing host lock file identity is unsafe")
    try:
        modes["file"] = int(item["mode"], 8)
    except (TypeError, ValueError) as error:
        raise RuntimeError("existing host lock file mode is malformed") from error
    if modes["ubuntu"] & 0o022 or modes["file"] & 0o600 != 0o600:
        raise RuntimeError("existing host lock file mode is unsafe")
    trust = identity["trust"]
    if (not isinstance(trust, dict) or
            set(trust) != {"policy", "acl", "mounts", "cache_nlink",
                           "host_directory_nlink"} or
            type(trust["cache_nlink"]) is not int or trust["cache_nlink"] < 1 or
            type(trust["host_directory_nlink"]) is not int or
            trust["host_directory_nlink"] < 1):
        raise RuntimeError("host lock private-ancestry proof is missing")
    expected_acl = {name: {"system.posix_acl_access": "absent_ENODATA",
                           "system.posix_acl_default": "absent_ENODATA"}
                    for name in ("cache", "host_directory")}
    expected_acl["file"] = {"system.posix_acl_access": "absent_ENODATA"}
    if trust["acl"] != expected_acl:
        raise RuntimeError("host lock POSIX ACL absence is unproven")
    mounts = trust["mounts"]
    required_mounts = {"mount_namespace", "mountinfo_sha256", "entry_count",
                       "root_mount_id", "root_mount_point", "root_mount_root",
                       "root_major_minor", "root_filesystem", "root_source",
                       "unsafe_aliases"}
    if (not isinstance(mounts, dict) or set(mounts) != required_mounts or
            not isinstance(mounts["mount_namespace"], str) or
            not isinstance(mounts["mountinfo_sha256"], str) or
            len(mounts["mountinfo_sha256"]) != 64 or
            any(letter not in "0123456789abcdef"
                for letter in mounts["mountinfo_sha256"]) or
            type(mounts["entry_count"]) is not int or mounts["entry_count"] < 1 or
            not isinstance(mounts["root_mount_id"], str) or
            mounts["root_mount_point"] != "/" or
            mounts["root_mount_root"] != "/" or
            mounts["root_major_minor"] != remote.MOUNT_MAJOR_MINOR or
            mounts["root_filesystem"] != "ext4" or
            mounts["root_source"] != remote.MOUNT_SOURCE or
            mounts["unsafe_aliases"] != []):
        raise RuntimeError("host lock mount-path proof is missing or unsafe")
    strict = all(modes[name] & 0o022 == 0
                 for name in ("cache", "host_directory", "file"))
    if strict:
        if trust["policy"] != "strict_r3_non_group_writable":
            raise RuntimeError("host lock strict mode policy is misbound")
    else:
        if (trust["policy"] !=
                "private_cache_exact_observed_modes_no_acl_no_same_ns_alias" or
                {name: modes[name] for name in ("cache", "host_directory", "file")} !=
                remote.LOCK_OBSERVED_MODES or
                any(identity[name]["gid"] != remote.LOCK_OBSERVED_GID
                    for name in ("cache", "host_directory", "file")) or
                any(identity[name]["inode"] != remote.LOCK_OBSERVED_INODES[name]
                    for name in ("cache", "host_directory", "file")) or
                trust["cache_nlink"] != remote.LOCK_OBSERVED_NLINKS["cache"] or
                trust["host_directory_nlink"] !=
                remote.LOCK_OBSERVED_NLINKS["host_directory"] or
                item["bytes"] != 0):
            raise RuntimeError("group-writable host lock lacks exact private parent")
    return identity


def _require_label(label, purpose):
    if (not isinstance(label, dict) or
            set(label) != {"rc", "matches", "match_count", "truncated",
                           "stderr_empty", "stdout_sha256", "started_utc",
                           "completed_utc"} or
            label.get("rc") != 0 or label.get("matches") != [] or
            label.get("match_count") != 0 or label.get("truncated") is not False or
            label.get("stderr_empty") is not True or
            label.get("stdout_sha256") != hashlib.sha256(b"").hexdigest()):
        raise RuntimeError(purpose + " science-label evidence missing or dirty")
    started = _time(label["started_utc"], purpose + " label start")
    completed = _time(label["completed_utc"], purpose + " label completion")
    if started > completed:
        raise RuntimeError(purpose + " science-label scan time is reversed")
    return started, completed


def _require_lock_fuser(item, owner_pid=None, lock_path=None):
    if (not isinstance(item, dict) or
            set(item) != {"path_exists", "rc", "pids", "stderr_empty",
                          "stdout_sha256", "stderr_sha256"} or
            item.get("path_exists") is not True):
        raise RuntimeError("host lock fuser evidence is missing or uncertain")
    if owner_pid is None:
        if (item.get("rc") != 1 or item.get("pids") != [] or
                item.get("stderr_empty") is not True or
                item.get("stderr_sha256") != hashlib.sha256(b"").hexdigest() or
                item.get("stdout_sha256") != hashlib.sha256(b"").hexdigest()):
            raise RuntimeError("host lock has a holder outside writer")
    elif (type(owner_pid) is not int or owner_pid <= 0 or item.get("rc") != 0 or
          item.get("pids") != [owner_pid] or
          not isinstance(lock_path, str) or
          item.get("stderr_empty") is not False or
          item.get("stderr_sha256") !=
          hashlib.sha256((lock_path + ":\n").encode("utf-8")).hexdigest() or
          not isinstance(item.get("stdout_sha256"), str) or
          len(item["stdout_sha256"]) != 64):
        raise RuntimeError("host lock self-only holder proof failed")
    return item


def _require_activity_gate(gate, remote, owner_pid=None):
    if not isinstance(gate, dict):
        raise RuntimeError("final activity gate missing")
    stops = set(remote.STOP_PATHS + [
        remote.CHECKOUT + "/research/agentic/STOP",
        remote.CHECKOUT + "/research/v2/STOP"])
    if (set(gate.get("targets", {})) != set(remote._target_paths()) or
            any(item.get("exists") is not False
                for item in gate["targets"].values()) or
            set(gate.get("stops", {})) != stops or
            any(item.get("exists") is not False
                for item in gate["stops"].values()) or
            gate.get("processes", {}).get("matched") != [] or
            gate.get("processes", {}).get("unreadable_cmdlines") != 0 or
            set(gate.get("checkpoint_fusers", {})) != set(remote.CHECKPOINTS)):
        raise RuntimeError("final activity target/STOP/process/holder gate failed")
    if any(item.get("path_exists") is not True or item.get("rc") != 1 or
           item.get("pids") != [] or item.get("stderr_empty") is not True
           for item in gate["checkpoint_fusers"].values()):
        raise RuntimeError("final activity checkpoint holder gate failed")
    _require_lock_fuser(gate.get("shared_lock_fuser"), owner_pid,
                        remote.LOCK_PATH if owner_pid is not None else None)
    neuron0, devices = gate.get("neuron0_fuser", {}), gate.get("neuron_ls", {})
    if (neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            devices.get("rc") != 0 or devices.get("stderr_empty") is not True or
            devices.get("json_valid") is not True or
            devices.get("schema_valid") is not True or
            devices.get("device_count", 0) < 1 or
            devices.get("non_diagnostic_pids") != []):
        raise RuntimeError("final activity Neuron gate failed")
    return gate


def _require_full_probe_inode_floor(probe):
    if not isinstance(probe, dict):
        raise RuntimeError("full probe is missing")
    # The inherited validator checks only identity_before. A full probe must
    # still have >= MIN_INODES at its terminal identity_after sample, whether
    # it is a preflight, writer prewrite probe, or final read-only postflight.
    for phase in ("identity_before", "identity_after"):
        identity = probe.get(phase)
        free = identity.get("root_free_inodes") if isinstance(identity, dict) else None
        if type(free) is not int or free < MIN_INODES:
            raise RuntimeError("full probe " + phase + " free-inode floor failed")
    return probe


def validate_probe(probe, *, before_write):
    _require_full_probe_inode_floor(probe)
    base.validate_probe(probe, before_write=before_write)
    _require_full_probe_inode_floor(probe)
    remote = remote_constants()
    identity = _require_lock_identity(probe.get("host_lock_identity"), remote)
    if identity["ubuntu"] != probe["identity_after"]["ancestry"]["ubuntu"]:
        raise RuntimeError("host lock parent diverges from probe ancestry")
    mounts = identity["trust"]["mounts"]
    if (mounts["mount_namespace"] != probe["identity_after"]["mount_namespace"] or
            mounts["root_mount_id"] != probe["identity_after"]["mount"]["mount_id"] or
            mounts["root_major_minor"] !=
            probe["identity_after"]["mount"]["major_minor"] or
            mounts["root_mount_point"] != probe["identity_after"]["mount"]["point"]):
        raise RuntimeError("host lock mount proof diverges from probe identity")
    _require_lock_fuser(probe.get("shared_lock_fuser"))
    _, label_end = _require_label(probe.get("science_label_scan"), "full probe")
    if label_end > _time(probe.get("utc"), "full probe"):
        raise RuntimeError("full probe label scan completed after its receipt")
    return probe


def compare_preflight_to_write(pre, write):
    """Bind the held launcher lock and last clean label scan to O_EXCL."""
    base.compare_preflight_to_write(pre, write)
    validate_probe(pre, before_write=True)
    prewrite = write["prewrite_probe"]
    validate_probe(prewrite, before_write=True)
    remote = remote_constants()
    lock_identity = pre["host_lock_identity"]
    lock_pin = remote._lock_identity_digest(lock_identity)
    final = write.get("final_creation_gate")
    if (not isinstance(final, dict) or
            prewrite["host_lock_identity"] != lock_identity or
            write.get("host_lock_identity") != lock_identity or
            write.get("preflight_lock_sha256") != lock_pin or
            lock_identity["ubuntu"] !=
            write["identity_before"]["ancestry"]["ubuntu"]):
        raise RuntimeError("preflight/write existing host lock continuity failed")
    first = _require_activity_gate(write.get("immediate_gate"), remote)
    pre_label = _require_activity_gate(final.get("pre_label_gate"), remote)
    lock = final.get("host_lock")
    if (not isinstance(lock, dict) or
            set(lock) != {"identity", "acquired_utc", "mode", "access",
                          "writer_process_identity", "fuser",
                          "identity_rechecked_before_final_label_scan"} or
            lock.get("identity") != lock_identity or
            lock.get("mode") != "LOCK_EX|LOCK_NB" or
            lock.get("access") !=
            "O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC" or
            lock.get("writer_process_identity") != write["process_identity"] or
            lock.get("identity_rechecked_before_final_label_scan") is not True):
        raise RuntimeError("final host lock acquisition evidence missing or misbound")
    owner_pid = write["process_identity"].get("pid")
    _require_activity_gate(final, remote, owner_pid=owner_pid)
    if (lock["fuser"] != final["shared_lock_fuser"] or
            write.get("host_lock_fuser") != lock["fuser"]):
        raise RuntimeError("final host lock self-only fuser evidence changed")
    label = final.get("science_label_scan")
    label_start, label_end = _require_label(label, "final pre-O_EXCL")
    if write.get("science_label_scan") != label:
        raise RuntimeError("final science-label evidence changed in write receipt")
    first_at = _time(first.get("utc"), "first activity")
    pre_label_at = _time(pre_label.get("utc"), "pre-lock activity")
    lock_at = _time(lock["acquired_utc"], "host lock acquisition")
    final_activity_at = _time(final.get("utc"), "post-lock activity")
    write_at = _time(write.get("utc"), "write receipt")
    if not (_time(prewrite.get("utc"), "writer full probe") < first_at <
            pre_label_at <= lock_at <= final_activity_at <=
            label_start <= label_end < write_at):
        raise RuntimeError("final host lock/activity/science-label order is not proven")
    return {"status": "PREFLIGHT_TO_FINAL_LOCKED_LABEL_GATE_PASS",
            "preflight_identity_sha256": write["preflight_identity_sha256"],
            "host_lock_identity_sha256": lock_pin,
            "host_lock_acquired_utc": lock["acquired_utc"],
            "post_lock_activity_utc": final["utc"],
            "final_label_started_utc": label["started_utc"],
            "final_label_completed_utc": label["completed_utc"],
            "final_label_stdout_sha256": label["stdout_sha256"]}


def compare_sessions(pre, write, verify):
    continuity = compare_preflight_to_write(pre, write)
    result = base.compare_sessions(pre, write, verify)
    remote = remote_constants()
    lock_identity = _require_lock_identity(verify.get("host_lock_identity"), remote)
    if (lock_identity != pre["host_lock_identity"] or
            lock_identity["ubuntu"] !=
            verify["identity_after"]["ancestry"]["ubuntu"]):
        raise RuntimeError("distinct readback host lock identity drift")
    _require_lock_fuser(verify.get("host_lock_fuser"))
    label_start, label_end = _require_label(verify.get("science_label_scan"),
                                          "distinct readback")
    if not (_time(write.get("utc"), "write receipt") < label_start <= label_end <=
            _time(verify.get("utc"), "distinct readback")):
        raise RuntimeError("distinct readback science-label order is not proven")
    return {**result,
            "host_lock_identity_sha256": continuity["host_lock_identity_sha256"],
            "final_label_stdout_sha256": continuity["final_label_stdout_sha256"],
            "readback_label_stdout_sha256": verify["science_label_scan"]["stdout_sha256"]}


def compare_postflight(pre, post, *, persistence=None):
    validate_probe(pre, before_write=True)
    validate_probe(post, before_write=False)
    result = base.compare_postflight(pre, post, persistence=persistence)
    if post["host_lock_identity"] != pre["host_lock_identity"]:
        raise RuntimeError("postflight host lock identity drift")
    lock_pin = remote_constants()._lock_identity_digest(pre["host_lock_identity"])
    if persistence is not None and (
            persistence.get("host_lock_identity_sha256") != lock_pin or
            persistence.get("final_label_stdout_sha256") !=
            hashlib.sha256(b"").hexdigest() or
            persistence.get("readback_label_stdout_sha256") !=
            hashlib.sha256(b"").hexdigest()):
        raise RuntimeError("postflight final label/lock proof is misbound")
    return {**result, "host_lock_identity_sha256": lock_pin,
            "postflight_label_stdout_sha256":
            post["science_label_scan"]["stdout_sha256"]}


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: this module only builds/reviews commands; it never opens SSH")
