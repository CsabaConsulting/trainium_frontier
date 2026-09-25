"""Offline evidence validators for disabled trn22 lock/final-label repair.

This package deliberately refuses to build SSH argv. A future enabled request
must be separately reviewed after static PASS and a one-shot assignment.
"""

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess


HERE = Path(__file__).resolve().parent
REMOTE = HERE / "stage_r4_remote_lock_hold.py"
REMOTE_SHA256 = "cdf06f700722e144dcb304e517a7e000fe0c6b2bcb210e4d2bb38f92757d5e27"
SOURCE = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz")
SOURCE_BYTES = 104154875
SOURCE_SHA256 = "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a"
TARGET = "/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz"
ENDPOINT = "ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com"
IDENTITY_FILE = "/home/poojith-devan/Desktop/Trainium/trn2.pem"
KNOWN_HOSTS = Path("/home/poojith-devan/.ssh/known_hosts")
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
HOST_FINGERPRINT = "SHA256:bXvW4P20oehuJWPMDLUS72E1aPY58rq/kOqUybiB44U"
SCHEMA = "trn22-c5c-a0-r4-transfer-private-cache-nonblock-lock-hold-v3"
WRITE_BINDING_SCHEMA = "trn22-r4-preflight-private-cache-nonblock-lock-write-binding-hold-v3"
STABLE_IDENTITY_KEYS = ("hostname", "uid", "boot_id", "mount_namespace",
                        "mount", "block_aliases", "ancestry")
PRE_FREE = 13177973911
POST_FREE = 10737418240
MIN_INODES = 10000
CHECKPOINT_DEV_INO = (66305, 2417091)
CHECKPOINT_BYTES = 469831009
CHECKPOINT_SHA256 = "1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b"
CACHE_DEV = 66305
CACHE_INODES = {"neff": 2430420, "local": 2430421, "hlo": 2430422,
                "inductor": 2430423, "compiler": 2430424, "debug": 2430425}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def argv_sha256(argv):
    return sha256(json.dumps(argv, separators=(",", ":")).encode("utf-8"))


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def reviewed_remote_source():
    data = REMOTE.read_bytes()
    if sha256(data) != REMOTE_SHA256:
        raise RuntimeError("reviewed remote source hash drift")
    return data.decode("utf-8")


def remote_constants():
    reviewed_remote_source()
    spec = importlib.util.spec_from_file_location("trn22_r4_remote_exact", REMOTE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_local_source():
    parent = SOURCE.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) & 0o077):
        raise RuntimeError("source artifact directory is not private")
    named = SOURCE.lstat()
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
            named.st_uid != os.getuid() or stat.S_IMODE(named.st_mode) != 0o600 or
            named.st_size != SOURCE_BYTES):
        raise RuntimeError("local source type, owner, mode, link or bytes drift")
    fd = os.open(SOURCE, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        identity = ("st_dev", "st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, key) != getattr(named, key) for key in identity):
            raise RuntimeError("source changed during no-follow open")
        digest, count = hashlib.sha256(), 0
        while True:
            block = os.read(fd, 4 * 1024 * 1024)
            if not block:
                break
            count += len(block)
            digest.update(block)
        if (count != SOURCE_BYTES or digest.hexdigest() != SOURCE_SHA256 or
                any(getattr(snapshot, key) != getattr(opened, key)
                    for snapshot in (os.fstat(fd), SOURCE.lstat()) for key in identity)):
            raise RuntimeError("source bytes/hash or inode changed while hashing")
        return {"path": str(SOURCE), "bytes": count, "sha256": digest.hexdigest(),
                "device": opened.st_dev, "inode": opened.st_ino, "mode": "0o600"}
    finally:
        os.close(fd)


def checked_ssh_identity():
    """Read-only pinned ED25519 host-key proof; never print key contents."""
    parent, named = KNOWN_HOSTS.parent.lstat(), KNOWN_HOSTS.lstat()
    private = Path(IDENTITY_FILE).lstat()  # Metadata only; never read the PEM.
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or
            stat.S_IMODE(parent.st_mode) & 0o077 or
            not stat.S_ISREG(named.st_mode) or named.st_uid != os.getuid() or
            named.st_nlink != 1 or stat.S_IMODE(named.st_mode) & 0o077 or
            not stat.S_ISREG(private.st_mode) or private.st_uid != os.getuid() or
            private.st_nlink != 1 or stat.S_IMODE(private.st_mode) != 0o600):
        raise RuntimeError("SSH known-hosts or private-key metadata is unsafe")
    fd = os.open(KNOWN_HOSTS, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
                named.st_dev, named.st_ino, named.st_size):
            raise RuntimeError("known-hosts inode changed before hash")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if (digest.hexdigest() != KNOWN_HOSTS_SHA256 or
                (os.fstat(fd).st_dev, os.fstat(fd).st_ino, os.fstat(fd).st_size) !=
                (named.st_dev, named.st_ino, named.st_size)):
            raise RuntimeError("pinned known-hosts bytes changed")
    finally:
        os.close(fd)
    hostname = ENDPOINT.split("@", 1)[1]
    found = subprocess.run(["ssh-keygen", "-F", hostname, "-f", str(KNOWN_HOSTS)],
                           capture_output=True, timeout=20, check=False)
    if found.returncode != 0:
        raise RuntimeError("pinned host entry not found")
    entries = [line for line in found.stdout.splitlines() if line and not line.startswith(b"#")]
    ed25519 = [line for line in entries if len(line.split()) >= 3 and
               line.split()[1] == b"ssh-ed25519"]
    if len(ed25519) != 1:
        raise RuntimeError("expected exactly one pinned ED25519 host entry")
    fingerprint = subprocess.run(["ssh-keygen", "-l", "-E", "sha256", "-f", "-"],
                                 input=ed25519[0] + b"\n", capture_output=True,
                                 timeout=20, check=False)
    if (fingerprint.returncode != 0 or
            HOST_FINGERPRINT.encode("ascii") not in fingerprint.stdout.split() or
            b"(ED25519)" not in fingerprint.stdout.split()):
        raise RuntimeError("trn22 ED25519 host fingerprint mismatch")
    # The second hash closes the read-only ssh-keygen race. Any later drift is
    # caught again immediately before each separately configured SSH process.
    checked = os.open(KNOWN_HOSTS, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(checked)
        if (info.st_dev, info.st_ino, info.st_size) != (
                named.st_dev, named.st_ino, named.st_size):
            raise RuntimeError("known-hosts inode changed after fingerprint check")
        again = hashlib.sha256()
        while True:
            block = os.read(checked, 1024 * 1024)
            if not block:
                break
            again.update(block)
        if again.hexdigest() != KNOWN_HOSTS_SHA256:
            raise RuntimeError("known-hosts changed while checking the host key")
    finally:
        os.close(checked)
    return {"known_hosts_sha256": KNOWN_HOSTS_SHA256,
            "host_fingerprint": HOST_FINGERPRINT,
            "host_key_algorithm": "ssh-ed25519"}


def write_binding(preflight):
    """Bind one fresh read-only probe to the sole remote write SSH process."""
    validate_probe(preflight, before_write=True)
    pin = {key: preflight["identity_after"][key] for key in STABLE_IDENTITY_KEYS}
    return {"schema": WRITE_BINDING_SCHEMA,
            "source_commit": remote_constants().SOURCE_COMMIT,
            "target": TARGET, "capsule_sha256": SOURCE_SHA256,
            "preflight_utc": preflight["utc"],
            "preflight_process_identity": preflight["process_identity"],
            "preflight_receipt_sha256": sha256(_canonical_bytes(preflight)),
            "identity_pin": pin,
            "lock_identity_pin": preflight["host_lock_identity"]}


def build_ssh_argv(operation, *, preflight=None):
    raise RuntimeError("DISPATCH HOLD: no SSH argv for unreviewed lock/final-label repair")


def _fresh(when, max_age_seconds=180):
    observed = datetime.datetime.fromisoformat(when)
    now = datetime.datetime.now(datetime.timezone.utc)
    if (observed.tzinfo is None or
            not -5 <= (now - observed).total_seconds() <= max_age_seconds):
        raise RuntimeError("remote read-only probe is stale or from future")


def _identity(receipt):
    before, after = receipt["identity_before"], receipt["identity_after"]
    for key in ("hostname", "uid", "boot_id", "mount_namespace", "mount",
                "block_aliases", "ancestry"):
        if before.get(key) != after.get(key):
            raise RuntimeError("host/boot/mount changed during one SSH process: " + key)
    if before["hostname"] != "ip-172-31-32-253" or before["uid"] != 1000:
        raise RuntimeError("wrong trn22 host or UID")
    mount = before["mount"]
    if (mount["filesystem"] != "ext4" or mount["source"] != "/dev/root" or
            mount["major_minor"] != "259:1"):
        raise RuntimeError("not the fixed trn22 persistent ext4 source")
    aliases = before["block_aliases"]
    names = aliases["historical_names"]
    if (aliases["mount_major_minor"] != mount["major_minor"] or
            set(names) != {"/dev/root", "/dev/nvme0n1p1"} or
            names["/dev/nvme0n1p1"].get("exists") is not True or
            names["/dev/nvme0n1p1"].get("resolved_rdev_major_minor") != "259:1" or
            (names["/dev/root"].get("exists") is True and
             names["/dev/root"].get("resolved_rdev_major_minor") != "259:1") or
            (names["/dev/root"].get("exists") is False and
             names["/dev/root"].get("mount_source_string_only") is not True) or
            not aliases["sysfs_realpath"].startswith("/sys/devices/") or
            not aliases["sysfs_realpath"].endswith("/nvme0n1p1")):
        raise RuntimeError("root-device aliases have not been reconciled")
    ancestry = before["ancestry"]
    for name, expected_uid in (("root", 0), ("home", 0), ("ubuntu", 1000)):
        item = ancestry[name]
        if (item["type"] != "directory" or item["uid"] != expected_uid or
                int(item["mode"], 8) & 0o022):
            raise RuntimeError("unsafe stage ancestor: " + name)
    if ancestry["ubuntu"]["device"] != int(os.makedev(*map(int, mount["major_minor"].split(":")))):
        raise RuntimeError("stage parent not on fixed ext4 device")
    return before


def _validate_lock_identity(record, identity):
    """Require a real pre-existing no-follow lock on the pinned ext4 device."""
    remote = remote_constants()
    device = identity["ancestry"]["ubuntu"]["device"]
    if (not isinstance(record, dict) or
            set(record) != {"path", "cache_directory", "lock_directory", "file"} or
            record["path"] != remote.HOST_LOCK):
        raise RuntimeError("missing existing cooperative-lock identity")
    for name in ("cache_directory", "lock_directory"):
        item = record[name]
        if (not isinstance(item, dict) or
                set(item) != {"type", "device", "inode", "uid", "gid", "mode", "acl"} or
                item["type"] != "directory" or item["device"] != device or
                item["uid"] != 1000 or item["gid"] != 1000 or
                not isinstance(item["inode"], int) or item["inode"] <= 0 or
                not isinstance(item["mode"], str) or
                item["acl"] != {"access": "ENODATA", "default": "ENODATA"}):
            raise RuntimeError("unsafe cooperative-lock ancestor: " + name)
        try:
            mode = int(item["mode"], 8)
        except ValueError as error:
            raise RuntimeError("malformed cooperative-lock ancestor mode") from error
        if name == "cache_directory":
            if mode != 0o700:
                raise RuntimeError("cooperative-lock cache is not private")
        elif (mode & 0o7000) or (mode & 0o002) or (
                (mode & 0o020) and mode != 0o775):
            raise RuntimeError("unsafe cooperative-lock inner directory")
    item = record["file"]
    if (not isinstance(item, dict) or
            set(item) != {"exists", "type", "device", "inode", "uid", "gid",
                          "mode", "nlink", "bytes", "acl"} or
            item["exists"] is not True or item["type"] != "regular" or
            item["device"] != device or item["uid"] != 1000 or
            item["gid"] != 1000 or item["nlink"] != 1 or
            not isinstance(item["inode"], int) or item["inode"] <= 0 or
            not isinstance(item["bytes"], int) or item["bytes"] < 0 or
            not isinstance(item["mode"], str) or
            item["acl"] != {"access": "ENODATA"}):
        raise RuntimeError("unsafe or absent cooperative-lock file")
    try:
        mode = int(item["mode"], 8)
    except ValueError as error:
        raise RuntimeError("malformed cooperative-lock mode") from error
    if (mode & 0o7000) or (mode & 0o002) or (
            (mode & 0o020) and mode != 0o664):
        raise RuntimeError("unsafe cooperative-lock file mode")
    return record


def validate_probe(probe, *, before_write):
    """Require a complete fresh safety and old-path inventory on every probe."""
    remote = remote_constants()
    if probe.get("schema") != SCHEMA or probe.get("operation") != "probe":
        raise RuntimeError("wrong probe schema/operation")
    _fresh(probe["utc"])
    identity = _identity(probe)
    _validate_lock_identity(probe.get("host_lock_identity"), identity)
    floor = PRE_FREE if before_write else POST_FREE
    if (identity["root_free_bytes"] < floor or
            probe["identity_after"]["root_free_bytes"] < floor or
            identity["root_free_inodes"] < MIN_INODES):
        raise RuntimeError("fresh persistent root space/inode gate failed")
    old = probe["old_inventory"]
    if old.get("catalog_complete") is not True or set(old["paths"]) != set(remote._old_paths()):
        raise RuntimeError("old r1/r2/r3 source/cache/output inventory incomplete")
    allowed = {remote.ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_" + name
               for name in ("r1", "r2", "r3", "r4")}
    if not set(old["top_level_matching_sources"]).issubset(allowed):
        raise RuntimeError("unknown old A0 source root; inventory needs independent review")
    for path, item in old["paths"].items():
        if (item.get("exists") is not True and item.get("exists") is not False) or (
                item.get("exists") is True and item.get("type") not in ("directory", "regular")):
            raise RuntimeError("unsafe or unclassified old path: " + path)
        # A Git worktree can use a regular .git file. The nested HEAD then
        # cannot exist, but its blocking file must be separately inventoried
        # and match the no-follow snapshot exactly. No other blocked ancestor
        # is accepted as an absent path.
        if path.endswith("/.git/HEAD"):
            git_file = old["paths"].get(path[:-len("/HEAD")], {})
            if git_file.get("type") == "regular":
                if item != {"exists": False, "file_backed_git": git_file}:
                    raise RuntimeError("file-backed old .git inventory mismatch: " + path)
            elif "file_backed_git" in item:
                raise RuntimeError("unexpected blocked old .git/HEAD path: " + path)
        elif "file_backed_git" in item:
            raise RuntimeError("unexpected blocked old path: " + path)
    for path, item in old["top_level_matching_sources"].items():
        if old["paths"].get(path) != item:
            raise RuntimeError("dynamic old source root differs from fixed inventory")
    for path in (remote.CHECKOUT, remote.MONITOR, remote.CHECKOUT + "/out",
                 remote.CHECKOUT + "/out/" + remote.LABEL,
                 remote.CHECKOUT + "/out/" + remote.LABEL + "-work",
                 remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".log",
                 remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".steps.jsonl",
                 remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".process.json",
                 remote.CHECKOUT + "/research/run-logs/" + remote.LABEL + ".postflight.json",
                 remote.OBSERVER + "/" + remote.LABEL + ".observer.json",
                 remote.OBSERVER + "/" + remote.LABEL + ".observer.stderr.log"):
        if probe["targets"].get(path, {}).get("exists") is not False:
            raise RuntimeError("future checkout/monitor/output path already exists")
    archive = probe["targets"].get(TARGET, {})
    if before_write and archive.get("exists") is not False:
        raise RuntimeError("new one-shot archive target already exists")
    if not before_write and archive.get("exists") not in (True, False):
        raise RuntimeError("postflight could not classify one-shot target")
    if archive.get("exists") is True and (
            archive.get("type") != "regular" or archive.get("mode") != "0o600" or
            archive.get("uid") != 1000 or archive.get("nlink") != 1 or
            archive.get("device") != identity["ancestry"]["ubuntu"]["device"]):
        raise RuntimeError("partial/completed archive target metadata unsafe")
    expected_stops = set(remote.STOP_PATHS + [
        remote.CHECKOUT + "/research/agentic/STOP",
        remote.CHECKOUT + "/research/v2/STOP"])
    if set(probe["stops"]) != expected_stops or any(
            item.get("exists") is not False for item in probe["stops"].values()):
        raise RuntimeError("STOP path present or incomplete")
    protected = probe["protected"]
    checkpoint = protected["checkpoint"]
    if (checkpoint.get("type") != "regular" or
            (checkpoint.get("device"), checkpoint.get("inode")) != CHECKPOINT_DEV_INO or
            checkpoint.get("uid") != 1000 or checkpoint.get("nlink") != 1 or
            checkpoint.get("bytes") != CHECKPOINT_BYTES or
            checkpoint.get("read_bytes") != CHECKPOINT_BYTES or
            checkpoint.get("sha256") != CHECKPOINT_SHA256):
        raise RuntimeError("protected checkpoint changed")
    if (protected["cache_root"].get("type") != "directory" or
            protected["cache_root"].get("device") != CACHE_DEV or
            protected["cache_root"].get("uid") != 1000 or
            set(protected["six_caches"]) != set(CACHE_INODES)):
        raise RuntimeError("protected warm-cache root changed")
    for name, inode in CACHE_INODES.items():
        item = protected["six_caches"][name]
        if (item.get("type") != "directory" or item.get("uid") != 1000 or
                item.get("device") != CACHE_DEV or
                item.get("inode") != inode):
            raise RuntimeError("protected warm-cache child changed: " + name)
    if (old["paths"][remote.CHECKPOINT] != {key: value for key, value in checkpoint.items()
             if key not in ("read_bytes", "sha256")} or
            old["paths"][remote.CACHE_ROOT] != protected["cache_root"]):
        raise RuntimeError("old inventory and protected checkpoint/cache disagree")
    if (probe["processes"].get("matched") != [] or
            probe["processes"].get("unreadable_cmdlines") != 0):
        raise RuntimeError("matched or unreadable trn22 process")
    label = probe["science_label_scan"]
    if (label.get("rc") != 0 or label.get("match_count") != 0 or
            label.get("matches") != [] or label.get("truncated") is not False or
            label.get("stderr_empty") is not True):
        raise RuntimeError("r4 science label is already used or scan uncertain")
    lock, neuron0, neuron_ls = (probe["shared_lock_fuser"],
                                probe["neuron0_fuser"], probe["neuron_ls"])
    checkpoint_fuser = probe["checkpoint_fuser"]
    if (checkpoint_fuser.get("path_exists") is not True or
            checkpoint_fuser.get("rc") != 1 or checkpoint_fuser.get("pids") != [] or
            checkpoint_fuser.get("stderr_empty") is not True or
            lock.get("path_exists") is not True or lock.get("pids") != [] or
            lock.get("rc") != 1 or lock.get("stderr_empty") is not True or
            neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            neuron_ls.get("rc") != 0 or neuron_ls.get("stderr_empty") is not True or
            neuron_ls.get("json_valid") is not True or
            neuron_ls.get("schema_valid") is not True or
            neuron_ls.get("device_count", 0) < 1 or
            neuron_ls.get("non_diagnostic_pids") != []):
        raise RuntimeError("shared lock or Neuron busy/uncertain")
    return probe


def _check_stop_and_process_snapshot(snapshot):
    remote = remote_constants()
    expected_stops = set(remote._all_stop_paths())
    if (not isinstance(snapshot, dict) or
            set(snapshot.get("stops", {})) != expected_stops or
            any(item.get("exists") is not False
                for item in snapshot["stops"].values()) or
            snapshot.get("processes", {}).get("matched") != [] or
            snapshot.get("processes", {}).get("unreadable_cmdlines") != 0):
        raise RuntimeError("live prewrite STOP/process safety receipt is incomplete or busy")


def _check_full_live_safety(safety, *, owned_lock=None):
    _check_stop_and_process_snapshot(safety)
    label = safety.get("science_label_scan", {})
    checkpoint_fuser = safety.get("checkpoint_fuser", {})
    lock = safety.get("shared_lock_fuser", {})
    held = safety.get("cooperative_lock")
    neuron0 = safety.get("neuron0_fuser", {})
    neuron_ls = safety.get("neuron_ls", {})
    if (checkpoint_fuser.get("path_exists") is not True or
            checkpoint_fuser.get("rc") != 1 or
            checkpoint_fuser.get("pids") != [] or
            checkpoint_fuser.get("stderr_empty") is not True or
            (owned_lock is None and (held is not None or
             lock.get("path_exists") is not True or lock.get("pids") != [] or
             lock.get("rc") != 1 or lock.get("stderr_empty") is not True)) or
            (owned_lock is not None and (lock is not None or held != owned_lock)) or
            neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            neuron_ls.get("rc") != 0 or neuron_ls.get("stderr_empty") is not True or
            neuron_ls.get("json_valid") is not True or
            neuron_ls.get("schema_valid") is not True or
            neuron_ls.get("device_count", 0) < 1 or
            neuron_ls.get("non_diagnostic_pids") != [] or
            label.get("rc") != 0 or label.get("match_count") != 0 or
            label.get("matches") != [] or label.get("truncated") is not False or
            label.get("stderr_empty") is not True):
        raise RuntimeError("live prewrite lock/Neuron/label safety receipt is busy")


def _validate_final_label_evidence(evidence, write):
    """Bind the last empty observation after lock acquisition to O_EXCL."""
    remote = remote_constants()
    final = evidence.get("final_label_observation")
    if (not isinstance(final, dict) or
            set(final) != {"started_utc", "completed_utc", "started_monotonic_ns",
                           "completed_monotonic_ns", "scan"}):
        raise RuntimeError("missing exact final-label observation")
    scan = final["scan"]
    if (not isinstance(scan, dict) or
            set(scan) != {"rc", "matches", "match_count", "truncated",
                          "stderr_empty", "stdout_sha256"} or scan.get("rc") != 0 or
            scan.get("match_count") != 0 or scan.get("matches") != [] or
            scan.get("truncated") is not False or scan.get("stderr_empty") is not True or
            scan.get("stdout_sha256") != sha256(b"")):
        raise RuntimeError("final label scan was nonempty or malformed")
    try:
        started = datetime.datetime.fromisoformat(final["started_utc"])
        completed = datetime.datetime.fromisoformat(final["completed_utc"])
        receipt = datetime.datetime.fromisoformat(write["utc"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("malformed final-label timestamps") from error
    if (started.tzinfo is None or completed.tzinfo is None or receipt.tzinfo is None or
            not started <= completed <= receipt):
        raise RuntimeError("final-label timestamp order failed")
    numbers = [evidence.get("cooperative_lock_acquired_monotonic_ns"),
               final["started_monotonic_ns"], final["completed_monotonic_ns"],
               evidence.get("target_open_attempt_monotonic_ns"),
               evidence.get("target_open_completed_monotonic_ns")]
    if (not all(isinstance(value, int) and value > 0 for value in numbers) or
            numbers != sorted(numbers) or
            numbers[3] - numbers[2] > remote.MAX_FINAL_SCAN_TO_OPEN_NS or
            numbers[4] - numbers[3] > remote.MAX_FINAL_SCAN_TO_OPEN_NS):
        raise RuntimeError("final-label observation is not immediately before O_EXCL")
    return final


def compare_preflight_write(preflight, write):
    """Close the preflight-to-write identity and live-safety continuity gap."""
    validate_probe(preflight, before_write=True)
    if write.get("schema") != SCHEMA or write.get("operation") != "write":
        raise RuntimeError("missing exact remote write receipt")
    _identity(write)
    for key in STABLE_IDENTITY_KEYS:
        if preflight["identity_after"][key] != write["identity_before"][key]:
            raise RuntimeError("preflight-to-write host/boot/mount drift: " + key)
    if preflight["process_identity"] == write["process_identity"]:
        raise RuntimeError("write did not enter a distinct SSH process")
    pre_time = datetime.datetime.fromisoformat(preflight["utc"])
    write_time = datetime.datetime.fromisoformat(write["utc"])
    if (pre_time.tzinfo is None or write_time.tzinfo is None or
            write_time < pre_time or (write_time - pre_time).total_seconds() > 600):
        raise RuntimeError("write timestamp not after its exact fresh preflight")
    evidence = write.get("prewrite_evidence", {})
    expected_binding = write_binding(preflight)
    if (evidence.get("preflight_binding_sha256") !=
            sha256(_canonical_bytes(expected_binding)) or
            evidence.get("preflight_receipt_sha256") !=
            expected_binding["preflight_receipt_sha256"] or
            evidence.get("preflight_lock_identity") !=
            expected_binding["lock_identity_pin"] or
            evidence.get("cooperative_lock_after_create") !=
            expected_binding["lock_identity_pin"] or
            evidence.get("cooperative_lock_after_write") !=
            expected_binding["lock_identity_pin"]):
        raise RuntimeError("remote write receipt is not bound to this preflight")
    _check_full_live_safety(evidence.get("live_prewrite_safety", {}))
    _check_full_live_safety(evidence.get("last_live_precreate_safety", {}),
                            owned_lock=expected_binding["lock_identity_pin"])
    _check_stop_and_process_snapshot(evidence.get("last_stop_process_check", {}))
    final_label = _validate_final_label_evidence(evidence, write)
    if (datetime.datetime.fromisoformat(final_label["started_utc"]) <
            datetime.datetime.fromisoformat(preflight["utc"])):
        raise RuntimeError("final-label observation predates its bound preflight")
    return {"status": "PREFLIGHT_TO_WRITE_IDENTITY_AND_LIVE_SAFETY_PASS",
            "preflight_receipt_sha256": expected_binding["preflight_receipt_sha256"],
            "preflight_binding_sha256": sha256(_canonical_bytes(expected_binding)),
            "cooperative_lock_identity_sha256": sha256(_canonical_bytes(
                expected_binding["lock_identity_pin"])),
            "final_label_observation_sha256": sha256(_canonical_bytes(final_label)),
            "boot_id": write["identity_before"]["boot_id"]}


def compare_sessions(write, verify, *, preflight):
    continuity = compare_preflight_write(preflight, write)
    if (write.get("schema") != SCHEMA or verify.get("schema") != SCHEMA or
            write.get("operation") != "write" or verify.get("operation") != "verify" or
            write.get("status") != "HASH_VERIFIED_IN_THIS_SESSION_ONLY" or
            verify.get("status") != write.get("status")):
        raise RuntimeError("missing write plus independent read-only verify")
    for receipt in (write, verify):
        _identity(receipt)
        if (receipt["path"] != TARGET or receipt["expected_bytes"] != SOURCE_BYTES or
                receipt["expected_sha256"] != SOURCE_SHA256 or
                receipt["readback_sha256"] != SOURCE_SHA256 or
                receipt["target_stat"]["bytes"] != SOURCE_BYTES or
                receipt["target_stat"]["mode"] != "0o600" or
                receipt["target_stat"]["nlink"] != 1):
            raise RuntimeError("receipt does not match exact capsule target")
    if (write["streamed_sha256"] != SOURCE_SHA256 or
            verify["streamed_sha256"] is not None or
            write["target_stat"] != verify["target_stat"]):
        raise RuntimeError("cross-session stream, inode or mode mismatch")
    for key in STABLE_IDENTITY_KEYS:
        if write["identity_after"][key] != verify["identity_before"][key]:
            raise RuntimeError("cross-session host/boot/mount drift: " + key)
    if write["process_identity"] == verify["process_identity"]:
        raise RuntimeError("verification not in a distinct SSH process")
    first, second = (datetime.datetime.fromisoformat(receipt["utc"])
                     for receipt in (write, verify))
    if first.tzinfo is None or second.tzinfo is None or second <= first:
        raise RuntimeError("read-only verification did not occur after write")
    return {"status": "CROSS_SESSION_PERSISTENCE_HASH_PASS", "path": TARGET,
            "bytes": SOURCE_BYTES, "sha256": SOURCE_SHA256,
            "target_stat": verify["target_stat"],
            "boot_id": verify["identity_after"]["boot_id"],
            "preflight_continuity": continuity}


def compare_postflight(pre, post, *, persistence=None):
    validate_probe(pre, before_write=True)
    validate_probe(post, before_write=False)
    for key in ("hostname", "uid", "boot_id", "mount_namespace", "mount",
                "block_aliases", "ancestry"):
        if pre["identity_after"][key] != post["identity_before"][key]:
            raise RuntimeError("pre/postflight host/boot/mount drift: " + key)
    if pre["old_inventory"] != post["old_inventory"]:
        raise RuntimeError("old r1/r2/r3 source/cache/output paths changed")
    if pre["protected"] != post["protected"]:
        raise RuntimeError("protected checkpoint/cache changed")
    if persistence is not None:
        target = post["targets"][TARGET]
        if target != persistence["target_stat"] or target["bytes"] != SOURCE_BYTES:
            raise RuntimeError("postflight archive inode differs from verified transfer")
    return {"status": "READ_ONLY_POSTFLIGHT_PASS", "archive_exists":
            post["targets"][TARGET]["exists"]}


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: this module only builds/reviews commands; it never opens SSH")
