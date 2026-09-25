"""Offline exact SSH argv builder and evidence validators; no dispatch CLI.

The only source stage is the frozen 446b capsule into one new trn21 ext4 path.
Importing or executing this module never opens SSH or changes host state.
"""

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess


HERE = Path(__file__).resolve().parent
REMOTE = HERE / "stage_r4_remote.py"
REMOTE_SHA256 = "be2e0aca21aaf5e29b330c5c0b004dd8a013e99c2aaaa29e820be53b0a15dd68"
SOURCE = Path("/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz")
SOURCE_BYTES = 104154875
SOURCE_SHA256 = "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a"
TARGET = "/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn21-stage-r1.tar.gz"
ENDPOINT = "ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com"
IDENTITY_FILE = "/home/poojith-devan/Desktop/Trainium/trn2.pem"
KNOWN_HOSTS = Path("/home/poojith-devan/.ssh/known_hosts")
KNOWN_HOSTS_SHA256 = "e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3"
HOST_FINGERPRINT = "SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc"
SCHEMA = "trn21-c5c-a0-r4-transfer-only-v1"
PRE_FREE = 13177973911
POST_FREE = 10737418240
MIN_INODES = 10000
CACHE_ROOT_INODE = 2945950
CHECKPOINTS = {
    "/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/out/"
    "fs-20260924-v2-35c9-lr015-control-trn21-r1-work/sandbox/out/final.pt":
    (469834849, "aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535", 2947631),
    "/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/out/"
    "fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/train/out/final.pt":
    (469831009, "5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9", 2938565),
}
CACHE_DEV = 66305
CACHE_INODES = {"neff": 2945951, "local": 2945952, "hlo": 2945953,
                "inductor": 2945954, "compiler": 2945955, "debug": 2945956}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def argv_sha256(argv):
    return sha256(json.dumps(argv, separators=(",", ":")).encode("utf-8"))


def reviewed_remote_source():
    data = REMOTE.read_bytes()
    if sha256(data) != REMOTE_SHA256:
        raise RuntimeError("reviewed remote source hash drift")
    return data.decode("utf-8")


def remote_constants():
    reviewed_remote_source()
    spec = importlib.util.spec_from_file_location("trn21_r4_remote_exact", REMOTE)
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
        raise RuntimeError("trn21 ED25519 host fingerprint mismatch")
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
    if before["hostname"] != "ip-172-31-34-85" or before["uid"] != 1000:
        raise RuntimeError("wrong trn21 host or UID")
    mount = before["mount"]
    if (mount["filesystem"] != "ext4" or mount["source"] != "/dev/root" or
            mount["major_minor"] != "259:1"):
        raise RuntimeError("not the fixed trn21 persistent ext4 source")
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


def validate_probe(probe, *, before_write):
    """Require a complete fresh safety and old-path inventory on every probe."""
    remote = remote_constants()
    if probe.get("schema") != SCHEMA or probe.get("operation") != "probe":
        raise RuntimeError("wrong probe schema/operation")
    _fresh(probe["utc"])
    identity = _identity(probe)
    floor = PRE_FREE if before_write else POST_FREE
    if (identity["root_free_bytes"] < floor or
            probe["identity_after"]["root_free_bytes"] < floor or
            identity["root_free_inodes"] < MIN_INODES):
        raise RuntimeError("fresh persistent root space/inode gate failed")
    old = probe["old_inventory"]
    if old.get("catalog_complete") is not True or set(old["paths"]) != set(remote._old_paths()):
        raise RuntimeError("old e65/r1/r2 source/cache/output inventory incomplete")
    allowed = {remote.ROOT + "/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_" + name
               for name in ("r1", "r2", "stage_r2", "r4")}
    if not set(old["top_level_matching_sources"]).issubset(allowed):
        raise RuntimeError("unknown old A0 source root; inventory needs independent review")
    if old["top_level_matching_sources"].get(remote.E65_CHECKOUT) != old["paths"][remote.E65_CHECKOUT]:
        raise RuntimeError("accepted e65 checkout omitted from dynamic old source inventory")
    for path, item in old["paths"].items():
        if (item.get("exists") is not True and item.get("exists") is not False) or (
                item.get("exists") is True and item.get("type") not in ("directory", "regular")):
            raise RuntimeError("unsafe or unclassified old path: " + path)
    for path, item in old["top_level_matching_sources"].items():
        if old["paths"].get(path) != item:
            raise RuntimeError("dynamic old source root differs from fixed inventory")
    if set(probe["targets"]) != set(remote._target_paths()):
        raise RuntimeError("future target inventory incomplete")
    for path in remote._target_paths():
        if path == TARGET:
            continue
        if probe["targets"].get(path, {}).get("exists") is not False:
            raise RuntimeError("future checkout/monitor/output path already exists")
    if old["paths"][remote.E65_CHECKOUT + "/out/" + remote.E65_LABEL].get("exists") is not False:
        raise RuntimeError("accepted e65 checkout science output appeared")
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
    if (set(protected["checkpoints"]) != set(CHECKPOINTS) or
            set(protected["e65_transfer_files"]) != set(remote.E65_TRANSFER_FILES) or
            set(protected["e65_source"]) != set(remote.E65_SOURCE)):
        raise RuntimeError("protected file inventory incomplete")
    for path, (size, digest, inode) in {**CHECKPOINTS,
                                        **remote.E65_TRANSFER_FILES}.items():
        item = protected["checkpoints"].get(path, protected["e65_transfer_files"].get(path))
        if (item.get("type") != "regular" or item.get("device") != 66305 or
                item.get("inode") != inode or item.get("uid") != 1000 or
                item.get("nlink") != 1 or item.get("bytes") != size or
                item.get("read_bytes") != size or item.get("sha256") != digest or
                old["paths"][path] != {key: value for key, value in item.items()
                                       if key not in ("read_bytes", "sha256")}):
            raise RuntimeError("protected checkpoint or earlier source-stage file changed")
    e65_checkout = protected["e65_checkout"]
    if (e65_checkout.get("type") != "directory" or
            e65_checkout.get("device") != 66305 or
            e65_checkout.get("inode") != remote.E65_CHECKOUT_INODE or
            e65_checkout.get("uid") != 1000 or
            e65_checkout.get("mode") != "0o700" or
            old["paths"][remote.E65_CHECKOUT] != e65_checkout):
        raise RuntimeError("accepted e65 checkout identity changed")
    for name, (size, digest, inode) in remote.E65_SOURCE.items():
        path = remote.E65_CHECKOUT + "/" + name
        item = protected["e65_source"][name]
        if (item.get("type") != "regular" or item.get("device") != 66305 or
                item.get("uid") != 1000 or item.get("nlink") != 1 or
                item.get("inode") != inode or
                item.get("bytes") != size or item.get("read_bytes") != size or
                item.get("sha256") != digest or
                old["paths"][path] != {key: value for key, value in item.items()
                                       if key not in ("read_bytes", "sha256")}):
            raise RuntimeError("accepted e65 source file changed: " + name)
    if (protected["cache_root"].get("type") != "directory" or
            protected["cache_root"].get("device") != CACHE_DEV or
            protected["cache_root"].get("uid") != 1000 or
            protected["cache_root"].get("inode") != CACHE_ROOT_INODE or
            set(protected["six_caches"]) != set(CACHE_INODES)):
        raise RuntimeError("protected warm-cache root changed")
    for name, inode in CACHE_INODES.items():
        item = protected["six_caches"][name]
        if (item.get("type") != "directory" or item.get("uid") != 1000 or
                item.get("device") != CACHE_DEV or
                item.get("inode") != inode):
            raise RuntimeError("protected warm-cache child changed: " + name)
    if old["paths"][remote.CACHE_ROOT] != protected["cache_root"]:
        raise RuntimeError("old inventory and protected warm cache disagree")
    if (probe["processes"].get("matched") != [] or
            probe["processes"].get("unreadable_cmdlines") != 0):
        raise RuntimeError("matched or unreadable trn21 process")
    label = probe["science_label_scan"]
    if (label.get("rc") != 0 or label.get("match_count") != 0 or
            label.get("matches") != [] or label.get("truncated") is not False or
            label.get("stderr_empty") is not True):
        raise RuntimeError("r4 science label is already used or scan uncertain")
    lock, neuron0, neuron_ls = (probe["shared_lock_fuser"],
                                probe["neuron0_fuser"], probe["neuron_ls"])
    if set(probe["checkpoint_fusers"]) != set(CHECKPOINTS):
        raise RuntimeError("protected checkpoint holder inventory incomplete")
    if any(item.get("path_exists") is not True or item.get("rc") != 1 or
           item.get("pids") != [] or item.get("stderr_empty") is not True
           for item in probe["checkpoint_fusers"].values()):
        raise RuntimeError("protected checkpoint is held")
    if (lock.get("pids") != [] or (lock.get("path_exists") and lock.get("rc") != 1) or
            (lock.get("path_exists") and lock.get("stderr_empty") is not True) or
            neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            neuron_ls.get("rc") != 0 or neuron_ls.get("stderr_empty") is not True or
            neuron_ls.get("json_valid") is not True or
            neuron_ls.get("schema_valid") is not True or
            neuron_ls.get("device_count", 0) < 1 or
            neuron_ls.get("non_diagnostic_pids") != []):
        raise RuntimeError("shared lock or Neuron busy/uncertain")
    return probe


def compare_preflight_to_write(pre, write):
    """Reject a writer on any new boot/mount/parent or safety state."""
    validate_probe(pre, before_write=True)
    if write.get("schema") != SCHEMA or write.get("operation") != "write":
        raise RuntimeError("missing exact writer receipt")
    prewrite = write.get("prewrite_probe")
    if not isinstance(prewrite, dict):
        raise RuntimeError("writer omitted its live full prewrite safety probe")
    validate_probe(prewrite, before_write=True)
    remote = remote_constants()
    expected_pin = remote._identity_digest(pre["identity_after"])
    if write.get("preflight_identity_sha256") != expected_pin:
        raise RuntimeError("writer did not bind the separate preflight identity")
    keys = ("hostname", "uid", "boot_id", "mount_namespace", "mount",
            "block_aliases", "ancestry")
    for before, after in ((pre["identity_after"], prewrite["identity_before"]),
                          (prewrite["identity_after"], write["identity_before"])):
        if any(before[key] != after[key] for key in keys):
            raise RuntimeError("preflight-to-write host/boot/mount/parent drift")
    if (pre["process_identity"] == prewrite["process_identity"] or
            prewrite["process_identity"] != write["process_identity"]):
        raise RuntimeError("prewrite probe did not run inside distinct writer SSH process")
    gate = write.get("immediate_gate")
    remote_stops = set(remote.STOP_PATHS + [
        remote.CHECKOUT + "/research/agentic/STOP",
        remote.CHECKOUT + "/research/v2/STOP"])
    if (not isinstance(gate, dict) or
            set(gate.get("targets", {})) != set(remote._target_paths()) or
            any(item.get("exists") is not False for item in gate["targets"].values()) or
            set(gate.get("stops", {})) != remote_stops or
            any(item.get("exists") is not False for item in gate["stops"].values()) or
            gate.get("processes", {}).get("matched") != [] or
            gate.get("processes", {}).get("unreadable_cmdlines") != 0 or
            set(gate.get("checkpoint_fusers", {})) != set(CHECKPOINTS)):
        raise RuntimeError("immediate writer STOP/process/holder evidence incomplete")
    for item in gate["checkpoint_fusers"].values():
        if (item.get("path_exists") is not True or item.get("rc") != 1 or
                item.get("pids") != [] or item.get("stderr_empty") is not True):
            raise RuntimeError("immediate writer checkpoint holder appeared")
    lock, neuron0, devices = (gate.get("shared_lock_fuser", {}),
                              gate.get("neuron0_fuser", {}), gate.get("neuron_ls", {}))
    if (lock.get("pids") != [] or
            (lock.get("path_exists") and
             (lock.get("rc") != 1 or lock.get("stderr_empty") is not True)) or
            neuron0.get("path_exists") is not True or neuron0.get("rc") != 1 or
            neuron0.get("pids") != [] or neuron0.get("stderr_empty") is not True or
            devices.get("rc") != 0 or devices.get("stderr_empty") is not True or
            devices.get("json_valid") is not True or devices.get("schema_valid") is not True or
            devices.get("device_count", 0) < 1 or devices.get("non_diagnostic_pids") != []):
        raise RuntimeError("immediate writer Neuron or lock evidence failed")
    if pre["old_inventory"] != prewrite["old_inventory"] or pre["protected"] != prewrite["protected"]:
        raise RuntimeError("protected or prior-stage assets changed before writer")
    times = [datetime.datetime.fromisoformat(item["utc"]) for item in (pre, prewrite, write)]
    if any(item.tzinfo is None for item in times) or not times[0] < times[1] < times[2]:
        raise RuntimeError("preflight/prewrite/write order is not proven")
    return {"status": "PREFLIGHT_TO_WRITE_LIVE_GATE_PASS",
            "preflight_identity_sha256": expected_pin}


def compare_sessions(pre, write, verify):
    compare_preflight_to_write(pre, write)
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
    for key in ("hostname", "uid", "boot_id", "mount_namespace", "mount",
                "block_aliases", "ancestry"):
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
            "boot_id": verify["identity_after"]["boot_id"]}


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
