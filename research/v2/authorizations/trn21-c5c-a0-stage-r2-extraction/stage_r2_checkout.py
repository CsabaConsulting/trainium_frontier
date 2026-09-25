"""Exact trn21 source checkout logic for a later, separately authorized stage.

This module has no dispatch entry point. The r2 transfer assignment permits only
three file transfers and cross-session verification; it does not permit calling
``stage_fixed_checkout``. A future Operator request must bind the reviewed
transfer receipts, these exact source bytes, fresh host preflight and one-shot
stage authority before any host invocation.

Extraction creates one previously absent checkout and never deletes, replaces,
renames or cleans any path. On failure, a partial checkout is retained.
"""

import datetime
import gzip
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import tarfile


SCHEMA = "trn21-c5c-a0-stage-r2-extraction-v1"
TRANSFER_SCHEMA = "trn21-c5c-a0-stage-r2-transfer-v1"
HOST = "ip-172-31-34-85"
UID = 1000
HOME = "/home/ubuntu"
CAPSULE_NAME = "trainium-c5c-e65-one-commit-20260925.trn21-stage-r2.tar.gz"
CAPSULE_PATH = HOME + "/" + CAPSULE_NAME
CHECKOUT_NAME = "trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2"
CHECKOUT_PATH = HOME + "/" + CHECKOUT_NAME
SCIENCE_OUTPUT = CHECKOUT_PATH + "/out/fs-20260925-v2-c5c-a0-control-trn21-r1"
ROOT_FREE_FLOOR = 10737418240
CAPSULE_BYTES = 104005787
CAPSULE_SHA256 = "32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264"
SOURCE_COMMIT = "e65a3097a5b00f6178188a935bb2acb9a5042c6c"
TRAIN_SHA256 = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA256 = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
GIT_CONFIG_SHA256 = "cfe7ba1238c9a78be7535d7c63bcaf5a4d5011d46b07c9b45d3bbf7d6c312dfe"
SOURCE_CONTRACT_SHA256 = "631f3e7c1a2c4ec962b58be3b3a5399d323bae22d41add1aaca7884cd4efe069"
TRANSFER_ASSIGNMENT_SHA256 = "0127358fed6e97787b8a7d60208ede3cadb0e870791baece2f12830300c09fd5"
EXPECTED_MEMBERS = 6338
EXPECTED_FILES = 6283
EXPECTED_DIRECTORIES = 55
EXPECTED_GNU_LONG_HEADERS = 4
EXPECTED_UNPACKED_BYTES = 188636254
EXPECTED_MANIFEST_SHA256 = "a9eff72e760f314717e712a5e3d23d300232db4feeb361896c57b16f7bdd1bf4"
CHUNK = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_UNPACKED_BYTES = 512 * 1024 * 1024
OPEN_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
OPEN_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
OPEN_NEW = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC

TRANSFER_ARTIFACTS = {
    "capsule": (CAPSULE_PATH, CAPSULE_BYTES, CAPSULE_SHA256),
    "preflight": (
        HOME + "/trainium-v2-g2-preflight-dryrun-20260925-trn21-stage-r2.py",
        23550, "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8"),
    "contained": (
        HOME + "/trainium-v2-g2-contained-20260925-trn21-stage-r2.py",
        9185, "0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d"),
}

PROTECTED_CHECKPOINTS = (
    ("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/"
     "out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/sandbox/out/final.pt",
     469834849, "aaa4b7c0ff58c56df564954c21f8abdd786930bfaa1cc447f6fb8e5b0da83535",
     66305, 2947631),
    ("/home/ubuntu/trainium_freshstart_20260924_c595_matrixlr015_confirm_trn21_r2/"
     "out/fs-20260924-c595-matrixlr015-confirm-trn21-r2-work/train/out/final.pt",
     469831009, "5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9",
     66305, 2938565),
)
CACHE_ROOT = ("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn21_r1/"
              "out/fs-20260924-v2-35c9-lr015-control-trn21-r1-work/caches/train")
PROTECTED_CACHE_INODES = {
    "": 2945950,
    "neff": 2945951,
    "local": 2945952,
    "hlo": 2945953,
    "inductor": 2945954,
    "compiler": 2945955,
    "debug": 2945956,
}
PROTECTED_CACHE_DEVICE = 66305

SAFE_NAME = re.compile(r"^[A-Za-z0-9._ /-]+$")
SECRET_MARKERS = (
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"Bearer[ \t]+eyJ[A-Za-z0-9_-]{24,}"),
)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, CHUNK)
        if not chunk:
            return size, digest.hexdigest()
        size += len(chunk)
        digest.update(chunk)


def _read_exact(stream, count):
    pieces = []
    remaining = count
    while remaining:
        chunk = stream.read(min(remaining, CHUNK))
        if not chunk:
            raise RuntimeError("truncated tar stream")
        pieces.append(chunk)
        remaining -= len(chunk)
    return b"".join(pieces)


def _skip_exact(stream, count):
    while count:
        chunk = stream.read(min(count, CHUNK))
        if not chunk:
            raise RuntimeError("truncated tar payload")
        count -= len(chunk)


def _octal(field):
    if field and field[0] & 0x80:
        raise RuntimeError("base-256 tar numeric field is not accepted")
    value = field.split(b"\0", 1)[0].strip(b" ")
    if not value:
        return 0
    if any(byte not in b"01234567" for byte in value):
        raise RuntimeError("non-octal tar numeric field")
    return int(value, 8)


def _safe_path(name, is_directory):
    if name in (".", "./") and is_directory:
        return "."
    if not name.startswith("./") or not SAFE_NAME.fullmatch(name):
        raise RuntimeError("tar path is absolute, ambiguous or contains unsafe characters")
    tail = name[2:]
    if is_directory and tail.endswith("/"):
        tail = tail[:-1]
    components = tail.split("/")
    if not tail or any(component in ("", ".", "..") for component in components):
        raise RuntimeError("tar path has empty, dot or traversal component")
    canonical = "/".join(components)
    lower = [component.lower() for component in components]
    basename = lower[-1]
    if lower[0] in ("data", "dataset", "datasets", "out", "runs", ".cache"):
        raise RuntimeError("data/output-like top-level tar path")
    if any(component in (".aws", ".ssh", ".gnupg") for component in lower):
        raise RuntimeError("credential directory in tar path")
    if (basename in ("id_rsa", "id_ed25519", "credentials", ".netrc", "config.json") or
            basename.endswith((".pem", ".key", ".p12", ".pfx", ".pt", ".pth",
                               ".ckpt", ".safetensors")) or
            (basename.startswith(".env") and basename != ".env.example")):
        raise RuntimeError("credential or checkpoint-like tar path")
    if canonical.startswith(".git/hooks/") and not basename.endswith(".sample"):
        raise RuntimeError("active Git hook in tar path")
    if (canonical in (".git/config.worktree", ".git/objects/info/alternates",
                      ".git/objects/info/http-alternates", ".gitmodules", ".gitattributes") or
            canonical.startswith((".git/modules/", ".git/refs/remotes/"))):
        raise RuntimeError("unsafe Git override or remote path")
    return canonical


def _raw_entries(stream):
    """Count raw headers too, so tarfile cannot hide a PAX/control record."""
    stream.seek(0)
    entries = []
    long_path = None
    long_headers = 0
    with gzip.GzipFile(fileobj=stream, mode="rb") as archive:
        while True:
            header = _read_exact(archive, 512)
            if header == b"\0" * 512:
                if _read_exact(archive, 512) != b"\0" * 512:
                    raise RuntimeError("tar trailer is not two zero blocks")
                while True:
                    tail = archive.read(CHUNK)
                    if not tail:
                        break
                    if any(tail):
                        raise RuntimeError("nonzero bytes after tar trailer")
                break
            recorded_checksum = _octal(header[148:156])
            actual_checksum = sum(header[:148]) + 8 * 32 + sum(header[156:])
            if recorded_checksum != actual_checksum:
                raise RuntimeError("tar header checksum mismatch")
            size = _octal(header[124:136])
            mode = _octal(header[100:108])
            kind = header[156:157]
            if size > MAX_MEMBER_BYTES or mode & 0o7000:
                raise RuntimeError("oversized or privileged tar member")
            raw_name = header[:100].split(b"\0", 1)[0]
            prefix = header[345:500].split(b"\0", 1)[0]
            if prefix:
                raw_name = prefix + b"/" + raw_name
            try:
                name = raw_name.decode("ascii")
            except UnicodeDecodeError as exc:
                raise RuntimeError("non-ASCII tar path") from exc
            padded = (size + 511) // 512 * 512
            if kind == b"L":
                if name != "././@LongLink" or long_path is not None or not 1 <= size <= 4096:
                    raise RuntimeError("unexpected GNU long-name header")
                body = _read_exact(archive, padded)
                value = body[:size]
                if not value.endswith(b"\0") or b"\0" in value[:-1] or any(body[size:]):
                    raise RuntimeError("malformed GNU long-name payload")
                try:
                    long_path = value[:-1].decode("ascii")
                except UnicodeDecodeError as exc:
                    raise RuntimeError("non-ASCII GNU long name") from exc
                long_headers += 1
                continue
            if kind not in (b"0", b"\0", b"5"):
                raise RuntimeError("link, special, PAX or unknown tar header")
            if kind == b"5" and size:
                raise RuntimeError("directory tar header has payload")
            effective_name = long_path if long_path is not None else name
            long_path = None
            canonical = _safe_path(effective_name, kind == b"5")
            entries.append((canonical, "directory" if kind == b"5" else "file", size, mode))
            if len(entries) > EXPECTED_MEMBERS:
                raise RuntimeError("too many tar members")
            _skip_exact(archive, padded)
    if long_path is not None:
        raise RuntimeError("unpaired GNU long-name header")
    return entries, long_headers


def scan_archive(stream, *, pinned=True):
    """Read the whole archive before any checkout creation; never extract here."""
    raw, long_headers = _raw_entries(stream)
    seen = set()
    directories = set()
    files = 0
    dirs = 0
    unpacked = 0
    observed = {}
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r:gz", errorlevel=2) as archive:
        if archive.pax_headers:
            raise RuntimeError("global PAX metadata is forbidden")
        for index, member in enumerate(archive):
            if index >= len(raw) or member.pax_headers:
                raise RuntimeError("PAX override or member-count mismatch")
            if member.isfile():
                kind = "file"
            elif member.isdir():
                kind = "directory"
            else:
                raise RuntimeError("tar link or special member")
            canonical = _safe_path(member.name, kind == "directory")
            metadata = (canonical, kind, member.size, member.mode)
            if metadata != raw[index] or member.linkname or getattr(member, "sparse", None):
                raise RuntimeError("tar parser disagrees with raw safe-member scan")
            if canonical in seen:
                raise RuntimeError("duplicate tar path")
            if canonical == ".":
                if index != 0 or kind != "directory":
                    raise RuntimeError("tar root member must be the first directory")
            else:
                parent = canonical.rsplit("/", 1)[0] if "/" in canonical else "."
                if parent not in directories:
                    raise RuntimeError("tar member parent is absent or not a directory")
            seen.add(canonical)
            if kind == "directory":
                dirs += 1
                directories.add(canonical)
                continue
            files += 1
            unpacked += member.size
            if unpacked > MAX_TOTAL_UNPACKED_BYTES:
                raise RuntimeError("tar unpacked byte bound exceeded")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("tar regular member cannot be read")
            digest = hashlib.sha256()
            captured = bytearray()
            tail = b""
            count = 0
            while True:
                chunk = source.read(CHUNK)
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
                if canonical in (".git/config", ".git/HEAD", ".git/shallow"):
                    if len(captured) + len(chunk) > 4096:
                        raise RuntimeError("Git metadata text file is unexpectedly large")
                    captured.extend(chunk)
                if not canonical.startswith(".git/objects/pack/"):
                    joined = tail + chunk
                    if any(marker.search(joined) for marker in SECRET_MARKERS):
                        raise RuntimeError("high-confidence secret marker in tar member")
                    tail = joined[-128:]
            if count != member.size:
                raise RuntimeError("truncated tar member payload")
            if canonical in (".git/config", ".git/HEAD", ".git/shallow",
                             "train.py", "prepare.py"):
                observed[canonical] = {"bytes": count, "sha256": digest.hexdigest()}
            if canonical == ".git/config":
                compact = re.sub(rb"\s+", b"", bytes(captured).lower())
                if any(token in compact for token in
                       (b"[remote", b"[filter", b"[include", b"credential",
                        b"hookspath", b"sshcommand", b"worktree=", b"fsmonitor")):
                    raise RuntimeError("unsafe Git configuration")
        if len(seen) != len(raw):
            raise RuntimeError("tar parser skipped or added a member")
    if pinned:
        if (len(raw), files, dirs, long_headers, unpacked) != (
                EXPECTED_MEMBERS, EXPECTED_FILES, EXPECTED_DIRECTORIES,
                EXPECTED_GNU_LONG_HEADERS, EXPECTED_UNPACKED_BYTES):
            raise RuntimeError("pinned tar count or size mismatch")
        expected = {
            ".git/config": GIT_CONFIG_SHA256,
            ".git/HEAD": hashlib.sha256((SOURCE_COMMIT + "\n").encode()).hexdigest(),
            ".git/shallow": hashlib.sha256((SOURCE_COMMIT + "\n").encode()).hexdigest(),
            "train.py": TRAIN_SHA256,
            "prepare.py": PREPARE_SHA256,
        }
        if set(observed) != set(expected):
            raise RuntimeError("pinned Git/science files missing from tar")
        if any(observed[name]["sha256"] != sha for name, sha in expected.items()):
            raise RuntimeError("pinned Git/science tar file hash mismatch")
    manifest_sha256 = hashlib.sha256(
        json.dumps(raw, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    if pinned and manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("pinned tar member manifest mismatch")
    return {
        "members": len(raw), "files": files, "directories": dirs,
        "gnu_long_headers": long_headers, "unpacked_bytes": unpacked,
        "manifest_sha256": manifest_sha256, "pinned_files": observed,
    }, raw


def _stat_record(info):
    return {
        "device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
        "gid": info.st_gid, "mode": oct(stat.S_IMODE(info.st_mode)),
        "nlink": info.st_nlink, "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
    }


def _private_directory(fd, owner_uid, device):
    info = os.fstat(fd)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_uid or
            info.st_dev != device or stat.S_IMODE(info.st_mode) != 0o700):
        raise RuntimeError("extraction directory is not private on the expected device")
    return info


def _open_relative_directory(root_fd, components, owner_uid, device):
    fd = os.dup(root_fd)
    try:
        _private_directory(fd, owner_uid, device)
        for component in components:
            next_fd = os.open(component, OPEN_DIR, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            _private_directory(fd, owner_uid, device)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _extract_archive(stream, root_fd, manifest, *, owner_uid):
    """Use openat/O_EXCL/O_NOFOLLOW only; leave every partial output on error."""
    root = _private_directory(root_fd, owner_uid, os.fstat(root_fd).st_dev)
    device = root.st_dev
    completed = 0
    bytes_written = 0
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r:gz", errorlevel=2) as archive:
        if archive.pax_headers:
            raise RuntimeError("PAX metadata appeared during extraction")
        for index, member in enumerate(archive):
            if index >= len(manifest) or member.pax_headers:
                raise RuntimeError("archive changed after safe-member scan")
            if member.isfile():
                kind = "file"
            elif member.isdir():
                kind = "directory"
            else:
                raise RuntimeError("archive gained a link or special member")
            canonical = _safe_path(member.name, kind == "directory")
            if (canonical, kind, member.size, member.mode) != manifest[index]:
                raise RuntimeError("archive member changed after safe-member scan")
            if canonical == ".":
                completed += 1
                continue
            components = canonical.split("/")
            parent_fd = _open_relative_directory(root_fd, components[:-1], owner_uid, device)
            try:
                name = components[-1]
                if kind == "directory":
                    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
                    directory_fd = os.open(name, OPEN_DIR, dir_fd=parent_fd)
                    try:
                        os.fchmod(directory_fd, 0o700)
                        _private_directory(directory_fd, owner_uid, device)
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    os.fsync(parent_fd)
                else:
                    mode = 0o700 if member.mode & 0o111 else 0o600
                    fd = os.open(name, OPEN_NEW, mode, dir_fd=parent_fd)
                    try:
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError("regular tar member cannot be read for extraction")
                        digest = hashlib.sha256()
                        count = 0
                        while True:
                            chunk = source.read(CHUNK)
                            if not chunk:
                                break
                            count += len(chunk)
                            if count > member.size:
                                raise RuntimeError("tar member exceeds its stated size")
                            digest.update(chunk)
                            view = memoryview(chunk)
                            while view:
                                written = os.write(fd, view)
                                if written <= 0:
                                    raise RuntimeError("zero-length extraction write")
                                view = view[written:]
                        if count != member.size:
                            raise RuntimeError("truncated member during extraction")
                        os.fchmod(fd, mode)
                        os.fsync(fd)
                        read_count, read_sha256 = _sha256_fd(fd)
                        opened = os.fstat(fd)
                        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                        if (read_count != count or read_sha256 != digest.hexdigest() or
                                not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode) or
                                opened.st_dev != device or opened.st_uid != owner_uid or
                                opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != mode or
                                (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino) or
                                stat.S_IMODE(named.st_mode) != mode):
                            raise RuntimeError("extracted file readback, type, inode or mode mismatch")
                        bytes_written += count
                    finally:
                        os.close(fd)
                    os.fsync(parent_fd)
                completed += 1
            finally:
                os.close(parent_fd)
        if completed != len(manifest):
            raise RuntimeError("archive ended before all scanned members were extracted")
    # Persist directory entries, including the new checkout, after all files pass.
    for canonical, kind, _size, _mode in reversed(manifest):
        if kind == "directory" and canonical != ".":
            fd = _open_relative_directory(root_fd, canonical.split("/"), owner_uid, device)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    os.fsync(root_fd)
    return {"members_extracted": completed, "file_bytes_written": bytes_written}


def _checked_file(path, *, expected_uid, expected_device=None, expected_inode=None,
                  expected_bytes=None, expected_sha256=None):
    before = os.lstat(path)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != expected_uid or
            before.st_nlink != 1 or
            (expected_device is not None and before.st_dev != expected_device) or
            (expected_inode is not None and before.st_ino != expected_inode) or
            (expected_bytes is not None and before.st_size != expected_bytes)):
        raise RuntimeError("protected or pinned file identity mismatch")
    fd = os.open(path, OPEN_READ)
    try:
        opened = os.fstat(fd)
        if _stat_record(opened) != _stat_record(before):
            raise RuntimeError("file changed between lstat and no-follow open")
        count, digest = _sha256_fd(fd)
        after = os.fstat(fd)
        named = os.lstat(path)
        if (_stat_record(after) != _stat_record(opened) or
                _stat_record(named) != _stat_record(opened) or count != opened.st_size or
                (expected_sha256 is not None and digest != expected_sha256)):
            raise RuntimeError("file changed or hash mismatched during read")
        result = _stat_record(opened)
        result["sha256"] = digest
        return result
    finally:
        os.close(fd)


def _checked_cache_directory(path, expected_inode):
    before = os.lstat(path)
    if (not stat.S_ISDIR(before.st_mode) or before.st_uid != UID or
            before.st_dev != PROTECTED_CACHE_DEVICE or before.st_ino != expected_inode):
        raise RuntimeError("protected warm-cache directory identity mismatch")
    fd = os.open(path, OPEN_DIR)
    try:
        opened = os.fstat(fd)
        after = os.lstat(path)
        if _stat_record(opened) != _stat_record(before) or _stat_record(after) != _stat_record(before):
            raise RuntimeError("protected warm-cache directory changed during check")
        return _stat_record(before)
    finally:
        os.close(fd)


def snapshot_protected_assets():
    """Read-only checkpoint hashes and the seven pinned cache-directory identities."""
    checkpoints = {}
    for path, size, digest, device, inode in PROTECTED_CHECKPOINTS:
        checkpoints[path] = _checked_file(
            path, expected_uid=UID, expected_device=device, expected_inode=inode,
            expected_bytes=size, expected_sha256=digest)
    caches = {}
    for child, inode in PROTECTED_CACHE_INODES.items():
        path = CACHE_ROOT + ("/" + child if child else "")
        caches[path] = _checked_cache_directory(path, inode)
    return {"checkpoints": checkpoints, "warm_cache_directories": caches}


def _mount_for_home():
    matches = []
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            before, separator, after = line.partition(" - ")
            if not separator:
                raise RuntimeError("malformed mountinfo")
            left = before.split()
            right = after.split()
            if len(left) < 5 or len(right) < 2:
                raise RuntimeError("malformed mountinfo fields")
            point = left[4]
            if point == HOME or HOME.startswith(point.rstrip("/") + "/"):
                matches.append((len(point), {
                    "mount_id": left[0], "major_minor": left[2],
                    "mount_point": point, "filesystem": right[0], "source": right[1],
                }))
    if not matches:
        raise RuntimeError("no mount covers the checkout parent")
    mount = max(matches, key=lambda row: row[0])[1]
    if mount["filesystem"] != "ext4" or mount["mount_point"] != "/":
        raise RuntimeError("checkout parent is not on the expected root ext4 mount")
    return mount


def host_identity(parent_fd):
    if socket.gethostname() != HOST or os.geteuid() != UID:
        raise RuntimeError("wrong host or effective UID for trn21 stage")
    ancestors = (("/", 0), ("/home", 0), (HOME, UID))
    for path, owner_uid in ancestors:
        info = os.lstat(path)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_uid or
                stat.S_IMODE(info.st_mode) & 0o022):
            raise RuntimeError("unsafe checkout parent ancestry")
    parent = os.fstat(parent_fd)
    named = os.lstat(HOME)
    if ((parent.st_dev, parent.st_ino) != (named.st_dev, named.st_ino) or
            not stat.S_ISDIR(parent.st_mode) or parent.st_uid != UID):
        raise RuntimeError("checkout parent path changed")
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        boot = stream.read().strip()
    if not boot:
        raise RuntimeError("missing host boot ID")
    free = os.statvfs(HOME)
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes < ROOT_FREE_FLOOR:
        raise RuntimeError("checkout would breach the 10-GiB source-stage floor")
    return {
        "hostname": socket.gethostname(), "uid": os.geteuid(), "gid": os.getegid(),
        "boot_id": boot, "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "mount": _mount_for_home(), "parent_device": parent.st_dev,
        "parent_inode": parent.st_ino, "parent_uid": parent.st_uid,
        "parent_gid": parent.st_gid, "parent_mode": oct(stat.S_IMODE(parent.st_mode)),
        "root_free_bytes": free_bytes,
    }


STABLE_IDENTITY_KEYS = (
    "hostname", "uid", "gid", "boot_id", "mount_namespace", "mount",
    "parent_device", "parent_inode", "parent_uid", "parent_gid", "parent_mode",
)


def _same_identity(left, right):
    if (not isinstance(left, dict) or not isinstance(right, dict) or
            any(key not in left or key not in right or
                left[key] is None or right[key] is None for key in STABLE_IDENTITY_KEYS)):
        raise RuntimeError("host, boot, mount or parent identity is incomplete")
    if any(left.get(key) != right.get(key) for key in STABLE_IDENTITY_KEYS):
        raise RuntimeError("host, boot, mount or parent identity changed")


def _receipt_time(receipt):
    try:
        timestamp = datetime.datetime.fromisoformat(receipt["utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("missing or malformed transfer timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != datetime.timedelta(0):
        raise RuntimeError("transfer timestamp is not UTC")
    return timestamp


def validate_persistence_bundle(bundle):
    """Require all nine raw writer receipts; independent transport review remains external."""
    if not isinstance(bundle, dict) or set(bundle) != set(TRANSFER_ARTIFACTS):
        raise RuntimeError("complete three-artifact persistence bundle required")
    baseline_identity = None
    seen_processes = set()
    capsule_final = None
    target_stats = {}
    first_verifications = []
    final_verifications = []
    for artifact, (path, expected_bytes, expected_sha256) in TRANSFER_ARTIFACTS.items():
        group = bundle[artifact]
        if not isinstance(group, dict) or set(group) != {"write", "verify", "final_verify"}:
            raise RuntimeError("each artifact needs write and two distinct verifications")
        receipts = [group[key] for key in ("write", "verify", "final_verify")]
        times = []
        target = None
        for index, receipt in enumerate(receipts):
            operation = "write" if index == 0 else "verify"
            if (not isinstance(receipt, dict) or receipt.get("schema") != TRANSFER_SCHEMA or
                    receipt.get("status") != "HASH_VERIFIED_IN_THIS_SESSION_ONLY" or
                    receipt.get("operation") != operation or receipt.get("artifact") != artifact or
                    receipt.get("path") != path or receipt.get("expected_bytes") != expected_bytes or
                    receipt.get("expected_sha256") != expected_sha256 or
                    receipt.get("readback_sha256") != expected_sha256 or
                    receipt.get("streamed_sha256") != (expected_sha256 if index == 0 else None)):
                raise RuntimeError("transfer receipt is not for the exact persistent artifact")
            metadata = receipt.get("target_stat")
            if (not isinstance(metadata, dict) or metadata.get("bytes") != expected_bytes or
                    metadata.get("mode") != "0o600" or metadata.get("nlink") != 1 or
                    metadata.get("uid") != UID or not isinstance(metadata.get("device"), int) or
                    not isinstance(metadata.get("inode"), int)):
                raise RuntimeError("transfer target metadata is incomplete or unsafe")
            if target is None:
                target = metadata
            elif metadata != target:
                raise RuntimeError("persistent artifact inode or metadata changed")
            before = receipt.get("identity_before")
            after = receipt.get("identity_after")
            if not isinstance(before, dict) or not isinstance(after, dict):
                raise RuntimeError("transfer host identity is missing")
            _same_identity(before, after)
            mount = after.get("mount")
            if (after.get("hostname") != HOST or after.get("uid") != UID or
                    not isinstance(after.get("boot_id"), str) or not after["boot_id"] or
                    not isinstance(after.get("mount_namespace"), str) or
                    not after["mount_namespace"].startswith("mnt:[") or
                    not isinstance(mount, dict) or mount.get("filesystem") != "ext4" or
                    mount.get("mount_point") != "/" or not mount.get("source") or
                    after.get("parent_device") != metadata["device"] or
                    after.get("parent_uid") != UID or
                    not isinstance(after.get("parent_inode"), int) or
                    before.get("root_free_bytes", 0) < ROOT_FREE_FLOOR or
                    after.get("root_free_bytes", 0) < ROOT_FREE_FLOOR):
                raise RuntimeError("transfer host, ext4 or disk-floor evidence failed")
            if baseline_identity is None:
                baseline_identity = after
            else:
                _same_identity(baseline_identity, after)
            identity = (receipt.get("process_id"), receipt.get("process_start_ticks"))
            if (not all(isinstance(value, int) and value > 0 for value in identity) or
                    identity in seen_processes):
                raise RuntimeError("transfer operations did not use distinct process identities")
            seen_processes.add(identity)
            times.append(_receipt_time(receipt))
        if not times[0] < times[1] < times[2]:
            raise RuntimeError("cross-session readbacks are not later than the write")
        first_verifications.append(times[1])
        final_verifications.append(times[2])
        target_stats[artifact] = target
        if artifact == "capsule":
            capsule_final = receipts[2]
    if min(final_verifications) <= max(first_verifications):
        raise RuntimeError("final read-only pass did not occur after all first verifications")
    return {"host_identity": baseline_identity,
            "capsule_target_stat": capsule_final["target_stat"],
            "target_stats": target_stats,
            "final_capsule_verify_utc": capsule_final["utc"]}


def _target_stat(fd, parent_fd, name):
    opened = os.fstat(fd)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode) or
            (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino) or
            opened.st_dev != os.fstat(parent_fd).st_dev or opened.st_uid != UID or
            opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != 0o600 or
            stat.S_IMODE(named.st_mode) != 0o600):
        raise RuntimeError("persistent capsule is not the same private ext4 regular file")
    return {
        "device": opened.st_dev, "inode": opened.st_ino, "bytes": opened.st_size,
        "uid": opened.st_uid, "gid": opened.st_gid,
        "mode": oct(stat.S_IMODE(opened.st_mode)), "nlink": opened.st_nlink,
    }


def _verify_transferred_tool(parent_fd, artifact, expected_stat):
    path, expected_bytes, expected_sha256 = TRANSFER_ARTIFACTS[artifact]
    name = path.rsplit("/", 1)[1]
    fd = os.open(name, OPEN_READ, dir_fd=parent_fd)
    try:
        before = _stat_record(os.fstat(fd))
        target = _target_stat(fd, parent_fd, name)
        bytes_read, digest = _sha256_fd(fd)
        after = _stat_record(os.fstat(fd))
        if (target != expected_stat or before != after or
                (bytes_read, digest) != (expected_bytes, expected_sha256) or
                _target_stat(fd, parent_fd, name) != target):
            raise RuntimeError("transferred CPU tool changed after persistence verification")
        return {"path": path, "bytes": bytes_read, "sha256": digest,
                "target_stat": target}
    finally:
        os.close(fd)


def _must_be_absent(path):
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError("exact new checkout or science output path is already present")


def _git(checkout, *arguments):
    command = ["git", "-C", checkout, *arguments]
    # Ignore inherited GIT_* and user/system configuration. No shell, fetch,
    # checkout, hooks, filters, remote or credential helper is invoked.
    environment = {
        "PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat",
    }
    result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False, timeout=180)
    if len(result.stdout) > 65536 or len(result.stderr) > 65536:
        raise RuntimeError("Git verification output exceeded the bounded receipt size")
    record = {
        "argv": command, "returncode": result.returncode,
        "stdout": result.stdout.decode("utf-8", "replace"),
        "stderr": result.stderr.decode("utf-8", "replace"),
    }
    if result.returncode:
        raise RuntimeError("Git verification command failed: " + " ".join(arguments))
    return record


def verify_checkout(checkout=CHECKOUT_PATH, *, owner_uid=UID):
    """Read-only exact Git/source check, including ignored untracked files."""
    root = os.lstat(checkout)
    if (not stat.S_ISDIR(root.st_mode) or root.st_uid != owner_uid or
            stat.S_IMODE(root.st_mode) != 0o700):
        raise RuntimeError("new checkout is not an owner-private directory")
    git_dir = os.lstat(checkout + "/.git")
    if not stat.S_ISDIR(git_dir.st_mode) or git_dir.st_uid != owner_uid:
        raise RuntimeError("extracted .git is not a real owned directory")
    for forbidden in (".git/config.worktree", ".git/objects/info/alternates",
                      ".git/objects/info/http-alternates", ".gitmodules", ".gitattributes"):
        _must_be_absent(checkout + "/" + forbidden)
    _must_be_absent(checkout + "/out/fs-20260925-v2-c5c-a0-control-trn21-r1")
    metadata = {
        "HEAD": _checked_file(checkout + "/.git/HEAD", expected_uid=owner_uid,
                              expected_sha256=hashlib.sha256((SOURCE_COMMIT + "\n").encode()).hexdigest()),
        "shallow": _checked_file(checkout + "/.git/shallow", expected_uid=owner_uid,
                                 expected_sha256=hashlib.sha256((SOURCE_COMMIT + "\n").encode()).hexdigest()),
        "config": _checked_file(checkout + "/.git/config", expected_uid=owner_uid,
                                expected_sha256=GIT_CONFIG_SHA256),
        "train.py": _checked_file(checkout + "/train.py", expected_uid=owner_uid,
                                  expected_sha256=TRAIN_SHA256),
        "prepare.py": _checked_file(checkout + "/prepare.py", expected_uid=owner_uid,
                                    expected_sha256=PREPARE_SHA256),
    }
    commands = {}
    commands["head"] = _git(checkout, "rev-parse", "HEAD")
    if commands["head"]["stdout"].strip() != SOURCE_COMMIT:
        raise RuntimeError("checkout HEAD is not the pinned e65 commit")
    commands["shallow"] = _git(checkout, "rev-parse", "--is-shallow-repository")
    if commands["shallow"]["stdout"].strip() != "true":
        raise RuntimeError("checkout is not a shallow repository")
    commands["head_count"] = _git(checkout, "rev-list", "--count", "HEAD")
    commands["all_count"] = _git(checkout, "rev-list", "--count", "--all")
    if (commands["head_count"]["stdout"].strip() != "1" or
            commands["all_count"]["stdout"].strip() != "1"):
        raise RuntimeError("checkout does not have exactly one reachable commit")
    commands["remotes"] = _git(checkout, "remote")
    if commands["remotes"]["stdout"].strip():
        raise RuntimeError("checkout has an unexpected Git remote")
    commands["fsck"] = _git(checkout, "fsck", "--full", "--strict", "--no-reflogs",
                            "--connectivity-only", "--no-progress")
    commands["status"] = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    commands["all_untracked"] = _git(checkout, "ls-files", "--others", "--full-name")
    if commands["status"]["stdout"] or commands["all_untracked"]["stdout"]:
        raise RuntimeError("checkout has tracked, untracked or ignored changes")
    root_after = os.lstat(checkout)
    if (not stat.S_ISDIR(root_after.st_mode) or
            (root_after.st_dev, root_after.st_ino, root_after.st_uid,
             stat.S_IMODE(root_after.st_mode)) !=
            (root.st_dev, root.st_ino, root.st_uid, stat.S_IMODE(root.st_mode))):
        raise RuntimeError("checkout root changed during Git verification")
    return {
        "checkout_path": checkout, "checkout_stat": _stat_record(root),
        "source_files": metadata, "git_checks": commands,
        "science_output_absent": True,
    }


def stage_fixed_checkout(persistence_bundle):
    """Later-stage API. The current transfer-only assignment forbids calling it.

    The caller must separately archive and independently review each raw SSH
    transport result, fresh full preflight, this exact code hash and the new
    one-shot stage request. Receipt consistency here is an additional gate,
    not authentication of the SSH transport itself.
    """
    proof = validate_persistence_bundle(persistence_bundle)
    parent_fd = os.open(HOME, OPEN_DIR)
    capsule_fd = None
    checkout_fd = None
    try:
        identity_before = host_identity(parent_fd)
        _same_identity(identity_before, proof["host_identity"])
        _must_be_absent(CHECKOUT_PATH)
        _must_be_absent(SCIENCE_OUTPUT)
        assets_before = snapshot_protected_assets()
        capsule_fd = os.open(CAPSULE_NAME, OPEN_READ, dir_fd=parent_fd)
        capsule_stat_before = _stat_record(os.fstat(capsule_fd))
        target = _target_stat(capsule_fd, parent_fd, CAPSULE_NAME)
        if target != proof["capsule_target_stat"] or target["bytes"] != CAPSULE_BYTES:
            raise RuntimeError("capsule inode differs from the final cross-session proof")
        observed_bytes, observed_sha256 = _sha256_fd(capsule_fd)
        if (observed_bytes, observed_sha256) != (CAPSULE_BYTES, CAPSULE_SHA256):
            raise RuntimeError("persistent capsule SHA-256 or byte count drift")
        tools_before = {artifact: _verify_transferred_tool(
            parent_fd, artifact, proof["target_stats"][artifact]
        ) for artifact in ("preflight", "contained")}
        with os.fdopen(os.dup(capsule_fd), "rb") as stream:
            archive_scan, manifest = scan_archive(stream, pinned=True)
        identity_after_scan = host_identity(parent_fd)
        _same_identity(identity_before, identity_after_scan)
        if snapshot_protected_assets() != assets_before:
            raise RuntimeError("protected asset changed during archive scan")
        _must_be_absent(CHECKOUT_PATH)
        # mkdir and every child creation are exclusive. No error path removes
        # the directory or a partially written file.
        os.mkdir(CHECKOUT_NAME, mode=0o700, dir_fd=parent_fd)
        checkout_fd = os.open(CHECKOUT_NAME, OPEN_DIR, dir_fd=parent_fd)
        os.fchmod(checkout_fd, 0o700)
        _private_directory(checkout_fd, UID, os.fstat(parent_fd).st_dev)
        os.fsync(parent_fd)
        with os.fdopen(os.dup(capsule_fd), "rb") as stream:
            extraction = _extract_archive(stream, checkout_fd, manifest, owner_uid=UID)
        target_after = _target_stat(capsule_fd, parent_fd, CAPSULE_NAME)
        capsule_stat_after = _stat_record(os.fstat(capsule_fd))
        post_bytes, post_sha256 = _sha256_fd(capsule_fd)
        if (target_after != target or capsule_stat_after != capsule_stat_before or
                (post_bytes, post_sha256) != (CAPSULE_BYTES, CAPSULE_SHA256)):
            raise RuntimeError("persistent capsule changed during extraction")
        checkout_named = os.stat(CHECKOUT_NAME, dir_fd=parent_fd, follow_symlinks=False)
        if (checkout_named.st_dev, checkout_named.st_ino) != (
                os.fstat(checkout_fd).st_dev, os.fstat(checkout_fd).st_ino):
            raise RuntimeError("new checkout path changed during extraction")
        checkout = verify_checkout(CHECKOUT_PATH, owner_uid=UID)
        checkout_named_after = os.stat(CHECKOUT_NAME, dir_fd=parent_fd, follow_symlinks=False)
        if (checkout_named_after.st_dev, checkout_named_after.st_ino) != (
                os.fstat(checkout_fd).st_dev, os.fstat(checkout_fd).st_ino):
            raise RuntimeError("checkout path changed during Git verification")
        assets_after = snapshot_protected_assets()
        if assets_after != assets_before:
            raise RuntimeError("protected checkpoint or warm-cache identity changed")
        tools_after = {artifact: _verify_transferred_tool(
            parent_fd, artifact, proof["target_stats"][artifact]
        ) for artifact in ("preflight", "contained")}
        if tools_after != tools_before:
            raise RuntimeError("transferred CPU tool identity changed during source stage")
        identity_after = host_identity(parent_fd)
        _same_identity(identity_before, identity_after)
        _same_identity(identity_after, proof["host_identity"])
        if checkout["checkout_stat"]["device"] != identity_after["parent_device"]:
            raise RuntimeError("checkout is not on the authenticated ext4 parent device")
        _must_be_absent(SCIENCE_OUTPUT)
        return {
            "schema": SCHEMA,
            "status": "SOURCE_CHECKOUT_VERIFIED_FOR_INDEPENDENT_REVIEW_ONLY",
            "utc": _utc(),
            "source_contract_sha256": SOURCE_CONTRACT_SHA256,
            "transfer_assignment_sha256": TRANSFER_ASSIGNMENT_SHA256,
            "persistence_bundle_sha256": hashlib.sha256(json.dumps(
                persistence_bundle, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "capsule": {"path": CAPSULE_PATH, "bytes": post_bytes,
                        "sha256": post_sha256, "target_stat": target_after},
            "archive_scan": archive_scan, "extraction": extraction,
            "transferred_tools_before": tools_before,
            "transferred_tools_after": tools_after,
            "checkout": checkout,
            "host_identity_before": identity_before,
            "host_identity_after": identity_after,
            "protected_assets_before": assets_before,
            "protected_assets_after": assets_after,
            "cpu_g2_authorized": False, "native_training_authorized": False,
            "evaluation_authorized": False,
        }
    finally:
        if checkout_fd is not None:
            os.close(checkout_fd)
        if capsule_fd is not None:
            os.close(capsule_fd)
        os.close(parent_fd)


if __name__ == "__main__":
    raise SystemExit("DISPATCH HOLD: transfer-only r2 assignment forbids checkout/extraction")
