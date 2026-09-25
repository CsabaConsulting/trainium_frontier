"""Local-only builder/verifier for the frozen trn22 A0 r4 source capsule.

This module contains no SSH, host writer, checkout, monitor, launch or AWS
operation. Its outputs are an owner-private *local* tar.gz outside Git and a
text member manifest. Neither output is an Operator assignment.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile


SOURCE_COMMIT = "446b657695f086f9b484fa601c9a3b61c2495912"
SOURCE_BRANCH = "codex/20260925-a0-trn22-r4-interlock"
LOCAL_CAPSULE = Path(
    "/home/poojith-devan/Desktop/Trainium-source-artifacts/"
    "trainium-c5c-r4-446b-one-commit-20260925.tar.gz"
)
MEMBER_MANIFEST = Path(__file__).with_name("MEMBERS.jsonl")
EXPECTED_CAPSULE = (
    104154875,
    "767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a",
)
EXPECTED_MANIFEST = (
    1229617,
    "7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a",
)
EXPECTED_TRACKED_FILES = 6278
EXPECTED_SHA256 = {
    "train.py": "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586",
    "prepare.py": "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb",
    "research/fresh_start_backup.py": "c978fd20b6620d0d80c25f8982974e8c4b2ad009edd9da376d784f5cecc3ead3",
    "research/fresh_start_guard.py": "df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc",
    "research/fresh_start_host.py": "b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3",
    "research/fresh_start_launch.py": "9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4",
    "research/fresh_start_monitor_interlock.py": "f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d",
    "research/fresh_start_sync.py": "ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c",
    "research/run_experiment.py": "88243a74f0f080929f93d0b88ddedbe6f5351ad37bf272229d9ae6d26efd5d2d",
    "research/v2/disk-monitor/root_free_guard_monitor.py": "91dae71468d5d029857371ccd2c960769b71874e855bb058130e738b91bb2c60",
    "research/v2/disk-monitor/paired_root_free_guard_monitor.py": "9e2d6ed26d918131c3e12b9937093e465b39aa895a9bf2be7339de8ae5094344",
}
CHUNK = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9._ /-]+$")
SECRET_MARKERS = {
    "aws_access_id": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "openai_project_key": re.compile(rb"sk-proj-[A-Za-z0-9_-]{20,}"),
    "private_key_pem": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer_jwt": re.compile(rb"Bearer[ \t]+eyJ[A-Za-z0-9_-]{24,}"),
    "bedrock_bearer_assignment": re.compile(
        rb"AWS_BEARER_TOKEN_BEDROCK[ \t]*[:=][ \t]*[\"']?[A-Za-z0-9_.-]{25,}"
    ),
}
OPEN_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
OPEN_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _git(clone, *args):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update({
        "PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
    })
    result = subprocess.run(["/usr/bin/git", "-C", str(clone), *args],
                            env=environment, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False, timeout=180)
    if result.returncode or len(result.stdout) > 2 * 1024 * 1024:
        raise RuntimeError("local exact-source Git check failed: " + args[0])
    return result.stdout


def _safe_path(relative, is_directory):
    if relative == "." and is_directory:
        return
    if not relative or relative.startswith("/") or not SAFE_NAME.fullmatch(relative):
        raise RuntimeError("unsafe archive member name")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError("empty or traversal archive component")
    lower = [part.lower() for part in parts]
    basename = lower[-1]
    if lower[0] in ("data", "dataset", "datasets", "out", "runs", ".cache"):
        raise RuntimeError("data or output directory is not source")
    if any(part in (".aws", ".ssh", ".gnupg") for part in lower):
        raise RuntimeError("credential directory in source")
    if (basename in ("id_rsa", "id_ed25519", "credentials", ".netrc", "config.json") or
            basename.endswith((".pem", ".key", ".p12", ".pfx", ".pt", ".pth",
                               ".ckpt", ".safetensors")) or
            (basename.startswith(".env") and basename != ".env.example")):
        raise RuntimeError("credential or checkpoint-like source member")
    if relative.startswith(".git/hooks/") and not basename.endswith(".sample"):
        raise RuntimeError("active Git hook in source")
    if (relative in (".git/config.worktree", ".git/objects/info/alternates",
                     ".git/objects/info/http-alternates", ".gitmodules",
                     ".gitattributes") or
            relative.startswith((".git/modules/", ".git/refs/remotes/"))):
        raise RuntimeError("unsafe Git override or remote path")


def _stat_key(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _digest_file(path, relative):
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError("source member is not a single-link regular file")
    if before.st_size > MAX_MEMBER_BYTES:
        raise RuntimeError("source member exceeds size cap")
    fd = os.open(path, OPEN_READ)
    try:
        opened = os.fstat(fd)
        if _stat_key(opened) != _stat_key(before):
            raise RuntimeError("source file changed between lstat and open")
        digest = hashlib.sha256()
        count = 0
        tail = b""
        while True:
            chunk = os.read(fd, CHUNK)
            if not chunk:
                break
            count += len(chunk)
            digest.update(chunk)
            joined = tail + chunk
            if any(pattern.search(joined) for pattern in SECRET_MARKERS.values()):
                raise RuntimeError("high-confidence secret marker in " + relative)
            tail = joined[-128:]
        if (count != opened.st_size or _stat_key(os.fstat(fd)) != _stat_key(opened) or
                _stat_key(os.lstat(path)) != _stat_key(opened)):
            raise RuntimeError("source file changed during safe scan")
        return count, digest.hexdigest(), _stat_key(opened)
    finally:
        os.close(fd)


def _tracked_modes(clone):
    modes = {}
    records = _git(clone, "ls-files", "--stage", "-z").split(b"\0")
    for record in records:
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, _object, stage = metadata.split(b" ")
        path = raw_path.decode("ascii")
        if mode not in (b"100644", b"100755") or stage != b"0" or path in modes:
            raise RuntimeError("non-regular, unresolved or duplicate Git index entry")
        _safe_path(path, False)
        modes[path] = 0o700 if mode == b"100755" else 0o600
    if len(modes) != EXPECTED_TRACKED_FILES:
        raise RuntimeError("frozen source tracked-file count changed")
    return modes


def validate_clone(clone):
    clone = Path(clone)
    if not stat.S_ISDIR(os.lstat(clone).st_mode):
        raise RuntimeError("clone is not a real directory")
    if _git(clone, "rev-parse", "HEAD").strip().decode() != SOURCE_COMMIT:
        raise RuntimeError("source clone HEAD is not frozen r4")
    if _git(clone, "rev-parse", "--is-shallow-repository").strip() != b"true":
        raise RuntimeError("source clone is not shallow")
    if (_git(clone, "rev-list", "--count", "HEAD").strip() != b"1" or
            _git(clone, "rev-list", "--count", "--all").strip() != b"1"):
        raise RuntimeError("source clone has more than one reachable commit")
    if (_git(clone, "remote") or
            _git(clone, "status", "--porcelain=v1", "--untracked-files=all") or
            _git(clone, "ls-files", "--others", "--full-name")):
        raise RuntimeError("source clone has a remote, dirt or untracked file")
    _git(clone, "fsck", "--full", "--strict", "--no-reflogs", "--no-progress")
    if (clone / ".git/HEAD").read_bytes() != (SOURCE_COMMIT + "\n").encode():
        raise RuntimeError("source clone is not detached at frozen r4")
    if (clone / ".git/shallow").read_bytes() != (SOURCE_COMMIT + "\n").encode():
        raise RuntimeError("source clone shallow boundary changed")
    config_keys = set(_git(clone, "config", "--local", "--name-only", "--list")
                      .decode("ascii").splitlines())
    if config_keys != {"core.repositoryformatversion", "core.filemode",
                       "core.bare", "core.logallrefupdates"}:
        raise RuntimeError("source clone Git config contains unexpected key")
    return _tracked_modes(clone)


def collect_members(clone, tracked_modes):
    clone = Path(clone)
    members = [(".", "directory", clone, 0o700)]
    observed_tracked = set()
    for current, directory_names, file_names in os.walk(clone, topdown=True,
                                                        followlinks=False):
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = Path(current, name)
            relative = path.relative_to(clone).as_posix()
            _safe_path(relative, True)
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                raise RuntimeError("source has a linked or special directory")
            members.append((relative, "directory", path, 0o700))
        for name in file_names:
            path = Path(current, name)
            relative = path.relative_to(clone).as_posix()
            _safe_path(relative, False)
            if relative.startswith(".git/"):
                mode = 0o600
            else:
                if relative not in tracked_modes:
                    raise RuntimeError("untracked source file in local clone")
                observed_tracked.add(relative)
                mode = tracked_modes[relative]
            members.append((relative, "file", path, mode))
    if observed_tracked != set(tracked_modes):
        raise RuntimeError("source worktree differs from exact Git index")
    members.sort(key=lambda row: (0 if row[0] == "." else row[0].count("/") + 1,
                                  row[0]))
    return members


class _HashingReader:
    def __init__(self, fd):
        self.fd = fd
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, count):
        chunk = os.read(self.fd, count)
        self.digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk


def _make_capsule(members, prescan):
    parent = LOCAL_CAPSULE.parent
    info = os.lstat(parent)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or
            stat.S_IMODE(info.st_mode) != 0o700):
        raise RuntimeError("local artifact directory must exist, be real and private")
    if LOCAL_CAPSULE.exists() or MEMBER_MANIFEST.exists():
        raise FileExistsError("local capsule or manifest already exists; no overwrite")
    output_fd = os.open(LOCAL_CAPSULE, OPEN_NEW, 0o600)
    records = []
    try:
        with os.fdopen(output_fd, "wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0,
                               compresslevel=6) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|",
                                  format=tarfile.GNU_FORMAT) as archive:
                    for relative, kind, path, mode in members:
                        item = tarfile.TarInfo("." if relative == "." else "./" + relative)
                        item.uid = item.gid = 0
                        item.uname = item.gname = ""
                        item.mtime = 0
                        item.mode = mode
                        if kind == "directory":
                            item.type = tarfile.DIRTYPE
                            archive.addfile(item)
                            records.append({"path": relative, "type": kind,
                                            "mode": oct(mode), "bytes": 0,
                                            "sha256": None})
                            continue
                        size, digest, before = prescan[relative]
                        item.type = tarfile.REGTYPE
                        item.size = size
                        fd = os.open(path, OPEN_READ)
                        try:
                            if _stat_key(os.fstat(fd)) != before:
                                raise RuntimeError("source file changed before archiving")
                            reader = _HashingReader(fd)
                            archive.addfile(item, reader)
                            if (reader.bytes_read != size or
                                    reader.digest.hexdigest() != digest or
                                    _stat_key(os.fstat(fd)) != before or
                                    _stat_key(os.lstat(path)) != before):
                                raise RuntimeError("source file changed during archiving")
                        finally:
                            os.close(fd)
                        records.append({"path": relative, "type": kind,
                                        "mode": oct(mode), "bytes": size,
                                        "sha256": digest})
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        # The unique partial local artifact is retained for diagnosis.
        raise
    return records


def _octal(field):
    if field and field[0] & 0x80:
        raise RuntimeError("base-256 tar numeric field forbidden")
    digits = field.split(b"\0", 1)[0].strip(b" ")
    if any(byte not in b"01234567" for byte in digits):
        raise RuntimeError("non-octal tar field")
    return int(digits or b"0", 8)


def _read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        chunk = stream.read(min(CHUNK, count - len(result)))
        if not chunk:
            raise RuntimeError("truncated capsule tar stream")
        result.extend(chunk)
    return bytes(result)


def _raw_scan():
    headers = []
    long_name = None
    long_headers = 0
    with LOCAL_CAPSULE.open("rb") as raw, gzip.GzipFile(fileobj=raw) as stream:
        while True:
            header = _read_exact(stream, 512)
            if header == b"\0" * 512:
                if _read_exact(stream, 512) != b"\0" * 512:
                    raise RuntimeError("tar trailer is not two zero blocks")
                while True:
                    tail = stream.read(CHUNK)
                    if not tail:
                        break
                    if any(tail):
                        raise RuntimeError("nonzero payload after tar trailer")
                break
            if _octal(header[148:156]) != sum(header[:148]) + 8 * 32 + sum(header[156:]):
                raise RuntimeError("tar header checksum mismatch")
            name = header[:100].split(b"\0", 1)[0]
            prefix = header[345:500].split(b"\0", 1)[0]
            if prefix:
                name = prefix + b"/" + name
            size, mode, kind = (_octal(header[124:136]), _octal(header[100:108]),
                                header[156:157])
            if size > MAX_MEMBER_BYTES or mode & 0o7000:
                raise RuntimeError("oversized or privileged tar member")
            padded = (size + 511) // 512 * 512
            if kind == b"L":
                if name != b"././@LongLink" or long_name is not None or not 1 <= size <= 4096:
                    raise RuntimeError("unexpected GNU long-name header")
                payload = _read_exact(stream, padded)
                value = payload[:size]
                if not value.endswith(b"\0") or b"\0" in value[:-1] or any(payload[size:]):
                    raise RuntimeError("malformed GNU long-name payload")
                long_name = value[:-1].decode("ascii")
                long_headers += 1
                continue
            if kind not in (b"0", b"\0", b"5") or (kind == b"5" and size):
                raise RuntimeError("tar link, PAX, special or unknown member")
            try:
                effective = long_name if long_name is not None else name.decode("ascii")
            except UnicodeDecodeError as exc:
                raise RuntimeError("non-ASCII tar member") from exc
            long_name = None
            directory = kind == b"5"
            if effective in (".", "./") and directory:
                relative = "."
            elif effective.startswith("./"):
                relative = effective[2:].rstrip("/") if directory else effective[2:]
            else:
                raise RuntimeError("absolute or ambiguous tar member")
            _safe_path(relative, directory)
            headers.append((relative, "directory" if directory else "file", size, mode))
            remaining = padded
            while remaining:
                chunk = _read_exact(stream, min(CHUNK, remaining))
                remaining -= len(chunk)
    if long_name is not None:
        raise RuntimeError("orphan GNU long-name header")
    return headers, long_headers


def verify_capsule(records):
    raw, long_headers = _raw_scan()
    if len(raw) != len(records):
        raise RuntimeError("raw tar header count differs from member manifest")
    seen = set()
    directories = set()
    file_bytes = 0
    files = 0
    with LOCAL_CAPSULE.open("rb") as stream, tarfile.open(
            fileobj=stream, mode="r:gz", errorlevel=2) as archive:
        if archive.pax_headers:
            raise RuntimeError("PAX archive metadata forbidden")
        for index, member in enumerate(archive):
            if index >= len(records) or member.pax_headers or member.linkname:
                raise RuntimeError("tar parser found PAX/link/extra member")
            kind = "directory" if member.isdir() else "file" if member.isfile() else None
            if kind is None:
                raise RuntimeError("tar parser found special member")
            if member.name not in (".", "./") and not member.name.startswith("./"):
                raise RuntimeError("tar parser found an ambiguous member name")
            relative = "." if member.name in (".", "./") and kind == "directory" else (
                member.name[2:].rstrip("/") if kind == "directory" else member.name[2:]
            )
            _safe_path(relative, kind == "directory")
            expected = records[index]
            if (relative, kind, member.size, member.mode) != raw[index] or (
                relative != expected["path"] or kind != expected["type"] or
                member.size != expected["bytes"] or oct(member.mode) != expected["mode"]
            ):
                raise RuntimeError("tar parser/member manifest mismatch")
            if relative in seen:
                raise RuntimeError("duplicate tar member")
            parent = relative.rsplit("/", 1)[0] if "/" in relative else "."
            if relative != "." and parent not in directories:
                raise RuntimeError("tar member parent was not an earlier directory")
            seen.add(relative)
            if kind == "directory":
                directories.add(relative)
                continue
            files += 1
            file_bytes += member.size
            if file_bytes > MAX_TOTAL_BYTES:
                raise RuntimeError("archive unpacked size exceeded cap")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("tar regular member has no payload")
            digest = hashlib.sha256()
            count = 0
            tail = b""
            while True:
                chunk = source.read(CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                count += len(chunk)
                joined = tail + chunk
                if any(pattern.search(joined) for pattern in SECRET_MARKERS.values()):
                    raise RuntimeError("high-confidence secret marker in capsule member " + relative)
                tail = joined[-128:]
            if count != member.size or digest.hexdigest() != expected["sha256"]:
                raise RuntimeError("tar payload differs from member manifest")
        if len(seen) != len(records):
            raise RuntimeError("tar parser skipped a member")
    return {"members": len(records), "files": files,
            "directories": len(records) - files,
            "gnu_long_headers": long_headers, "unpacked_file_bytes": file_bytes}


def _file_digest(path, *, private=False):
    named_before = os.lstat(path)
    if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
        raise RuntimeError("local capsule/manifest is not a regular single-link file")
    fd = os.open(path, OPEN_READ)
    try:
        info = os.fstat(fd)
        if (_stat_key(info) != _stat_key(named_before) or
                info.st_uid != os.geteuid() or
                (private and stat.S_IMODE(info.st_mode) != 0o600)):
            raise RuntimeError("local capsule/manifest owner, inode or mode changed")
        digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(fd, CHUNK)
            if not chunk:
                break
            count += len(chunk)
            digest.update(chunk)
        if (_stat_key(os.fstat(fd)) != _stat_key(info) or
                _stat_key(os.lstat(path)) != _stat_key(info)):
            raise RuntimeError("local capsule/manifest changed during hashing")
        return count, digest.hexdigest()
    finally:
        os.close(fd)


def _write_manifest(records):
    fd = os.open(MEMBER_MANIFEST, OPEN_NEW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True,
                                    separators=(",", ":")).encode("ascii") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_manifest():
    records = []
    fd = os.open(MEMBER_MANIFEST, OPEN_READ)
    with os.fdopen(fd, "rb") as stream:
        for line in stream:
            record = json.loads(line)
            if set(record) != {"path", "type", "mode", "bytes", "sha256"}:
                raise RuntimeError("member manifest schema changed")
            records.append(record)
    return records


def _source_inputs(clone):
    tracked_modes = validate_clone(clone)
    members = collect_members(clone, tracked_modes)
    prescan = {}
    total = 0
    for relative, kind, path, _mode in members:
        if kind == "file":
            result = _digest_file(path, relative)
            total += result[0]
            if total > MAX_TOTAL_BYTES:
                raise RuntimeError("source exceeds bounded unpacked-size cap")
            prescan[relative] = result
    for relative, expected in EXPECTED_SHA256.items():
        if prescan[relative][1] != expected:
            raise RuntimeError("frozen r4 source SHA-256 differs at " + relative)
    return tracked_modes, members, prescan


def _summary(tracked_modes, scan):
    capsule_bytes, capsule_sha256 = _file_digest(LOCAL_CAPSULE, private=True)
    manifest_bytes, manifest_sha256 = _file_digest(MEMBER_MANIFEST)
    _require_frozen_artifacts((capsule_bytes, capsule_sha256),
                              (manifest_bytes, manifest_sha256))
    return {"source_commit": SOURCE_COMMIT, "source_branch": SOURCE_BRANCH,
            "tracked_files": len(tracked_modes), "secret_marker_hits": 0,
            "local_capsule": str(LOCAL_CAPSULE), "capsule_bytes": capsule_bytes,
            "capsule_sha256": capsule_sha256, "member_manifest": str(MEMBER_MANIFEST),
            "manifest_bytes": manifest_bytes, "manifest_sha256": manifest_sha256,
            **scan}


def _require_frozen_artifacts(capsule, manifest):
    if capsule != EXPECTED_CAPSULE or manifest != EXPECTED_MANIFEST:
        raise RuntimeError("local capsule or member manifest differs from frozen r4 pins")


def build(clone):
    tracked_modes, members, prescan = _source_inputs(clone)
    records = _make_capsule(members, prescan)
    scan = verify_capsule(records)
    _write_manifest(records)
    return _summary(tracked_modes, scan)


def finish_existing(clone):
    """Finish the first local build after a verifier-only parser error.

    The capsule is never reopened for writing. Its full member payload must
    match a newly scanned immutable clone before the absent manifest is made.
    This is not a host-stage retry, cleanup or overwrite mechanism.
    """
    if MEMBER_MANIFEST.exists():
        raise FileExistsError("member manifest already exists; no overwrite")
    if _file_digest(LOCAL_CAPSULE, private=True) != EXPECTED_CAPSULE:
        raise RuntimeError("existing local capsule differs from frozen r4 pin")
    tracked_modes, members, prescan = _source_inputs(clone)
    records = []
    for relative, kind, _path, mode in members:
        size, digest = (0, None) if kind == "directory" else prescan[relative][:2]
        records.append({"path": relative, "type": kind, "mode": oct(mode),
                        "bytes": size, "sha256": digest})
    scan = verify_capsule(records)
    _write_manifest(records)
    return _summary(tracked_modes, scan)


def verify():
    records = _read_manifest()
    scan = verify_capsule(records)
    capsule_bytes, capsule_sha256 = _file_digest(LOCAL_CAPSULE, private=True)
    manifest_bytes, manifest_sha256 = _file_digest(MEMBER_MANIFEST)
    _require_frozen_artifacts((capsule_bytes, capsule_sha256),
                              (manifest_bytes, manifest_sha256))
    return {"source_commit": SOURCE_COMMIT,
            "capsule_bytes": capsule_bytes, "capsule_sha256": capsule_sha256,
            "manifest_bytes": manifest_bytes, "manifest_sha256": manifest_sha256,
            **scan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-from-private-clone", metavar="PATH")
    group.add_argument("--finish-existing-local-capsule-from-private-clone",
                       metavar="PATH")
    group.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()
    if args.build_from_private_clone:
        summary = build(args.build_from_private_clone)
    elif args.finish_existing_local_capsule_from_private_clone:
        summary = finish_existing(args.finish_existing_local_capsule_from_private_clone)
    else:
        summary = verify()
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
