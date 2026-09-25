"""Exact, fail-closed scratch bootstrap for the disabled A2 CPU G2 request.

The reviewed command carries these bytes as ``python -B -c <source>``; this
file is not staged to a host by the disabled package. ``prepare`` may be used
only after a fresh preflight and one-attempt marker in a separately enabled
request. ``run`` checks TMPDIR in the G2 launcher's own interpreter before
executing the unchanged, pinned CPU G2 tool with runpy. No fallback is allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import sys
import tempfile


PINNED_CHECKOUT = "/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2"
PINNED_ROOT = "/home/ubuntu/trainium-v2-g2-a2-c5c-m64d128-20260925-trn22-r3"
PINNED_TOOL = PINNED_CHECKOUT + "/workflow/tools/preflight_dryrun.py"
PINNED_TRAIN = PINNED_CHECKOUT + "/train.py"
PINNED_PREPARE = PINNED_CHECKOUT + "/prepare.py"
G2_SHA = "6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8"
TRAIN_SHA = "c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586"
PREPARE_SHA = "5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
NOFOLLOW_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _exact_paths(root: str, tmp: str) -> tuple[str, str]:
    require(os.path.isabs(root) and os.path.normpath(root) == root,
            "scratch root is not canonical absolute")
    require(tmp == root + "/tmp" and os.path.normpath(tmp) == tmp,
            "TMPDIR is not exact scratch child")
    parent = os.path.dirname(root)
    require(parent != "/" and os.path.realpath(parent) == parent,
            "scratch parent is symlinked or broad")
    require(os.environ.get("TMPDIR") == tmp, "clean child TMPDIR changed")
    return parent, os.path.basename(root)


def _owned_dir(fd: int, expected_device: int, *, name: str,
               required_mode: int | None = 0o700) -> os.stat_result:
    st = os.fstat(fd)
    require(stat.S_ISDIR(st.st_mode) and st.st_dev == expected_device,
            name + " is not a directory on the preflight filesystem")
    mode = stat.S_IMODE(st.st_mode)
    require(st.st_uid == os.getuid() and
            (mode == required_mode if required_mode is not None else not (mode & 0o002)),
            name + " ownership/mode changed")
    return st


def _open_existing(parent_fd: int, basename: str, device: int, *, name: str,
                   required_mode: int | None = 0o700) -> int:
    entry = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    require(stat.S_ISDIR(entry.st_mode), name + " is not a real directory")
    fd = os.open(basename, NOFOLLOW_DIR, dir_fd=parent_fd)
    st = _owned_dir(fd, device, name=name, required_mode=required_mode)
    require((entry.st_dev, entry.st_ino) == (st.st_dev, st.st_ino),
            name + " path/fd identity changed")
    return fd


def _positive_decimal(text: str, name: str) -> int:
    require(text.isdecimal(), name + " not bound to decimal bytes")
    number = int(text)
    require(number > 0, name + " must be positive")
    return number


def _free_space(parent: str, *, min_free: int, min_inodes: int,
                projection: int, floor: int) -> None:
    vfs = os.statvfs(parent)
    free = vfs.f_bavail * vfs.f_frsize
    inodes = vfs.f_favail
    require(free >= min_free and inodes >= min_inodes,
            "fresh scratch free bytes/inodes below hard floor")
    require(projection + floor <= free, "scratch projection plus protected floor does not fit")


def _inspect(root: str, tmp: str, *, create: bool, allow_process: bool,
             min_free: int, min_inodes: int, projection: int, floor: int) -> dict:
    parent, basename = _exact_paths(root, tmp)
    parent_fd = os.open(parent, NOFOLLOW_DIR)
    try:
        parent_st = os.fstat(parent_fd)
        require(stat.S_ISDIR(parent_st.st_mode) and parent_st.st_uid == os.getuid(),
                "scratch parent ownership changed")
        _free_space(parent, min_free=min_free, min_inodes=min_inodes,
                    projection=projection, floor=floor)
        if create:
            require(not os.path.lexists(root), "scratch root was not absent")
            os.umask(0o077)
            os.mkdir(basename, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        root_fd = _open_existing(parent_fd, basename, parent_st.st_dev, name="scratch root")
        try:
            if create:
                require(not os.path.lexists(tmp), "TMPDIR was not absent")
                os.mkdir("tmp", 0o700, dir_fd=root_fd)
                os.fsync(root_fd)
            tmp_fd = _open_existing(root_fd, "tmp", parent_st.st_dev, name="TMPDIR")
            try:
                entries = set(os.listdir(root_fd))
                expected = {"tmp", "process"} if allow_process else {"tmp"}
                require(entries == expected and not os.listdir(tmp_fd),
                        "scratch/TMPDIR contains unexpected paths")
                if allow_process:
                    # contained.py creates only the process directory before
                    # starting this child; reject symlinks or other siblings.
                    process_fd = _open_existing(root_fd, "process", parent_st.st_dev,
                                                name="contained process directory",
                                                required_mode=None)
                    os.close(process_fd)
                # tempfile normally searches fallback roots if TMPDIR is missing.  Reject
                # that behavior, then pin its module-global root in this interpreter.
                tempfile.tempdir = None
                chosen = tempfile.gettempdir()
                require(chosen == tmp, "tempfile silently fell back from exact TMPDIR")
                tempfile.tempdir = tmp
                root_st, tmp_st = os.fstat(root_fd), os.fstat(tmp_fd)
                return {"root": root, "tmp": tmp, "device": root_st.st_dev,
                        "root_inode": root_st.st_ino, "tmp_inode": tmp_st.st_ino,
                        "tempfile_root": chosen, "owner_uid": root_st.st_uid,
                        "mode": "0700"}
            finally:
                os.close(tmp_fd)
        finally:
            os.close(root_fd)
    finally:
        os.close(parent_fd)


def _run_tool(tool: str, argv: list[str], root: str) -> None:
    require(root == PINNED_ROOT and tool == PINNED_TOOL,
            "G2 tool or scratch binding changed")
    require(_hash(tool) == G2_SHA and _hash(PINNED_TRAIN) == TRAIN_SHA and
            _hash(PINNED_PREPARE) == PREPARE_SHA, "G2 source bytes changed")
    require("--real-data" not in argv and "--stub-prepare" not in argv,
            "G2 real-data/stub path forbidden")
    require(argv[argv.index("--train-py") + 1] == PINNED_TRAIN and
            argv[argv.index("--prepare-dir") + 1] == PINNED_CHECKOUT and
            argv[argv.index("--out") + 1] == root + "/g2.json",
            "G2 path arguments changed")
    sys.argv = [tool, *argv]
    runpy.run_path(tool, run_name="__main__")


def main(argv: list[str]) -> None:
    require(len(argv) >= 8 and argv[0] in ("prepare", "guard", "run"),
            "scratch guard mode/arguments missing")
    mode, root, tmp = argv[:3]
    min_free = _positive_decimal(argv[3], "minimum free bytes")
    min_inodes = _positive_decimal(argv[4], "minimum free inodes")
    projection = _positive_decimal(argv[5], "reviewed scratch projection")
    floor = _positive_decimal(argv[6], "protected operational floor")
    if mode == "run":
        require(len(argv) > 8, "G2 command missing")
    else:
        require(len(argv) == 8 and argv[7] == "--end", "scratch command shape changed")
    result = _inspect(root, tmp, create=mode == "prepare",
                      allow_process=mode == "run", min_free=min_free,
                      min_inodes=min_inodes, projection=projection, floor=floor)
    print(json.dumps({"scratch_guard": "PASS", "mode": mode, **result}, sort_keys=True), flush=True)
    if mode == "run":
        _run_tool(argv[7], argv[8:], root)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, IndexError) as error:
        print("SCRATCH_HOLD: " + str(error), file=sys.stderr)
        raise SystemExit(2) from error
