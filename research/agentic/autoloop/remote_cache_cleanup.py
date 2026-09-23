#!/usr/bin/env python3
"""Remove only one completed fixed-queue job's unique compile caches; stdlib only."""

import json
import os
import shutil
import stat
import sys
import time
import re

BASE = "/home/ubuntu/autoloop/caches"
KEYS = (
    "TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "NEURON_COMPILE_CACHE_DIR",
)


def fail(message):
    print(json.dumps({"error": message}))
    raise SystemExit(2)


def regular_json(path):
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        fail("cleanup spec is not a regular file")
    with open(path) as fh:
        return json.load(fh)


def main():
    if len(sys.argv) != 2:
        fail("usage: remote_cache_cleanup.py <cleanup.json>")
    spec = regular_json(sys.argv[1])
    label = spec.get("label", "")
    if not re.fullmatch(r"al-[A-Za-z0-9_.-]+", label):
        fail("invalid cleanup label")
    root = os.path.join(BASE, label)
    expected = []
    for group in ("train", "eval"):
        for key in KEYS:
            expected.append(os.path.join(root, group, key.lower()))
    if spec.get("cache_paths") != expected or spec.get("cache_root") != root:
        fail("cache paths differ from exact label-owned schema")
    if os.path.realpath(root) != root or not os.path.isdir(root):
        fail("cache root missing, symlinked, or non-directory")
    cursor = BASE
    for component in (label,):
        cursor = os.path.join(cursor, component)
        if stat.S_ISLNK(os.lstat(cursor).st_mode):
            fail("cache root contains a symlink")
    for path in expected:
        if os.path.lexists(path) and (os.path.islink(path) or os.path.realpath(path) != path):
            fail("cache path is symlinked")
    # No surviving process may have cwd/root/exe/fd beneath this exact cache root.
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        for leaf in ("cwd", "root", "exe"):
            try:
                target = os.path.realpath(os.path.join("/proc", pid, leaf))
            except OSError:
                continue
            if target == root or target.startswith(root + os.sep):
                fail("live process references cache root")
        fd_dir = os.path.join("/proc", pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.path.realpath(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target == root or target.startswith(root + os.sep):
                fail("live process has cache file open")
    files = 0
    total_bytes = 0
    for base, dirs, names in os.walk(root, followlinks=False):
        if any(os.path.islink(os.path.join(base, name)) for name in dirs + names):
            fail("cache tree contains a symlink")
        for name in names:
            st = os.lstat(os.path.join(base, name))
            if not stat.S_ISREG(st.st_mode):
                fail("cache tree contains a non-regular file")
            files += 1
            total_bytes += st.st_size
    shutil.rmtree(root)
    if os.path.lexists(root):
        fail("cache root remains after cleanup")
    receipt = {
        "label": label, "cache_root": root, "cache_paths": expected,
        "files_removed": files, "bytes_removed": total_bytes,
        "absent_after": True, "cleaned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_path = spec.get("receipt_path", "")
    if receipt_path != os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "receipt.json"):
        fail("cleanup receipt path differs from exact schema")
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(receipt, fh, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
