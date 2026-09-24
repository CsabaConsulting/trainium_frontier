#!/usr/bin/env python3
"""No-follow, O_EXCL artifact primitives for the Track A cohort."""

import hashlib
import json
import os
from pathlib import Path
import stat


def canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise SystemExit(f"symlink rejected: {current}")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"not a regular file: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data = b"".join(chunks)
        if len(data) != opened.st_size or opened.st_size != before.st_size:
            raise SystemExit(f"file changed while reading: {path}")
        return data
    finally:
        os.close(fd)


def identity(data):
    return {"sha256": digest(data), "bytes": len(data)}


class Store:
    def __init__(self, root, label):
        self.root = Path(root).absolute()
        self.label = label
        self.path = self.root / "out" / label

    @staticmethod
    def directory(path):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise SystemExit(f"unsafe directory: {path}")
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def create(self):
        root_fd = self.directory(self.root)
        try:
            try:
                os.mkdir("out", 0o700, dir_fd=root_fd)
                os.fsync(root_fd)
            except FileExistsError:
                pass
            out_fd = os.open("out", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                try:
                    os.mkdir(self.label, 0o700, dir_fd=out_fd)
                    os.fsync(out_fd)
                except FileExistsError as error:
                    raise SystemExit("artifact output already exists") from error
            finally:
                os.close(out_fd)
        finally:
            os.close(root_fd)

    def fd(self):
        out_fd = self.directory(self.root / "out")
        try:
            return os.open(self.label, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=out_fd)
        finally:
            os.close(out_fd)

    def write(self, name, data):
        if Path(name).name != name or name in {"", ".", ".."}:
            raise SystemExit("artifact path escape")
        directory = self.fd()
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
            try:
                offset = 0
                while offset < len(data):
                    written = os.write(fd, data[offset:])
                    if written <= 0:
                        raise SystemExit("short artifact write")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(directory)
        finally:
            os.close(directory)
        if self.read(name) != data:
            raise SystemExit("artifact persistence mismatch")
        return identity(data)

    def write_json(self, name, value):
        return self.write(name, canonical_json(value))

    def read(self, name):
        if Path(name).name != name:
            raise SystemExit("artifact path escape")
        directory = self.fd()
        try:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise SystemExit(f"artifact not regular: {name}")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                opened = os.fstat(fd)
                chunks = []
                while True:
                    block = os.read(fd, 1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
                data = b"".join(chunks)
                if len(data) != opened.st_size or opened.st_size != info.st_size:
                    raise SystemExit(f"artifact changed while reading: {name}")
                return data
            finally:
                os.close(fd)
        finally:
            os.close(directory)

    def inspect(self, expected):
        directory = self.fd()
        try:
            names = set(os.listdir(directory))
        finally:
            os.close(directory)
        if names != set(expected):
            raise SystemExit("artifact allowlist mismatch")
        return {name: identity(self.read(name)) for name in sorted(names)}
