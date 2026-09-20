#!/usr/bin/env python3
"""Pin a minimal, unmodified NKI Library import closure in an isolated namespace.

This is source vendoring, not package installation. In particular it never puts
``nkilib_src`` on sys.path or replaces the compiler's bundled ``nkilib``.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


SEEDS = ("core/attention/attention_cte.py", "core/attention/attention_bwd.py",
         "core/mlp/mlp.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(root):
    pending = [Path(p) for p in SEEDS]
    seen, external = set(), set()

    def add_module(parts):
        for p in (Path(*parts).with_suffix(".py"), Path(*parts) / "__init__.py"):
            if (root / p).is_file() and p not in seen:
                pending.append(p)

    while pending:
        p = pending.pop()
        if p in seen:
            continue
        seen.add(p)
        for parent in p.parents:
            if parent == Path("."):
                break
            f = parent / "__init__.py"
            if (root / f).is_file() and f not in seen:
                pending.append(f)
        for node in ast.walk(ast.parse((root / p).read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = list(p.parent.parts)
                    base = base[:len(base) - node.level + 1]
                    parts = base + (node.module.split(".") if node.module else [])
                    add_module(parts)
                    for name in node.names:
                        add_module(parts + [name.name])
                else:
                    external.add(node.module)
            elif isinstance(node, ast.Import):
                external.update(n.name for n in node.names)
    if any(n.startswith(("nkilib", "nkilib_src")) for n in external):
        raise ValueError("Absolute upstream imports would escape the isolated namespace")
    return sorted(seen), sorted(external)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    library = args.library.resolve()
    source = library / "src/nkilib_src/nkilib"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=library, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=library).strip():
        raise ValueError("Upstream clone must be clean before pinning")
    files, external = closure(source)
    args.destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(upstream="https://github.com/aws-neuron/nki-library", commit=head,
                    seeds=list(SEEDS), external_imports=external, files={},
                    namespace="research.vendor." + args.destination.name,
                    no_upstream_kernel_edits=True, package_installation=False)
    for rel in files:
        dest = args.destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dest)
        manifest["files"][str(rel)] = dict(sha256=digest(dest), bytes=dest.stat().st_size)
    # The upstream root __init__ redirects bundled imports when installed. This
    # deliberately empty root does not run that package replacement mechanism.
    (args.destination / "__init__.py").write_text(
        '"""Isolated, pinned NKI Library sources; see manifest.json and LICENSE."""\n')
    manifest["generated_root_sha256"] = digest(args.destination / "__init__.py")
    shutil.copyfile(library / "LICENSE", args.destination / "LICENSE")
    for name in ("NOTICE", "NOTICE.txt"):
        if (library / name).is_file():
            shutil.copyfile(library / name, args.destination / name)
    old = {str(p.relative_to(args.installed)): digest(p) for p in args.installed.rglob("*.py")}
    new = {str(p.relative_to(source)): digest(p) for p in source.rglob("*.py")}
    manifest["installed_comparison"] = dict(
        installed_python_files=len(old), clone_python_files=len(new),
        equal=sum(old[k] == new[k] for k in old.keys() & new.keys()),
        different=sum(old[k] != new[k] for k in old.keys() & new.keys()),
        only_in_clone=len(new.keys() - old.keys()), only_in_installed=len(old.keys() - new.keys()),
        selected={p: dict(installed_sha256=old.get(p), clone_sha256=new.get(p)) for p in
                  (*SEEDS, "experimental/optimizer/fused_adam.py", "experimental/loss/cross_entropy.py")})
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(commit=head, files=len(files), bytes=sum(v["bytes"] for v in manifest["files"].values()),
                          destination=str(args.destination), comparison=manifest["installed_comparison"])))


if __name__ == "__main__":
    main()
