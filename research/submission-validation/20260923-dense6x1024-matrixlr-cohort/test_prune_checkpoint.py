#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, os, pathlib, subprocess, tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prune", HERE / "prune_checkpoint.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

def digest(data): return hashlib.sha256(data).hexdigest()

def setup(root):
    arm, label = "control", "fs-test"
    p.ARMS = {arm: (label, root)}
    run = root / "out" / label; checkpoint = run / "train/out/final.pt"
    archive = root / "out/archives" / f"{label}.tar.gz"; proof = run / "evidence-pushed-receipt.json"
    receipt = run / "checkpoint-prune-receipt.json"
    checkpoint.parent.mkdir(parents=True); archive.parent.mkdir(parents=True); archive.write_bytes(b"archive")
    checkpoint.write_bytes(b"checkpoint")
    fields = {"schema":"dense6x1024-cohort-evidence-push-v1","arm":arm,"label":label,
              "archive_path":str(archive),"archive_sha256":digest(b"archive"),
              "checkpoint_path":str(checkpoint),"checkpoint_sha256":digest(b"checkpoint"),
              "checkpoint_bytes":10,"origin":p.ORIGIN,"branch":p.BRANCH,
              "retrieved_verified_committed_pushed":True,"evidence_commit":"a"*40}
    proof.write_text(json.dumps(fields))
    args=argparse.Namespace(arm=arm,label=label,checkpoint=checkpoint,expected_sha256=digest(b"checkpoint"),
        expected_bytes=10,archive=archive,archive_sha256=digest(b"archive"),evidence_pushed_receipt=proof,
        evidence_pushed_receipt_sha256=digest(proof.read_bytes()),receipt=receipt,
        expected_script_sha256=p.sha(pathlib.Path(p.__file__)),
        expected_core_commit=subprocess.run(["git","rev-parse","HEAD"],cwd=HERE.parents[3],
            text=True,capture_output=True,check=True).stdout.strip())
    return args, fields

with tempfile.TemporaryDirectory() as d:
    args,_=setup(pathlib.Path(d)); p.execute(args, lambda:"a"*40)
    assert not args.checkpoint.exists() and json.loads(args.receipt.read_text())["deleted"] is True

for mutation in ("arm","label","archive_path","archive_sha256","checkpoint_path","checkpoint_sha256",
                 "checkpoint_bytes","origin","branch","retrieved_verified_committed_pushed","evidence_commit"):
    with tempfile.TemporaryDirectory() as d:
        args,fields=setup(pathlib.Path(d)); fields[mutation] = ("b"*40 if mutation=="evidence_commit" else "wrong")
        args.evidence_pushed_receipt.write_text(json.dumps(fields)); args.evidence_pushed_receipt_sha256=digest(args.evidence_pushed_receipt.read_bytes())
        try: p.execute(args, lambda:"a"*40)
        except (SystemExit, TypeError): pass
        else: raise AssertionError(mutation)
        assert args.checkpoint.exists()

for collision in ("receipt", "temp"):
    with tempfile.TemporaryDirectory() as d:
        args,_=setup(pathlib.Path(d)); target=args.receipt if collision=="receipt" else args.receipt.with_name(args.receipt.name+".complete.tmp")
        target.write_text("collision")
        try: p.execute(args, lambda:"a"*40)
        except (FileExistsError, SystemExit): pass
        else: raise AssertionError(collision)
        assert args.checkpoint.exists()

with tempfile.TemporaryDirectory() as d:
    root=pathlib.Path(d); args,_=setup(root); args.checkpoint.write_bytes(b"changed")
    try: p.execute(args, lambda:"a"*40)
    except SystemExit: pass
    else: raise AssertionError("changed checkpoint")
    assert args.checkpoint.exists()

for symlink_parent in (False, True):
    with tempfile.TemporaryDirectory() as d:
        base=pathlib.Path(d); real=base/"real"; args,_=setup(real)
        if symlink_parent:
            alias=base/"alias"; alias.symlink_to(real, target_is_directory=True); p.ARMS={"control":("fs-test",alias)}
            args.checkpoint=alias/"out/fs-test/train/out/final.pt"; args.archive=alias/"out/archives/fs-test.tar.gz"; args.evidence_pushed_receipt=alias/"out/fs-test/evidence-pushed-receipt.json"; args.receipt=alias/"out/fs-test/checkpoint-prune-receipt.json"
        else:
            target=args.checkpoint; target.unlink(); target.symlink_to(real/"other"); (real/"other").write_bytes(b"checkpoint")
        try: p.execute(args, lambda:"a"*40)
        except (SystemExit, FileNotFoundError): pass
        else: raise AssertionError("symlink")
        assert (real/"other").exists() if not symlink_parent else (real/"out/fs-test/train/out/final.pt").exists()
print("prune positive/adversarial tests: PASS")
