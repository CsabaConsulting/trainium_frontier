#!/usr/bin/env python3
"""Authenticated exact-path c595 checkpoint prune after evidence is pushed."""
import argparse, hashlib, json, os, pathlib, re, stat, subprocess

ORIGIN="https://github.com/CsabaConsulting/trainium_frontier.git"
BRANCH="experiment/20260923-c595-dual-control"
ARMS={
 "control_trn21":("fs-20260923-c595-control-trn21-r1",pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn21")),
 "control_trn22":("fs-20260923-c595-control-trn22-r1",pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn22")),
}

def nofollow(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode): raise SystemExit("not regular")
        return os.read(fd,info.st_size),info.st_size
    finally: os.close(fd)
def sha(path): return hashlib.sha256(nofollow(path)[0]).hexdigest()
def safe_components(path,leaf_required=True):
    if not path.is_absolute(): raise SystemExit("absolute path required")
    current=pathlib.Path(path.anchor)
    for index,part in enumerate(path.parts[1:]):
        current/=part
        try: info=os.lstat(current)
        except FileNotFoundError:
            if index==len(path.parts[1:])-1 and not leaf_required:return
            raise
        if stat.S_ISLNK(info.st_mode):raise SystemExit("symlink rejected")
def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
def remote_head():
    fields=subprocess.check_output(["git","ls-remote",ORIGIN,f"refs/heads/{BRANCH}"],text=True).split()
    if len(fields)!=2:raise SystemExit("remote head malformed")
    return fields[0]

def execute(args,head_lookup=remote_head):
    script=pathlib.Path(__file__)
    if sha(script)!=args.expected_script_sha256:raise SystemExit("script hash mismatch")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=script.parents[3],text=True).strip()
    if head!=args.expected_core_commit:raise SystemExit("core commit mismatch")
    label,root=ARMS[args.arm]
    checkpoint=root/"out"/label/"train"/"out"/"final.pt"
    archive=root/"out"/"archives"/f"{label}.tar.gz"
    proof=root/"out"/label/"evidence-pushed-receipt.json"
    receipt=root/"out"/label/"checkpoint-prune-receipt.json"
    supplied=(args.label,args.checkpoint,args.archive,args.evidence_pushed_receipt,args.receipt)
    expected=(label,checkpoint,archive,proof,receipt)
    if tuple(map(str,supplied))!=tuple(map(str,expected)):raise SystemExit("lexical path contract mismatch")
    for path in (checkpoint,archive,proof):safe_components(path)
    safe_components(receipt,False)
    proof_bytes,_=nofollow(proof)
    if hashlib.sha256(proof_bytes).hexdigest()!=args.evidence_pushed_receipt_sha256:raise SystemExit("proof hash mismatch")
    record=json.loads(proof_bytes)
    exact={"schema":"c595-control-evidence-push-v1","arm":args.arm,"label":label,
           "archive_path":str(archive),"archive_sha256":args.archive_sha256,
           "checkpoint_path":str(checkpoint),"checkpoint_sha256":args.expected_sha256,
           "checkpoint_bytes":args.expected_bytes,"origin":ORIGIN,"branch":BRANCH,
           "retrieved_verified_committed_pushed":True}
    if set(record)!=set(exact)|{"evidence_commit"} or any(record.get(k)!=v for k,v in exact.items()):raise SystemExit("proof contract mismatch")
    evidence_commit=record["evidence_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}",evidence_commit) or head_lookup()!=evidence_commit:raise SystemExit("evidence not remote head")
    if sha(archive)!=args.archive_sha256:raise SystemExit("archive mismatch")
    checkpoint_data,size=nofollow(checkpoint)
    if size!=args.expected_bytes or hashlib.sha256(checkpoint_data).hexdigest()!=args.expected_sha256:raise SystemExit("checkpoint mismatch")
    temporary=receipt.with_name(receipt.name+".complete.tmp")
    safe_components(temporary,False)
    intent_fd=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    temp_fd=None
    try:
        temp_fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        intent={**exact,"evidence_commit":evidence_commit,"state":"durable_intent_reserved","deleted":False,"other_deletions":[]}
        os.write(intent_fd,(json.dumps(intent,sort_keys=True)+"\n").encode());os.fsync(intent_fd);fsync_dir(receipt.parent)
        os.unlink(checkpoint);fsync_dir(checkpoint.parent)
        complete={**intent,"state":"complete","deleted":True}
        os.write(temp_fd,(json.dumps(complete,sort_keys=True)+"\n").encode());os.fsync(temp_fd)
        os.close(temp_fd);temp_fd=None
        os.replace(temporary,receipt);fsync_dir(receipt.parent)
    finally:
        os.close(intent_fd)
        if temp_fd is not None:os.close(temp_fd)
    return receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=ARMS,required=True);ap.add_argument("--label",required=True)
    ap.add_argument("--checkpoint",type=pathlib.Path,required=True);ap.add_argument("--archive",type=pathlib.Path,required=True)
    ap.add_argument("--archive-sha256",required=True);ap.add_argument("--expected-sha256",required=True);ap.add_argument("--expected-bytes",type=int,required=True)
    ap.add_argument("--evidence-pushed-receipt",type=pathlib.Path,required=True);ap.add_argument("--evidence-pushed-receipt-sha256",required=True)
    ap.add_argument("--receipt",type=pathlib.Path,required=True);ap.add_argument("--expected-core-commit",required=True);ap.add_argument("--expected-script-sha256",required=True)
    print(execute(ap.parse_args()))
if __name__=="__main__":main()
