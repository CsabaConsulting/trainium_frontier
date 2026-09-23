#!/usr/bin/env python3
"""Deterministic post-run retrieve/verify/push/proof/prune/cache-cleanup operator."""
import argparse, hashlib, json, os, pathlib, subprocess, tarfile

ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
ORIGIN="https://github.com/CsabaConsulting/trainium_frontier.git";BRANCH="experiment/20260923-c595-dual-control"
TRAIN_SHA="c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
SSH=["ssh","-i","/home/poojith-devan/Desktop/Trainium/trn2.pem","-o","BatchMode=yes","-o","ConnectTimeout=12"]
SCP=["scp","-i","/home/poojith-devan/Desktop/Trainium/trn2.pem","-o","BatchMode=yes","-o","ConnectTimeout=12"]
ARMS={
 "control_trn21":("ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com",pathlib.PurePosixPath("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn21"),"fs-20260923-c595-control-trn21-r1"),
 "control_trn22":("ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com",pathlib.PurePosixPath("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn22"),"fs-20260923-c595-control-trn22-r1"),
}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def config(arm):
 host,root,label=ARMS[arm];sandbox=root/"out"/label
 return host,root,label,sandbox,root/"out"/"archives"/(label+".tar.gz"),sandbox/"archive-completion.json"
def verify(arm,archive,completion):
 _host,_root,label,_sandbox,remote_archive,_remote_completion=config(arm)
 c=json.loads(completion.read_text())
 exact={"schema":"c595-control-archive-completion-v1","arm":arm,"label":label,
        "candidate_sha256":TRAIN_SHA,"archive_path":str(remote_archive),"ready_to_retrieve":True,
        "archive_immutable_after_sha256":True}
 if any(c.get(k)!=v for k,v in exact.items()):raise SystemExit("completion identity mismatch")
 if sha(archive)!=c["archive_sha256"] or archive.stat().st_size!=c["archive_bytes"]:raise SystemExit("archive identity mismatch")
 with tarfile.open(archive,"r:gz") as bundle:
  if set(bundle.getnames())!=set(c["archive_members"]):raise SystemExit("archive members mismatch")
  for name,want in c["archive_members"].items():
   stream=bundle.extractfile(name)
   if stream is None or hashlib.sha256(stream.read()).hexdigest()!=want:raise SystemExit("archive member hash mismatch")
 return c
def remote_ref():
 fields=subprocess.check_output(["git","ls-remote",ORIGIN,f"refs/heads/{BRANCH}"],text=True).split()
 if len(fields)!=2:raise SystemExit("remote ref malformed")
 return fields[0]
def durable(path,value):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:os.write(fd,(json.dumps(value,indent=2,sort_keys=True)+"\n").encode());os.fsync(fd)
 finally:os.close(fd)

def retrieve(args):
 host,_root,label,_sandbox,archive,completion=config(args.arm)
 destination=HERE/"retrieved"/label;destination.mkdir(parents=True,exist_ok=False)
 subprocess.run([*SCP,f"{host}:{archive}",str(destination/"evidence.tar.gz")],check=True)
 subprocess.run([*SCP,f"{host}:{completion}",str(destination/"archive-completion.json")],check=True)
 c=verify(args.arm,destination/"evidence.tar.gz",destination/"archive-completion.json")
 record={"schema":"c595-control-retrieval-v1","arm":args.arm,"label":label,
         "archive_sha256":c["archive_sha256"],"archive_bytes":c["archive_bytes"],
         "archive_members":c["archive_members"],"completion_sha256":sha(destination/"archive-completion.json"),
         "core_commit":c["core_commit"],"candidate_sha256":TRAIN_SHA,"verified":True}
 durable(destination/"retrieval-verification.json",record);print(destination)

def commit_proof(args):
 host,_root,label,sandbox,archive,_completion=config(args.arm)
 destination=HERE/"retrieved"/label;verification=destination/"retrieval-verification.json"
 c=verify(args.arm,destination/"evidence.tar.gz",destination/"archive-completion.json")
 if not verification.is_file():raise SystemExit("retrieval verification missing")
 subprocess.run(["git","add","--",str(verification.relative_to(ROOT)),str((destination/"archive-completion.json").relative_to(ROOT))],cwd=ROOT,check=True)
 subprocess.run(["git","commit","-m",f"evidence: verify {label} archive"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin",f"HEAD:refs/heads/{BRANCH}"],cwd=ROOT,check=True)
 evidence_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 if remote_ref()!=evidence_commit:raise SystemExit("evidence push not remote head")
 proof={"schema":"c595-control-evidence-push-v1","arm":args.arm,"label":label,
        "archive_path":str(archive),"archive_sha256":c["archive_sha256"],
        "checkpoint_path":c["checkpoint_path"],"checkpoint_sha256":c["checkpoint_sha256"],
        "checkpoint_bytes":c["checkpoint_bytes"],"origin":ORIGIN,"branch":BRANCH,
        "retrieved_verified_committed_pushed":True,"evidence_commit":evidence_commit}
 local=destination/"evidence-pushed-receipt.json";durable(local,proof)
 subprocess.run([*SCP,str(local),f"{host}:{sandbox/'evidence-pushed-receipt.json'}"],check=True)
 print(local)

def prune(args):
 host,root,label,sandbox,archive,_completion=config(args.arm);destination=HERE/"retrieved"/label
 c=json.loads((destination/"archive-completion.json").read_text());proof=destination/"evidence-pushed-receipt.json"
 script=root/"research/submission-validation/20260923-c595-dual-host-seed42-controls/prune_checkpoint.py"
 receipt=sandbox/"checkpoint-prune-receipt.json";remote_proof=sandbox/"evidence-pushed-receipt.json"
 argv=["cd",str(root),"&&","/home/ubuntu/workspace/native_venv/bin/python",str(script),"--arm",args.arm,"--label",label,
       "--checkpoint",c["checkpoint_path"],"--archive",str(archive),"--archive-sha256",c["archive_sha256"],
       "--expected-sha256",c["checkpoint_sha256"],"--expected-bytes",str(c["checkpoint_bytes"]),
       "--evidence-pushed-receipt",str(remote_proof),"--evidence-pushed-receipt-sha256",sha(proof),
       "--receipt",str(receipt),"--expected-core-commit",c["core_commit"],"--expected-script-sha256",args.prune_script_sha256]
 subprocess.run([*SSH,host," ".join(argv)],check=True)
 local=destination/"checkpoint-prune-receipt.json";subprocess.run([*SCP,f"{host}:{receipt}",str(local)],check=True)
 record=json.loads(local.read_text())
 if record.get("state")!="complete" or record.get("deleted") is not True:raise SystemExit("prune receipt invalid")
 subprocess.run(["git","add","--",str(local.relative_to(ROOT))],cwd=ROOT,check=True);subprocess.run(["git","commit","-m",f"evidence: preserve {label} prune"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin",f"HEAD:refs/heads/{BRANCH}"],cwd=ROOT,check=True)
 if remote_ref()!=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip():raise SystemExit("prune receipt not pushed")
 print(local)

def cleanup(args):
 host,root,label,sandbox,_archive,_completion=config(args.arm);destination=HERE/"retrieved"/label
 proof=destination/"evidence-pushed-receipt.json";prune_receipt=destination/"checkpoint-prune-receipt.json"
 script=root/"research/submission-validation/20260923-c595-dual-host-seed42-controls/cleanup_caches.py";remote_receipt=sandbox/"cache-cleanup-receipt.json"
 argv=["cd",str(root),"&&","/home/ubuntu/workspace/native_venv/bin/python",str(script),"--arm",args.arm,"--label",label,
       "--evidence-pushed-receipt",str(sandbox/"evidence-pushed-receipt.json"),"--evidence-pushed-receipt-sha256",sha(proof),
       "--prune-receipt",str(sandbox/"checkpoint-prune-receipt.json"),"--prune-receipt-sha256",sha(prune_receipt),
       "--receipt",str(remote_receipt),"--expected-core-commit",json.loads((destination/"archive-completion.json").read_text())["core_commit"],
       "--expected-script-sha256",args.cleanup_script_sha256]
 subprocess.run([*SSH,host," ".join(argv)],check=True)
 local=destination/"cache-cleanup-receipt.json";subprocess.run([*SCP,f"{host}:{remote_receipt}",str(local)],check=True)
 record=json.loads(local.read_text())
 if record.get("state")!="complete" or record.get("preserved_logs_results") is not True:raise SystemExit("cleanup receipt invalid")
 subprocess.run(["git","add","--",str(local.relative_to(ROOT))],cwd=ROOT,check=True);subprocess.run(["git","commit","-m",f"evidence: preserve {label} cache cleanup"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin",f"HEAD:refs/heads/{BRANCH}"],cwd=ROOT,check=True)
 if remote_ref()!=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip():raise SystemExit("cleanup receipt not pushed")
 print(local)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("retrieve","commit-proof","prune","cleanup"));ap.add_argument("--arm",choices=ARMS,required=True)
 ap.add_argument("--prune-script-sha256");ap.add_argument("--cleanup-script-sha256");args=ap.parse_args()
 {"retrieve":retrieve,"commit-proof":commit_proof,"prune":prune,"cleanup":cleanup}[args.stage](args)
if __name__=="__main__":main()
