#!/usr/bin/env python3
"""Deterministic post-run retrieve/verify/push/proof/prune/cache-cleanup operator."""
import argparse, hashlib, json, os, pathlib, re, subprocess, tarfile

ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
ORIGIN="https://github.com/CsabaConsulting/trainium_frontier.git";BRANCH="experiment/20260923-c595-dual-control"
TRAIN_SHA="c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
SSH=["ssh","-i","/home/poojith-devan/Desktop/Trainium/trn2.pem","-o","BatchMode=yes","-o","ConnectTimeout=12"]
SCP=["scp","-i","/home/poojith-devan/Desktop/Trainium/trn2.pem","-o","BatchMode=yes","-o","ConnectTimeout=12"]
ARMS={
 "control_trn21":("ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com",pathlib.PurePosixPath("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn21"),"fs-20260923-c595-control-trn21-r1"),
 "control_trn22":("ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com",pathlib.PurePosixPath("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn22"),"fs-20260923-c595-control-trn22-r1"),
 "control_trn21_r2":("ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com",pathlib.PurePosixPath("/home/ubuntu/trainium_freshstart_20260924_c595_control_trn21_r2"),"fs-20260924-c595-control-trn21-r2"),
 "control_trn22_r2":("ubuntu@ec2-16-26-154-27.ap-southeast-4.compute.amazonaws.com",pathlib.PurePosixPath("/home/ubuntu/trainium_freshstart_20260924_c595_control_trn22_r2"),"fs-20260924-c595-control-trn22-r2"),
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

def exact_proof(arm,label,archive,c,evidence_commit):
 return {"schema":"c595-control-evidence-push-v1","arm":arm,"label":label,
  "archive_path":str(archive),"archive_sha256":c["archive_sha256"],
  "checkpoint_path":c["checkpoint_path"],"checkpoint_sha256":c["checkpoint_sha256"],
  "checkpoint_bytes":c["checkpoint_bytes"],"origin":ORIGIN,"branch":BRANCH,
  "retrieved_verified_committed_pushed":True,"evidence_commit":evidence_commit}
def validate_exact(value,expected,name):
 if set(value)!=set(expected) or value!=expected:raise SystemExit(name+" schema/identity mismatch")
def validate_retrieval(value,arm,label,c,completion_sha):
 expected={"schema":"c595-control-retrieval-v1","arm":arm,"label":label,
  "archive_sha256":c["archive_sha256"],"archive_bytes":c["archive_bytes"],
  "archive_members":c["archive_members"],"completion_sha256":completion_sha,
  "core_commit":c["core_commit"],"candidate_sha256":TRAIN_SHA,"verified":True}
 validate_exact(value,expected,"retrieval verification")
def require_clean_evidence_state(root,commit_paths,allowed_untracked):
 staged=subprocess.check_output(["git","diff","--cached","--name-only"],cwd=root,text=True).splitlines()
 dirty=subprocess.check_output(["git","diff","--name-only"],cwd=root,text=True).splitlines()
 untracked=subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=root,text=True).splitlines()
 if staged or dirty or set(untracked)!=set(allowed_untracked):raise SystemExit("repository state not exclusive to retrieved evidence")
 for path in commit_paths:
  if path not in untracked:raise SystemExit("evidence path was pretracked or missing")
def commit_exact_evidence(paths,allowed_untracked,message):
 relative=[str(path.relative_to(ROOT)) for path in paths]
 require_clean_evidence_state(ROOT,relative,allowed_untracked)
 subprocess.run(["git","add","--",*relative],cwd=ROOT,check=True)
 staged=subprocess.check_output(["git","diff","--cached","--name-only"],cwd=ROOT,text=True).splitlines()
 if set(staged)!=set(relative) or len(staged)!=len(relative):raise SystemExit("staged stage-evidence path mismatch")
 subprocess.run(["git","commit","--only","-m",message,"--",*relative],cwd=ROOT,check=True)
 committed=subprocess.check_output(["git","diff-tree","--no-commit-id","--name-only","-r","HEAD"],cwd=ROOT,text=True).splitlines()
 if set(committed)!=set(relative) or len(committed)!=len(relative):raise SystemExit("commit contains non-stage evidence")
 subprocess.run(["git","push","origin",f"HEAD:refs/heads/{BRANCH}"],cwd=ROOT,check=True)
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 if remote_ref()!=head:raise SystemExit("stage evidence not pushed")
 return head
def lowercase_sha(value,name):
 if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}",value):raise SystemExit(name+" is not lowercase sha256")
 return value
def validate_terminal_cleanup(journal,receipt,arm,label,core,proof_sha,prune_sha,caches,journal_sha):
 absence={path:"ENOENT" for path in caches}
 base={"arm":arm,"label":label,"core_commit":core,"candidate_sha256":TRAIN_SHA,
  "proof_sha256":lowercase_sha(proof_sha,"proof sha"),"prune_receipt_sha256":lowercase_sha(prune_sha,"prune sha"),
  "allowlist":caches,"preserved_logs_results":True,"completed":caches,"active_path":None,
  "state":"complete","absence_lstat":absence}
 expected_journal={"schema":"c595-cache-cleanup-journal-v1",**base}
 validate_exact(journal,expected_journal,"terminal cleanup journal")
 expected_receipt={"schema":"c595-cache-cleanup-receipt-v1",**base,"journal_sha256":lowercase_sha(journal_sha,"journal sha")}
 validate_exact(receipt,expected_receipt,"terminal cleanup receipt")

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
 completion=destination/"archive-completion.json";archive_local=destination/"evidence.tar.gz"
 validate_retrieval(json.loads(verification.read_text()),args.arm,label,c,sha(completion))
 commit_paths=[str(verification.relative_to(ROOT)),str(completion.relative_to(ROOT))]
 require_clean_evidence_state(ROOT,commit_paths,[*commit_paths,str(archive_local.relative_to(ROOT))])
 subprocess.run(["git","add","--",*commit_paths],cwd=ROOT,check=True)
 if set(subprocess.check_output(["git","diff","--cached","--name-only"],cwd=ROOT,text=True).splitlines())!=set(commit_paths):raise SystemExit("staged evidence path mismatch")
 subprocess.run(["git","commit","-m",f"evidence: verify {label} archive"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin",f"HEAD:refs/heads/{BRANCH}"],cwd=ROOT,check=True)
 evidence_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 if remote_ref()!=evidence_commit:raise SystemExit("evidence push not remote head")
 proof=exact_proof(args.arm,label,archive,c,evidence_commit)
 local=destination/"evidence-pushed-receipt.json";durable(local,proof)
 subprocess.run([*SCP,str(local),f"{host}:{sandbox/'evidence-pushed-receipt.json'}"],check=True)
 print(local)

def prune(args):
 host,root,label,sandbox,archive,_completion=config(args.arm);destination=HERE/"retrieved"/label
 c=json.loads((destination/"archive-completion.json").read_text());proof=destination/"evidence-pushed-receipt.json"
 script=root/"research/submission-validation/20260923-c595-dual-host-seed42-controls/prune_checkpoint.py"
 receipt=sandbox/"checkpoint-prune-receipt.json";remote_proof=sandbox/"evidence-pushed-receipt.json"
 allowed=[str((destination/"evidence.tar.gz").relative_to(ROOT)),str(proof.relative_to(ROOT))]
 require_clean_evidence_state(ROOT,[],allowed)
 argv=["cd",str(root),"&&","/home/ubuntu/workspace/native_venv/bin/python",str(script),"--arm",args.arm,"--label",label,
       "--checkpoint",c["checkpoint_path"],"--archive",str(archive),"--archive-sha256",c["archive_sha256"],
       "--expected-sha256",c["checkpoint_sha256"],"--expected-bytes",str(c["checkpoint_bytes"]),
       "--evidence-pushed-receipt",str(remote_proof),"--evidence-pushed-receipt-sha256",sha(proof),
       "--receipt",str(receipt),"--expected-core-commit",c["core_commit"],"--expected-script-sha256",args.prune_script_sha256]
 subprocess.run([*SSH,host," ".join(argv)],check=True)
 local=destination/"checkpoint-prune-receipt.json";subprocess.run([*SCP,f"{host}:{receipt}",str(local)],check=True)
 record=json.loads(local.read_text());proof_data=json.loads(proof.read_text())
 validate_exact(record,{**proof_data,"state":"complete","deleted":True,"other_deletions":[]},"prune receipt")
 commit_exact_evidence([local],[*allowed,str(local.relative_to(ROOT))],f"evidence: preserve {label} prune")
 print(local)

def cleanup(args):
 host,root,label,sandbox,_archive,_completion=config(args.arm);destination=HERE/"retrieved"/label
 proof=destination/"evidence-pushed-receipt.json";prune_receipt=destination/"checkpoint-prune-receipt.json"
 script=root/"research/submission-validation/20260923-c595-dual-host-seed42-controls/cleanup_caches.py";remote_receipt=sandbox/"cache-cleanup-receipt.json";remote_journal=sandbox/"cache-cleanup-journal.json"
 allowed=[str((destination/"evidence.tar.gz").relative_to(ROOT)),str(proof.relative_to(ROOT))]
 require_clean_evidence_state(ROOT,[],allowed)
 argv=["cd",str(root),"&&","/home/ubuntu/workspace/native_venv/bin/python",str(script),"--arm",args.arm,"--label",label,
       "--evidence-pushed-receipt",str(sandbox/"evidence-pushed-receipt.json"),"--evidence-pushed-receipt-sha256",sha(proof),
       "--prune-receipt",str(sandbox/"checkpoint-prune-receipt.json"),"--prune-receipt-sha256",sha(prune_receipt),
       "--receipt",str(remote_receipt),"--expected-core-commit",json.loads((destination/"archive-completion.json").read_text())["core_commit"],
       "--expected-script-sha256",args.cleanup_script_sha256]
 subprocess.run([*SSH,host," ".join(argv)],check=True)
 local=destination/"cache-cleanup-receipt.json";local_journal=destination/"cache-cleanup-journal.json"
 subprocess.run([*SCP,f"{host}:{remote_journal}",str(local_journal)],check=True)
 subprocess.run([*SCP,f"{host}:{remote_receipt}",str(local)],check=True)
 record=json.loads(local.read_text());journal_record=json.loads(local_journal.read_text());proof_data=json.loads(proof.read_text());prune_data=json.loads(prune_receipt.read_text())
 validate_exact(prune_data,{**proof_data,"state":"complete","deleted":True,"other_deletions":[]},"prune receipt")
 caches=[str(sandbox/"train"/"caches"/name) for name in ("neff","local","hlo","inductor","compiler","debug")]
 core=json.loads((destination/"archive-completion.json").read_text())["core_commit"]
 validate_terminal_cleanup(journal_record,record,args.arm,label,core,sha(proof),sha(prune_receipt),caches,sha(local_journal))
 incoming=[str(local_journal.relative_to(ROOT)),str(local.relative_to(ROOT))]
 commit_exact_evidence([local_journal,local],[*allowed,*incoming],f"evidence: preserve {label} cache cleanup")
 print(local)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("retrieve","commit-proof","prune","cleanup"));ap.add_argument("--arm",choices=ARMS,required=True)
 ap.add_argument("--prune-script-sha256");ap.add_argument("--cleanup-script-sha256");args=ap.parse_args()
 {"retrieve":retrieve,"commit-proof":commit_proof,"prune":prune,"cleanup":cleanup}[args.stage](args)
if __name__=="__main__":main()
