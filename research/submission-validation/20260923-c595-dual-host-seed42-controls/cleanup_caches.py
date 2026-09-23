#!/usr/bin/env python3
"""Authenticated, crash-safe removal of only job-owned fresh caches."""
import argparse, hashlib, json, os, pathlib, shutil, stat, subprocess

ORIGIN="https://github.com/CsabaConsulting/trainium_frontier.git"
BRANCH="experiment/20260923-c595-dual-control"
ARMS={
 "control_trn21":("fs-20260923-c595-control-trn21-r1",pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn21")),
 "control_trn22":("fs-20260923-c595-control-trn22-r1",pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn22")),
 "control_trn21_r2":("fs-20260924-c595-control-trn21-r2",pathlib.Path("/home/ubuntu/trainium_freshstart_20260924_c595_control_trn21_r2")),
 "control_trn22_r2":("fs-20260924-c595-control-trn22-r2",pathlib.Path("/home/ubuntu/trainium_freshstart_20260924_c595_control_trn22_r2")),
}
CACHE_NAMES=("neff","local","hlo","inductor","compiler","debug")
PROOF_KEYS={"schema","arm","label","archive_path","archive_sha256","checkpoint_path",
 "checkpoint_sha256","checkpoint_bytes","origin","branch","retrieved_verified_committed_pushed","evidence_commit"}

def file_bytes(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  info=os.fstat(fd)
  if not stat.S_ISREG(info.st_mode):raise SystemExit("not regular")
  with os.fdopen(fd,"rb",closefd=False) as stream:return stream.read()
 finally:os.close(fd)
def sha(path):return hashlib.sha256(file_bytes(path)).hexdigest()
def no_symlink(path,leaf=True):
 current=pathlib.Path(path.anchor)
 for index,part in enumerate(path.parts[1:]):
  current/=part
  try:info=os.lstat(current)
  except FileNotFoundError:
   if index==len(path.parts[1:])-1 and not leaf:return
   raise
  if stat.S_ISLNK(info.st_mode):raise SystemExit("symlink rejected: "+str(current))
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
def write_exclusive(path,value):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:os.write(fd,(json.dumps(value,sort_keys=True)+"\n").encode());os.fsync(fd)
 finally:os.close(fd)
 fsync_dir(path.parent)
def reject_tree_symlinks(root):
 no_symlink(root)
 if not root.is_dir():raise SystemExit("cache path not directory")
 for current,dirs,files in os.walk(root,followlinks=False):
  for name in [*dirs,*files]:
   if stat.S_ISLNK(os.lstat(pathlib.Path(current)/name).st_mode):raise SystemExit("cache tree symlink rejected")

def validate_proof(value,arm,label,root=None):
 if set(value)!=PROOF_KEYS:raise SystemExit("proof schema mismatch")
 if (value["schema"]!="c595-control-evidence-push-v1" or value["arm"]!=arm or value["label"]!=label
  or value["origin"]!=ORIGIN or value["branch"]!=BRANCH or value["retrieved_verified_committed_pushed"] is not True):raise SystemExit("proof identity mismatch")
 if type(value["checkpoint_bytes"]) is not int or value["checkpoint_bytes"]<=0:raise SystemExit("proof size mismatch")
 for key,size in (("archive_sha256",64),("checkpoint_sha256",64),("evidence_commit",40)):
  if type(value[key]) is not str or len(value[key])!=size or any(c not in "0123456789abcdef" for c in value[key]):raise SystemExit("proof digest mismatch")
 if root is not None:
  expected_archive=root/"out"/"archives"/(label+".tar.gz")
  expected_checkpoint=root/"out"/label/"train"/"out"/"final.pt"
  if (value["archive_path"],value["checkpoint_path"])!=(str(expected_archive),str(expected_checkpoint)):raise SystemExit("proof path mismatch")
def validate_prune(value,proof):
 expected={**proof,"state":"complete","deleted":True,"other_deletions":[]}
 if set(value)!=set(expected) or value!=expected:raise SystemExit("prune receipt contract mismatch")
def bound_identity(arm,label,head,proof_sha,prune_sha,caches):
 return {"schema":"c595-cache-cleanup-journal-v1","arm":arm,"label":label,"core_commit":head,
  "proof_sha256":proof_sha,"prune_receipt_sha256":prune_sha,"allowlist":list(map(str,caches)),"preserved_logs_results":True}
def validate_journal(value,bound,caches):
 if set(value)!=set(bound)|{"completed","active_path","state"} or any(value[k]!=v for k,v in bound.items()):raise SystemExit("cleanup journal identity/schema mismatch")
 allow=list(map(str,caches));completed=value["completed"]
 if type(completed) is not list or completed!=allow[:len(completed)] or len(completed)>len(allow):raise SystemExit("cleanup journal non-monotone prefix")
 active=value["active_path"]
 if active is not None and (len(completed)==len(allow) or active!=allow[len(completed)]):raise SystemExit("cleanup journal active path mismatch")
 if value["state"] not in ("in_progress","complete"):raise SystemExit("cleanup journal state mismatch")
 if value["state"]=="complete" and (completed!=allow or active is not None):raise SystemExit("premature cleanup complete")
 for path in completed:
  if os.path.lexists(path):raise SystemExit("completed cache path still exists")
 return value
def validate_transition(old,new,bound,caches):
 validate_journal(old,bound,caches);validate_journal(new,bound,caches);a,b=old["completed"],new["completed"]
 if len(b)<len(a) or len(b)>len(a)+1:raise SystemExit("invalid cleanup journal transition")
 if len(b)==len(a)+1 and old["active_path"]!=b[-1]:raise SystemExit("completion without durable active intent")
 if len(b)==len(a) and old["active_path"] is not None and new["active_path"] not in (old["active_path"],None):raise SystemExit("invalid active transition")
def replace_journal(path,value,bound,caches):
 temp=path.with_name(path.name+".tmp")
 if os.path.lexists(temp):raise SystemExit("unreconciled cleanup journal temp")
 if path.exists():validate_transition(json.loads(file_bytes(path)),value,bound,caches)
 else:validate_journal(value,bound,caches)
 write_exclusive(temp,value);os.replace(temp,path);fsync_dir(path.parent)
def reconcile_temp(path,bound,caches,initial):
 temp=path.with_name(path.name+".tmp")
 if not os.path.lexists(temp):return
 no_symlink(temp);pending=json.loads(file_bytes(temp))
 if path.exists():validate_transition(json.loads(file_bytes(path)),pending,bound,caches)
 elif pending!=initial:raise SystemExit("invalid initial cleanup journal temp")
 os.replace(temp,path);fsync_dir(path.parent)
def validate_receipt(value,bound,caches,journal_sha):
 expected={**bound,"schema":"c595-cache-cleanup-receipt-v1","completed":list(map(str,caches)),"active_path":None,"state":"complete","journal_sha256":journal_sha}
 if set(value)!=set(expected) or value!=expected:raise SystemExit("cleanup receipt contract mismatch")
 if any(os.path.lexists(path) for path in caches):raise SystemExit("cleanup receipt claims present cache")

def execute(args):
 script=pathlib.Path(__file__)
 if sha(script)!=args.expected_script_sha256:raise SystemExit("cleanup script hash mismatch")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=script.parents[3],text=True).strip()
 if head!=args.expected_core_commit:raise SystemExit("core mismatch")
 label,root=ARMS[args.arm];sandbox=root/"out"/label
 proof=sandbox/"evidence-pushed-receipt.json";prune=sandbox/"checkpoint-prune-receipt.json";journal=sandbox/"cache-cleanup-journal.json";receipt=sandbox/"cache-cleanup-receipt.json"
 if tuple(map(str,(args.label,args.evidence_pushed_receipt,args.prune_receipt,args.receipt)))!=tuple(map(str,(label,proof,prune,receipt))):raise SystemExit("lexical cleanup contract mismatch")
 for path in (proof,prune):no_symlink(path)
 if sha(proof)!=args.evidence_pushed_receipt_sha256 or sha(prune)!=args.prune_receipt_sha256:raise SystemExit("prerequisite receipt hash mismatch")
 proof_data=json.loads(file_bytes(proof));prune_data=json.loads(file_bytes(prune));validate_proof(proof_data,args.arm,label,root);validate_prune(prune_data,proof_data)
 rehearsal=sandbox/"train"/"rehearsal.json";no_symlink(rehearsal);caches=[sandbox/"train"/"caches"/name for name in CACHE_NAMES]
 if set(json.loads(file_bytes(rehearsal)).get("fresh_empty_sdk_caches",{}).values())!=set(map(str,caches)):raise SystemExit("fresh-cache ownership receipt mismatch")
 bound=bound_identity(args.arm,label,head,args.evidence_pushed_receipt_sha256,args.prune_receipt_sha256,caches)
 initial={**bound,"completed":[],"active_path":None,"state":"in_progress"};reconcile_temp(journal,bound,caches,initial)
 if not journal.exists():replace_journal(journal,initial,bound,caches)
 current=validate_journal(json.loads(file_bytes(journal)),bound,caches)
 if receipt.exists():validate_receipt(json.loads(file_bytes(receipt)),bound,caches,sha(journal));return receipt
 for cache in caches:
  name=str(cache)
  if name in current["completed"]:continue
  if current["active_path"] is None:
   if not cache.exists():raise SystemExit("missing cache before durable intent")
   reject_tree_symlinks(cache);current={**current,"active_path":name};replace_journal(journal,current,bound,caches)
  if cache.exists():reject_tree_symlinks(cache);shutil.rmtree(cache);fsync_dir(cache.parent)
  current={**current,"completed":[*current["completed"],name],"active_path":None};replace_journal(journal,current,bound,caches)
 current={**current,"state":"complete"};replace_journal(journal,current,bound,caches)
 result={**bound,"schema":"c595-cache-cleanup-receipt-v1","completed":list(map(str,caches)),"active_path":None,"state":"complete","journal_sha256":sha(journal)}
 write_exclusive(receipt,result);return receipt

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=ARMS,required=True);ap.add_argument("--label",required=True)
 ap.add_argument("--evidence-pushed-receipt",type=pathlib.Path,required=True);ap.add_argument("--evidence-pushed-receipt-sha256",required=True);ap.add_argument("--prune-receipt",type=pathlib.Path,required=True);ap.add_argument("--prune-receipt-sha256",required=True)
 ap.add_argument("--receipt",type=pathlib.Path,required=True);ap.add_argument("--expected-core-commit",required=True);ap.add_argument("--expected-script-sha256",required=True);print(execute(ap.parse_args()))
if __name__=="__main__":main()
