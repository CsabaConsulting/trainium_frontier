#!/usr/bin/env python3
"""Remove only job-owned fresh caches after pushed evidence and checkpoint prune."""
import argparse, hashlib, json, os, pathlib, shutil, stat, subprocess

HERE=pathlib.Path(__file__).resolve().parent
ARMS={
 "control_trn21":("fs-20260923-c595-control-trn21-r1",pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn21")),
 "control_trn22":("fs-20260923-c595-control-trn22-r1",pathlib.Path("/home/ubuntu/trainium_freshstart_20260923_c595_control_trn22")),
}
CACHE_NAMES=("neff","local","hlo","inductor","compiler","debug")

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
def atomic_replace(path,value):
 temp=path.with_name(path.name+".tmp")
 fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:os.write(fd,(json.dumps(value,sort_keys=True)+"\n").encode());os.fsync(fd)
 finally:os.close(fd)
 os.replace(temp,path);fsync_dir(path.parent)
def reject_tree_symlinks(root):
 no_symlink(root)
 if not root.is_dir():raise SystemExit("cache path not directory")
 for current,dirs,files in os.walk(root,followlinks=False):
  for name in [*dirs,*files]:
   if stat.S_ISLNK(os.lstat(pathlib.Path(current)/name).st_mode):raise SystemExit("cache tree symlink rejected")

def execute(args):
 script=pathlib.Path(__file__)
 if sha(script)!=args.expected_script_sha256:raise SystemExit("cleanup script hash mismatch")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=script.parents[3],text=True).strip()
 if head!=args.expected_core_commit:raise SystemExit("core mismatch")
 label,root=ARMS[args.arm];sandbox=root/"out"/label
 proof=sandbox/"evidence-pushed-receipt.json";prune=sandbox/"checkpoint-prune-receipt.json"
 journal=sandbox/"cache-cleanup-journal.json";receipt=sandbox/"cache-cleanup-receipt.json"
 if tuple(map(str,(args.label,args.evidence_pushed_receipt,args.prune_receipt,args.receipt))) != tuple(map(str,(label,proof,prune,receipt))):raise SystemExit("lexical cleanup contract mismatch")
 for path in (proof,prune):no_symlink(path)
 if sha(proof)!=args.evidence_pushed_receipt_sha256 or sha(prune)!=args.prune_receipt_sha256:raise SystemExit("prerequisite receipt hash mismatch")
 proof_data=json.loads(file_bytes(proof));prune_data=json.loads(file_bytes(prune))
 if (proof_data.get("schema")!="c595-control-evidence-push-v1" or proof_data.get("arm")!=args.arm
     or proof_data.get("label")!=label or proof_data.get("retrieved_verified_committed_pushed") is not True):raise SystemExit("evidence not pushed/bound")
 if (prune_data.get("arm")!=args.arm or prune_data.get("label")!=label
     or prune_data.get("state")!="complete" or prune_data.get("deleted") is not True):raise SystemExit("checkpoint not pruned/bound")
 rehearsal=sandbox/"train"/"rehearsal.json";no_symlink(rehearsal)
 rehearsal_data=json.loads(file_bytes(rehearsal));declared=set(rehearsal_data.get("fresh_empty_sdk_caches",{}).values())
 caches=[sandbox/"train"/"caches"/name for name in CACHE_NAMES]
 if declared!=set(map(str,caches)):raise SystemExit("fresh-cache ownership receipt mismatch")
 base={"schema":"c595-cache-cleanup-journal-v1","arm":args.arm,"label":label,
       "core_commit":head,"proof_sha256":args.evidence_pushed_receipt_sha256,
       "prune_receipt_sha256":args.prune_receipt_sha256,"allowlist":list(map(str,caches)),
       "completed":[],"active_path":None,"state":"in_progress","preserved_logs_results":True}
 if journal.exists():
  current=json.loads(file_bytes(journal))
  for key,value in base.items():
   if key not in ("completed","active_path","state") and current.get(key)!=value:raise SystemExit("cleanup journal identity mismatch")
 else:atomic_replace(journal,base);current=base
 if receipt.exists():
  done=json.loads(file_bytes(receipt))
  if done.get("state")=="complete" and done.get("completed")==list(map(str,caches)):return receipt
  raise SystemExit("invalid existing cleanup receipt")
 for cache in caches:
  name=str(cache)
  if name in current["completed"]:continue
  if current.get("active_path") not in (None,name):raise SystemExit("cleanup journal active path mismatch")
  if current.get("active_path") is None:
   reject_tree_symlinks(cache);current["active_path"]=name;atomic_replace(journal,current)
  if cache.exists():reject_tree_symlinks(cache);shutil.rmtree(cache);fsync_dir(cache.parent)
  current["completed"].append(name);current["active_path"]=None;atomic_replace(journal,current)
 current["state"]="complete";atomic_replace(journal,current)
 result={**current,"schema":"c595-cache-cleanup-receipt-v1","journal_sha256":sha(journal)}
 fd=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:os.write(fd,(json.dumps(result,sort_keys=True)+"\n").encode());os.fsync(fd)
 finally:os.close(fd)
 fsync_dir(receipt.parent);return receipt

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=ARMS,required=True);ap.add_argument("--label",required=True)
 ap.add_argument("--evidence-pushed-receipt",type=pathlib.Path,required=True);ap.add_argument("--evidence-pushed-receipt-sha256",required=True)
 ap.add_argument("--prune-receipt",type=pathlib.Path,required=True);ap.add_argument("--prune-receipt-sha256",required=True)
 ap.add_argument("--receipt",type=pathlib.Path,required=True);ap.add_argument("--expected-core-commit",required=True);ap.add_argument("--expected-script-sha256",required=True)
 print(execute(ap.parse_args()))
if __name__=="__main__":main()
