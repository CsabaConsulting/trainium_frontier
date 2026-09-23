#!/usr/bin/env python3
import hashlib,importlib.util,json,pathlib,subprocess,tempfile
HERE=pathlib.Path(__file__).resolve().parent
def load(name):
 spec=importlib.util.spec_from_file_location(name,HERE/(name+".py"));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
operator=load("post_run_operator");cleanup=load("cleanup_caches");prune=load("prune_checkpoint")
contract=json.loads((HERE/"post-run-contract.json").read_text())
assert contract["ordering"]==["retrieve","commit-proof","prune","cleanup"]
assert contract["cache_allowlist"]==[f"train/caches/{name}" for name in cleanup.CACHE_NAMES]
for arm,entry in contract["arms"].items():
 host,root,label=operator.ARMS[arm]
 assert entry["ssh"]==host and entry["root"]==str(root) and entry["label"]==label
 assert cleanup.ARMS[arm]==(label,pathlib.Path(root)) and prune.ARMS[arm]==(label,pathlib.Path(root))
 assert set(entry["argv"])==set(contract["ordering"])
 for stage,argv in entry["argv"].items():
  assert argv[:2]==["/usr/bin/python3",contract["operator_script"]]
  assert argv[2:5]==[stage,"--arm",arm]
  subprocess.run([argv[0],argv[1],"--help"],cwd=HERE.parents[2],check=True,stdout=subprocess.DEVNULL)
digest=lambda name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
assert contract["operator_script_sha256"]==digest("post_run_operator.py")
assert contract["prune_script_sha256"]==digest("prune_checkpoint.py")
assert contract["cleanup_script_sha256"]==digest("cleanup_caches.py")
source=(HERE/"post_run_operator.py").read_text()
for required in ("retrieval-verification.json","archive-completion.json","evidence-pushed-receipt.json",
                 "checkpoint-prune-receipt.json","cache-cleanup-receipt.json","git","ls-remote","scp","ssh"):
 assert required in source
cleanup_source=(HERE/"cleanup_caches.py").read_text()
for required in ("O_NOFOLLOW","O_EXCL","active_path","shutil.rmtree","preserved_logs_results"):
 assert required in cleanup_source

# Exact receipt schemas reject minimal and extra-key forgeries.
sample={"schema":"c595-control-evidence-push-v1","arm":"a"}
for actual in ({"schema":"c595-control-evidence-push-v1"},{**sample,"extra":1}):
 try:operator.validate_exact(actual,sample,"test")
 except SystemExit:pass
 else:raise AssertionError("forged exact receipt accepted")

caches=[f"/remote/train/caches/{name}" for name in cleanup.CACHE_NAMES]
base={"arm":"control_trn21_r2","label":"label","core_commit":"c"*40,
 "candidate_sha256":operator.TRAIN_SHA,"proof_sha256":"a"*64,"prune_receipt_sha256":"b"*64,
 "allowlist":caches,"preserved_logs_results":True,"completed":caches,"active_path":None,
 "state":"complete","absence_lstat":{path:"ENOENT" for path in caches}}
journal={"schema":"c595-cache-cleanup-journal-v1",**base}
receipt={"schema":"c595-cache-cleanup-receipt-v1",**base,"journal_sha256":"d"*64}
operator.validate_terminal_cleanup(journal,receipt,"control_trn21_r2","label","c"*40,"a"*64,"b"*64,caches,"d"*64)
for bad_journal,bad_receipt in (
 (journal,{**receipt,"journal_sha256":"z"*64}),
 ({**journal,"absence_lstat":{**journal["absence_lstat"],caches[0]:"PRESENT"}},receipt),
 ({**journal,"completed":caches[:-1]},receipt),
 (journal,{**receipt,"candidate_sha256":"0"*64}),
):
 try:operator.validate_terminal_cleanup(bad_journal,bad_receipt,"control_trn21_r2","label","c"*40,"a"*64,"b"*64,caches,"d"*64)
 except SystemExit:pass
 else:raise AssertionError("forged/mutated cleanup evidence accepted")

# Evidence commits fail closed on either unrelated dirt or pre-staged content.
with tempfile.TemporaryDirectory(prefix="c595-git-state-") as temporary:
 root=pathlib.Path(temporary);subprocess.run(["git","init","-q"],cwd=root,check=True)
 subprocess.run(["git","config","user.email","test@example.invalid"],cwd=root,check=True);subprocess.run(["git","config","user.name","test"],cwd=root,check=True)
 (root/"base").write_text("base");subprocess.run(["git","add","base"],cwd=root,check=True);subprocess.run(["git","commit","-qm","base"],cwd=root,check=True)
 (root/"verification").write_text("v");(root/"completion").write_text("c");(root/"archive").write_text("a")
 operator.require_clean_evidence_state(root,["verification","completion"],["verification","completion","archive"])
 (root/"unrelated").write_text("x")
 try:operator.require_clean_evidence_state(root,["verification","completion"],["verification","completion","archive"])
 except SystemExit:pass
 else:raise AssertionError("unrelated dirt accepted")
 (root/"unrelated").unlink();subprocess.run(["git","add","verification"],cwd=root,check=True)
 try:operator.require_clean_evidence_state(root,["verification","completion"],["verification","completion","archive"])
 except SystemExit:pass
 else:raise AssertionError("pre-staged evidence accepted")
assert source.count("require_clean_evidence_state(ROOT,[],allowed)")==2
assert "cache-cleanup-journal.json" in source and "sha(local_journal)" in source
print("c595 post-run literal path/lifecycle gates: PASS")
