#!/usr/bin/env python3
import hashlib,importlib.util,json,pathlib,subprocess,tempfile,types
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("cleanup",HERE/"cleanup_caches.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="c595-cleanup-") as temporary:
 root=pathlib.Path(temporary)/"remote";label="test-label";arm="test_arm";sandbox=root/"out"/label
 module.ARMS={arm:(label,root)}
 (sandbox/"train/caches").mkdir(parents=True)
 caches=[sandbox/"train/caches"/name for name in module.CACHE_NAMES]
 target=pathlib.Path(temporary)/"outside";target.mkdir();(target/"must-preserve").write_text("yes")
 caches[0].symlink_to(target,target_is_directory=True)
 for cache in caches[1:]:cache.mkdir();(cache/"entry").write_text("cache")
 (sandbox/"training.log").write_text("preserve")
 proof={"schema":"c595-control-evidence-push-v1","arm":arm,"label":label,
  "archive_path":str(root/"out/archives"/(label+".tar.gz")),"archive_sha256":"a"*64,
  "checkpoint_path":str(sandbox/"train/out/final.pt"),"checkpoint_sha256":"b"*64,
  "checkpoint_bytes":123,"origin":module.ORIGIN,"branch":module.BRANCH,
  "retrieved_verified_committed_pushed":True,"evidence_commit":"c"*40}
 prune={**proof,"state":"complete","deleted":True,"other_deletions":[]}
 proof_path=sandbox/"evidence-pushed-receipt.json";proof_path.write_text(json.dumps(proof))
 prune_path=sandbox/"checkpoint-prune-receipt.json";prune_path.write_text(json.dumps(prune))
 rehearsal=sandbox/"train/rehearsal.json";rehearsal.write_text(json.dumps({"fresh_empty_sdk_caches":{str(i):str(p) for i,p in enumerate(caches)}}))
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 args=types.SimpleNamespace(arm=arm,label=label,evidence_pushed_receipt=proof_path,
   evidence_pushed_receipt_sha256=sha(proof_path),prune_receipt=prune_path,prune_receipt_sha256=sha(prune_path),
   receipt=sandbox/"cache-cleanup-receipt.json",expected_core_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(),
   expected_script_sha256=sha(HERE/"cleanup_caches.py"))
 try:module.execute(args)
 except SystemExit:pass
 else:raise AssertionError("symlink cache accepted")
 assert (target/"must-preserve").read_text()=="yes" and (sandbox/"training.log").is_file()
 caches[0].unlink();caches[0].mkdir();(caches[0]/"entry").write_text("cache")
 receipt=module.execute(args);record=json.loads(receipt.read_text())
 assert record["state"]=="complete" and record["preserved_logs_results"] is True
 assert not any(path.exists() for path in caches) and (sandbox/"training.log").read_text()=="preserve"
 assert module.execute(args)==receipt

 # Exact-schema validators reject forged minimal prerequisites and receipts.
 for bad in ({"arm":arm}, {**proof,"extra":True}):
  try:module.validate_proof(bad,arm,label)
  except SystemExit:pass
  else:raise AssertionError("forged proof accepted")
 try:module.validate_prune({"arm":arm,"state":"complete"},proof)
 except SystemExit:pass
 else:raise AssertionError("forged prune receipt accepted")

 # A completed journal/receipt can never authenticate a still-present cache.
 bound=module.bound_identity(arm,label,args.expected_core_commit,args.evidence_pushed_receipt_sha256,args.prune_receipt_sha256,caches)
 caches[0].mkdir()
 forged={**bound,"completed":list(map(str,caches)),"active_path":None,"state":"complete","absence_lstat":{str(path):"ENOENT" for path in caches}}
 try:module.validate_journal(forged,bound,caches)
 except SystemExit:pass
 else:raise AssertionError("all-completed journal accepted present cache")
 caches[0].rmdir()

 # Invalid and valid stale temp handling is decided before any deletion.
 journal=sandbox/"cache-cleanup-journal-recovery.json";initial={**bound,"completed":[],"active_path":None,"state":"in_progress","absence_lstat":{}}
 bad_temp=journal.with_name(journal.name+".tmp");bad_temp.write_text(json.dumps({"forged":True}))
 try:module.reconcile_temp(journal,bound,caches,initial)
 except SystemExit:pass
 else:raise AssertionError("invalid stale temp accepted")
 bad_temp.unlink();bad_temp.write_text(json.dumps(initial));module.reconcile_temp(journal,bound,caches,initial)
 assert journal.exists() and not bad_temp.exists()
print("c595 cache cleanup symlink/idempotence/preservation: PASS")
