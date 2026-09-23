#!/usr/bin/env python3
import importlib.util,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
def load(name):
 spec=importlib.util.spec_from_file_location(name,HERE/(name+".py"));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
operator=load("post_run_operator");cleanup=load("cleanup_caches");prune=load("prune_checkpoint")
contract=json.loads((HERE/"post-run-contract.json").read_text())
assert [stage["name"] for stage in contract["stages"]]==["retrieve","commit-proof","prune","cleanup"]
assert contract["cache_allowlist"]==[f"train/caches/{name}" for name in cleanup.CACHE_NAMES]
for arm,entry in contract["arms"].items():
 host,root,label=operator.ARMS[arm]
 assert entry["ssh"]==host and entry["root"]==str(root) and entry["label"]==label
 assert cleanup.ARMS[arm]==(label,pathlib.Path(root)) and prune.ARMS[arm]==(label,pathlib.Path(root))
source=(HERE/"post_run_operator.py").read_text()
for required in ("retrieval-verification.json","archive-completion.json","evidence-pushed-receipt.json",
                 "checkpoint-prune-receipt.json","cache-cleanup-receipt.json","git","ls-remote","scp","ssh"):
 assert required in source
cleanup_source=(HERE/"cleanup_caches.py").read_text()
for required in ("O_NOFOLLOW","O_EXCL","active_path","shutil.rmtree","preserved_logs_results"):
 assert required in cleanup_source
print("c595 post-run literal path/lifecycle gates: PASS")
