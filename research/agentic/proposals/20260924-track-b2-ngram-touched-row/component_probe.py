#!/usr/bin/env python3
"""Fail-closed native identity/component probe. It never trains or mutates a checkpoint."""
import argparse,hashlib,importlib.util,json,os,pathlib,re,socket,stat,subprocess,sys

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CONTROL_SHA="35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
PREPARE_SHA="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
TABLE_KEYS=tuple(f"ngram_embeds.{key}.weight" for key in ("b1","t1","b5","t5"))
AUTH_KEYS={"schema","state","probe_label","expected_hostname","expected_core_commit","expected_request_sha256","control_source_commit","control_train_py","control_train_sha256","candidate_train_py","candidate_train_sha256","track_a_receipt_path","track_a_receipt_sha256","checkpoint_path","checkpoint_sha256","checkpoint_bytes","world_size","lnc","training_authorized","checkpoint_mutation_authorized","prune_authorized","cleanup_authorized"}
RECEIPT_KEYS={"schema","status","ready_to_retrieve","label","hostname","source_commit","candidate_sha256","checkpoint_path","checkpoint_sha256","checkpoint_bytes","training_completed","finite","artifact_manifest_sha256","request_sha256","authorization_sha256","runtime_job_sha256"}
HEX64=re.compile(r"[0-9a-f]{64}")

def nofollow_bytes(path):
    path=path.absolute();current=pathlib.Path(path.anchor)
    for part in path.parts[1:]:
        current/=part;mode=os.lstat(current).st_mode
        if stat.S_ISLNK(mode):raise SystemExit("symlink rejected")
    if not stat.S_ISREG(os.lstat(path).st_mode):raise SystemExit("regular file required")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        chunks=[]
        while True:
            value=os.read(fd,1<<20)
            if not value:break
            chunks.append(value)
        return b"".join(chunks)
    finally:os.close(fd)

def sha(data):return hashlib.sha256(data).hexdigest()
def strict_json_bytes(data,name):
    try:value=json.loads(data)
    except (UnicodeDecodeError,json.JSONDecodeError):raise SystemExit(name+" invalid JSON")
    if type(value) is not dict:raise SystemExit(name+" must be object")
    return value

def validate_authorization(auth,auth_sha):
    if set(auth)!=AUTH_KEYS or auth["schema"]!="track-b2-native-component-probe-authorization-v1":raise SystemExit("authorization schema mismatch")
    if auth["state"]!="authorized_component_probe_only":raise SystemExit("probe remains blocked pending authenticated bindings")
    if auth["training_authorized"] is not False or auth["checkpoint_mutation_authorized"] is not False or auth["prune_authorized"] is not False or auth["cleanup_authorized"] is not False:raise SystemExit("mutation authorization rejected")
    if auth["control_source_commit"]!="c51f6fdc5ca77793a1524148a563cafac0ab22b0" or auth["control_train_sha256"]!=CONTROL_SHA or auth["control_train_py"]!="submissions/dense6x1024_public0992851/train.py":raise SystemExit("control identity mismatch")
    for key in ("candidate_train_sha256","track_a_receipt_sha256","checkpoint_sha256","expected_request_sha256"):
        if type(auth[key]) is not str or not HEX64.fullmatch(auth[key]):raise SystemExit(key+" invalid")
    if type(auth["checkpoint_bytes"]) is not int or auth["checkpoint_bytes"]<=0 or auth["world_size"]!=4 or auth["lnc"]!=2:raise SystemExit("shape/size binding mismatch")
    for key in ("expected_hostname","expected_core_commit","candidate_train_py","track_a_receipt_path","checkpoint_path"):
        if type(auth[key]) is not str or not auth[key]:raise SystemExit(key+" unbound")
    if not re.fullmatch(r"[0-9a-f]{40}",auth["expected_core_commit"]):raise SystemExit("core invalid")
    if auth["candidate_train_py"]==auth["control_train_py"] or auth["candidate_train_sha256"]==CONTROL_SHA:raise SystemExit("candidate must be a distinct authenticated implementation")
    return auth_sha

def validate_track_a_receipt(auth):
    path=ROOT/auth["track_a_receipt_path"];data=nofollow_bytes(path)
    if sha(data)!=auth["track_a_receipt_sha256"]:raise SystemExit("Track A receipt digest mismatch")
    receipt=strict_json_bytes(data,"Track A receipt")
    if set(receipt)!=RECEIPT_KEYS or receipt["schema"]!="track-a-35c9-control-completion-v1" or receipt["status"]!="success" or receipt["ready_to_retrieve"] is not True or receipt["training_completed"] is not True or receipt["finite"] is not True:raise SystemExit("Track A receipt result mismatch")
    exact=(receipt["source_commit"]==auth["control_source_commit"] and receipt["candidate_sha256"]==CONTROL_SHA and receipt["hostname"]==auth["expected_hostname"] and receipt["checkpoint_path"]==auth["checkpoint_path"] and receipt["checkpoint_sha256"]==auth["checkpoint_sha256"] and receipt["checkpoint_bytes"]==auth["checkpoint_bytes"])
    if not exact:raise SystemExit("Track A receipt binding mismatch")
    for key in ("artifact_manifest_sha256","request_sha256","authorization_sha256","runtime_job_sha256"):
        if type(receipt[key]) is not str or not HEX64.fullmatch(receipt[key]):raise SystemExit("Track A evidence digest invalid")
    return sha(data)

def tensor_bytes(tensor):
    value=tensor.detach().cpu().contiguous()
    if str(value.dtype)=="torch.bfloat16":value=value.view(__import__("torch").uint16)
    return value.numpy().tobytes()

def load_observed(module,checkpoint,device,torch):
    calls=[];original=torch.nn.Module.load_state_dict
    def observed(self,state_dict,*args,**kwargs):
        strict=kwargs.get("strict",args[0] if args else True);calls.append((tuple(state_dict.keys()),strict));return original(self,state_dict,*args,**kwargs)
    torch.nn.Module.load_state_dict=observed
    try:model=module.load_for_eval(str(checkpoint),device)
    finally:torch.nn.Module.load_state_dict=original
    if len(calls)!=1 or calls[0][1] is not True:raise SystemExit("strict load was not observed exactly once")
    return model,calls[0][0]

def import_path(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--authorization",type=pathlib.Path,required=True);ap.add_argument("--authorization-sha256",required=True);ap.add_argument("--request-sha256",required=True);ap.add_argument("--core-commit",required=True);ap.add_argument("--output",type=pathlib.Path,required=True);args=ap.parse_args()
    auth_bytes=nofollow_bytes(args.authorization);auth_sha=sha(auth_bytes)
    if auth_sha!=args.authorization_sha256:raise SystemExit("authorization digest mismatch")
    auth=strict_json_bytes(auth_bytes,"authorization");validate_authorization(auth,auth_sha)
    if args.request_sha256!=auth["expected_request_sha256"] or args.core_commit!=auth["expected_core_commit"]:raise SystemExit("request/core mismatch")
    if socket.gethostname()!=auth["expected_hostname"]:raise SystemExit("host mismatch")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=args.core_commit or subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True):raise SystemExit("checkout mismatch")
    if sha(nofollow_bytes(ROOT/auth["control_train_py"]))!=CONTROL_SHA or sha(nofollow_bytes(ROOT/"prepare.py"))!=PREPARE_SHA or sha(nofollow_bytes(ROOT/auth["candidate_train_py"]))!=auth["candidate_train_sha256"]:raise SystemExit("source digest mismatch")
    validate_track_a_receipt(auth);checkpoint=pathlib.Path(auth["checkpoint_path"]);checkpoint_data=nofollow_bytes(checkpoint)
    if len(checkpoint_data)!=auth["checkpoint_bytes"] or sha(checkpoint_data)!=auth["checkpoint_sha256"]:raise SystemExit("checkpoint identity mismatch")
    if args.output.exists():raise SystemExit("non-overwrite output exists")
    # Heavy imports and any Neuron access happen only after every static/evidence binding passes.
    import torch,torch.distributed as dist,torch_neuronx
    if not dist.is_initialized() or dist.get_world_size()!=4 or int(os.environ.get("NEURON_LOGICAL_NC_CONFIG","0"))!=2:raise SystemExit("native four-rank LNC2 required")
    device=torch.device("neuron");control=import_path(ROOT/auth["control_train_py"],"track_b2_control");candidate=import_path(ROOT/auth["candidate_train_py"],"track_b2_candidate")
    control_model,control_keys=load_observed(control,checkpoint,device,torch);candidate_model,candidate_keys=load_observed(candidate,checkpoint,device,torch)
    if control_keys!=candidate_keys:raise SystemExit("ordered checkpoint key mismatch")
    cs=control_model.state_dict();ds=candidate_model.state_dict()
    table_hashes={}
    for key in TABLE_KEYS:
        if key not in cs or key not in ds or tensor_bytes(cs[key])!=tensor_bytes(ds[key]):raise SystemExit("ngram table byte mismatch")
        table_hashes[key]=sha(tensor_bytes(cs[key]))
    fixed=((torch.arange(1024,dtype=torch.int64)*7919+17)%8192).reshape(1,1024).to(device)
    with torch.no_grad():a=control_model(fixed);b=candidate_model(fixed)
    if isinstance(a,tuple):a=a[0]
    if isinstance(b,tuple):b=b[0]
    if not torch.equal(a,b):raise SystemExit("fixed-input logits mismatch")
    export=getattr(candidate,"track_b2_native_component_probe",None)
    if not callable(export):raise SystemExit("candidate component export missing")
    rank=dist.get_rank();evidence=export(device=device,rank=rank,world_size=4)
    required={"schema","duplicates_present","avg_world_size","dense_param","touched_param","dense_moment","touched_logical_moment","untouched_param_before","untouched_param_after","untouched_state_before","untouched_state_after","ordering"}
    if type(evidence) is not dict or set(evidence)!=required or evidence["schema"]!="track-b2-native-bf16-ordering-v1" or evidence["duplicates_present"] is not True or evidence["avg_world_size"]!=4:raise SystemExit("component evidence schema mismatch")
    for left,right in (("dense_param","touched_param"),("dense_moment","touched_logical_moment"),("untouched_param_before","untouched_param_after"),("untouched_state_before","untouched_state_after")):
        if not torch.equal(evidence[left],evidence[right]):raise SystemExit("component exactness mismatch: "+left)
    if evidence["ordering"]!={"occurrence_accumulation":"bf16","collective":"sum_then_divide_world","state":"candidate_reported_and_tensor_checked"}:raise SystemExit("ordering evidence mismatch")
    local_evidence={"rank":rank,"dense_param_sha256":sha(tensor_bytes(evidence["dense_param"])),"touched_param_sha256":sha(tensor_bytes(evidence["touched_param"])),"dense_moment_sha256":sha(tensor_bytes(evidence["dense_moment"])),"touched_logical_moment_sha256":sha(tensor_bytes(evidence["touched_logical_moment"])),"untouched_param_sha256":sha(tensor_bytes(evidence["untouched_param_after"])),"untouched_state_sha256":sha(tensor_bytes(evidence["untouched_state_after"]))}
    rank_evidence=[None]*4;dist.all_gather_object(rank_evidence,local_evidence)
    if [item["rank"] for item in rank_evidence]!=list(range(4)):raise SystemExit("rank evidence consensus mismatch")
    versions={"python":sys.version,"torch":torch.__version__,"torch_neuronx":getattr(torch_neuronx,"__version__",None),"neuron_cc":subprocess.run(["neuronx-cc","--version"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30).stdout.strip()}
    result={"schema":"track-b2-native-component-probe-result-v1","label":auth["probe_label"],"hostname":socket.gethostname(),"core_commit":args.core_commit,"request_sha256":args.request_sha256,"authorization_sha256":auth_sha,"checkpoint_sha256":auth["checkpoint_sha256"],"ordered_keys":list(control_keys),"table_hashes":table_hashes,"input_sha256":sha(tensor_bytes(fixed)),"logits_sha256":sha(tensor_bytes(a)),"rank_evidence":rank_evidence,"versions":versions,"native_bf16_ordering":evidence["ordering"],"training_executed":False,"checkpoint_mutated":False,"training_authorized":False,"ready_to_retrieve":True}
    if rank!=0:
        dist.barrier();return
    args.output.parent.mkdir(parents=True,exist_ok=True);fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    data=(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode()
    try:
        view=memoryview(data)
        while view:
            count=os.write(fd,view)
            if count<=0:raise OSError("short result write")
            view=view[count:]
        os.fsync(fd)
    finally:os.close(fd)
    if nofollow_bytes(args.output)!=data:raise SystemExit("result persistence mismatch")
    dist.barrier()
    print(json.dumps({"result_sha256":sha(data),"ready_to_retrieve":True},sort_keys=True))
if __name__=="__main__":main()
