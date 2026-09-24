#!/usr/bin/env python3
"""Fail-closed native identity/component probe. It never trains or mutates a checkpoint."""
import argparse,hashlib,importlib.util,json,os,pathlib,re,socket,stat,subprocess,sys

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CONTROL_SHA="35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
PREPARE_SHA="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
TABLE_KEYS=tuple(f"ngram_embeds.{key}.weight" for key in ("b1","t1","b5","t5"))
AUTH_KEYS={"schema","state","probe_label","expected_hostname","expected_core_commit","expected_request_contract_sha256","control_source_commit","control_train_py","control_train_sha256","candidate_train_py","candidate_train_sha256","track_a_receipt_path","track_a_receipt_sha256","checkpoint_path","checkpoint_sha256","checkpoint_bytes","world_size","lnc","training_authorized","checkpoint_mutation_authorized","prune_authorized","cleanup_authorized"}
RECEIPT_KEYS={"schema","status","ready_to_retrieve","label","hostname","source_commit","candidate_sha256","checkpoint_path","checkpoint_sha256","checkpoint_bytes","training_completed","finite","artifact_manifest_sha256","request_sha256","authorization_sha256","runtime_job_sha256"}
REQUEST_KEYS={"schema","state","label","expected_commit","expected_hostname","authorization","authorization_sha256","fingerprint_paths","expected_sha256","environment","command","planned_command","blocked_fields","dispatch_authorized","training_authorized","output_root","output"}
HEX64=re.compile(r"[0-9a-f]{64}")

def nofollow_fd(path,flags=os.O_RDONLY):
    path=path.absolute();parts=path.parts;fd=os.open(path.anchor,os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        result=os.open(parts[-1],flags|os.O_NOFOLLOW,dir_fd=fd)
    except OSError as exc:raise SystemExit("unsafe no-follow path: "+str(exc))
    finally:os.close(fd)
    return result

def nofollow_dir_fd(path):
    path=path.absolute();fd=os.open(path.anchor,os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        return fd
    except OSError as exc:
        os.close(fd);raise SystemExit("unsafe no-follow directory: "+str(exc))

def nofollow_bytes(path):
    fd=nofollow_fd(path)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise SystemExit("regular file required")
        chunks=[]
        while True:
            value=os.read(fd,1<<20)
            if not value:break
            chunks.append(value)
        return b"".join(chunks)
    finally:os.close(fd)

def nofollow_identity(path):
    fd=nofollow_fd(path);digest=hashlib.sha256();size=0
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise SystemExit("regular file required")
        while True:
            value=os.read(fd,1<<20)
            if not value:break
            digest.update(value);size+=len(value)
    finally:os.close(fd)
    return digest.hexdigest(),size

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
    for key in ("candidate_train_sha256","track_a_receipt_sha256","checkpoint_sha256","expected_request_contract_sha256"):
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

def request_contract(request):
    value={key:request[key] for key in REQUEST_KEYS-{"authorization_sha256","command","planned_command","blocked_fields","state","dispatch_authorized"}}
    value["command"]=["<BOUND_AT_FINALIZATION>"]
    return sha(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode())

def validate_request(request,data,args,auth):
    if set(request)!=REQUEST_KEYS or request["schema"]!="track-b2-native-component-probe-request-v1" or request["state"]!="authorized_component_probe_only" or request["dispatch_authorized"] is not True or request["training_authorized"] is not False:raise SystemExit("request schema/state mismatch")
    if request_contract(request)!=auth["expected_request_contract_sha256"] or request["expected_commit"]!=args.core_commit or request["expected_hostname"]!=auth["expected_hostname"]:raise SystemExit("request identity mismatch")
    if request["authorization"]!=str(args.authorization.relative_to(ROOT)) or request["authorization_sha256"]!=args.authorization_sha256:raise SystemExit("request authorization mismatch")
    if request["output"]!=str(args.output) or type(request["output_root"]) is not str or pathlib.Path(request["output_root"]).absolute()!=args.output.parent.absolute():raise SystemExit("request output binding mismatch")
    expected_command=["torchrun","--standalone","--nproc_per_node=4",str(pathlib.Path(__file__).relative_to(ROOT)),"--request-file",str(args.request_file.relative_to(ROOT)),"--authorization",str(args.authorization.relative_to(ROOT)),"--authorization-sha256",args.authorization_sha256,"--core-commit",args.core_commit,"--output",str(args.output)]
    if request["command"]!=expected_command:raise SystemExit("request argv mismatch")
    return sha(data)

def validate_candidate_schema(value):
    keys={"schema","param","logical_moment","physical_untouched_before","physical_untouched_after"}
    if type(value) is not dict or set(value)!=keys or value["schema"]!="track-b2-native-touched-result-v1":raise SystemExit("candidate touched-result schema mismatch")
    return value

def validate_rank_evidence(items):
    keys={"rank","local_contribution_sha256","global_gradient_sha256","param_sha256","logical_moment_sha256","untouched_state_sha256"}
    if type(items) is not list or len(items)!=4 or any(type(item) is not dict or set(item)!=keys for item in items) or [item["rank"] for item in items]!=list(range(4)):raise SystemExit("rank evidence consensus mismatch")
    for item in items:
        if any(type(item[key]) is not str or not HEX64.fullmatch(item[key]) for key in keys-{"rank"}):raise SystemExit("rank evidence digest mismatch")
    if len({item["local_contribution_sha256"] for item in items})!=4:raise SystemExit("rank contributions missing or duplicated")
    for key in ("global_gradient_sha256","param_sha256","logical_moment_sha256"):
        if len({item[key] for item in items})!=1:raise SystemExit("global rank consensus mismatch: "+key)
    return items

def secure_write_new_tree(root,output,data):
    root=root.absolute();output=output.absolute()
    if output.parent!=root or root.exists():raise SystemExit("output root must be absent and exclusively owned")
    parent=root.parent;parent_fd=nofollow_dir_fd(parent)
    try:os.mkdir(root.name,0o700,dir_fd=parent_fd)
    except OSError as exc:raise SystemExit("exclusive output-root creation failed: "+str(exc))
    finally:os.close(parent_fd)
    root_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:fd=os.open(output.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=root_fd)
    finally:os.close(root_fd)
    try:
        view=memoryview(data)
        while view:
            count=os.write(fd,view)
            if count<=0:raise OSError("short result write")
            view=view[count:]
        os.fsync(fd)
    finally:os.close(fd)

def tensor_bytes(tensor):
    value=tensor.detach().cpu().contiguous()
    if str(value.dtype)=="torch.bfloat16":value=value.view(__import__("torch").uint16)
    return value.numpy().tobytes()

def reduce_clipped_with_evidence(clipped,dist,bytes_fn,world_size=4):
    local_sha=sha(b"".join(bytes_fn(value) for value in clipped))
    for value in clipped:
        dist.all_reduce(value,op=dist.ReduceOp.SUM);value.div_(world_size)
    global_sha=sha(b"".join(bytes_fn(value) for value in clipped))
    return local_sha,global_sha

def native_fixture_and_control(torch,dist,rank,device):
    rows,dim,step,beta2,lr,eps=257,128,7,0.91,0.03,1e-10;raw={};dense=[]
    for table in range(4):
        indices=torch.tensor([3,17+table,3,31+rank,80+table*4+rank],dtype=torch.int64,device=device)
        base=torch.arange(indices.numel()*dim,dtype=torch.float32).reshape(indices.numel(),dim)
        values=(((base+rank*37+table*101)%211)-105).mul_(1/256).to(torch.bfloat16).to(device)
        local=torch.zeros((rows,dim),dtype=torch.bfloat16,device=device);local.index_add_(0,indices,values);raw[table]={"indices":indices,"values":values};dense.append(local)
    params=[torch.nn.Parameter(torch.zeros_like(value),requires_grad=True) for value in dense]
    for param,value in zip(params,dense):param.grad=value
    torch.nn.utils.clip_grad_norm_(params,2.0,foreach=True)
    clipped=[param.grad.detach().clone() for param in params]
    local_contribution_sha,global_gradient_sha=reduce_clipped_with_evidence(clipped,dist,tensor_bytes)
    initial_param=[];initial_moment=[];expected_param=[];expected_moment=[]
    for table,grad in enumerate(clipped):
        p=(((torch.arange(rows*dim,dtype=torch.float32).reshape(rows,dim)+table*13)%97)-48).mul_(1/128).to(torch.bfloat16).to(device)
        v=(((torch.arange(rows*dim,dtype=torch.float32).reshape(rows,dim)+table*7)%43)+1).mul_(1/4096).to(torch.bfloat16).to(device)
        initial_param.append(p.clone());initial_moment.append(v.clone());next_v=v.clone();next_v.lerp_(grad.square(),1-beta2);update=grad/((next_v/(1-beta2**step)).sqrt()+eps);next_p=p.clone();next_p.add_(update,alpha=-lr);expected_param.append(next_p);expected_moment.append(next_v)
    return {"rows":rows,"dim":dim,"step":step,"beta2":beta2,"lr":lr,"eps":eps,"max_norm":2.0,"tables":raw,"initial_param":initial_param,"initial_moment":initial_moment},torch.stack(expected_param),torch.stack(expected_moment),local_contribution_sha,global_gradient_sha

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
    ap=argparse.ArgumentParser();ap.add_argument("--request-file",type=pathlib.Path,required=True);ap.add_argument("--authorization",type=pathlib.Path,required=True);ap.add_argument("--authorization-sha256",required=True);ap.add_argument("--core-commit",required=True);ap.add_argument("--output",type=pathlib.Path,required=True);args=ap.parse_args()
    auth_bytes=nofollow_bytes(args.authorization);auth_sha=sha(auth_bytes)
    if auth_sha!=args.authorization_sha256:raise SystemExit("authorization digest mismatch")
    auth=strict_json_bytes(auth_bytes,"authorization");validate_authorization(auth,auth_sha)
    request_bytes=nofollow_bytes(args.request_file);request=strict_json_bytes(request_bytes,"request");validate_request(request,request_bytes,args,auth)
    request_sha=sha(request_bytes)
    if args.core_commit!=auth["expected_core_commit"]:raise SystemExit("request/core mismatch")
    if socket.gethostname()!=auth["expected_hostname"]:raise SystemExit("host mismatch")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=args.core_commit or subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True):raise SystemExit("checkout mismatch")
    if sha(nofollow_bytes(ROOT/auth["control_train_py"]))!=CONTROL_SHA or sha(nofollow_bytes(ROOT/"prepare.py"))!=PREPARE_SHA or sha(nofollow_bytes(ROOT/auth["candidate_train_py"]))!=auth["candidate_train_sha256"]:raise SystemExit("source digest mismatch")
    validate_track_a_receipt(auth);checkpoint=pathlib.Path(auth["checkpoint_path"]);checkpoint_pre_sha,checkpoint_pre_bytes=nofollow_identity(checkpoint)
    if checkpoint_pre_bytes!=auth["checkpoint_bytes"] or checkpoint_pre_sha!=auth["checkpoint_sha256"]:raise SystemExit("checkpoint identity mismatch")
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
    rank=dist.get_rank();fixture,expected_param,expected_moment,local_contribution_sha,global_gradient_sha=native_fixture_and_control(torch,dist,rank,device)
    evidence=validate_candidate_schema(export(fixture=fixture,device=device,rank=rank,world_size=4))
    if not torch.equal(evidence["param"],expected_param) or not torch.equal(evidence["logical_moment"],expected_moment):raise SystemExit("candidate differs from harness-owned dense control")
    if not torch.equal(evidence["physical_untouched_before"],evidence["physical_untouched_after"]):raise SystemExit("candidate mutated untouched physical state")
    local_evidence={"rank":rank,"local_contribution_sha256":local_contribution_sha,"global_gradient_sha256":global_gradient_sha,"param_sha256":sha(tensor_bytes(evidence["param"])),"logical_moment_sha256":sha(tensor_bytes(evidence["logical_moment"])),"untouched_state_sha256":sha(tensor_bytes(evidence["physical_untouched_after"]))}
    rank_evidence=[None]*4;dist.all_gather_object(rank_evidence,local_evidence)
    validate_rank_evidence(rank_evidence)
    versions={"python":sys.version,"torch":torch.__version__,"torch_neuronx":getattr(torch_neuronx,"__version__",None),"neuron_cc":subprocess.run(["neuronx-cc","--version"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30).stdout.strip()}
    checkpoint_post_sha,checkpoint_post_bytes=nofollow_identity(checkpoint)
    if (checkpoint_post_sha,checkpoint_post_bytes)!=(checkpoint_pre_sha,checkpoint_pre_bytes):raise SystemExit("checkpoint mutated during probe")
    args.request_sha256=request_sha
    result={"schema":"track-b2-native-component-probe-result-v1","label":auth["probe_label"],"hostname":socket.gethostname(),"core_commit":args.core_commit,"request_sha256":args.request_sha256,"authorization_sha256":auth_sha,"checkpoint_pre_sha256":checkpoint_pre_sha,"checkpoint_pre_bytes":checkpoint_pre_bytes,"checkpoint_post_sha256":checkpoint_post_sha,"checkpoint_post_bytes":checkpoint_post_bytes,"ordered_keys":list(control_keys),"table_hashes":table_hashes,"input_sha256":sha(tensor_bytes(fixed)),"logits_sha256":sha(tensor_bytes(a)),"rank_evidence":rank_evidence,"versions":versions,"native_bf16_ordering":{"fixture":"harness-owned","occurrence_accumulation":"native_bf16_index_add","clip":"one_joint_coefficient_four_tables_per_rank","collective":"four_rank_sum_then_divide_world","comparison":"bit_exact"},"training_executed":False,"checkpoint_mutated":False,"training_authorized":False,"ready_to_retrieve":True}
    if rank!=0:
        dist.barrier();return
    data=(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode()
    secure_write_new_tree(pathlib.Path(request["output_root"]),args.output,data)
    if nofollow_bytes(args.output)!=data:raise SystemExit("result persistence mismatch")
    dist.barrier()
    print(json.dumps({"result_sha256":sha(data),"ready_to_retrieve":True},sort_keys=True))
if __name__=="__main__":main()
