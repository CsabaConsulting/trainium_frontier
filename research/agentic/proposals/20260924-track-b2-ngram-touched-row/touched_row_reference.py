#!/usr/bin/env python3
"""NumPy CPU mathematical reference; not a Neuron implementation or launch harness."""
import io
import numpy as np


def bf16(value):
    """Round FP32 to BF16 (round-to-nearest-even), returned in an FP32 container."""
    x=np.asarray(value,dtype=np.float32);bits=x.view(np.uint32)
    rounded=bits+np.uint32(0x7FFF)+((bits>>np.uint32(16))&np.uint32(1))
    return (rounded&np.uint32(0xFFFF0000)).view(np.float32)


def coalesce(indices,values,rows):
    dense=np.zeros((rows,values.shape[1]),dtype=np.float32)
    for index,value in zip(indices.tolist(),values):dense[index]=np.float32(dense[index]+value)
    touched=np.unique(indices).astype(np.int64)
    return touched,dense[touched]


def locally_clip(indices,values,rows,max_norm):
    touched,summed=coalesce(indices,values,rows)
    norm=float(np.linalg.norm(summed));scale=min(1.0,max_norm/(norm+1e-6))
    return touched,np.float32(summed*np.float32(scale))


def dense_reference(param,moment,rank_records,step,beta2,lr,eps,max_norm,state_bf16):
    gathered=[]
    for indices,values in rank_records:
        touched,clipped=locally_clip(indices,values,param.shape[0],max_norm)
        gathered.append((touched,clipped))
    touched,compact=coalesce(np.concatenate([x[0] for x in gathered]),np.concatenate([x[1] for x in gathered]),param.shape[0])
    grad=np.zeros_like(moment,dtype=np.float32);grad[touched]=np.float32(compact/np.float32(len(gathered)))
    moment=np.float32(moment*np.float32(beta2)+grad*grad*np.float32(1.0-beta2))
    if state_bf16:moment=bf16(moment)
    update=grad/(np.sqrt(moment/np.float32(1.0-beta2**step))+np.float32(eps))
    return bf16(param-np.float32(lr)*update),moment


def materialize(moment,last_touch,touched,step,beta2,state_bf16,round_final=True):
    value=moment[touched].copy();delta=step-last_touch[touched]
    # Repeated decay, including state rounding, is the exact dense RMSProp contract.
    for age in range(1,int(delta.max(initial=0))+1):
        active=delta>=age;value[active]=np.float32(value[active]*np.float32(beta2))
        if state_bf16:
            rounded=active if round_final else (delta>age)
            value[rounded]=bf16(value[rounded])
    return value


def lazy_step(param,physical,last_touch,rank_records,step,beta2,lr,eps,max_norm,state_bf16):
    gathered=[locally_clip(i,v,param.shape[0],max_norm) for i,v in rank_records]
    indices=np.concatenate([x[0] for x in gathered]);values=np.concatenate([x[1] for x in gathered])
    touched,grad=coalesce(indices,values,param.shape[0]);grad=np.float32(grad/np.float32(len(gathered)));untouched=np.ones(param.shape[0],dtype=bool);untouched[touched]=False
    old=(param.copy(),physical.copy(),last_touch.copy())
    value=materialize(physical,last_touch,touched,step,beta2,state_bf16,round_final=False)
    value=np.float32(value+grad*grad*np.float32(1.0-beta2))
    if state_bf16:value=bf16(value)
    update=grad/(np.sqrt(value/np.float32(1.0-beta2**step))+np.float32(eps))
    param[touched]=bf16(param[touched]-np.float32(lr)*update);physical[touched]=value;last_touch[touched]=step
    assert np.array_equal(param[untouched],old[0][untouched])
    assert np.array_equal(physical[untouched],old[1][untouched])
    assert np.array_equal(last_touch[untouched],old[2][untouched])


def logical_moment(physical,last_touch,step,beta2,state_bf16):
    rows=np.arange(physical.shape[0],dtype=np.int64)
    return materialize(physical,last_touch,rows,step,beta2,state_bf16)


def run(state_bf16):
    rng=np.random.default_rng(42);rows,width=11,5;p0=bf16(rng.standard_normal((rows,width),dtype=np.float32))
    rank_records=[
        (np.array([7,2,7,4,2]),rng.standard_normal((5,width),dtype=np.float32)),
        (np.array([4,9,4,2,9,9]),rng.standard_normal((6,width),dtype=np.float32)),
        (np.array([2,2,7]),rng.standard_normal((3,width),dtype=np.float32)),
        (np.array([9,4,7,7]),rng.standard_normal((4,width),dtype=np.float32))]
    beta2,lr,eps,max_norm=0.91,0.03,1e-10,1.25
    dense_p=p0.copy();dense_v=np.zeros((rows,width),dtype=np.float32);lazy_p=p0.copy();lazy_v=dense_v.copy();last=np.zeros(rows,dtype=np.int64)
    history=[]
    for step in range(1,7):
        records=[(i,np.float32(v*np.float32(1.0+step/10.0))) for i,v in rank_records if not(step==3 and int(i[0])==7)]
        history.append(records);dense_p,dense_v=dense_reference(dense_p,dense_v,records,step,beta2,lr,eps,max_norm,state_bf16)
        lazy_step(lazy_p,lazy_v,last,records,step,beta2,lr,eps,max_norm,state_bf16)
        assert np.array_equal(lazy_p,dense_p),(step,float(np.max(np.abs(lazy_p-dense_p))))
        logical=logical_moment(lazy_v,last,step,beta2,state_bf16)
        if state_bf16:np.testing.assert_allclose(logical,dense_v,rtol=8e-3,atol=2e-5)
        else:assert np.array_equal(logical,dense_v)
    snapshot=(lazy_p.copy(),lazy_v.copy(),last.copy());replay=(p0.copy(),np.zeros_like(lazy_v),np.zeros_like(last))
    for step,records in enumerate(history,1):lazy_step(*replay,records,step,beta2,lr,eps,max_norm,state_bf16)
    assert all(np.array_equal(a,b) for a,b in zip(snapshot,replay))
    saved=io.BytesIO();np.savez(saved,model_table=snapshot[0],moment=snapshot[1],last_touch=snapshot[2]);saved.seek(0)
    restored=np.load(saved,allow_pickle=False);assert np.array_equal(restored["model_table"],snapshot[0])
    indices=np.array([0,2,4,7,9]);assert np.array_equal(restored["model_table"][indices],snapshot[0][indices])


if __name__=="__main__":
    run(False);run(True)
    print("touched-row lazy RMSProp CPU reference: PASS")
