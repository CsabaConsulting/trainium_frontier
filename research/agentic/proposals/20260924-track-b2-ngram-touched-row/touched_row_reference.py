#!/usr/bin/env python3
"""Sparse CPU oracle for Track B2; it makes no native-Neuron rounding claim."""
import hashlib
import io
import pathlib
import numpy as np

ROWS=128*8192
DIM=128
TABLES=("b1","t1","b5","t5")
ROOT=pathlib.Path(__file__).resolve().parents[4]
SOURCE=ROOT/"submissions/dense6x1024_public0992851/train.py"
PREPARE=ROOT/"prepare.py"


def bf16(value):
    x=np.asarray(value,dtype=np.float32);bits=x.view(np.uint32)
    rounded=bits+np.uint32(0x7FFF)+((bits>>np.uint32(16))&np.uint32(1))
    return (rounded&np.uint32(0xFFFF0000)).view(np.float32)


def validate(indices,values):
    if not isinstance(indices,np.ndarray) or indices.dtype!=np.int64 or indices.ndim!=1:raise ValueError("indices must be rank-1 int64")
    if np.any(indices<0) or np.any(indices>=ROWS):raise ValueError("row out of range")
    if not isinstance(values,np.ndarray) or values.dtype!=np.float32 or values.shape!=(len(indices),DIM):raise ValueError("values must be [N,128] float32")


def add(a,b,bf16_rounding):
    value=np.float32(a+b)
    return bf16(value) if bf16_rounding else value


def coalesce(indices,values,bf16_rounding):
    validate(indices,values);out={}
    for row,value in zip(indices.tolist(),values):out[row]=add(out.get(row,np.zeros(DIM,dtype=np.float32)),value,bf16_rounding)
    return dict(sorted(out.items()))


def rank_clip(records,max_norm,bf16_rounding):
    """One clip coefficient jointly covers all four table gradients on this rank."""
    if set(records)!=set(TABLES):raise ValueError("exactly four table records required")
    compact={key:coalesce(*records[key],bf16_rounding) for key in TABLES}
    sumsq=np.float32(0)
    for key in TABLES:
        for value in compact[key].values():sumsq=np.float32(sumsq+np.sum(value*value,dtype=np.float32))
    coefficient=np.float32(min(1.0,max_norm/(float(np.sqrt(sumsq))+1e-6)))
    for key in TABLES:
        for row in compact[key]:
            value=np.float32(compact[key][row]*coefficient)
            compact[key][row]=bf16(value) if bf16_rounding else value
    return compact,coefficient


def global_avg(rank_records,max_norm,bf16_rounding):
    clipped=[];coefficients=[]
    for records in rank_records:
        value,coefficient=rank_clip(records,max_norm,bf16_rounding);clipped.append(value);coefficients.append(coefficient)
    merged={key:{} for key in TABLES}
    for rank in clipped:
        for key in TABLES:
            for row,value in rank[key].items():merged[key][row]=add(merged[key].get(row,np.zeros(DIM,dtype=np.float32)),value,bf16_rounding)
    for key in TABLES:
        for row in merged[key]:
            value=np.float32(merged[key][row]/np.float32(len(clipped)))
            merged[key][row]=bf16(value) if bf16_rounding else value
    return merged,coefficients


def empty_state():
    return {"param":{key:{} for key in TABLES},"moment":{key:{} for key in TABLES},"last":{key:{} for key in TABLES},"step":0}


def initial_row(key,row):
    # Actual control initializes every n-gram row to zero; sparse storage avoids a 2 GiB fixture.
    return np.zeros(DIM,dtype=np.float32)


def dense_step(state,grads,beta2,lr,eps,state_bf16):
    state["step"]+=1;step=state["step"]
    for key in TABLES:
        known=set(state["moment"][key])|set(grads[key])
        for row in sorted(known):
            old=state["moment"][key].get(row,np.zeros(DIM,dtype=np.float32))
            grad=grads[key].get(row,np.zeros(DIM,dtype=np.float32))
            value=np.float32(old*np.float32(beta2)+grad*grad*np.float32(1-beta2));value=bf16(value) if state_bf16 else value
            state["moment"][key][row]=value
            if row in grads[key]:
                param=state["param"][key].get(row,initial_row(key,row));update=grad/(np.sqrt(value/np.float32(1-beta2**step))+np.float32(eps))
                state["param"][key][row]=bf16(param-np.float32(lr)*update)


def lazy_step(state,grads,beta2,lr,eps,state_bf16):
    state["step"]+=1;step=state["step"]
    for key in TABLES:
        untouched=set(state["moment"][key])-set(grads[key]);before={row:state["moment"][key][row].copy() for row in untouched}
        for row,grad in grads[key].items():
            value=state["moment"][key].get(row,np.zeros(DIM,dtype=np.float32)).copy();previous=state["last"][key].get(row,0);delta=step-previous
            # Zero-gradient intervening steps round state; current step combines decay+new square before its state write.
            for age in range(1,delta+1):
                value=np.float32(value*np.float32(beta2))
                if state_bf16 and age<delta:value=bf16(value)
            value=np.float32(value+grad*grad*np.float32(1-beta2));value=bf16(value) if state_bf16 else value
            param=state["param"][key].get(row,initial_row(key,row));update=grad/(np.sqrt(value/np.float32(1-beta2**step))+np.float32(eps))
            state["param"][key][row]=bf16(param-np.float32(lr)*update);state["moment"][key][row]=value;state["last"][key][row]=step
        assert all(np.array_equal(state["moment"][key][row],value) for row,value in before.items())


def logical_moment(state,key,row,beta2,state_bf16):
    value=state["moment"][key].get(row,np.zeros(DIM,dtype=np.float32)).copy();delta=state["step"]-state["last"][key].get(row,0)
    for _ in range(delta):
        value=np.float32(value*np.float32(beta2));value=bf16(value) if state_bf16 else value
    return value


def state_equal(dense,lazy,beta2,state_bf16):
    assert dense["step"]==lazy["step"]
    for key in TABLES:
        assert set(dense["param"][key])==set(lazy["param"][key])
        for row in dense["param"][key]:assert np.array_equal(dense["param"][key][row],lazy["param"][key][row])
        for row,value in dense["moment"][key].items():
            logical=logical_moment(lazy,key,row,beta2,state_bf16)
            if state_bf16:np.testing.assert_allclose(logical,value,rtol=8e-3,atol=2e-5)
            else:assert np.array_equal(logical,value)


def physical_equal(left,right):
    assert left["step"]==right["step"]
    for section in ("param","moment","last"):
        for key in TABLES:
            assert set(left[section][key])==set(right[section][key])
            for row in left[section][key]:
                a,b=left[section][key][row],right[section][key][row]
                assert a==b if section=="last" else np.array_equal(a,b)


def serialize(state):
    payload={"step":np.array([state["step"]],dtype=np.int64)}
    for section in ("param","moment","last"):
        for key in TABLES:
            rows=np.array(sorted(state[section][key]),dtype=np.int64);payload[f"{section}_{key}_rows"]=rows
            if section=="last":payload[f"{section}_{key}_values"]=np.array([state[section][key][row] for row in rows],dtype=np.int64)
            else:payload[f"{section}_{key}_values"]=np.stack([state[section][key][row] for row in rows]) if len(rows) else np.empty((0,DIM),dtype=np.float32)
    output=io.BytesIO();np.savez(output,**payload);return output.getvalue()


def reload(data):
    values=np.load(io.BytesIO(data),allow_pickle=False);state=empty_state();state["step"]=int(values["step"][0])
    for section in ("param","moment","last"):
        for key in TABLES:
            rows=values[f"{section}_{key}_rows"];items=values[f"{section}_{key}_values"]
            state[section][key]={int(row):(int(value) if section=="last" else value.copy()) for row,value in zip(rows,items)}
    return state


def records(rng,step):
    result=[]
    for rank in range(4):
        arm={}
        for table_index,key in enumerate(TABLES):
            base=(rank*100003+table_index*20011+step*997)%ROWS
            indices=np.array([base,(base+17)%ROWS,base,(base+rank+31)%ROWS],dtype=np.int64)
            arm[key]=(indices,rng.standard_normal((len(indices),DIM),dtype=np.float32))
        result.append(arm)
    return result


def source_contract():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=="35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2"
    assert hashlib.sha256(PREPARE.read_bytes()).hexdigest()=="5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb"
    text=SOURCE.read_text();assert "'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}" in text
    assert "model.load_state_dict(state, strict=True)" in text and "self.ngram_embeds = nn.ModuleDict(tables)" in text
    expected={f"ngram_embeds.{key}.weight" for key in TABLES};checkpoint_keys={f"ngram_embeds.{key}.weight" for key in TABLES}
    assert checkpoint_keys==expected
    table={key:bf16(np.arange(DIM,dtype=np.float32)[None,:]) for key in TABLES};loaded={key:value.copy() for key,value in table.items()}
    assert all(np.array_equal(loaded[key],table[key]) for key in TABLES) # eval reads identical model-table bytes; optimizer metadata is separate.


def invalid_inputs():
    good=np.zeros((1,DIM),dtype=np.float32)
    bad=[(np.array([-1],dtype=np.int64),good),(np.array([ROWS],dtype=np.int64),good),(np.array([1],dtype=np.int32),good),(np.array([[1]],dtype=np.int64),good),(np.array([1],dtype=np.int64),np.zeros((1,DIM-1),dtype=np.float32)),(np.array([1],dtype=np.int64),np.zeros((1,DIM),dtype=np.float64))]
    for indices,values in bad:
        try:coalesce(indices,values,False)
        except ValueError:pass
        else:raise AssertionError("invalid sparse record accepted")


def run(state_bf16,bf16_collective):
    rng=np.random.default_rng(42);beta2,lr,eps,max_norm=.91,.03,1e-10,2.0;dense=empty_state();lazy=empty_state();history=[]
    for step in range(1,5):
        batch=records(rng,step);history.append(batch);grads,coefficients=global_avg(batch,max_norm,bf16_collective)
        assert len(coefficients)==4 and all(0<value<=1 for value in coefficients);dense_step(dense,grads,beta2,lr,eps,state_bf16);lazy_step(lazy,grads,beta2,lr,eps,state_bf16);state_equal(dense,lazy,beta2,state_bf16)
        if step==3:resumed=reload(serialize(lazy))
    # Resume must produce the same NEXT update, not merely reload equal bytes.
    replay=reload(serialize(resumed));grads,_=global_avg(history[3],max_norm,bf16_collective);lazy_step(replay,grads,beta2,lr,eps,state_bf16)
    physical_equal(replay,lazy)


if __name__=="__main__":
    source_contract();invalid_inputs()
    for state_bf16 in (False,True):
        for bf16_collective in (False,True):run(state_bf16,bf16_collective)
    print("four-table touched-row lazy RMSProp CPU oracle: PASS (native BF16 ordering remains gated)")
