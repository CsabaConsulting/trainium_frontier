#!/usr/bin/env python3
"""Source-line attribution + NEFF role mapping for a neuron-explorer parquet profile."""
import sys, os, re, json
import numpy as np
import pandas as pd

NE = "/home/ubuntu/ne-data/profiles/global"
name = sys.argv[1]
cands = [d for d in os.listdir(NE) if d.startswith(name + "_i-") or d == name + "@latest"]
# prefer the long form (full ingest)
cands.sort(key=len, reverse=True)
P = os.path.join(NE, cands[0])
print("PROFILE_DIR", P)

md = pd.read_parquet(f"{P}/Metadata.parquet").iloc[0]
inst = pd.read_parquet(f"{P}/Instruction.parquet")
act = pd.read_parquet(f"{P}/ActiveTime.parquet")
try:
    ei = pd.read_parquet(f"{P}/ExecutionInfo.parquet")
except Exception:
    ei = None

t0 = 0
t1 = int(md["last_hw_timestamp"])
total = t1 - t0
print("TOTAL_NS", total, "TOTAL_MS", round(total/1e6, 3))


def interval_merge_ns(starts, ends, a, b):
    if len(starts) == 0:
        return 0
    s = np.maximum(np.asarray(starts, dtype=np.int64), a)
    e = np.minimum(np.asarray(ends, dtype=np.int64), b)
    v = s < e
    s, e = s[v], e[v]
    if len(s) == 0:
        return 0
    o = np.argsort(s)
    s, e = s[o], e[o]
    tot = 0
    cur = -1
    for i in range(len(s)):
        if s[i] >= cur:
            tot += e[i] - s[i]
            cur = e[i]
        elif e[i] > cur:
            tot += e[i] - cur
            cur = e[i]
    return int(tot)


# ---------- TensorE busy intervals -> build "TE idle" complement for exposure ----------
te = act[act["engine"] == "tensor"]
te_s = np.sort(te["start_ts"].values.astype(np.int64))
order = np.argsort(te["start_ts"].values.astype(np.int64))
te_e = te["end_ts"].values.astype(np.int64)[order]
# merge TE busy
mb_s, mb_e = [], []
for s, e in zip(te_s, te_e):
    if mb_s and s <= mb_e[-1]:
        if e > mb_e[-1]:
            mb_e[-1] = e
    else:
        mb_s.append(s); mb_e.append(e)
mb_s = np.array(mb_s, dtype=np.int64); mb_e = np.array(mb_e, dtype=np.int64)
te_busy = int((mb_e - mb_s).sum())
print("TE_BUSY_MS", round(te_busy/1e6, 3), "TE_BUSY_PCT", round(100*te_busy/total, 2))

# TE idle intervals (complement within [t0,t1])
idle_s, idle_e = [], []
prev = t0
for s, e in zip(mb_s, mb_e):
    if s > prev:
        idle_s.append(prev); idle_e.append(s)
    prev = max(prev, e)
if prev < t1:
    idle_s.append(prev); idle_e.append(t1)
idle_s = np.array(idle_s, dtype=np.int64); idle_e = np.array(idle_e, dtype=np.int64)
print("TE_IDLE_MS", round(int((idle_e-idle_s).sum())/1e6, 3), "N_IDLE_INTERVALS", len(idle_s))


def exposed_ns(starts, ends):
    """ns of these intervals that overlaps TE-idle windows (interval-merged)."""
    if len(starts) == 0:
        return 0
    # clip each instruction interval against idle intervals; accumulate merged coverage
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    res_s, res_e = [], []
    j_idx = np.searchsorted(idle_s, s, side="right") - 1
    for k in range(len(s)):
        j = max(j_idx[k], 0)
        while j < len(idle_s) and idle_s[j] < e[k]:
            a = max(s[k], idle_s[j]); b = min(e[k], idle_e[j])
            if a < b:
                res_s.append(a); res_e.append(b)
            j += 1
    return interval_merge_ns(res_s, res_e, t0, t1)


# ---------- window instructions ----------
w = inst[(inst["start_ts"] >= t0) & (inst["start_ts"] < t1)].copy()
w["end_ts_c"] = w["start_ts"] + w["duration_ns"]
print("N_INST", len(w))
srccol = "bir_debug_info_source_location"
w[srccol] = w[srccol].fillna("<none>")
print("ATTRIBUTED_PCT", round(100*(w[srccol] != "<none>").mean(), 2))


def norm_src(x):
    if x == "<none>":
        return x
    # keep last file:line-ish token
    return x


w["src"] = w[srccol]

# ---------- per-source aggregation ----------
rows = []
for src, g in w.groupby("src"):
    dur = int(g["duration_ns"].sum())
    wall = interval_merge_ns(g["start_ts"].values, g["end_ts_c"].values, t0, t1)
    exp = exposed_ns(g["start_ts"].values, g["end_ts_c"].values)
    rows.append(dict(src=src, n=len(g), dur_ms=dur/1e6, wall_ms=wall/1e6,
                     exposed_ms=exp/1e6,
                     engines=",".join(sorted(g["engine"].astype(str).unique())[:5])))
agg = pd.DataFrame(rows).sort_values("wall_ms", ascending=False)
agg["wall_pct"] = 100*agg["wall_ms"]/(total/1e6)
agg["exposed_pct"] = 100*agg["exposed_ms"]/(total/1e6)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 110)
print("\n===== TOP 40 SOURCE LOCATIONS by wall (interval-merged, any engine) =====")
print(agg.head(40).to_string(index=False))
print("\n===== TOP 25 SOURCE LOCATIONS by TE-EXPOSED time =====")
print(agg.sort_values("exposed_ms", ascending=False).head(25).to_string(index=False))

# ---------- tensor-engine-only per-source ----------
tw = w[w["engine"] == "Tensor"]
rows = []
for src, g in tw.groupby("src"):
    rows.append(dict(src=src, n=len(g), te_dur_ms=int(g["duration_ns"].sum())/1e6,
                     flops=g["adjusted_flops"].sum() if "adjusted_flops" in g else np.nan))
tagg = pd.DataFrame(rows).sort_values("te_dur_ms", ascending=False)
tagg["te_pct_of_te_busy"] = 100*tagg["te_dur_ms"]/(te_busy/1e6)
print("\n===== TOP 25 SOURCE LOCATIONS on TENSOR engine =====")
print(tagg.head(25).to_string(index=False))

# ---------- keyword probes ----------
print("\n===== KEYWORD PROBES (substring match on source location) =====")
probes = ["local_conv", "CausalConv", "conv", "F.linear", "linear", "rope", "norm",
          "softcap", "cross_entropy", "attention", "scaled_dot", "relu", "embed",
          "2658", "2637", "1750", "1751", "341", "332"]
for p in probes:
    m = agg[agg["src"].str.contains(p, case=False, na=False, regex=False)]
    if len(m):
        print(f"  {p:16s} n_srcs={len(m):3d} wall={m.wall_ms.sum():9.3f}ms "
              f"({100*m.wall_ms.sum()/(total/1e6):5.2f}%) exposed={m.exposed_ms.sum():8.3f}ms "
              f"({100*m.exposed_ms.sum()/(total/1e6):5.2f}%) inst={int(m.n.sum())}")
    else:
        print(f"  {p:16s} NO MATCH")

# ---------- hlo_name / layer breakdown for the unattributed ----------
for col in ["hlo_name", "layer", "opcode", "framework_op_name", "nki_source_location"]:
    if col in w.columns:
        na = w[w["src"] == "<none>"]
        if len(na):
            top = na.groupby(na[col].fillna("<null>"))["duration_ns"].agg(["count", "sum"])
            top["ms"] = top["sum"]/1e6
            top = top.sort_values("ms", ascending=False).head(15)
            print(f"\n===== UNATTRIBUTED rows grouped by {col} (top 15) =====")
            print(top[["count", "ms"]].to_string())

# ---------- NEFF role map ----------
if ei is not None:
    print("\n===== ExecutionInfo columns =====")
    print(list(ei.columns))
    e = ei.copy()
    if "start_ts" in e.columns:
        e = e[(e["start_ts"] >= t0) & (e["start_ts"] < t1)]
    print("N_EXEC", len(e))
    print(e.head(80).to_string())
