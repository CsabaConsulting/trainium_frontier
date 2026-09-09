"""Launch-count / dispatch-gap comparison between two ingested neuron-explorer profiles.

Reads ExecutionInfo (one row per NEFF execution) and reports, for each profile:
  * N_EXEC          -- number of NEFF launches in the window (the "launch count")
  * N_DISTINCT_NEFF -- number of separate compiled NEFFs involved
  * busy             -- union of [exec_start, exec_end] intervals (interval-merged, so
                        overlapping executions are not double counted)
  * gap              -- window time with NO execution in flight = pure dispatch overhead
This is the metric the optimizer-dispatch change is supposed to move.
"""
import os
import sys

import pandas as pd

NE = "/home/ubuntu/ne-data/profiles/global"


def load(name):
    cands = [d for d in os.listdir(NE) if d.startswith(name + "_i-") or d == name + "@latest"]
    cands.sort(key=len, reverse=True)
    P = os.path.join(NE, cands[0])
    ei = pd.read_parquet(f"{P}/ExecutionInfo.parquet")
    md = pd.read_parquet(f"{P}/Metadata.parquet").iloc[0]
    return P, ei, md


def merge(iv):
    """Union of closed intervals -> total covered length."""
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


rows = []
for name in sys.argv[1:]:
    P, ei, md = load(name)
    total_ns = int(md["last_hw_timestamp"])
    iv = list(zip(ei["execution_start_ts"].astype("int64"), ei["execution_end_ts"].astype("int64")))
    busy = merge(iv)
    span = max(e for _, e in iv) - min(s for s, _ in iv)
    rows.append(dict(
        profile=name,
        n_exec=len(ei),
        n_distinct_neff=ei["neff_name"].nunique(),
        window_ms=total_ns / 1e6,
        exec_span_ms=span / 1e6,
        busy_ms=busy / 1e6,
        gap_ms=(span - busy) / 1e6,
        gap_pct_of_span=100.0 * (span - busy) / span,
        mean_exec_us=(sum(e - s for s, e in iv) / len(iv)) / 1e3,
    ))

df = pd.DataFrame(rows).set_index("profile")
pd.set_option("display.width", 200)
print(df.T.to_string(float_format=lambda v: f"{v:,.2f}"))

if len(rows) == 2:
    a, b = rows[0], rows[1]
    print(f"\ndelta ({b['profile']} vs {a['profile']}):")
    for k in ("n_exec", "n_distinct_neff", "exec_span_ms", "busy_ms", "gap_ms", "gap_pct_of_span"):
        d = b[k] - a[k]
        pct = 100.0 * d / a[k] if a[k] else float("nan")
        print(f"  {k:20s} {a[k]:>12,.2f} -> {b[k]:>12,.2f}   ({d:+,.2f}, {pct:+.1f}%)")
