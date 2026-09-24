#!/usr/bin/env python3
import importlib.util,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("host",ROOT/"research/fresh_start_host.py");host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
floor=13*1024**3//2
assert host.MIN_OPERATIONAL_FREE_BYTES==floor
assert not host.sufficient_free_space(floor-1)
assert host.sufficient_free_space(floor)
launch=(ROOT/"research/fresh_start_launch.py").read_text()
assert "assert sufficient_free_space(free)" in launch and "preflight(r, m['label'])" in launch
assert launch.index("free = shutil.disk_usage(r).free") < launch.index("preflight(r, m['label'])")
assert "8 * 1024**3" not in launch and "less than 8 GiB" not in launch
assert "8 * 1024**3" not in (ROOT/"research/fresh_start_host.py").read_text()
import json
proposal=json.loads((ROOT/"research/submission-validation/20260924-current-base-dual-control/EXPERIMENT_PROPOSAL.json").read_text())
assert proposal["operational_disk_floor_gib"]==6.5 and proposal["hard_abort_disk_floor_gib"]==5.0
assert int(proposal["operational_disk_floor_gib"]*1024**3)==floor
protocol=(ROOT/"research/experiment-protocol.md").read_text();assert "6.5GiB" in protocol and "below5GiB" in protocol
print("6.5 GiB operational floor boundaries/recheck: PASS")
