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
print("6.5 GiB operational floor boundaries/recheck: PASS")
