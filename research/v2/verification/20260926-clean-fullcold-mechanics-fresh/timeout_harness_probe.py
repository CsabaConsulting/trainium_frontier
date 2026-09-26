"""Run the exact harness's Popen node, with an explicitly harmless CPU command."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import time

directory = Path(__file__).resolve().parent
tree = ast.parse(Path('/home/poojith-devan/clean-fullcold-review-DrcoWi/submission_rehearsal.py').read_text())
node = next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
    and n.value.func.attr=='Popen')
command=['/usr/bin/python3','-I','-B',str(directory/'timeout_child_probe.py'),sys.argv[1],sys.argv[2]]
sandbox=directory
env={'PATH':'/usr/bin:/bin'}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
    'actual_harness_Popen_statement.py','exec',optimize=0))
while process.poll() is None:
    time.sleep(0.05)
