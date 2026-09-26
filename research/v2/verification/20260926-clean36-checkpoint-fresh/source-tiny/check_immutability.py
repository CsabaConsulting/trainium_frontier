import importlib.util
import json
from pathlib import Path
import sys
base=Path(__file__).parent
s=importlib.util.spec_from_file_location('candidate',base/'candidate.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
pin={'sha256':'45470f257bbc1ca7e2baed4762b644739490b6c81a9341dce42fc699a4cb7618','bytes':22}
try:
    h.authenticate_files({str(base/'immutability-fixture.txt'):pin})
except h.CheckFailure as e:
    if sys.argv[1]!='post' or 'FILE_IDENTITY' not in str(e):raise
    print(json.dumps({'phase':'post','status':'MUTATION_REJECTED','message':str(e)}))
else:
    if sys.argv[1]!='pre':raise RuntimeError('mutation accepted')
    print(json.dumps({'phase':'pre','status':'AUTHENTICATED'}))
