"""Independent tiny-only replay into reviewer-owned outputs, never host work."""
import hashlib
import importlib.util
import os
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = Path('/home/poojith-devan/Desktop/Trainium-architecture-initial-cache-refresh-20260925/workflow/proposals/research-v2/train.py')
EXPECTED = '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
    raise ValueError('exact architecture candidate required')
os.environ['CACHE_TEST_VARIANT'] = 'architecture'
os.environ['CACHE_TEST_SOURCE'] = str(SOURCE)
script = ROOT / 'research/v2/lanes/initial-cache-refresh/test_initial_cache.py'
spec = importlib.util.spec_from_file_location('cache_review_tiny', script)
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
test.RECEIPTS = HERE / 'independent-tiny'
if test.RECEIPTS.exists():
    raise ValueError('review output must be new')
test.run_tiny()
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test))
raise SystemExit(0 if result.wasSuccessful() else 1)
