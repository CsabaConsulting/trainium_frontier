#!/usr/bin/env python3
"""TEST-ONLY stand-in for `prepare.py eval-public`: prints a val_bpb line (FAKE_BPB)."""
import os
print(f"val_bpb: {float(os.environ.get('FAKE_BPB', '0.9920'))}")
