# Minimal CPU stub of the organiser prepare.py: only the names train.py imports.
# Used solely to import the candidate module for CPU reference tests. No data, no scoring.
EVAL_TIMEOUT_SECONDS = 3600
TOKENIZER_VOCAB_SIZE = 8192
TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS = 900
TRAIN_STARTUP_STEPS_EXCLUDED = 5
TRAIN_TIME_BUDGET_SECONDS = 1800
class Tokenizer: pass
def _document_batches(*a, **k): raise RuntimeError('stub')
def _row_groups_for_split(*a, **k): raise RuntimeError('stub')
def causality_check(*a, **k): raise RuntimeError('stub')
def ensure_tokenizer(*a, **k): raise RuntimeError('stub')
def evaluate_bpb(*a, **k): raise RuntimeError('stub')
def get_dist_info(): return (False, 0, 0, 1)
def get_token_bytes(*a, **k): raise RuntimeError('stub')
def make_dataloader(*a, **k): raise RuntimeError('stub')
def print0(*a, **k): print(*a, **k)
def record_step_boundary(step): pass
