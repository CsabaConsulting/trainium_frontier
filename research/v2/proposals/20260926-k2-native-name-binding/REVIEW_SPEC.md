# Neutral review specification: K2 consumer namespace successor

Review the one-line outer-scope change in the attached complete kernel diff.
The original experimental module is disabled unless explicitly enabled and
has no production import. Its contract is fixed V256/C128/T256, supplied
synthetic stable metadata, BF16 or FP32 input and token-ordered FP32 sums.

Permitted change: expose the imported `nl` and `nisa` aliases in module globals
through a declaration inside the existing factory. The nested kernel body,
factory signature/default, decorator, lazy-import behavior, checker API,
arithmetic/dtypes, all original cases and exact-byte thresholds must remain
unchanged. No production integration or external execution is permitted.

The original four native rank receipts and driver terminal are supplied as
raw failure evidence, including the original source/checker/reference pins.
They do not establish the installed frontend's exact namespace semantics.
The successor must not erase their failure or claim native acceptance.

Use your own bounded CPU/stdlib test to assess the scope/default-disabled
contract and any risks. No NKI/Torch installation, full model, training data,
remote host access or native launch. Report source compatibility separately
from unmeasured native compile/exactness/performance. A review pass is not
source adoption or native launch permission.
