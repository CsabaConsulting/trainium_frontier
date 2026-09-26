# Narrow repair after independent false-acceptance finding

Original packetd24465fb/leaf52eab7d8 and every receipt remain in history. Independent review found that an empty byte payload could be labeled FP32[256,128], allowing consistent empty weight/forward records through validation. The previous self-check results did not cover this failure and do not override the failed independent gate.

Leaf634a789c adds validation at the start of blob(), before any bytes enter the receipt map: supported dtype exactly FP32 or int32, a nonempty list/tuple of strictly positive integer dimensions (booleans excluded), and exact four-byte element count. All tensors in this fixed diagnostic are nonempty. Validator calls reconstructing expected weight/output and actual tensor metadata share this rejection. Scientific fixtures, IEEE754 arithmetic, ordered routes, compiler configuration, tolerances and native lifecycle are unchanged.

Own new CPU33 tests include the reported all-consistent empty weight/forward forgery, empty/truncated/overlong data, int32 truncation, unsupported dtype and malformed dimensions; normal/-O outputs are cpu-byte-repair-*.json. Rebound packet41 and aggregation11 checks have separate packet-byte-repair-*.json and aggregation-byte-repair-*.json receipts. These remain self-checks on local Torch2.13, not independent or native acceptance. Fresh review reassessment is required; all host/staging/launch/adoption flags remain false.

Use REVIEW_BYTE_REPAIR_INPUTS.json and the exact repair diff/spec/raw for reassessment; this note is not reviewer context. No host, network, scientific source, ledger or state action was taken.
