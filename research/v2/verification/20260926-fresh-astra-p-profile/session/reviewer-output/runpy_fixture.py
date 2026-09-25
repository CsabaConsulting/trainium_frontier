import sys
import torch
OBSERVED = (list(sys.argv), sys.path[0], __name__)
torch.manual_seed(42)
VALUE = torch.randn(2, 8)
if "--raise" in sys.argv:
    raise RuntimeError("fixture failure")
