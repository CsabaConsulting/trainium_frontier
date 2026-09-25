"""Reuse the reviewed, explicit non-secret session-archive operation."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "archive_operation", HERE.parent / "20260926-fresh-astra-k2-namespace/archive_session.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HERE = HERE

if __name__ == "__main__":
    module.main()
