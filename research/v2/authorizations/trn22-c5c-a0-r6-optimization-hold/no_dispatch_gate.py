#!/usr/bin/env python3
"""Offline r6 HOLD sentinel, not a native launcher or remote dispatcher.

The frozen 446b launcher has assertion-based local and remote safety checks.
This executable intentionally has no path to SSH, receipt creation, or guard
spawn. A future source change must install equivalent non-assert gates in the
actual local and remote launcher before an enabled request is considered.
"""

import argparse
import json
import os
from pathlib import Path
import sys


class DispatchHold(RuntimeError):
    """A mechanical refusal that must survive Python optimization."""


def require_unoptimized_python(stage):
    """Fail closed in either local or remote interpreter context."""
    if stage not in ("local", "remote"):
        raise DispatchHold("R6_UNRECOGNIZED_LAUNCH_STAGE")
    if sys.flags.optimize != 0:
        raise DispatchHold("R6_OPTIMIZED_PYTHON_REFUSED")
    if "PYTHONOPTIMIZE" in os.environ:
        raise DispatchHold("R6_PYTHONOPTIMIZE_ENV_REFUSED")


def refuse_unrepaired_source(stage, request_path=None):
    require_unoptimized_python(stage)
    if stage == "local":
        if request_path is None:
            raise DispatchHold("R6_REQUEST_REQUIRED")
        request = json.loads(Path(request_path).read_text())
        if request.get("dispatch_authorized") is not False:
            raise DispatchHold("R6_REQUEST_NOT_DISABLED")
    raise DispatchHold("R6_FROZEN_446B_LAUNCHER_UNSAFE_HOLD")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("local", "remote"))
    parser.add_argument("request", nargs="?", type=Path)
    args = parser.parse_args(argv)
    refuse_unrepaired_source(args.stage, args.request)


if __name__ == "__main__":
    main()
