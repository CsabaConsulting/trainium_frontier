"""Pinned remote one-attempt adapter; transported as python3 -c source, never staged.

The adapter refuses the disabled draft request. A future, separately reviewed
request and Supervisor assignment must change its authorization state and hash.
"""

import hashlib
import json
import sys
import traceback


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("exact request/source/bundle SHA-256 arguments required")
    expected_request, expected_source, expected_bundle = sys.argv[1:]
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict) or set(payload) != {
                "request_text", "stage_source", "persistence_bundle"}:
            raise RuntimeError("incomplete exact checkout payload")
        request_bytes = payload["request_text"].encode("utf-8")
        source_bytes = payload["stage_source"].encode("utf-8")
        bundle_bytes = json.dumps(payload["persistence_bundle"], sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")
        if (digest(request_bytes) != expected_request or
                digest(source_bytes) != expected_source or
                digest(bundle_bytes) != expected_bundle):
            raise RuntimeError("exact request, extraction code or persistence bundle drift")
        request = json.loads(request_bytes)
        if (request.get("schema") != "trn21-c5c-a0-checkout-only-request-v1" or
                request.get("checkout_authorized") is not True or
                request.get("operator_one_attempt_only") is not True or
                request.get("extraction_code_sha256") != expected_source or
                request.get("persistence_bundle_sha256") != expected_bundle):
            raise RuntimeError("checkout request is disabled or disagrees with exact inputs")
        namespace = {"__name__": "reviewed_stage_r2_checkout",
                     "__file__": "<sha256-pinned-stage-r2-checkout>"}
        exec(compile(source_bytes, namespace["__file__"], "exec"), namespace)
        receipt = namespace["stage_fixed_checkout"](payload["persistence_bundle"])
        if (receipt.get("status") != "SOURCE_CHECKOUT_VERIFIED_FOR_INDEPENDENT_REVIEW_ONLY" or
                receipt.get("cpu_g2_authorized") is not False or
                receipt.get("native_training_authorized") is not False or
                receipt.get("evaluation_authorized") is not False):
            raise RuntimeError("checkout implementation returned an invalid stage receipt")
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except BaseException as error:
        failure = {
            "schema": "trn21-c5c-a0-checkout-attempt-failure-v1",
            "status": "HOLD_PRESERVE_PARTIAL_NO_RETRY",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "traceback_tail": traceback.format_exc(limit=5)[-3000:],
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
