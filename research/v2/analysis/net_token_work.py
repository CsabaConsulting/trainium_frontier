"""Human-requested work estimate; not evaluator, BPB selector or launch authority."""
import argparse
import json
import math


def net_work(cold_startup_seconds, steady_tokens_per_second):
    if not math.isfinite(cold_startup_seconds) or cold_startup_seconds < 0:
        raise ValueError("Actual cold startup must be finite and nonnegative")
    if not math.isfinite(steady_tokens_per_second) or steady_tokens_per_second <= 0:
        raise ValueError("Actual steady tokens/s must be finite and positive")
    charge = max(0.0, cold_startup_seconds - 900.0)
    available = max(0.0, 1800.0 - charge)
    return {"cold_startup_seconds": cold_startup_seconds,
            "charged_startup_excess_seconds": charge,
            "available_training_seconds_for_work_estimate": available,
            "steady_tokens_per_second": steady_tokens_per_second,
            "net_tokens": available * steady_tokens_per_second}


def compare(candidate_startup, candidate_rate, base_startup, base_rate):
    candidate, base = net_work(candidate_startup, candidate_rate), net_work(base_startup, base_rate)
    if base["net_tokens"] <= 0:
        raise ValueError("The comparison base must have positive net work")
    ratio = candidate["net_tokens"] / base["net_tokens"]
    return {"candidate": candidate, "base": base, "net_token_ratio": ratio,
            "net_token_delta_percent": (ratio - 1) * 100,
            "human_net_threshold_passed": candidate["net_tokens"] >= base["net_tokens"] * 1.03,
            "scope": "Work estimate only; cold provenance, exact timing, correctness, official total-budget, BPB and submission checks remain separate"}


def test():
    assert net_work(859, 100)["net_tokens"] == 180000
    assert net_work(900, 100)["charged_startup_excess_seconds"] == 0
    assert net_work(1000, 100)["net_tokens"] == 170000
    assert net_work(2700, 100)["net_tokens"] == 0
    assert net_work(3000, 100)["net_tokens"] == 0
    assert compare(859, 103, 859, 100)["human_net_threshold_passed"]
    assert not compare(1000, 103, 859, 100)["human_net_threshold_passed"]
    assert not compare(859, 102.9, 859, 100)["human_net_threshold_passed"]
    assert compare(1000, 110, 859, 100)["human_net_threshold_passed"]
    for startup, rate in [(-1,100),(math.nan,100),(math.inf,100),(859,0),(859,-1),(859,math.nan),(859,math.inf)]:
        try:
            net_work(startup, rate)
        except ValueError:
            continue
        raise AssertionError((startup,rate))
    print("PASS: 9 work/threshold cases and 7 invalid-input cases; no native result implied")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    for name in ("candidate-startup", "candidate-rate", "base-startup", "base-rate"):
        parser.add_argument("--" + name, type=float)
    args = parser.parse_args()
    if args.self_test:
        test()
    else:
        values = [args.candidate_startup,args.candidate_rate,args.base_startup,args.base_rate]
        if any(x is None for x in values):
            parser.error("All four actual measurements are required; unknown is not zero")
        print(json.dumps(compare(*values), indent=2, allow_nan=False))
