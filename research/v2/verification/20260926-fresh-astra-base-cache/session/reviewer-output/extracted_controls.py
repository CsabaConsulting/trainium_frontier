parser.add_argument('--initial-cache-refresh', action=argparse.BooleanOptionalAction, default=False, help='Refresh differentiable weight casts inside the timed first step before its first forward; default OFF.')

if step == 0 and args.initial_cache_refresh:
    for m in cached_linears:
        m.refresh_weight_cache(COMPUTE_DTYPE)

for m in cached_linears:
    m.refresh_weight_cache(COMPUTE_DTYPE)
