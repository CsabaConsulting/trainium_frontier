if step + 1 == TRAIN_STARTUP_STEPS_EXCLUDED:
    now = time.monotonic()
    observed_startup = now - _PROCESS_STARTED
    startup_allowance_est = min(observed_startup, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
    excess = max(0.0, observed_startup + STARTUP_LAUNCH_MARGIN_SECONDS - TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS)
    budget_started = now
    recent_step_times.clear()
