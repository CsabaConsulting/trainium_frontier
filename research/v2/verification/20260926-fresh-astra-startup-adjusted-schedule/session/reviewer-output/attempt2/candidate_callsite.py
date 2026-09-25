schedule_elapsed = startup_adjusted_schedule_elapsed(step, elapsed, budget_started) if args.startup_adjusted_schedule else elapsed
lrm = lr_multiplier(step, elapsed=schedule_elapsed, max_train_seconds=args.max_train_seconds)
lrm_flat = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
wd_progress = min(1.0, schedule_elapsed / max(1e-06, args.max_train_seconds))
demon_b1 = demon_beta1_scale(wd_progress)
muon_b2 = muon_beta2(wd_progress)
for group in optimizer.param_groups:
    group['lr'] = group['initial_lr'] * batch_lr_scale * (lrm_flat if group.get('no_lr_decay') else lrm)
    if group['kind'] == 'muon':
        group['momentum'] = muon_momentum(step, wd_progress)
        group['weight_decay'] = WEIGHT_DECAY * (1.0 - wd_progress) if WD_DECAY_TO_ZERO else WEIGHT_DECAY
        group['beta2'] = muon_b2
    elif group['kind'] == 'adamw' and DEMON_BETA1:
        _b1, _b2 = group['initial_betas']
        group['betas'] = (_b1 * demon_b1, _b2)
