import torch


class TrainerBase:
    """Shared plumbing between SFNOSingleStepTrainer and SFNOMultiStepTrainer:
    device setup, GPU-utilization sampling, LR-scheduler construction, and
    optimizer-state resume - identical (or, for the scheduler, made identical
    via an explicit ``scheduler_type``) between the two trainers.
    """

    def _init_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            raise RuntimeError("Device is now CPU !")

    def _gpu_utilization(self):
        try:
            return float(torch.cuda.utilization(self.device))
        except Exception:
            # utilization() needs pynvml; fall back to memory footprint (MB).
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def _build_scheduler(self, optimizer, epochs, warmup_epochs,
                         warmup_start_factor, scheduler_type, restart,
                         restart_period, restart_mult):
        """LR schedule = optional linear warmup, then either a constant rate,
        a single cosine decay, or cosine annealing with warm restarts.

        warmup_epochs        : linear ramp from warmup_start_factor*lr up to lr
                               over this many epochs (0 disables warmup).
        scheduler_type        : 'cosine' (default) - CosineAnnealingLR (or, with
                               restart=True, CosineAnnealingWarmRestarts) over the
                               post-warmup span. 'constant' - lr held flat at its
                               post-warmup value for the rest of training (restart
                               is ignored in this case).
        restart               : if True (and scheduler_type='cosine') use
                               CosineAnnealingWarmRestarts (periodic restarts)
                               instead of a one-shot CosineAnnealingLR.
        restart_period        : length (epochs) of the first restart cycle (T_0);
                               defaults to the post-warmup span when unset.
        restart_mult          : cycle-length growth factor after each restart (T_mult).
        """
        sched = torch.optim.lr_scheduler
        if scheduler_type not in ("cosine", "constant"):
            raise ValueError(
                f"scheduler_type must be 'cosine' or 'constant', got {scheduler_type!r}")

        if scheduler_type == "constant":
            main = sched.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            # cosine runs over whatever epochs remain after the warmup ramp
            remaining = max(epochs - warmup_epochs, 1)
            if restart:
                t0 = restart_period if restart_period is not None else remaining
                main = sched.CosineAnnealingWarmRestarts(
                    optimizer, T_0=t0, T_mult=restart_mult)
            else:
                main = sched.CosineAnnealingLR(optimizer, T_max=remaining)

        if warmup_epochs > 0:
            warmup = sched.LinearLR(
                optimizer, start_factor=warmup_start_factor, end_factor=1.0,
                total_iters=warmup_epochs)
            return sched.SequentialLR(
                optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
        return main

    @staticmethod
    def _restore_optimizer_state(optimizer, resume_checkpoint, lr, weight_decay):
        """Restores Adam's momentum/variance buffers, but also restores the
        OLD lr/weight_decay into param_groups - training config is allowed
        to differ on a continuation, so re-apply the current run's values
        rather than silently keeping whatever the matched run last used.
        """
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = weight_decay
            # Drop the stale initial_lr restored from the checkpoint's own
            # optimizer state - the scheduler built above only fills this in
            # via setdefault(), so leaving the old value in place would
            # silently keep driving the LR schedule off the PREVIOUS run's lr
            # instead of the one just set above.
            pg.pop("initial_lr", None)
