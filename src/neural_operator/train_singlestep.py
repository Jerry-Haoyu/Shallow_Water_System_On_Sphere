import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))


import torch
import torch.nn as nn
import torch_harmonics
from torch.utils.data import Dataset, DataLoader
from src.helpers.print import *
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO

from src.neural_operator.dataset import SWEDataset
from src.neural_operator.loss import spectral_l2loss_sphere
from src.helpers.run_model import neural_model_path

import csv
import math
import os
import time
import json
import warnings
import yaml
from types import SimpleNamespace 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm




class SFNOSingleStepTrainer:
    # Roots of the neural-operator checkpoint / log trees. The per-run
    # directories underneath them are built from the model + training
    # configuration in _setup_run (see README.md for the convention).
    CHECKPOINT_ROOT = os.path.join("checkpoints", "neural_operator")
    LOG_ROOT = os.path.join("training_logs", "neural_operator")

    def __init__(self,
        training_data_dir,
        n_future=1,
        num_layers=4,
        scale_factor=1,
        embed_dim=16,
        residual_prediction=True,
        pos_embed='learnable lat'
    ):
        print("😗 😗 Starting SFNO Single Step Training 😗 😗 ".center(100))

        self.start_time = time.perf_counter()

        # Architecture knobs are stored so both the run directory name (built in
        # train(), since it also encodes the optimization hyperparameters) and
        # the model construction below can reference them.
        self.n_future = n_future
        self.num_layers = num_layers
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.residual_prediction = residual_prediction
        self.pos_embed = pos_embed
        self.training_data_dir = training_data_dir

        # check if gpu is available here, if not, stop the training immediately
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            raise RuntimeError("Device is now CPU !")

        # Initialize the dataset
        self.ds = SWEDataset(simulation_data_dir=training_data_dir, n_future=n_future)
        validation_split = 0.15
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.ds, [1 - validation_split, validation_split]
        )
        self.nlat, self.nlon = self.ds.solver.nlat, self.ds.solver.nlon
        self.inp_mean_gp, self.inp_mean_zeta, self.inp_mean_delta  = self.ds.inp_mean[0].item(), self.ds.inp_mean[1].item(), self.ds.inp_mean[2].item()
        self.inp_std_gp, self.inp_std_zeta, self.inp_std_delta = self.ds.inp_std[0].item(), self.ds.inp_std[1].item(), self.ds.inp_std[2].item()

        # Initalize the model
        self.model = SFNO(
            img_size=(self.nlat, self.nlon), grid=self.ds.solver.grid,
            num_layers=num_layers, scale_factor=scale_factor, embed_dim=embed_dim,
            residual_prediction=residual_prediction,
            pos_embed=pos_embed, use_mlp=True,
            normalization_layer="none",
        ).to(self.device)

        # Loss is a plain function (solver, prd, tar) -> scalar; the solver used
        # for the spherical harmonic transform lives on the dataset.
        self.loss = spectral_l2loss_sphere
        self.solver = self.ds.solver

    @staticmethod
    def _next_run_index(base_dir):
        """Smallest non-negative integer whose sub-directory does not yet exist
        under ``base_dir`` (the 0/, 1/, 2/, ... training-config slots)."""
        if not os.path.isdir(base_dir):
            return 0
        existing = {int(d) for d in os.listdir(base_dir)
                    if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))}
        i = 0
        while i in existing:
            i += 1
        return i

    def _architecture_signature(self):
        """Everything that defines the model itself (fixed at __init__) -
        deliberately EXCLUDES optimization hyperparameters (lr, batch_size,
        epochs, schedule, ...), which can freely change between a run and the
        one continuing it. Two runs sharing this signature have weights that
        mean the same thing, so continuing training just means loading them
        and picking up wherever the optimizer/epoch count leaves off.
        """
        return {
            "nlat": self.nlat, "nlon": self.nlon, "grid": self.solver.grid,
            "n_future": self.n_future, "num_layers": self.num_layers,
            "pos_embed": self.pos_embed, "scale_factor": self.scale_factor,
            "embed_dim": self.embed_dim, "residual_prediction": self.residual_prediction,
            "normalization_layer": "none", "training_data": self.training_data_dir,
        }

    @staticmethod
    def _find_matching_run(base_dir, signature):
        """Scan ``base_dir``'s 0/, 1/, ... sub-directories for one whose
        model_info.json matches ``signature`` (exact equality on every key the
        candidate file actually has). Returns ``(index, model_info)`` for the
        first match, else ``(None, None)``.

        A key that's present in the candidate but differs still fails the
        match. A key that's simply *absent* (e.g. an older run trained before
        that knob was added to model_info.json's schema) does not block a
        match on its own - there's nothing in the file to contradict, and
        treating "never recorded" the same as "recorded and different" would
        permanently orphan every run trained before a new knob was tracked.
        A directory whose model_info.json is missing/unreadable is skipped.
        """
        if not os.path.isdir(base_dir):
            return None, None
        for name in sorted((d for d in os.listdir(base_dir) if d.isdigit()), key=int):
            sub_dir = os.path.join(base_dir, name)
            info_path = os.path.join(sub_dir, "model_info.json")
            if not (os.path.isdir(sub_dir) and os.path.isfile(info_path)):
                continue
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if all(key not in info or info[key] == value for key, value in signature.items()):
                return int(name), info
        return None, None

    def _load_history_from_csv(self):
        """Reconstruct in-memory history (for the dashboard) from an existing
        train_log.csv, so a resumed run's plot/log stay continuous instead of
        resetting at the resume point."""
        history = {"epoch": [], "train_loss": [], "validation_loss": [],
                   "lr": [], "gpu_utillization": [], "per_epoch_time": []}
        if not os.path.isfile(self.log_file):
            return history
        with open(self.log_file, "r", newline="") as f:
            for row in csv.DictReader(f):
                history["epoch"].append(int(row["epoch"]))
                history["train_loss"].append(float(row["train_loss"]))
                history["validation_loss"].append(
                    float(row["validation_loss"]) if row["validation_loss"] else None)
                history["lr"].append(float(row["lr"]))
                history["gpu_utillization"].append(float(row["gpu_utillization"]))
                history["per_epoch_time"].append(float(row["per_epoch_time"]))
        return history

    def _setup_run(self,
                   epochs,
                    lr,
                    batch_size,
                    weight_decay,
                    warmup_epochs,
                    warmup_start_factor,
                    restart,
                    restart_period,
                    restart_mult,
                    validation_cadence,
                    continue_training,
        ):
        """Build the task name and create the per-run output files/directories.

        The task name encodes both the architecture (fixed at __init__) and the
        optimization hyperparameters (only known here), so this must run once at
        the start of train().
        """
        # ------------------------------------------------------------------ #
        # Build the per-run directory following the neural-operator convention
        # (README.md). neural_model_path is the single source of truth for the
        # tree so inference.py resolves exactly what the trainer writes:
        #
        #   resol_*/nfuture_*/nlayer_*/embed_*/trainData_*/posEmbed_*/grid_*/<index>/
        #
        # The optimization hyperparameters are NOT encoded in the path; instead
        # each new training config for the same architecture lands in the next
        # integer sub-directory (0/, 1/, 2/, ...) and is documented in
        # model_info.json. The log tree mirrors the checkpoint tree under LOG_ROOT.
        # ------------------------------------------------------------------ #
        self.train_data_tag = os.path.basename(os.path.normpath(self.training_data_dir))
        model_path_kwargs = dict(
            resol=(self.nlat, self.nlon),
            n_future=self.n_future,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            pos_embed=self.pos_embed,
            trainData_path=self.train_data_tag,
            grid=self.solver.grid,
        )

        # If continue_training is set, search 0/, 1/, ... for a run with the
        # SAME ARCHITECTURE (training hyperparameters need not match - lr,
        # epochs, schedule, etc. are all free to change on a continuation) and
        # resume it instead of starting over with reinitialized weights.
        # Otherwise (the default) always claim a fresh index, full stop.
        probe_dir, *_ = neural_model_path(index=0, **model_path_kwargs)
        base_ckpt_dir = os.path.dirname(probe_dir)

        match_index, match_info = (
            self._find_matching_run(base_ckpt_dir, self._architecture_signature())
            if continue_training else (None, None)
        )

        if match_index is not None and match_info is not None:
            self.resume = True
            self.resume_from_epoch = int(match_info["epochs"])
            self.run_index = match_index
        else:
            self.resume = False
            self.resume_from_epoch = 0
            self.run_index = self._next_run_index(base_ckpt_dir)

        run_ckpt_dir, single_name, _multi_name, info_name = neural_model_path(
            index=self.run_index, **model_path_kwargs)
        # log tree mirrors the checkpoint tree, only the root differs
        run_log_dir = run_ckpt_dir.replace(self.CHECKPOINT_ROOT, self.LOG_ROOT, 1)

        self.task_name = os.path.relpath(run_ckpt_dir, self.CHECKPOINT_ROOT)
        print(f" task_name = {self.task_name}".center(60))

        os.makedirs(run_ckpt_dir, exist_ok=True)
        os.makedirs(run_log_dir, exist_ok=True)

        # -------------------------------------------------------------- #
        # Make the resume decision impossible to miss in the run's stdout.
        # -------------------------------------------------------------- #
        if self.resume:
            print_in_box({
                "title": "🔁🔁🔁  continue=True: RESUMING EXISTING RUN (architecture match found)  🔁🔁🔁",
                "lines": [
                    f"That run already completed {self.resume_from_epoch} epoch(s).",
                    (f"Continuing training from epoch {self.resume_from_epoch} up to {epochs} epoch(s) total."
                     if self.resume_from_epoch < epochs else
                     f"It already covers the requested {epochs} epoch(s) - nothing left to train."),
                ],
            })
        elif continue_training:
            print_in_box({
                "title": "🆕  continue=True but NO ARCHITECTURE MATCH found: NEW RUN  🆕",
                "lines": [
                    f"Searched 0/, 1/, ... under {base_ckpt_dir}; none share this architecture.",
                    f"Using a fresh index: {self.run_index}/",
                ],
            })
        else:
            print_in_box({
                "title": "🆕  continue=False: STARTING FROM SCRATCH (no search performed)  🆕",
                "lines": [
                    f"Using a fresh index: {self.run_index}/ under {base_ckpt_dir}",
                    f"Set 'continue: true' in config to resume a matching architecture instead.",
                ],
            })

        # Document the model structure for future reference
        self.info_file = os.path.join(run_ckpt_dir, info_name)
        # The model checkpoint (mirrors the path, single-step suffix)
        self.checkpoint = os.path.join(run_ckpt_dir, single_name)
        # Training log
        self.log_file = os.path.join(run_log_dir, "train_log.csv")
        # A 2x2 matplotlib dashboard, refreshed every validation step
        self.dashboard_plot = os.path.join(run_log_dir, "dashboard_plot.png")

        archi_content = {
            "title": "SFNO Model Basic Architecture and Training Config",
            "lines": [
                f"nlat = {self.nlat} | nlon = {self.nlon} : Resolution",
                f"n_future = {self.n_future} : M(D_t) = D_(t+n_future*0.5h)",
                f"num_layers = {self.num_layers} : number of SFNO layers",
                f"pos_embed = {self.pos_embed} : positional embedding before SFNO layers",
                f"scale_factor = {self.scale_factor} : downsampling ratio",
                f"embed_dim = {self.embed_dim} : up-projection from 3 channels",
                f"training_data = {self.training_data_dir}",
                f"inp_mean = ({self.inp_mean_gp:.2f}, {self.inp_mean_zeta:.2f}, {self.inp_mean_delta:.2f})",
                f"inp_std = ({self.inp_std_gp:.2f}, {self.inp_std_zeta:.2f}, {self.inp_std_delta:.2f})",
                f"epochs = {epochs} ({'RESUMING from ' + str(self.resume_from_epoch) if self.resume else 'fresh run'})",
                f"lr = {lr}",
                f"batch_size = {batch_size}" ,
                f"weight_decay = {weight_decay}" ,
                f"warmup_epochs = {warmup_epochs}" ,
                f"warmup_start_factor = {warmup_start_factor}" ,
                f"restart = {restart}",
                f"restart_period = {restart_period}",
                f"restart_mult = {restart_mult} ",
                f"validation_cadence = {validation_cadence}",
            ]
        }

        print_in_box(archi_content)

        # Persist the model description alongside the checkpoint. Keep this in
        # sync with _config_signature()'s keys - anything compared there for
        # resume-matching should be recorded here too.
        model_info = {
            "type" : "neural_operator",
            "nlat" : self.nlat, # Resolution
            "nlon" : self.nlon, # Resolution
            "grid" : self.solver.grid, # quadrature grid (needed to rebuild SFNO)
            "n_future" : self.n_future, # M(D_t) = D_(t+n_future*0.5h)
            "num_layers" : self.num_layers, # number of SFNO layers
            "pos_embed" : self.pos_embed , # positional embedding before SFNO layers"
            "scale_factor" : self.scale_factor , # downsampling ratio"
            "embed_dim" : self.embed_dim , # up-projection from 3 channels"
            "residual_prediction" : self.residual_prediction ,
            "normalization_layer" : "none" ,
            "run_index" : self.run_index ,
            "training_data" : self.training_data_dir ,
            "inp_mean_gp" : self.inp_mean_gp ,
            "inp_mean_zeta" : self.inp_mean_zeta ,
            "inp_mean_delta" : self.inp_mean_delta ,
            "inp_std_gp" : self.inp_std_gp ,
            "inp_std_zeta" : self.inp_std_zeta ,
            "inp_std_delta" : self.inp_std_delta ,
            "epochs" : epochs,
            "lr" : lr ,
            "batch_size" : batch_size ,
            "weight_decay" : weight_decay ,
            "warmup_epochs" : warmup_epochs ,
            "warmup_start_factor" : warmup_start_factor ,
            "restart" : restart ,
            "restart_period" : restart_period,
            "restart_mult" : restart_mult,
            "validation_cadence" : validation_cadence,
        }

        with open(self.info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=4)

        if self.resume:
            # Preserve the existing log so the dashboard/CSV stay continuous
            # across the resume point instead of resetting.
            self.history = self._load_history_from_csv()
        else:
            # CSV header (also the schema consumed by __update_log / the dashboard)
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "validation_loss", "lr",
                     "gpu_utillization", "per_epoch_time"]
                )
            self.history = {
                "epoch": [], "train_loss": [], "validation_loss": [],
                "lr": [], "gpu_utillization": [], "per_epoch_time": [],
            }
        self.best_val_loss = math.inf

    def _gpu_utilization(self):
        try:
            return float(torch.cuda.utilization(self.device))
        except Exception:
            # utilization() needs pynvml; fall back to memory footprint (MB).
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def __update_log(self, epoch, train_loss, val_loss, lr, gpu_util,
                     epoch_time, total_epochs, is_validation):
        """Append one row to the CSV, print an ETA, and (on validation steps)
        redraw the dashboard plot."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["validation_loss"].append(val_loss)  # None on non-val steps
        self.history["lr"].append(lr)
        self.history["gpu_utillization"].append(gpu_util)
        self.history["per_epoch_time"].append(epoch_time)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6e}",
                "" if val_loss is None else f"{val_loss:.6e}",
                f"{lr:.6e}", f"{gpu_util:.2f}", f"{epoch_time:.3f}",
            ])

        # ETA from the running average epoch time.
        avg_epoch_time = sum(self.history["per_epoch_time"]) / len(self.history["per_epoch_time"])
        remaining = max(total_epochs - (epoch + 1), 0)
        eta_sec = avg_epoch_time * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        msg = (f"[epoch {epoch + 1}/{total_epochs}] "
               f"train={train_loss:.4e} "
               f"{'val=' + format(val_loss, '.4e') + ' ' if val_loss is not None else ''}"
               f"lr={lr:.2e} gpu={gpu_util:.0f} "
               f"epoch_time={epoch_time:.1f}s ETA={eta_str}")
        print(msg)

        if is_validation:
            self._draw_dashboard()

    def _draw_dashboard(self):
        h = self.history
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # train + validation loss on the same axes
        ax = axes[0, 0]
        ax.plot(h["epoch"], h["train_loss"], label="train", color="tab:blue")
        val_epochs = [e for e, v in zip(h["epoch"], h["validation_loss"]) if v is not None]
        val_losses = [v for v in h["validation_loss"] if v is not None]
        if val_losses:
            ax.plot(val_epochs, val_losses, label="validation",
                    color="tab:orange", marker="o")
        ax.set_title("Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("spectral L2")
        ax.set_yscale("log")
        ax.legend()

        axes[0, 1].plot(h["epoch"], h["lr"], color="tab:green")
        axes[0, 1].set_title("Learning rate")
        axes[0, 1].set_xlabel("epoch")

        axes[1, 0].plot(h["epoch"], h["gpu_utillization"], color="tab:red")
        axes[1, 0].set_title("GPU utilization")
        axes[1, 0].set_xlabel("epoch")

        axes[1, 1].plot(h["epoch"], h["per_epoch_time"], color="tab:purple")
        axes[1, 1].set_title("Per-epoch time (s)")
        axes[1, 1].set_xlabel("epoch")

        fig.suptitle(self.task_name)
        fig.tight_layout()
        fig.savefig(self.dashboard_plot, dpi=120)
        plt.close(fig)

    def _run_epoch(self, loader, train, optimizer=None):
        """Run a single pass over ``loader``; returns the mean loss."""
        self.model.train(train)
        total_loss, n_batches = 0.0, 0
        torch.set_grad_enabled(train)
        for inp, tar, _ in loader:
            inp = inp.to(self.device)
            tar = tar.to(self.device)

            prd = self.model(inp)
            loss = self.loss(self.solver, prd, tar, relative=True, squared=False)

            if train:
                assert optimizer is not None, "optimizer required when train=True"
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        torch.set_grad_enabled(True)
        return total_loss / max(n_batches, 1)

    def _build_scheduler(self, optimizer, epochs, warmup_epochs,
                         warmup_start_factor, restart, restart_period,
                         restart_mult):
        """LR schedule = optional linear warmup, then either a single cosine
        decay or cosine annealing with warm restarts.

        warmup_epochs        : linear ramp from warmup_start_factor*lr up to lr
                               over this many epochs (0 disables warmup).
        restart              : if True use CosineAnnealingWarmRestarts (periodic
                               restarts) instead of a one-shot CosineAnnealingLR.
        restart_period       : length (epochs) of the first restart cycle (T_0);
                               defaults to the post-warmup span when unset.
        restart_mult         : cycle-length growth factor after each restart (T_mult).
        """
        sched = torch.optim.lr_scheduler
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

    def train(self,
        batch_size=128,
        lr=5e-3,
        epochs=800,
        weight_decay=1e-3,
        validation_cadence=10,
        warmup_epochs=0,            # linear LR warmup length (0 = off)
        warmup_start_factor=0.01,   # warmup starts at this fraction of lr
        restart=False,              # cosine annealing with warm restarts
        restart_period=None,        # first restart cycle length T_0 (epochs)
        restart_mult=1,             # restart cycle growth factor T_mult
        compile=True,  # Whether to compile the model
        continue_training=False,    # resume a matching-architecture run instead of starting fresh
        ):

        self._setup_run(epochs,
                        lr,
                        batch_size,
                        weight_decay,
                        warmup_epochs,
                        warmup_start_factor,
                        restart,
                        restart_period,
                        restart_mult,
                        validation_cadence,
                        continue_training,
                        )

        if self.resume and self.resume_from_epoch >= epochs:
            # Nothing left to do - already reported in the RESUMING box above.
            return

        resume_checkpoint = None
        if self.resume:
            resume_checkpoint = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(resume_checkpoint["model_state_dict"])
            self.best_val_loss = resume_checkpoint.get("val_loss", self.best_val_loss)

        train_content = {
            "title": "SFNO Training Configuration",
            "lines": [
                f"batch_size = {batch_size}",
                f"lr = {lr}",
                f"epochs= {epochs}",
                f"weight_decay = {weight_decay}",
                f"validation_cadence = {validation_cadence}",
                f"warmup_epochs = {warmup_epochs} (start_factor {warmup_start_factor})",
                f"restart = {restart} (T_0={restart_period}, T_mult={restart_mult})",
                f"compile = {compile}",
                f"train / validation samples = {len(self.train_dataset)} / {len(self.test_dataset)}",
            ],
        }
        print_in_box(train_content)

        # The dataset produces tensors already living on the GPU, so keep the
        # DataLoader single-process (CUDA tensors cannot cross worker boundaries).
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

        if compile:
            try:
                self.model = torch.compile(self.model)
            except RuntimeError as e:
                # e.g. "Dynamo is not supported on Python 3.13+"; fall back to
                # eager mode rather than aborting the run.
                print(f"⚠️  torch.compile unavailable, running eager: {e}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        if resume_checkpoint is not None:
            # Restores Adam's momentum/variance buffers, but also restores the
            # OLD lr/weight_decay into param_groups - training config is allowed
            # to differ on a continuation, so re-apply the current run's values
            # rather than silently keeping whatever the matched run last used.
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                pg["weight_decay"] = weight_decay

        scheduler = self._build_scheduler(
            optimizer, epochs, warmup_epochs, warmup_start_factor,
            restart, restart_period, restart_mult)
        # Scheduler state isn't checkpointed; fast-forward it so a resumed run's
        # LR trajectory matches what a single 0..epochs run would have had at
        # this point, instead of restarting the schedule from epoch 0. This
        # necessarily calls scheduler.step() without any optimizer.step() in
        # between (we're replaying epoch boundaries, not training), which
        # trips PyTorch's "step() before optimizer.step()" UserWarning even
        # though no LR value is actually being skipped here - suppress just
        # that expected warning rather than let it worry future readers.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*lr_scheduler.step\\(\\) before `optimizer.step\\(\\)`.*")
            for _ in range(self.resume_from_epoch):
                scheduler.step()

        # (1) mini-batch gradient descent, logging every epoch
        # (2) every validation_cadence, evaluate; keep the best checkpoint and
        #     refresh the dashboard plot.
        for ep in tqdm.tqdm(range(self.resume_from_epoch, epochs), desc="epochs",
                            initial=self.resume_from_epoch, total=epochs):
            epoch_start = time.perf_counter()
            train_loss = self._run_epoch(train_loader, train=True, optimizer=optimizer)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            is_validation = ((ep + 1) % validation_cadence == 0) or (ep == epochs - 1)
            val_loss = None
            if is_validation:
                val_loss = self._run_epoch(test_loader, train=False)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    # Save the underlying (uncompiled) module's state dict.
                    model_to_save = getattr(self.model, "_orig_mod", self.model)
                    torch.save({
                        "epoch": ep,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "task_name": self.task_name,
                    }, self.checkpoint)

            epoch_time = time.perf_counter() - epoch_start
            gpu_util = self._gpu_utilization()
            self.__update_log(ep, train_loss, val_loss, current_lr, gpu_util,
                              epoch_time, epochs, is_validation)

        total_time = time.perf_counter() - self.start_time
        print(f"㊣ ㊣ ㊣ Finshed Traning Single-Step ! Total time spent {total_time:3f} ㊣ ㊣ ㊣")
        print(f"Best validation loss = {self.best_val_loss:.6e} -> {self.checkpoint}")


def main():
    # Accept config file from CLI arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    DEFAULT_CONFIG = {
        "n_future" : 1,
        "num_layers" : 4,
        "scale_factor" : 3,
        "embed_dim" : 16,
        "residual_prediction" : True,
        "pos_embed" : 'learnable lat',
        "batch_size" : 128,
        "lr" : 5e-4,
        "epochs" : 100,
        "weight_decay" : 0.01,
        "validation_cadence" : 25,
        "warmup_epochs" : 5,
        "warmup_start_factor" : 0.01,
        "restart" : True,
        "restart_period" : 30,
        "restart_mult" : 2,
        "compile" : False,
        "continue" : False,   # resume a matching-architecture run instead of starting fresh
    }

    raw_train_cfg = DEFAULT_CONFIG | config.get("train_single", {}) # config.get syntax : get(key, fallback)

    # Ensure types are correct regardless of YAML formatting
    raw_train_cfg["lr"] = float(raw_train_cfg["lr"])
    raw_train_cfg["weight_decay"] = float(raw_train_cfg["weight_decay"])
    raw_train_cfg["warmup_start_factor"] = float(raw_train_cfg["warmup_start_factor"])
    raw_train_cfg["epochs"] = int(raw_train_cfg["epochs"])
    raw_train_cfg["batch_size"] = int(raw_train_cfg["batch_size"])

    # "continue" is a Python keyword - can't be a SimpleNamespace attribute
    # accessed via dot syntax, so pull it out before the conversion below.
    continue_training = bool(raw_train_cfg.pop("continue"))

    # this allows yml file dict access to go from file['attribute'] to file.attribute
    train_config = SimpleNamespace(**raw_train_cfg)
    
    trainer = SFNOSingleStepTrainer(
        training_data_dir=train_config.training_data_dir,
        n_future=train_config.n_future,
        num_layers=train_config.num_layers,
        scale_factor=train_config.scale_factor,
        embed_dim=train_config.embed_dim,
        residual_prediction=train_config.residual_prediction,
        pos_embed=train_config.pos_embed,
    )
    
    trainer.train(
        batch_size=train_config.batch_size,
        lr=train_config.lr,
        epochs=train_config.epochs,
        weight_decay=train_config.weight_decay,
        validation_cadence=train_config.validation_cadence,
        warmup_epochs=train_config.warmup_epochs,
        warmup_start_factor=train_config.warmup_start_factor,
        restart=train_config.restart,
        restart_period=train_config.restart_period,
        restart_mult=train_config.restart_mult,
        compile=train_config.compile,
        continue_training=continue_training,
    )


if __name__ == "__main__":
    main()
