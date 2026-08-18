import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.helpers.print import *
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO

from src.neural_operator.dataset import SWEMultiStepDataset
from src.neural_operator.loss import LOSS_FUNCTIONS
from src.helpers.run_model import neural_model_path

import csv
import json
import math
import os
import random
import time
import warnings
import yaml
import re
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm


class SFNOMultiStepTrainer:
    """Teacher/student curriculum fine-tuning on top of an already-trained
    single-step SFNO (see multi_step_curriculum_training.png):

        Initialize teacher and student from the single-step checkpoint.
        Freeze the teacher.
        for curr_step = 1 .. max_subsequent_steps:
            while student not converged:
                teacher_step ~ Uniform{1, .., curr_step}
                obs1 = t(1), obs2 = t(teacher_step - 1), target = t(teacher_step)
                teacher_output = teacher rolled out (teacher_step - 1) times from obs1
                loss = loss(student(teacher_output), target) + loss(student(obs2), target)
            teacher = student ; freeze teacher

    Deviations from the literal pseudocode, both standard for batched training:
      - "while not converged" is a fixed budget of `epochs_per_stage` epochs
        per curriculum stage rather than an explicit convergence test.
      - `teacher_step` is sampled once per mini-batch (not per example), so a
        whole batch shares one rollout length per step - otherwise every
        example in the batch would need its own teacher rollout length.

    The run lives in the SAME checkpoint directory as the single-step run it
    builds on (README.md: checkpoints_single.pt / checkpoints_multi.pt sit
    side by side) - located via `neural_model_path` from the identical
    architecture knobs, exactly like inference.py's single_step switch does.
    """

    CHECKPOINT_ROOT = os.path.join("checkpoints", "neural_operator")
    LOG_ROOT = os.path.join("training_logs", "neural_operator")

    def __init__(self,
        training_data_dir,
        nlat,
        nlon,
        n_future,
        num_layers,
        embed_dim,
        pos_embed,
        grid,
        index,
        max_subsequent_steps,
        scale_factor=1,
        residual_prediction=True,
        normalization_layer="none",
        loss_type="spectral",
    ):
        print("🧑‍🏫 🧑‍🎓 Starting SFNO Multi-Step Curriculum Training 🧑‍🎓 🧑‍🏫".center(100))
        self.start_time = time.perf_counter()

        self.training_data_dir = training_data_dir
        self.nlat, self.nlon = nlat, nlon
        self.n_future = n_future
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.pos_embed = pos_embed
        self.grid = grid
        self.scale_factor = scale_factor
        self.residual_prediction = residual_prediction
        self.max_subsequent_steps = max_subsequent_steps

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            raise RuntimeError("Device is now CPU !")

        # ------------------------------------------------------------------ #
        # Locate the pretrained single-step run this curriculum builds on -
        # same architecture-identification convention as inference.py.
        # normalization_layer/loss_type must match the single-step run's own
        # config exactly (like every other field here) since they're baked
        # into the tree's norm_*/loss_* nodes.
        # ------------------------------------------------------------------ #
        train_data_tag = re.findall(r"dataset_(\w+)",self.training_data_dir)[0]
        model_path_kwargs = dict(
            resol=(nlat, nlon), n_future=n_future, num_layers=num_layers,
            embed_dim=embed_dim, pos_embed=pos_embed, trainData_path=train_data_tag,
            grid=grid, normalization_layer=normalization_layer, loss_type=loss_type,
            index=index,
        )
        self.run_dir, single_name, multi_name, info_name = neural_model_path(**model_path_kwargs)
        source_ckpt_path = os.path.join(self.run_dir, single_name)
        self.info_file = os.path.join(self.run_dir, info_name)
        self.checkpoint = os.path.join(self.run_dir, multi_name)

        if not os.path.isfile(source_ckpt_path) or not os.path.isfile(self.info_file):
            raise FileNotFoundError(
                f"No pretrained single-step model found at {self.run_dir} "
                f"(expected {single_name} and {info_name}). Train it first with `make train_single`."
            )
        with open(self.info_file, "r", encoding="utf-8") as f:
            self.source_info = json.load(f)
        print(f"📦 Building curriculum on top of single-step run: {source_ckpt_path}")

        self.dataset_name = self.source_info.get("dataset_name")
        self.pressure = self.source_info.get("pressure")

        # non-dimensionalization scales come from the pretrained model itself
        # (README: checkpoints are self-describing) rather than being
        # recomputed from training_data_dir, so the student/teacher keep
        # seeing inputs scaled exactly the way their pretrained weights
        # expect - even if the curriculum trains on a different (e.g. longer)
        # trajectory set.
        self.T = self.source_info["T"]
        self.U = self.source_info["U"]

        self.ds = SWEMultiStepDataset(
            simulation_data_dir=training_data_dir, n_future=n_future,
            max_subsequent_steps=max_subsequent_steps, T=self.T, U=self.U,
        )
        if (self.ds.solver.nlat, self.ds.solver.nlon) != (nlat, nlon):
            raise ValueError(
                f"training_data_dir '{training_data_dir}' has resolution "
                f"({self.ds.solver.nlat}, {self.ds.solver.nlon}), which does not match "
                f"the pretrained model's ({nlat}, {nlon})."
            )
        validation_split = 0.15
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.ds, [1 - validation_split, validation_split]
        )
        self.solver = self.ds.solver

        def _build_model():
            return SFNO(
                img_size=(nlat, nlon), grid=grid,
                num_layers=num_layers, scale_factor=scale_factor, embed_dim=embed_dim,
                residual_prediction=residual_prediction,
                pos_embed=pos_embed, use_mlp=True,
                normalization_layer=self.source_info.get("normalization_layer", "none"),
            ).to(self.device)

        source_state = torch.load(source_ckpt_path, map_location=self.device, weights_only=True)

        # Initialize both the teacher and student to the pretrained single-step
        # model (see pseudocode above).
        self.student = _build_model()
        self.student.load_state_dict(source_state["model_state_dict"])

        self.teacher = _build_model()
        self._sync_teacher_to_student()

        # sourced from source_info (not the raw loss_type constructor arg)
        # for the same reason normalization_layer is above: it's the
        # authoritative record of what this checkpoint actually trained with.
        self.loss = LOSS_FUNCTIONS[self.source_info.get("loss_type", "spectral")]

    def _sync_teacher_to_student(self):
        """teacher = student; freeze the weights of the teacher model."""
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _load_history_from_csv(self):
        """Reconstruct in-memory history (for the dashboard) from an existing
        train_log_multistep.csv, so a resumed run's plot/log stay continuous."""
        history = {"epoch": [], "curr_step": [], "train_loss": [], "validation_loss": [],
                   "lr": [], "gpu_utillization": [], "per_epoch_time": []}
        if not os.path.isfile(self.log_file):
            return history
        with open(self.log_file, "r", newline="") as f:
            for row in csv.DictReader(f):
                history["epoch"].append(int(row["epoch"]))
                history["curr_step"].append(int(row["curr_step"]))
                history["train_loss"].append(float(row["train_loss"]))
                history["validation_loss"].append(
                    float(row["validation_loss"]) if row["validation_loss"] else None)
                history["lr"].append(float(row["lr"]))
                history["gpu_utillization"].append(float(row["gpu_utillization"]))
                history["per_epoch_time"].append(float(row["per_epoch_time"]))
        return history

    def _setup_run(self,
        epochs_per_stage,
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
        self.total_epochs = self.max_subsequent_steps * epochs_per_stage

        run_log_dir = self.run_dir.replace(self.CHECKPOINT_ROOT, self.LOG_ROOT, 1)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(run_log_dir, exist_ok=True)

        self.task_name = os.path.relpath(self.run_dir, self.CHECKPOINT_ROOT) + " (multistep)"
        print(f" task_name = {self.task_name}".center(60))

        self.log_file = os.path.join(run_log_dir, "train_log_multistep.csv")
        self.dashboard_plot = os.path.join(run_log_dir, "dashboard_plot_multistep.png")

        # ------------------------------------------------------------------ #
        # Resume: unlike train_single (which searches 0/, 1/, ... for a
        # matching architecture), there is exactly one multistep checkpoint
        # slot per single-step run (checkpoints_multi.pt lives next to
        # checkpoints_single.pt - README.md), so resuming just means
        # continuing THIS checkpoint. The curriculum schedule itself
        # (max_subsequent_steps, epochs_per_stage) is fixed once a run starts
        # - changing it mid-curriculum would desync which epoch belongs to
        # which curr_step, so a mismatch on resume is a hard error rather than
        # silently reinterpreted.
        # ------------------------------------------------------------------ #
        self.resume = False
        self.resume_from_epoch = 0
        resume_checkpoint = None
        if continue_training and os.path.isfile(self.checkpoint):
            resume_checkpoint = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            stored_max_steps = resume_checkpoint.get("max_subsequent_steps")
            stored_eps = resume_checkpoint.get("epochs_per_stage")
            if stored_max_steps != self.max_subsequent_steps or stored_eps != epochs_per_stage:
                raise ValueError(
                    f"Existing {self.checkpoint} was trained with max_subsequent_steps="
                    f"{stored_max_steps}, epochs_per_stage={stored_eps}; config asks for "
                    f"{self.max_subsequent_steps}/{epochs_per_stage}. The curriculum schedule "
                    f"can't change mid-run - set continue: false to start a fresh curriculum."
                )
            self.resume = True
            self.resume_from_epoch = int(resume_checkpoint["epoch"]) + 1

        if self.resume:
            print_in_box({
                "title": "🔁🔁🔁  continue=True: RESUMING EXISTING MULTISTEP RUN  🔁🔁🔁",
                "lines": [
                    f"{self.checkpoint} already completed epoch {self.resume_from_epoch - 1} "
                    f"(curr_step={self._curr_step_for_epoch(self.resume_from_epoch - 1, epochs_per_stage)}).",
                    (f"Continuing from epoch {self.resume_from_epoch} up to {self.total_epochs} total."
                     if self.resume_from_epoch < self.total_epochs else
                     "It already covers the full curriculum - nothing left to train."),
                ],
            })
        else:
            print_in_box({
                "title": "🆕  STARTING A NEW CURRICULUM (multistep)  🆕",
                "lines": [
                    f"checkpoint -> {self.checkpoint}",
                    f"max_subsequent_steps={self.max_subsequent_steps} x epochs_per_stage={epochs_per_stage} "
                    f"= {self.total_epochs} total epochs",
                ],
            })

        archi_content = {
            "title": "SFNO Multi-Step Curriculum Config",
            "lines": [
                f"nlat = {self.nlat} | nlon = {self.nlon} | n_future = {self.n_future}",
                f"max_subsequent_steps = {self.max_subsequent_steps}",
                f"epochs_per_stage = {epochs_per_stage} -> total_epochs = {self.total_epochs}",
                f"lr = {lr} | batch_size = {batch_size} | weight_decay = {weight_decay}",
                f"warmup_epochs = {warmup_epochs} | restart = {restart}",
                f"validation_cadence = {validation_cadence}",
                f"normalization_layer = {self.source_info.get('normalization_layer', 'none')} "
                f"| loss_type = {self.source_info.get('loss_type', 'spectral')}",
                f"source single-step checkpoint = {self.run_dir}",
            ]
        }
        print_in_box(archi_content)

        if self.resume:
            self.history = self._load_history_from_csv()
        else:
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "curr_step", "train_loss", "validation_loss", "lr",
                     "gpu_utillization", "per_epoch_time"]
                )
            self.history = {
                "epoch": [], "curr_step": [], "train_loss": [], "validation_loss": [],
                "lr": [], "gpu_utillization": [], "per_epoch_time": [],
            }
        self.best_val_loss = math.inf if resume_checkpoint is None else resume_checkpoint.get("val_loss", math.inf)

        # Persist multistep provenance alongside the single-step fields already
        # in model_info.json (nested under "multistep" so nothing collides).
        model_info = dict(self.source_info)
        model_info["multistep"] = {
            "max_subsequent_steps": self.max_subsequent_steps,
            "epochs_per_stage": epochs_per_stage,
            "total_epochs": self.total_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "warmup_start_factor": warmup_start_factor,
            "restart": restart,
            "restart_period": restart_period,
            "restart_mult": restart_mult,
            "validation_cadence": validation_cadence,
            "training_data_dir": self.training_data_dir,
        }
        with open(self.info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=4)

        return resume_checkpoint

    def _curr_step_for_epoch(self, epoch, epochs_per_stage):
        """1-indexed curriculum stage active during `epoch` (0-indexed)."""
        return min(epoch // epochs_per_stage + 1, self.max_subsequent_steps)

    def _gpu_utilization(self):
        try:
            return float(torch.cuda.utilization(self.device))
        except Exception:
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def __update_log(self, epoch, curr_step, train_loss, val_loss, lr, gpu_util,
                     epoch_time, total_epochs, is_validation):
        self.history["epoch"].append(epoch)
        self.history["curr_step"].append(curr_step)
        self.history["train_loss"].append(train_loss)
        self.history["validation_loss"].append(val_loss)
        self.history["lr"].append(lr)
        self.history["gpu_utillization"].append(gpu_util)
        self.history["per_epoch_time"].append(epoch_time)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, curr_step, f"{train_loss:.6e}",
                "" if val_loss is None else f"{val_loss:.6e}",
                f"{lr:.6e}", f"{gpu_util:.2f}", f"{epoch_time:.3f}",
            ])

        avg_epoch_time = sum(self.history["per_epoch_time"]) / len(self.history["per_epoch_time"])
        remaining = max(total_epochs - (epoch + 1), 0)
        eta_sec = avg_epoch_time * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        msg = (f"[epoch {epoch + 1}/{total_epochs} | curr_step={curr_step}] "
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

        ax = axes[0, 0]
        ax.plot(h["epoch"], h["train_loss"], label="train", color="tab:blue")
        val_epochs = [e for e, v in zip(h["epoch"], h["validation_loss"]) if v is not None]
        val_losses = [v for v in h["validation_loss"] if v is not None]
        if val_losses:
            ax.plot(val_epochs, val_losses, label="validation", color="tab:orange", marker="o")
        ax.set_title("Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("spectral L2 (multi + single step)")
        ax.set_yscale("log")
        ax.legend()

        axes[0, 1].step(h["epoch"], h["curr_step"], color="tab:green", where="post")
        axes[0, 1].set_title("Curriculum stage (curr_step)")
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

    def _rollout_teacher(self, obs1, steps):
        """Frozen teacher, `steps` autoregressive calls starting from obs1
        (steps == 0 returns obs1 itself, i.e. teacher_step == 1)."""
        state = obs1
        if steps > 0:
            with torch.no_grad():
                for _ in range(steps):
                    state += self.teacher(state)
        return state

    def _run_stage_epoch(self, loader, curr_step, train, optimizer=None):
        """One pass over `loader` at curriculum stage `curr_step`; returns the
        mean (multi_step_loss + single_step_loss)."""
        self.student.train(train)
        total_loss, n_batches = 0.0, 0
        torch.set_grad_enabled(train)
        for window, _ in loader:
            window = window.to(self.device)  # (B, max_subsequent_steps+1, 3, nlat, nlon)
            obs1 = window[:, 0]

            # sampled once per batch (see class docstring)
            teacher_step = random.randint(1, curr_step)
            obs2 = window[:, teacher_step - 1]
            target = window[:, teacher_step]

            teacher_output = self._rollout_teacher(obs1, teacher_step - 1)

            multi_step_prd = self.student(teacher_output)
            single_step_prd = self.student(obs2)

            multi_step_loss = self.loss(self.solver, multi_step_prd, target, relative=True, squared=False)
            single_step_loss = self.loss(self.solver, single_step_prd, target, relative=True, squared=False)
            loss = multi_step_loss + single_step_loss

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
        sched = torch.optim.lr_scheduler
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
        epochs_per_stage=20,
        batch_size=16,
        lr=1e-5,
        weight_decay=1e-3,
        validation_cadence=5,
        warmup_epochs=0,
        warmup_start_factor=0.01,
        restart=False,
        restart_period=None,
        restart_mult=1,
        compile=False,
        continue_training=True,
    ):
        resume_checkpoint = self._setup_run(
            epochs_per_stage, lr, batch_size, weight_decay, warmup_epochs,
            warmup_start_factor, restart, restart_period, restart_mult,
            validation_cadence, continue_training,
        )

        if self.resume and self.resume_from_epoch >= self.total_epochs:
            return

        if resume_checkpoint is not None:
            self.student.load_state_dict(resume_checkpoint["model_state_dict"])
            # the teacher is restored from its OWN saved state (not the
            # resumed student) since it tracks whichever stage boundary it
            # last crossed, not the resumed student weights.
            teacher_state = resume_checkpoint.get("teacher_state_dict")
            if teacher_state is not None:
                self.teacher.load_state_dict(teacher_state)
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad_(False)
            else:
                self._sync_teacher_to_student()

        train_content = {
            "title": "SFNO Multi-Step Training Configuration",
            "lines": [
                f"batch_size = {batch_size}", f"lr = {lr}",
                f"epochs_per_stage = {epochs_per_stage} | total_epochs = {self.total_epochs}",
                f"weight_decay = {weight_decay}", f"validation_cadence = {validation_cadence}",
                f"warmup_epochs = {warmup_epochs} (start_factor {warmup_start_factor})",
                f"restart = {restart} (T_0={restart_period}, T_mult={restart_mult})",
                f"compile = {compile}",
                f"train / validation samples = {len(self.train_dataset)} / {len(self.test_dataset)}",
            ],
        }
        print_in_box(train_content)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

        if compile:
            try:
                self.student = torch.compile(self.student)
                self.teacher = torch.compile(self.teacher)
            except RuntimeError as e:
                print(f"⚠️  torch.compile unavailable, running eager: {e}")

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                pg["weight_decay"] = weight_decay

        scheduler = self._build_scheduler(
            optimizer, self.total_epochs, warmup_epochs, warmup_start_factor,
            restart, restart_period, restart_mult)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*lr_scheduler.step\\(\\) before `optimizer.step\\(\\)`.*")
            for _ in range(self.resume_from_epoch):
                scheduler.step()

        prev_curr_step = self._curr_step_for_epoch(max(self.resume_from_epoch - 1, 0), epochs_per_stage)
        for ep in tqdm.tqdm(range(self.resume_from_epoch, self.total_epochs), desc="epochs",
                            initial=self.resume_from_epoch, total=self.total_epochs):
            curr_step = self._curr_step_for_epoch(ep, epochs_per_stage)

            # "teacher = student; freeze the weights of the teacher model" -
            # fires exactly once, at the start of a new curriculum stage.
            if curr_step != prev_curr_step:
                self._sync_teacher_to_student()
                prev_curr_step = curr_step

            epoch_start = time.perf_counter()
            train_loss = self._run_stage_epoch(train_loader, curr_step, train=True, optimizer=optimizer)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            is_validation = ((ep + 1) % validation_cadence == 0) or (ep == self.total_epochs - 1)
            val_loss = None
            if is_validation:
                val_loss = self._run_stage_epoch(test_loader, curr_step, train=False)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    student_to_save = getattr(self.student, "_orig_mod", self.student)
                    teacher_to_save = getattr(self.teacher, "_orig_mod", self.teacher)
                    torch.save({
                        "epoch": ep,
                        "curr_step": curr_step,
                        "model_state_dict": student_to_save.state_dict(),
                        "teacher_state_dict": teacher_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "task_name": self.task_name,
                        "max_subsequent_steps": self.max_subsequent_steps,
                        "epochs_per_stage": epochs_per_stage,
                    }, self.checkpoint)

            epoch_time = time.perf_counter() - epoch_start
            gpu_util = self._gpu_utilization()
            self.__update_log(ep, curr_step, train_loss, val_loss, current_lr, gpu_util,
                              epoch_time, self.total_epochs, is_validation)

        total_time = time.perf_counter() - self.start_time
        print(f"㊣ ㊣ ㊣ Finished Multi-Step Curriculum Training ! Total time spent {total_time:3f} ㊣ ㊣ ㊣")
        print(f"Best validation loss = {self.best_val_loss:.6e} -> {self.checkpoint}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    DEFAULT_CONFIG = {
        # identifies the pretrained single-step run to build on (same
        # convention as the `inference` config section)
        "nlat": 128,
        "nlon": 256,
        "n_future": 1,
        "num_layers": 4,
        "scale_factor": 1,
        "embed_dim": 16,
        "residual_prediction": True,
        "pos_embed": "learnable lat",
        "trainData": "equiangular",
        "grid": "equiangular",
        "normalization_layer": "none",  # must match the single-step run's own value
        "loss_type": "spectral",        # must match the single-step run's own value
        "index": 0,
        # curriculum
        "max_subsequent_steps": 4,
        "epochs_per_stage": 20,
        # optimization
        "batch_size": 16,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "validation_cadence": 5,
        "warmup_epochs": 0,
        "warmup_start_factor": 0.01,
        "restart": False,
        "restart_period": None,
        "restart_mult": 1,
        "compile": False,
        "continue": True,
    }

    raw_train_cfg = DEFAULT_CONFIG | config.get("train_multi", {})

    raw_train_cfg["nlat"] = int(raw_train_cfg["nlat"])
    raw_train_cfg["nlon"] = int(raw_train_cfg["nlon"])
    raw_train_cfg["n_future"] = int(raw_train_cfg["n_future"])
    raw_train_cfg["num_layers"] = int(raw_train_cfg["num_layers"])
    raw_train_cfg["scale_factor"] = int(raw_train_cfg["scale_factor"])
    raw_train_cfg["embed_dim"] = int(raw_train_cfg["embed_dim"])
    raw_train_cfg["index"] = int(raw_train_cfg["index"])
    raw_train_cfg["max_subsequent_steps"] = int(raw_train_cfg["max_subsequent_steps"])
    raw_train_cfg["epochs_per_stage"] = int(raw_train_cfg["epochs_per_stage"])
    raw_train_cfg["batch_size"] = int(raw_train_cfg["batch_size"])
    raw_train_cfg["lr"] = float(raw_train_cfg["lr"])
    raw_train_cfg["weight_decay"] = float(raw_train_cfg["weight_decay"])
    raw_train_cfg["warmup_epochs"] = int(raw_train_cfg["warmup_epochs"])
    raw_train_cfg["warmup_start_factor"] = float(raw_train_cfg["warmup_start_factor"])
    raw_train_cfg["restart_mult"] = int(raw_train_cfg["restart_mult"])
    if raw_train_cfg["restart_period"] is not None:
        raw_train_cfg["restart_period"] = int(raw_train_cfg["restart_period"])

    # "continue" is a Python keyword - can't be a SimpleNamespace attribute
    # accessed via dot syntax, so pull it out before the conversion below.
    continue_training = bool(raw_train_cfg.pop("continue"))

    train_config = SimpleNamespace(**raw_train_cfg)

    trainer = SFNOMultiStepTrainer(
        training_data_dir=train_config.training_data_dir,
        nlat=train_config.nlat,
        nlon=train_config.nlon,
        n_future=train_config.n_future,
        num_layers=train_config.num_layers,
        embed_dim=train_config.embed_dim,
        pos_embed=train_config.pos_embed,
        grid=train_config.grid,
        index=train_config.index,
        max_subsequent_steps=train_config.max_subsequent_steps,
        scale_factor=train_config.scale_factor,
        residual_prediction=train_config.residual_prediction,
        normalization_layer=train_config.normalization_layer,
        loss_type=train_config.loss_type,
    )

    trainer.train(
        epochs_per_stage=train_config.epochs_per_stage,
        batch_size=train_config.batch_size,
        lr=train_config.lr,
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
