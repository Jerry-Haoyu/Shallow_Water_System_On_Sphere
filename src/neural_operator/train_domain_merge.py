"""Gradually merge a second training domain into an already-trained SFNO.

Given a model M1 already trained on domain D1 (`make train_single`), this
fine-tunes it on a MIXTURE of D1 and a second domain D2 - e.g. an ERA5-trained
model gradually exposed to Galewsky trajectories, or the reverse. The mixture
is a curriculum: the probability of any given training sample being drawn from
D2 follows a sigmoid from 0% at the first epoch to `p_max` (default 50%) at the
last, so the model is never handed a hard domain switch.

Why a ramp rather than simply training on the union from the start: M1's
weights encode D1, and a cold 50/50 mixture applies the full distribution
shift on the first optimizer step, which is exactly the regime where fine-
tuning forgets the original domain. Ramping D2 in keeps every early step close
to the distribution M1 already fits, and the sigmoid's flat tails give both a
gentle onset and a settled final mixture rather than a schedule still moving
when training stops.

Both domains are validated SEPARATELY every validation_cadence epochs, since
the question this pipeline exists to answer is two-sided: D2's validation loss
says whether the new domain is being acquired, D1's says whether the old one is
being forgotten. The checkpoint kept is the one minimizing their MEAN, i.e. the
best joint model rather than the best on either domain alone.

The merged run lands in its own `trainData_(<D1>+<D2>)/<index>/` slot under the
usual neural-operator tree (README.md), so it is an ordinary single-step
checkpoint as far as inference.py is concerned - run it by pointing
`inference.trainData` at that same merged tag.

Configuration is read from the ``train_domain_merge`` section of config.yml
(pass an alternative path as the first CLI argument).

@author Haoyu Tang hytang2@illinois.edu
"""
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust .parent steps as needed
sys.path.insert(0, str(SRC_DIR))

import csv
import json
import math
import os
import re
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm

from src.helpers.print import print_in_box
from src.helpers.config import load_raw_config
from src.helpers.run_model import neural_model_path
from src.neural_operator.sfno_model import SphericalFourierNeuralOperator as SFNO
from src.neural_operator.dataset import SWEDataset, DomainMixtureDataset
from src.neural_operator.loss import LOSS_FUNCTIONS
from src.neural_operator.trainer_base import TrainerBase


def sigmoid_domain_probability(epoch, epochs, p_max=0.5, steepness=10.0, midpoint=0.5):
    """Probability of drawing a training sample from D2 at `epoch` (0-indexed).

    A logistic in normalized training progress t = epoch/(epochs-1), affinely
    rescaled so the endpoints are met EXACTLY: p(first epoch) == 0.0 and
    p(last epoch) == p_max. A raw logistic only approaches its asymptotes, so
    without the rescaling the schedule would start at a small but nonzero D2
    fraction and stop short of p_max - both of which contradict what the
    curriculum is specified to do, and the first of which quietly removes the
    pure-D1 warm start that motivates ramping at all.

    steepness : logistic slope. Larger = a flatter pure-D1 plateau early, a
                sharper transition around `midpoint`, and a flatter settled
                mixture at the end. The default 10 spends roughly the first
                and last fifth of training on the plateaus.
    midpoint  : where in normalized progress (0..1) the mixture passes
                p_max/2 - lower shifts D2 in earlier.
    """
    if steepness <= 0:
        raise ValueError(f"steepness must be > 0, got {steepness}")
    if epochs <= 1:
        return float(p_max)
    t = epoch / (epochs - 1)

    def logistic(x):
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))

    lo, hi = logistic(0.0), logistic(1.0)
    return float(p_max) * (logistic(t) - lo) / (hi - lo)


def training_data_tag(training_data_dir):
    """The `trainData_(...)` node this directory maps to (README.md's
    convention) - its `dataset_<name>` node when it has one, else its
    `ic_<name>` node (galewsky-only data has no dataset of its own). Same
    resolution train_singlestep.py's _setup_run performs, so a merged run's
    tag is built from exactly the tags its two source runs would have used.
    """
    dataset_match = re.findall(r"dataset_(\w+)", training_data_dir)
    if dataset_match:
        return dataset_match[0]
    ic_match = re.findall(r"ic_(\w+)", training_data_dir)
    return ic_match[0] if ic_match else "unknown"


class SFNODomainMergeTrainer(TrainerBase):
    CHECKPOINT_ROOT = os.path.join("checkpoints", "neural_operator")
    LOG_ROOT = os.path.join("training_logs", "neural_operator")
    CHANNEL_NAMES = ("phi", "vorticity", "divergence")

    def __init__(self,
        d1_training_data_dir,
        d2_training_data_dir,
        nlat,
        nlon,
        n_future,
        num_layers,
        embed_dim,
        pos_embed,
        grid,
        index,
        normalization_layer="none",
        loss_type="spectral",
        samples_per_file=1,
        cache_in_memory=True,
        p_max=0.5,
        sigmoid_steepness=10.0,
        sigmoid_midpoint=0.5,
    ):
        print("🌍 🤝 🌀 Starting SFNO Domain-Merge Fine-Tuning 🌀 🤝 🌍".center(100))
        self.start_time = time.perf_counter()

        self.d1_training_data_dir = d1_training_data_dir
        self.d2_training_data_dir = d2_training_data_dir
        self.nlat, self.nlon = nlat, nlon
        self.n_future = n_future
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.pos_embed = pos_embed
        self.grid = grid
        self.source_index = index
        self.normalization_layer = normalization_layer
        self.loss_type = loss_type
        self.samples_per_file = samples_per_file
        self.cache_in_memory = cache_in_memory
        self.p_max = p_max
        self.sigmoid_steepness = sigmoid_steepness
        self.sigmoid_midpoint = sigmoid_midpoint

        self._init_device()

        # ------------------------------------------------------------------ #
        # (1) Locate and load M1, the single-step run trained on D1. Same
        #     architecture-identification convention as train_multistep.py /
        #     inference.py: the fields above name a path in the tree, they do
        #     not describe a model to build from scratch.
        # ------------------------------------------------------------------ #
        self.d1_tag = training_data_tag(d1_training_data_dir)
        self.d2_tag = training_data_tag(d2_training_data_dir)
        if self.d1_tag == self.d2_tag:
            raise ValueError(
                f"D1 and D2 resolve to the same trainData tag '{self.d1_tag}' "
                f"({d1_training_data_dir!r} and {d2_training_data_dir!r}) - there is "
                f"no second domain to merge, and the merged run would collide with "
                f"the source run's own directory."
            )

        source_dir, single_name, _multi_name, info_name = neural_model_path(
            resol=(nlat, nlon), n_future=n_future, num_layers=num_layers,
            embed_dim=embed_dim, pos_embed=pos_embed, trainData_path=self.d1_tag,
            grid=grid, normalization_layer=normalization_layer, loss_type=loss_type,
            index=index,
        )
        source_ckpt_path = os.path.join(source_dir, single_name)
        source_info_path = os.path.join(source_dir, info_name)
        if not os.path.isfile(source_ckpt_path) or not os.path.isfile(source_info_path):
            raise FileNotFoundError(
                f"No pretrained single-step model (M1) found at {source_dir} "
                f"(expected {single_name} and {info_name}). Train it first with "
                f"`make train_single` on D1."
            )
        with open(source_info_path, "r", encoding="utf-8") as f:
            self.source_info = json.load(f)
        self.source_ckpt_path = source_ckpt_path
        print(f"📦 Fine-tuning on top of M1: {source_ckpt_path}")

        # Architecture comes from M1's OWN record, never from this config -
        # the weights loaded below have to land in an identically-shaped
        # model. residual_prediction/target_mode in particular add no
        # parameters, so a mismatch would not even fail load_state_dict();
        # it would silently train against the wrong target framing (see
        # train_multistep.py's identical note).
        self.scale_factor = self.source_info.get("scale_factor", 1)
        self.residual_prediction = self.source_info.get("residual_prediction", True)
        self.target_mode = self.source_info.get("target_mode", "absolute")
        self.inner_skip = self.source_info.get("inner_skip", "none")
        self.hard_thresholding_fraction = self.source_info.get("hard_thresholding_fraction", 1.0)

        # M1's non-dimensionalization scales drive BOTH domains (see
        # SWEDataset's T/U override): the fine-tuned weights keep the input
        # scaling they were trained under, and inference.py reconstructs
        # physical units from the single T/U recorded below - so there is
        # exactly one admissible scale for the merged model, and it is M1's.
        self.T = self.source_info["T"]
        self.U = self.source_info["U"]
        self.h_avg = self.source_info.get("h_avg")
        self.h_amp = self.source_info.get("h_amp")

        # ------------------------------------------------------------------ #
        # (2) Both domains, non-dimensionalized identically, split per-domain
        #     so each keeps its own held-out validation set.
        # ------------------------------------------------------------------ #
        self.d1 = self._build_domain(d1_training_data_dir, "D1")
        self.d2 = self._build_domain(d2_training_data_dir, "D2")
        self._check_domains_compatible()

        validation_split = 0.15
        d1_train, self.d1_val = torch.utils.data.random_split(
            self.d1, [1 - validation_split, validation_split])
        d2_train, self.d2_val = torch.utils.data.random_split(
            self.d2, [1 - validation_split, validation_split])
        self.mixture = DomainMixtureDataset(base=d1_train, new=d2_train)

        self.true_interval_minutes = self.d1.true_interval_minutes
        self.solver = self.d1.solver

        # ------------------------------------------------------------------ #
        # (3) Rebuild M1's architecture exactly and load its weights.
        # ------------------------------------------------------------------ #
        self.model = SFNO(
            img_size=(nlat, nlon), grid=grid,
            num_layers=num_layers, scale_factor=self.scale_factor, embed_dim=embed_dim,
            residual_prediction=self.residual_prediction,
            inner_skip=self.inner_skip,
            hard_thresholding_fraction=self.hard_thresholding_fraction,
            pos_embed=pos_embed, use_mlp=True,
            normalization_layer=self.normalization_layer,
        ).to(self.device)
        source_state = torch.load(source_ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(source_state["model_state_dict"])
        print(f"✅ Loaded M1 weights (best val_loss on D1 was "
              f"{source_state.get('val_loss', float('nan')):.6e})")

        # sourced from M1's own record for the same reason the architecture
        # is: it names what those weights were actually optimized against.
        self.loss = LOSS_FUNCTIONS[self.source_info.get("loss_type", loss_type)]

    def _build_domain(self, training_data_dir, label):
        print(f"💿 Loading {label} from {training_data_dir}")
        return SWEDataset(
            simulation_data_dir=training_data_dir, n_future=self.n_future,
            mode=self.target_mode, samples_per_file=self.samples_per_file,
            cache_in_memory=self.cache_in_memory,
            # both domains share M1's scales - see the T/U note in __init__
            T=self.T, U=self.U,
        )

    def _check_domains_compatible(self):
        """Both domains must be the same spatial resolution as M1 and share a
        frame cadence. Resolution is a hard architectural constraint (the SFNO
        is built for one img_size). Cadence matters because `n_future` frames
        is what fixes the model's physical lead time: mixing domains whose
        frames are spaced differently would train one network on two different
        forecast horizons while recording only one true_interval_minutes in
        model_info.json, so inference would mislabel its own rollout times.
        """
        for label, ds in (("D1", self.d1), ("D2", self.d2)):
            if (ds.solver.nlat, ds.solver.nlon) != (self.nlat, self.nlon):
                raise ValueError(
                    f"{label} ('{ds.simulation_data_dir}') has resolution "
                    f"({ds.solver.nlat}, {ds.solver.nlon}), which does not match M1's "
                    f"({self.nlat}, {self.nlon})."
                )
        d1_cadence, d2_cadence = self.d1.true_interval_minutes, self.d2.true_interval_minutes
        if not math.isclose(d1_cadence, d2_cadence, rel_tol=1e-6):
            raise ValueError(
                f"D1 and D2 have different frame cadences ({d1_cadence:.4f} vs "
                f"{d2_cadence:.4f} min): n_future={self.n_future} would mean a different "
                f"physical lead time in each domain, and only one value can be recorded "
                f"in the merged model's model_info.json."
            )

    @staticmethod
    def _next_run_index(base_dir):
        """Smallest non-negative integer whose sub-directory does not yet
        exist under ``base_dir`` (the 0/, 1/, 2/, ... training-config slots).
        Mirrors train_singlestep.py's identical helper."""
        if not os.path.isdir(base_dir):
            return 0
        existing = {int(d) for d in os.listdir(base_dir)
                    if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))}
        i = 0
        while i in existing:
            i += 1
        return i

    def _history_columns(self):
        cols = ["epoch", "p_d2_scheduled", "p_d2_realized", "train_loss",
                "val_loss_d1", "val_loss_d2", "val_loss_mean"]
        cols += [f"train_loss_{name}" for name in self.CHANNEL_NAMES]
        cols += [f"val_loss_d1_{name}" for name in self.CHANNEL_NAMES]
        cols += [f"val_loss_d2_{name}" for name in self.CHANNEL_NAMES]
        cols += ["lr", "gpu_utillization", "per_epoch_time"]
        return cols

    def _setup_run(self, epochs, lr, batch_size, weight_decay, warmup_epochs,
                   warmup_start_factor, scheduler_type, restart, restart_period,
                   restart_mult, validation_cadence):
        """Claim a fresh `trainData_(<D1>+<D2>)/<index>/` slot and write the
        run's model_info.json / log files.

        The merged run deliberately gets its own trainData node rather than
        another index under D1's: the two are not interchangeable training
        configurations of one model, they are models fit to different data,
        and inference.py selects a checkpoint by exactly this tag.
        """
        self.merged_tag = f"{self.d1_tag}+{self.d2_tag}"
        model_path_kwargs = dict(
            resol=(self.nlat, self.nlon),
            n_future=self.n_future,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            pos_embed=self.pos_embed,
            trainData_path=self.merged_tag,
            grid=self.grid,
            normalization_layer=self.normalization_layer,
            loss_type=self.loss_type,
        )
        probe_dir, *_ = neural_model_path(index=0, **model_path_kwargs)
        base_ckpt_dir = os.path.dirname(probe_dir)
        self.run_index = self._next_run_index(base_ckpt_dir)

        run_ckpt_dir, single_name, _multi_name, info_name = neural_model_path(
            index=self.run_index, **model_path_kwargs)
        run_log_dir = run_ckpt_dir.replace(self.CHECKPOINT_ROOT, self.LOG_ROOT, 1)

        self.task_name = os.path.relpath(run_ckpt_dir, self.CHECKPOINT_ROOT)
        print(f" task_name = {self.task_name}".center(60))

        os.makedirs(run_ckpt_dir, exist_ok=True)
        os.makedirs(run_log_dir, exist_ok=True)

        self.info_file = os.path.join(run_ckpt_dir, info_name)
        # saved as the ordinary single-step checkpoint name: the merged model
        # IS a single-step model, so inference.py picks it up unchanged.
        self.checkpoint = os.path.join(run_ckpt_dir, single_name)
        self.log_file = os.path.join(run_log_dir, "train_log_domain_merge.csv")
        self.dashboard_plot = os.path.join(run_log_dir, "dashboard_plot_domain_merge.png")

        schedule_preview = ", ".join(
            f"ep{e}:{sigmoid_domain_probability(e, epochs, self.p_max, self.sigmoid_steepness, self.sigmoid_midpoint):.0%}"
            for e in sorted({0, epochs // 4, epochs // 2, (3 * epochs) // 4, epochs - 1})
        )
        print_in_box({
            "title": "SFNO Domain-Merge Configuration",
            "lines": [
                f"M1 (source)  = {self.source_ckpt_path}",
                f"D1 (base)    = {self.d1_training_data_dir}  [{len(self.d1)} samples, tag '{self.d1_tag}']",
                f"D2 (merging) = {self.d2_training_data_dir}  [{len(self.d2)} samples, tag '{self.d2_tag}']",
                f"merged tag   = {self.merged_tag} -> index {self.run_index}/",
                f"nlat = {self.nlat} | nlon = {self.nlon} | grid = {self.grid}",
                f"n_future = {self.n_future} : M(D_t) = D_(t+n_future*{self.true_interval_minutes:.2f}min)",
                f"num_layers = {self.num_layers} | embed_dim = {self.embed_dim} | scale_factor = {self.scale_factor}",
                f"residual_prediction = {self.residual_prediction} | target_mode = {self.target_mode} (from M1)",
                f"inner_skip = {self.inner_skip} | hard_thresholding_fraction = {self.hard_thresholding_fraction} (from M1)",
                f"normalization_layer = {self.normalization_layer} | loss_type = {self.loss_type}",
                f"T = {self.T:.2f} | U = {self.U:.2f} (M1's scales, applied to BOTH domains)",
                f"p_max = {self.p_max:.0%} : final D2 sampling probability",
                f"sigmoid_steepness = {self.sigmoid_steepness} | sigmoid_midpoint = {self.sigmoid_midpoint}",
                f"schedule: {schedule_preview}",
                f"epochs = {epochs} | lr = {lr} | batch_size = {batch_size}",
                f"weight_decay = {weight_decay} | validation_cadence = {validation_cadence}",
                f"warmup_epochs = {warmup_epochs} (start_factor {warmup_start_factor})",
                f"scheduler_type = {scheduler_type} : cosine | constant, post-warmup LR schedule",
                f"restart = {restart} (T_0={restart_period}, T_mult={restart_mult}) (cosine only)",
                f"no_height_loss = {self.no_height_loss}",
                f"train (mixture) / val D1 / val D2 = "
                f"{len(self.mixture)} / {len(self.d1_val)} / {len(self.d2_val)}",
            ],
        })

        model_info = {
            "type": "neural_operator",
            # --- architecture: inherited from M1 so this checkpoint is
            # --- loadable by exactly the same reconstruction path.
            "nlat": self.nlat,
            "nlon": self.nlon,
            "grid": self.grid,
            "n_future": self.n_future,
            "num_layers": self.num_layers,
            "pos_embed": self.pos_embed,
            "scale_factor": self.scale_factor,
            "embed_dim": self.embed_dim,
            "residual_prediction": self.residual_prediction,
            "inner_skip": self.inner_skip,
            "target_mode": self.target_mode,
            "hard_thresholding_fraction": self.hard_thresholding_fraction,
            "normalization_layer": self.normalization_layer,
            "loss_type": self.loss_type,
            "run_index": self.run_index,
            # --- provenance: what this model was actually fit to. Note
            # --- `training_data` is D1's directory (the domain whose
            # --- reference numerical solver inference.py recovers from this
            # --- string - see numerical_model_info_from_neural_opeartor_model_info);
            # --- both domains are recorded explicitly alongside it.
            "training_data": self.d1_training_data_dir,
            "d1_training_data": self.d1_training_data_dir,
            "d2_training_data": self.d2_training_data_dir,
            "d1_tag": self.d1_tag,
            "d2_tag": self.d2_tag,
            "source_checkpoint": self.source_ckpt_path,
            "source_index": self.source_index,
            "dataset_name": self.d1.dataset_name,
            "pressure": self.d1.pressure,
            "d2_dataset_name": self.d2.dataset_name,
            "d2_pressure": self.d2.pressure,
            "h_avg": self.h_avg,
            "h_amp": self.h_amp,
            "U": self.U,
            "T": self.T,
            "true_interval_minutes": self.true_interval_minutes,
            # --- curriculum
            "p_max": self.p_max,
            "sigmoid_steepness": self.sigmoid_steepness,
            "sigmoid_midpoint": self.sigmoid_midpoint,
            # --- optimization
            "samples_per_file": self.samples_per_file,
            "cache_in_memory": self.cache_in_memory,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "warmup_start_factor": warmup_start_factor,
            "scheduler_type": scheduler_type,
            "restart": restart,
            "restart_period": restart_period,
            "restart_mult": restart_mult,
            "validation_cadence": validation_cadence,
            "no_height_loss": self.no_height_loss,
        }
        with open(self.info_file, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=4)

        with open(self.log_file, "w", newline="") as f:
            csv.writer(f).writerow(self._history_columns())
        self.history = {col: [] for col in self._history_columns()}
        self.best_val_loss = math.inf

    def _update_log(self, epoch, p_scheduled, p_realized, train_loss, train_loss_channels,
                    val_d1, val_d1_channels, val_d2, val_d2_channels, val_mean,
                    lr, gpu_util, epoch_time, total_epochs, is_validation):
        h = self.history
        h["epoch"].append(epoch)
        h["p_d2_scheduled"].append(p_scheduled)
        h["p_d2_realized"].append(p_realized)
        h["train_loss"].append(train_loss)
        h["val_loss_d1"].append(val_d1)
        h["val_loss_d2"].append(val_d2)
        h["val_loss_mean"].append(val_mean)
        for i, name in enumerate(self.CHANNEL_NAMES):
            h[f"train_loss_{name}"].append(train_loss_channels[i])
            h[f"val_loss_d1_{name}"].append(val_d1_channels[i])
            h[f"val_loss_d2_{name}"].append(val_d2_channels[i])
        h["lr"].append(lr)
        h["gpu_utillization"].append(gpu_util)
        h["per_epoch_time"].append(epoch_time)

        def fmt(v):
            return "" if v is None else f"{v:.6e}"

        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{p_scheduled:.6f}", f"{p_realized:.6f}", f"{train_loss:.6e}",
                fmt(val_d1), fmt(val_d2), fmt(val_mean),
                *[f"{v:.6e}" for v in train_loss_channels],
                *[fmt(v) for v in val_d1_channels],
                *[fmt(v) for v in val_d2_channels],
                f"{lr:.6e}", f"{gpu_util:.2f}", f"{epoch_time:.3f}",
            ])

        avg_epoch_time = sum(h["per_epoch_time"]) / len(h["per_epoch_time"])
        eta_sec = avg_epoch_time * max(total_epochs - (epoch + 1), 0)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        val_str = ""
        if val_mean is not None:
            val_str = f"val(D1={val_d1:.4e} D2={val_d2:.4e} mean={val_mean:.4e}) "
        print(f"[epoch {epoch + 1}/{total_epochs} | p_D2={p_scheduled:.1%} "
              f"(realized {p_realized:.1%})] train={train_loss:.4e} {val_str}"
              f"lr={lr:.2e} gpu={gpu_util:.0f} epoch_time={epoch_time:.1f}s ETA={eta_str}")

        if is_validation:
            self._draw_dashboard()

    def _draw_dashboard(self):
        h = self.history
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # (0,0) the headline: is D2 being learned without losing D1?
        ax = axes[0, 0]
        ax.plot(h["epoch"], h["train_loss"], label="train (mixture)", color="tab:blue")
        for key, label, color in (("val_loss_d1", "val D1 (forgetting)", "tab:red"),
                                  ("val_loss_d2", "val D2 (acquisition)", "tab:green"),
                                  ("val_loss_mean", "val mean (checkpointed)", "tab:orange")):
            eps = [e for e, v in zip(h["epoch"], h[key]) if v is not None]
            vals = [v for v in h[key] if v is not None]
            if vals:
                ax.plot(eps, vals, label=label, color=color, marker="o", markersize=3)
        ax.set_title("Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{self.loss_type} loss")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

        # (0,1) the curriculum actually applied
        ax = axes[0, 1]
        ax.plot(h["epoch"], h["p_d2_scheduled"], label="scheduled", color="tab:purple")
        ax.plot(h["epoch"], h["p_d2_realized"], label="realized", color="tab:gray",
                linestyle="dotted")
        ax.set_title("D2 sampling probability (sigmoid curriculum)")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

        # (1,0) per-channel, both domains on shared axes for direct comparison
        ax = axes[1, 0]
        for i, name in enumerate(self.CHANNEL_NAMES):
            color = f"C{i}"
            for key, style, tag in ((f"val_loss_d1_{name}", "solid", "D1"),
                                    (f"val_loss_d2_{name}", "dashed", "D2")):
                eps = [e for e, v in zip(h["epoch"], h[key]) if v is not None]
                vals = [v for v in h[key] if v is not None]
                if vals:
                    ax.plot(eps, vals, color=color, linestyle=style, label=f"{name} ({tag})")
        ax.set_title("Per-channel validation loss")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)

        axes[1, 1].plot(h["epoch"], h["lr"], color="tab:green")
        axes[1, 1].set_title("Learning rate")
        axes[1, 1].set_xlabel("epoch")

        fig.suptitle(self.task_name)
        fig.tight_layout()
        fig.savefig(self.dashboard_plot, dpi=120)
        plt.close(fig)

    def _run_epoch(self, loader, train, optimizer=None):
        """One pass over `loader`; returns (mean_loss, per_channel_loss,
        realized_d2_fraction). The third value is only meaningful for the
        mixture loader (the validation loaders are single-domain by
        construction) and measures what the weights actually saw, as opposed
        to the probability that was scheduled.
        """
        self.model.train(train)
        total_loss, n_batches = 0.0, 0
        total_channel_loss = torch.zeros(len(self.CHANNEL_NAMES), device=self.device)
        n_d2, n_samples = 0, 0
        torch.set_grad_enabled(train)
        for inp, tar, domain_id in loader:
            inp = inp.to(self.device)
            tar = tar.to(self.device)

            prd = self.model(inp)
            channel_loss = self.loss(self.solver, prd, tar, relative=True,
                                     squared=False, reduce_channels=False)
            loss = channel_loss[1:].mean() if self.no_height_loss else channel_loss.mean()

            if train:
                assert optimizer is not None, "optimizer required when train=True"
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_channel_loss += channel_loss.detach()
            n_batches += 1
            if torch.is_tensor(domain_id):
                n_d2 += int((domain_id == DomainMixtureDataset.NEW_DOMAIN_ID).sum())
                n_samples += domain_id.numel()
        torch.set_grad_enabled(True)
        n_batches = max(n_batches, 1)
        realized = (n_d2 / n_samples) if n_samples else 0.0
        return total_loss / n_batches, (total_channel_loss / n_batches).tolist(), realized

    def train(self,
        batch_size=16,
        lr=1e-4,
        epochs=100,
        weight_decay=0.01,
        validation_cadence=3,
        warmup_epochs=0,
        warmup_start_factor=0.01,
        scheduler_type="cosine",
        restart=False,
        restart_period=None,
        restart_mult=1,
        compile=False,
        no_height_loss=False,
        ):
        self.no_height_loss = no_height_loss

        self._setup_run(epochs, lr, batch_size, weight_decay, warmup_epochs,
                        warmup_start_factor, scheduler_type, restart,
                        restart_period, restart_mult, validation_cadence)

        # Mixture is reshuffled each epoch; the two validation loaders are
        # fixed, single-domain, and never shuffled so their losses are
        # directly comparable epoch to epoch.
        train_loader = DataLoader(self.mixture, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        d1_val_loader = DataLoader(self.d1_val, batch_size=batch_size,
                                   shuffle=False, num_workers=0)
        d2_val_loader = DataLoader(self.d2_val, batch_size=batch_size,
                                   shuffle=False, num_workers=0)

        if compile:
            try:
                self.model = torch.compile(self.model)
            except RuntimeError as e:
                print(f"⚠️  torch.compile unavailable, running eager: {e}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = self._build_scheduler(
            optimizer, epochs, warmup_epochs, warmup_start_factor,
            scheduler_type, restart, restart_period, restart_mult)

        for ep in tqdm.tqdm(range(epochs), desc="epochs", total=epochs):
            epoch_start = time.perf_counter()

            # The curriculum: advance the mixture BEFORE the epoch runs, so
            # epoch 0 is pure D1 (p=0) and the final epoch is the full p_max
            # blend - see sigmoid_domain_probability.
            p_scheduled = sigmoid_domain_probability(
                ep, epochs, self.p_max, self.sigmoid_steepness, self.sigmoid_midpoint)
            self.mixture.set_new_domain_probability(p_scheduled)

            train_loss, train_loss_channels, p_realized = self._run_epoch(
                train_loader, train=True, optimizer=optimizer)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            is_validation = ((ep + 1) % validation_cadence == 0) or (ep == epochs - 1)
            val_d1 = val_d2 = val_mean = None
            val_d1_channels = [None] * len(self.CHANNEL_NAMES)
            val_d2_channels = [None] * len(self.CHANNEL_NAMES)
            if is_validation:
                val_d1, val_d1_channels, _ = self._run_epoch(d1_val_loader, train=False)
                val_d2, val_d2_channels, _ = self._run_epoch(d2_val_loader, train=False)
                # The merged model is selected on the JOINT objective: a
                # checkpoint that is excellent on one domain and broken on the
                # other is precisely the outcome this pipeline exists to
                # avoid, and would win on either domain's loss alone.
                val_mean = 0.5 * (val_d1 + val_d2)
                if val_mean < self.best_val_loss:
                    self.best_val_loss = val_mean
                    model_to_save = getattr(self.model, "_orig_mod", self.model)
                    torch.save({
                        "epoch": ep,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_mean,
                        "val_loss_d1": val_d1,
                        "val_loss_d2": val_d2,
                        "p_d2": p_scheduled,
                        "task_name": self.task_name,
                    }, self.checkpoint)

            epoch_time = time.perf_counter() - epoch_start
            self._update_log(ep, p_scheduled, p_realized, train_loss, train_loss_channels,
                             val_d1, val_d1_channels, val_d2, val_d2_channels, val_mean,
                             current_lr, self._gpu_utilization(), epoch_time,
                             epochs, is_validation)

        total_time = time.perf_counter() - self.start_time
        print(f"㊣ ㊣ ㊣ Finished Domain-Merge Fine-Tuning! Total time {total_time:.3f}s ㊣ ㊣ ㊣")
        print(f"Best joint (mean) validation loss = {self.best_val_loss:.6e} -> {self.checkpoint}")


def main():
    DEFAULT_CONFIG = {
        # D1: the domain M1 was trained on. Its trainData tag also LOCATES
        # M1 in the checkpoint tree, together with the architecture fields
        # below - so these must name an existing `make train_single` run.
        "d1_training_data_dir": None,
        # D2: the domain being merged in.
        "d2_training_data_dir": None,
        # --- which trained M1 to fine-tune (identifies a path, see above) ---
        "nlat": 128,
        "nlon": 256,
        "n_future": 1,
        "num_layers": 4,
        "embed_dim": 128,
        "pos_embed": "learnable lat",
        "grid": "equiangular",
        "normalization_layer": "none",
        "loss_type": "grid",
        "index": 0,
        # --- curriculum: D2's sampling probability, 0 -> p_max by sigmoid ---
        "p_max": 0.5,
        "sigmoid_steepness": 10.0,
        "sigmoid_midpoint": 0.5,
        # --- optimization (same knobs/semantics as train_single) ---
        # lr defaults an order of magnitude below train_single's: this starts
        # from converged weights, and the whole point of the ramp is to avoid
        # large early updates that overwrite D1.
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 16,
        "weight_decay": 0.01,
        "validation_cadence": 3,
        "warmup_epochs": 0,
        "warmup_start_factor": 0.001,
        "scheduler_type": "cosine",   # cosine | constant, post-warmup LR schedule
        "restart": False,
        "restart_period": 20,
        "restart_mult": 2,
        "samples_per_file": 30,
        "cache_in_memory": True,
        "compile": False,
        "no_height_loss": False,
    }

    raw = load_raw_config("train_domain_merge", DEFAULT_CONFIG)

    for key in ("d1_training_data_dir", "d2_training_data_dir"):
        if not raw[key]:
            raise ValueError(f"train_domain_merge.{key} is required.")
    raw["lr"] = float(raw["lr"])
    raw["weight_decay"] = float(raw["weight_decay"])
    raw["warmup_start_factor"] = float(raw["warmup_start_factor"])
    raw["p_max"] = float(raw["p_max"])
    raw["sigmoid_steepness"] = float(raw["sigmoid_steepness"])
    raw["sigmoid_midpoint"] = float(raw["sigmoid_midpoint"])
    raw["epochs"] = int(raw["epochs"])
    raw["batch_size"] = int(raw["batch_size"])
    raw["samples_per_file"] = int(raw["samples_per_file"])
    raw["cache_in_memory"] = bool(raw["cache_in_memory"])
    raw["index"] = int(raw["index"])
    if not 0.0 <= raw["p_max"] <= 1.0:
        raise ValueError(f"train_domain_merge.p_max must be in [0, 1], got {raw['p_max']}")

    cfg = SimpleNamespace(**raw)

    trainer = SFNODomainMergeTrainer(
        d1_training_data_dir=cfg.d1_training_data_dir,
        d2_training_data_dir=cfg.d2_training_data_dir,
        nlat=cfg.nlat,
        nlon=cfg.nlon,
        n_future=cfg.n_future,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        pos_embed=cfg.pos_embed,
        grid=cfg.grid,
        index=cfg.index,
        normalization_layer=cfg.normalization_layer,
        loss_type=cfg.loss_type,
        samples_per_file=cfg.samples_per_file,
        cache_in_memory=cfg.cache_in_memory,
        p_max=cfg.p_max,
        sigmoid_steepness=cfg.sigmoid_steepness,
        sigmoid_midpoint=cfg.sigmoid_midpoint,
    )

    trainer.train(
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        validation_cadence=cfg.validation_cadence,
        warmup_epochs=cfg.warmup_epochs,
        warmup_start_factor=cfg.warmup_start_factor,
        scheduler_type=cfg.scheduler_type,
        restart=cfg.restart,
        restart_period=cfg.restart_period,
        restart_mult=cfg.restart_mult,
        compile=cfg.compile,
        no_height_loss=bool(cfg.no_height_loss),
    )


if __name__ == "__main__":
    main()
