#!/usr/bin/env python3
"""
CIFAR-10 FINAL V2: calibrated 3-seed paired shock validation
============================================================

Purpose
-------
Paper-oriented validation of optimizer response to an isolated gradient shock.

Key improvements over the previous version
------------------------------------------
1. Lightweight learning-rate calibration on TRAINING-DATA validation split only.
2. Selected learning rates are frozen before all shock experiments.
3. Explicit pre-shock parameter-equality verification for every matched pair.
4. Adds RelativeCausalUpdateToControlUpdate:
       ||Δθ_shock - Δθ_control|| / ||Δθ_control||
   alongside the injected-shock-normalized TransmissionRatio.
5. Reports mean, sample SD, median, min, max across 3 independent seeds.
   No bootstrap CI is used for n=3.
6. Final CIFAR probe set is separate from LR calibration.
7. Short-horizon causal analysis remains the primary deep-learning validation.

Default workload
----------------
Calibration:
    SGD:     [0.01, 0.03, 0.05, 0.10]
    AdaGrad: [0.003, 0.01, 0.03, 0.05]
    Adam:    [0.0003, 0.001, 0.003]
    3 epochs, 10k calibration-train images, 2k calibration-validation images.

Final:
    3 optimizers x 3 seeds x {control, shock} = 18 trajectories
    20k final-training images, 8 epochs, 2k fixed CIFAR-10 test probe images
    shock at step 500, 20x local classifier-gradient norm
    short-horizon causal window = 300 steps

Run
---
    python cifar10_final_v2_calibrated_3seed.py

Outputs
-------
cifar10_final_v2_results/
    calibration_results.csv
    selected_learning_rates.json
    per_run_summary.csv
    aggregate_summary.csv
    pre_shock_equality_checks.csv
    *_paired_trace.csv
    *_aggregate_causal_trace.csv
    figures/
    config.json
"""

import argparse
import csv
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class SmallCIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, batch_size: int, seed: int, num_workers: int, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


# ---------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------

def make_optimizer(name: str, model: nn.Module, lr: float):
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    if name == "AdaGrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, eps=1e-10)
    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.numel()

    model.train()
    return loss_sum / n, correct / n


@torch.no_grad()
def parameter_vector(model):
    return torch.cat([
        p.detach().reshape(-1).cpu()
        for p in model.parameters()
    ])


# ---------------------------------------------------------------------
# Deterministic dataset partition
# ---------------------------------------------------------------------

def load_cifar(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)

    try:
        train_full = datasets.CIFAR10(
            root=root, train=True, download=False, transform=transform
        )
        test_full = datasets.CIFAR10(
            root=root, train=False, download=False, transform=transform
        )
        print(f"Using local CIFAR-10 from: {root.resolve()}")
    except RuntimeError:
        print("Local CIFAR-10 not found. Downloading once...")
        train_full = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        test_full = datasets.CIFAR10(
            root=root, train=False, download=False, transform=transform
        )

    # Fixed, disjoint training partitions.
    rng = np.random.default_rng(args.partition_seed)
    perm = rng.permutation(len(train_full))

    n_final = args.train_limit
    n_cal_train = args.calibration_train_limit
    n_cal_val = args.calibration_val_size
    required = n_final + n_cal_train + n_cal_val

    if required > len(train_full):
        raise ValueError(
            f"Requested disjoint train/calibration partitions need {required} "
            f"images but CIFAR-10 train has {len(train_full)}."
        )

    final_idx = perm[:n_final]
    cal_train_idx = perm[n_final:n_final + n_cal_train]
    cal_val_idx = perm[n_final + n_cal_train:required]

    final_train = Subset(train_full, final_idx.tolist())
    cal_train = Subset(train_full, cal_train_idx.tolist())
    cal_val = Subset(train_full, cal_val_idx.tolist())

    probe_size = min(args.probe_size, len(test_full))
    probe = Subset(test_full, list(range(probe_size)))

    return final_train, cal_train, cal_val, probe


# ---------------------------------------------------------------------
# LR calibration
# ---------------------------------------------------------------------

def run_calibration_candidate(
    optimizer_name,
    lr,
    args,
    cal_train,
    cal_val_loader,
):
    seed = args.calibration_seed
    set_seed(seed)

    model = SmallCIFARCNN()
    optimizer = make_optimizer(optimizer_name, model, lr)

    loader = make_loader(
        cal_train,
        batch_size=args.batch_size,
        seed=seed + 111,
        num_workers=args.num_workers,
        shuffle=True,
    )

    t0 = time.time()

    for _epoch in range(args.calibration_epochs):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

    val_loss, val_acc = evaluate(model, cal_val_loader)

    return {
        "Optimizer": optimizer_name,
        "LearningRate": lr,
        "CalibrationEpochs": args.calibration_epochs,
        "CalibrationTrainSize": len(cal_train),
        "CalibrationValSize": len(cal_val_loader.dataset),
        "ValidationLoss": val_loss,
        "ValidationAccuracy": val_acc,
        "RuntimeSeconds": time.time() - t0,
    }


def calibrate_learning_rates(args, cal_train, cal_val):
    val_loader = make_loader(
        cal_val,
        batch_size=args.batch_size,
        seed=args.calibration_seed,
        num_workers=args.num_workers,
        shuffle=False,
    )

    grids = {
        "SGD": args.sgd_lr_grid,
        "AdaGrad": args.adagrad_lr_grid,
        "Adam": args.adam_lr_grid,
    }

    rows = []
    selected = {}

    print("\n" + "=" * 78)
    print("LEARNING-RATE CALIBRATION")
    print("=" * 78)
    print("Selection rule: highest validation accuracy; tie -> lower validation loss.")
    print("Calibration uses only disjoint CIFAR-10 training-data partitions.\n")

    for opt in ["SGD", "AdaGrad", "Adam"]:
        candidates = []

        for lr in grids[opt]:
            print(f"Calibrating {opt}, lr={lr:g} ...")
            row = run_calibration_candidate(
                opt, lr, args, cal_train, val_loader
            )
            rows.append(row)
            candidates.append(row)

            print(
                f"  val_acc={row['ValidationAccuracy']:.4f} | "
                f"val_loss={row['ValidationLoss']:.4f} | "
                f"time={row['RuntimeSeconds']:.1f}s"
            )

        # Deterministic selection.
        best = sorted(
            candidates,
            key=lambda r: (-r["ValidationAccuracy"], r["ValidationLoss"])
        )[0]

        selected[opt] = float(best["LearningRate"])
        print(
            f"SELECTED {opt}: lr={selected[opt]:g} "
            f"(acc={best['ValidationAccuracy']:.4f}, "
            f"loss={best['ValidationLoss']:.4f})\n"
        )

    return rows, selected


# ---------------------------------------------------------------------
# Shock
# ---------------------------------------------------------------------

def inject_relative_shock(model, multiplier, direction_seed):
    """
    Inject shock into classifier.weight gradient.

    shock norm = multiplier * ||current classifier gradient||_2

    The same random unit direction is used for a given seed across optimizers,
    while the shock magnitude is optimizer-local and therefore relative.
    """
    p = model.classifier.weight
    if p.grad is None:
        raise RuntimeError("classifier.weight gradient is missing at shock step.")

    g = p.grad
    base_norm = torch.linalg.vector_norm(g.detach()).item()

    gen = torch.Generator()
    gen.manual_seed(direction_seed)
    u = torch.randn(g.shape, generator=gen, dtype=g.dtype)
    u = u / (torch.linalg.vector_norm(u) + 1e-20)

    injected_norm = multiplier * max(base_norm, 1e-12)
    g.add_(injected_norm * u)

    return base_norm, injected_norm


# ---------------------------------------------------------------------
# Evaluation schedule
# ---------------------------------------------------------------------

def build_eval_steps(total_steps, shock_step, sparse_interval):
    dense_offsets = [
        -100, -50, -20, -10, -5, -2, -1,
        0, 1, 2, 5, 10, 20, 30, 50,
        75, 100, 150, 200, 250, 300
    ]

    steps = {0, total_steps, shock_step}

    for off in dense_offsets:
        s = shock_step + off
        if 0 <= s <= total_steps:
            steps.add(s)

    for s in range(sparse_interval, total_steps + 1, sparse_interval):
        steps.add(s)

    return sorted(steps)


# ---------------------------------------------------------------------
# Final paired trajectories
# ---------------------------------------------------------------------

def run_trajectory(
    optimizer_name,
    lr,
    seed,
    shocked,
    args,
    train_dataset,
    probe_loader,
    initial_state,
    eval_steps_set,
    control_shock_update_vector=None,
    control_pre_shock_vector=None,
):
    set_seed(seed)

    model = SmallCIFARCNN()
    model.load_state_dict(deepcopy(initial_state))
    optimizer = make_optimizer(optimizer_name, model, lr)

    loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        seed=seed + 12345,
        num_workers=args.num_workers,
        shuffle=True,
    )

    steps = []
    losses = []
    accs = []

    l0, a0 = evaluate(model, probe_loader)
    steps.append(0)
    losses.append(l0)
    accs.append(a0)

    shock_update_vector = None
    pre_shock_vector = None
    pre_shock_equality_norm = float("nan")

    meta = {
        "base_classifier_grad_norm": float("nan"),
        "injected_shock_norm": 0.0,
        "immediate_update_norm": float("nan"),
        "causal_update_vector_norm": float("nan"),
        "transmission_ratio": float("nan"),
        "relative_causal_update_to_control_update": float("nan"),
    }

    global_step = 0
    t0 = time.time()

    for _epoch in range(args.epochs):
        for x, y in loader:
            global_step += 1

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            if global_step == args.shock_step:
                # This vector is the parameter state immediately BEFORE
                # applying either control or shocked update at shock_step.
                pre_shock_vector = parameter_vector(model)

                if shocked and control_pre_shock_vector is not None:
                    pre_shock_equality_norm = torch.linalg.vector_norm(
                        pre_shock_vector - control_pre_shock_vector
                    ).item()

                    if pre_shock_equality_norm > args.pre_shock_tolerance:
                        raise RuntimeError(
                            f"Matched-pair pre-shock equality FAILED for "
                            f"{optimizer_name}, seed={seed}: "
                            f"||theta_shock-theta_control||="
                            f"{pre_shock_equality_norm:.3e} > "
                            f"{args.pre_shock_tolerance:.3e}"
                        )

                before = pre_shock_vector

                if shocked:
                    base_norm, injected_norm = inject_relative_shock(
                        model,
                        args.shock_multiplier,
                        seed + 777,
                    )
                    meta["base_classifier_grad_norm"] = base_norm
                    meta["injected_shock_norm"] = injected_norm

                optimizer.step()

                after = parameter_vector(model)
                shock_update_vector = after - before

                update_norm = torch.linalg.vector_norm(
                    shock_update_vector
                ).item()
                meta["immediate_update_norm"] = update_norm

                if shocked and control_shock_update_vector is not None:
                    causal_update = (
                        shock_update_vector - control_shock_update_vector
                    )
                    causal_norm = torch.linalg.vector_norm(
                        causal_update
                    ).item()

                    control_update_norm = torch.linalg.vector_norm(
                        control_shock_update_vector
                    ).item()

                    meta["causal_update_vector_norm"] = causal_norm
                    meta["transmission_ratio"] = (
                        causal_norm /
                        max(meta["injected_shock_norm"], 1e-20)
                    )
                    meta["relative_causal_update_to_control_update"] = (
                        causal_norm /
                        max(control_update_norm, 1e-20)
                    )
            else:
                optimizer.step()

            if global_step in eval_steps_set:
                pl, pa = evaluate(model, probe_loader)
                steps.append(global_step)
                losses.append(pl)
                accs.append(pa)

    if shock_update_vector is None:
        raise RuntimeError(
            f"Shock step {args.shock_step} was never reached."
        )

    if steps[-1] != global_step:
        pl, pa = evaluate(model, probe_loader)
        steps.append(global_step)
        losses.append(pl)
        accs.append(pa)

    return {
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "seed": seed,
        "shocked": shocked,
        "steps": np.asarray(steps, dtype=int),
        "probe_loss": np.asarray(losses, dtype=float),
        "probe_acc": np.asarray(accs, dtype=float),
        "shock_update_vector": shock_update_vector,
        "pre_shock_vector": pre_shock_vector,
        "pre_shock_equality_norm": pre_shock_equality_norm,
        "shock_metadata": meta,
        "runtime_seconds": time.time() - t0,
        "final_probe_loss": float(losses[-1]),
        "final_probe_acc": float(accs[-1]),
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def analyze_pair(control, shock, args):
    if not np.array_equal(control["steps"], shock["steps"]):
        raise RuntimeError(
            "Control and shock evaluation steps do not match."
        )

    steps = control["steps"]
    delta_loss = shock["probe_loss"] - control["probe_loss"]

    shock_idx = np.where(steps == args.shock_step)[0]
    if len(shock_idx) != 1:
        raise RuntimeError(
            "Shock step was not evaluated exactly once."
        )
    shock_idx = int(shock_idx[0])

    horizon_end = args.shock_step + args.short_horizon_steps
    mask = (
        (steps >= args.shock_step)
        & (steps <= horizon_end)
    )

    hsteps = steps[mask]
    hdelta = delta_loss[mask]

    if len(hsteps) == 0:
        raise RuntimeError(
            "No short-horizon evaluations were found."
        )

    abs_hdelta = np.abs(hdelta)
    peak_i = int(np.argmax(abs_hdelta))
    peak_step = int(hsteps[peak_i])

    positive = np.maximum(hdelta, 0.0)

    cumulative_positive = (
        float(np.trapezoid(positive, x=hsteps))
        if len(hsteps) > 1 else 0.0
    )
    cumulative_absolute = (
        float(np.trapezoid(abs_hdelta, x=hsteps))
        if len(hsteps) > 1 else 0.0
    )

    sm = shock["shock_metadata"]
    cm = control["shock_metadata"]

    row = {
        "Optimizer": shock["optimizer"],
        "Seed": shock["seed"],
        "LearningRate": shock["learning_rate"],
        "ShockMultiplier": args.shock_multiplier,
        "ShockStep": args.shock_step,
        "ShortHorizonSteps": args.short_horizon_steps,

        "PreShockParameterDifferenceNorm":
            shock["pre_shock_equality_norm"],

        "ImmediateControlUpdateNorm":
            cm["immediate_update_norm"],
        "ImmediateShockUpdateNorm":
            sm["immediate_update_norm"],
        "CausalUpdateVectorNorm":
            sm["causal_update_vector_norm"],
        "InjectedShockNorm":
            sm["injected_shock_norm"],
        "TransmissionRatio":
            sm["transmission_ratio"],
        "RelativeCausalUpdateToControlUpdate":
            sm["relative_causal_update_to_control_update"],

        "ImmediateProbeLossDeviation":
            float(delta_loss[shock_idx]),

        "ShortHorizonPeakPositiveLossDamage":
            float(np.max(positive)),
        "ShortHorizonPeakAbsoluteLossDeviation":
            float(abs_hdelta[peak_i]),
        "ShortHorizonPeakDeviationStep":
            peak_step,
        "ShortHorizonPeakDelaySteps":
            peak_step - args.shock_step,

        "ShortHorizonCumulativePositiveLossDamage":
            cumulative_positive,
        "ShortHorizonCumulativeAbsoluteLossDeviation":
            cumulative_absolute,

        "ControlFinalProbeLoss":
            control["final_probe_loss"],
        "ShockFinalProbeLoss":
            shock["final_probe_loss"],
        "FinalProbeLossDifference":
            shock["final_probe_loss"] - control["final_probe_loss"],

        "ControlFinalProbeAcc":
            control["final_probe_acc"],
        "ShockFinalProbeAcc":
            shock["final_probe_acc"],
        "FinalProbeAccuracyDifference":
            shock["final_probe_acc"] - control["final_probe_acc"],

        "ControlRuntimeSeconds":
            control["runtime_seconds"],
        "ShockRuntimeSeconds":
            shock["runtime_seconds"],
    }

    return row, delta_loss


# ---------------------------------------------------------------------
# Aggregation: appropriate descriptive statistics for n=3
# ---------------------------------------------------------------------

def finite_stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "N": 0,
            "Mean": np.nan,
            "SampleSD": np.nan,
            "Median": np.nan,
            "Min": np.nan,
            "Max": np.nan,
        }

    return {
        "N": len(x),
        "Mean": float(np.mean(x)),
        "SampleSD": (
            float(np.std(x, ddof=1))
            if len(x) >= 2 else np.nan
        ),
        "Median": float(np.median(x)),
        "Min": float(np.min(x)),
        "Max": float(np.max(x)),
    }


def compute_aggregate_rows(per_run_rows):
    metrics = [
        "TransmissionRatio",
        "RelativeCausalUpdateToControlUpdate",
        "CausalUpdateVectorNorm",
        "ImmediateProbeLossDeviation",
        "ShortHorizonPeakPositiveLossDamage",
        "ShortHorizonPeakAbsoluteLossDeviation",
        "ShortHorizonPeakDelaySteps",
        "ShortHorizonCumulativePositiveLossDamage",
        "ShortHorizonCumulativeAbsoluteLossDeviation",
        "FinalProbeAccuracyDifference",
        "FinalProbeLossDifference",
    ]

    rows = []

    for opt in ["SGD", "AdaGrad", "Adam"]:
        group = [
            r for r in per_run_rows
            if r["Optimizer"] == opt
        ]

        for metric in metrics:
            stats = finite_stats([
                r[metric] for r in group
            ])

            rows.append({
                "Optimizer": opt,
                "Metric": metric,
                "NSeeds": stats["N"],
                "Mean": stats["Mean"],
                "SampleSD": stats["SampleSD"],
                "Median": stats["Median"],
                "Min": stats["Min"],
                "Max": stats["Max"],
            })

    return rows


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_pair_trace(path, control, shock, delta_loss):
    rows = []

    for i, step in enumerate(control["steps"]):
        rows.append({
            "Step": int(step),
            "ControlProbeLoss":
                float(control["probe_loss"][i]),
            "ShockProbeLoss":
                float(shock["probe_loss"][i]),
            "DeltaProbeLoss":
                float(delta_loss[i]),
            "AbsoluteDeltaProbeLoss":
                float(abs(delta_loss[i])),
            "ControlProbeAcc":
                float(control["probe_acc"][i]),
            "ShockProbeAcc":
                float(shock["probe_acc"][i]),
        })

    write_csv(path, rows)


def make_aggregate_trace(out_dir, optimizer_name, trace_bank):
    steps = trace_bank[0]["steps"]

    for item in trace_bank[1:]:
        if not np.array_equal(steps, item["steps"]):
            raise RuntimeError(
                f"Trace step mismatch for {optimizer_name}."
            )

    matrix = np.vstack([
        item["delta"]
        for item in trace_bank
    ])

    rows = []

    for j, step in enumerate(steps):
        vals = matrix[:, j]

        rows.append({
            "Step": int(step),
            "MeanDeltaProbeLoss":
                float(np.mean(vals)),
            "SampleSDDeltaProbeLoss":
                float(np.std(vals, ddof=1))
                if len(vals) >= 2 else np.nan,
            "MedianDeltaProbeLoss":
                float(np.median(vals)),
            "MinDeltaProbeLoss":
                float(np.min(vals)),
            "MaxDeltaProbeLoss":
                float(np.max(vals)),
            "MedianAbsoluteDeltaProbeLoss":
                float(np.median(np.abs(vals))),
        })

    write_csv(
        out_dir /
        f"{optimizer_name.lower()}_aggregate_causal_trace.csv",
        rows,
    )

    return rows


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_plots(
    out_dir,
    per_run_rows,
    aggregate_traces,
    shock_step,
    short_horizon_steps,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping figures because matplotlib failed: {exc}")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    opts = ["SGD", "AdaGrad", "Adam"]

    # Transmission ratio: individual seed points + median.
    fig = plt.figure(figsize=(7, 4.5))
    for i, opt in enumerate(opts):
        vals = [
            r["TransmissionRatio"]
            for r in per_run_rows
            if r["Optimizer"] == opt
        ]
        xs = np.full(len(vals), i, dtype=float)
        plt.scatter(xs, vals)
        plt.scatter([i], [np.median(vals)], marker="_", s=300)

    plt.xticks(range(len(opts)), opts)
    plt.yscale("log")
    plt.ylabel("Causal transmission ratio")
    plt.title("Instantaneous shock transmission across seeds")
    plt.tight_layout()
    plt.savefig(
        fig_dir / "transmission_ratio_seed_points.png",
        dpi=220,
    )
    plt.close(fig)

    # Relative-to-control update response.
    fig = plt.figure(figsize=(7, 4.5))
    for i, opt in enumerate(opts):
        vals = [
            r["RelativeCausalUpdateToControlUpdate"]
            for r in per_run_rows
            if r["Optimizer"] == opt
        ]
        xs = np.full(len(vals), i, dtype=float)
        plt.scatter(xs, vals)
        plt.scatter([i], [np.median(vals)], marker="_", s=300)

    plt.xticks(range(len(opts)), opts)
    plt.yscale("log")
    plt.ylabel("Causal update / ordinary control update")
    plt.title("Shock response relative to ordinary update")
    plt.tight_layout()
    plt.savefig(
        fig_dir / "relative_update_response_seed_points.png",
        dpi=220,
    )
    plt.close(fig)

    # Median signed causal loss response.
    fig = plt.figure(figsize=(8, 5))

    for opt in opts:
        rows = aggregate_traces[opt]
        steps = np.asarray([r["Step"] for r in rows])
        values = np.asarray([
            r["MedianDeltaProbeLoss"]
            for r in rows
        ])

        mask = (
            (steps >= shock_step - 100)
            & (steps <= shock_step + short_horizon_steps)
        )

        plt.plot(
            steps[mask],
            values[mask],
            label=opt,
        )

    plt.axhline(0, linewidth=1)
    plt.axvline(
        shock_step,
        linestyle="--",
        label="Shock",
    )
    plt.xlabel("Training step")
    plt.ylabel("Median paired probe-loss deviation")
    plt.title("Short-horizon causal response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        fig_dir / "median_short_horizon_causal_response.png",
        dpi=220,
    )
    plt.close(fig)

    # Median absolute causal loss response.
    fig = plt.figure(figsize=(8, 5))

    for opt in opts:
        rows = aggregate_traces[opt]
        steps = np.asarray([r["Step"] for r in rows])
        values = np.asarray([
            r["MedianAbsoluteDeltaProbeLoss"]
            for r in rows
        ])

        mask = (
            (steps >= shock_step - 100)
            & (steps <= shock_step + short_horizon_steps)
        )

        plt.plot(
            steps[mask],
            values[mask],
            label=opt,
        )

    plt.axvline(
        shock_step,
        linestyle="--",
        label="Shock",
    )
    plt.xlabel("Training step")
    plt.ylabel("Median |paired probe-loss deviation|")
    plt.title("Short-horizon absolute causal response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        fig_dir / "median_short_horizon_absolute_response.png",
        dpi=220,
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="./data",
    )
    p.add_argument(
        "--output-dir",
        default="cifar10_final_v2_results",
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028],
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=8,
    )
    p.add_argument(
        "--train-limit",
        type=int,
        default=20000,
    )
    p.add_argument(
        "--probe-size",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--shock-step",
        type=int,
        default=500,
    )
    p.add_argument(
        "--shock-multiplier",
        type=float,
        default=20.0,
    )
    p.add_argument(
        "--short-horizon-steps",
        type=int,
        default=300,
    )
    p.add_argument(
        "--sparse-eval-interval",
        type=int,
        default=200,
    )

    p.add_argument(
        "--pre-shock-tolerance",
        type=float,
        default=1e-8,
    )

    # Calibration.
    p.add_argument(
        "--calibration-seed",
        type=int,
        default=4242,
    )
    p.add_argument(
        "--partition-seed",
        type=int,
        default=13579,
    )
    p.add_argument(
        "--calibration-epochs",
        type=int,
        default=3,
    )
    p.add_argument(
        "--calibration-train-limit",
        type=int,
        default=10000,
    )
    p.add_argument(
        "--calibration-val-size",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--sgd-lr-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.10],
    )
    p.add_argument(
        "--adagrad-lr-grid",
        nargs="+",
        type=float,
        default=[0.003, 0.01, 0.03, 0.05],
    )
    p.add_argument(
        "--adam-lr-grid",
        nargs="+",
        type=float,
        default=[0.0003, 0.001, 0.003],
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    torch.set_num_threads(
        max(1, min(8, os.cpu_count() or 4))
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        final_train,
        calibration_train,
        calibration_val,
        probe_dataset,
    ) = load_cifar(args)

    probe_loader = make_loader(
        probe_dataset,
        batch_size=args.batch_size,
        seed=0,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # -------------------------------------------------------------
    # Stage 1: LR calibration
    # -------------------------------------------------------------
    calibration_rows, selected_lrs = calibrate_learning_rates(
        args,
        calibration_train,
        calibration_val,
    )

    write_csv(
        out_dir / "calibration_results.csv",
        calibration_rows,
    )

    with open(
        out_dir / "selected_learning_rates.json",
        "w",
    ) as f:
        json.dump(
            selected_lrs,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Stage 2: final paired experiment
    # -------------------------------------------------------------
    steps_per_epoch = math.ceil(
        len(final_train) / args.batch_size
    )
    total_steps = steps_per_epoch * args.epochs

    if args.shock_step >= total_steps:
        raise ValueError(
            f"shock_step={args.shock_step} must be below "
            f"total_steps={total_steps}."
        )

    if (
        args.shock_step + args.short_horizon_steps
        > total_steps
    ):
        raise ValueError(
            "Short-horizon analysis window extends beyond "
            "training. Increase epochs or reduce horizon."
        )

    eval_steps = build_eval_steps(
        total_steps,
        args.shock_step,
        args.sparse_eval_interval,
    )
    eval_steps_set = set(eval_steps)

    config = vars(args).copy()
    config["selected_learning_rates"] = selected_lrs
    config["steps_per_epoch"] = steps_per_epoch
    config["total_steps"] = total_steps
    config["evaluation_steps"] = eval_steps
    config["final_train_size"] = len(final_train)
    config["probe_size_actual"] = len(probe_dataset)

    with open(
        out_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("FINAL PAIRED SHOCK EXPERIMENT")
    print("=" * 78)
    print(f"Seeds: {args.seeds}")
    print(f"Selected LRs: {selected_lrs}")
    print(f"Final train size: {len(final_train)}")
    print(f"Probe size: {len(probe_dataset)}")
    print(f"Epochs: {args.epochs}")
    print(f"Steps/epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Shock step: {args.shock_step}")
    print(f"Short horizon: {args.short_horizon_steps}")
    print(
        f"Total trajectories: "
        f"{len(args.seeds) * 3 * 2}"
    )

    per_run_rows = []
    equality_rows = []
    traces = {
        "SGD": [],
        "AdaGrad": [],
        "Adam": [],
    }

    t_all = time.time()

    for seed in args.seeds:
        set_seed(seed)

        base_model = SmallCIFARCNN()

        initial_state = {
            k: v.detach().clone()
            for k, v in base_model.state_dict().items()
        }

        print("\n" + "#" * 78)
        print(f"SEED {seed}")
        print("#" * 78)

        for opt in ["SGD", "AdaGrad", "Adam"]:
            lr = selected_lrs[opt]

            print(
                f"\n{opt}, seed={seed}, lr={lr:g}: CONTROL"
            )

            control = run_trajectory(
                optimizer_name=opt,
                lr=lr,
                seed=seed,
                shocked=False,
                args=args,
                train_dataset=final_train,
                probe_loader=probe_loader,
                initial_state=initial_state,
                eval_steps_set=eval_steps_set,
            )

            print(
                f"{opt}, seed={seed}, lr={lr:g}: SHOCK"
            )

            shock = run_trajectory(
                optimizer_name=opt,
                lr=lr,
                seed=seed,
                shocked=True,
                args=args,
                train_dataset=final_train,
                probe_loader=probe_loader,
                initial_state=initial_state,
                eval_steps_set=eval_steps_set,
                control_shock_update_vector=
                    control["shock_update_vector"],
                control_pre_shock_vector=
                    control["pre_shock_vector"],
            )

            equality_rows.append({
                "Optimizer": opt,
                "Seed": seed,
                "LearningRate": lr,
                "PreShockParameterDifferenceNorm":
                    shock["pre_shock_equality_norm"],
                "Tolerance":
                    args.pre_shock_tolerance,
                "Passed":
                    int(
                        shock["pre_shock_equality_norm"]
                        <= args.pre_shock_tolerance
                    ),
            })

            row, delta_loss = analyze_pair(
                control,
                shock,
                args,
            )
            per_run_rows.append(row)

            traces[opt].append({
                "steps": control["steps"],
                "delta": delta_loss,
            })

            save_pair_trace(
                out_dir /
                f"{opt.lower()}_seed{seed}_paired_trace.csv",
                control,
                shock,
                delta_loss,
            )

            print(
                f"  pre-shock diff="
                f"{row['PreShockParameterDifferenceNorm']:.3e} | "
                f"transmission="
                f"{row['TransmissionRatio']:.6g} | "
                f"relative-update="
                f"{row['RelativeCausalUpdateToControlUpdate']:.6g} | "
                f"immediate Δloss="
                f"{row['ImmediateProbeLossDeviation']:.6g} | "
                f"peak |Δloss|="
                f"{row['ShortHorizonPeakAbsoluteLossDeviation']:.6g} | "
                f"delay="
                f"{row['ShortHorizonPeakDelaySteps']}"
            )

    write_csv(
        out_dir / "per_run_summary.csv",
        per_run_rows,
    )

    write_csv(
        out_dir / "pre_shock_equality_checks.csv",
        equality_rows,
    )

    aggregate_rows = compute_aggregate_rows(
        per_run_rows
    )

    write_csv(
        out_dir / "aggregate_summary.csv",
        aggregate_rows,
    )

    aggregate_traces = {}

    for opt in ["SGD", "AdaGrad", "Adam"]:
        aggregate_traces[opt] = make_aggregate_trace(
            out_dir,
            opt,
            traces[opt],
        )

    make_plots(
        out_dir,
        per_run_rows,
        aggregate_traces,
        args.shock_step,
        args.short_horizon_steps,
    )

    elapsed_minutes = (
        time.time() - t_all
    ) / 60.0

    print("\n" + "=" * 78)
    print("FINISHED")
    print("=" * 78)
    print(
        f"Final paired-experiment runtime: "
        f"{elapsed_minutes:.1f} minutes"
    )
    print(
        "Calibration results:",
        out_dir / "calibration_results.csv",
    )
    print(
        "Selected LRs:",
        out_dir / "selected_learning_rates.json",
    )
    print(
        "Per-run summary:",
        out_dir / "per_run_summary.csv",
    )
    print(
        "Aggregate summary:",
        out_dir / "aggregate_summary.csv",
    )
    print(
        "Pre-shock checks:",
        out_dir / "pre_shock_equality_checks.csv",
    )
    print(
        "Figures:",
        out_dir / "figures",
    )


if __name__ == "__main__":
    main()
