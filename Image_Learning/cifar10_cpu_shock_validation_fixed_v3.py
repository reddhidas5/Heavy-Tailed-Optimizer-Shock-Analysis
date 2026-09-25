#!/usr/bin/env python3
"""
CPU-friendly CIFAR-10 paired shock validation for SGD, AdaGrad, and Adam.

Purpose
-------
Validate the synthetic V4 mechanism on a real image-learning problem:
    gradient shock -> immediate parameter update -> delayed loss deviation -> recovery

Default pilot = 6 training runs:
    3 optimizers x {matched control, shocked trajectory} x 1 seed

The control and shocked run for each optimizer use the same:
    - model initialization
    - CIFAR-10 ordering
    - preprocessing
    - seed

A single controlled gradient shock is injected into the classifier-weight gradient
at a fixed training step. The script measures:
    - immediate full-parameter update norm
    - causal update-vector difference vs matched control
    - probe-set loss deviation
    - peak positive causal damage
    - peak absolute deviation
    - cumulative positive causal damage
    - peak-deviation delay
    - T50 / T90 recovery (with sustained-hold criterion)

"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ----------------------------- Model -----------------------------------------

class SmallCIFARCNN(nn.Module):
    """Small CPU-friendly CNN (~100k parameters)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# --------------------------- Reproducibility ---------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_deterministic_loader(dataset, batch_size: int, seed: int, num_workers: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


# ----------------------------- Optimizer -------------------------------------

def make_optimizer(name: str, model: nn.Module):
    name = name.lower()
    if name == "sgd":
        # Plain SGD to match the synthetic mechanism as closely as possible.
        return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=0.01, eps=1e-10)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    raise ValueError(name)


# ------------------------- Gradient Perturbations ----------------------------

def add_background_t3_noise(model: nn.Module, scale: float):
    """
    Add variance-standardized Student-t(df=3) noise to each gradient.

    Standard Student-t(3) has variance 3, so division by sqrt(3)
    yields unit variance before applying `scale`.

    Noise scale is relative to each tensor's gradient RMS:
        noise_std ~= scale * grad_rms
    """
    if scale <= 0:
        return

    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        rms = torch.sqrt(torch.mean(g.detach() ** 2) + 1e-20)
        dist = torch.distributions.StudentT(df=torch.tensor(3.0, device=g.device))
        z = dist.sample(g.shape).to(dtype=g.dtype) / math.sqrt(3.0)
        g.add_(scale * rms * z)


def inject_classifier_shock(
    model: SmallCIFARCNN,
    multiplier: float,
    shock_seed: int,
) -> Tuple[float, float]:
    """
    Inject a controlled gradient shock into classifier.weight.

    Let g be the current classifier gradient and u a fixed random unit vector.
    We add:
        shock = multiplier * ||g||_2 * u

    Thus the injected shock norm is multiplier times the local gradient norm.

    Returns
    -------
    base_grad_norm, injected_shock_norm
    """
    p = model.classifier.weight
    if p.grad is None:
        raise RuntimeError("classifier.weight has no gradient at shock step.")

    g = p.grad
    base_norm = torch.linalg.vector_norm(g.detach()).item()

    gen = torch.Generator(device=g.device)
    gen.manual_seed(shock_seed)
    u = torch.randn(g.shape, generator=gen, device=g.device, dtype=g.dtype)
    u_norm = torch.linalg.vector_norm(u)
    u = u / (u_norm + 1e-20)

    shock_norm = multiplier * max(base_norm, 1e-12)
    g.add_(shock_norm * u)
    return base_norm, shock_norm


# ------------------------------ Metrics --------------------------------------

@torch.no_grad()
def parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


@torch.no_grad()
def evaluate_probe(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    model.train()
    return loss_sum / n, correct / n


def sustained_recovery_time(
    steps: np.ndarray,
    deviation: np.ndarray,
    peak_index: int,
    fraction: float,
    hold: int,
):
    """
    First evaluation step after the absolute-deviation peak for which
    |delta| stays <= fraction * peak for `hold` consecutive evaluation points.
    """
    peak = abs(deviation[peak_index])
    if peak <= 1e-15:
        return 0.0

    threshold = fraction * peak
    absdev = np.abs(deviation)
    for i in range(peak_index + 1, len(absdev) - hold + 1):
        if np.all(absdev[i:i + hold] <= threshold):
            return float(steps[i] - steps[peak_index])
    return float("nan")


# ----------------------------- One Trajectory --------------------------------

def run_trajectory(
    optimizer_name: str,
    shocked: bool,
    args,
    train_dataset,
    probe_loader,
    initial_state: Dict[str, torch.Tensor],
    control_shock_update_vector: torch.Tensor = None,
):
    set_seed(args.seed)

    device = torch.device("cpu")
    model = SmallCIFARCNN().to(device)
    model.load_state_dict(initial_state)
    optimizer = make_optimizer(optimizer_name, model)

    loader = make_deterministic_loader(
        train_dataset,
        batch_size=args.batch_size,
        seed=args.seed + 12345,
        num_workers=args.num_workers,
    )

    # Evaluate at step 0.
    probe_steps = [0]
    probe_losses = []
    probe_accs = []
    loss0, acc0 = evaluate_probe(model, probe_loader, device)
    probe_losses.append(loss0)
    probe_accs.append(acc0)

    global_step = 0
    shock_metadata = {
        "base_classifier_grad_norm": float("nan"),
        "injected_shock_norm": 0.0,
        "immediate_update_norm": float("nan"),
        "causal_update_vector_norm": float("nan"),
        "transmission_ratio": float("nan"),
    }

    shock_update_vector = None

    model.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        for x, y in loader:
            global_step += 1
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            if args.background_noise == "t3":
                add_background_t3_noise(model, args.background_noise_scale)

            is_shock_step = (global_step == args.shock_step)
            before = None
            if is_shock_step:
                before = parameter_vector(model)

                if shocked:
                    base_norm, shock_norm = inject_classifier_shock(
                        model,
                        multiplier=args.shock_multiplier,
                        shock_seed=args.seed + 777,
                    )
                    shock_metadata["base_classifier_grad_norm"] = base_norm
                    shock_metadata["injected_shock_norm"] = shock_norm

            optimizer.step()

            if is_shock_step:
                after = parameter_vector(model)
                shock_update_vector = after - before
                shock_metadata["immediate_update_norm"] = torch.linalg.vector_norm(
                    shock_update_vector
                ).item()

                if shocked and control_shock_update_vector is not None:
                    causal_vec = shock_update_vector - control_shock_update_vector
                    causal_norm = torch.linalg.vector_norm(causal_vec).item()
                    shock_metadata["causal_update_vector_norm"] = causal_norm
                    denom = max(shock_metadata["injected_shock_norm"], 1e-20)
                    shock_metadata["transmission_ratio"] = causal_norm / denom

                # Evaluate immediately after the optimizer step.
                pl, pa = evaluate_probe(model, probe_loader, device)
                probe_steps.append(global_step)
                probe_losses.append(pl)
                probe_accs.append(pa)

            elif global_step % args.eval_interval == 0:
                pl, pa = evaluate_probe(model, probe_loader, device)
                probe_steps.append(global_step)
                probe_losses.append(pl)
                probe_accs.append(pa)

    # Evaluating final state if needed.
    if probe_steps[-1] != global_step:
        pl, pa = evaluate_probe(model, probe_loader, device)
        probe_steps.append(global_step)
        probe_losses.append(pl)
        probe_accs.append(pa)

    elapsed = time.time() - t0

    return {
        "optimizer": optimizer_name,
        "shocked": shocked,
        "steps": np.asarray(probe_steps, dtype=int),
        "probe_loss": np.asarray(probe_losses, dtype=float),
        "probe_acc": np.asarray(probe_accs, dtype=float),
        "shock_update_vector": shock_update_vector,
        "shock_metadata": shock_metadata,
        "elapsed_seconds": elapsed,
        "final_probe_loss": float(probe_losses[-1]),
        "final_probe_acc": float(probe_accs[-1]),
    }


# ---------------------------- Paired Analysis --------------------------------

def analyze_pair(control, shock, args):
    if not np.array_equal(control["steps"], shock["steps"]):
        raise RuntimeError("Control and shock evaluation steps do not match.")

    steps = control["steps"]
    delta_loss = shock["probe_loss"] - control["probe_loss"]
    abs_delta = np.abs(delta_loss)
    pos_delta = np.maximum(delta_loss, 0.0)

    post = np.where(steps >= args.shock_step)[0]
    if len(post) == 0:
        raise RuntimeError(f"No post-shock evaluations: shock_step={args.shock_step}, last recorded step={steps[-1]}. Increase --epochs, increase --train-limit, or lower --shock-step.")

    local_abs = abs_delta[post]
    peak_local_idx = int(np.argmax(local_abs))
    peak_idx = int(post[peak_local_idx])

    peak_pos_damage = float(np.max(pos_delta[post]))
    peak_abs_dev = float(abs_delta[peak_idx])
    peak_step = int(steps[peak_idx])

    # Integrate positive causal probe-loss deviation over step index.
    cum_pos = float(np.trapezoid(pos_delta[post], x=steps[post]))

    t50 = sustained_recovery_time(
        steps, delta_loss, peak_idx, fraction=0.50, hold=args.recovery_hold
    )
    t90 = sustained_recovery_time(
        steps, delta_loss, peak_idx, fraction=0.10, hold=args.recovery_hold
    )

    sm = shock["shock_metadata"]

    return {
        "Optimizer": shock["optimizer"],
        "Seed": args.seed,
        "BackgroundNoise": args.background_noise,
        "ShockMultiplier": args.shock_multiplier,
        "ShockStep": args.shock_step,
        "ImmediateControlUpdateNorm": control["shock_metadata"]["immediate_update_norm"],
        "ImmediateShockUpdateNorm": sm["immediate_update_norm"],
        "CausalUpdateVectorNorm": sm["causal_update_vector_norm"],
        "InjectedShockNorm": sm["injected_shock_norm"],
        "TransmissionRatio": sm["transmission_ratio"],
        "PeakPositiveCausalProbeLossDamage": peak_pos_damage,
        "PeakAbsoluteProbeLossDeviation": peak_abs_dev,
        "PeakDeviationStep": peak_step,
        "PeakDelaySteps": peak_step - args.shock_step,
        "CumulativePositiveProbeLossDamage": cum_pos,
        "T50StepsAfterPeak": t50,
        "T50Recovered": int(not math.isnan(t50)),
        "T90StepsAfterPeak": t90,
        "T90Recovered": int(not math.isnan(t90)),
        "ControlFinalProbeLoss": control["final_probe_loss"],
        "ShockFinalProbeLoss": shock["final_probe_loss"],
        "ControlFinalProbeAcc": control["final_probe_acc"],
        "ShockFinalProbeAcc": shock["final_probe_acc"],
        "ControlRuntimeSeconds": control["elapsed_seconds"],
        "ShockRuntimeSeconds": shock["elapsed_seconds"],
    }, delta_loss


# ----------------------------- Data ------------------------------------------


def resolve_shock_step(args, train_dataset):

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    if total_steps < 2:
        raise ValueError(
            f"Run is too short: only {total_steps} total training step(s). "
            "Increase --epochs or --train-limit."
        )

    if args.shock_step >= total_steps:
        old_step = args.shock_step
        args.shock_step = max(1, total_steps // 2)
        print()
        print("WARNING: requested shock step is outside this run.")
        print(f"  Requested shock step : {old_step}")
        print(f"  Total training steps : {total_steps}")
        print(f"  Automatically using  : {args.shock_step} (midpoint of run)")
        print()

    return steps_per_epoch, total_steps


def prepare_data(args):
    # No random augmentation in the pilot: matched control/shock runs should see exactly the same inputs in exactly the same order.
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
        print(f"Using local CIFAR-10 dataset from: {root.resolve()}")
    except RuntimeError:
        print("Local CIFAR-10 copy was not found or failed its integrity check.")
        print("Attempting download with retries...")

        import time as _time

        last_error = None
        for attempt in range(1, 6):
            try:
                print(f"CIFAR-10 download attempt {attempt}/5")
                train_full = datasets.CIFAR10(
                    root=root, train=True, download=True, transform=transform
                )
                test_full = datasets.CIFAR10(
                    root=root, train=False, download=False, transform=transform
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(f"Download attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt < 5:
                    wait_s = 10 * attempt
                    print(f"Waiting {wait_s} seconds before retrying...")
                    _time.sleep(wait_s)

        if last_error is not None:
            raise RuntimeError(
                "\\nCIFAR-10 could not be downloaded after 5 attempts.\\n"
                f"Expected data folder: {root.resolve()}\\n\\n"
                "This is a network/download problem, not a model-training problem.\\n"
                "You can download the archive manually in a browser or with Windows "
                "curl, extract it into the data folder, and rerun the script.\\n"
                "Expected extracted folder: data/cifar-10-batches-py\\n"
            ) from last_error

    # Deterministic optional training subset for faster CPU pilot.
    if args.train_limit and args.train_limit < len(train_full):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_full), size=args.train_limit, replace=False)
        train_dataset = Subset(train_full, idx.tolist())
    else:
        train_dataset = train_full

    probe_size = min(args.probe_size, len(test_full))
    probe_dataset = Subset(test_full, list(range(probe_size)))
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_dataset, probe_loader


# ------------------------------ Outputs --------------------------------------

def save_trace_csv(path: Path, control, shock, delta):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Step",
            "ControlProbeLoss",
            "ShockProbeLoss",
            "DeltaProbeLoss",
            "ControlProbeAcc",
            "ShockProbeAcc",
        ])
        for row in zip(
            control["steps"],
            control["probe_loss"],
            shock["probe_loss"],
            delta,
            control["probe_acc"],
            shock["probe_acc"],
        ):
            w.writerow(row)


def save_summary_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def maybe_make_plots(out_dir: Path, pairs: Dict[str, tuple]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    # Probe loss: control vs shock, one figure per optimizer.
    for opt, (control, shock, delta) in pairs.items():
        fig = plt.figure(figsize=(7, 4.5))
        plt.plot(control["steps"], control["probe_loss"], label="Matched control")
        plt.plot(shock["steps"], shock["probe_loss"], label="Shock")
        plt.xlabel("Training step")
        plt.ylabel("Probe cross-entropy")
        plt.title(f"{opt}: matched control vs shock")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{opt.lower()}_probe_loss.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 4.5))
        plt.plot(shock["steps"], delta, label="Shock - control")
        plt.axhline(0, linewidth=1)
        plt.xlabel("Training step")
        plt.ylabel("Causal probe-loss deviation")
        plt.title(f"{opt}: paired causal deviation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{opt.lower()}_causal_deviation.png", dpi=180)
        plt.close(fig)


# ------------------------------ Main -----------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="cifar10_cpu_validation_results")

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)

    # Using full CIFAR-10 by default.
    p.add_argument("--train-limit", type=int, default=20000)
    p.add_argument("--probe-size", type=int, default=512)

    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--recovery-hold", type=int, default=3)

    # With 20k images and batch 128, ~157 steps/epoch.
    # Shock near the beginning of epoch 4 by default.
    p.add_argument("--shock-step", type=int, default=500)
    p.add_argument("--shock-multiplier", type=float, default=20.0)

    p.add_argument(
        "--background-noise",
        choices=["clean", "t3"],
        default="clean",
        help="Optional persistent standardized t3 gradient noise.",
    )
    p.add_argument(
        "--background-noise-scale",
        type=float,
        default=0.25,
        help="Gradient-RMS-relative scale for persistent t3 noise.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Reducing CPU oversubscription risk on laptops.
    max_threads = max(1, min(8, (os.cpu_count() or 4)))
    torch.set_num_threads(max_threads)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CPU-FRIENDLY CIFAR-10 PAIRED SHOCK VALIDATION")
    print("=" * 72)
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Epochs: {args.epochs}")
    print(f"Train limit: {args.train_limit}")
    print(f"Probe size: {args.probe_size}")
    print(f"Shock step: {args.shock_step}")
    print(f"Shock multiplier: {args.shock_multiplier}")
    print(f"Background noise: {args.background_noise}")
    print()

    train_dataset, probe_loader = prepare_data(args)

    steps_per_epoch, total_steps = resolve_shock_step(args, train_dataset)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Effective shock step: {args.shock_step}")
    print()

    # Saving the FINAL resolved configuration, including any smoke-test or shock-step adjustment.
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # One common initial network state for all optimizers.
    set_seed(args.seed)
    base_model = SmallCIFARCNN()
    initial_state = {
        k: v.detach().clone()
        for k, v in base_model.state_dict().items()
    }

    summary_rows = []
    pairs = {}

    for opt in ["SGD", "AdaGrad", "Adam"]:
        print(f"\n{'-' * 72}")
        print(f"{opt}: matched CONTROL")
        print(f"{'-' * 72}")

        control = run_trajectory(
            optimizer_name=opt,
            shocked=False,
            args=args,
            train_dataset=train_dataset,
            probe_loader=probe_loader,
            initial_state=initial_state,
        )

        print(f"{opt}: SHOCKED trajectory")
        shock = run_trajectory(
            optimizer_name=opt,
            shocked=True,
            args=args,
            train_dataset=train_dataset,
            probe_loader=probe_loader,
            initial_state=initial_state,
            control_shock_update_vector=control["shock_update_vector"],
        )

        row, delta = analyze_pair(control, shock, args)
        summary_rows.append(row)
        pairs[opt] = (control, shock, delta)

        save_trace_csv(
            out_dir / f"{opt.lower()}_paired_trace.csv",
            control, shock, delta
        )

        print(
            f"{opt}: immediate shock update={row['ImmediateShockUpdateNorm']:.6g}, "
            f"causal update={row['CausalUpdateVectorNorm']:.6g}, "
            f"peak delay={row['PeakDelaySteps']} steps, "
            f"T50={row['T50StepsAfterPeak']}, "
            f"T90={row['T90StepsAfterPeak']}"
        )

    save_summary_csv(out_dir / "paired_shock_summary.csv", summary_rows)
    maybe_make_plots(out_dir, pairs)

    print("\n" + "=" * 72)
    print("FINISHED")
    print("=" * 72)
    print(f"Summary: {out_dir / 'paired_shock_summary.csv'}")
    print("Per-optimizer traces and PNG plots are in the same folder.")
    print("\nTo display the summary in Python:")
    print("  import pandas as pd")
    print(f"  print(pd.read_csv(r'{out_dir / 'paired_shock_summary.csv'}').to_string(index=False))")


if __name__ == "__main__":
    main()