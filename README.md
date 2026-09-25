# Beyond Update Tails: Shock Attenuation and Trajectory Response of Adaptive Optimizers Under Heavy-Tailed Noise

Reproducibility code and experimental outputs for:

> **Beyond Update Tails: Shock Attenuation and Trajectory Response of Adaptive Optimizers Under Heavy-Tailed Noise**
> Reddhi Das — Department of Electrical and Computer Engineering, North Carolina State University

This project studies **SGD, AdaGrad, and Adam** under extreme stochastic-gradient events. The central question is:

> Does stronger attenuation of an extreme gradient at the current update necessarily imply a smaller downstream trajectory response?

The main empirical conclusion is that **instantaneous attenuation and downstream trajectory response are distinct properties**. An optimizer can strongly suppress the update associated with an extreme gradient event while still exhibiting a larger or more delayed deviation later in training.

---

## Experimental Overview

| Study | Purpose | Intervention / Noise | Primary Measurements |
|---|---|---|---|
| Quadratic tail study | Compare how optimizers transform controlled noise distributions | Gaussian, Student-t5, Student-t3; unclipped and clipped | Update-norm quantiles, conditional-tail mean, exceedance probabilities, Hill diagnostics |
| Quadratic paired-shock study | Isolate the effect of one extreme event | Single shock at iteration 50; M in {2,5,10,20,50,100} | Immediate shocked update, transmission, causal objective deviation, recovery |
| CIFAR-10 paired intervention | Test whether the attenuation/trajectory separation persists in nonlinear training | 20x relative perturbation of final classifier-weight gradient at step 500 | Causal update transmission, relative update, probe-loss deviation, peak delay, cumulative response |

---

## 1. Controlled Quadratic Problem

The synthetic objective is:

```
F(x) = (1/2) * x^T * Q * x,    grad F(x) = Q * x
```

with x in R^10. The implementation constructs:

```matlab
eigvals = linspace(0.5, 5.0, 10);
Q = diag(eigvals);
```

The ten eigenvalues are equally spaced over [0.5, 5]. The objective is 0.5-strongly convex and 5-smooth, with minimizer x* = 0.

### Heavy-Tailed Perturbations

The distributional study compares Gaussian noise with Student-t noise for nu in {3, 5}. Student-t samples are rescaled to match coordinate-wise Gaussian variance:

```
xi_t = sigma * sqrt((nu - 2) / nu) * z_t,    z_{t,j} ~ t_nu i.i.d.
```

with sigma = 1. Both raw and coordinate-wise clipped gradients are studied, with clipping threshold c = 5.

The primary distributional variable is the parameter-update norm:

```
U_t = || x_{t+1} - x_t ||_2
```

### Representative Unclipped Update Statistics

| Optimizer | Noise | Median | Q95 | Q99 | C95 | Hill Diagnostic |
|---|---|---:|---:|---:|---:|---:|
| SGD | Gaussian | 0.02496 | 0.04850 | 0.06200 | 0.05672 | 16.81 |
| SGD | Student-t5 | 0.02519 | 0.05202 | 0.06791 | 0.06190 | 6.28 |
| SGD | Student-t3 | 0.02327 | 0.05337 | 0.08522 | 0.07940 | 3.66 |
| AdaGrad | Gaussian | 0.02055 | 0.04022 | 0.05032 | 0.04736 | 16.07 |
| AdaGrad | Student-t5 | 0.02089 | 0.04212 | 0.05608 | 0.05066 | 7.01 |
| AdaGrad | Student-t3 | 0.01978 | 0.04147 | 0.05769 | 0.05290 | 4.17 |
| Adam | Gaussian | 0.06263 | 0.12710 | 0.16386 | 0.14832 | 15.89 |
| Adam | Student-t5 | 0.06353 | 0.12965 | 0.16903 | 0.15151 | 13.69 |
| Adam | Student-t3 | 0.06017 | 0.13556 | 0.19508 | 0.16812 | 7.08 |

Under Student-t3, clipping reduces Q99 from 0.0852 to 0.0555 for SGD, from 0.0577 to 0.0482 for AdaGrad, and from 0.1951 to 0.1468 for Adam.

**Key distinction:** the tail of the injected gradient noise and the distribution of optimizer updates are not the same object.

---

## 2. Quadratic Paired Shock-Control Experiment

The paired experiment isolates one extreme gradient event. Shocked and control trajectories use the same initialization, optimizer, hyperparameters, and background stochastic sequence, and coincide before the intervention.

### Configuration

```
Shock iteration:       50
Shock magnitudes:      2, 5, 10, 20, 50, 100
Paired trials:         500 per optimizer/magnitude
Trajectory horizon:    750 iterations
Background noise SD:   0.10
Shock dimension:       1
Recovery hold:         10 consecutive iterations
```

### Transmission Definition (Quadratic)

The quadratic transmission statistic is defined as:

```
R_trans = || Delta_x_tau_shocked ||_2 / M
```

where Delta_x_tau_shocked is the **total shocked update** at the intervention — not the shocked-minus-control difference. This differs from the CIFAR-10 definition below.

### Trajectory Metrics

For the quadratic objective, D_t = F(x_t_shocked) - F(x_t_control). The study reports:

```
D_peak_positive = max_t [D_t]+
C_positive      = sum_t [D_t]+
```

Recovery times T50 and T90 are measured after the absolute-deviation peak and require the response to remain below 50% or 10% of that peak for 10 consecutive iterations. Trials not satisfying the criterion within the horizon are right-censored.

### Selected Shock Results

| M | Optimizer | R_trans | D_peak+ | C+ | T90 Failure |
|---:|---|---:|---:|---:|---:|
| 20 | SGD | 0.01004 | 0.00945 | 0.903 | 0.000 |
| 20 | AdaGrad | 0.01015 | 0.01075 | 0.974 | 0.000 |
| 20 | Adam | 0.00167 | 0.01063 | 0.819 | 0.006 |
| 100 | SGD | 0.01000 | 0.24705 | 24.546 | 0.000 |
| 100 | AdaGrad | 0.00204 | 0.01099 | 3.983 | 0.952 |
| 100 | Adam | 0.00033 | 0.01331 | 2.843 | 0.006 |

At M=20, median absolute-deviation peaks occur at iterations 50, 51, and 80 for SGD, AdaGrad, and Adam (delays of 0, 1, and 30 iterations from the intervention).

At M=100, AdaGrad's median T50 is approximately 333 iterations and 95.2% of trials do not satisfy the T90 criterion. Adam's median T50 is 157 iterations with 0.6% T90 failure. SGD has T50=70 and T90=230 across the shock sweep.

---

## 3. Why Optimizer State Matters

**SGD:** a sufficiently large shock produces an immediate update scaling as O(M).

**AdaGrad:** a dominant shock contributes O(M^2) to the cumulative squared-gradient accumulator. The immediate normalized update can remain O(1) as M grows, while the shock contribution remains in the accumulator indefinitely (no forgetting factor).

**Adam:** a dominant shock contributes at order M to the first moment and M^2 to the second moment. Normalization bounds the leading immediate response, while the direct moment contribution subsequently decays geometrically according to the optimizer's exponential-memory factors.

These are leading-order mechanism arguments. Once shocked and control parameters diverge, later gradient differences also contribute to evolving optimizer states and trajectories.

---

## 4. CIFAR-10 Image-Learning Validation

The CIFAR-10 experiment tests whether the attenuation-versus-trajectory distinction persists during nonlinear neural-network training.

**Important:** this experiment does not claim that naturally occurring CIFAR-10 gradients follow a heavy-tailed distribution. It applies a controlled extreme gradient perturbation to otherwise matched training trajectories.

### Network Architecture

```
Conv(32) -> ReLU -> MaxPool
Conv(64) -> ReLU -> MaxPool
Conv(128) -> ReLU
Global Average Pooling
Linear(128 -> 10)
```

Random data augmentation is intentionally omitted to preserve exact pairwise matching.

### Learning-Rate Calibration

```
Training images:       10,000
Validation images:      2,000 (disjoint)
Calibration epochs:         3
Calibration seed:         fixed
```

Candidate rates:

```
SGD:      0.01, 0.03, 0.05, 0.10
AdaGrad:  0.003, 0.01, 0.03, 0.05
Adam:     0.0003, 0.001, 0.003
```

Selected operating points:

| Optimizer | Selected LR | Calibration Accuracy |
|---|---:|---:|
| SGD | 0.03 | 0.2495 |
| AdaGrad | 0.01 | 0.3455 |
| Adam | 0.003 | 0.4105 |

These are validation-selected operating points, not globally optimal hyperparameters.

### Paired Protocol

```
Seeds:   2026, 2027, 2028
```

Within each optimizer-seed pair, shocked and control runs have identical initialization, data ordering, preprocessing, and pre-intervention training history. All nine recorded pre-shock parameter differences are exactly zero and pass the 1e-8 equality tolerance.

### Intervention

At training step 500, only the final classifier-weight gradient is perturbed:

```
q_tau = 20 * || g_tau_cls_control ||_2 * u,    || u ||_2 = 1
```

The perturbation has the same relative severity (20x the current classifier-gradient norm) but not necessarily the same absolute norm across optimizers or seeds. Probe-loss deviation is evaluated for 300 post-intervention training steps.

### Transmission Definition (CIFAR-10)

Unlike the quadratic metric, CIFAR-10 uses the causal shocked-minus-control update:

```
delta_theta = Delta_theta_shocked - Delta_theta_control

R_trans = || delta_theta ||_2 / || q_tau ||_2

R_rel   = || delta_theta ||_2 / || Delta_theta_control ||_2
```

### CIFAR-10 Results

| Metric | SGD | AdaGrad | Adam |
|---|---:|---:|---:|
| Median transmission ratio | 0.0300 | 0.0114 | 0.0090 |
| Median relative causal update | 10.67 | 2.73 | 1.28 |
| Median peak absolute deviation | 0.00151 | 0.00290 | 0.01214 |
| Observed peak-delay range (steps) | 0-1 | 5-300 | 50-150 |
| Median cumulative absolute deviation | 0.1594 | 0.1956 | 1.4591 |

Only three seeds are used — these values are descriptive replications, not population-level significance estimates.

---

## Project Structure

```
Heavy-Tailed-Optimizer-Shock-Analysis/
|
|-- README.md
|-- matlab/
|   `-- <quadratic V4 MATLAB scripts>
|-- python/
|   `-- cifar10_final_v2_calibrated_3seed.py
|-- results/
|   |-- hp_adaptive_v4_summary.csv
|   |-- hp_adaptive_v4_violation_summary.csv
|   |-- hp_adaptive_v4_shockonly_paired_shock_summary.csv
|   |-- aggregate_summary.csv
|   |-- per_run_summary.csv
|   |-- calibration_results.csv
|   `-- pre_shock_equality_checks.csv
|-- figures/
|   |-- hp_adaptive_v4_shockonly_paired_shock_immediate_update.png
|   |-- hp_adaptive_v4_shockonly_paired_shock_transmission_ratio.png
|   |-- hp_adaptive_v4_shockonly_paired_shock_peak_causal_damage.png
|   |-- hp_adaptive_v4_shockonly_paired_shock_cumulative_damage.png
|   |-- hp_adaptive_v4_shockonly_paired_shock_T50.png
|   |-- hp_adaptive_v4_shockonly_paired_shock_T90_failure.png
|   |-- transmission_ratio_seed_points.png
|   `-- median_short_horizon_absolute_response.png
`-- paper/
    `-- manuscript.pdf
```

---

## Requirements

### MATLAB Experiments

Requires MATLAB with the **Statistics and Machine Learning Toolbox** (for `trnd` and `prctile`).

### CIFAR-10 Experiment

```
torch
torchvision
numpy
pandas
matplotlib
```

PyTorch version used during reported CPU runs: `2.11.0+cpu`.

Generate `requirements.txt` directly from the released environment rather than treating the list above as an exact lockfile.

---

## Running the Experiments

### Quadratic Experiments (MATLAB)

Run the released V4 MATLAB entry point. The shock-only implementation:

1. Builds the 10-dimensional quadratic problem
2. Tunes optimizer learning rates on the Gaussian/unclipped reference condition
3. Freezes the selected rates
4. Runs the paired shock-control sweep
5. Computes immediate, trajectory, recovery, and effective-scale diagnostics
6. Generates shock-response figures
7. Exports the paired-shock summary to `hp_adaptive_v4_shockonly_paired_shock_summary.csv`

Full shock experiment state is stored in `hp_adaptive_v4_shockonly_results.mat`.

### CIFAR-10 Experiment (Python)

```bash
python cifar10_final_v2_calibrated_3seed.py
```

The script runs learning-rate calibration, matched control/shock runs, pre-shock equality checks, aggregation, and figure generation. Use the script's help output for exact CLI arguments and dataset paths.

---

## Key Output Files

| File | Contents |
|---|---|
| `hp_adaptive_v4_summary.csv` | Quadratic distributional and tail summary |
| `hp_adaptive_v4_violation_summary.csv` | Empirical threshold-exceedance probabilities |
| `hp_adaptive_v4_shockonly_paired_shock_summary.csv` | Aggregated paired quadratic shock metrics |
| `aggregate_summary.csv` | Aggregated CIFAR-10 results across seeds |
| `per_run_summary.csv` | Optimizer/seed-level CIFAR-10 measurements |
| `calibration_results.csv` | CIFAR-10 learning-rate calibration results |
| `pre_shock_equality_checks.csv` | Matched-pair pre-intervention equality checks |

---

## Reproducibility Notes

- Gaussian and Student-t comparisons use variance matching.
- Learning rates are selected before final evaluation and not retuned for individual tail conditions.
- Paired experiments use matched randomness and pre-intervention trajectories.
- The quadratic and CIFAR-10 transmission ratios have **different numerators** — see Section 2 and Section 4 for explicit definitions.
- Clipped Hill estimates are not interpreted as asymptotic tail exponents.
- Recovery statistics must be interpreted together with their censoring/failure fractions.
- The CIFAR-10 experiment is a controlled extreme-gradient intervention, not evidence that the natural CIFAR-10 gradient distribution is heavy-tailed.
- The three-seed CIFAR-10 results are descriptive.
- Different metrics (immediate attenuation, peak response, cumulative response, delay, recovery) can produce different optimizer orderings under the same intervention.

---

## Citation

```bibtex
@misc{das2026beyondupdatetails,
  author = {Das, Reddhi},
  title  = {Beyond Update Tails: Shock Attenuation and Trajectory Response
            of Adaptive Optimizers Under Heavy-Tailed Noise},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Replace with final DOI, venue, and publication details upon acceptance.

---

## Author

**Reddhi Das**
Department of Electrical and Computer Engineering
North Carolina State University
rdas5@ncsu.edu | [GitHub](https://github.com/reddhidas5)

---

## Keywords

Stochastic optimization, heavy-tailed noise, adaptive optimization, SGD, AdaGrad, Adam, shock response, trajectory response
