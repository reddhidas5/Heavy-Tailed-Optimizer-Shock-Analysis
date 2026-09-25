# Beyond Update Tails: Shock Attenuation and Trajectory Response of Adaptive Optimizers Under Heavy-Tailed Noise

Reproducibility code and experimental outputs for:

> **Beyond Update Tails: Shock Attenuation and Trajectory Response of Adaptive Optimizers Under Heavy-Tailed Noise**  
> Reddhi Das — Department of Electrical and Computer Engineering, North Carolina State University

This project studies **SGD, AdaGrad, and Adam** under extreme stochastic-gradient events. The central question is:

> **Does stronger attenuation of an extreme gradient at the current update necessarily imply a smaller downstream trajectory response?**

The project addresses this question in two complementary settings:

- a controlled strongly convex quadratic problem with variance-matched Gaussian and Student-\(t\) noise; and
- a paired CIFAR-10 intervention experiment using a controlled extreme classifier-gradient perturbation.

The main empirical conclusion is that **instantaneous attenuation and downstream trajectory response are distinct properties**. An optimizer can strongly suppress the update associated with an extreme gradient event while still exhibiting a larger or more delayed deviation later in training.

---

## Experimental overview

| Study | Purpose | Main intervention / noise | Primary measurements |
|---|---|---|---|
| Quadratic tail study | Compare how optimizers transform controlled noise distributions | Gaussian, Student-\(t_5\), Student-\(t_3\); unclipped and coordinate-wise clipped | update-norm quantiles, conditional-tail mean, exceedance probabilities, Hill diagnostics |
| Quadratic paired-shock study | Isolate the effect of one extreme event | single shock at iteration 50; \(M\in\{2,5,10,20,50,100\}\) | immediate shocked update, transmission, causal objective deviation, recovery, effective scaling |
| CIFAR-10 paired intervention | Test whether the attenuation/trajectory separation persists in nonlinear training | \(20\times\) relative perturbation of final classifier-weight gradient at step 500 | causal update transmission, relative update, probe-loss deviation, peak delay, cumulative response |

---

## 1. Controlled quadratic problem

The synthetic objective is

\[
F(x)=\frac{1}{2}x^\top Qx,
\qquad \nabla F(x)=Qx,
\]

with \(x\in\mathbb{R}^{10}\). The implementation constructs

```matlab
eigvals = linspace(0.5, 5.0, 10);
Q = diag(eigvals);
```

so the ten eigenvalues are equally spaced over \([0.5,5]\). The objective is therefore \(0.5\)-strongly convex and \(5\)-smooth, with minimizer \(x^\star=0\).

### Heavy-tailed perturbations

The distributional study compares Gaussian noise with Student-\(t_\nu\) noise for \(\nu\in\{3,5\}\). Student-\(t\) samples are rescaled to match the coordinate-wise Gaussian variance:

\[
\xi_t=\sigma\sqrt{\frac{\nu-2}{\nu}}\,z_t,
\qquad z_{t,j}\overset{\mathrm{i.i.d.}}{\sim}t_\nu,
\]

with \(\sigma=1\).

Both raw and coordinate-wise clipped gradients are studied, with clipping threshold \(c=5\).

The primary distributional variable is the parameter-update norm

\[
U_t=\|x_{t+1}-x_t\|_2.
\]

The analysis uses empirical medians and upper quantiles, conditional-tail means, threshold-exceedance probabilities, and Hill diagnostics. Hill estimates are treated only as finite-sample upper-tail diagnostics: Gaussian data are not regularly varying, and clipping bounds the processed gradient.

### Representative unclipped update statistics

| Optimizer | Noise | Median | \(Q_{95}\) | \(Q_{99}\) | \(C_{95}\) | Hill diagnostic |
|---|---|---:|---:|---:|---:|---:|
| SGD | Gaussian | 0.02496 | 0.04850 | 0.06200 | 0.05672 | 16.81 |
| SGD | Student-\(t_5\) | 0.02519 | 0.05202 | 0.06791 | 0.06190 | 6.28 |
| SGD | Student-\(t_3\) | 0.02327 | 0.05337 | 0.08522 | 0.07940 | 3.66 |
| AdaGrad | Gaussian | 0.02055 | 0.04022 | 0.05032 | 0.04736 | 16.07 |
| AdaGrad | Student-\(t_5\) | 0.02089 | 0.04212 | 0.05608 | 0.05066 | 7.01 |
| AdaGrad | Student-\(t_3\) | 0.01978 | 0.04147 | 0.05769 | 0.05290 | 4.17 |
| Adam | Gaussian | 0.06263 | 0.12710 | 0.16386 | 0.14832 | 15.89 |
| Adam | Student-\(t_5\) | 0.06353 | 0.12965 | 0.16903 | 0.15151 | 13.69 |
| Adam | Student-\(t_3\) | 0.06017 | 0.13556 | 0.19508 | 0.16812 | 7.08 |

Under Student-\(t_3\), clipping reduces \(Q_{99}\) from 0.0852 to 0.0555 for SGD, from 0.0577 to 0.0482 for AdaGrad, and from 0.1951 to 0.1468 for Adam.

These results motivate a key distinction: **the tail of the injected gradient noise and the distribution of optimizer updates are not the same object**.

---

## 2. Quadratic paired shock-control experiment

The paired experiment isolates one extreme gradient event. Within each pair, shocked and control trajectories use the same initialization, optimizer, hyperparameters, and background stochastic sequence and coincide before the intervention.

The final shock-only configuration uses:

```text
Shock iteration:       50
Shock magnitudes:      2, 5, 10, 20, 50, 100
Paired trials:         500 per optimizer/magnitude
Trajectory horizon:    750 iterations
Background noise SD:   0.10
Shock dimension:       1
Recovery hold:         10 consecutive iterations
```

### Important definition: quadratic transmission

The MATLAB implementation defines the quadratic transmission statistic as

\[
R_{\mathrm{trans}}^{(Q)}
=
\frac{\|\Delta x_\tau^{(s)}\|_2}{M},
\]

where \(\Delta x_\tau^{(s)}\) is the **total shocked update** at the intervention. It is not the shocked-minus-control update difference.

This distinction matters when comparing the synthetic experiment with the CIFAR-10 experiment below.

### Trajectory metrics

For the quadratic objective,

\[
D_t=F(x_t^{(s)})-F(x_t^{(c)}).
\]

The study reports, among other quantities,

\[
D_{\mathrm{peak}}^+
=\max_t[D_t]_+,
\qquad
C^+=\sum_t[D_t]_+.
\]

Recovery is computed from the absolute causal deviation. \(T_{50}\) and \(T_{90}\) are measured after the absolute-deviation peak and require the response to remain below 50% or 10% of that peak, respectively, for 10 consecutive iterations. Trials that do not satisfy the criterion within the horizon are right-censored.

### Selected shock results

| \(M\) | Optimizer | \(R_{\mathrm{trans}}^{(Q)}\) | \(D_{\mathrm{peak}}^+\) | \(C^+\) | \(T_{90}\) failure |
|---:|---|---:|---:|---:|---:|
| 20 | SGD | 0.01004 | 0.00945 | 0.903 | 0.000 |
| 20 | AdaGrad | 0.01015 | 0.01075 | 0.974 | 0.000 |
| 20 | Adam | 0.00167 | 0.01063 | 0.819 | 0.006 |
| 100 | SGD | 0.01000 | 0.24705 | 24.546 | 0.000 |
| 100 | AdaGrad | 0.00204 | 0.01099 | 3.983 | 0.952 |
| 100 | Adam | 0.00033 | 0.01331 | 2.843 | 0.006 |

At \(M=20\), the median absolute-deviation peaks occur at iterations 50, 51, and 80 for SGD, AdaGrad, and Adam, respectively, corresponding to delays of 0, 1, and 30 iterations from the intervention.

At \(M=100\), AdaGrad's median \(T_{50}\) is approximately 333 iterations and 95.2% of trials do not satisfy the \(T_{90}\) criterion within the observation horizon. Adam's median \(T_{50}\) is 157 iterations, with 0.6% \(T_{90}\) failure. SGD has \(T_{50}=70\) and \(T_{90}=230\) across the shock sweep.

The experiment therefore separates **immediate attenuation, peak trajectory response, accumulated response, and recovery**.

---

## 3. Why optimizer state matters

The controlled quadratic experiment provides a simple mechanism for the observed temporal differences.

For SGD, a sufficiently large shock produces an immediate update that scales as \(O(M)\).

For AdaGrad, a dominant shock contributes \(O(M^2)\) to the cumulative squared-gradient accumulator. The immediate normalized update can remain \(O(1)\) as \(M\) grows, while the shock contribution remains in the accumulator because AdaGrad has no explicit forgetting factor.

For Adam, a dominant shock contributes at order \(M\) to the first moment and \(M^2\) to the second moment. Normalization can therefore bound the leading immediate response, while the direct moment contribution subsequently decays geometrically according to the optimizer's exponential-memory factors.

These are leading-order mechanism arguments. Once shocked and control parameters diverge, later gradient differences also contribute to the evolving optimizer states and trajectories.

---

## 4. CIFAR-10 image-learning validation

The CIFAR-10 experiment asks whether the attenuation-versus-trajectory distinction persists during nonlinear neural-network training.

**It does not claim that naturally occurring CIFAR-10 gradients follow a heavy-tailed distribution.** Instead, it applies a controlled extreme gradient perturbation to otherwise matched training trajectories.

### Network

The compact convolutional model contains three convolutional stages with 32, 64, and 128 output channels:

```text
Conv -> ReLU -> MaxPool
Conv -> ReLU -> MaxPool
Conv -> ReLU
Global Average Pooling
Linear(128 -> 10)
```

Random data augmentation is intentionally omitted to preserve exact pairwise matching.

### Learning-rate calibration

Calibration uses:

```text
Training images:       10,000
Validation images:      2,000 (disjoint)
Calibration epochs:         3
Calibration seed:         fixed
```

Candidate rates are:

```text
SGD:      0.01, 0.03, 0.05, 0.10
AdaGrad:  0.003, 0.01, 0.03, 0.05
Adam:     0.0003, 0.001, 0.003
```

The highest-validation-accuracy candidates are:

| Optimizer | Selected LR | Calibration accuracy |
|---|---:|---:|
| SGD | 0.03 | 0.2495 |
| AdaGrad | 0.01 | 0.3455 |
| Adam | 0.003 | 0.4105 |

These are **validation-selected operating points**, not claims of globally optimal hyperparameters.

### Paired protocol

The final experiment uses seeds:

```text
2026
2027
2028
```

Within each optimizer-seed pair, shocked and control runs have identical initialization, data ordering, preprocessing, and pre-intervention training history. All nine recorded pre-shock parameter differences are exactly zero and pass the \(10^{-8}\) equality tolerance.

### Controlled classifier-gradient intervention

At training step \(\tau=500\), only the final classifier-weight gradient is perturbed:

\[
q_\tau
=
20\|g_{\tau,\mathrm{cls}}^{(c)}\|_2u,
\qquad
\|u\|_2=1.
\]

The perturbation therefore has the same **relative severity**—20 times the current classifier-gradient norm—but not necessarily the same absolute norm across optimizers or seeds.

Probe-loss deviation is evaluated for 300 post-intervention training steps.

### Important definition: CIFAR transmission

Unlike the quadratic metric, CIFAR-10 uses the causal shocked-minus-control update:

\[
\delta\Delta\theta_\tau
=
\Delta\theta_\tau^{(s)}
-
\Delta\theta_\tau^{(c)},
\]

\[
R_{\mathrm{trans}}^{(C)}
=
\frac{\|\delta\Delta\theta_\tau\|_2}{\|q_\tau\|_2}.
\]

The relative causal update is

\[
R_{\mathrm{rel}}
=
\frac{\|\delta\Delta\theta_\tau\|_2}
{\|\Delta\theta_\tau^{(c)}\|_2}.
\]

### CIFAR-10 results

| Metric | SGD | AdaGrad | Adam |
|---|---:|---:|---:|
| Median transmission ratio | 0.0300 | 0.0114 | 0.0090 |
| Median relative causal update | 10.67 | 2.73 | 1.28 |
| Median peak \(|D_t|\) | 0.00151 | 0.00290 | 0.01214 |
| Observed peak-delay range (steps) | 0–1 | 5–300 | 50–150 |
| Median cumulative \(|D_t|\) | 0.1594 | 0.1956 | 1.4591 |

Only three seeds are used, so these values are reported as **descriptive replications**, not population-level significance estimates.

Under the calibrated operating points, Adam has the smallest median immediate transmission but the largest median peak and cumulative absolute probe-loss deviations. Its peak occurs 50–150 steps after intervention, compared with 0–1 steps for SGD. The AdaGrad range reaches the 300-step observation boundary for one seed, so it should not be interpreted as a stable characteristic time scale.

The paired design establishes that the within-pair difference is initiated by the controlled intervention. It does **not** imply that optimizer memory is the sole cause of every later difference: after the intervention, nonlinear training dynamics can amplify, contract, or otherwise transform the initial parameter and optimizer-state perturbations.

---

## Repository layout

The repository can be organized as follows (rename entries if your local filenames differ):

```text
.
├── README.md
├── matlab/
│   └── <quadratic V4 MATLAB scripts>
├── python/
│   └── cifar10_final_v2_calibrated_3seed.py
├── results/
│   ├── hp_adaptive_v4_summary.csv
│   ├── hp_adaptive_v4_violation_summary.csv
│   ├── hp_adaptive_v4_shockonly_paired_shock_summary.csv
│   ├── aggregate_summary.csv
│   ├── per_run_summary.csv
│   ├── calibration_results.csv
│   └── pre_shock_equality_checks.csv
├── figures/
│   ├── hp_adaptive_v4_shockonly_paired_shock_immediate_update.png
│   ├── hp_adaptive_v4_shockonly_paired_shock_transmission_ratio.png
│   ├── hp_adaptive_v4_shockonly_paired_shock_peak_causal_damage.png
│   ├── hp_adaptive_v4_shockonly_paired_shock_cumulative_damage.png
│   ├── hp_adaptive_v4_shockonly_paired_shock_T50.png
│   ├── hp_adaptive_v4_shockonly_paired_shock_T90_failure.png
│   ├── transmission_ratio_seed_points.png
│   └── median_short_horizon_absolute_response.png
└── paper/
    └── <manuscript.pdf>
```

The angle-bracketed names are intentionally placeholders where the exact public repository filename has not been established here.

---

## Requirements

### MATLAB experiments

The MATLAB experiments use functions including `trnd` and `prctile`, so the **Statistics and Machine Learning Toolbox** is required.

### CIFAR-10 experiment

The Python pipeline uses PyTorch and the standard scientific Python stack. The exact environment used during the reported CPU runs included PyTorch 2.11.0+cpu.

At minimum, the repository should provide the dependencies actually imported by the released script. A typical environment includes:

```text
torch
torchvision
numpy
pandas
matplotlib
```

For a public reproducibility repository, generate `requirements.txt` directly from the final released implementation/environment rather than treating this list as an exact lockfile.

---

## Running the experiments

### Quadratic experiments

Run the released V4 MATLAB entry point from MATLAB. The shock-only implementation:

1. builds the 10-dimensional quadratic problem;
2. tunes optimizer learning rates on the Gaussian/unclipped reference condition;
3. freezes the selected rates;
4. runs the paired shock-control sweep;
5. computes immediate, trajectory, recovery, and effective-scale diagnostics;
6. generates the shock-response figures; and
7. exports the paired-shock summary.

The primary shock-summary output is:

```text
hp_adaptive_v4_shockonly_paired_shock_summary.csv
```

The implementation also stores the full shock experiment state in:

```text
hp_adaptive_v4_shockonly_results.mat
```

### CIFAR-10 experiment

The final calibrated three-seed pipeline is:

```text
cifar10_final_v2_calibrated_3seed.py
```

It performs learning-rate calibration, matched control/shock runs, pre-shock equality checks, aggregation, and figure generation.

Because command-line arguments and dataset paths are implementation-dependent, use the script's current CLI/help output for the exact invocation rather than copying machine-specific paths into this README.

---

## Key output files

| File | Contents |
|---|---|
| `hp_adaptive_v4_summary.csv` | quadratic distributional/tail summary |
| `hp_adaptive_v4_violation_summary.csv` | empirical threshold-exceedance probabilities |
| `hp_adaptive_v4_shockonly_paired_shock_summary.csv` | aggregated paired quadratic shock metrics |
| `aggregate_summary.csv` | aggregated CIFAR-10 results across seeds |
| `per_run_summary.csv` | optimizer/seed-level CIFAR-10 measurements |
| `calibration_results.csv` | CIFAR-10 learning-rate calibration results |
| `pre_shock_equality_checks.csv` | matched-pair pre-intervention equality checks |

If your exported CIFAR files currently contain suffixes such as `(1)`, renaming them to the clean names above before publishing the repository is recommended.

---

## Reproducibility and interpretation notes

- Gaussian and Student-\(t\) comparisons use variance matching.
- Learning rates are selected before final evaluation rather than retuned for individual tail conditions.
- Paired experiments use matched randomness and pre-intervention trajectories.
- The quadratic and CIFAR transmission ratios have **different implemented numerators**; the superscripts \(Q\) and \(C\) above make this explicit.
- Clipped Hill estimates are not interpreted as asymptotic tail exponents.
- Recovery statistics must be interpreted together with their censoring/failure fractions.
- The CIFAR-10 experiment is a controlled extreme-gradient intervention, not evidence that the natural CIFAR-10 gradient distribution is heavy-tailed.
- The three-seed CIFAR results are descriptive.
- The experiments do not establish a universal optimizer ranking. Different metrics—immediate attenuation, peak response, cumulative response, delay, and recovery—can produce different orderings.

---

## Paper

This repository accompanies the manuscript:

**Beyond Update Tails: Shock Attenuation and Trajectory Response of Adaptive Optimizers Under Heavy-Tailed Noise**

The repository is intended to make the reported experiments and numerical results auditable and reproducible. If a manuscript PDF is included, place it under `paper/` and update the filename in the repository tree above.

---

## Citation

If you use this code or build on the experimental methodology, please cite the associated paper. Until final publication metadata are available, use a placeholder citation rather than assigning conference proceedings information prematurely.

```bibtex
@misc{das2026beyondupdatetails,
  author = {Das, Reddhi},
  title  = {Beyond Update Tails: Shock Attenuation and Trajectory Response
            of Adaptive Optimizers Under Heavy-Tailed Noise},
  year   = {2026},
  note   = {Manuscript}
}
```

Replace this entry with the final DOI, venue, pages, and publication details if/when the paper is published.

---

## Author

**Reddhi Das**  
Department of Electrical and Computer Engineering  
North Carolina State University

---

## License

No license is asserted by this README. Before making the repository public, add an explicit `LICENSE` file if you intend others to reuse or redistribute the code, and ensure that the chosen license is compatible with all included third-party material.

---

## Keywords

Stochastic optimization · heavy-tailed noise · adaptive optimization · SGD · AdaGrad · Adam · shock response · trajectory response
