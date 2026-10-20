# Code walkthrough: correlated Gaussian perturbations

This document explains [correlated_gaussian_experiment.py](correlated_gaussian_experiment.py) so that each component can be described to an advisor.

## 1. What stays fixed?

The original 5,000-point stationary Gaussian AR(1) dataset, the chronological split, the 32-step historical input windows, the training-only normalization, the trained Transformer weights, and the original test targets all remain unchanged.

A fitted AR(1) linear baseline is rebuilt from the existing training segment. Its slope is learned by ordinary least squares. No new Transformer training occurs.

Imports reuse established helpers from:

- [transformer_model.py](transformer_model.py) for \`load_split\`, \`WINDOW\`, and output paths.
- [directional_perturbation.py](directional_perturbation.py) for frozen model loading and prediction.
- [linear_baseline_comparison.py](linear_baseline_comparison.py) for linear fitting and prediction and consistent metrics.

## 2. Designing the Gaussian covariance

\`correlation_matrix(profile, length)\` returns a symmetric 32×32 matrix C with \`C[i,i]=1\`. The profiles are independent, positive temporal, negative temporal, and common mode. A successful Cholesky factorization tests positive definiteness.

Two examples:

- Positive temporal: \`C[i,j] = 0.8 ** abs(i-j)\`.
- Negative temporal: \`C[i,j] = (-0.8) ** abs(i-j)\`.

Negative correlation at adjacent positions does **not** mean that the matrix is invalid; this alternating-sign Toeplitz matrix is positive definite for \`abs(rho)<1\`.

## 3. Generating multivariate Gaussian noise

\`correlated_noise(z, C, sigma)\` receives an \`(N,32)\` matrix of independent standard normal draws and computes \`sigma * z @ chol(C).T\` (row-vector notation). The output is a matrix with row covariance \`sigma² C\`. Each value is cast to float32 before adding it to the PyTorch input.

Every profile has equal diagonal variance, so total expected squared perturbation norm is **32 × sigma²**. Only off-diagonal covariance is deliberately changed.

## 4. Paired experiment loop

The outer loop runs \`repeats\` times. For each repeat, we draw one base \`z\` for every test window. We then transform **that same z** using each of the four covariance matrices.

Both forecasters evaluate the exact same perturbed windows for a given profile/draw. The clean forecast for each model is computed once and is subtracted to get the forecast response. The unchanged target is used to compute RMSE and NMSE.

This is a paired Monte Carlo comparison, which is preferable to changing random noise whenever covariance structure changes.

## 5. Metrics, paired effect, and output files

\`correlated_gaussian_per_repeat.csv\` records mean absolute/signed output response, RMSE, NMSE, and sampled squared noise norm.

\`correlated_gaussian_summary.csv\` averages over noise draws and includes the difference between perturbed RMSE and clean RMSE.

The pairing step joins each profile result with the *independent* result from the **same repetition and model**. It saves:

- \`correlated_gaussian_paired_deltas.csv\`: all matched differences.
- \`correlated_gaussian_paired_effects.csv\`: averages and across-draw standard errors.

Three plots separate the questions:

- **Covariance heatmaps** — what distributional change was introduced?
- **Response and RMSE bars** — how much did each forecast change, and how accurate was it?
- **Paired-difference bars** — how much did correlated noise change the result relative to independent noise at the same marginal variance?

Do not interpret error bars across repeated noise draws as variability from retraining the Transformer.

## 6. Analytical control

For linear AR(1), the forecast depends only on the newest input, so off-diagonal covariance cannot change its **expected** response. The expected mean absolute response is \`abs(slope) * sigma * training_std * sqrt(2/pi)\` for all four covariance choices.

This acts as a check of the experimental implementation. Differences in observed linear response across conditions are finite-sample Monte Carlo effects.

## 7. Interpreting small differences

The initial local run found the response of the Transformer changed very little across the four correlation structures. This is a **valid outcome** for this model and data generator; we should not tune the experiment to produce a preferred ranking.

The next scientific task is to replicate over multiple trained seeds or to change the generator in a controlled way, not to claim that a covariance structure is inherently best.

For the mathematical formulation, limitations, and instructions, see [CORRELATED_GAUSSIAN.md](../reports/CORRELATED_GAUSSIAN.md).
