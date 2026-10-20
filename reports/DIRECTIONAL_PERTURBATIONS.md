# Directional perturbation study: experiment protocol

**Status:** The researcher has now generated the directional heatmap on Windows and provided a screenshot (October 20, 2026). The qualitative outcome is documented in [directional findings](transformer_experiment/directional_findings_2026-10-20.md). The exact native PNG/CSVs still need to be archived from that computer; see [publishing instructions](figures/README.md).

## Objective

Earlier experiments independently perturbed *every* position in a 32-value window and found a larger average change in the fitted linear AR(1) forecast than in the Transformer forecast. That does not identify **which past observations** drive the predictions. This experiment probes one input coordinate at a time while holding every other coordinate, model weight, and target fixed.

## Definitions

At target time t, the input is the 32-vector X[t] = (x[t-32], ..., x[t-1]), normalized using training data only. The clean prediction is f(X[t]). Let e[j] be the unit vector at position j, where j=1 means the oldest observation x[t-32] and j=32 means the newest x[t-1].

For each j and test window, independently draw **z ~ N(0, sigma^2)** in normalized units and form:

**X_perturbed = X + z * e[j]**

The response is **delta = f(X_perturbed) - f(X)**. The primary statistic is the mean of |delta| over the 750 held-out target indices and then over 12 independent draws per coordinate.

Use sigma=0.10 by default, matching the earlier magnitude sweep's normalized unit system. The same z draws for each test window are used for both the fitted AR(1) and frozen Transformer, making responses directly comparable.

Crucially, **sigma here applies to just one coordinate**. Earlier isotropic experiments perturbed all 32 coordinates simultaneously. The per-coordinate and all-coordinate experiments have different total noise energy, so their response magnitudes should not be directly interpreted as equivalent exposure levels.

## Expected behavior of the fitted AR(1)

The fitted linear forecast is f(X) = a + b * X[32], followed by conversion back to original series units. It depends only on the most recent observation.

- Positions 1–31: response should be exactly zero (up to numerical precision).
- Position 32: mean |delta| should approach **|b| × sigma × training_std × sqrt(2/pi)** when averaged across many independent Gaussian perturbations.
- The fit learns b from the training set; it does not use the generator coefficient 0.7 directly.

The Transformer can respond to any of the 32 positions because its attention operates over the full history. Which positions matter more is an *empirical* finding and should not be assumed in advance.

## Implementation

Scripts:

- [Directional analysis](../code/directional_perturbation.py) — loads training data, same frozen Transformer checkpoint, and the OLS-fitted linear baseline. Evaluates each of the 32 positions using paired noise draws.
- [Directional smoke test](../code/test_directional_perturbation.py) — verifies only one coordinate changes, clean inputs stay unchanged, AR(1) ignores old positions, and out-of-range positions fail cleanly.
- [Transformer walkthrough](../code/TRANSFORMER_WALKTHROUGH.md) — explains the forecaster architecture and current training assumptions.

The new script checks that the checkpoint's window length, data split and normalization parameters match the current data, rather than silently evaluating a stale model. Predictions and RMSE are expressed in original signal units; perturbation sigma is expressed relative to training standard deviation.

## Run in VS Code (PowerShell)

Start in your Git-cloned project folder. **Do not retrain the Transformer if its checkpoint and generator data are still present**.

    git pull origin main
    py code/test_directional_perturbation.py
    py code/directional_perturbation.py

For more perturbation repeats or a different magnitude:

    py code/directional_perturbation.py --sigma 0.05 --repeats 20

The analysis prints progress for positions 1 through 32 and, at the end, shows the five most responsive positions for each model and the fitted AR(1) theoretical response.

## Generated files (ignored by Git)

In reports/transformer_experiment/:

| File | Contents |
| --- | --- |
| directional_per_repeat.csv | Each model, input position, noise repeat, absolute/signed response and original-target RMSE |
| directional_summary.csv | Averaged response, across-repeat standard deviation and averaged RMSE for each model/position |
| directional_sensitivity.png | Two-row sensitivity heatmap plus line comparison (positions 1–32) |

The across-repeat standard deviation quantifies **variation from perturbation draws for one fixed model**. It does not measure uncertainty across different training seeds or datasets.

## Development verification versus researcher's results

The deterministic smoke test passed locally. A complete default 32-position run with 12 repeats also completed locally using the retained starter model: the fitted AR(1) had zero response at positions 1–31, and position 32 gave approximately **0.08247** versus theoretical **0.08250**. The strongest Transformer response was also at position 32 (approximately **0.06702**), with smaller responses at some older observations.

**The numbers in the previous paragraph are from the separate development-environment verification, NOT exact values read from the researcher's screenshot.** The user subsequently supplied a Windows-generated directional plot with the same qualitative pattern: AR(1) responds at position 32 only, while Transformer also responds weakly at older positions. The plot visually shows normalized sigma=0.1; the exact 32-position data will be published from the Windows files using the [publishing helper](../code/publish_directional_results.py).

## Limitations and later extensions

- This is a **coordinate perturbation** experiment, not yet a general directional-gradient or Fisher-geometry analysis.
- Repeated noise draws and overlapping historical windows are not independent training replications. Avoid reporting confidence intervals as if every response were a separate independent data sample.
- Measuring one coordinate at a time changes a specific feature. It does not necessarily preserve the joint distribution of a Gaussian AR process.
- The next step is to compare **norm-matched multi-coordinate directions** (e.g., uniform/alternating/eigenvectors) and later correlated multivariate Gaussian noise, then examine behavior across multiple trained seeds and longer horizons.


## Preserve this specific Windows run

The current full-resolution figure and exact machine-generated measurements are in the locally ignored `reports/transformer_experiment/` directory. To publish these for your advisor, **do not rerun the experiment first**. Pull the latest code, then run:

```powershell
git pull origin main
py code/publish_directional_results.py --label 2026-10-20
git add reports/figures/
git commit -m "Archive October 20 directional sensitivity figure and metrics"
git push origin main
```

The publisher checks for 32 positions × 2 models, consistent repetitions and a single sigma, then copies the native PNG, summarized and per-repeat CSVs, and model configuration to versioned `reports/figures/` paths. It generates a linked Markdown report with the exact highest-response positions. It does **not** upload checkpoint weights. The qualitative screenshot report is at [directional findings](transformer_experiment/directional_findings_2026-10-20.md).
