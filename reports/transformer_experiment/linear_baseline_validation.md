# Fitted AR(1) versus Transformer: confirmed Windows comparison

**Date:** October 20, 2026  
**Source:** researcher-provided VS Code PowerShell output after running the GitHub scripts (the summary values below were transcribed from the displayed output and rounded to six decimals). The raw per-repeat CSV and checkpoint remain on the local computer.

## What was run

After a successful git pull, these commands completed:

    py code/test_linear_baseline.py
    py code/linear_baseline_comparison.py

The test printed: **PASS: AR(1) fitting, directional response, and metrics**. The fitted AR(1) slope was **0.716396**, with a normalized intercept of approximately **-0.000109**. The generator coefficient is 0.7.

## Why this is a fair baseline

For the Gaussian AR(1) generator (x[t] = 0.7*x[t-1] + independent N(0,1) noise), the theoretically appropriate one-step conditional mean depends on the most recent observation. The linear predictor is fitted by ordinary least squares using **training target indices only**; the true coefficient is not supplied to the fit. Transformer predictions and linear predictions are evaluated on identical held-out test target times, with the same normalized input windows, original targets, and paired Gaussian noise draws. The trained Transformer remains unchanged throughout.

### Clean test accuracy

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Fitted linear AR(1) | 1.004406 | 0.790132 |
| Transformer | 1.022539 | 0.808803 |

This fitted linear AR(1) has lower test error on this one dataset and seed. That is consistent with the known linear data-generating process; it does **not** establish that linear models are generally better than Transformers.

### Paired Gaussian input perturbations (30 draws at each sigma)

| Noise sigma / train std | Linear mean abs response | Transformer mean abs response | Linear RMSE | Transformer RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.000000 | 0.000000 | 1.004406 | 1.022539 |
| 0.01 | 0.008284 | 0.007119 | 1.004521 | 1.022638 |
| 0.05 | 0.041157 | 0.035094 | 1.005535 | 1.023279 |
| 0.10 | 0.082423 | 0.070765 | 1.010917 | 1.027424 |
| 0.20 | 0.164668 | 0.140584 | 1.027068 | 1.040837 |

[Recorded full summary table (CSV)](observed_linear_comparison_2026-10-20.csv)

For Gaussian independent input noise, the fitted AR(1) has an analytical expected response magnitude:

**E|change| = |fitted slope| × sigma × training std × sqrt(2/pi).**

In the console output, the theoretical values at sigma = 0.01, 0.05, 0.10, 0.20 were approximately 0.008250, 0.041250, 0.082499, 0.164998; they closely match the empirical means. The comparison therefore also checks that the experiment behaves as expected for a simple known model.

## Interpretation and limitations

1. The Transformer is **less sensitive to this isotropic Gaussian perturbation** at every tested nonzero sigma, but its original-target RMSE is **higher**. Sensitivity and accuracy are distinct.
2. Responses are averages of absolute changes in each model's own forecast, not differences between the two models' forecasts.
3. Both models receive exactly the same noise realizations within the comparison. Results from the separately run perturbation script are consistent but are not a substitute for matched noise when making a paired comparison.
4. Only one trained Transformer seed and one Gaussian AR(1) series are included. A difference in response does not demonstrate broad robustness under distribution shifts.
5. Training-only normalization and fixed original targets prevent data leakage and changes of target definition during the test.

The checkpoint, generated PNG, and per-repeat outputs are excluded from Git. The recorded table and methodology are committed. For the next experiment, test each of the 32 historical positions separately with [directional_perturbation.py](../../code/directional_perturbation.py); this is an additional *directional sensitivity* test, not a result already confirmed from Windows.

## Reproduce on the same machine

    git pull origin main
    py code/test_linear_baseline.py
    py code/linear_baseline_comparison.py

Generated files live under **reports/transformer_experiment/**: linear_vs_transformer_clean_metrics.csv, linear_vs_transformer_per_draw.csv, linear_vs_transformer_summary.csv, linear_baseline_parameters.json, and linear_vs_transformer.png.
