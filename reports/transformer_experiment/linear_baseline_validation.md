# Fitted AR(1) versus Transformer — local validation

## Why this benchmark?

The generator is **Gaussian AR(1)**: `x[t] = 0.7 * x[t-1] + eta[t]`, where eta[t] is independent Gaussian noise with standard deviation 1. For this process, the correct conditional-mean one-step forecast depends only on the most recent observation. A simple fitted AR(1) therefore provides a scientifically informative baseline.

We fit `normalized_x[t] = a + b * normalized_x[t-1]` by least squares on **training target indices only**. We do *not* supply the known coefficient 0.7 to the fit. Predictions are transformed back into original units. The evaluation uses the exact same test indices, fixed Transformer weights, same original targets and **paired Gaussian input perturbations** for both models.

## Results from a separate local validation

The comparison script was tested using the locally retained research starter's generated series and model checkpoint (same seed and core model configuration as the project). This is **not yet a comparison run confirmed on the student's Windows machine**; rerun it locally after pulling the updated GitHub repository.

| Clean test metric | Fitted AR(1) | Transformer |
| --- | ---: | ---: |
| RMSE | 1.004406 | 1.022539 |
| MAE | 0.790132 | 0.808803 |

Estimated normalized AR(1) slope: **0.716396**, close to the generator's value of 0.7.

| Perturbation σ (normalized) | Fitted AR(1) mean absolute response | Transformer mean absolute response | Fitted AR(1) mean RMSE | Transformer mean RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.000000 | 0.000000 | 1.004406 | 1.022539 |
| 0.01 | 0.008284 | 0.007119 | 1.004521 | 1.022638 |
| 0.05 | 0.041157 | 0.035094 | 1.005535 | 1.023279 |
| 0.10 | 0.082423 | 0.070765 | 1.010917 | 1.027424 |
| 0.20 | 0.164668 | 0.140584 | 1.027068 | 1.040837 |

In this one local comparison, the fitted linear model has **lower clean test error** and **larger average input-perturbation response** than the Transformer. A larger response means a larger change in the forecast, not necessarily a worse predictor. The simple fitted AR(1) reflects the known data-generating structure.

For the fitted AR(1), the theoretical expected absolute forecast change under independent Gaussian perturbation is:

`E|Δ forecast| = |b| * σ * training_std * sqrt(2/pi)`.

Only the perturbation of the most recent input coordinate affects this AR(1) forecast; the earlier 31 coordinates contribute nothing. The Transformer can use all 32 coordinates.

## Reproduce this comparison on your Windows checkout

After the existing generator, Transformer training, and perturbation scripts have finished:

```powershell
git pull origin main
py code/test_linear_baseline.py
py code/linear_baseline_comparison.py
```

The comparison writes:
- `linear_vs_transformer_clean_metrics.csv`
- `linear_vs_transformer_per_draw.csv`
- `linear_vs_transformer_summary.csv`
- `linear_baseline_parameters.json`
- `linear_vs_transformer.png`

All appear under `reports/transformer_experiment/`. These **machine-generated** outputs are intentionally ignored by Git; this Markdown page preserves the local validation summary, and your observed Windows Transformer results are preserved separately in [initial_results.md](initial_results.md).

## Limitations and next steps

Results are from one seed and one dataset. The linear generator makes the fitted AR(1) unusually appropriate; do not extrapolate a general ranking of Transformer and classical methods from this test. Next experiments should repeat training seeds, change distribution families, test perturbations along individual time coordinates, and examine correlated Gaussian directions.
