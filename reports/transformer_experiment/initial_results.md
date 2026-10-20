# Initial Transformer perturbation experiment (exploratory)

These measurements come from the first locally run starter experiment (training seed 42, perturbation seed 123). The original local experiment used the same AR(1) generator and core model/training/perturbation settings as the documented scripts. **They are not the result of a new GitHub Actions or remote training run.** Re-run the scripts to produce checkpoint, per-draw CSV, and updated metrics on your machine.

## Test accuracy of the clean Transformer

| Metric | First run |
| --- | ---: |
| Test MSE (original units) | 1.045586 |
| Test MAE (original units) | 0.808803 |
| Best validation MSE (normalized units) | 0.495638 |

These results come from a single fitted Transformer on 750 held-out target time indices.

## Input perturbation response

Mean absolute change in prediction, averaged over held-out test windows and 30 perturbation draws per sigma:

| Gaussian input noise sigma (normalized units) | Mean absolute forecast change |
| ---: | ---: |
| 0.00 | 0.000000 |
| 0.01 | 0.007119 |
| 0.05 | 0.035094 |
| 0.10 | 0.070765 |
| 0.20 | 0.140584 |

![Initial response curve](perturbation_response.svg)

The first curve increases approximately with sigma over these five points. This describes **sensitivity**, not automatic deterioration in forecast accuracy. The perturbation script also computes RMSE and NMSE against unchanged test targets so accuracy can be analyzed independently.

## Interpretation and cautions

- This data is a stationary Gaussian AR(1) series with no intentionally inserted regime changes.
- Isotropic independent noise modifies temporal covariance, not merely variance at each time.
- Fixed perturbation repeats do not capture variability due to retraining the Transformer.
- A suitable next comparison is the optimal one-step AR(1) linear forecast, against which the Transformer can be evaluated for both forecast accuracy and response structure.
- Response magnitude depends on the units and scaling; use the same scale for cross-model comparison.
- Numerical results may vary with PyTorch/hardware versions. The source code, parameters, and seeds are recorded for reproducing the experiment.
