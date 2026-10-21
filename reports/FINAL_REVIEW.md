# Final research-stage audit

## Preserved materials

The baseline and complex synthetic datasets, historical ARIMA/Kalman results and figures, and all Transformer/perturbation experiment scripts were preserved. The four uploaded plots are [directional sensitivity](figures/directional_sensitivity.png), [Gaussian covariance matrices](figures/correlated_gaussian_matrices.png), [forecast response/error](figures/correlated_gaussian_response.png), and [paired covariance effects](figures/correlated_gaussian_paired_effects.png).

## Correctness fixes

- Fixed early synthetic generator, ARIMA and Kalman output paths, which previously depended on running the terminal from the wrong directory.
- Repaired the syntax corruption at the end of combined_kalman.py.
- Removed mandatory TkAgg backend in the early ARIMA script.
- Replaced artificially zero pre-shift ARIMA error (copied training observations) with N/A because the historical script does not forecast that period.
- Clarified that older Kalman curves are posterior filtered estimates, not comparable pre-observation one-step forecasts.
- Corrected unsupported descriptions of chaos and variance collapse in the complex generator.
- Improved covariance plot layout to avoid a colorbar overlapping the final panel when figures are regenerated.
- Updated documents to reflect the completed Windows covariance experiment and original uploaded figures.

## Experimental findings and limitations

In this single-seed stationary Gaussian AR(1) baseline, fitted linear AR(1) achieved slightly lower clean test error than Transformer, while Transformer showed lower average absolute response to input perturbations. Directional noise revealed the models depend most on the latest observation; covariance changes with fixed marginal variance produced small response differences. These results cannot establish universal robustness or statistical significance across different trained models. The old ARIMA/Kalman metrics and figures remain historical; this cleanup did not rerun them or overwrite their results.

Next: multiple training seeds, controlled longer-lag generators, norm-matched perturbation directions, and long-horizon forecasting.
