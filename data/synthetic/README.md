# Synthetic Dataset: synthetic_shift_v1

## Description
This dataset contains a synthetic time series with a controlled distribution shift.

## Generation Process
- First half (t = 0–199):
  - X_t ~ N(10, 1)
- Second half (t = 200–399):
  - X_t ~ N(20, 2)

The observed series is generated as:
Y_t = X_t^2 + 3 + noise

where noise ~ N(0, 1)

## Purpose
This dataset is used to evaluate how time series forecasting models perform under distribution shift.

## Notes
- The shift occurs at t = 200
- The shift is both a mean and variance change

## Other synthetic datasets in this directory

- **complex_time_series.csv**: 600 points from the earlier three-regime generator in [generate_complex_data.py](../../code/generate_complex_data.py), with non-linear AR(2)-style lag dependence, changing seasonality/noise, offsets, trend and outliers. It is deliberately challenging but many factors change simultaneously, so effects cannot always be causally attributed to a single shift.
- **perturbation_ar1.csv** (generated locally, not checked into Git): 5,000 points after burn-in from a **stationary Gaussian AR(1)** process with coefficient 0.7 and N(0,1) innovations, created by [generate_perturbation_data.py](../../code/generate_perturbation_data.py). This is the control dataset for one-step Transformer and fitted AR(1) perturbation-response experiments.
- **arima_fit_log.csv**, **mse_results.csv**, **full_metrics.csv**: saved exploratory metrics from the earlier ARIMA/Kalman phase, not metrics from the new Transformer experiment.

### Notation note

In the first shifting dataset, the second argument in NumPy's normal sampling call is a **standard deviation**, not a variance. The observed output is a nonlinear function of the latent Gaussian values; it should not itself be described as Gaussian.

The generated stationary AR(1) baseline intentionally has **no generator distribution shift**. Later input perturbations modify historical windows supplied to the predictors, not the generation process or original forecast targets. These are different experiments.
