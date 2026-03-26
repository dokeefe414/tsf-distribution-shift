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
