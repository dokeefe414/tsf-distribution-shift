# Time Series Forecasting Under Distribution Shift

## Overview
This project studies how time series forecasting models perform when the underlying data distribution changes over time (distribution shift).

The goal is to evaluate whether models such as ARIMA can:
- maintain prediction accuracy under shifting conditions
- detect changes in the data
- adapt to new patterns

## Motivation
In real-world applications, time series data is often non-stationary, meaning its statistical properties (mean, variance, trend) change over time. Most forecasting models assume stable data, which can lead to poor performance when shifts occur.

This project explores model robustness in controlled settings using synthetic data.

## Methodology
- Generate synthetic time series data using stochastic processes
- Introduce controlled distribution shifts by modifying mean and variance
- Train baseline forecasting models (ARIMA)
- Evaluate performance before and after shifts

## Current Progress
- Studied stochastic process-based data generation
- Designed synthetic data framework
- Planning implementation of distribution shift scenarios

## Next Steps
- Implement synthetic data generator
- Visualize distribution shifts
- Train ARIMA model
- Evaluate prediction error

## Repository Structure
- `data/` → datasets
- `code/` → scripts for data generation and models
- `plots/` → visualizations

## Tools
- Python
- NumPy, Pandas
- Matplotlib
- Statsmodels (ARIMA)

## Author
Daniel O'Keefe
