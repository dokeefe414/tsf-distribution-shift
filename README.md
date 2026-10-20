# Time Series Forecasting Under Distribution Shift and Perturbations

Research project exploring how time-series forecasting models respond to nonstationarity, changing temporal dependencies, and **controlled input perturbations**.

**Researcher:** Daniel O'Keefe  
**Research focus:** understanding forecasting behavior, not just ranking models by prediction error.

## Research questions

1. How do forecasting models behave when the data-generating distribution changes?
2. How sensitive are one-step forecasts to small, controlled perturbations in historical observations?
3. How do perturbation **magnitude, direction, covariance structure, and distribution family** affect the forecast response?
4. How does model architecture or training change the response?
5. Later: what happens at longer forecasting horizons and different time scales?

The current priority is question 2, using a deliberately simple synthetic baseline. This isolates input sensitivity from the many simultaneous shifts in the earlier complex generator.

## Project stages

| Stage | Contents | Status |
| --- | --- | --- |
| 1. Simple shifting synthetic data | Abrupt mean/variance changes and nonlinear observation rule | Implemented |
| 2. More challenging time series | AR dependencies, nonlinear term, changing seasonality, outliers and varying noise | Implemented |
| 3. Classical methods | Rolling ARIMA, standard KF and adaptive KF comparisons | Implemented; methodology needs further refinement |
| **4. Perturbation response** | **AR(1) baseline, custom Transformer, Gaussian input perturbations** | **Initial implementation complete; exploratory results** |
| 5. Broader analysis | Covariance directions, distribution families, long-horizon forecasts, surrogate fidelity, information geometry | Planned |

## Run the new Transformer / perturbation experiment

Open this repository in Visual Studio Code. In a terminal opened at the repository root, run:

    python -m pip install -r requirements-perturbation.txt
    python code/generate_perturbation_data.py
    python code/test_perturbation_pipeline.py
    python code/transformer_model.py
    python code/perturbation_experiment.py

If Windows uses the Python launcher, replace `python` with `py`. Running the scripts from the root is easiest, although the scripts resolve their project paths relative to their own files.

The first script creates the data. The test script checks splits, shapes, and the network's forward/backward pass without training. Training writes a model checkpoint and held-out predictions. The final script generates perturbation measurements and a response plot.

**No pretrained forecasting model is used.** The Transformer-style block explicitly implements multi-head scaled dot-product attention using query, key and value projections in PyTorch.

### Experiment specification

| Element | Current setup |
| --- | --- |
| Generator | Stationary Gaussian AR(1): `x[t] = 0.7*x[t-1] + eta[t]` |
| Innovations | Independent `eta[t] ~ N(0, 1)` |
| Length | 5,000 observations after 200 burn-in observations |
| Split | Chronological 70% train / 15% validation / 15% test by *target time* |
| Scaling | Fit mean and standard deviation on the training portion only |
| Predictor | One-step-ahead custom Transformer-style encoder |
| Historical input length | `L = 32` |
| Network | 32-dimensional embeddings, four attention heads, one encoder block |
| Training | AdamW, MSE, early stopping by validation loss |
| Perturbation | `epsilon ~ N(0, sigma² I)` added to normalized **test inputs only** |
| Sigma | `0, 0.01, 0.05, 0.10, 0.20` times the training standard deviation |
| Repeats | 30 independent perturbation draws for each sigma |
| Response | Mean absolute difference between perturbed and clean forecasts |
| Accuracy | RMSE and NMSE measured against **unchanged original targets** |

The same trained forecaster is used for all perturbation levels. The model sees only historical inputs for each target, not the target itself or later values.

### Understand the method

- [Research proposal (HTML)](reports/perturbation_proposal.html) — objectives, mathematical setup, justified distributions and future research directions.
- [Perturbation methodology](reports/PERTURBATION_METHODS.md) — data flow, attention mathematics, leakage prevention, loss and evaluation.
- [Transformer code walkthrough](code/TRANSFORMER_WALKTHROUGH.md) — explanations tied to the source code.
- [Exploratory first-run results](reports/transformer_experiment/initial_results.md) — observations and limitations; regenerated outputs are local.
- [New generator](code/generate_perturbation_data.py), [model/trainer](code/transformer_model.py), [experiment](code/perturbation_experiment.py).

### What the first results do *not* demonstrate

A larger forecast response does **not** automatically imply worse forecasting accuracy; they are measured separately. This is a stationary baseline, not yet a controlled distribution-shift experiment. A single training seed does not establish general behavior; repeated training seeds, simple statistical/linear baselines, uncertainty intervals and sensitivity to the chosen window length belong in later experiments.

## Earlier distribution-shift research

- `code/generate_data.py`: a 400-point synthetic example with one abrupt distribution shift.
- `code/generate_complex_data.py`: a 600-point, three-regime synthetic process with lag dependence, nonlinearity, varying seasonality, variance shifts, trend and anomalies.
- `code/arima_complex.py`: an early rolling ARIMA experiment.
- `code/model_comparison.py`: rolling/windowed ARIMA versus standard and adaptive Kalman predictors. Prints selected ARIMA fitting information and records MSE, RMSE, MAE and normalized MSE.
- `kalman/code/`: earlier Kalman variants developed for coursework and initial research investigations.
- `data/synthetic/full_metrics.csv`: saved metrics from the earlier three-model comparison.
- `plots/`, `kalman/plots/`: earlier visualizations.

**Interpret earlier plots cautiously:** the adaptive KF in `model_comparison.py` can conflate distribution shifts with measurement noise, slowing its response. The older `arima_complex.py` copies training observations into its plotted prediction array, so its reported pre-shift MSE of zero is **not** a genuine forecast error. Some older scripts use working-directory-relative paths and have not been standardized. These are retained as historical experiments rather than treated as validated benchmarks.

## Repository organization

    code/
      generate_data.py
      generate_complex_data.py
      arima_complex.py
      model_comparison.py
      generate_perturbation_data.py       # NEW
      transformer_model.py                # NEW
      perturbation_experiment.py          # NEW
      test_perturbation_pipeline.py       # NEW
      TRANSFORMER_WALKTHROUGH.md          # NEW
    data/synthetic/                      # baseline datasets and historical metrics
    kalman/                              # earlier KF work
    plots/                               # earlier visualizations
    notebooks/
    reports/
      perturbation_proposal.html          # NEW
      PERTURBATION_METHODS.md             # NEW
      transformer_experiment/
        initial_results.md               # NEW
    requirements-perturbation.txt         # NEW

The new experiment generates its CSV, model weights, per-draw statistics, and plots locally. Those derived outputs are git-ignored to keep the repository focused on code, documentation, and reproducibility.

## Reproducibility and limitations

The generator, training and perturbation stages use fixed random seeds. Results can still vary across PyTorch versions, hardware, and platform configuration. The train/validation/test split uses chronological target indices; the validation/test *windows* may contain observations immediately preceding their split boundary, which is permitted in one-step forecasting when those observations are known at prediction time.

This project is exploratory work in progress. The next research step is to compare perturbation responses across independent training seeds and a simple baseline before moving to correlated Gaussian directions and longer-horizon responses.
