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
| **4. Perturbation response** | **AR(1) generator, custom Transformer, Gaussian input perturbations** | **Executed successfully on Windows; first results recorded** |
| **4b. Model baseline comparison** | **Fitted linear AR(1) vs. frozen Transformer, shared perturbations** | **Completed successfully in VS Code; metrics recorded** |
| **4c. Directional sensitivity** | **Perturb each historical input position independently** | **Completed in VS Code; original plot uploaded to GitHub** |
| **4d. Correlated Gaussian perturbations** | **Change input covariance without changing marginal noise variance** | **Implemented and locally tested; Windows run next** |
| 5. Broader analysis | Covariance directions, distribution families, long-horizon forecasts, surrogate fidelity, information geometry | Planned |

## Run the new Transformer / perturbation experiment

Open this repository in Visual Studio Code. In a terminal opened at the repository root, run:

    python -m pip install -r requirements-perturbation.txt
    python code/generate_perturbation_data.py
    python code/test_perturbation_pipeline.py
    python code/transformer_model.py
    python code/perturbation_experiment.py
    python code/test_linear_baseline.py
    python code/linear_baseline_comparison.py
    python code/test_directional_perturbation.py
    python code/directional_perturbation.py
    python code/test_correlated_gaussian.py
    python code/correlated_gaussian_experiment.py

If Windows uses the Python launcher, replace `python` with `py`. Running the scripts from the root is easiest, although the scripts resolve their project paths relative to their own files.

The generator creates the data; the first smoke test verifies split boundaries and forward/backward computation. Training saves a checkpoint and held-out predictions. The perturbation script generates the noise-magnitude response plot. Next, the baseline test and comparison evaluate a fitted linear AR(1) on the same held-out examples and paired noise draws as the frozen Transformer. The final two commands test and measure **position-by-position directional response** for all 32 historical observations. Existing local checkpoints are reused; retraining is not needed for directional evaluation.

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
| Repeats | 30 independent perturbation draws for each sigma in the isotropic sweep; 12 per input position by default in the new directional test |
| Response | Mean absolute difference between perturbed and clean forecasts |
| Accuracy | RMSE and NMSE measured against **unchanged original targets** |

The same trained forecaster is used for all perturbation levels. The model sees only historical inputs for each target, not the target itself or later values.

### Successful execution and recorded results (October 20, 2026)

The scripts were run successfully in VS Code on Windows. The training smoke test passed, and Transformer training stopped after 10 epochs with the best validation loss at epoch 5.

| Confirmed result from Windows run | Value |
| --- | ---: |
| Clean test MSE | 1.045586 |
| Clean test MAE | 0.808803 |
| Best normalized validation MSE | 0.495638 |
| Mean absolute response, sigma=0.01 | 0.007119 |
| Mean absolute response, sigma=0.10 | 0.070765 |
| Mean absolute response, sigma=0.20 | 0.140584 |

See [confirmed training and perturbation results](reports/transformer_experiment/initial_results.md), with [epoch-by-epoch training losses](reports/transformer_experiment/observed_training_2026-10-20.csv) and the [recorded 30-repeat summary](reports/transformer_experiment/observed_perturbation_2026-10-20.csv).

The fitted AR(1) versus Transformer comparison **also completed successfully in the researcher's Windows VS Code terminal**. Its fitted slope was 0.716396, with clean test RMSE **1.004406** (linear) versus **1.022539** (Transformer). At sigma=0.10, mean absolute forecast response was **0.082423** (linear) versus **0.070765** (Transformer). See the [confirmed comparison report](reports/transformer_experiment/linear_baseline_validation.md) and [recorded table](reports/transformer_experiment/observed_linear_comparison_2026-10-20.csv).

**Directional sensitivity was also run successfully on Windows.** The fitted AR(1) responds only to the most recent input (position 32), while the Transformer shows smaller nonzero responses to earlier positions. The [original directional PNG](reports/figures/directional_sensitivity.png) was subsequently uploaded directly to GitHub. The exact numeric directional CSVs have **not** been published. [Read the directional findings](reports/transformer_experiment/directional_findings_2026-10-20.md).

**Next research experiment:** multivariate Gaussian input perturbations with four covariance structures but the same marginal noise variance. The new code and tests have been **verified in a separate local development environment**; this experiment has **not yet been executed on your Windows checkpoint**. See [correlated Gaussian methodology](reports/CORRELATED_GAUSSIAN.md).

### Understand the method

- [Research proposal (HTML)](reports/perturbation_proposal.html) — objectives, mathematical setup, justified distributions and future research directions.
- [Perturbation methodology](reports/PERTURBATION_METHODS.md) — data flow, attention mathematics, leakage prevention, loss and evaluation.
- [Transformer code walkthrough](code/TRANSFORMER_WALKTHROUGH.md) — explanations tied to the source code.
- [Confirmed Windows run](reports/transformer_experiment/initial_results.md) — actual console results, training losses, sensitivity and forecast error.
- [Confirmed AR(1) versus Transformer comparison](reports/transformer_experiment/linear_baseline_validation.md) — Windows results, paired perturbations, theory and interpretation.
- [Directional perturbation protocol](reports/DIRECTIONAL_PERTURBATIONS.md) — all 32 input positions, formulas, outputs and interpretation.
- [Directional findings, October 20](reports/transformer_experiment/directional_findings_2026-10-20.md) — confirmed screenshot observations, with precise-value limitations.
- [Uploaded directional graph](reports/figures/directional_sensitivity.png) — native image from the researcher's experiment.
- [Correlated Gaussian experiment](reports/CORRELATED_GAUSSIAN.md) — covariance patterns, formulas, evaluation design, locally verified behavior and next commands.
- [Correlated Gaussian code walkthrough](code/CORRELATED_GAUSSIAN_WALKTHROUGH.md) — line-by-line conceptual explanation of the new experiment.
- [Research status and advisor roadmap](reports/RESEARCH_STATUS.md) — progress through the September meeting goals.
- [New generator](code/generate_perturbation_data.py), [model/trainer](code/transformer_model.py), [experiment](code/perturbation_experiment.py).

### What the first results do *not* demonstrate

A larger forecast response does **not** automatically imply worse forecasting accuracy; they are measured separately. This is a stationary baseline, not yet a controlled distribution-shift experiment. A single training seed does not establish general behavior; repeated training seeds, uncertainty intervals and sensitivity to the chosen window length belong in later experiments. The new linear comparison is a **baseline**, not evidence that one architecture is always superior.

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
      linear_baseline_comparison.py        # NEW
      test_linear_baseline.py              # NEW
      directional_perturbation.py          # NEW
      test_directional_perturbation.py     # NEW
      correlated_gaussian_experiment.py    # NEW
      test_correlated_gaussian.py          # NEW
      CORRELATED_GAUSSIAN_WALKTHROUGH.md   # NEW
      publish_directional_results.py       # NEW, archive exact Windows outputs
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
        observed_training_2026-10-20.csv  # NEW, printed training losses
        observed_perturbation_2026-10-20.csv # NEW, printed summaries
        linear_baseline_validation.md     # NEW
        observed_linear_comparison_2026-10-20.csv # NEW
        directional_findings_2026-10-20.md # NEW, screenshot-based findings
    DIRECTIONAL_PERTURBATIONS.md           # NEW
    CORRELATED_GAUSSIAN.md                # NEW
    figures/
      directional_sensitivity.png         # Original uploaded graph
    RESEARCH_STATUS.md                     # NEW
    requirements-perturbation.txt         # NEW

The experiments generate datasets, checkpoint weights, per-draw statistics and PNG plots locally. Those large/reproducible outputs are git-ignored; the repository **does retain** small CSV snapshots transcribed from the confirmed Windows console output, code, and Markdown reports for reproducibility.

## Reproducibility and limitations

The generator, training and perturbation stages use fixed random seeds. Results can still vary across PyTorch versions, hardware, and platform configuration. The train/validation/test split uses chronological target indices; the validation/test *windows* may contain observations immediately preceding their split boundary, which is permitted in one-step forecasting when those observations are known at prediction time.

This project is exploratory work in progress. The linear and coordinate-wise comparisons ran on Windows; the directional image is uploaded. The new covariance experiment was tested in a separate local environment but still needs a run in VS Code using your existing trained model. After that, compare multiple training seeds, analyze norm-matched directions, and investigate longer-horizon forecasting.
