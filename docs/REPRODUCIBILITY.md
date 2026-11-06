# Reproducibility and execution guide

## Requirements

Use Python 3.10+ (the code uses `dict[str, ...]` and `list[dict]` type annotations), NumPy and Matplotlib. The ODE workflow does **not** require torch, pandas, SciPy or Jupyter. Install from the repository root:

```powershell
py -m pip install -r requirements-ode.txt
```

On Linux/macOS replace `py` with `python3`. Use `python` if that points to your intended interpreter on Windows.

## Validate the environment

```powershell
py -m unittest discover -s code -p "test_controlled_ode.py" -v
py -m unittest discover -s code -p "test_ode_sensitivity.py" -v
```

The legacy controlled-ODE suite contains 5 checks; the new sensitivity suite contains 8 checks. Tests cover deterministic replays, intervention causality, the sampling grid, next-step indexing, variational integration, numerical derivative convergence, Wasserstein properties and seed-level bootstrap determinism. If a test fails on a different platform, report the complete traceback and library versions rather than adjusting scientific thresholds without analysis.

## Generate trajectories and early response metrics

```powershell
py code/run_controlled_ode.py
```

Writes into `reports/ode_experiment/`:
- `ode_trajectories.png`: paired unperturbed/control/parameter trajectories and irregular samples;
- `response_summary.csv`: three intervention types, \(\mathbb E|\Delta_G|\), \(\mathbb E|\Delta_F|\), mismatch and RMSE;
- `configuration.json`: the time grid, seeds, training and evaluation settings.

## Derivative validation, independent control seeds and OT

```powershell
py code/run_sensitivity_validation.py --seeds 12 --bootstrap 500 --epsilon 0.04
```

Writes into `reports/sensitivity_experiment/`:
- `tangent_vs_finite_difference.png`: \(s_u,s_a\) alongside centered finite differences at \(h=0.01\);
- `derivative_validation.csv`: mean and maximum derivative error for both interventions at four h values;
- `per_seed_response.csv`: paired generator/model response and RMSE for each unseen control seed;
- `response_bootstrap_summary.csv`: mean and bootstrap 95% percentile interval per metric/intervention;
- `distribution_at_t100.csv`: empirical Wasserstein-1/2 for state marginals at fixed time t=100, plus paired absolute change;
- `configuration.json`: actual experiment configuration and seed ranges.

The default generator seed (2026) supplies the **training** trajectory. Evaluation control seeds start at 3000 and are not used for fitting the linear forecaster. The first experiment fits a 12-lag linear model to target times before 58. Evaluation compares target times at and after 82, with the intervention starting at 80.

The bootstrap resamples **complete control realizations** (seed-level aggregate statistics), not time steps. Fixed bootstrap seed 11 gives deterministic replication. Confidence intervals do not capture network-training variability or systematic model bias.

Change `--epsilon` to investigate different physical intervention strengths. The derivative-validation h grid is intentionally fixed across such runs; it probes the limit as h tends to zero.

## Windows and directories

All paths in the new scripts resolve from `Path(__file__).resolve().parents[1]`, so the scripts can be executed from different current working directories. Unit tests expect discovery using `-s code`. Generated experiment directories are ignored by Git; the **small archived pilot report** is committed separately.

If `py` is not found, check `python --version` and installation. If NumPy or Matplotlib is missing, confirm that the Python used for installation is the same one running the scripts. If the plot does not open automatically, use:

```powershell
ii reports\sensitivity_experiment\tangent_vs_finite_difference.png
```

## Trust and limitations

A green test suite demonstrates software/numerical consistency for the covered cases; it is not proof of scientific generality. For a numerical convergence claim with respect to the *continuous* ODE, change the RK4 integration step and compare solutions on aligned grids; current finite-difference refinement checks an independent axis (intervention amplitude).

Do not infer that a benchmark is reproducible simply from a single saved graph. Report the input/control seed, actual package versions, commit, model checkpoint (when applicable), saved CSVs and complete configuration. Keep experimental inputs and historical results separate.
