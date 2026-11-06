# Research status and verification ledger

**Updated:** November 6, 2026  
**Research question:** Can one-step forecasting models follow the local and finite perturbation responses of an underlying dynamical system, beyond merely achieving low prediction error?

This ledger distinguishes mathematical identities, demonstrated code behavior, empirical observations and planned work. For instructions see [README](../README.md).

## Completed: nonlinear ODE and sampling

The scalar cubic ODE \(\dot x=a(t)x-x^3+u(t)\) is integrated via RK4 with a seeded piecewise-constant random control plus sinusoid. It includes a built-in parameter switch at t=60 and separate interventions at t=80. The regular sampler takes measurements every 0.25 time units; an irregular sampler preserves actual timestamps.

- Implementation: [controlled_ode_framework.py](../code/controlled_ode_framework.py).
- First comparison: [run_controlled_ode.py](../code/run_controlled_ode.py).
- Tests: [test_controlled_ode.py](../code/test_controlled_ode.py) (five original checks).
- Initial archived metrics: [CONTROLLED_ODE_FIRST_RUN.csv](CONTROLLED_ODE_FIRST_RUN.csv).
- The single-run trajectory plot is locally regenerated, not committed.

**Interpretation:** Controlled generator perturbations can be separated from observation-only noise. This does not establish phase-transition detection.

## Completed: mathematical sensitivity baseline

The control and coefficient variational equations were derived and implemented; the solution is compared with centered finite differences at four amplitudes. The new study also computes 12 independent control-seed paired-response measurements and fixed-time empirical W1/W2 state-marginal distances.

- Equations: [MATHEMATICAL_DERIVATIONS.md](../docs/MATHEMATICAL_DERIVATIONS.md).
- Implementation: [ode_sensitivity.py](../code/ode_sensitivity.py), [run_sensitivity_validation.py](../code/run_sensitivity_validation.py).
- Verification: [test_ode_sensitivity.py](../code/test_ode_sensitivity.py) (eight tested checks).
- Numerical record: [SENSITIVITY_VALIDATION.md](SENSITIVITY_VALIDATION.md).
- Uncertainty, leakage and competing baselines: [EXPERIMENT_PROTOCOL.md](../docs/EXPERIMENT_PROTOCOL.md).

**Pilot findings:** For h=0.01, full-grid derivative MAE is approximately 0.00018776 for the control offset and 0.00001169 for the parameter offset. With intervention amplitude 0.04, the mean response mismatch over 12 evaluation control seeds is approximately 0.00072526 (control) and 0.00045510 (parameter). These are specific to the pilot generator, linear baseline and numerical scheme; there is no established general significance.

## Completed: earlier time-series research

| Work | Evidence | Limitation |
| --- | --- | --- |
| Piecewise/nonlinear synthetic distribution shifts | [generate_data.py](../code/generate_data.py) and [generate_complex_data.py](../code/generate_complex_data.py) | Exploratory generators, not validated physical systems |
| Early ARIMA/Kalman comparisons | [model_comparison.py](../code/model_comparison.py), [kalman/](../kalman/) | Some historical methods/plots are not methodologically comparable |
| Stationary Gaussian AR(1), 5000 observations | [generate_perturbation_data.py](../code/generate_perturbation_data.py) | No controlled external input |
| Custom attention forecaster | [transformer_model.py](../code/transformer_model.py), [recorded results](transformer_experiment/initial_results.md) | One small model and training seed |
| Fitted AR(1) comparison | [validation record](transformer_experiment/linear_baseline_validation.md) | Easier for a linear model on linear AR(1) data |
| Directional perturbations | [plot](figures/directional_sensitivity.png), [findings](transformer_experiment/directional_findings_2026-10-20.md) | Observational input sensitivity, not generator perturbation |
| Correlated Gaussian input perturbations | [report](CORRELATED_GAUSSIAN.md), [figures](figures/) | Fixed input marginal variance, limited repetitions |

**Historical checked figures:** AR(1) fitted slope approximately 0.716396; clean held-out RMSE 1.004406 (AR(1)) vs 1.022539 (Transformer); at normalized sigma 0.10, corresponding mean absolute input-noise responses 0.082423 and 0.070765. These results are from the earlier stationary AR(1) program, *not* the new ODE sensitivity experiment.

## Planned, not yet implemented

- RK4 dt-refinement with a finer reference integration.
- Independent model-fitting seeds and larger random-control trajectory ensembles.
- Matched persistence/AR, GRU, LSTM and Transformer forecasters on controlled ODE data.
- Equal information sets: with/without known control as a covariate.
- Gradual parameter ramps with properly defined pre-transition detection, held-out false-alarm rates and lead time.
- Online adaptation, separately evaluated from detection.
- Statistically defined information geometry, Fisher information, and more advanced transport analysis only after the relevant probability model is specified.

The mathematical sensitivity equations, the bootstrap, and empirical 1D Wasserstein calculations are established methods. Their existence alone is not a novel research contribution.
