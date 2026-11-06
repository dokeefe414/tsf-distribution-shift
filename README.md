# Time-Series Forecasting Under Distribution Shift

**Researcher:** Daniel O'Keefe  
**Focus:** mathematically grounded sensitivity analysis of dynamical systems and one-step time-series forecasters.

This repository investigates a question that goes beyond ordinary forecasting accuracy: **when a system responds to a controlled change, does a forecasting model reproduce that response faithfully?**

The project combines a nonlinear ODE generator, stochastic control sequences, trajectory sampling, one-step forecasting, and *separate* measurements of generator sensitivity, model sensitivity and distribution change. It also retains earlier Gaussian AR(1), Transformer, ARIMA and Kalman experiments as exploratory baselines.

> **Research status (November 2026):** The ODE generator and a linear one-step benchmark are implemented. Tangent-linear sensitivity equations, centered finite-difference checks, multi-seed evaluations and empirical 1D optimal transport are now implemented and tested. **GRU/LSTM training on the ODE generator, mathematical detection guarantees, Riemannian/Fisher geometry and online adaptation have not been completed.** The sensitivity equations are established mathematics, not a claimed new theorem.

## Scientific motivation

Forecast error and response fidelity answer different questions. A model can predict the next value accurately on ordinary data, but still change its prediction incorrectly after a physical intervention. Conversely, a model that barely responds to noisy inputs is not necessarily a good physical forecaster.

We therefore distinguish:

1. **System response** \(\Delta_G\): change in the *true* trajectory after modifying a control or generator parameter.
2. **Model response** \(\Delta_F\): change in a *frozen model's* prediction when presented with the corresponding altered observation window.
3. **Response mismatch** \(\lvert \Delta_F-\Delta_G\rvert\): how closely the predicted change tracks the actual change.
4. **Distribution response**: how the law of states across independent random control realizations changes, initially measured with empirical one-dimensional Wasserstein distances.

These measurements should be reported together with forecast RMSE, numerical error and independent-run uncertainty. Their significance and any claim of novelty must be tested against established alternatives.

## Current generator

The nonlinear state follows:

\[
\frac{dx}{dt} = a(t)x-x^3+b\,u(t), \qquad x(0)=0.1,\quad b=1.
\]

- \(a(t)=-0.8\) before time 60 and \(a(t)=0.6\) afterwards. The switch is *built into the generator*.
- \(u(t)=0.08\sin(0.2t)+0.08Z_k\), where each \(Z_k\) is a seeded standard-normal random control level held for about one time unit.
- The state is integrated with fourth-order Runge–Kutta at \(dt=0.05\) and observed regularly every 0.25 time units (or via an optional irregular sampler).
- At time 80 an experiment can change either \(u(t)\) or \(a(t)\). Baseline and intervention runs use the **same initial condition and random control realization**.
- Random control does not make this an Itô SDE. The simulation is a deterministic ODE **conditional on a sampled control path**.

The baseline forecaster is a 12-lag ridge linear predictor trained on one pre-switch trajectory. It is deliberately simple and kept frozen for sensitivity comparison. Future neural networks will be compared on matched data, horizons and information sets.

## Run it locally

Open a terminal at the repository root. On Windows with the Python launcher:

```powershell
git pull origin main
py -m pip install -r requirements-ode.txt
py -m unittest discover -s code -p "test_controlled_ode.py" -v
py -m unittest discover -s code -p "test_ode_sensitivity.py" -v
py code/run_controlled_ode.py
py code/run_sensitivity_validation.py --seeds 12 --bootstrap 500 --epsilon 0.04
```

Use `python` instead of `py` on systems without the Windows launcher. The ODE workflow requires only NumPy and Matplotlib, not PyTorch. The earlier Transformer experiments have their own dependency list and commands below.

**Generated local files** (ignored by Git): `reports/ode_experiment/` and `reports/sensitivity_experiment/`. These include plots, per-seed CSVs, derivative errors, bootstrap summaries, Wasserstein distances and configuration JSONs.

For installation, expected outputs, seed behavior, troubleshooting, and numerical checks, see [reproducibility guide](docs/REPRODUCIBILITY.md).

## Mathematically validated sensitivity

For an intervention beginning at \(t_0\), define the local control sensitivity \(s_u=\partial x/\partial\varepsilon\) and parameter sensitivity \(s_a=\partial x/\partial\delta\), both evaluated at zero intervention:

\[
\dot s_u=(a-3x^2)s_u+b\,\mathbf{1}_{t\ge t_0},
\qquad
\dot s_a=(a-3x^2)s_a+x\,\mathbf{1}_{t\ge t_0}.
\]

These equations follow from the chain rule and are integrated alongside the baseline trajectory. They are checked against centered differences \([x_{+h}-x_{-h}]/(2h)\) at several step sizes. The validation compares numerical derivatives **for the same discretized model**, not exact continuous-time integration error.

In a 12-seed exploratory evaluation, interventions at time 80 were compared with a frozen linear forecaster using samples whose target time is at least 82. Results from the first run:

| Intervention, amplitude 0.04 | Mean system absolute response | Mean forecast absolute response | Response mismatch |
| --- | ---: | ---: | ---: |
| Control offset | 0.038001 | 0.037353 | 0.000725 |
| Generator-parameter offset | 0.025836 | 0.025393 | 0.000455 |

These are **illustrative pilot measurements**, not evidence that linear forecasting is generally optimal or that a phase change can be predicted. The model sees only past states, not future control. See [full numerical report](reports/SENSITIVITY_VALIDATION.md) for finite-difference accuracy, seed-level bootstrap intervals, distribution distances and limitations.

## Methodology and project map

| Component | Code | Method |
| --- | --- | --- |
| Controlled ODE and samplers | [controlled_ode_framework.py](code/controlled_ode_framework.py) | Time-varying cubic dynamics, seeded control, RK4, regular/irregular observation |
| Initial ODE experiment | [run_controlled_ode.py](code/run_controlled_ode.py) | Control/parameter/noise interventions and fixed linear forecast |
| Variational sensitivity and 1D OT | [ode_sensitivity.py](code/ode_sensitivity.py) | Coupled tangent equations, centered differences, exact empirical 1D W1/W2 |
| Multi-seed numerical study | [run_sensitivity_validation.py](code/run_sensitivity_validation.py) | Numerical consistency, paired responses, seed bootstrap, fixed-time marginal distributions |
| Unit checks | [test_controlled_ode.py](code/test_controlled_ode.py), [test_ode_sensitivity.py](code/test_ode_sensitivity.py) | Reproducibility, time causality, differential consistency and metric invariants |
| Stationary AR(1) generator | [generate_perturbation_data.py](code/generate_perturbation_data.py) | 5,000 observations, 32-lag one-step prediction |
| Custom attention forecaster | [transformer_model.py](code/transformer_model.py) | PyTorch training from scratch; chronological 70/15/15 split |
| AR(1) versus attention | [linear_baseline_comparison.py](code/linear_baseline_comparison.py) | Matched Gaussian input perturbations |
| Directional response | [directional_perturbation.py](code/directional_perturbation.py) | Position-by-position input sensitivity |
| Correlated Gaussian response | [correlated_gaussian_experiment.py](code/correlated_gaussian_experiment.py) | Input-noise covariance structure at controlled variance |

### Earlier experiments

The original AR(1) and Transformer experiments are kept separate from the controlled-ODE program. They tested **observational input noise**, not changes in the underlying dynamics. Their confirmed historical results, including test RMSE and positional sensitivity, are archived in [the research status](reports/RESEARCH_STATUS.md), [the perturbation methods](reports/PERTURBATION_METHODS.md) and [the correlated-noise report](reports/CORRELATED_GAUSSIAN.md). The older Kalman/ARIMA work is exploratory and not presented as a fully fair benchmark.

To reproduce the older experiment on a machine with PyTorch installed:

```powershell
py -m pip install -r requirements-perturbation.txt
py code/generate_perturbation_data.py
py code/test_perturbation_pipeline.py
py code/transformer_model.py
py code/perturbation_experiment.py
py code/test_linear_baseline.py
py code/linear_baseline_comparison.py
py code/test_directional_perturbation.py
py code/directional_perturbation.py
py code/test_correlated_gaussian.py
py code/correlated_gaussian_experiment.py
```

Training artifacts are generated locally. The plots and archived summaries from previously completed runs remain under `reports/`.

## Research documentation

- [Mathematical derivations](docs/MATHEMATICAL_DERIVATIONS.md): chain rule, variational ODEs, Jacobian stability, finite differences and sensitivity mismatch.
- [Probability, geometry and physics](docs/PROBABILITY_GEOMETRY_AND_PHYSICS.md): random control paths, stochastic versus deterministic dynamics, empirical optimal transport, Fisher information and physical interpretation.
- [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md): falsifiable hypotheses, independent replicates, baselines, leakage, numerical safeguards and failure cases.
- [Reproducibility guide](docs/REPRODUCIBILITY.md): install and run commands, output definitions and troubleshooting.
- [Roadmap and learning plan](docs/ROADMAP_AND_READING.md): feasible staged progression toward neural forecasting, regime detection and geometric statistics.
- [Numerical validation and first results](reports/SENSITIVITY_VALIDATION.md): recorded finite-difference, bootstrap and optimal-transport outcomes.
- [Controlled generator overview](reports/CONTROLLED_ODE_FRAMEWORK.md): experiment narrative, sampling and perturbation concepts.
- [Research status](reports/RESEARCH_STATUS.md): completed versus planned work and historical caveats.

## Scientific limitations and open questions

The current regime change is imposed at a known time; it is **not** an algorithmically detected transition, and its forcing means the observed trajectory should not be called an ideal unforced pitchfork bifurcation. The 12 independent evaluation seeds provide a useful pilot but are too few for broad claims. The bootstrap reflects *control-seed* variability only and excludes retraining uncertainty. Distribution comparisons currently cover a **single scalar marginal at time 100**, not a joint path distribution. The tangent equations, bootstrap and Wasserstein implementation are established tools; scientific novelty remains to be demonstrated by rigorous comparative experiments.

Next steps: step-size convergence; larger control ensembles; explicit-control versus state-only forecasting; GRU/LSTM/Transformer comparisons with matched information; gradual parameter ramps; early-warning false-alarm and lead-time benchmarks; and only then more ambitious statistical-geometric analysis.

## References

- Stoer and Bulirsch, *Introduction to Numerical Analysis*, sensitivity analysis of initial-value problems: https://www.math.kent.edu/~reichel/courses/numerical.analysis.2/Numerical-book.pdf
- Peyré and Cuturi, *Computational Optimal Transport*: https://www.math.ucdavis.edu/~saito/data/acha.read.f19/peyre-cuturi_computational-optxport.pdf
- Kuehn, *A mathematical framework for critical transitions*: https://www.sciencedirect.com/science/article/pii/S0167278911000443
- Scheffer et al., *Early-warning signals for critical transitions*: https://www.nature.com/articles/nature08227

**This repository is an evolving research implementation, not a validated physical forecasting system.**
