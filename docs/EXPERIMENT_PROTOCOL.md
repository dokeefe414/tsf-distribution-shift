# Experimental protocol and falsifiable research questions

## Scientific objective

Determine whether a forecasting model's response to a *controlled generator intervention* accurately tracks the actual dynamical response, and under what conditions its response mismatch is informative beyond ordinary forecast error.

This is a **research hypothesis**, not an established novel finding. Ordinary variational sensitivity, input perturbation, Wasserstein distance and early-warning indicators have substantial prior literature. Any publication-level novelty claim needs a focused literature comparison.

## Hypotheses and potential falsifications

| Question | Measurement | What could falsify a stronger claim? |
|---|---|---|
| Does the ODE derivative match known numerical behavior? | RK4 tangent sensitivities vs centered finite differences as h decreases | Failure to converge or reproducible pre-onset response |
| Can a frozen forecast track physical response? | Per-time \( |\Delta_F-\Delta_G| \), averaged by independent seed | A naïve last-value predictor performs equally well or better |
| Does response fidelity add information beyond RMSE? | Matched model/seed comparisons of RMSE and response mismatch | Response mismatch gives no independent predictive/detection value |
| Do perturbation responses warn of dynamical transitions? | Pre-transition detector metric, false positive rate and detection lead time | No better performance than simple error/variance/autocorrelation baselines |
| Do different interventions have distinct physical effects? | Control vs parameter derivative and trajectory response | No detectable distinction after unit-matched comparisons |

Do not compare an unobservable future intervention against an allegedly omniscient predictor. Explicitly define whether future controls are known, forecast, or unavailable.

## Experimental units and leakage

- **Training:** a single baseline control realization with seed 2026, using only pre-switch one-step targets at \(t<58\). This is an initial pilot, not a representative training ensemble.
- **Evaluation:** seeds 3000–3011 by default; independently generated control sequences, paired baseline/perturbed trajectories for *each* seed. The model coefficients remain frozen.
- **Perturbation:** a control offset or parameter offset is introduced at \(t=80\); the default intervention amplitude is 0.04 in its own units.
- **Response evaluation:** regular observations with target times \(t\ge82\); horizon is **one step** and window length is 12 samples.
- **Generator derivative validation:** centered differences for \(h\in\{0.08,0.04,0.02,0.01\}\) compared with coupled RK4 variational integration.
- **Distribution evaluation:** one-time marginal at \(t=100\), using equal-sized empirical samples across independent seeds.
- **Uncertainty:** 500 percentile bootstrap draws (by default) over *per-seed* statistics, not time-step observations. This does not include retraining variability.

The same generator's control seed is reused within each baseline/perturbed pair so the treatment response is not dominated by unrelated random draws.

## Metrics and reporting

1. Clean one-step forecast RMSE against the clean target.
2. Perturbed one-step forecast RMSE against the **perturbed** physical target.
3. Mean absolute generator response \(E|\Delta_G|\).
4. Mean absolute forecast response \(E|\Delta_F|\).
5. Mean absolute response mismatch \(E|\Delta_F-\Delta_G|\).
6. Per-seed bootstrap confidence intervals for the ensemble mean.
7. Empirical 1D Wasserstein-1/2 for fixed-time state marginals (do not confuse with paired response).
8. Numerical derivative MAE and maximum absolute error, reported per step size.

Report **magnitude and mismatch together**: a small forecast response is not automatically desirable. A constant predictor is very insensitive but could be uninformative.

## Required controls before scientific claims

- Run independent training seeds and separate model-fitting uncertainty from evaluation-seed variation.
- Add a persistence/last-observation baseline, ordinary AR models, and matched-capacity GRU/LSTM/Transformer models, on identical horizons and splits.
- Compare mean response mismatch **and** response shape/correlation, including cases of low generator sensitivity and near transition.
- Report matched units and normalized interventions; numerically equal control/parameter offsets are not equivalent physical changes.
- Validate the continuous-time solution with RK4 step-size refinement.
- Distinguish abrupt regime changes, gradual bifurcation approaches, observation noise, structural parameter shifts and control interventions.
- For phase-change detection, define advance detection and false-alarm criteria before inspecting results and test against no-change controls.
- Evaluate both forecasters that observe the control signal and forecasters that do not; avoid giving future information to one model alone.
- Record code version, seed selection, complete configuration, plots and small numerical summaries.
- Repeat the experiment at multiple intervention amplitudes and onset times, including negative offsets.

## Failure modes worth testing

**High forecasting accuracy but poor response tracking:** a model predicts baseline values well yet systematically underestimates treatment-induced change.

**Low forecast response but poor accuracy:** a constant or heavily regularized model looks robust under an input-noise metric while failing to follow the real system.

**Transient mismatch:** a control change happens too quickly for a model without future-control information to react at the first post-intervention step.

**Spurious early warning:** an anomaly score fires only *after* an abrupt forced switch, then is incorrectly presented as early detection.

**Finite-difference artifact:** a large intervention produces substantial nonlinearity so finite differences no longer approximate the derivative at zero.

## Current scope versus future work

The implemented benchmark validates **one scalar, controlled cubic ODE, one linear frozen forecaster and a small ensemble**. GRU/LSTM/Transformer comparisons and statistical-manifold inference are milestones, not completed evidence. This scoped baseline is deliberately small enough to audit mathematically before introducing harder methods.
