# Controlled nonlinear ODE framework

**Version:** November 2026. **Status:** nonlinear generator, sampling, paired interventions, a frozen linear forecast benchmark, variational sensitivity validation and first multi-seed probabilistic comparisons are implemented.

This page gives the experimental story in one place. Detailed mathematical proofs and caveats are in [MATHEMATICAL_DERIVATIONS.md](../docs/MATHEMATICAL_DERIVATIONS.md), the probabilistic interpretation is in [PROBABILITY_GEOMETRY_AND_PHYSICS.md](../docs/PROBABILITY_GEOMETRY_AND_PHYSICS.md), and the actual numerical record is in [SENSITIVITY_VALIDATION.md](SENSITIVITY_VALIDATION.md).

## Conceptual pipeline

```text
control u(t), parameter a(t), initial x0
                   |
   g(x,u,t)  -- RK4 numerical integration --> state trajectory x(t)
                   |                                |
        intervene in u or a                  sampler (regular / irregular)
                                                    |
                                      observed time series and timestamps
                                                    |
                                    train on pre-change windows (one-step)
                                                    |
                                        freeze predictor F
                                                    |
                      paired baseline and perturbed input windows
                             |                      |
                       true response         forecast response
                             +--------- compare -----+
                                      |
                                response mismatch
```

## Generator and random sequence

We use \(\dot x=a(t)x-x^3+u(t)\). The time-varying parameter switches from -0.8 to +0.6 at time 60. The control adds a weak sinusoid to normally distributed random levels held constant for approximately one unit of time. The control is a *sampled random signal*, not a white-noise SDE. Each random seed determines an entire control sequence and hence an entire trajectory. Given that sequence, the ODE is deterministic.

For the baseline, \(\mathrm{d}x/\mathrm{d}t\) depends on the state, parameter and contemporaneous control. Changing a parameter changes the ODE vector field; changing the control modifies the forcing; adding measurement noise to sampled observations leaves the generator completely unchanged.

A classical RK4 solver integrates on a 0.05 grid; the regular sampler observes each 0.25 time units, while the irregular sampler varies the gaps but returns timestamps. Only regularly spaced samples are used by the initial linear one-step predictor. Ignoring irregular gaps in a time-series model would be a modeling assumption requiring separate evaluation.

## Paired interventions and causality

The default control-offset and parameter-offset interventions begin at time 80. Baseline and perturbed simulations reuse the **same random realization and initial state**, so their difference estimates a controlled pathwise effect. Observation noise instead alters already-sampled values.

The 2026-10 preliminary baseline experiment used intervention amounts 0.08 (control and observation noise standard deviation) and 0.12 (parameter). Those historical numbers are preserved in [CONTROLLED_ODE_FIRST_RUN.csv](CONTROLLED_ODE_FIRST_RUN.csv). They should not be compared directly to the new **0.04 amplitude** sensitivity-validation experiment, which uses different settings.

## One-step forecasting experiment

The baseline one-step prediction is a linear ridge model \(F(S)=w_0+w^TS\) on 12 historical observations. It is fitted on a single control realization from **before** the built-in coefficient change. The baseline is a sanity check for response measurement, not evidence that a linear model is state-of-the-art or a surrogate for a neural-network comparison.

Because the model sees historical states only and does **not** know future control values, immediate unpredictable control changes are not generally forecastable. A later controlled experiment should compare this state-only predictor against forecasters with explicitly available control information.

## What was added mathematically

For two separately controlled intervention amplitudes, the tangent states \(s_u\) and \(s_a\) satisfy

\[
\dot s_u=(a-3x^2)s_u+\mathbf 1_{t\ge80},\qquad
\dot s_a=(a-3x^2)s_a+x\,\mathbf 1_{t\ge80}.
\]

They predict infinitesimal changes in the trajectory, *not* general large shocks. These equations are integrated with RK4 and compared to centered differences using the same control realization. The numerical consistency experiment demonstrates decreasing derivative error at smaller amplitudes. It does not yet test continuous-time convergence in the integration step.

At a later stage, sensitivity matching can be compared across forecaster families to determine whether one-step RMSE alone hides important failures to track physical response. That is a research question rather than a result.

## Probabilistic extension

The experiment holds a forecaster fixed and simulates multiple independent random-control realizations, using paired control paths within each intervention. It reports the distribution of seed-level response statistics with simple percentile bootstrap intervals. It also compares two empirical state marginals at t=100 using exact one-dimensional Wasserstein-1 and Wasserstein-2.

These are elementary and interpretable distribution metrics. They do not imply that a Riemannian or Fisher-information geometry has already been constructed.

## Current constraints and next work

There is no GRU/LSTM training on this generator yet, no online parameter update, no statistical significance established over independent model fits, no gradual critical-transition early-warning experiment and no theorem relating forecast mismatch to physical stability. The switch at time 60 is known by design; the study has not yet discovered a change point.

The next high-value steps are solver timestep convergence, more independent training and evaluation seeds, explicit-control baselines, and matched models before early-warning claims. See [experimental protocol](../docs/EXPERIMENT_PROTOCOL.md) and [roadmap](../docs/ROADMAP_AND_READING.md).
