# Controlled ODE generator: meeting follow-up (October 2026)

**Status:** First end-to-end research scaffold implemented and smoke-tested. **Not** a completed GRU/LSTM/Transformer comparison, online-learning system, or phase-change detector.

## 1. What Fanchao proposed

The long-term research pipeline is:

    ODE generator g(x, u, t) -> trajectories -> sampler -> time series -> one-step forecaster F
                    |                                        |
             perturb g or u                        perturb observed inputs
                    +----------------------------------------+
                          compare changes in the system
                          with changes in forecasts

The generator evolves according to dx/dt = g(x, u, t), where the control u(t) can be random and the dynamics can change with time. Numerical integration creates trajectories, while the sampler turns trajectories into observations. This is different from adding noise to an already-existing input window.

## 2. Implemented mathematical example

We intentionally start with a **nonlinear one-dimensional ODE**:

    dx/dt = a(t) x - x^3 + u(t)

with time-varying coefficient:

    a(t) = -0.8 for t < 60;  +0.6 for t >= 60

The forcing u(t) combines a weak sine wave with seeded Gaussian random values held approximately one second at a time. This is a **random control sequence**, not an independent Gaussian diffusion term inside a stochastic differential equation. The same control realization is reused in paired simulations.

At u=0 the unforced system changes equilibrium structure as a crosses zero (pitchfork-bifurcation example). The nonzero forcing breaks exact symmetry, so this demo should be called a **known regime change**, not proof of automatic phase-transition detection.

The integrator uses fourth-order Runge-Kutta with dt=0.05 from t=0 to t=120, starting at x(0)=0.1. Random control generation is seeded for reproducibility.

## 3. Three different interventions

All interventions begin at t=80; the baseline and intervened runs share the **same initial condition and random control draw**.

| Intervention | Precisely what changes? | Generator trajectory changes? |
| --- | --- | --- |
| Control signal | Add 0.08 to u(t) after t=80 | Yes |
| Generator parameter | Add 0.12 to a(t) after t=80 | Yes |
| Observation noise | Add iid N(0, 0.08^2) to the sampled values | **No**; only observed inputs change |

We compare the new and baseline trajectories sampled on the same timestamps. The first two are changes in g / its control, whereas the last reproduces the kind of observational perturbation used by earlier AR(1) experiments.

## 4. Sampler and initial predictor

- A regular sampler reads the ODE state every 0.25 time units, producing 481 values.
- A seeded irregular sampler also exists to study sampling intervals; it keeps timestamps. The baseline predictor below deliberately uses **regular** samples only.
- A simple **ridge-regularized autoregression** uses 12 historical samples to predict the next sample. Training uses only targets with t < 58, before the known change at t=60.
- We keep its fitted weights frozen and compare paired predictions for the clean and perturbed time series at t >= 82.
- The predictor is a minimal, interpretable placeholder. **GRU, LSTM, Transformer re-training, and online learning are future work.**

For an intervention, define the system response at one-step target time t:

    Delta_G(t) = x_perturbed(t) - x_baseline(t)

and the frozen forecaster response:

    Delta_F(t) = F(perturbed input window) - F(baseline input window)

We report mean |Delta_G|, mean |Delta_F| and mean |Delta_F - Delta_G| on a matched set of times. The last number measures **one-step response tracking**, not an intrinsic, generator-independent sensitivity of F. To compare model accuracy, we use the corresponding perturbed target for interventions on g, and the clean target for observation-only noise.

For **observation noise**, the physical generator is unchanged, so Delta_G=0 by definition. This distinction helps disentangle generator response from observation-driven model response.

## 5. Run and tests

From the repository root in VS Code / PowerShell:

```powershell
py -m pip install -r requirements-perturbation.txt
py -m unittest discover -s code -p "test_controlled_ode.py" -v
py code/run_controlled_ode.py
```

Or use `python` in place of `py`. The demo itself only requires numpy and matplotlib, already in the repository's dependency file.

Regenerable local outputs are written under `reports/ode_experiment/`:

- `configuration.json` captures the dynamics, perturbation levels and evaluation settings.
- `response_summary.csv` contains the three paired-response diagnostics.
- `ode_trajectories.png` shows the system trajectories, known change time and irregular sampling example.

**Reproducible smoke-test results** from the initial execution (one control seed, one fitted linear model):

| Intervention | Mean absolute generator response | Mean absolute forecast response | Mean absolute response mismatch |
| --- | ---: | ---: | ---: |
| Control offset | 0.08291 | 0.08148 | 0.00168 |
| Parameter offset | 0.07377 | 0.07249 | 0.00131 |
| Observation noise | 0.00000 | 0.11697 | 0.11697 |

These are early diagnostics from this synthetic example. They do **not** establish causal attribution for arbitrary real-world systems, statistical significance, or a model ranking. Repeating multiple seeds and varying change timing and strengths are necessary before drawing general conclusions.

## 6. Next experiments (planned, not completed)

1. Compare a GRU, LSTM and the existing Transformer on the **same sampled trajectories** using matched chronological splits, window lengths, horizon and target scaling.
2. Train on multiple random control realizations, and evaluate unseen control sequences plus regime changes without leakage.
3. Sweep control and generator-parameter perturbation magnitudes; estimate distributions and confidence intervals for Delta_G and Delta_F using paired replicates.
4. Define a measurable phase-change detection criterion, then compare response-based warning signals against standard change-point tests and forecast error.
5. Investigate online updating after the regime change, separating detection, model adaptation and evaluation.
6. Explore irregular sampling with time-aware inputs rather than treating varying gaps as equal steps.

**Important scientific distinction:** High raw forecast response may simply reflect large generator response. We must compare both, not label the largest response as the weakest model.
