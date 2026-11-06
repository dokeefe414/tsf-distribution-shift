# Research roadmap and mathematics learning sequence

The project is deliberately staged. Each new algorithm must answer a precise question, have justified assumptions and be compared with a simpler baseline. “Interesting graph” is not a research question. **The central falsifiable claim is whether response-fidelity measurements reveal behavior not captured by standard forecast error.**

## Completed foundation (current)

- Synthetic distribution shifts; ARIMA and Kalman exploratory baselines.
- Stationary AR(1), a custom attention-based one-step model and Gaussian input-noise response studies.
- Nonlinear time-varying controlled ODE, regular/irregular sampling and frozen linear one-step forecasting.
- Pathwise variational sensitivities, finite-difference validation, multi-seed paired-response estimates and elementary one-dimensional optimal transport.

## Next: numerical and experimental credibility

**Milestone A — Derivative and solver verification.** Vary RK4 dt; compare trajectories against a substantially finer aligned reference; check convergence away from event boundaries. Vary h for both tangent finite-difference studies. Run multiple onset times and positive/negative interventions. Acceptance: numerical error reported separately from scientific model error.

**Milestone B — Statistical design.** Increase independent control seeds and perform repeated fitting on multiple independent training trajectories, not just multiple test trajectories. Compare held-out control sequences, empirical uncertainty and bootstrap stability. Acceptance: tables and confidence intervals with an explicit independent unit.

**Milestone C — Fair forecasting baselines.** Add persistence and fitted autoregressions, then train GRU, LSTM and an attention model on identical splits, lookback windows, targets and physical units. Compare **state-only** input against **state + known control** input. Acceptance: matched RMSE, response mismatch, uncertainty and runtime.

## Later: informative change detection

A sudden predetermined coefficient switch is **not** an early-warning benchmark. Generate a slow, controlled parameter ramp near a bifurcation; define the bifurcation/time-of-instability precisely and determine whether the noise/control schedule makes the early-warning question identifiable. Set thresholds on validation trajectories, evaluate false alarms on no-change controls, and measure *lead time* on unseen trajectories. Baselines should include moving variance, lag-1 autocorrelation, forecast error and a simple change-point detector.

Only after reproducible detection results should online adaptation be introduced. Split detection from adaptation; record how much data the model receives before the next prediction.

## Mathematics program: concepts to derive before code

| Order | Study | Prove or compute on the project |
| --- | --- | --- |
| 1 | Linear algebra and calculus | Gradient, chain rule, Jacobian, eigenvalues; derivative of \(ax-x^3\) |
| 2 | Probability and statistics | Conditional expectation, covariance, independent controls, paired variance, bootstrap units |
| 3 | ODE and numerical analysis | RK4, equilibrium stability, variational equations, numerical consistency |
| 4 | Dynamical systems | Bifurcation normal forms, forcing, local Lyapunov behavior, observable vs hidden control |
| 5 | Geometry | Tangent vectors, coordinate changes, metrics; identify the relevant manifold before curvature |
| 6 | Optimal transport | Derive one-dimensional quantile matching and empirical W1/W2; understand coupling versus marginal comparison |
| 7 | Information geometry | Define \(p_\theta\), score, Fisher information and regularity; compare to Euclidean and OT metrics |
| 8 | Statistical physics | Fluctuation/response, ensembles, critical slowing down and where noisy forcing invalidates simple indicators |

## Sources and limits of originality

- Stoer & Bulirsch, *Introduction to Numerical Analysis*, §7.2.19: https://www.math.kent.edu/~reichel/courses/numerical.analysis.2/Numerical-book.pdf
- Peyré & Cuturi, *Computational Optimal Transport*: https://www.math.ucdavis.edu/~saito/data/acha.read.f19/peyre-cuturi_computational-optxport.pdf
- Kuehn, *A mathematical framework for critical transitions*: https://www.sciencedirect.com/science/article/pii/S0167278911000443
- Scheffer et al., *Early-warning signals for critical transitions*: https://www.nature.com/articles/nature08227

Variational equations, bifurcation theory, elementary OT, Fisher information and bootstrap CIs are *established*. A meaningful contribution could come from a new theoretically justified relationship between **physical response, learned response and predictive error** under a well-defined observation regime. Such a claim requires a dedicated literature review and appropriately strong counterexamples and baselines.
