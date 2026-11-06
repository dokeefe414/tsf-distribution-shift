# Mathematical sensitivity validation — pilot record (November 6, 2026)

## Questions tested

1. Do directly integrated variational equations agree with independent centered finite-difference estimates of the generator's local response?
2. How well does a frozen linear one-step predictor follow generator responses across independent random-control trajectories?
3. How far do empirical scalar state distributions move at one fixed time under the same physical interventions?

**Status:** Software/numerical pilot run locally with NumPy and Matplotlib; 8 new sensitivity tests passed. The results below are specific to this parameterization, time step, intervention onset, control realization and 12 evaluation seeds. They are *not* evidence of a new mathematical theorem or universal neural-network behavior.

## Controlled ODE and comparisons

\[
\dot{x}=a(t)x-x^3+u(t), \quad
a(t)=-0.8 \text{ for } t<60,\; a(t)=0.6 \text{ thereafter}.
\]

Gaussian-held random control is seeded; interventions begin at t=80. Classical RK4 uses dt=0.05. The linear one-step forecaster uses 12 regular historical observations, fit on a training trajectory with seed 2026 using targets at t<58 and frozen for all evaluations (t>=82). The independent evaluation seeds are 3000 through 3011. Per-seed, each intervention and baseline share an identical control realization.

## 1. Finite-difference vs variational derivative

The theoretical equations are

\[
\dot{s}_u=(a-3x^2)s_u+\mathbf 1_{t\ge80},\quad
\dot{s}_a=(a-3x^2)s_a+x\mathbf 1_{t\ge80}.
\]

The centered difference is \(D_h=[x_{+h}-x_{-h}]/(2h)\). Values below are the mean absolute difference between the computed \(D_h\) trajectory and variational \(s\) trajectory across the full integration grid, using training seed 2026.

| Finite-difference h | Control derivative MAE | Parameter derivative MAE |
| ---: | ---: | ---: |
| 0.08 | 0.01527111 | 0.00076707 |
| 0.04 | 0.00315028 | 0.00018819 |
| 0.02 | 0.00075795 | 0.00004683 |
| 0.01 | 0.00018776 | 0.00001169 |

The error decreases as h shrinks, consistent with the \(O(h^2)\) centered-difference truncation behavior (until roundoff dominates). This validates the implementation against a separate numerical approximation **for the discretized generator**. It does *not* bound error against the exact continuous ODE solution; dt-refinement is still necessary.

[Archived CSV](SENSITIVITY_DERIVATIVE_ERRORS.csv). The runnable script generates `tangent_vs_finite_difference.png` locally.

## 2. Paired generator and model responses

Each entry is the mean over **12 independent seed-level evaluation statistics**, with intervention amplitude +0.04. “Response mismatch” is \(E|\Delta_F-\Delta_G|\) on a per-seed test interval, then averaged across seeds.

| Intervention | Generator mean absolute response | Forecast mean absolute response | Mean response mismatch | Mismatch bootstrap 95% percentile interval |
| --- | ---: | ---: | ---: | --- |
| Control offset | 0.03800067 | 0.03735287 | 0.00072526 | [0.00066476, 0.00079551] |
| Parameter offset | 0.02583602 | 0.02539295 | 0.00045510 | [0.00043877, 0.00047547] |

Bootstrap uses 500 draws of the **independent-seed units**, seed 11. The interval measures variability across these 12 fixed control sequences and does not include refitting the forecasting model or uncertainty about the physical ODE.

The response mismatch is small for this highly smooth scalar system and simple one-step predictor; there is no basis yet to call the approach universally strong or novel.

[Archived seed-level bootstrap CSV](SENSITIVITY_BOOTSTRAP_SUMMARY.csv). A separate local per-seed CSV records each paired result.

## 3. Empirical distribution comparisons at t=100

Across 12 independent control seeds, each baseline state and intervened state form equally weighted one-dimensional empirical measures. Exact quantile matching for these empirical measures gives:

| Intervention | Empirical Wasserstein-1 | Empirical Wasserstein-2 | Paired mean absolute state change |
| --- | ---: | ---: | ---: |
| Control offset | 0.04808586 | 0.04860246 | 0.04808586 |
| Parameter offset | 0.03020518 | 0.03033588 | 0.03020518 |

These are distances **between distributions at one point in time**, not distances between whole time-series distributions and not predictive-performance metrics. That the empirical W1 matches paired mean absolute change in this pilot reflects sample ordering; it is not a general identity. Twelve samples are not a robust estimate of the underlying population measure.

[Archived fixed-time OT CSV](SENSITIVITY_EMPIRICAL_OT.csv).

## 4. What the tests establish — and what they do not

- **Established:** reproducibility for fixed seeds; generator and tangent consistency; no pre-intervention sensitivity; basic quantile-transport metric properties; seed-level bootstrap determinism; executable finite-difference and ensemble experiments.
- **Not established:** GRU/LSTM/Transformer performance on the controlled-ODE data; a formal change-point detector; guaranteed generalization; information-geometric or Riemannian metrics; physical real-world validity; independent repeated training uncertainty.
- **Known caveat:** stepwise coefficient change at t=60 is part of the generator. It is not predicted by the model. Unseen future controls cannot be assumed available to a state-only forecaster.

## Reproduction

```powershell
py -m pip install -r requirements-ode.txt
py -m unittest discover -s code -p "test_ode_sensitivity.py" -v
py code/run_sensitivity_validation.py --seeds 12 --bootstrap 500 --epsilon 0.04
```

The experiment saves a configuration JSON, full per-seed CSVs, a tangent/finite-difference figure and diagnostic summaries under the ignored directory `reports/sensitivity_experiment/`. See [reproducibility](../docs/REPRODUCIBILITY.md) and [mathematical derivations](../docs/MATHEMATICAL_DERIVATIONS.md).
