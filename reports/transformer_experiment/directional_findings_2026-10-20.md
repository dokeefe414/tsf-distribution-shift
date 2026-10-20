# Directional perturbation results — first Windows run

**Date:** October 20, 2026  
**Evidence:** Researcher-provided screenshot of `directional_sensitivity.png` produced by the project’s directional perturbation script in VS Code.

## Status and provenance

The researcher has now run the directional sensitivity experiment and supplied a screenshot of its generated heatmap and line plot. This verifies that the graph was produced on the Windows machine. **The full-resolution PNG was subsequently uploaded to GitHub** at [reports/figures/directional_sensitivity.png](../figures/directional_sensitivity.png). The exact numeric CSVs remain on that computer. Therefore the values below are qualitative observations from the screenshot, not transcriptions of exact 32-position measurements.

For the exact figure, see [the original uploaded PNG](../figures/directional_sensitivity.png). For the exact per-position numerical values, the local CSVs can still be archived with [the optional publishing helper](../../code/publish_directional_results.py); preserve the current outputs before rerunning.

## Experimental setup

- Generator: stationary Gaussian AR(1), approximately `x[t] = 0.7 x[t-1] + eta[t]`, with Gaussian independent innovations.
- Models: a one-step, fitted linear AR(1) and a custom trained Transformer.
- Inputs: 32 historical observations, position 1 oldest and position 32 newest.
- Experiment: add Gaussian noise to exactly one input position per test input, leaving the other 31 positions unchanged.
- The screenshot title shows normalized perturbation standard deviation **sigma = 0.1**.
- The script defaults to 12 draws per position and a fixed seed, but the screenshot alone does not independently establish the command-line arguments.
- Response: mean absolute change in the model’s forecast, expressed in original signal units.

## What the screenshot shows

| Finding | Observation | Interpretation |
| --- | --- | --- |
| AR(1) positions 1–31 | Response is at or visually indistinguishable from zero | Matches the known one-step AR(1) dependence on only the newest observation |
| AR(1) position 32 | Dominant peak, visually about 0.08 | Expected because the model uses the most recent observation |
| Transformer positions 1–31 | Small nonzero responses at multiple positions, with some local peaks | Indicates input sensitivity to historical observations beyond the newest lag |
| Transformer position 32 | Largest response for that model, visually about 0.07 | Suggests strongest local sensitivity to the newest observation in this model/run |
| Comparison | AR(1) newest-lag response appears larger | Less sensitivity does **not** by itself imply better forecasting accuracy |

The earlier confirmed **clean forecast RMSE** measurements were 1.004406 for fitted AR(1) and 1.022539 for the Transformer. They came from the previous clean/paired-noise experiment; they are not directional RMSE measurements read from this screenshot.

## Key explanation for the advisor

For fitted AR(1), the one-step forecast is `f(X) = a + b X[32]`. Under one-coordinate noise, changing positions 1–31 cannot affect the prediction, whereas perturbing position 32 does. In the Transformer, self-attention and the final forecasting head permit influence from any input position. The screenshot suggests the model mainly relies on the newest observation, but also reacts to older inputs.

**Do not equate a small nonzero perturbation response with useful predictive information.** The learned position-specific responses could reflect redundant features, small incidental dependencies, or characteristics of a single training run.

## Limitations and next experiment

This is **one model checkpoint on one Gaussian AR(1) dataset**. There is no repeated-training uncertainty estimate. The perturbation acts on a single coordinate and need not preserve the original joint temporal distribution. It is not yet a correlated multivariate Gaussian, directional-score, Fisher-information, or statistical-manifold experiment.

Next: archive the exact PNG and two directional CSV files, then try **norm-matched multi-coordinate perturbations** and **temporally correlated Gaussian noise**, before moving to longer-horizon behavior.

[Experiment method and implementation](../DIRECTIONAL_PERTURBATIONS.md) · [Overall research progress](../RESEARCH_STATUS.md)
