# Correlated multivariate Gaussian perturbation experiment

**Research status:** Completed by the researcher in Windows VS Code, with 20 of 20 perturbation repetitions. The three original figures are uploaded. Earlier development-environment tests were implementation checks, not independent model-training replications.

## 1. Motivation

Previous research stages measured how a Transformer and fitted linear AR(1) respond to:

1. Different strengths of **independent Gaussian noise** added to all 32 input values.
2. Noise at **one input position at a time**, revealing which past observations affect a prediction.

The advisor's next question concerns the **distribution of the perturbation**, specifically multivariate Gaussian perturbations with tunable covariance. This experiment changes how noise at different historical positions is **correlated**, while controlling its marginal variance.

A nontrivial result can be that covariance has little effect on a particular model. The purpose is to **measure** the response, not to engineer a difference.

## 2. Precisely defined experiment

For each held-out time t, the forecaster receives a normalized history of L=32 values:

\`X[t] = (x[t-32], ..., x[t-1])\`.

Two frozen forecasting rules are evaluated:

- **Fitted AR(1):** Ordinary-least-squares one-step predictor, fitted using training target indices only.
- **Transformer:** Custom one-step historical-window Transformer, using its existing trained checkpoint, with **no retraining** during this study.

For each covariance condition C:

\`epsilon ~ N(0, sigma^2 C)\`

\`X_perturbed = X + epsilon\`

\`response = F(X_perturbed) - F(X)\`

The initial default is \`sigma = 0.10\` **in training-normalized units**, with **20 perturbation draws per covariance profile**.

### Controlled covariance families

| Code name | Definition | Meaning |
| --- | --- | --- |
| independent | C = I | Independent noise at every historical position |
| positive_temporal | C[i,j] = 0.8^abs(i-j) | Nearby perturbations tend to share a sign; correlation decays with separation |
| negative_temporal | C[i,j] = (-0.8)^abs(i-j) | Neighboring perturbations tend to have opposite signs; the sign alternates with lag |
| common_mode | C[i,i]=1, C[i,j]=0.5 for i != j | Every pair of positions has the same positive correlation |

These choices are **mathematically valid positive-definite correlation matrices**, verified by Cholesky factorization and tests. They are fixed design choices, not estimates fitted using test data.

### Why equal marginal variance is important

Every C has unit diagonal, so each perturbed input coordinate has variance **sigma²**, independent of profile. The total expected squared noise norm is:

\`E(||epsilon||²) = trace(sigma² C) = L * sigma²\`.

At L=32 and sigma=0.10, this equals **0.32** in normalized squared units for **all four** conditions.

Thus the experiment varies **correlation structure**, not the expected noise energy. Correlation patterns can still produce different individual noise vector norms and differently aligned directions; the equality is in expectation.

### Gaussian sampling: Cholesky transform

The code draws an independent standard Gaussian vector z and computes:

\`epsilon = sigma * chol(C) * z\`

where \`chol(C) chol(C)^T = C\`.

With 750 held-out test windows, this draws 750 perturbed histories per repetition and covariance profile. For each repetition, the **same base standard-normal draws z** are used for all four correlation profiles, and both forecasters see the **same perturbed windows**. This makes covariance comparisons paired, reducing unpaired Monte Carlo variation.

The original forecast targets **do not change**. The generator itself also remains the original stationary Gaussian AR(1); this is an **input perturbation experiment**, not a generator distribution-shift experiment.

## 3. Response measurements

The experiment computes, for each model/profile/repetition:

- **Mean absolute response:** mean |F(X+epsilon)-F(X)| across test windows, in original signal units.
- **Mean signed response:** average directional forecast change; can cancel when noise has opposite signs.
- **RMSE against the unchanged targets:** prediction accuracy, in original units.
- **NMSE:** perturbed forecast MSE divided by the fixed **training-set** variance.
- **Empirical squared noise norm:** check against the common expected total noise energy.

Results are summarized across the 20 draws per condition. Each per-draw record is retained locally.

### Paired comparison to independent noise

For each draw, calculate:

\`D_response(profile) = response(profile) - response(independent)\`

\`D_RMSE(profile) = RMSE(profile) - RMSE(independent)\`

These are **paired differences of summary measures for the same base z**, evaluated separately for each forecasting model. They are not differences between the Transformer forecast and the AR(1) forecast.

The script also computes a standard error across 20 Monte Carlo draws for these paired differences. This describes **noise-draw variability for one fixed dataset and model**, not uncertainty across multiple trained Transformers or synthetic datasets.

## 4. A useful analytical prediction for AR(1)

The fitted linear model predicts from only the newest historical value, \`F_linear(X) = a + b * X[last]\`. It ignores the first 31 positions.

Because all four covariance profiles have the **same variance at the last coordinate**, linear AR(1) has identical *expected* mean absolute forecast response across profiles:

\`E|Delta_linear| = |b| * sigma * training_std * sqrt(2/pi)\`.

Small empirical differences between profile means can occur because the finite random draws are different after Cholesky transformation. They are **not** a systematic covariance effect on the AR(1) forecaster in expectation.

The Transformer can combine historical positions, so correlation may change how input perturbations interact with its sensitivity pattern. Any observed change should be interpreted alongside paired noise-draw variation and baseline accuracy.

## 5. Development-environment validation (not the researcher's Windows run)

The covariance test passed with positive-definite matrices, empirical sampled covariances, matched diagonal variance and the AR(1) invariance check.

The full default experiment also completed in a separate local validation on the retained starter checkpoint with sigma=0.10 and 20 repeats. The approximate mean absolute forecast response was:

| Perturbation covariance | Fitted AR(1) | Transformer |
| --- | ---: | ---: |
| Independent | 0.08205 | 0.07012 |
| Positive temporal | 0.08187 | 0.06993 |
| Negative temporal | 0.08333 | 0.07040 |
| Common-mode | 0.08193 | 0.06977 |

**Observation from this local check:** differences among covariance structures were small for both forecasters, and the paired differences for the Transformer were typically a few ten-thousandths in the average response. The simple linear Gaussian generator may help explain why the Transformer has a strong newest-lag dependence, but this is not a causal or cross-seed conclusion.

These values are intentionally **not labeled as the researcher's confirmed Windows results**. Run the script using your current checkpoint to generate the actual experiment tables.

## 6. Run it in VS Code

Pull your latest code from GitHub, then run:

\`\`\`powershell
git pull origin main
py code/test_correlated_gaussian.py
py code/correlated_gaussian_experiment.py
\`\`\`

**Do not rerun the generator or retrain the Transformer** if the previous \`perturbation_ar1.csv\`, \`transformer.pt\`, and \`config.json\` already exist. The script uses them and checks for compatible split/normalization through the existing model loader.

To change the noise magnitude or number of draws, for example:

\`\`\`powershell
py code/correlated_gaussian_experiment.py --sigma 0.05 --repeats 30
\`\`\`

### Generated files

All appear under \`reports/transformer_experiment/\` on your computer (ignored by Git):

| File | What it contains |
| --- | --- |
| correlated_gaussian_matrices.png | Heatmaps of the four 32×32 correlation patterns |
| correlated_gaussian_response.png | Absolute response and RMSE for each model/profile |
| correlated_gaussian_paired_effects.png | Small paired differences from independent noise, with Monte Carlo standard errors |
| correlated_gaussian_summary.csv | Mean response, forecast error and noise energy per model/profile |
| correlated_gaussian_per_repeat.csv | All repetition-level measurements |
| correlated_gaussian_paired_deltas.csv | Paired profile-vs-independent differences for each draw |
| correlated_gaussian_paired_effects.csv | Mean paired effects and their draw-to-draw variation |
| correlated_gaussian_covariances.csv | All entries of all four designed C matrices |
| correlated_gaussian_config.json | Experimental parameters, seeds, clean RMSE, learned AR(1) slope |

Original figures: [covariance matrices](figures/correlated_gaussian_matrices.png), [response and RMSE](figures/correlated_gaussian_response.png), [paired effects](figures/correlated_gaussian_paired_effects.png). Exact raw measurements remain in local CSVs.

## 7. Limitations and next research direction

- Fixed model, one dataset, one perturbation sigma by default: results are **exploratory**.
- Same marginal covariance does **not** guarantee fidelity to the original autoregressive joint input distribution.
- Input history windows overlap and perturbation draws are not independent *training* replications.
- Changing covariance is not the same as moving along an information-geometric geodesic; **no Fisher metric or Riemannian distance is computed here**.
- Future: repeat experiments over model-training seeds; vary sigma with the same covariance families; test covariance directions derived from the original data distribution; consider norm-matched eigenvector perturbations and only then formal statistical-manifold questions.

[Experiment source](../code/correlated_gaussian_experiment.py) · [Deterministic test](../code/test_correlated_gaussian.py) · [Code walkthrough](../code/CORRELATED_GAUSSIAN_WALKTHROUGH.md)


## Confirmed Windows results (sigma=0.10; 20 repetitions)

The fitted AR(1) theoretical expected mean absolute response is **0.082499** under all four covariance profiles. The following values were read from the researcher's Windows terminal output:

| Correlation profile | Fitted AR(1) mean abs response | Transformer mean abs response |
| --- | ---: | ---: |
| Independent | 0.082047 | 0.070116 |
| Positive temporal | 0.081871 | 0.069930 |
| Negative temporal | 0.083333 | 0.070397 |
| Common-mode | 0.081933 | 0.069768 |

Changing covariance with fixed marginal noise variance produced relatively small response changes for both models on this stationary Gaussian AR(1) dataset. The linear model's response stayed close to mathematical theory. The Transformer was less responsive, but this alone does **not** mean it was more accurate. Error bars in the paired effects graph reflect the variability between repeated perturbation draws for a *single trained model*; they do not measure uncertainty across independently trained models.

The original uploaded covariance heatmap has an overlapping colorbar. The updated plotting script fixes the layout for future regenerated figures; the original upload has been preserved rather than silently replaced.
