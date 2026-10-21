# Perturbation-response research: methodology and evaluation

## Goal and relationship to earlier work

Earlier experiments changed the time-series-generating process across regimes and evaluated the resulting forecasting error. The new experiment instead holds a generator and a **trained forecaster** fixed, then perturbs its **input window**. The goal is to measure the resulting output sensitivity and accuracy change separately.

The simpler baseline matters: simultaneously varying trend, variance, seasonality, model dynamics and outliers would make it hard to identify what produced a response.

## Generator G

First experiment: a stationary autoregressive Gaussian process.

`x_t = 0.7 x_(t-1) + eta_t`, with `eta_t ~ N(0,1)` independent across t.

A fixed seed generates 5,200 observations; the first 200 are discarded as burn-in. Remaining samples are cast to float32. There are no designed distribution shifts in this first experiment.

## Forecaster F

The trained model maps `X_t = (x_(t-L), ..., x_(t-1))` to a one-step prediction of `x_t`, with `L=32`. It is an explicitly coded, small Transformer-style encoder.

Use chronological splits on *target indices*: training 32–3499, validation 3500–4249, test 4250–4999. The training segment determines normalization statistics. Validation determines the selected checkpoint. The test targets are held out until final evaluation.

## Perturbation P

For a normalized historical input window of length L:

`epsilon ~ N(0, sigma_p^2 I_L)` and `X'_t = X_t + epsilon`.

Because the inputs are normalized by the training standard deviation, sigma is expressed in training-standard-deviation units. The values 0.01, 0.05, 0.10 and 0.20 represent increasingly strong, but still controlled, perturbations. Sigma=0 is the exact no-perturbation control.

30 independently generated draws are used for each sigma. All comparisons reuse the same trained weights. The unperturbed target `x_t` is **not** modified.

## Response and accuracy are distinct

Clean forecast: `f_t = F(X_t)`.

Perturbed forecast: `f'_t = F(X_t + epsilon)`.

Response: `delta_t = f'_t - f_t`.

Primary measure: average absolute response, `mean_t |delta_t|`, further averaged over perturbation repeats.

Accuracy is evaluated separately against **original** targets, using RMSE and NMSE. NMSE divides MSE by the *training-set variance*, keeping the denominator fixed across perturbation strengths and avoiding misleading changes due to separate window variances.

A large response is sensitivity; it is not by itself proof that the prediction becomes less accurate. Some perturbations can accidentally improve predictions.

## Interpreting perturbation size and sample fidelity

A small sigma means each coordinate of the normalized input moves only a fraction of its training standard deviation. For L dimensions, the expected squared Euclidean size under isotropic Gaussian perturbations is `L*sigma_p^2` (in normalized coordinates), so `sqrt(L)*sigma_p` is the root-mean-squared vector norm.

This controls *per-coordinate* scale, but does not guarantee the perturbed windows follow the original AR(1) distribution: independent added noise changes temporal covariance. In future experiments, distribution fidelity can be assessed using covariance diagnostics or divergences between original and perturbed distributions.

## Distribution choices for follow-up experiments

| Choice | Why consider it | What it changes |
| --- | --- | --- |
| Isotropic Gaussian (current) | Analytically transparent, repeatable, controlled variance | Noise magnitude without a preferred direction |
| Correlated multivariate Gaussian | Test sensitivity to temporal correlation and covariance directions | Covariance eigenvalues and eigenvectors |
| Laplace | Heavier tails reveal sensitivity to occasional larger deviations | Tail behavior |
| Uniform | Bounded noise provides a control without large deviations | Support shape |
| Gaussian mixtures | Study multimodality and different local density regions | Number/position/weight of modes |

Start with isotropic Gaussian variance sweeps. Then choose a fixed direction or covariance eigenvector and apply a scaled perturbation along it, changing one factor at a time. Do not label input noise as a distribution shift in the underlying generator.

## Model training as an experimental variable

Hold architecture and dataset fixed while varying one training feature, e.g. window length, weight decay, random seed or loss. Compare the **response curve** and **unperturbed test accuracy**; improvements in one do not guarantee improvements in the other.

## Long-horizon extension

For one-step autoregressive rollouts, future predictions can be fed back into future inputs. Compare response magnitude as rollout length increases, and measure accumulated prediction drift. This is not yet implemented and should not be inferred from the one-step figures.

## Statistical manifolds and surrogate models: future direction

The Gaussian family can be parameterized by mean and covariance. One can later define paths through this parameter space, perturb data drawn from those distributions, and compare forecaster response. A Fisher-information metric is a possible way to formalize local distribution distance, but it is **not** calculated by the current experiment.

A future interpretable surrogate `S` could be trained to approximate Transformer `F`. Two different fidelity questions then matter: output fidelity `|S(X)-F(X)|` and response fidelity `|(S(X+epsilon)-S(X))-(F(X+epsilon)-F(X))|`. These should be studied separately; the advisor's abbreviated "mutual D2" description should be clarified before a formal loss is selected.

## Validity checks and reporting

1. Verify forward/backward propagation and split boundaries using the smoke test.
2. Report both mean sensitivity and forecast accuracy at each perturbation strength.
3. Compare multiple independently trained model seeds in future work; 30 noise draws for one trained model do **not** provide uncertainty across training seeds.
4. Include a linear AR(1) baseline, which is appropriate for the selected generator.
5. Use matched clean and perturbed test inputs, a fixed pretrained checkpoint, and common evaluation targets.
6. Be explicit that the current experiments are exploratory and do not prove robustness under regime shifts.

## Implemented AR(1) baseline comparison

The new script `code/linear_baseline_comparison.py` fits `z[t] = intercept + slope * z[t-1]` by ordinary least squares on **training target indices only**. It uses the same normalized series as the Transformer. The fit learns its slope; it does not use the generator's true slope 0.7. Both models produce one-step forecasts for the same test times.

For every sigma and repetition, the script draws Gaussian noise **once**, applies it to the same historical test windows for both models, and evaluates responses and accuracy against the same original targets. The Transformer checkpoint is held fixed. These paired perturbations reduce between-model variation from unrelated random noise.

Because only the last coordinate affects the fitted AR(1), its theoretical mean absolute response under isotropic Gaussian perturbations is:

`E|delta| = abs(slope) * sigma * training_std * sqrt(2/pi)`.

This provides a transparent analytical prediction to compare against empirical perturbation results. The Transformer may depend on any input position, so its response reflects the learned attention and forecasting head, not necessarily the underlying generator.

Both scripts save per-draw and mean-summary CSVs, along with original-unit RMSE and fixed-training-variance NMSE. The comparison generates a two-panel plot separating output sensitivity from forecast error.

The initial Windows Transformer results are recorded in [initial_results.md](transformer_experiment/initial_results.md). The researcher's Windows baseline comparison also completed successfully; the confirmed results are in [linear_baseline_validation.md](transformer_experiment/linear_baseline_validation.md). The next coordinate-based experiment has been implemented and tested locally, and is described in [DIRECTIONAL_PERTURBATIONS.md](DIRECTIONAL_PERTURBATIONS.md).

## Next implemented experiment: one-coordinate directional perturbations

For each position j = 1,...,32, perturb ONLY that historical input coordinate with Gaussian noise of normalized standard deviation 0.10. Reuse the same perturbed test windows for both the fitted AR(1) and frozen Transformer. Record the change in prediction and accuracy relative to unchanged original targets. The [directional methods document](DIRECTIONAL_PERTURBATIONS.md) spells out the noise protocol, interpretation and expected analytical AR(1) behavior, and how to run the new code.

The coordinate study probes which historical positions influence prediction and separates the results by model. It does **not** yet implement joint covariance perturbations, Riemannian distances, long-horizon rollouts or a surrogate fidelity loss.


## Correlated multivariate Gaussian experiment (completed in Windows VS Code)

The new [covariance study](CORRELATED_GAUSSIAN.md) introduces epsilon ~ N(0, sigma² C), with C chosen from four symmetric positive-definite 32x32 correlation patterns. Every C has diagonal 1, so each coordinate has noise variance sigma² and the expected squared perturbation norm equals 32*sigma². The only deliberate change is in off-diagonal temporal covariance.

We use one independent Gaussian draw z per test window and repetition, then transform it using Cholesky factors of each C. Reusing z across conditions and applying the same resulting noisy windows to the frozen Transformer and training-fitted AR(1) provides a paired comparison. The original targets and forecaster weights stay fixed.

We record the mean absolute forecast change and original-target RMSE separately; per-repeat paired differences relative to independent noise make small covariance effects easier to assess. The expected AR(1) response is theoretically invariant under these covariance conditions, since AR(1) depends only on the newest input and all profiles give that coordinate the same marginal variance.

The researcher completed the experiment in Windows VS Code and uploaded the figures; see [confirmed covariance results](CORRELATED_GAUSSIAN.md). The run should be interpreted as controlled input-noise sensitivity, not proof of robustness to a changed generating distribution or a calculated statistical-manifold distance.
