"""Compare one-step forecasts under controlled multivariate Gaussian input noise.

Run AFTER training the Transformer:
    py code/test_correlated_gaussian.py
    py code/correlated_gaussian_experiment.py

Noise: epsilon ~ N(0, sigma^2 C), with C a 32x32 correlation matrix.
Every C has unit diagonal, so the marginal variance of each position and
E||epsilon||^2 = WINDOW * sigma^2 remain fixed. Only correlation changes.

Conditions:
    independent:       C[i,j] = 1(i=j)
    positive_temporal: C[i,j] = ( 0.8)^abs(i-j)
    negative_temporal: C[i,j] = (-0.8)^abs(i-j)
    common_mode:       C[i,j] = 0.5 if i!=j, else 1

The same base Gaussian draws are reused across profiles per repeat, and the
two models receive the same perturbed test windows. Models and targets remain
fixed. This does not implement information geometry or shift the generator.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from directional_perturbation import load_frozen_model, predict_transformer
from linear_baseline_comparison import fit_ar1, predict_ar1, score
from transformer_model import ART, WINDOW, load_split

PROFILES = ("independent", "positive_temporal", "negative_temporal", "common_mode")
DEFAULT_SIGMA = 0.10
DEFAULT_REPEATS = 20
DEFAULT_SEED = 2026


def correlation_matrix(name: str, length: int = WINDOW) -> np.ndarray:
    """Create a symmetric positive-definite correlation matrix (diagonal 1).

    Positive/negative temporal profiles are Toeplitz AR(1) covariance
    patterns; the common-mode profile is equicorrelated with off-diagonal
    correlation 0.5. None is estimated using held-out test observations.
    """
    if not isinstance(length, int) or length <= 1:
        raise ValueError("length must be an integer >= 2")
    offsets = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    if name == "independent":
        matrix = np.eye(length, dtype=np.float64)
    elif name == "positive_temporal":
        matrix = np.power(0.8, offsets, dtype=np.float64)
    elif name == "negative_temporal":
        matrix = np.power(-0.8, offsets, dtype=np.float64)
    elif name == "common_mode":
        matrix = 0.5 * np.ones((length, length), dtype=np.float64)
        np.fill_diagonal(matrix, 1.0)
    else:
        raise ValueError(f"Unknown correlation profile {name!r}: {PROFILES}")
    np.linalg.cholesky(matrix)  # Fail early if a modified matrix is invalid.
    return matrix


def correlated_noise(z: np.ndarray, correlation: np.ndarray, sigma: float) -> np.ndarray:
    """Map iid standard-normal rows z into N(0, sigma^2 * correlation).

    z has shape (num_test_windows, WINDOW). Reusing z across conditions
    creates paired Monte Carlo perturbations. Output is float32 for PyTorch.
    """
    z = np.asarray(z, dtype=np.float64)
    matrix = np.asarray(correlation, dtype=np.float64)
    if z.ndim != 2 or matrix.shape != (z.shape[1], z.shape[1]):
        raise ValueError("z must be (samples,L) and correlation must be (L,L)")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 1):
        raise ValueError("Correlation matrix must be symmetric with unit diagonal")
    chol = np.linalg.cholesky(matrix)
    return (sigma * (z @ chol.T)).astype(np.float32)


def plot_correlation_profiles(matrices, destination):
    """Visualize designed covariance correlations, not empirical estimates."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(14, 3.5), sharex=True, sharey=True)
    for axis, (name, matrix) in zip(axes, matrices.items()):
        im = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        axis.set_title(name.replace("_", " "), fontsize=10)
        axis.set_xlabel("Input position")
        axis.set_xticks([0, WINDOW - 1], [1, WINDOW])
        axis.set_yticks([0, WINDOW - 1], [1, WINDOW])
    axes[0].set_ylabel("Input position")
    fig.colorbar(im, ax=axes.tolist(), label="Correlation", shrink=0.78)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.83, bottom=0.17, wspace=0.15)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_responses(summary, destination, sigma):
    """Keep perturbation sensitivity and forecast error as separate results."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    x = np.arange(len(PROFILES))
    for model_name, offset in (("Fitted AR(1)", -0.19), ("Transformer", 0.19)):
        group = summary[summary["model"] == model_name].set_index("profile").loc[list(PROFILES)]
        axes[0].bar(x + offset, group["mean_absolute_response"], width=0.36, label=model_name)
        axes[1].bar(x + offset, group["mean_rmse_original_targets"], width=0.36, label=model_name)
    axes[0].set_ylabel("Mean absolute forecast change (original units)")
    axes[0].set_title(f"Response to N(0, sigma^2 C), normalized sigma = {sigma:g}")
    axes[1].set_ylabel("Mean RMSE against original targets")
    axes[1].set_title("Forecast error (lower is better)")
    for axis in axes:
        axis.set_xticks(x, [label.replace("_", "\n") for label in PROFILES])
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_paired_effects(effects, destination):
    """Show small changes relative to independent noise using paired draws."""
    fig, (sensitivity_ax, rmse_ax) = plt.subplots(2, 1, figsize=(10, 8))
    x = np.arange(len(PROFILES))
    for model_name, offset in (("Fitted AR(1)", -0.19), ("Transformer", 0.19)):
        group = effects[effects["model"] == model_name].set_index("profile").loc[list(PROFILES)]
        sensitivity_ax.bar(
            x + offset, group["mean_response_difference"],
            yerr=group["response_sem_draws"], capsize=3, width=0.36, label=model_name
        )
        rmse_ax.bar(
            x + offset, group["mean_rmse_difference"],
            yerr=group["rmse_sem_draws"], capsize=3, width=0.36, label=model_name
        )
    sensitivity_ax.set(
        title="Paired sensitivity difference relative to independent noise",
        ylabel="Change in mean absolute response"
    )
    rmse_ax.set(
        title="Paired RMSE difference relative to independent noise",
        ylabel="Change in RMSE"
    )
    for axis in (sensitivity_ax, rmse_ax):
        axis.set_xticks(x, [label.replace("_", "\n") for label in PROFILES])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        axis.set_axisbelow(True)
    fig.text(
        0.5, 0.005,
        "Error bars: standard error across noise draws, NOT model-training uncertainty",
        ha="center", fontsize=9
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Marginal noise std, in training-normalized units")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be positive and finite")
    if args.repeats < 2:
        parser.error("--repeats must be >= 2 for across-draw std")

    train, _, test, stats = load_split()
    transformer = load_frozen_model(stats)  # Reuse the frozen saved checkpoint.
    intercept, slope = fit_ar1(train)        # Training-only OLS fit.

    model_inputs, targets = [], []
    for x, y in DataLoader(test, batch_size=256, shuffle=False):
        model_inputs.append(x)
        targets.append(y)
    windows = torch.cat(model_inputs)
    actual = torch.cat(targets).numpy() * stats["std"] + stats["mean"]

    def linear_predict(inputs):
        return predict_ar1(inputs.numpy(), intercept, slope) * stats["std"] + stats["mean"]

    def transformer_predict(inputs):
        return predict_transformer(transformer, inputs, stats["mean"], stats["std"])

    predictors = {"Fitted AR(1)": linear_predict, "Transformer": transformer_predict}
    clean = {name: predictor(windows) for name, predictor in predictors.items()}
    clean_rmse = {
        name: score(actual, prediction, stats["std"])["rmse"]
        for name, prediction in clean.items()
    }
    matrices = {name: correlation_matrix(name) for name in PROFILES}
    rng = np.random.default_rng(args.seed)
    records = []
    print(f"Comparing {len(PROFILES)} Gaussian correlations, marginal sigma={args.sigma}")
    print(f"Each C is positive definite, has unit diagonal, and trace={WINDOW}.")
    print("Common base noise across profiles and both models; unchanged original targets.")

    # New base Gaussian per repetition; SAME base z reused for every covariance
    # profile in that repetition and for both forecasting models.
    for rep in range(args.repeats):
        base_z = rng.standard_normal((len(windows), WINDOW))
        for profile, c in matrices.items():
            eps = correlated_noise(base_z, c, args.sigma)
            perturbed = windows + torch.from_numpy(eps[:, :, None])
            for name, predict in predictors.items():
                forecast = predict(perturbed)
                response = forecast - clean[name]
                accuracy = score(actual, forecast, stats["std"])
                records.append({
                    "repeat": rep,
                    "profile": profile,
                    "model": name,
                    "sigma_normalized": args.sigma,
                    "mean_absolute_response": float(np.mean(np.abs(response))),
                    "mean_signed_response": float(np.mean(response)),
                    "rmse_original_targets": accuracy["rmse"],
                    "nmse_train_variance": accuracy["nmse_train_variance"],
                    "empirical_noise_squared_norm": float(np.mean(np.sum(eps ** 2, axis=1))),
                })
        if (rep + 1) % 5 == 0 or (rep + 1) == args.repeats:
            print(f"Completed {rep + 1}/{args.repeats} repetitions", flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    per_repeat = pd.DataFrame(records)
    summary = per_repeat.groupby(
        ["profile", "model", "sigma_normalized"], as_index=False
    ).agg(
        mean_absolute_response=("mean_absolute_response", "mean"),
        response_std_across_repeats=("mean_absolute_response", "std"),
        mean_rmse_original_targets=("rmse_original_targets", "mean"),
        mean_nmse_train_variance=("nmse_train_variance", "mean"),
        mean_noise_squared_norm=("empirical_noise_squared_norm", "mean"),
    )
    summary["clean_rmse"] = summary["model"].map(clean_rmse)
    summary["rmse_increase_vs_clean"] = summary["mean_rmse_original_targets"] - summary["clean_rmse"]

    reference = per_repeat.loc[
        per_repeat["profile"] == "independent",
        ["repeat", "model", "mean_absolute_response", "rmse_original_targets"]
    ].rename(columns={
        "mean_absolute_response": "independent_mean_absolute_response",
        "rmse_original_targets": "independent_rmse_original_targets"
    })
    paired = per_repeat.merge(reference, on=["repeat", "model"], how="left", validate="many_to_one")
    paired["response_difference_vs_independent"] = (
        paired["mean_absolute_response"] - paired["independent_mean_absolute_response"]
    )
    paired["rmse_difference_vs_independent"] = (
        paired["rmse_original_targets"] - paired["independent_rmse_original_targets"]
    )
    paired_effects = paired.groupby(["profile", "model"], as_index=False).agg(
        mean_response_difference=("response_difference_vs_independent", "mean"),
        std_response_difference=("response_difference_vs_independent", "std"),
        mean_rmse_difference=("rmse_difference_vs_independent", "mean"),
        std_rmse_difference=("rmse_difference_vs_independent", "std")
    )
    paired_effects["response_sem_draws"] = paired_effects["std_response_difference"] / np.sqrt(args.repeats)
    paired_effects["rmse_sem_draws"] = paired_effects["std_rmse_difference"] / np.sqrt(args.repeats)

    per_repeat.to_csv(ART / "correlated_gaussian_per_repeat.csv", index=False)
    summary.to_csv(ART / "correlated_gaussian_summary.csv", index=False)
    paired.to_csv(ART / "correlated_gaussian_paired_deltas.csv", index=False)
    paired_effects.to_csv(ART / "correlated_gaussian_paired_effects.csv", index=False)
    pd.DataFrame([
        {"profile": profile, "row_position": i + 1, "column_position": j + 1,
         "correlation": float(c[i, j])}
        for profile, c in matrices.items()
        for i in range(WINDOW) for j in range(WINDOW)
    ]).to_csv(ART / "correlated_gaussian_covariances.csv", index=False)
    (ART / "correlated_gaussian_config.json").write_text(json.dumps({
        "generator": "stationary Gaussian AR(1)",
        "frozen_model": "previously trained transformer.pt; no retraining",
        "window": WINDOW, "sigma_normalized": args.sigma,
        "repeats": args.repeats, "seed": args.seed,
        "profiles": list(PROFILES),
        "noise_covariance": "sigma^2 * C; diag(C)=1",
        "expected_squared_norm_normalized": WINDOW * args.sigma ** 2,
        "fitted_ar1_intercept_normalized": intercept,
        "fitted_ar1_slope_normalized": slope,
        "training_std": stats["std"], "clean_rmse": clean_rmse
    }, indent=2), encoding="utf-8")

    plot_correlation_profiles(matrices, ART / "correlated_gaussian_matrices.png")
    plot_responses(summary, ART / "correlated_gaussian_response.png", args.sigma)
    plot_paired_effects(paired_effects, ART / "correlated_gaussian_paired_effects.png")

    expected_linear_response = abs(slope) * args.sigma * stats["std"] * np.sqrt(2.0 / np.pi)
    print("\nTheoretical fitted AR(1) E|response| for EACH covariance profile:",
          f"{expected_linear_response:.6f}")
    print("AR(1) sensitivity is theoretically invariant because last-position noise variance stays fixed.")
    print(summary[[
        "profile", "model", "mean_absolute_response",
        "mean_rmse_original_targets", "rmse_increase_vs_clean"
    ]].to_string(index=False))
    print(f"\nSaved covariance matrices, tables, and plots to {ART}")


if __name__ == "__main__":
    main()
