"""Compare a fitted AR(1) predictor with the frozen Transformer on paired inputs.

Run AFTER generator and Transformer training:
    py code/linear_baseline_comparison.py

Both models predict identical target times and use original-unit error metrics.
Fit the linear baseline on normalized training targets only. Apply the same
Gaussian noise draws to the same normalized test input windows for both models.
The Transformer weights are not changed. Derived outputs are git-ignored.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformer_model import ART, Forecaster, load_split, WINDOW
from perturbation_experiment import (
    NUM_REPEATS,
    PERTURBATION_SEED,
    PERTURBATION_STDS,
)


def fit_ar1(training_dataset):
    """Fit y[t] = intercept + slope * y[t-1] by OLS, using training targets.

    Variables are normalized using training-only mean and standard deviation.
    Fitting is independent of the known generator coefficient of 0.7.
    """
    series = np.asarray(training_dataset.series, dtype=np.float64)
    times = training_dataset.indices
    previous, targets = series[times - 1], series[times]
    design = np.column_stack([np.ones(len(times)), previous])
    intercept, slope = np.linalg.lstsq(design, targets, rcond=None)[0]
    return float(intercept), float(slope)


def predict_ar1(windows, intercept, slope):
    """One prediction per sample from the last position of (samples,32,1)."""
    data = np.asarray(windows)
    if data.ndim != 3 or data.shape[1:] != (WINDOW, 1):
        raise ValueError(f"Expected (samples,{WINDOW},1); got {data.shape}")
    return intercept + slope * data[:, -1, 0]


def score(actual, forecast, training_std):
    """Evaluate in original units, normalizing NMSE by fixed train variance."""
    residual = np.asarray(actual) - np.asarray(forecast)
    mse = float(np.mean(residual ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "nmse_train_variance": float(mse / (training_std ** 2)),
    }


def main():
    train, _, test, stats = load_split()
    checkpoint = ART / "transformer.pt"
    config_path = ART / "config.json"
    if not checkpoint.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Train first: py code/transformer_model.py (missing checkpoint/config)."
        )

    # Guard against comparing a checkpoint with a newly regenerated dataset.
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    for field in ("window", "n", "train_end", "val_end"):
        current = WINDOW if field == "window" else stats[field]
        if int(saved[field]) != int(current):
            raise ValueError(f"Model config {field} differs from current dataset")
    for field in ("mean", "std"):
        if not np.isclose(saved[field], stats[field], rtol=1e-6, atol=1e-7):
            raise ValueError(f"Model config {field} differs from current dataset")

    intercept, slope = fit_ar1(train)
    print(f"Fitted AR(1), normalized units: intercept={intercept:.6f}, slope={slope:.6f}")

    test_windows, test_targets = [], []
    for inputs, targets in DataLoader(test, batch_size=256, shuffle=False):
        test_windows.append(inputs)
        test_targets.append(targets)
    windows = torch.cat(test_windows)
    targets = torch.cat(test_targets).numpy() * stats["std"] + stats["mean"]

    transformer = Forecaster(length=WINDOW)
    transformer.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    transformer.eval()

    def transformer_predict(batch):
        with torch.no_grad():
            return transformer(batch).numpy() * stats["std"] + stats["mean"]

    def linear_predict(batch):
        return (
            predict_ar1(batch.numpy(), intercept, slope) * stats["std"]
            + stats["mean"]
        )

    predictors = {
        "Fitted AR(1)": linear_predict,
        "Transformer": transformer_predict,
    }
    clean_forecasts = {name: fn(windows) for name, fn in predictors.items()}
    clean_metrics = []
    for name, clean in clean_forecasts.items():
        metrics = score(targets, clean, stats["std"])
        clean_metrics.append({"model": name, **metrics})
        print(f"Clean {name}: RMSE={metrics['rmse']:.6f}; MAE={metrics['mae']:.6f}")

    rng = np.random.default_rng(PERTURBATION_SEED)
    per_draw = []
    # One draw for each sigma/repetition, reused by both models.
    for sigma in PERTURBATION_STDS:
        for repeat in range(NUM_REPEATS):
            epsilon = torch.from_numpy(
                rng.normal(0, sigma, size=tuple(windows.shape)).astype(np.float32)
            )
            perturbed = windows + epsilon
            for name, predict in predictors.items():
                changed = predict(perturbed)
                diff = changed - clean_forecasts[name]
                metrics = score(targets, changed, stats["std"])
                per_draw.append({
                    "model": name,
                    "sigma_normalized": sigma,
                    "repeat": repeat,
                    "mean_absolute_response": float(np.mean(np.abs(diff))),
                    "rmse_original_targets": metrics["rmse"],
                    "nmse_train_variance": metrics["nmse_train_variance"],
                })

    ART.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean_metrics).to_csv(
        ART / "linear_vs_transformer_clean_metrics.csv", index=False
    )
    pd.DataFrame(per_draw).to_csv(
        ART / "linear_vs_transformer_per_draw.csv", index=False
    )
    summary = pd.DataFrame(per_draw).groupby(
        ["model", "sigma_normalized"], as_index=False
    ).agg(
        mean_absolute_response=("mean_absolute_response", "mean"),
        response_std_across_repeats=("mean_absolute_response", "std"),
        mean_rmse_original_targets=("rmse_original_targets", "mean"),
        mean_nmse_train_variance=("nmse_train_variance", "mean"),
    )
    summary.to_csv(ART / "linear_vs_transformer_summary.csv", index=False)
    (ART / "linear_baseline_parameters.json").write_text(
        json.dumps({
            "normalized_intercept": intercept,
            "normalized_slope": slope,
            "fit": "ordinary least squares, training targets only",
            "perturbation_seed": PERTURBATION_SEED,
            "repetitions_per_sigma": NUM_REPEATS,
        }, indent=2), encoding="utf-8"
    )

    fig, (ax_response, ax_rmse) = plt.subplots(1, 2, figsize=(12, 5))
    for name in predictors:
        group = summary[summary["model"] == name]
        ax_response.plot(
            group["sigma_normalized"], group["mean_absolute_response"],
            marker="o", label=name
        )
        ax_rmse.plot(
            group["sigma_normalized"], group["mean_rmse_original_targets"],
            marker="o", label=name
        )
    ax_response.set(
        title="Response sensitivity",
        xlabel="Perturbation std / training std",
        ylabel="Mean absolute forecast change (original units)",
    )
    ax_rmse.set(
        title="Forecast accuracy",
        xlabel="Perturbation std / training std",
        ylabel="RMSE vs. original targets",
    )
    for axis in (ax_response, ax_rmse):
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Fitted linear AR(1) vs Transformer: paired noise draws")
    fig.tight_layout()
    fig.savefig(ART / "linear_vs_transformer.png", dpi=160)
    plt.close(fig)

    # Analytical Gaussian expected absolute response for fitted AR(1).
    # Only the last coordinate matters; epsilon_last ~ N(0, sigma^2).
    print("\nExpected fitted-AR(1) absolute response (theory):")
    for sigma in PERTURBATION_STDS:
        expectation = abs(slope) * sigma * stats["std"] * np.sqrt(2 / np.pi)
        print(f"sigma={sigma:.2f} theoretical E|response|={expectation:.6f}")
    print("\nPaired empirical results:")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison CSVs, parameters, and plot to {ART}")


if __name__ == "__main__":
    main()
