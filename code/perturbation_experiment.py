"""Measure a fixed trained Transformer's response to Gaussian input perturbations.

Perturbations change ONLY historical test inputs, not the model weights or the
original forecast targets. Sigma is measured in training-standard-deviation
units because model input windows are normalized.

Run after training:
    python code/perturbation_experiment.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformer_model import ART, Forecaster, load_split

PERTURBATION_STDS = [0.0, 0.01, 0.05, 0.10, 0.20]
NUM_REPEATS = 30
PERTURBATION_SEED = 123


def main():
    _, _, test_dataset, statistics = load_split()

    checkpoint = ART / "transformer.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint}. Run code/transformer_model.py first."
        )

    model = Forecaster()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    windows, targets = [], []
    for input_window, target in DataLoader(test_dataset, batch_size=256):
        windows.append(input_window)
        targets.append(target)
    input_windows = torch.cat(windows)
    target_original_units = (
        torch.cat(targets).numpy() * statistics["std"] + statistics["mean"]
    )

    with torch.no_grad():
        clean_forecast = (
            model(input_windows).numpy() * statistics["std"] + statistics["mean"]
        )

    rng = np.random.default_rng(PERTURBATION_SEED)
    rows = []

    for sigma in PERTURBATION_STDS:
        for repeat in range(NUM_REPEATS):
            # Gaussian perturbation for every position in every test window.
            # Sigma = 0 is a control condition (no change to the input).
            perturbation = torch.from_numpy(
                rng.normal(0.0, sigma, size=tuple(input_windows.shape)).astype(
                    np.float32
                )
            )
            with torch.no_grad():
                perturbed_forecast = (
                    model(input_windows + perturbation).numpy()
                    * statistics["std"]
                    + statistics["mean"]
                )

            prediction_change = perturbed_forecast - clean_forecast
            mse = float(
                np.mean((perturbed_forecast - target_original_units) ** 2)
            )
            rows.append({
                "sigma_normalized": sigma,
                "repeat": repeat,
                "mean_absolute_response": float(
                    np.mean(np.abs(prediction_change))
                ),
                "rmse_original_targets": float(np.sqrt(mse)),
                "nmse_train_variance": mse / statistics["std"] ** 2,
            })

    ART.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(ART / "perturbation_metrics.csv", index=False)

    summary = results.groupby("sigma_normalized", as_index=False).agg(
        mean_absolute_response=("mean_absolute_response", "mean"),
        mean_rmse_original_targets=("rmse_original_targets", "mean"),
        mean_nmse_train_variance=("nmse_train_variance", "mean"),
        response_std_across_repeats=("mean_absolute_response", "std"),
    )
    summary.to_csv(ART / "perturbation_summary.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        summary["sigma_normalized"],
        summary["mean_absolute_response"],
        marker="o",
    )
    axis.set(
        xlabel="Perturbation std / training std",
        ylabel="Mean absolute prediction change (original units)",
        title="Transformer response to Gaussian input perturbations",
    )
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(ART / "perturbation_response.png", dpi=160)
    plt.close(figure)

    print(summary.to_string(index=False))
    print("Saved perturbation metrics, summary, and plot to", ART)


if __name__ == "__main__":
    main()
