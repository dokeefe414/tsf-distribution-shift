"""Measure input-position sensitivity for a frozen Transformer and fitted AR(1).

For each of the 32 historical input coordinates, add N(0, sigma**2)
noise to that coordinate *only*, leaving the other 31 unchanged. Reuse
identical perturbations for both models and evaluate on original test targets.

Usage from the repository root (after training the Transformer):
    py code/directional_perturbation.py
    py code/directional_perturbation.py --sigma 0.05 --repeats 20

Outputs are written to reports/transformer_experiment and ignored by Git.
This is a one-step, coordinate-direction experiment, NOT a multivariate
covariance or statistical-manifold experiment.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from linear_baseline_comparison import fit_ar1, predict_ar1
from transformer_model import ART, WINDOW, Forecaster, load_split

DEFAULT_SIGMA = 0.10
DEFAULT_REPEATS = 12
DEFAULT_SEED = 2026


def perturb_one_position(windows, position, values):
    """Return a COPY of (N,L,1) windows with changes only at one position.

    Position 0 is oldest, position L-1 is most recent. The values array has
    one Gaussian draw for each test window, in normalized units.
    """
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW, 1):
        raise ValueError(f"Expected windows of shape (N,{WINDOW},1)")
    if not 0 <= position < WINDOW:
        raise ValueError(f"Position must be between 0 and {WINDOW - 1}")
    if values.ndim != 1 or values.shape[0] != windows.shape[0]:
        raise ValueError("values must have one entry per window")
    result = windows.clone()
    result[:, position, 0] += torch.as_tensor(values, dtype=result.dtype)
    return result


def load_frozen_model(stats):
    """Load saved Transformer and verify that the scaler/split still match."""
    checkpoint = ART / "transformer.pt"
    config_path = ART / "config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Missing trained model/config. Run py code/transformer_model.py first."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for field in ("n", "train_end", "val_end"):
        if int(cfg[field]) != int(stats[field]):
            raise ValueError(f"Saved model and dataset disagree on {field}")
    if int(cfg["window"]) != WINDOW:
        raise ValueError("Saved model has an incompatible input window length")
    for field in ("mean", "std"):
        if not np.isclose(cfg[field], stats[field], rtol=1e-6, atol=1e-7):
            raise ValueError(f"Saved model and dataset disagree on {field}")
    model = Forecaster(length=WINDOW)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    return model.eval()


def predict_transformer(model, windows, training_mean, training_std, batch_size=256):
    """Evaluate frozen model in manageable batches; return original-unit predictions."""
    predictions = []
    with torch.inference_mode():
        for block in windows.split(batch_size):
            predictions.append(model(block).cpu().numpy())
    return np.concatenate(predictions) * training_std + training_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be positive and finite")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    train, _, test, stats = load_split()
    transformer = load_frozen_model(stats)
    intercept, slope = fit_ar1(train)
    inputs, targets = [], []
    for x, y in DataLoader(test, batch_size=256, shuffle=False):
        inputs.append(x)
        targets.append(y)
    windows = torch.cat(inputs)
    original_targets = (
        torch.cat(targets).numpy() * stats["std"] + stats["mean"]
    )

    def linear_predict(batch):
        return (
            predict_ar1(batch.numpy(), intercept, slope) * stats["std"]
            + stats["mean"]
        )

    def transformer_predict(batch):
        return predict_transformer(transformer, batch, stats["mean"], stats["std"])

    predictors = {"Fitted AR(1)": linear_predict, "Transformer": transformer_predict}
    clean = {name: predict(windows) for name, predict in predictors.items()}
    rng = np.random.default_rng(args.seed)
    rows = []

    for position in range(WINDOW):
        for repeat in range(args.repeats):
            # A scalar perturbation per test sample, applied to a single lag.
            # The SAME noisy windows are used to evaluate both predictors.
            noise = rng.normal(0.0, args.sigma, size=len(windows)).astype(np.float32)
            modified_windows = perturb_one_position(windows, position, noise)
            for model_name, predictor in predictors.items():
                forecast = predictor(modified_windows)
                response = forecast - clean[model_name]
                rows.append({
                    "model": model_name,
                    "position_oldest_to_newest": position + 1,
                    "lag_before_target": WINDOW - position,
                    "repeat": repeat,
                    "sigma_normalized": args.sigma,
                    "mean_absolute_response": float(np.mean(np.abs(response))),
                    "mean_signed_response": float(np.mean(response)),
                    "rmse_original_targets": float(
                        np.sqrt(np.mean((forecast - original_targets) ** 2))
                    ),
                })
        print(f"Processed position {position + 1:02d}/{WINDOW}", flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    per_repeat = pd.DataFrame(rows)
    summary = per_repeat.groupby(
        ["model", "position_oldest_to_newest", "lag_before_target", "sigma_normalized"],
        as_index=False,
    ).agg(
        mean_absolute_response=("mean_absolute_response", "mean"),
        response_std_across_repeats=("mean_absolute_response", "std"),
        mean_rmse_original_targets=("rmse_original_targets", "mean"),
    )
    per_repeat.to_csv(ART / "directional_per_repeat.csv", index=False)
    summary.to_csv(ART / "directional_summary.csv", index=False)

    names = ["Fitted AR(1)", "Transformer"]
    array = np.vstack([
        summary[summary["model"] == name]
        .sort_values("position_oldest_to_newest")["mean_absolute_response"]
        .to_numpy()
        for name in names
    ])

    figure, (heatmap_axis, line_axis) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1.1, 1.3]}
    )
    image = heatmap_axis.imshow(array, aspect="auto", interpolation="nearest", cmap="viridis")
    heatmap_axis.set_yticks([0, 1], names)
    heatmap_axis.set_xticks(np.arange(0, WINDOW, 4), np.arange(1, WINDOW + 1, 4))
    heatmap_axis.set_xlabel("Input position (1 = oldest, 32 = most recent)")
    heatmap_axis.set_title(f"Mean absolute response to one-coordinate Gaussian noise (sigma={args.sigma})")
    figure.colorbar(image, ax=heatmap_axis, label="Forecast change (original units)")

    positions = np.arange(1, WINDOW + 1)
    for i, name in enumerate(names):
        line_axis.plot(positions, array[i], label=name, marker=".")
    line_axis.set(
        xlabel="Input position (1 = oldest, 32 = most recent)",
        ylabel="Mean absolute forecast change (original units)",
        title="Sensitivity by historical observation",
        xlim=(1, WINDOW),
    )
    line_axis.grid(alpha=0.3)
    line_axis.legend()
    figure.tight_layout()
    figure.savefig(ART / "directional_sensitivity.png", dpi=160)
    plt.close(figure)

    expected_ar1_newest = (
        abs(slope) * args.sigma * stats["std"] * np.sqrt(2 / np.pi)
    )
    print(f"\nFitted AR(1) slope (normalized): {slope:.6f}")
    print(f"Theoretical AR(1) mean absolute response at position {WINDOW}: {expected_ar1_newest:.6f}")
    print("All earlier positions should have exactly zero AR(1) response.")
    print("\nHighest-response positions:")
    for name in names:
        subset = summary[summary["model"] == name].nlargest(5, "mean_absolute_response")
        print(f"\n{name}:\n{subset[['position_oldest_to_newest', 'lag_before_target', 'mean_absolute_response']].to_string(index=False)}")
    print(f"\nSaved directional tables and heatmap to {ART}")


if __name__ == "__main__":
    main()
