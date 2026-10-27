"""First end-to-end controlled-ODE perturbation example.

Run from repository root: python code/run_controlled_ode.py
Only NumPy and Matplotlib are required. Outputs are regenerable and ignored.
"""

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Works in terminal/headless environments.
import matplotlib.pyplot as plt

from controlled_ode_framework import (
    DynamicsConfig, fit_linear_forecaster, make_windows, predict_linear,
    sample_irregular, sample_regular, simulate,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "ode_experiment"


def main() -> None:
    cfg = DynamicsConfig()
    baseline = simulate(cfg)
    changed_control = simulate(cfg, control_offset=0.08)
    changed_parameter = simulate(cfg, parameter_offset=0.12)

    time, clean = sample_regular(baseline)
    _, control_series = sample_regular(changed_control)
    _, parameter_series = sample_regular(changed_parameter)
    irregular_time, irregular_values = sample_irregular(baseline)

    # Model training stays entirely before the phase change; model stays frozen.
    window = 12
    x_train_all, y_train_all = make_windows(clean, window)
    target_times = time[window:]
    train_mask = target_times < cfg.phase_time - 2.0
    eval_mask = target_times >= 82.0  # After the intervention at t=80.
    coefficients = fit_linear_forecaster(x_train_all[train_mask], y_train_all[train_mask])
    clean_forecast = predict_linear(x_train_all, coefficients)

    # The same timestamps and exact same input positions are used in every run.
    changes = {"control_offset": control_series, "parameter_offset": parameter_series}
    rng = np.random.default_rng(2027)
    # Observation noise perturbs S, not g. No generator response in this case.
    changes["observation_noise"] = clean + rng.normal(0.0, 0.08, size=len(clean))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, series in changes.items():
        features, targets = make_windows(series, window)
        changed_forecast = predict_linear(features, coefficients)
        delta_generator = targets - y_train_all if label != "observation_noise" else np.zeros_like(targets)
        delta_forecast = changed_forecast - clean_forecast
        # Paired response measures, in original physical units.
        rows.append({
            "intervention": label,
            "generator_mean_absolute_response": float(np.mean(np.abs(delta_generator[eval_mask]))),
            "forecast_mean_absolute_response": float(np.mean(np.abs(delta_forecast[eval_mask]))),
            "response_mismatch_mae": float(np.mean(np.abs((delta_forecast - delta_generator)[eval_mask]))),
            "perturbed_forecast_rmse": float(np.sqrt(np.mean((changed_forecast[eval_mask] - (targets if label != "observation_noise" else y_train_all)[eval_mask])**2))),
        })

    with (OUTPUT / "response_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump({"generator": vars(cfg), "window": window, "train_end_before": cfg.phase_time - 2,
                   "perturbation_onset": 80.0, "evaluation_start": 82.0,
                   "control_offset": 0.08, "parameter_offset": 0.12,
                   "observation_noise_std": 0.08, "irregular_sampler_seed": 99}, handle, indent=2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time, clean, label="Baseline trajectory", linewidth=1.5)
    axes[0].plot(time, control_series, label="Perturbed control u(t)", alpha=0.8)
    axes[0].plot(time, parameter_series, label="Perturbed parameter a(t)", alpha=0.8)
    axes[0].axvline(cfg.phase_time, linestyle="--", color="gray", label="Regime change (t=60)")
    axes[0].axvline(80.0, linestyle=":", color="black", label="Interventions (t=80)")
    axes[0].set(ylabel="State x(t)", title="Paired controlled ODE trajectories")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(time, clean, label="Continuous-grid trajectory", alpha=0.6)
    axes[1].scatter(irregular_time, irregular_values, s=7, label="Irregular samples")
    axes[1].set(xlabel="Time", ylabel="Sampled state", title="Sampling the baseline trajectory")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ode_trajectories.png", dpi=150)
    plt.close(fig)

    print(f"Saved experiment outputs to: {OUTPUT}")
    print(f"Samples: {len(time)} regular, {len(irregular_time)} irregular")
    print(f"Linear baseline: {train_mask.sum()} training targets; {eval_mask.sum()} evaluation targets")
    for row in rows:
        print(f"{row['intervention']}: generator MAE response {row['generator_mean_absolute_response']:.5f}; "
              f"forecast MAE response {row['forecast_mean_absolute_response']:.5f}; "
              f"response mismatch {row['response_mismatch_mae']:.5f}")
    print("Note: these are first-pass diagnostics, not GRU/LSTM/Transformer results.")


if __name__ == "__main__":
    main()
