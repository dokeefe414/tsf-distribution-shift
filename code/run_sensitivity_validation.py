"""Validate ODE tangent equations, forecast response and one-dimensional OT.

Run: python code/run_sensitivity_validation.py --seeds 12 --bootstrap 500
Produces regenerable local CSV, JSON and PNG files (no neural networks).
"""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controlled_ode_framework import (
    DynamicsConfig, fit_linear_forecaster, make_windows, predict_linear,
    sample_regular, simulate,
)
from ode_sensitivity import (
    central_difference, paired_bootstrap_ci, simulate_variations, wasserstein_1d,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "sensitivity_experiment"
ONSET = 80.0
EVAL_START = 82.0
WINDOW = 12


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot save empty results")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(seeds: int = 12, bootstrap: int = 500, epsilon: float = 0.04, output: Path = DEFAULT_OUTPUT) -> dict:
    if seeds < 3 or bootstrap < 10 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Require >=3 independent seeds, >=10 bootstraps and epsilon > 0")
    output.mkdir(parents=True, exist_ok=True)
    cfg = DynamicsConfig()
    tangent = simulate_variations(cfg, onset=ONSET)
    time, reference_series = sample_regular(simulate(cfg, onset=ONSET))
    training_windows, training_targets = make_windows(reference_series, WINDOW)
    target_times = time[WINDOW:]
    training_mask = target_times < cfg.phase_time - 2.0
    eval_mask = target_times >= EVAL_START
    coefficients = fit_linear_forecaster(training_windows[training_mask], training_targets[training_mask])

    # Mathematical tangent responses versus finite differences.
    validation_rows = []
    epsilon_grid = [0.08, 0.04, 0.02, 0.01]
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, kind in zip(axes, ("control", "parameter")):
        sensitivity = tangent["s_" + kind]
        for h in epsilon_grid:
            numerical = central_difference(cfg, kind=kind, epsilon=h, onset=ONSET)
            error = numerical - sensitivity
            validation_rows.append({
                "intervention": kind, "epsilon": h,
                "mean_absolute_derivative_error": float(np.mean(np.abs(error))),
                "max_absolute_derivative_error": float(np.max(np.abs(error))),
            })
        numerical = central_difference(cfg, kind=kind, epsilon=epsilon_grid[-1], onset=ONSET)
        axis.plot(tangent["time"], sensitivity, label="Variational equation", linewidth=2)
        axis.plot(tangent["time"], numerical, "--", label="Central difference h=0.01", alpha=0.9)
        axis.axvline(ONSET, linestyle=":", color="black")
        axis.set(ylabel="dx/d(intervention)", title=f"Local {kind} sensitivity")
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time")
    figure.tight_layout()
    figure.savefig(output / "tangent_vs_finite_difference.png", dpi=160)
    plt.close(figure)
    write_csv(output / "derivative_validation.csv", validation_rows)

    # Frozen forecaster trained on one pre-change realization. Evaluation seeds
    # differ from training seed, and paired runs share identical random input.
    per_seed_rows = []
    time_index = int(round(100.0 / cfg.dt))
    state_at_100 = {"baseline": [], "control": [], "parameter": []}
    for seed in range(3000, 3000 + seeds):
        test_cfg = replace(cfg, seed=seed)
        base = simulate(test_cfg, onset=ONSET)
        _, baseline = sample_regular(base)
        features_base, target_base = make_windows(baseline, WINDOW)
        forecast_base = predict_linear(features_base, coefficients)
        state_at_100["baseline"].append(base["state"][time_index])
        for kind in ("control", "parameter"):
            changes = {"control_offset" if kind == "control" else "parameter_offset": epsilon}
            altered = simulate(test_cfg, onset=ONSET, **changes)
            _, perturbed = sample_regular(altered)
            features_pert, target_pert = make_windows(perturbed, WINDOW)
            forecast_pert = predict_linear(features_pert, coefficients)
            dg = (target_pert - target_base)[eval_mask]
            df = (forecast_pert - forecast_base)[eval_mask]
            per_seed_rows.append({
                "seed": seed, "intervention": kind,
                "generator_response_mae": float(np.mean(np.abs(dg))),
                "forecaster_response_mae": float(np.mean(np.abs(df))),
                "response_mismatch_mae": float(np.mean(np.abs(df - dg))),
                "perturbed_forecast_rmse": float(np.sqrt(np.mean((forecast_pert[eval_mask] - target_pert[eval_mask])**2))),
            })
            state_at_100[kind].append(altered["state"][time_index])
    write_csv(output / "per_seed_response.csv", per_seed_rows)

    aggregate_rows = []
    for kind in ("control", "parameter"):
        rows = [row for row in per_seed_rows if row["intervention"] == kind]
        for metric in ("generator_response_mae", "forecaster_response_mae", "response_mismatch_mae", "perturbed_forecast_rmse"):
            values = [row[metric] for row in rows]
            mean, low, high = paired_bootstrap_ci(values, draws=bootstrap, seed=11)
            aggregate_rows.append({"intervention": kind, "metric": metric, "mean": mean, "bootstrap_95_low": low, "bootstrap_95_high": high, "independent_control_seeds": seeds})
    write_csv(output / "response_bootstrap_summary.csv", aggregate_rows)

    # 1-D OT compares the single-time marginal at t=100. The response statistics
    # above preserve each paired trajectory identity, unlike sorted OT samples.
    baseline_draws = np.array(state_at_100["baseline"])
    ot_rows = []
    for kind in ("control", "parameter"):
        changed = np.array(state_at_100[kind])
        ot_rows.append({
            "intervention": kind, "time": 100.0,
            "empirical_w1": wasserstein_1d(baseline_draws, changed, order=1),
            "empirical_w2": wasserstein_1d(baseline_draws, changed, order=2),
            "mean_paired_absolute_change": float(np.mean(np.abs(changed - baseline_draws))),
            "independent_control_seeds": seeds,
        })
    write_csv(output / "distribution_at_t100.csv", ot_rows)
    config_record = {
        "generator": vars(cfg), "training_seed": cfg.seed, "evaluation_seed_range": [3000, 3000+seeds-1],
        "evaluation_time_start": EVAL_START, "intervention_onset": ONSET,
        "epsilon": epsilon, "window": WINDOW, "training_target_time_upper_bound_exclusive": cfg.phase_time - 2.0,
        "bootstrap_draws": bootstrap, "bootstrap_seed": 11,
        "finite_difference_epsilons": epsilon_grid,
        "ot_comparison_time": 100.0,
        "note": "Seed-level uncertainty; no neural network training; OT is a single-time marginal metric."
    }
    (output / "configuration.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    print(f"Output directory: {output}")
    for row in validation_rows:
        if row["epsilon"] == epsilon_grid[-1]:
            print(f"{row['intervention']} central difference h=0.01 MAE: {row['mean_absolute_derivative_error']:.8f}")
    for row in ot_rows:
        print(f"{row['intervention']} at t=100: empirical W1={row['empirical_w1']:.6f}, W2={row['empirical_w2']:.6f}")
    print(f"Evaluated {seeds} unseen random control seeds; frozen linear baseline; outputs are exploratory.")
    return {"derivative": validation_rows, "per_seed": per_seed_rows, "aggregate": aggregate_rows, "ot": ot_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12, help="independent evaluation control sequences")
    parser.add_argument("--bootstrap", type=int, default=500, help="bootstrap replicates across independent seeds")
    parser.add_argument("--epsilon", type=float, default=0.04, help="positive physical intervention amplitude")
    args = parser.parse_args()
    run(seeds=args.seeds, bootstrap=args.bootstrap, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
