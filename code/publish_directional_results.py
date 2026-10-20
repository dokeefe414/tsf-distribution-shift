"""Stage exact directional results for a versioned GitHub research report.

Run from the project root after directional_perturbation.py has finished:
    py code/publish_directional_results.py --label 2026-10-20

Does not push to GitHub, retrain any models, or silently overwrite an existing
different run. Copies the local full-resolution PNG, CSVs, and model config
into reports/figures for a deliberate Git commit.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "reports" / "transformer_experiment"
PUBLISHED_DIR = ROOT / "reports" / "figures"


def digest(path):
    """Return a SHA-256 checksum for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_summary(summary, per_repeat):
    """Reject missing, inconsistent or mixed-experiment outputs."""
    required = {
        "model", "position_oldest_to_newest", "lag_before_target",
        "sigma_normalized", "mean_absolute_response"
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Directional summary missing columns: {sorted(missing)}")

    models = {"Fitted AR(1)", "Transformer"}
    if set(summary["model"]) != models or len(summary) != 64:
        raise ValueError("Expected two models with 32 positions each (64 rows)")
    sigmas = summary["sigma_normalized"].unique()
    if len(sigmas) != 1 or not (0 < sigmas[0] <= 10):
        raise ValueError("Expected one positive, finite perturbation sigma")
    for name in models:
        subset = summary[summary["model"] == name]
        if set(subset["position_oldest_to_newest"]) != set(range(1, 33)):
            raise ValueError(f"Missing or repeated positions for {name}")
        if any(int(row.position_oldest_to_newest) + int(row.lag_before_target) != 33
               for row in subset.itertuples()):
            raise ValueError("Historical position does not match lag convention")
        if subset["mean_absolute_response"].isna().any():
            raise ValueError("Mean absolute response is missing")

    per_repeat_cols = {
        "model", "position_oldest_to_newest", "repeat",
        "sigma_normalized", "mean_absolute_response"
    }
    if not per_repeat_cols.issubset(per_repeat.columns):
        raise ValueError("Per-repeat measurements have missing columns")
    counts = per_repeat.groupby(["model", "position_oldest_to_newest"]).size()
    if len(counts) != 64 or counts.min() != counts.max():
        raise ValueError("Inconsistent perturbation repetition counts")
    return float(sigmas[0]), int(counts.min())


def assemble_report(label, summary, sigma, repeats, source_files):
    """Produce a self-contained Markdown record from actual local CSVs."""
    lines = [
        f"# Directional sensitivity - published run {label}",
        "",
        "This report was built from locally generated experiment outputs by",
        "code/publish_directional_results.py. It is an archived run,",
        "not a new training experiment.",
        "",
        f"- Gaussian input noise standard deviation (normalized): **{sigma:g}**",
        "- Historical input length: **32** (position 1 oldest, 32 newest)",
        f"- Perturbation draws per input position: **{repeats}**",
        "- Models: **Fitted AR(1)** and **frozen Transformer**",
        "- Both models receive the same perturbed historical test inputs",
        "- Target values are unchanged and all responses use original signal units",
        "",
        f"![Directional sensitivity](directional_sensitivity_{label}.png)",
        "",
        f"[Exact summary CSV](directional_summary_{label}.csv) | "
        f"[Per-repeat data](directional_per_repeat_{label}.csv) | "
        f"[Model configuration](directional_model_config_{label}.json)",
        "",
        "## Five most responsive positions",
        ""
    ]
    for name in ("Fitted AR(1)", "Transformer"):
        lines.extend([
            f"### {name}",
            "",
            "| Position (oldest to newest) | Lag before target | Mean absolute response |",
            "| ---: | ---: | ---: |"
        ])
        top = summary[summary["model"] == name].nlargest(
            5, "mean_absolute_response"
        )
        for row in top.itertuples():
            lines.append(
                f"| {int(row.position_oldest_to_newest)} | "
                f"{int(row.lag_before_target)} | "
                f"{float(row.mean_absolute_response):.6f} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation and limitations",
        "",
        "AR(1) depends only on the newest observation, so its responses at",
        "positions 1-31 should be zero. The Transformer can be sensitive to",
        "older observations, but sensitivity alone does not establish",
        "their predictive usefulness.",
        "",
        "One trained model and one synthetic time series do not establish",
        "general behavior across random seeds or data distributions.",
        "Perturbing one coordinate also need not preserve the AR process's",
        "joint input distribution.",
        "",
        "## Source file SHA-256 checksums",
        ""
    ])
    for filename, path in source_files.items():
        lines.append(f"- {filename}: {digest(path)}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", required=True,
        help="Version label, e.g. 2026-10-20; letters, numbers and hyphens only"
    )
    args = parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in args.label):
        parser.error("--label may contain only letters, numbers, or hyphens")

    source_files = {
        "directional_sensitivity.png": SOURCE_DIR / "directional_sensitivity.png",
        "directional_summary.csv": SOURCE_DIR / "directional_summary.csv",
        "directional_per_repeat.csv": SOURCE_DIR / "directional_per_repeat.csv",
        "config.json": SOURCE_DIR / "config.json"
    }
    for src in source_files.values():
        if not src.is_file():
            raise FileNotFoundError(
                f"Required local output missing: {src}. "
                "Run directional_perturbation.py first."
            )

    summary = pd.read_csv(source_files["directional_summary.csv"])
    per_repeat = pd.read_csv(source_files["directional_per_repeat.csv"])
    sigma, repeats = validate_summary(summary, per_repeat)
    report = assemble_report(args.label, summary, sigma, repeats, source_files)

    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        SOURCE_DIR / "directional_sensitivity.png":
            PUBLISHED_DIR / f"directional_sensitivity_{args.label}.png",
        SOURCE_DIR / "directional_summary.csv":
            PUBLISHED_DIR / f"directional_summary_{args.label}.csv",
        SOURCE_DIR / "directional_per_repeat.csv":
            PUBLISHED_DIR / f"directional_per_repeat_{args.label}.csv",
        SOURCE_DIR / "config.json":
            PUBLISHED_DIR / f"directional_model_config_{args.label}.json"
    }
    report_path = PUBLISHED_DIR / f"directional_results_{args.label}.md"

    # Check every destination before copying anything. No silent overwrites.
    for src, dst in outputs.items():
        if dst.exists() and src.read_bytes() != dst.read_bytes():
            raise FileExistsError(f"Different run is already published at {dst}")
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report:
        raise FileExistsError(f"Different report is already published at {report_path}")

    for src, dst in outputs.items():
        if not dst.exists():
            shutil.copy2(src, dst)
    if not report_path.exists():
        report_path.write_text(report, encoding="utf-8")

    print(f"Staged exact graph, summary, raw responses, config and report in {PUBLISHED_DIR}")
    print("Review these files, then commit and push them using Git.")


if __name__ == "__main__":
    main()
