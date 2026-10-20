"""Generate the controlled stationary Gaussian AR(1) baseline for sensitivity research.

Run from the repository root:
    python code/generate_perturbation_data.py

The first 200 observations are discarded as burn-in. The resulting dataset
has 5,000 observations and no deliberately introduced distribution shifts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "data" / "synthetic"
OUTPUT_FILE = OUTPUT_DIRECTORY / "perturbation_ar1.csv"

NUM_OBSERVATIONS = 5000
BURN_IN = 200
AR_COEFFICIENT = 0.7
INNOVATION_STD = 1.0
RANDOM_SEED = 42


def generate_ar1():
    """Generate x[t] = phi * x[t-1] + eta[t], eta ~ N(0, 1)."""
    rng = np.random.default_rng(RANDOM_SEED)
    innovations = rng.normal(
        loc=0.0,
        scale=INNOVATION_STD,
        size=NUM_OBSERVATIONS + BURN_IN,
    )
    values = np.zeros(NUM_OBSERVATIONS + BURN_IN, dtype=np.float64)
    for t in range(1, len(values)):
        values[t] = AR_COEFFICIENT * values[t - 1] + innovations[t]

    # Use float32 for model input, after computing the recurrence.
    return values[BURN_IN:].astype(np.float32)


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    values = generate_ar1()
    pd.DataFrame(
        {"time": np.arange(len(values)), "value": values}
    ).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(values)} observations to {OUTPUT_FILE}")
    print(f"Empirical mean: {values.mean():.4f}, std: {values.std():.4f}")


if __name__ == "__main__":
    main()
