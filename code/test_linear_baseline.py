"""Fast deterministic tests of the AR(1) linear comparison. No training needed.

    py code/test_linear_baseline.py
"""
import numpy as np

from linear_baseline_comparison import fit_ar1, predict_ar1, score
from transformer_model import WINDOW


class SyntheticTraining:
    """Simple object with attributes matching the training-window dataset."""

    def __init__(self, series):
        self.series = np.asarray(series, dtype=np.float32)
        self.indices = np.arange(1, len(series))


def main():
    # Known recurrence with intercept and slope: linear fit should recover both.
    values = np.zeros(101, dtype=np.float32)
    values[0] = -1.25
    for t in range(1, len(values)):
        values[t] = 0.25 + 0.7 * values[t - 1]
    intercept, slope = fit_ar1(SyntheticTraining(values))
    assert np.isclose(intercept, 0.25, atol=1e-6)
    assert np.isclose(slope, 0.7, atol=1e-6)

    windows = np.zeros((2, WINDOW, 1), dtype=np.float32)
    windows[0, -1, 0] = 3.0
    windows[1, -1, 0] = -2.0
    clean = predict_ar1(windows, intercept, slope)
    perturbed = windows.copy()
    perturbed[:, 0, 0] += 10.0  # Oldest input is irrelevant to AR(1).
    np.testing.assert_allclose(predict_ar1(perturbed, intercept, slope), clean)
    perturbed[:, -1, 0] += 0.15
    np.testing.assert_allclose(
        predict_ar1(perturbed, intercept, slope) - clean,
        np.full(2, 0.15 * slope), rtol=1e-5,
    )

    errors = score(np.array([0.0, 2.0]), np.array([1.0, 1.0]), 2.0)
    assert np.isclose(errors["mse"], 1.0)
    assert np.isclose(errors["nmse_train_variance"], 0.25)
    print("PASS: AR(1) fitting, directional response, and metrics")


if __name__ == "__main__":
    main()
