"""Small deterministic tests for coordinate-wise Gaussian perturbations.

Run without a trained model: py code/test_directional_perturbation.py
"""
import numpy as np
import torch

from directional_perturbation import perturb_one_position
from linear_baseline_comparison import predict_ar1
from transformer_model import WINDOW


def main():
    clean = torch.arange(4 * WINDOW, dtype=torch.float32).reshape(4, WINDOW, 1)
    untouched = clean.clone()
    values = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

    oldest = perturb_one_position(clean, 0, values)
    np.testing.assert_allclose((oldest - clean)[:, 0, 0].numpy(), values, atol=1e-5)
    assert torch.equal(oldest[:, 1:, :], clean[:, 1:, :])
    np.testing.assert_allclose(
        predict_ar1(oldest.numpy(), 0.0, 0.7),
        predict_ar1(clean.numpy(), 0.0, 0.7),
    )

    newest = perturb_one_position(clean, WINDOW - 1, values)
    assert torch.equal(newest[:, :-1, :], clean[:, :-1, :])
    np.testing.assert_allclose(
        predict_ar1(newest.numpy(), 0.0, 0.7)
        - predict_ar1(clean.numpy(), 0.0, 0.7),
        0.7 * values,
        atol=1e-5,
    )
    assert torch.equal(clean, untouched), "The original windows were modified"

    for position in (-1, WINDOW):
        try:
            perturb_one_position(clean, position, values)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-bounds position must raise ValueError")
    print("PASS: coordinate perturbation, nonmutation, and AR(1) direction checks")


if __name__ == "__main__":
    main()
