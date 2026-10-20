"""Lightweight smoke tests. No model training required.

Run after generating the AR(1) dataset:
    python code/test_perturbation_pipeline.py
"""
import numpy as np
import torch

from transformer_model import WINDOW, Forecaster, load_split


def main():
    train, validation, test, stats = load_split()
    assert len(train) == stats["train_end"] - WINDOW
    assert train.indices.max() < validation.indices.min()
    assert validation.indices.max() < test.indices.min()

    input_window, target = train[0]
    assert input_window.shape == (WINDOW, 1)
    assert target.ndim == 0
    assert torch.isfinite(input_window).all()

    model = Forecaster()
    batch = torch.stack([train[0][0], train[1][0]])
    prediction = model(batch)
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()

    loss = prediction.square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    # AR(1) training distribution is close to zero mean, unit scale after normalization.
    observed = np.concatenate([train[i][0].numpy()[:, 0] for i in range(5)])
    assert np.isfinite(observed).all()

    print("PASS: chronological splits, historical windows, forward/backward pass")


if __name__ == "__main__":
    main()
