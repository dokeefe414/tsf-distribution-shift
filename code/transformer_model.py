"""Train a small custom Transformer-style encoder for one-step time-series prediction.

All attention projections and the network are built using basic PyTorch layers;
no pretrained model or torch.nn.Transformer module is used.

Run in the repository:
    python code/generate_perturbation_data.py
    python code/transformer_model.py

The input at time t is [x[t-32], ..., x[t-1]] and target is x[t].
Splits use target indices, chronological order, and training-only scaling.
"""
from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "perturbation_ar1.csv"
ART = ROOT / "reports" / "transformer_experiment"

WINDOW = 32
SEED = 42
MODEL_DIMENSION = 32
ATTENTION_HEADS = 4
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5


class Windows(Dataset):
    """Supervised samples with WINDOW past values and one future target.

    A validation or test input may include *past* observations that precede
    that split. The target itself never overlaps across splits.
    """

    def __init__(self, series, target_indices, length=WINDOW):
        self.series = series
        self.indices = np.asarray(list(target_indices), dtype=np.int64)
        self.length = length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t = int(self.indices[index])
        history = self.series[t - self.length:t, None]
        target = self.series[t]
        return (
            torch.tensor(history, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


class Attention(nn.Module):
    """Implement multi-head scaled dot-product self-attention explicitly.

    Q, K, V are learned linear projections. Each head computes
    softmax(Q K^T / sqrt(d_head)) V. Heads are concatenated and projected.
    Attention is bidirectional *inside a strictly historical window*.
    """

    def __init__(self, d_model=MODEL_DIMENSION, heads=ATTENTION_HEADS):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.num_heads = heads
        self.head_dim = d_model // heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, steps, channels = x.shape

        def into_heads(tensor):
            return tensor.reshape(
                batch, steps, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = into_heads(self.query_projection(x))
        key = into_heads(self.key_projection(x))
        value = into_heads(self.value_projection(x))
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = attention_weights @ value
        concatenated = attended_values.transpose(1, 2).contiguous().reshape(
            batch, steps, channels
        )
        return self.output_projection(concatenated)


class Block(nn.Module):
    """One attention block with residual connections, normalization and an MLP."""

    def __init__(self, d_model=MODEL_DIMENSION):
        super().__init__()
        self.attention = Attention(d_model)
        self.normalization_1 = nn.LayerNorm(d_model)
        self.normalization_2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )

    def forward(self, x):
        x = self.normalization_1(x + self.attention(x))
        return self.normalization_2(x + self.feed_forward(x))


class Forecaster(nn.Module):
    """A historical-window encoder with a scalar one-step prediction head."""

    def __init__(self, length=WINDOW, d_model=MODEL_DIMENSION):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, length, d_model))
        self.block = Block(d_model)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length * d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        embedded = self.embedding(x) + self.position
        features = self.block(embedded)
        return self.head(features).squeeze(-1)


def load_split():
    """Return normalized train/validation/test windows and training statistics."""
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run code/generate_perturbation_data.py first."
        )

    series = pd.read_csv(DATA)["value"].to_numpy(dtype=np.float32)
    n = len(series)
    train_end, validation_end = int(0.70 * n), int(0.85 * n)
    mean = float(series[:train_end].mean())
    std = float(series[:train_end].std())
    if std <= 0:
        raise ValueError("Training-series standard deviation must be positive.")

    normalized = (series - mean) / std
    train = Windows(normalized, range(WINDOW, train_end))
    validation = Windows(normalized, range(train_end, validation_end))
    test = Windows(normalized, range(validation_end, n))
    metadata = {
        "mean": mean,
        "std": std,
        "n": n,
        "train_end": train_end,
        "val_end": validation_end,
    }
    return train, validation, test, metadata


def main():
    """Train with validation-based early stopping; save model and held-out results."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    ART.mkdir(parents=True, exist_ok=True)

    train, validation, test, stats = load_split()
    model = Forecaster()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.001
    )
    loss_function = nn.MSELoss()
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=128)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        training_losses = []
        for input_window, target in train_loader:
            optimizer.zero_grad()
            prediction = model(input_window)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for input_window, target in validation_loader:
                loss = loss_function(model(input_window), target)
                validation_losses.append(loss.item())

        validation_loss = float(np.mean(validation_losses))
        print(
            f"Epoch {epoch:02d}: "
            f"train MSE={np.mean(training_losses):.5f}, "
            f"validation MSE={validation_loss:.5f} (normalized)"
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ART / "transformer.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping: validation loss stopped improving.")
                break

    model.load_state_dict(
        torch.load(ART / "transformer.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    normalized_predictions, normalized_targets = [], []
    with torch.no_grad():
        for input_window, target in DataLoader(test, batch_size=128):
            normalized_predictions.extend(model(input_window).numpy())
            normalized_targets.extend(target.numpy())

    predicted = np.asarray(normalized_predictions) * stats["std"] + stats["mean"]
    actual = np.asarray(normalized_targets) * stats["std"] + stats["mean"]
    metrics = {
        "test_mse": float(np.mean((predicted - actual) ** 2)),
        "test_mae": float(np.mean(np.abs(predicted - actual))),
        "validation_mse_normalized": best_validation_loss,
    }
    configuration = {"window": WINDOW, "seed": SEED, **stats, **metrics}
    (ART / "config.json").write_text(json.dumps(configuration, indent=2))
    pd.DataFrame({
        "target_time": test.indices,
        "actual": actual,
        "forecast": predicted,
    }).to_csv(ART / "test_predictions.csv", index=False)
    print("Saved model, metadata, and test predictions to", ART)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
