# Transformer implementation: component-by-component walkthrough

This document accompanies [transformer_model.py](transformer_model.py). It explains the code so the architecture can be discussed and defended, rather than treated as a black box.

## 1. Forecasting task

At a target index `t`, we supply only the previous `L=32` observations:

`X_t = [x_(t-32), ..., x_(t-1)]`.

The model predicts one scalar `x_t`. Although attention can examine all 32 historical positions, **it cannot access x_t** during its forward pass.

Each training item has input tensor shape `(32, 1)` and a scalar target. A batch of 64 is `(64, 32, 1)`.

## 2. Data preparation: `Windows` and `load_split`

`Windows.__getitem__` slices from `t - 32` through `t - 1`. The split boundaries are applied to the **target indices**, so no target is shared across training, validation and testing.

From 5,000 samples, training targets are indexes 32–3499, validation 3500–4249 and test 4250–4999. At the start of validation, the first input includes the 32 observations immediately before that target. This is intentional: those observations exist at forecast time.

The scaler mean and standard deviation are calculated from only the first 3,500 training observations. The same values normalize validation, test, and perturbation inputs. The model never fits a scaler using held-out data.

## 3. Embedding and positional information: `Forecaster`

`nn.Linear(1, 32)` maps each scalar observation to a 32-dimensional representation. A learned positional tensor of shape `(1, 32, 32)` is added, giving the attention block a way to distinguish time positions. These positional parameters begin at zero and are optimized during training.

## 4. Multi-head attention: `Attention`

We implement query (Q), key (K), and value (V) projections using three separate `nn.Linear(32, 32)` layers. Four heads each process eight channels.

For each head:

`Attention(Q,K,V) = softmax(Q @ K.T / sqrt(8)) @ V`

The division by `sqrt(8)` helps keep attention scores from becoming excessively large. Softmax assigns weights over historical positions. Weighted values from all four heads are concatenated and projected back to 32 channels.

This is self-attention **inside one historical input window**. It does not use a causal attention mask because the entire window is already in the past relative to the target. A different formulation that predicted a value at every intermediate input position would need additional masking.

## 5. Transformer-style block: `Block`

The attention output is added back to its input (a residual connection), followed by layer normalization. Next, a two-layer feedforward network uses the GELU activation. A second residual connection and normalization finish the block.

There is one block to keep the first experiment transparent and computationally manageable.

## 6. Output: `Forecaster.head`

The 32 positions × 32 features are flattened to 1,024 features, passed through a hidden 64-unit layer and mapped to one number. That number is the normalized, one-step-ahead forecast.

This is a **custom Transformer-style encoder**, not a reproduction of a particular named time-series foundation model.

## 7. Training procedure

- Loss: mean squared error between normalized forecasts and normalized targets.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.001.
- Training batches: 64, shuffled within the training set.
- Validation: evaluated without gradient updates after each epoch.
- Early stopping: stop after five consecutive epochs without sufficient improvement; save the lowest-validation-loss model.
- Testing: performed only after selecting the model using validation loss.
- Saved outputs: checkpoint, model configuration and original-unit test forecasts.

The `test_mse` and `test_mae` are computed after undoing normalization, so they use the original series units. Validation MSE printed during training is in normalized units.

## 8. Why this implementation is suitable for perturbation response

The same weights and input windows are used in two passes:
- Clean: `F(X)`
- Perturbed: `F(X + epsilon)`

The difference isolates the change in the model output caused by a controlled change in input. Because no training occurs during perturbation evaluation, changes cannot be attributed to a newly fitted model.

## 9. Limits and next experiments

The model uses full-window attention rather than autoregressive generation of multiple future points. It has a small learned positional representation, no dropout, one encoder block and a flattening head. These are deliberate simplicity choices rather than proven optimal architecture choices.

The first dataset is a linear Gaussian AR(1) process, so a simple AR(1) regression is an important comparison baseline. The Transformer may have similar or worse forecasting error even if its perturbation-response pattern is interesting.

Next: run multiple training seeds; compare to a linear baseline; measure per-time-position sensitivity; use directional and correlated perturbations; investigate multi-step forecasts.
