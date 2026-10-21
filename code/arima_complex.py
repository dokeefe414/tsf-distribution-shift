import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'synthetic' / 'complex_time_series.csv'
PLOT = ROOT / 'plots' / 'arima_complex.png'
PLOT.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Load Complex Data
# -----------------------------
df = pd.read_csv(DATA)

y = df["value"].values
t = df["time"].values

# -----------------------------
# 2. Train/Test Split
# -----------------------------
train_size = 200  # train only on first regime

train = y[:train_size]
test = y[train_size:]

# -----------------------------
# 3. Fit ARIMA
# -----------------------------
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

# -----------------------------
# 4. Rolling Forecast (IMPORTANT)
# -----------------------------
predictions = np.full(len(y), np.nan, dtype=float)  # No training-regime predictions

history = list(train)

for i in range(len(test)):
    model = ARIMA(history, order=(2,1,2))
    model_fit = model.fit()

    yhat = model_fit.forecast()[0]
    predictions[train_size + i] = yhat

    # Add actual value to history (this is key)
    history.append(test[i])

# -----------------------------
# 5. Convert to array
# -----------------------------
# Training positions remain NaN: not genuine held-out forecasts

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(14,6))

plt.plot(t, y, label="Actual Data")
plt.plot(t, predictions, label="ARIMA Rolling Forecast", linewidth=2)

plt.axvline(x=200, color='red', linestyle='--', label="Shift 1")
plt.axvline(x=400, color='green', linestyle='--', label="Shift 2")

plt.title("ARIMA on Complex Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

plt.savefig(PLOT)
plt.show()

# -----------------------------
# 7. Evaluate
# -----------------------------
mse_after = np.mean((y[train_size:] - predictions[train_size:])**2)
print("Pre-shift MSE: N/A (no training-regime held-out forecasts)")
print("Post-shift rolling one-step MSE:", mse_after)
