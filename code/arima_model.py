import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib

# Fix for VS Code plot issues
matplotlib.use('TkAgg')

# -----------------------------
# 1. Load your dataset
# -----------------------------
print("Loading data...")

df = pd.read_csv("tsf-distribution-shift/data/synthetic/synthetic_shift_v1.csv")

y = df["value"].values
t = df["time"].values

print("Data loaded successfully")

# -----------------------------
# 2. Train/Test Split
# -----------------------------
train_size = len(y) // 2

train = y[:train_size]
test = y[train_size:]

print("Training ARIMA model...")

# -----------------------------
# 3. Fit ARIMA Model
# -----------------------------
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

print("Model trained")

# -----------------------------
# 4. Forecast
# -----------------------------
forecast = model_fit.forecast(steps=len(test))

# Combine train + forecast for full plot
full_pred = np.concatenate([train, forecast])

print("Forecast complete")

# -----------------------------
# 5. Plot Results
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(t, y, label="Actual Data")
plt.plot(t, full_pred, label="ARIMA Forecast", linewidth=2)

plt.axvline(x=train_size, linestyle='--', label="Distribution Shift")

plt.title("ARIMA Forecast Under Distribution Shift")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Save plot
plt.savefig("tsf-distribution-shift/plots/arima_result.png")

print("Plot saved to tsf-distribution-shift/plots/arima_result.png")

# Show plot
plt.show()

# -----------------------------
# 6. Evaluate Error
# -----------------------------
mse_before = np.mean((y[:train_size] - full_pred[:train_size])**2)
mse_after = np.mean((y[train_size:] - full_pred[train_size:])**2)

print("\nRESULTS:")
print("MSE Before Shift:", mse_before)
print("MSE After Shift:", mse_after)

print("\nScript finished successfully")
