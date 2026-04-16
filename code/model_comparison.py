import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("tsf-distribution-shift/data/synthetic/complex_time_series.csv")

y = df["value"].values
t = df["time"].values
n = len(y)

# -----------------------------
# 2. ARIMA (Rolling Forecast)
#    - Only stores ACTUAL forecasts (no training data leakage)
#    - Trains on fixed window of last 100 points to avoid
#      growing history masking shift impact
# -----------------------------
train_size = 200
WINDOW = 100  # sliding window size

arima_preds = []  # only forecasted values (t=200 onward)
history = list(y[:train_size])

print("Running ARIMA rolling forecast...")
for i in range(train_size, n):
    # Use only the last WINDOW points to train
    window = history[-WINDOW:]

    model = ARIMA(window, order=(2, 1, 2))
    model_fit = model.fit()

    yhat = model_fit.forecast()[0]
    arima_preds.append(yhat)

    history.append(y[i])  # add actual value to history

arima_preds = np.array(arima_preds)  # shape: (n - train_size,)

# -----------------------------
# 3. Standard Kalman Filter
#    - One-step-ahead prediction (store x_pred BEFORE update)
# -----------------------------
kf_preds = np.zeros(n)

x_est = y[0]
P = 1.0
Q = 0.1   # lower = smoother
R = 10.0

for i in range(n):
    # Predict
    x_pred = x_est
    P_pred = P + Q

    # Store one-step-ahead prediction BEFORE seeing y[i]
    kf_preds[i] = x_pred

    # Update with actual observation
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * (y[i] - x_pred)
    P = (1 - K) * P_pred

# -----------------------------
# 4. Adaptive Kalman Filter
#    - Adapts R with clamping to prevent divergence
#    - One-step-ahead prediction (store x_pred BEFORE update)
# -----------------------------
akf_preds = np.zeros(n)

x_est = y[0]
P = 1.0
Q = 0.1
R = 10.0
alpha = 0.05   # slower adaptation rate

for i in range(n):
    # Predict
    x_pred = x_est
    P_pred = P + Q

    # Store one-step-ahead prediction BEFORE seeing y[i]
    akf_preds[i] = x_pred

    # Innovation
    innovation = y[i] - x_pred

    # Adapt R — clamped so it can't explode
    R = (1 - alpha) * R + alpha * (innovation ** 2)
    R = np.clip(R, 0.1, 100.0)

    # Kalman gain and update
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * innovation
    P = (1 - K) * P_pred

# -----------------------------
# 5. MSE Calculations
#    - ARIMA: only valid for t=200 onward
#    - KF / AKF: valid for all t, split into 3 windows
# -----------------------------
def mse(a, b):
    return np.mean((a - b) ** 2)

# ARIMA windows (forecasts start at t=200)
mse_arima_pre      = None  # no valid pre-shift forecast for ARIMA
mse_arima_shift1   = mse(y[200:400], arima_preds[:200])   # between shift 1 and 2
mse_arima_shift2   = mse(y[400:],    arima_preds[200:])   # after shift 2

# KF windows
mse_kf_pre    = mse(y[:200],    kf_preds[:200])
mse_kf_shift1 = mse(y[200:400], kf_preds[200:400])
mse_kf_shift2 = mse(y[400:],    kf_preds[400:])

# Adaptive KF windows
mse_akf_pre    = mse(y[:200],    akf_preds[:200])
mse_akf_shift1 = mse(y[200:400], akf_preds[200:400])
mse_akf_shift2 = mse(y[400:],    akf_preds[400:])

# -----------------------------
# 6. Save MSE Results to CSV
# -----------------------------
os.makedirs("tsf-distribution-shift/data/synthetic", exist_ok=True)

results_df = pd.DataFrame({
    "Model":          ["ARIMA (Windowed)", "Kalman Filter", "Adaptive Kalman"],
    "MSE_Pre_Shift":  ["N/A",              mse_kf_pre,      mse_akf_pre],
    "MSE_Shift1_to_2":[mse_arima_shift1,   mse_kf_shift1,   mse_akf_shift1],
    "MSE_After_Shift2":[mse_arima_shift2,  mse_kf_shift2,   mse_akf_shift2],
})

results_df.to_csv("tsf-distribution-shift/data/synthetic/mse_results.csv", index=False)

print("\n=== MSE RESULTS ===")
print(results_df.to_string(index=False))

# -----------------------------
# 7. Plot
# -----------------------------
os.makedirs("tsf-distribution-shift/plots", exist_ok=True)

# Build full-length ARIMA array for plotting (pad pre-shift with NaN)
arima_plot = np.full(n, np.nan)
arima_plot[train_size:] = arima_preds

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# --- Top plot: all models ---
ax = axes[0]
ax.plot(t, y,          label="Actual Data",      color="blue",   linewidth=1.5)
ax.plot(t, arima_plot, label="ARIMA (Windowed)", color="orange", linewidth=1.5)
ax.plot(t, kf_preds,   label="Kalman Filter",    color="green",  linewidth=1.5)
ax.plot(t, akf_preds,  label="Adaptive Kalman",  color="red",    linewidth=1.5, linestyle="--")
ax.axvline(x=200, color="red",   linestyle=":", linewidth=1.5, label="Shift 1 (t=200)")
ax.axvline(x=400, color="green", linestyle=":", linewidth=1.5, label="Shift 2 (t=400)")
ax.set_title("Model Comparison Under Distribution Shift")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Bottom plot: residuals (error) for post-shift only ---
ax2 = axes[1]
ax2.plot(t[train_size:], y[train_size:] - arima_preds,          label="ARIMA Error",          color="orange")
ax2.plot(t[train_size:], y[train_size:] - kf_preds[train_size:],label="Kalman Error",          color="green")
ax2.plot(t[train_size:], y[train_size:] - akf_preds[train_size:],label="Adaptive Kalman Error",color="red", linestyle="--")
ax2.axvline(x=200, color="red",   linestyle=":", linewidth=1.5)
ax2.axvline(x=400, color="green", linestyle=":", linewidth=1.5)
ax2.axhline(y=0,   color="black", linestyle="-", linewidth=0.8)
ax2.set_title("Forecast Error (Post-Shift Only)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Residual")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tsf-distribution-shift/plots/model_comparison.png", dpi=150)
plt.show()

print("\nPlot saved. Done.")