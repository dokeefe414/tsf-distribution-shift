import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib

matplotlib.use('TkAgg')

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("tsf-distribution-shift/data/synthetic/complex_time_series.csv")

y = df["value"].values
t = df["time"].values

n = len(y)

# -----------------------------
# 2. ARIMA (Rolling Forecast)
# -----------------------------
train_size = 200
history = list(y[:train_size])
arima_preds = list(y[:train_size])

for i in range(train_size, n):
    model = ARIMA(history, order=(2,1,2))
    model_fit = model.fit()

    yhat = model_fit.forecast()[0]
    arima_preds.append(yhat)

    history.append(y[i])

arima_preds = np.array(arima_preds)

# -----------------------------
# 3. Standard Kalman Filter
# -----------------------------
kf_preds = np.zeros(n)

x_est = y[0]
P = 1.0
Q = 1.0
R = 10.0

for i in range(n):
    # Prediction
    x_pred = x_est
    P_pred = P + Q

    # Update
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * (y[i] - x_pred)
    P = (1 - K) * P_pred

    kf_preds[i] = x_est

# -----------------------------
# 4. Adaptive Kalman Filter (Innovation-based)
# -----------------------------
akf_preds = np.zeros(n)

x_est = y[0]
P = 1.0
Q = 1.0
R = 10.0

alpha = 0.1  # adaptation rate

for i in range(n):
    # Prediction
    x_pred = x_est
    P_pred = P + Q

    # Innovation
    innovation = y[i] - x_pred

    # Adapt R based on error
    R = (1 - alpha) * R + alpha * (innovation**2)

    # Kalman gain
    K = P_pred / (P_pred + R)

    # Update
    x_est = x_pred + K * innovation
    P = (1 - K) * P_pred

    akf_preds[i] = x_est

# -----------------------------
# 5. Plot Everything
# -----------------------------
plt.figure(figsize=(14,6))

plt.plot(t, y, label="Actual Data")
plt.plot(t, arima_preds, label="ARIMA (Rolling)", linewidth=2)
plt.plot(t, kf_preds, label="Kalman Filter")
plt.plot(t, akf_preds, label="Adaptive Kalman", linestyle="--")

plt.axvline(x=200, color='red', linestyle='--', label="Shift 1")
plt.axvline(x=400, color='green', linestyle='--', label="Shift 2")

plt.title("Model Comparison Under Distribution Shift")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

plt.savefig("tsf-distribution-shift/plots/model_comparison.png")
plt.show()

# -----------------------------
# 6. MSE Comparison
# -----------------------------
def mse(a, b):
    return np.mean((a - b)**2)

print("\n=== MSE RESULTS ===")

print("ARIMA Before:", mse(y[:200], arima_preds[:200]))
print("ARIMA After:", mse(y[200:], arima_preds[200:]))

print("KF Before:", mse(y[:200], kf_preds[:200]))
print("KF After:", mse(y[200:], kf_preds[200:]))

print("Adaptive KF Before:", mse(y[:200], akf_preds[:200]))
print("Adaptive KF After:", mse(y[200:], akf_preds[200:]))

print("\nDone.")