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
# 2. Metrics
# -----------------------------
def mse(true, pred):
    return np.mean((true - pred) ** 2)
 
def nmse(true, pred):
    """Normalized MSE: MSE divided by variance of true signal.
       NMSE < 1 means model beats a naive mean predictor.
       NMSE = 1 means model is as bad as predicting the mean.
       NMSE > 1 means model is worse than predicting the mean."""
    return np.mean((true - pred) ** 2) / np.var(true)
 
def mae(true, pred):
    """Mean Absolute Error: more interpretable than MSE,
       same units as the data, less sensitive to outliers."""
    return np.mean(np.abs(true - pred))
 
def rmse(true, pred):
    """Root MSE: same units as data, easier to interpret than MSE."""
    return np.sqrt(np.mean((true - pred) ** 2))
 
# -----------------------------
# 3. ARIMA Rolling Forecast
# -----------------------------
train_size = 200
WINDOW = 100
 
arima_preds = []
history = list(y[:train_size])
 
# Track ARIMA details across fits
arima_log = []
 
print("=" * 60)
print("RUNNING ARIMA ROLLING FORECAST")
print("=" * 60)
 
for i in range(train_size, n):
    window = history[-WINDOW:]
 
    model = ARIMA(window, order=(2, 1, 2))
    model_fit = model.fit()
 
    yhat = model_fit.forecast()[0]
    arima_preds.append(yhat)
    history.append(y[i])
 
    # Print model details at shift points and every 100 steps
    step = i - train_size
    if step == 0 or i == 200 or i == 400 or step % 100 == 0:
        params = model_fit.params
        print(f"\n--- ARIMA fit at t={i} (forecast step {step}) ---")
        print(f"  AIC        : {model_fit.aic:.2f}")
        print(f"  BIC        : {model_fit.bic:.2f}")
        print(f"  AR params  : {params[:2].round(4).tolist()}")
        print(f"  MA params  : {params[2:4].round(4).tolist()}")
        print(f"  Forecast   : {yhat:.4f}  |  Actual: {y[i]:.4f}  |  Error: {abs(y[i]-yhat):.4f}")
 
        arima_log.append({
            "t": i,
            "step": step,
            "AIC": model_fit.aic,
            "BIC": model_fit.bic,
            "forecast": yhat,
            "actual": y[i],
            "abs_error": abs(y[i] - yhat)
        })
 
arima_preds = np.array(arima_preds)
 
# Save ARIMA log
os.makedirs("tsf-distribution-shift/data/synthetic", exist_ok=True)
pd.DataFrame(arima_log).to_csv(
    "tsf-distribution-shift/data/synthetic/arima_fit_log.csv", index=False
)
print("\nARIMA fit log saved.")
 
# -----------------------------
# 4. Standard Kalman Filter
# -----------------------------
print("\n" + "=" * 60)
print("RUNNING STANDARD KALMAN FILTER")
print("=" * 60)
 
kf_preds = np.zeros(n)
x_est = y[0]
P = 1.0
Q = 0.1
R = 10.0
 
kf_gains = []
 
for i in range(n):
    x_pred = x_est
    P_pred = P + Q
 
    kf_preds[i] = x_pred  # store BEFORE update
 
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * (y[i] - x_pred)
    P = (1 - K) * P_pred
 
    kf_gains.append(K)
 
    if i in [0, 100, 200, 300, 400, 500, 599]:
        print(f"  t={i:3d} | x_pred={x_pred:.4f} | K={K:.4f} | P={P:.4f} | R={R:.4f} | Q={Q:.4f}")
 
print(f"\nKalman Summary:")
print(f"  Fixed Q      : {Q}")
print(f"  Fixed R      : {R}")
print(f"  Avg K (gain) : {np.mean(kf_gains):.4f}")
print(f"  Min K        : {np.min(kf_gains):.4f}")
print(f"  Max K        : {np.max(kf_gains):.4f}")
 
# -----------------------------
# 5. Adaptive Kalman Filter
# -----------------------------
print("\n" + "=" * 60)
print("RUNNING ADAPTIVE KALMAN FILTER")
print("=" * 60)
 
akf_preds = np.zeros(n)
x_est = y[0]
P = 1.0
Q = 0.1
R = 10.0
alpha = 0.1
 
akf_gains = []
R_history = []
 
for i in range(n):
    x_pred = x_est
    P_pred = P + Q
 
    akf_preds[i] = x_pred  # store BEFORE update
 
    innovation = y[i] - x_pred
 
    R = (1 - alpha) * R + alpha * (innovation ** 2)
    R = np.clip(R, 1, 50)
 
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * innovation
    P = (1 - K) * P_pred
 
    akf_gains.append(K)
    R_history.append(R)
 
    if i in [0, 100, 200, 300, 400, 500, 599]:
        print(f"  t={i:3d} | x_pred={x_pred:.4f} | K={K:.4f} | P={P:.4f} | R={R:.4f} | innovation={innovation:.4f}")
 
print(f"\nAdaptive Kalman Summary:")
print(f"  Alpha (adapt rate) : {alpha}")
print(f"  Initial R          : 10.0")
print(f"  Final R            : {R_history[-1]:.4f}")
print(f"  Min R              : {min(R_history):.4f}")
print(f"  Max R              : {max(R_history):.4f}")
print(f"  Avg K (gain)       : {np.mean(akf_gains):.4f}")
 
# -----------------------------
# 6. Metrics — Per Window
# -----------------------------
print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)
 
# ARIMA only valid post train_size
arima_post = arima_preds        # t=200 to 600
y_post     = y[train_size:]
 
arima_s1   = arima_preds[:200]  # t=200-400
arima_s2   = arima_preds[200:]  # t=400-600
y_s1       = y[200:400]
y_s2       = y[400:]
 
windows = {
    "Pre-Shift (t=0-199)":        (y[:200],  None,       kf_preds[:200],    akf_preds[:200]),
    "Post-Shift1 (t=200-399)":    (y_s1,     arima_s1,   kf_preds[200:400], akf_preds[200:400]),
    "Post-Shift2 (t=400-599)":    (y_s2,     arima_s2,   kf_preds[400:],    akf_preds[400:]),
    "Full Post-Shift (t=200-599)":(y_post,   arima_post, kf_preds[200:],    akf_preds[200:]),
}
 
rows = []
for window_name, (true, arima_w, kf_w, akf_w) in windows.items():
    print(f"\n  {window_name}")
    print(f"  {'Metric':<10} {'ARIMA':>12} {'KF':>12} {'Adaptive KF':>14}")
    print(f"  {'-'*50}")
 
    for metric_name, fn in [("MSE", mse), ("RMSE", rmse), ("MAE", mae), ("NMSE", nmse)]:
        a  = f"{fn(true, arima_w):.4f}" if arima_w is not None else "N/A"
        k  = f"{fn(true, kf_w):.4f}"
        ak = f"{fn(true, akf_w):.4f}"
        print(f"  {metric_name:<10} {a:>12} {k:>12} {ak:>14}")
 
        rows.append({
            "Window": window_name,
            "Metric": metric_name,
            "ARIMA": a,
            "KF": k,
            "Adaptive_KF": ak
        })
 
# Save full results
results_df = pd.DataFrame(rows)
results_df.to_csv("tsf-distribution-shift/data/synthetic/full_metrics.csv", index=False)
print("\nFull metrics saved to tsf-distribution-shift/data/synthetic/full_metrics.csv")
 
# -----------------------------
# 7. Plot
# -----------------------------
os.makedirs("tsf-distribution-shift/plots", exist_ok=True)
 
arima_plot = np.full(n, np.nan)
arima_plot[train_size:] = arima_preds
 
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
 
# --- Top: predictions ---
ax = axes[0]
ax.plot(t, y,          label="Actual Data",      color="blue",   linewidth=1.2)
ax.plot(t, arima_plot, label="ARIMA (Windowed)", color="orange", linewidth=1.5)
ax.plot(t, kf_preds,   label="Kalman Filter",    color="green",  linewidth=1.5)
ax.plot(t, akf_preds,  label="Adaptive Kalman",  color="red",    linewidth=1.5, linestyle="--")
ax.axvline(x=200, color="red",   linestyle=":", linewidth=1.5, label="Shift 1 (t=200)")
ax.axvline(x=400, color="green", linestyle=":", linewidth=1.5, label="Shift 2 (t=400)")
ax.set_title("Model Comparison Under Distribution Shift")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True, alpha=0.3)
 
# --- Middle: residuals ---
ax2 = axes[1]
ax2.plot(t[train_size:], y[train_size:] - arima_preds,            label="ARIMA Error",          color="orange")
ax2.plot(t[train_size:], y[train_size:] - kf_preds[train_size:],  label="Kalman Error",          color="green")
ax2.plot(t[train_size:], y[train_size:] - akf_preds[train_size:], label="Adaptive Kalman Error", color="red", linestyle="--")
ax2.axvline(x=200, color="red",   linestyle=":", linewidth=1.5)
ax2.axvline(x=400, color="green", linestyle=":", linewidth=1.5)
ax2.axhline(y=0,   color="black", linestyle="-", linewidth=0.8)
ax2.set_title("Forecast Residuals (Post-Shift Only)")
ax2.set_ylabel("Error")
ax2.legend()
ax2.grid(True, alpha=0.3)
 
# --- Bottom: adaptive R over time ---
ax3 = axes[2]
ax3.plot(t, R_history, color="purple", label="Adaptive KF — R (measurement noise)")
ax3.axvline(x=200, color="red",   linestyle=":", linewidth=1.5, label="Shift 1")
ax3.axvline(x=400, color="green", linestyle=":", linewidth=1.5, label="Shift 2")
ax3.set_title("Adaptive Kalman: How Measurement Noise R Changes Over Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("R value")
ax3.legend()
ax3.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("tsf-distribution-shift/plots/model_comparison.png", dpi=150)
plt.show()
 
print("\nPlot saved. Done.")
