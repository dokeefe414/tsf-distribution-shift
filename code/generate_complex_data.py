import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
 
# -----------------------------
# 1. Setup
# -----------------------------
np.random.seed(42)
 
n = 600
time = np.arange(n)
y = np.zeros(n)
 
# -----------------------------
# 2. Nonlinear Time-Varying AR Process (2 lags now)
#    - Uses tanh nonlinearity so linear models struggle
#    - AR coefficients shift dramatically per regime
# -----------------------------
for i in range(2, n):
 
    if i < 200:
        # Regime 1: moderate AR, low noise
        a1, a2 = 0.6, -0.2
        noise = np.random.normal(0, 5)
        y[i] = a1 * y[i-1] + a2 * y[i-2] + noise
 
    elif i < 400:
        # Regime 2: near-unit-root (very persistent), high noise
        # nonlinear term added — tanh creates saturation effect
        a1, a2 = 0.95, -0.3
        noise = np.random.normal(0, 20)
        y[i] = a1 * y[i-1] + a2 * y[i-2] + 10 * np.tanh(y[i-1] / 100) + noise
 
    else:
        # Regime 3: sign flip on a1, chaotic behavior
        a1, a2 = -0.4, 0.6
        noise = np.random.normal(0, 12)
        y[i] = a1 * y[i-1] + a2 * y[i-2] + noise
 
# -----------------------------
# 3. Time-Varying Seasonality
#    - Frequency CHANGES across regimes (not fixed 0.1)
#    - This breaks models that learn a fixed seasonal pattern
# -----------------------------
seasonality = np.zeros(n)
for i in range(n):
    if i < 200:
        seasonality[i] = 20 * np.sin(0.05 * i)        # slow frequency
    elif i < 400:
        seasonality[i] = 35 * np.sin(0.15 * i + 1.0)  # faster + phase shift
    else:
        seasonality[i] = 15 * np.sin(0.08 * i - 0.5)  # medium + different phase
 
y = y + seasonality
 
# -----------------------------
# 4. Heteroscedastic Noise
#    - Variance itself changes over time (not just mean)
#    - This is what really breaks ARIMA assumptions
# -----------------------------
het_noise = np.zeros(n)
for i in range(n):
    if i < 200:
        het_noise[i] = np.random.normal(0, 3)
    elif i < 400:
        # Noise variance ramps up linearly
        sigma = 3 + (i - 200) * 0.1
        het_noise[i] = np.random.normal(0, sigma)
    else:
        het_noise[i] = np.random.normal(0, 25)
 
y = y + het_noise
 
# -----------------------------
# 5. Mean Shifts (larger jumps than before)
# -----------------------------
y[:200]   += 100
y[200:400] += 350   # bigger jump
y[400:]   += 550
 
# -----------------------------
# 6. Accelerating Trend (nonlinear)
#    - Quadratic growth makes linear trend models fail late
# -----------------------------
trend = 0.0003 * (time ** 1.8)
y = y + trend
 
# -----------------------------
# 7. Hard Cases to Break Models
# -----------------------------
 
# Multiple sudden spikes (not just one)
y[150] += 250
y[250] += 400
y[430] += -300   # negative spike too
 
# Noise burst region (wider)
y[300:360] += np.random.normal(0, 60, 60)
 
# Abrupt variance collapse (models won't expect it)
y[370:400] += np.random.normal(0, 1, 30)  # suddenly very quiet
 
# Slow gradual drift that accelerates
y += 0.001 * (time ** 2) * 0.05
 
# -----------------------------
# 8. Convert to float32
# -----------------------------
y = y.astype(np.float32)
 
# -----------------------------
# 9. Save
# -----------------------------
os.makedirs("tsf-distribution-shift/data/synthetic", exist_ok=True)
 
df = pd.DataFrame({"time": time, "value": y})
df.to_csv("tsf-distribution-shift/data/synthetic/complex_time_series.csv", index=False)
 
print("=== Data Summary ===")
print(f"Total points : {n}")
print(f"Regime 1 (0-199)   | mean={y[:200].mean():.1f}, std={y[:200].std():.1f}")
print(f"Regime 2 (200-399) | mean={y[200:400].mean():.1f}, std={y[200:400].std():.1f}")
print(f"Regime 3 (400-599) | mean={y[400:].mean():.1f}, std={y[400:].std():.1f}")
print("Data saved to tsf-distribution-shift/data/synthetic/complex_time_series.csv")
 
# -----------------------------
# 10. Plot with annotations
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
 
# Top: full series
ax = axes[0]
ax.plot(time, y, label="Complex Time Series", color="steelblue", linewidth=1)
ax.axvline(x=200, color='red',    linestyle='--', linewidth=1.5, label="Shift 1 (t=200)")
ax.axvline(x=400, color='green',  linestyle='--', linewidth=1.5, label="Shift 2 (t=400)")
ax.axvspan(300, 360, alpha=0.1, color='orange', label="Noise Burst")
ax.axvspan(370, 400, alpha=0.1, color='purple', label="Variance Collapse")
ax.set_title("Challenging Synthetic Time Series")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True, alpha=0.3)
 
# Bottom: rolling std to show heteroscedasticity
window = 20
rolling_std = pd.Series(y).rolling(window).std()
ax2 = axes[1]
ax2.plot(time, rolling_std, color="darkorange", label=f"Rolling Std (window={window})")
ax2.axvline(x=200, color='red',   linestyle='--', linewidth=1.5)
ax2.axvline(x=400, color='green', linestyle='--', linewidth=1.5)
ax2.set_title("Rolling Standard Deviation (shows variance changes)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Std Dev")
ax2.legend()
ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
os.makedirs("tsf-distribution-shift/plots", exist_ok=True)
plt.savefig("tsf-distribution-shift/plots/complex_data.png", dpi=150)
plt.show()
