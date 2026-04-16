import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Setup
# -----------------------------
np.random.seed(42)

n = 600
t = np.arange(n)

# -----------------------------
# 2. Components
# -----------------------------

# Trend
trend = 0.05 * t

# Seasonality
seasonality = 20 * np.sin(0.1 * t)

# Noise
noise1 = np.random.normal(0, 5, 200)
noise2 = np.random.normal(0, 15, 200)
noise3 = np.random.normal(0, 10, 200)

# -----------------------------
# 3. Create Segments (distribution shifts)
# -----------------------------

# Segment 1: normal behavior
seg1 = trend[:200] + seasonality[:200] + noise1 + 100

# Segment 2: higher variance + amplified seasonality
seg2 = trend[200:400] + 2 * seasonality[200:400] + noise2 + 300

# Segment 3: multiplicative effect + stronger trend
seg3 = (trend[400:] * 2) * (1 + 0.01 * seasonality[400:]) + noise3 + 500

# Combine all
y = np.concatenate([seg1, seg2, seg3]).astype(np.float32)

# -----------------------------
# 4. Save Data
# -----------------------------
df = pd.DataFrame({
    "time": t,
    "value": y
})

# Make sure folder exists
os.makedirs("tsf-distribution-shift/data/synthetic", exist_ok=True)

df.to_csv("tsf-distribution-shift/data/synthetic/complex_time_series.csv", index=False)

print("Data saved to data/synthetic/complex_time_series.csv")

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(14,6))

plt.plot(t, y, label="Complex Time Series")

plt.axvline(x=200, color='red', linestyle='--', label="Shift 1")
plt.axvline(x=400, color='green', linestyle='--', label="Shift 2")

plt.title("Complex Synthetic Time Series with Multiple Distribution Shifts")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Save plot
os.makedirs("tsf-distribution-shift/plots", exist_ok=True)
plt.savefig("tsf-distribution-shift/plots/complex_data.png")

plt.show()