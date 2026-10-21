import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_OUTPUT = ROOT / 'data' / 'synthetic' / 'synthetic_shift_v1.csv'
PLOT_OUTPUT = ROOT / 'plots' / 'synthetic_shift.png'
DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Number of points
n = 400

# Create time index
t = np.arange(n)

# --- First segment (before shift) ---
mean1 = 10
std1 = 1
x1 = np.random.normal(mean1, std1, n // 2)

# --- Second segment (after shift) ---
mean2 = 20
std2 = 2
x2 = np.random.normal(mean2, std2, n // 2)

# Combine the two segments
x = np.concatenate([x1, x2])

# Apply transformation (like your advisor suggested)
# Y_t = X_t^2 + 3 + noise
noise = np.random.normal(0, 1, n)
y = x**2 + 3 + noise

# Create DataFrame
df = pd.DataFrame({
    "time": t,
    "value": y
})

# Save to CSV
df.to_csv(DATA_OUTPUT, index=False)
# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(t, y, label="Synthetic Time Series")

# Mark where the shift happens
plt.axvline(x=n//2, color='red', linestyle='--', label="Distribution Shift")

plt.title("Synthetic Time Series with Distribution Shift")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Save the plot
plt.savefig(PLOT_OUTPUT)

# Show the plot
plt.show()
