import numpy as np
import matplotlib.pyplot as plt

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
plt.savefig("../plots/synthetic_shift.png")

# Show the plot
plt.show()
