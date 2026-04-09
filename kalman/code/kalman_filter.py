import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Synthetic Data 
# -----------------------------
np.random.seed(42)

n = 400
t = np.arange(n)

# First half
x1 = np.random.normal(10, 1, n // 2)

# Second half (distribution shift)
x2 = np.random.normal(20, 2, n // 2)

x = np.concatenate([x1, x2])

# Noise
noise = np.random.normal(0, 1, n)

# Final series
y = (x**2 + 3 + noise).astype(np.float32)

# -----------------------------
# 2. Kalman Filter Implementation
# -----------------------------

# Initialize arrays
x_hat = np.zeros(n)  # predictions
P = np.zeros(n)      # error covariance

# Initial guesses
x_hat[0] = y[0]
P[0] = 1.0

# Parameters (tune these if needed)
Q = 1e-3   # process noise
R = 1.0    # measurement noise

for k in range(1, n):
    # Prediction step
    x_pred = x_hat[k-1]
    P_pred = P[k-1] + Q

    # Update step
    K = P_pred / (P_pred + R)   # Kalman Gain
    x_hat[k] = x_pred + K * (y[k] - x_pred)
    P[k] = (1 - K) * P_pred

# -----------------------------
# 3. Plot Results
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(t, y, label="Actual Data", alpha=0.7)
plt.plot(t, x_hat, label="Kalman Prediction", linewidth=2)

# Mark shift
plt.axvline(x=n//2, color='red', linestyle='--', label="Distribution Shift")

plt.title("Kalman Filter on Distribution Shift Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Save plot
plt.savefig("tsf-distribution-shift/kalman/plots/kalman_result.png")

plt.show()
