import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate SAME Synthetic Data
# -----------------------------
np.random.seed(42)

n = 400
t = np.arange(n)

x1 = np.random.normal(10, 1, n // 2)
x2 = np.random.normal(20, 2, n // 2)
x = np.concatenate([x1, x2])

noise = np.random.normal(0, 1, n)
y = (x**2 + 3 + noise).astype(np.float32)

# -----------------------------
# 2. Adaptive Kalman Filter
# -----------------------------

x_hat = np.zeros(n)
P = np.zeros(n)

x_hat[0] = y[0]
P[0] = 1.0

# Initial parameters
Q = 1e-3
R = 1.0

# Adaptive parameters
alpha = 0.01   # learning rate for adaptation

for k in range(1, n):
    # Prediction
    x_pred = x_hat[k-1]
    P_pred = P[k-1] + Q

    # Innovation (error)
    innovation = y[k] - x_pred

    #  Adapt R based on error magnitude
    R = (1 - alpha) * R + alpha * min(innovation**2, 100)

    # Update
    K = P_pred / (P_pred + R)
    x_hat[k] = x_pred + K * innovation
    P[k] = (1 - K) * P_pred

# -----------------------------
# 3. Plot Results
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(t, y, label="Actual Data", alpha=0.7)
plt.plot(t, x_hat, label="Adaptive Kalman", linewidth=2)

plt.axvline(x=n//2, color='red', linestyle='--', label="Distribution Shift")

plt.title("Adaptive Kalman Filter on Distribution Shift Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Save plot
plt.savefig("tsf-distribution-shift/kalman/plots/adaptive_kalman_result.png")

plt.show()
