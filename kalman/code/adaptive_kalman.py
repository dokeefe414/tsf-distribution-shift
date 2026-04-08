import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Synthetic Data
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
P[0] = 10.0   # slightly higher initial uncertainty

Q = 0.01      # higher than before → more flexibility
R = 5.0       # reasonable starting measurement noise

alpha = 0.05  # moderate adaptation speed

for k in range(1, n):

    # Prediction
    x_pred = x_hat[k-1]
    P_pred = P[k-1] + Q

    # Innovation (error)
    innovation = y[k] - x_pred

    # -----------------------------
    # Controlled Adaptive Update
    # -----------------------------
    error_sq = innovation**2

    # Smooth + cap extreme spikes
    error_sq = np.clip(error_sq, 0, 5000)

    # Update R slowly (measurement noise)
    R = (1 - alpha) * R + alpha * error_sq

    # Update Q moderately (process flexibility)
    Q = (1 - alpha) * Q + alpha * (error_sq * 0.01)

    # Keep Q and R in safe ranges
    R = np.clip(R, 1, 1000)
    Q = np.clip(Q, 1e-4, 5)

    # -----------------------------
    # Kalman Update
    # -----------------------------
    K = P_pred / (P_pred + R)

    x_hat[k] = x_pred + K * innovation
    P[k] = (1 - K) * P_pred

# -----------------------------
# 3. Plot Results
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(t, y, label="Actual Data", alpha=0.6)
plt.plot(t, x_hat, label="Adaptive Kalman (Fixed)", linewidth=2)

plt.axvline(x=n//2, color='red', linestyle='--', label="Distribution Shift")

plt.title("Improved Adaptive Kalman Filter")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

plt.savefig("tsf-distribution-shift/kalman/plots/adaptive_kalman_fixed.png")
plt.show()
