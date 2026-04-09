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
# 2. Helper Function
# -----------------------------
def kalman_step(x_prev, P_prev, y_k, Q, R):
    x_pred = x_prev
    P_pred = P_prev + Q

    K = P_pred / (P_pred + R)
    x_new = x_pred + K * (y_k - x_pred)
    P_new = (1 - K) * P_pred

    return x_new, P_new

# -----------------------------
# 3. Initialize
# -----------------------------
x_std = np.zeros(n)
x_adapt = np.zeros(n)
x_window = np.zeros(n)
x_cp = np.zeros(n)

x_std[0] = x_adapt[0] = x_window[0] = x_cp[0] = y[0]

P_std = P_adapt = P_window = P_cp = 10.0

Q_std, R_std = 0.01, 5.0

Q_adapt, R_adapt = 0.01, 5.0
alpha = 0.05

Q_window, R_window = 0.01, 5.0
window_size = 20

Q_cp, R_cp = 0.01, 5.0
threshold = 200

# -----------------------------
# 4. Run Filters
# -----------------------------
for k in range(1, n):

    # ---- Standard KF ----
    x_std[k], P_std = kalman_step(x_std[k-1], P_std, y[k], Q_std, R_std)

    # ---- Adaptive KF ----
    innovation = y[k] - x_adapt[k-1]
    error_sq = np.clip(innovation**2, 0, 5000)

    R_adapt = (1 - alpha) * R_adapt + alpha * error_sq
    Q_adapt = (1 - alpha) * Q_adapt + alpha * (error_sq * 0.01)

    R_adapt = np.clip(R_adapt, 1, 1000)
    Q_adapt = np.clip(Q_adapt, 1e-4, 5)

    x_adapt[k], P_adapt = kalman_step(x_adapt[k-1], P_adapt, y[k], Q_adapt, R_adapt)

    # ---- Sliding Window KF ----
    if k > window_size:
        window = y[k-window_size:k]
        var = np.var(window)

        R_window = np.clip(var * 0.01, 0.5, 50)

        Q_window = np.clip(var * 0.02, 0.01, 10)

    x_window[k], P_window = kalman_step(x_window[k-1], P_window, y[k], Q_window, R_window)

    # ---- Change-Point KF ----
    innovation_cp = y[k] - x_cp[k-1]

    if abs(innovation_cp) > threshold:
        P_cp = 200  # strong reset

    x_cp[k], P_cp = kalman_step(x_cp[k-1], P_cp, y[k], Q_cp, R_cp)

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(14, 7))

plt.plot(t, y, label="Actual Data", alpha=0.5)

plt.plot(t, x_std, label="Standard KF", linewidth=2)
plt.plot(t, x_adapt, label="Adaptive KF", linewidth=2)
plt.plot(t, x_window, label="Sliding Window KF", linewidth=2)
plt.plot(t, x_cp, label="Change-Point KF", linewidth=2)

plt.axvline(x=n//2, color='red', linestyle='--', label="Distribution Shift")

plt.title("Kalman Filter Comparison")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

plt.savefig("tsf-distribution-shift/kalman/plots/combined_kalman.png")
plt.show()combined_kalman.png")
plt.show()
