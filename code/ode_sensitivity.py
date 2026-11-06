"""Tangent-linear ODE sensitivities for the controlled cubic generator.

For dx/dt = a(t)x - x^3 + gain*u(t), and scalar interventions
u -> u + eps*1[t>=onset], a -> a + delta*1[t>=onset],
the variations at zero intervention satisfy

  s_u' = (a - 3*x*x)*s_u + gain*1[t>=onset]
  s_a' = (a - 3*x*x)*s_a + x*1[t>=onset].

Both sensitivities start at zero if x(0) is held fixed. The original
simulation treats the random control and switching coefficient as
piecewise constant on each RK4 integration step; this routine uses exactly
that same discretization. These are pathwise derivatives conditional on a
fixed random control draw, not a derivative of a probability density.
"""

import numpy as np

from controlled_ode_framework import DynamicsConfig, simulate


def simulate_variations(config: DynamicsConfig, *, onset: float = 80.0) -> dict[str, np.ndarray]:
    """Simulate x, dx/d(control_offset), dx/d(parameter_offset) jointly via RK4."""
    original = simulate(config, onset=onset)
    times = original["time"]
    n = len(times) - 1
    dt = config.dt
    x = np.empty(n + 1, dtype=float)
    s_control = np.zeros(n + 1, dtype=float)
    s_parameter = np.zeros(n + 1, dtype=float)
    x[0] = config.x0

    for i in range(n):
        a = original["parameter"][i]
        u = original["control"][i]
        active = float(times[i] + 0.5 * dt >= onset)

        def rhs(y: np.ndarray) -> np.ndarray:
            value, du, da = y
            jacobian = a - 3.0 * value**2
            return np.array([
                a * value - value**3 + config.control_gain * u,
                jacobian * du + config.control_gain * active,
                jacobian * da + value * active,
            ], dtype=float)

        previous = np.array([x[i], s_control[i], s_parameter[i]])
        k1 = rhs(previous)
        k2 = rhs(previous + 0.5 * dt * k1)
        k3 = rhs(previous + 0.5 * dt * k2)
        k4 = rhs(previous + dt * k3)
        updated = previous + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x[i+1], s_control[i+1], s_parameter[i+1] = updated

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(s_control)) or not np.all(np.isfinite(s_parameter)):
        raise ArithmeticError("Sensitivity simulation diverged")
    return {
        "time": times,
        "state": x,
        "s_control": s_control,
        "s_parameter": s_parameter,
        "jacobian": original["parameter"] - 3.0*x[:-1]**2,
    }


def central_difference(config: DynamicsConfig, *, kind: str, epsilon: float, onset: float = 80.0) -> np.ndarray:
    """Centered derivative with the same random control realization in both runs."""
    if kind not in ("control", "parameter"):
        raise ValueError("kind must be 'control' or 'parameter'")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    args = {"control_offset" if kind == "control" else "parameter_offset": epsilon}
    plus = simulate(config, onset=onset, **args)["state"]
    args = {"control_offset" if kind == "control" else "parameter_offset": -epsilon}
    minus = simulate(config, onset=onset, **args)["state"]
    return (plus - minus) / (2.0 * epsilon)


def wasserstein_1d(first: np.ndarray, second: np.ndarray, order: int = 1) -> float:
    """Exact empirical W1 or W2 for equally weighted, equal-sized 1-D samples.

    For n observations in each measure, optimal transport matches the
    sorted order statistics. This discards paired trajectory identity;
    pairing is retained separately for physical response statistics.
    """
    if order not in (1, 2):
        raise ValueError("only orders 1 and 2 are supported")
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) == 0:
        raise ValueError("inputs must be nonempty 1-D arrays of equal length")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("inputs must be finite")
    diff = np.sort(x) - np.sort(y)
    return float(np.mean(np.abs(diff))) if order == 1 else float(np.sqrt(np.mean(diff**2)))


def paired_bootstrap_ci(values: np.ndarray, *, draws: int = 1000, seed: int = 17, confidence: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap of a per-trajectory statistic, resampling seeds.

    Treats independent control seeds as the replicate units, not time points.
    Does not account for model-fitting uncertainty or prove population coverage.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)):
        raise ValueError("at least two finite independent-seed statistics required")
    if not 0 < confidence < 1 or draws < 10:
        raise ValueError("invalid confidence level or bootstrap draw count")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(v), size=(draws, len(v)))
    means = v[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(v.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0-alpha))
