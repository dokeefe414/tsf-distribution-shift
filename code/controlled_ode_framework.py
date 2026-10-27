"""Controlled nonlinear ODE generator and sampler for paired perturbation studies.

We simulate dx/dt = a(t) * x - x**3 + gain * u(t), where a(t) changes
sign at a known time. The random control sequence is seeded and held fixed
between baseline/perturbed runs to isolate the effect of an intervention.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DynamicsConfig:
    duration: float = 120.0
    dt: float = 0.05
    phase_time: float = 60.0
    parameter_before: float = -0.8
    parameter_after: float = 0.6
    control_gain: float = 1.0
    control_std: float = 0.08
    control_hold: float = 1.0
    x0: float = 0.1
    seed: int = 2026


def _validated_steps(config: DynamicsConfig) -> int:
    if config.duration <= 0 or config.dt <= 0 or config.control_hold <= 0:
        raise ValueError("duration, dt, and control_hold must be positive")
    if config.control_std < 0:
        raise ValueError("control_std must not be negative")
    steps = round(config.duration / config.dt)
    if not np.isclose(steps * config.dt, config.duration, atol=1e-10):
        raise ValueError("duration must be an integer multiple of dt")
    return steps


def make_control(config: DynamicsConfig) -> np.ndarray:
    """Fixed-integration-step control, combining sinusoid and random holds."""
    n = _validated_steps(config)
    midpoint_times = (np.arange(n) + 0.5) * config.dt
    buckets = np.floor(midpoint_times / config.control_hold).astype(int)
    random_levels = np.random.default_rng(config.seed).normal(size=buckets.max() + 1)
    return 0.08 * np.sin(0.2 * midpoint_times) + config.control_std * random_levels[buckets]


def simulate(
    config: DynamicsConfig,
    *,
    control_offset: float = 0.0,
    parameter_offset: float = 0.0,
    onset: float = 80.0,
) -> dict[str, np.ndarray]:
    """Integrate with RK4; paired calls use the identical random control path.

    Offsets begin at onset, and are independent intervention types:
    control_offset changes u(t), whereas parameter_offset changes a(t).
    These perturb the generator, NOT post-hoc sampled observations.
    """
    n = _validated_steps(config)
    if onset < 0 or onset > config.duration:
        raise ValueError("onset must lie inside the simulated time interval")
    t = np.arange(n + 1, dtype=float) * config.dt
    control = make_control(config)
    midpoints = t[:-1] + 0.5 * config.dt
    control = control + control_offset * (midpoints >= onset)
    parameter = np.where(
        midpoints < config.phase_time,
        config.parameter_before,
        config.parameter_after,
    ) + parameter_offset * (midpoints >= onset)

    x = np.empty(n + 1, dtype=float)
    x[0] = config.x0
    for i in range(n):
        a, u = parameter[i], control[i]

        def derivative(value: float) -> float:
            return a * value - value**3 + config.control_gain * u

        k1 = derivative(x[i])
        k2 = derivative(x[i] + 0.5 * config.dt * k1)
        k3 = derivative(x[i] + 0.5 * config.dt * k2)
        k4 = derivative(x[i] + config.dt * k3)
        x[i + 1] = x[i] + (config.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    if not np.all(np.isfinite(x)):
        raise ArithmeticError("ODE simulation diverged; adjust step size or coefficients")
    return {"time": t, "state": x, "control": control, "parameter": parameter}


def sample_regular(run: dict[str, np.ndarray], interval: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Sample trajectory at uniform intervals aligned to the integration grid."""
    dt = run["time"][1] - run["time"][0]
    stride = round(interval / dt)
    if stride < 1 or not np.isclose(stride * dt, interval, atol=1e-10):
        raise ValueError("regular interval must be a positive integer multiple of dt")
    return run["time"][::stride].copy(), run["state"][::stride].copy()


def sample_irregular(
    run: dict[str, np.ndarray], *, min_gap: float = 0.2, max_gap: float = 0.65, seed: int = 99
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly spaced observations, returning timestamps alongside values.

    A predictor trained on these samples should explicitly account for gaps.
    """
    dt = run["time"][1] - run["time"][0]
    low, high = int(np.ceil(min_gap / dt - 1e-10)), int(np.floor(max_gap / dt + 1e-10))
    if low < 1 or high < low:
        raise ValueError("sampling gaps must define at least one positive grid step")
    rng = np.random.default_rng(seed)
    indices = [0]
    while indices[-1] < len(run["time"]) - 1:
        next_index = indices[-1] + int(rng.integers(low, high + 1))
        if next_index >= len(run["time"]):
            break
        indices.append(next_index)
    return run["time"][indices].copy(), run["state"][indices].copy()


def make_windows(values: np.ndarray, window: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Historical windows predict the NEXT sample, with no future leakage."""
    values = np.asarray(values, dtype=float)
    if window < 1 or len(values) <= window:
        raise ValueError("window must be positive and smaller than series length")
    features = np.lib.stride_tricks.sliding_window_view(values, window)[:-1]
    targets = values[window:]
    return features, targets


def fit_linear_forecaster(features: np.ndarray, targets: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Fit fixed one-step linear model, with intercept and ridge on lag weights."""
    if ridge < 0:
        raise ValueError("ridge penalty must be nonnegative")
    design = np.column_stack((np.ones(len(features)), features))
    penalty = np.diag([0.0] + [ridge] * features.shape[1])
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def predict_linear(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + features @ coefficients[1:]
