"""Fast self-contained checks: python -m unittest discover -s code -p 'test_controlled_ode.py'"""

import unittest
import numpy as np

from controlled_ode_framework import (
    DynamicsConfig, fit_linear_forecaster, make_windows, predict_linear,
    sample_irregular, sample_regular, simulate,
)


class TestControlledODE(unittest.TestCase):
    def setUp(self):
        self.cfg = DynamicsConfig(duration=20.0, dt=0.05, phase_time=10.0, seed=17)

    def test_deterministic_and_unperturbed(self):
        first = simulate(self.cfg, onset=12)
        second = simulate(self.cfg, onset=12)
        zero = simulate(self.cfg, onset=12, control_offset=0, parameter_offset=0)
        np.testing.assert_array_equal(first["state"], second["state"])
        np.testing.assert_array_equal(first["state"], zero["state"])

    def test_intervention_cannot_rewrite_the_past(self):
        base = simulate(self.cfg, onset=12)
        modified = simulate(self.cfg, onset=12, control_offset=0.2)
        untouched = base["time"] <= 12
        np.testing.assert_allclose(base["state"][untouched], modified["state"][untouched], atol=0, rtol=0)
        self.assertGreater(np.max(np.abs(modified["state"]-base["state"])), 1e-3)
        changed_a = simulate(self.cfg, onset=12, parameter_offset=0.2)
        np.testing.assert_allclose(base["state"][untouched], changed_a["state"][untouched], atol=0, rtol=0)

    def test_sampled_time_alignment(self):
        run = simulate(self.cfg, onset=12)
        t, x = sample_regular(run, interval=0.25)
        self.assertEqual(len(t), len(x))
        np.testing.assert_allclose(np.diff(t), 0.25)
        ti, xi = sample_irregular(run, seed=9)
        self.assertEqual(len(ti), len(xi))
        self.assertTrue(np.all(np.diff(ti) > 0))
        self.assertGreater(np.std(np.diff(ti)), 0)

    def test_one_step_windows_and_frozen_forecast(self):
        series = np.arange(30.0)
        features, targets = make_windows(series, window=4)
        np.testing.assert_array_equal(features[0], [0, 1, 2, 3])
        self.assertEqual(targets[0], 4)
        coef = fit_linear_forecaster(features[:15], targets[:15])
        self.assertEqual(len(predict_linear(features, coef)), len(targets))
        self.assertLess(np.mean(np.abs(predict_linear(features[:15], coef)-targets[:15])), 0.01)

    def test_invalid_grid_or_sampling(self):
        with self.assertRaises(ValueError):
            simulate(DynamicsConfig(duration=3.0, dt=0.7))
        with self.assertRaises(ValueError):
            sample_regular(simulate(self.cfg, onset=12), interval=0.23)


if __name__ == "__main__":
    unittest.main()
