"""Numerical and probabilistic tests for variational ODE sensitivity."""
import unittest
import numpy as np

from controlled_ode_framework import DynamicsConfig, simulate
from ode_sensitivity import central_difference, paired_bootstrap_ci, simulate_variations, wasserstein_1d


class TestODEVariations(unittest.TestCase):
    def setUp(self):
        self.cfg = DynamicsConfig(duration=24.0, dt=0.05, phase_time=12.0, seed=71)
        self.onset = 16.0

    def test_base_replay_matches_existing_generator(self):
        expected = simulate(self.cfg, onset=self.onset)
        got = simulate_variations(self.cfg, onset=self.onset)
        np.testing.assert_allclose(got["state"], expected["state"], atol=1e-12, rtol=1e-12)

    def test_causal_sensitivities_zero_before_intervention(self):
        result = simulate_variations(self.cfg, onset=self.onset)
        before = result["time"] <= self.onset
        np.testing.assert_array_equal(result["s_control"][before], 0)
        np.testing.assert_array_equal(result["s_parameter"][before], 0)
        self.assertGreater(np.max(np.abs(result["s_control"])), 0.01)

    def test_analytic_tangent_matches_centered_differences(self):
        solution = simulate_variations(self.cfg, onset=self.onset)
        for kind in ("control", "parameter"):
            central = central_difference(self.cfg, kind=kind, epsilon=0.0005, onset=self.onset)
            self.assertLess(np.max(np.abs(central - solution["s_" + kind])), 1e-5)

    def test_second_order_finite_difference_refinement(self):
        solution = simulate_variations(self.cfg, onset=self.onset)["s_control"]
        coarse = np.mean(np.abs(central_difference(self.cfg, kind="control", epsilon=0.08, onset=self.onset) - solution))
        fine = np.mean(np.abs(central_difference(self.cfg, kind="control", epsilon=0.04, onset=self.onset) - solution))
        self.assertGreater(coarse, fine * 2.5)

    def test_perturbation_does_not_modify_preintervention_trajectory(self):
        original = simulate(self.cfg, onset=self.onset)["state"]
        altered = simulate(self.cfg, onset=self.onset, control_offset=0.04)["state"]
        index = int(round(self.onset / self.cfg.dt))
        np.testing.assert_array_equal(original[:index + 1], altered[:index + 1])

    def test_wasserstein_sorted_quantiles(self):
        self.assertAlmostEqual(wasserstein_1d([0, 2], [1, 3], order=1), 1.0)
        self.assertAlmostEqual(wasserstein_1d([2, 0], [3, 1], order=2), 1.0)
        self.assertAlmostEqual(wasserstein_1d([0, 1], [0, 1]), 0.0)
        with self.assertRaises(ValueError):
            wasserstein_1d([1, 2], [1])
        with self.assertRaises(ValueError):
            wasserstein_1d([1], [2], order=3)

    def test_seed_level_bootstrap_is_reproducible(self):
        one = paired_bootstrap_ci([1, 2, 4, 3], seed=7, draws=100)
        two = paired_bootstrap_ci([1, 2, 4, 3], seed=7, draws=100)
        self.assertEqual(one, two)
        self.assertLessEqual(one[1], one[0])
        self.assertGreaterEqual(one[2], one[0])

    def test_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            central_difference(self.cfg, kind="control", epsilon=0.0, onset=self.onset)
        with self.assertRaises(ValueError):
            central_difference(self.cfg, kind="invalid", epsilon=0.01, onset=self.onset)


if __name__ == "__main__":
    unittest.main()
