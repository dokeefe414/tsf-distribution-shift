"""Deterministic covariance, Gaussian-sampling and AR(1) tests.

Run with no trained checkpoint:
    py code/test_correlated_gaussian.py
"""
import numpy as np

from correlated_gaussian_experiment import PROFILES, correlation_matrix, correlated_noise


def main():
    rng = np.random.default_rng(2026)
    length = 32
    for name in PROFILES:
        c = correlation_matrix(name, length)
        assert c.shape == (length, length)
        assert np.allclose(c, c.T)
        assert np.allclose(np.diag(c), 1.0)
        assert np.linalg.eigvalsh(c).min() > 0
        assert np.isclose(np.trace(c), length)
        assert np.isclose(correlated_noise(np.zeros((7, length)), c, 0).sum(), 0)
        assert correlated_noise(rng.normal(size=(3, length)), c, 0.10).shape == (3, length)

    positive = correlation_matrix("positive_temporal", length)
    negative = correlation_matrix("negative_temporal", length)
    common = correlation_matrix("common_mode", length)
    assert np.isclose(positive[0, 1], 0.8) and np.isclose(positive[0, 2], 0.64)
    assert np.isclose(negative[0, 1], -0.8) and np.isclose(negative[0, 2], 0.64)
    assert np.isclose(common[0, 10], 0.5)

    z = rng.standard_normal((18000, length))
    sigma = .1
    for name in PROFILES:
        c = correlation_matrix(name, length)
        noise = correlated_noise(z, c, sigma).astype(np.float64)
        empirical_cov = np.cov(noise, rowvar=False, bias=True)
        np.testing.assert_allclose(np.diag(empirical_cov), np.full(length, sigma ** 2), rtol=.06)
        assert np.max(np.abs(empirical_cov - sigma ** 2 * c)) < .0015, name

        # Fitted AR(1) uses only the last coordinate. Its expected absolute
        # response is unchanged when off-diagonal correlations change.
        slope = .716396
        observed = np.mean(np.abs(slope * noise[:, -1]))
        expected = abs(slope) * sigma * np.sqrt(2 / np.pi)
        assert abs(observed - expected) < .002, (name, observed, expected)

    for invalid in ["not_a_profile", "gaussian_mixture"]:
        try:
            correlation_matrix(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Unrecognized covariance profile accepted")
    try:
        correlated_noise(np.zeros((4, length)), np.eye(length), -.1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative sigma accepted")
    print("PASS: positive-definite unit-diagonal covariance, sampled correlation, equal noise variance, AR(1) invariance")


if __name__ == "__main__":
    main()
