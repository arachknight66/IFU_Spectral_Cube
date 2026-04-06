"""Tests for preprocessing: noise estimation, continuum, smoothing, DQ, defringe."""
from __future__ import annotations

import numpy as np
import pytest

from ifu_spectral_cube.preprocessing import (
    apply_dq_mask,
    apply_dq_mask_1d,
    apply_gaussian_smoothing,
    apply_savgol_smoothing,
    compare_smoothing_metrics,
    defringe_1d,
    estimate_continuum_1d,
    estimate_continuum_iterative,
    estimate_noise_sigma,
    subtract_continuum_1d,
)


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

class TestNoiseEstimation:
    def test_pure_gaussian_noise(self) -> None:
        rng = np.random.default_rng(10)
        noise = rng.normal(0.0, 1.0, size=5000)
        sigma = estimate_noise_sigma(noise)
        assert np.isfinite(sigma)
        assert 0.8 < sigma < 1.2

    def test_noise_with_emission_lines(self) -> None:
        """MAD-based estimator should be robust against emission spikes."""
        rng = np.random.default_rng(42)
        spectrum = rng.normal(0.0, 0.5, size=2000)
        # Add 5% outlier spikes (emission lines)
        spike_idx = rng.choice(2000, size=100, replace=False)
        spectrum[spike_idx] += rng.uniform(5, 20, size=100)
        sigma = estimate_noise_sigma(spectrum)
        assert 0.3 < sigma < 0.8  # Should still estimate ~0.5

    def test_short_spectrum(self) -> None:
        sigma = estimate_noise_sigma(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(sigma)  # < 4 finite points

    def test_all_nan(self) -> None:
        sigma = estimate_noise_sigma(np.full(100, np.nan))
        assert np.isnan(sigma)


# ---------------------------------------------------------------------------
# DQ masking
# ---------------------------------------------------------------------------

class TestDQMask:
    def test_basic_masking(self) -> None:
        data = np.ones((10, 3, 3))
        dq = np.zeros((10, 3, 3), dtype=int)
        dq[5, 1, 1] = 1  # Flag one voxel
        result = apply_dq_mask(data, dq, bad_dq_flags=1)
        assert np.isnan(result[5, 1, 1])
        assert result[0, 0, 0] == 1.0

    def test_bitfield_masking(self) -> None:
        data = np.ones((5, 2, 2))
        dq = np.zeros((5, 2, 2), dtype=int)
        dq[2, 0, 0] = 4  # bit 2 set
        # Only mask bit 0 → should NOT be masked
        result = apply_dq_mask(data, dq, bad_dq_flags=1)
        assert result[2, 0, 0] == 1.0
        # Mask bit 2 → SHOULD be masked
        result2 = apply_dq_mask(data, dq, bad_dq_flags=4)
        assert np.isnan(result2[2, 0, 0])

    def test_no_dq(self) -> None:
        data = np.ones((5, 2, 2))
        result = apply_dq_mask(data, None)
        np.testing.assert_array_equal(result, data)

    def test_1d_wrapper(self) -> None:
        spec = np.ones(10)
        dq = np.zeros(10, dtype=int)
        dq[3] = 1
        result = apply_dq_mask_1d(spec, dq)
        assert np.isnan(result[3])


# ---------------------------------------------------------------------------
# Continuum estimation
# ---------------------------------------------------------------------------

class TestContinuum:
    def test_flat_continuum(self) -> None:
        spec = np.ones(500)
        cont = estimate_continuum_1d(spec, window_length=51)
        np.testing.assert_allclose(cont, 1.0, atol=1e-10)

    def test_sloped_continuum(self) -> None:
        x = np.linspace(0, 10, 500)
        spec = 2.0 * x + 1.0
        cont = estimate_continuum_1d(spec, window_length=51)
        # Median filter should follow the slope reasonably well (within edge effects)
        np.testing.assert_allclose(cont[50:-50], spec[50:-50], atol=0.5)

    def test_iterative_rejects_emission(self) -> None:
        rng = np.random.default_rng(99)
        spec = np.ones(1000) + rng.normal(0, 0.01, 1000)
        # Add strong emission spike
        spec[500:510] += 10.0
        cont_iter = estimate_continuum_iterative(spec, window_length=51, sigma_clip=3.0)
        # The iterative continuum near the spike should be close to 1.0
        assert abs(cont_iter[505] - 1.0) < 1.0

    def test_subtract_continuum_basic(self) -> None:
        spec = np.ones(200)
        residual, cont = subtract_continuum_1d(spec, window_length=51)
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_savgol_preserves_flat(self) -> None:
        spec = np.ones(100)
        smoothed = apply_savgol_smoothing(spec, window_length=11, polyorder=3)
        np.testing.assert_allclose(smoothed, 1.0, atol=1e-10)

    def test_savgol_preserves_quadratic(self) -> None:
        """SG with polyorder=3 should preserve quadratic signal exactly."""
        x = np.linspace(-5, 5, 200)
        spec = 3.0 * x ** 2 + 2.0 * x + 1.0
        smoothed = apply_savgol_smoothing(spec, window_length=11, polyorder=3)
        np.testing.assert_allclose(smoothed[20:-20], spec[20:-20], atol=0.5)

    def test_gaussian_reduces_noise(self) -> None:
        rng = np.random.default_rng(77)
        spec = rng.normal(0, 1, 500)
        smoothed = apply_gaussian_smoothing(spec, sigma_channels=2.0)
        assert np.std(smoothed) < np.std(spec)

    def test_compare_metrics(self) -> None:
        rng = np.random.default_rng(11)
        orig = rng.normal(0, 1, 500)
        sg = apply_savgol_smoothing(orig)
        gauss = apply_gaussian_smoothing(orig)
        m = compare_smoothing_metrics(orig, sg, gauss)
        assert m["noise_reduction_savgol"] > 1.0
        assert m["noise_reduction_gaussian"] > 1.0


# ---------------------------------------------------------------------------
# Defringing
# ---------------------------------------------------------------------------

class TestDefringe:
    def test_removes_sinusoidal_fringe(self) -> None:
        wl = np.linspace(10, 20, 2000)
        fringe_period = 0.05  # micron
        fringe = 0.1 * np.sin(2 * np.pi * wl / fringe_period)
        signal = fringe.copy()
        defringed, info = defringe_1d(signal, wl, period_range_micron=(0.01, 0.1))
        assert info["n_components_removed"] >= 1
        assert np.std(defringed) < np.std(signal)

    def test_no_fringe_no_subtraction(self) -> None:
        wl = np.linspace(10, 20, 500)
        rng = np.random.default_rng(33)
        signal = rng.normal(0, 0.01, 500)
        # Use high significance threshold so noise doesn't trigger false fringes
        defringed, info = defringe_1d(signal, wl, significance_power_ratio=15.0)
        # Should not find significant fringes in pure noise
        assert info["n_components_removed"] <= 1
