"""Tests for Gaussian/Voigt fitting, multiplet deblending, and redshift estimation."""
from __future__ import annotations

import numpy as np
import pytest

from ifu_spectral_cube.fitting import (
    _compute_diagnostics,
    estimate_redshift_from_matches,
    fit_all_peaks,
    fit_multiplet,
    fit_peak_gaussian,
    fit_peak_voigt,
)
from ifu_spectral_cube.models import GaussianFit, LineMatch, Peak


def _inject_gaussian(wl: np.ndarray, mu: float, amp: float, sig: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((wl - mu) / sig) ** 2)


class TestGaussianFit:
    def test_recovers_parameters(self) -> None:
        wl = np.linspace(12.5, 13.1, 200)
        flux = _inject_gaussian(wl, 12.8, 2.0, 0.015)
        peak = Peak(index=100, wavelength_micron=12.8, flux=2.0, prominence=2.0, width_samples=5, snr=10.0)
        gf = fit_peak_gaussian(wl, flux, peak, window_half_width_micron=0.2)
        assert gf is not None
        assert abs(gf.center_micron - 12.8) < 0.01
        assert abs(gf.sigma_micron - 0.015) < 0.005
        assert gf.amplitude > 1.5

    def test_noisy_fit(self) -> None:
        rng = np.random.default_rng(42)
        wl = np.linspace(12.5, 13.1, 200)
        flux = _inject_gaussian(wl, 12.8, 2.0, 0.015) + rng.normal(0, 0.1, 200)
        peak = Peak(index=100, wavelength_micron=12.8, flux=2.0, prominence=2.0, width_samples=5, snr=10.0)
        gf = fit_peak_gaussian(wl, flux, peak)
        assert gf is not None
        assert abs(gf.center_micron - 12.8) < 0.02

    def test_returns_none_for_insufficient_data(self) -> None:
        wl = np.array([12.79, 12.80, 12.81])
        flux = np.array([0.1, 2.0, 0.1])
        peak = Peak(index=1, wavelength_micron=12.8, flux=2.0, prominence=1.9, width_samples=1, snr=10.0)
        assert fit_peak_gaussian(wl, flux, peak) is None

    def test_diagnostics_populated(self) -> None:
        wl = np.linspace(12.5, 13.1, 200)
        flux = _inject_gaussian(wl, 12.8, 2.0, 0.015)
        peak = Peak(index=100, wavelength_micron=12.8, flux=2.0, prominence=2.0, width_samples=5, snr=10.0)
        gf = fit_peak_gaussian(wl, flux, peak, window_half_width_micron=0.2)
        assert gf is not None
        assert np.isfinite(gf.residual_normality_p)
        assert np.isfinite(gf.durbin_watson)
        assert gf.durbin_watson > 0


class TestVoigtFit:
    def test_voigt_fit_on_gaussian_data(self) -> None:
        """Voigt should be able to fit pure Gaussian data (gamma_L → 0)."""
        wl = np.linspace(12.5, 13.1, 200)
        flux = _inject_gaussian(wl, 12.8, 2.0, 0.015)
        peak = Peak(index=100, wavelength_micron=12.8, flux=2.0, prominence=2.0, width_samples=5, snr=10.0)
        vf = fit_peak_voigt(wl, flux, peak, window_half_width_micron=0.2)
        assert vf is not None
        assert abs(vf.center_micron - 12.8) < 0.01
        assert np.isfinite(vf.bic)


class TestMultipletDeblending:
    def test_deblend_two_lines(self) -> None:
        wl = np.linspace(14.0, 14.7, 300)
        flux = (_inject_gaussian(wl, 14.32, 1.5, 0.012) +
                _inject_gaussian(wl, 14.37, 1.0, 0.012))
        peaks = [
            Peak(index=130, wavelength_micron=14.32, flux=1.5, prominence=1.5, width_samples=5, snr=8.0),
            Peak(index=150, wavelength_micron=14.37, flux=1.0, prominence=1.0, width_samples=5, snr=6.0),
        ]
        fits = fit_multiplet(wl, flux, peaks, window_half_width_micron=0.25)
        assert len(fits) == 2
        centers = sorted(f.center_micron for f in fits)
        assert abs(centers[0] - 14.32) < 0.02
        assert abs(centers[1] - 14.37) < 0.02


class TestFitAll:
    def test_fit_all_returns_both(self) -> None:
        wl = np.linspace(12.5, 13.1, 200)
        flux = _inject_gaussian(wl, 12.8, 2.0, 0.015)
        peak = Peak(index=100, wavelength_micron=12.8, flux=2.0, prominence=2.0, width_samples=5, snr=10.0)
        gauss, voigt = fit_all_peaks(wl, flux, [peak], try_voigt=True)
        assert len(gauss) == 1


class TestDiagnostics:
    def test_normal_residuals(self) -> None:
        rng = np.random.default_rng(123)
        r = rng.normal(0, 1, 100)
        sw_p, dw = _compute_diagnostics(r)
        assert sw_p > 0.01  # Should pass normality test
        assert 1.5 < dw < 2.5  # Should be uncorrelated

    def test_autocorrelated_residuals(self) -> None:
        # Create positively autocorrelated residuals
        x = np.linspace(0, 4 * np.pi, 100)
        r = np.sin(x)
        _, dw = _compute_diagnostics(r)
        assert dw < 1.5  # Should show positive autocorrelation


class TestRedshiftEstimation:
    def test_basic_redshift(self) -> None:
        z_true = 0.01
        matches = [
            LineMatch(
                peak_index=10, observed_wavelength_micron=12.8136 * (1 + z_true),
                line_name="[Ne II]", rest_wavelength_micron=12.8136,
                ionization_class="low", delta_micron=0.0, redshift=z_true,
                confidence=0.8, ambiguous=False,
            ),
        ]
        z, scatter = estimate_redshift_from_matches(matches)
        assert abs(z - z_true) < 1e-4

    def test_empty_matches(self) -> None:
        z, scatter = estimate_redshift_from_matches([])
        assert z == 0.0
        assert np.isnan(scatter)
