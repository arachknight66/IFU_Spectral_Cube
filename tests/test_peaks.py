"""Tests for peak detection, CWT, merging, and local SNR refinement."""
from __future__ import annotations

import numpy as np
import pytest

from ifu_spectral_cube.models import Peak
from ifu_spectral_cube.peaks import (
    detect_peaks_1d,
    detect_peaks_cwt,
    merge_peak_lists,
    refine_peak_local_snr,
    suggest_miri_peak_params,
)


def _make_spectrum_with_peaks() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic spectrum: noise + 3 Gaussian emission lines."""
    rng = np.random.default_rng(42)
    wl = np.linspace(10.0, 20.0, 3000)
    flux = rng.normal(0.0, 0.12, size=wl.size)

    def g(mu: float, amp: float, sig: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((wl - mu) / sig) ** 2)

    flux += g(12.8136, 1.8, 0.012)
    flux += g(15.5551, 1.4, 0.014)
    flux += g(17.0348, 0.8, 0.010)
    return wl, flux


class TestFindPeaks:
    def test_detects_injected_lines(self) -> None:
        wl, flux = _make_spectrum_with_peaks()
        peaks, _, sigma = detect_peaks_1d(wl, flux, prominence_sigma=5.0, width=(2, 40), distance=5)
        assert len(peaks) >= 2
        peak_wls = [p.wavelength_micron for p in peaks]
        # Check that Ne II and Ne III are found
        assert any(abs(w - 12.8136) < 0.05 for w in peak_wls)
        assert any(abs(w - 15.5551) < 0.05 for w in peak_wls)

    def test_noise_only_no_peaks(self) -> None:
        rng = np.random.default_rng(99)
        wl = np.linspace(10, 20, 1000)
        flux = rng.normal(0, 0.1, 1000)
        peaks, _, _ = detect_peaks_1d(wl, flux, prominence_sigma=8.0)
        assert len(peaks) == 0

    def test_peak_snr_positive(self) -> None:
        wl, flux = _make_spectrum_with_peaks()
        peaks, _, _ = detect_peaks_1d(wl, flux, prominence_sigma=3.0)
        for p in peaks:
            assert p.snr > 0


class TestCWTPeaks:
    def test_cwt_detects_lines(self) -> None:
        wl, flux = _make_spectrum_with_peaks()
        peaks, sigma = detect_peaks_cwt(wl, flux, min_snr=3.0)
        assert len(peaks) >= 1

    def test_cwt_returns_valid_indices(self) -> None:
        wl, flux = _make_spectrum_with_peaks()
        peaks, _ = detect_peaks_cwt(wl, flux, min_snr=3.0)
        for p in peaks:
            assert 0 <= p.index < len(wl)


class TestMergePeaks:
    def test_merge_removes_duplicates(self) -> None:
        pa = [Peak(index=100, wavelength_micron=12.8, flux=1.0, prominence=0.5, width_samples=3, snr=5.0)]
        pb = [Peak(index=101, wavelength_micron=12.81, flux=1.1, prominence=0.6, width_samples=3, snr=6.0)]
        merged = merge_peak_lists(pa, pb, merge_radius_channels=3)
        assert len(merged) == 1
        assert merged[0].snr == 6.0  # Higher SNR kept

    def test_merge_keeps_distant_peaks(self) -> None:
        pa = [Peak(index=100, wavelength_micron=12.8, flux=1.0, prominence=0.5, width_samples=3, snr=5.0)]
        pb = [Peak(index=200, wavelength_micron=15.5, flux=1.1, prominence=0.6, width_samples=3, snr=6.0)]
        merged = merge_peak_lists(pa, pb, merge_radius_channels=3)
        assert len(merged) == 2

    def test_merge_empty_lists(self) -> None:
        assert merge_peak_lists([], []) == []
        pa = [Peak(index=10, wavelength_micron=12.0, flux=1.0, prominence=0.5, width_samples=3, snr=5.0)]
        assert merge_peak_lists(pa, []) == pa


class TestLocalSNR:
    def test_refines_snr(self) -> None:
        wl, flux = _make_spectrum_with_peaks()
        peaks, _, _ = detect_peaks_1d(wl, flux, prominence_sigma=4.0, width=(2, 40))
        if peaks:
            peaks = refine_peak_local_snr(wl, flux, peaks, local_window_channels=80)
            for p in peaks:
                assert np.isfinite(p.local_snr)
                assert p.local_snr > 0


class TestSuggestParams:
    def test_returns_valid_params(self) -> None:
        wl = np.linspace(5, 28, 2000)
        params = suggest_miri_peak_params(wl, resolving_power=2500.0)
        assert "width" in params
        assert "distance" in params
        assert params["distance"] >= 1
