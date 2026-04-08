"""
Gaussian Profile Fitting for Detected Spectral Peaks.

For each detected peak, fits a Gaussian profile g(λ) = A·exp(-(λ-μ)²/(2σ²))
to extract physical parameters: amplitude, centroid, line width (σ and FWHM),
and integrated flux.

The relationship FWHM = 2√(2·ln2)·σ ≈ 2.3548·σ connects the Gaussian
width to the full-width at half-maximum commonly used in spectroscopy.

Integrated line flux = A·σ·√(2π), which gives the total flux in the line
independent of the spectral resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit


FWHM_FACTOR: float = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


@dataclass
class GaussianFit:
    """Result of a single Gaussian fit to a spectral peak.

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the Gaussian.
    center : float
        Centroid wavelength in µm.
    sigma : float
        Standard deviation of the Gaussian in µm.
    fwhm : float
        Full-width at half-maximum in µm.
    integrated_flux : float
        Total flux under the Gaussian (A·σ·√(2π)).
    fit_success : bool
        Whether the fit converged.
    residual_rms : float
        RMS of the fit residuals in the fit window.
    """
    amplitude: float
    center: float
    sigma: float
    fwhm: float
    integrated_flux: float
    fit_success: bool
    residual_rms: float


def _gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian profile function."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def fit_gaussian_peaks(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    peak_indices: np.ndarray,
    fit_window_channels: int = 10,
) -> list[GaussianFit]:
    """Fit Gaussian profiles to detected peaks.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength axis in µm.
    spectrum : np.ndarray
        1D flux array.
    peak_indices : np.ndarray
        Indices of detected peaks.
    fit_window_channels : int
        Half-width of the fitting window around each peak (in channels).

    Returns
    -------
    list of GaussianFit
        One result per input peak.
    """
    results: list[GaussianFit] = []
    n = len(spectrum)

    for idx in peak_indices:
        idx = int(idx)
        lo = max(0, idx - fit_window_channels)
        hi = min(n, idx + fit_window_channels + 1)

        wl_window = wavelength[lo:hi]
        flux_window = spectrum[lo:hi]

        # Initial guesses
        a0 = float(spectrum[idx])
        mu0 = float(wavelength[idx])
        # Estimate sigma from half-width of peak region above half-max
        half_max = a0 / 2.0
        above_half = np.where(flux_window >= half_max)[0]
        if len(above_half) >= 2:
            sig0 = (wl_window[above_half[-1]] - wl_window[above_half[0]]) / FWHM_FACTOR
        else:
            sig0 = abs(wavelength[1] - wavelength[0]) * 2.0

        sig0 = max(sig0, abs(wavelength[1] - wavelength[0]) * 0.5)

        try:
            popt, _ = curve_fit(
                _gaussian, wl_window, flux_window,
                p0=[a0, mu0, sig0],
                bounds=([0, wl_window.min(), 0],
                        [np.inf, wl_window.max(), wl_window.max() - wl_window.min()]),
                maxfev=5000,
            )
            amp, center, sigma = popt
            fwhm = sigma * FWHM_FACTOR
            integrated = amp * sigma * np.sqrt(2.0 * np.pi)

            fitted = _gaussian(wl_window, *popt)
            residual_rms = float(np.sqrt(np.mean((flux_window - fitted) ** 2)))

            results.append(GaussianFit(
                amplitude=float(amp),
                center=float(center),
                sigma=float(sigma),
                fwhm=float(fwhm),
                integrated_flux=float(integrated),
                fit_success=True,
                residual_rms=residual_rms,
            ))
        except (RuntimeError, ValueError):
            results.append(GaussianFit(
                amplitude=float(a0),
                center=float(mu0),
                sigma=float(sig0),
                fwhm=float(sig0 * FWHM_FACTOR),
                integrated_flux=0.0,
                fit_success=False,
                residual_rms=float("inf"),
            ))

    return results
