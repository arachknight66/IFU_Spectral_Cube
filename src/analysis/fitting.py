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
    """Result of a single Gaussian fit to a spectral peak with uncertainty propagation.

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the Gaussian.
    amplitude_err : float
        1-sigma uncertainty on amplitude.
    center : float
        Centroid wavelength in µm.
    center_err : float
        1-sigma uncertainty on centroid wavelength in µm.
    sigma : float
        Standard deviation of the Gaussian in µm.
    sigma_err : float
        1-sigma uncertainty on sigma in µm.
    fwhm : float
        Full-width at half-maximum in µm.
    fwhm_err : float
        1-sigma uncertainty on FWHM in µm.
    integrated_flux : float
        Total flux under the Gaussian (A·σ·√(2π)).
    integrated_flux_err : float
        1-sigma uncertainty on integrated flux.
    fit_success : bool
        Whether the fit converged.
    residual_rms : float
        RMS of the fit residuals in the fit window.
    reduced_chi2 : float
        Reduced chi-squared (χ² / DOF) of the fit.
    dof : int
        Degrees of freedom in the fit window (N - 3).
    """
    amplitude: float
    amplitude_err: float
    center: float
    center_err: float
    sigma: float
    sigma_err: float
    fwhm: float
    fwhm_err: float
    integrated_flux: float
    integrated_flux_err: float
    fit_success: bool
    residual_rms: float
    reduced_chi2: float
    dof: int


def _gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian profile function."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def fit_gaussian_peaks(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    peak_indices: np.ndarray,
    fit_window_channels: int = 10,
    err: np.ndarray | None = None,
) -> list[GaussianFit]:
    """Fit Gaussian profiles to detected peaks with weighted error propagation.

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
    err : np.ndarray, optional
        1D standard error/uncertainty array.

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
        err_window = err[lo:hi] if err is not None else None

        if err_window is not None:
            # Avoid 0 or negative errors
            err_window = np.maximum(1e-12, err_window)

        # Initial guesses
        a0 = float(spectrum[idx])
        mu0 = float(wavelength[idx])
        half_max = a0 / 2.0
        above_half = np.where(flux_window >= half_max)[0]
        if len(above_half) >= 2:
            sig0 = (wl_window[above_half[-1]] - wl_window[above_half[0]]) / FWHM_FACTOR
        else:
            sig0 = abs(wavelength[1] - wavelength[0]) * 2.0

        sig0 = max(sig0, abs(wavelength[1] - wavelength[0]) * 0.5)

        try:
            if err_window is not None:
                popt, pcov = curve_fit(
                    _gaussian, wl_window, flux_window,
                    p0=[a0, mu0, sig0],
                    sigma=err_window,
                    absolute_sigma=True,
                    bounds=([0, wl_window.min(), 0],
                            [np.inf, wl_window.max(), wl_window.max() - wl_window.min()]),
                    maxfev=5000,
                )
            else:
                popt, pcov = curve_fit(
                    _gaussian, wl_window, flux_window,
                    p0=[a0, mu0, sig0],
                    bounds=([0, wl_window.min(), 0],
                            [np.inf, wl_window.max(), wl_window.max() - wl_window.min()]),
                    maxfev=5000,
                )

            amp, center, sigma = popt
            perr = np.sqrt(np.maximum(0.0, np.diag(pcov))) if pcov is not None else np.zeros(3)
            amp_err, center_err, sigma_err = perr[0], perr[1], perr[2]

            fwhm = sigma * FWHM_FACTOR
            fwhm_err = sigma_err * FWHM_FACTOR
            integrated = amp * sigma * np.sqrt(2.0 * np.pi)

            # Covariance between amp and sigma for integrated flux error propagation
            cov_amp_sig = pcov[0, 2] if pcov is not None else 0.0
            var_integrated = (2.0 * np.pi) * (
                (sigma ** 2) * (amp_err ** 2) +
                (amp ** 2) * (sigma_err ** 2) +
                2.0 * amp * sigma * cov_amp_sig
            )
            integrated_err = float(np.sqrt(np.maximum(0.0, var_integrated)))

            fitted = _gaussian(wl_window, *popt)
            residuals = flux_window - fitted
            residual_rms = float(np.sqrt(np.mean(residuals ** 2)))

            dof = max(1, len(wl_window) - 3)
            if err_window is not None:
                chi2 = float(np.sum((residuals / err_window) ** 2))
            else:
                chi2 = float(np.sum(residuals ** 2) / (residual_rms ** 2 if residual_rms > 0 else 1.0))
            reduced_chi2 = float(chi2 / dof)

            results.append(GaussianFit(
                amplitude=float(amp),
                amplitude_err=float(amp_err),
                center=float(center),
                center_err=float(center_err),
                sigma=float(sigma),
                sigma_err=float(sigma_err),
                fwhm=float(fwhm),
                fwhm_err=float(fwhm_err),
                integrated_flux=float(integrated),
                integrated_flux_err=float(integrated_err),
                fit_success=True,
                residual_rms=residual_rms,
                reduced_chi2=reduced_chi2,
                dof=dof,
            ))
        except (RuntimeError, ValueError):
            results.append(GaussianFit(
                amplitude=float(a0),
                amplitude_err=0.0,
                center=float(mu0),
                center_err=0.0,
                sigma=float(sig0),
                sigma_err=0.0,
                fwhm=float(sig0 * FWHM_FACTOR),
                fwhm_err=0.0,
                integrated_flux=0.0,
                integrated_flux_err=0.0,
                fit_success=False,
                residual_rms=float("inf"),
                reduced_chi2=999.0,
                dof=max(1, len(wl_window) - 3),
            ))

    return results
