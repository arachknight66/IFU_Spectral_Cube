"""Spectral line fitting: Gaussian, Voigt, and multi-component models.

Physical motivation
-------------------
* **Gaussian profile** — appropriate when line broadening is dominated by
  thermal motions (Doppler broadening) or instrumental resolution.  The
  MIRI MRS LSF is approximately Gaussian with FWHM ≈ λ/R.

* **Voigt profile** — convolution of Gaussian (thermal + instrumental)
  and Lorentzian (natural + pressure broadening).  Relevant when:
  - Pressure broadening is significant (dense nebulae, stellar atmospheres).
  - The line sits on broad wings from underlying transitions.
  At MIRI resolution R~1500–3500, Lorentzian wings are usually sub-dominant
  but can matter for strong AGN coronal lines.

* **Multi-Gaussian deblending** — simultaneous fit of 2–4 Gaussians
  sharing a linear baseline, for resolving blended multiplets (e.g.,
  [Ne V] 14.32 + [Cl II] 14.37, or PAH 12.7 + [Ne II] 12.81).

Fit diagnostics
---------------
* **Reduced χ²** — goodness-of-fit (≈1 ideal, >>1 poor, <<1 overfitting).
* **Shapiro-Wilk normality test** — tests whether residuals are Gaussian;
  low p-value suggests systematic structure in residuals.
* **Durbin-Watson statistic** — tests for autocorrelation of residuals;
  value ≈ 2 = uncorrelated, < 1.5 = positive autocorrelation (e.g.
  residual fringes or unmodeled blends).
* **BIC (Bayesian Information Criterion)** — for model selection between
  Gaussian and Voigt: BIC = n·ln(RSS/n) + k·ln(n), lower is better.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.stats import shapiro

from .models import GaussianFit, LineMatch, Peak, VoigtFit


# ---------------------------------------------------------------------------
# Profile functions
# ---------------------------------------------------------------------------

def _gaussian_linear(
    x: np.ndarray,
    amp: float,
    mu: float,
    sigma: float,
    c0: float,
    c1: float,
) -> np.ndarray:
    """Gaussian peak + linear baseline: A·exp(−(x−μ)²/2σ²) + c0 + c1·(x−μ)."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c0 + c1 * (x - mu)


def _voigt_linear(
    x: np.ndarray,
    amp: float,
    mu: float,
    sigma_g: float,
    gamma_l: float,
    c0: float,
    c1: float,
) -> np.ndarray:
    """Voigt profile + linear baseline.

    Uses ``scipy.special.voigt_profile`` which evaluates the normalized Voigt
    function V(x; σ, γ) = Real[Faddeeva(z)] / (σ√(2π)).  We scale by
    amplitude and add a linear baseline.
    """
    vp = voigt_profile(x - mu, sigma_g, gamma_l)
    # Normalize so that the peak value of the Voigt function maps to amplitude
    vp_peak = voigt_profile(0.0, sigma_g, gamma_l)
    if vp_peak > 0:
        vp = vp / vp_peak
    return amp * vp + c0 + c1 * (x - mu)


def _multi_gaussian_linear(x: np.ndarray, *params: float) -> np.ndarray:
    """Sum of N Gaussians + shared linear baseline.

    Parameters are packed as: [amp1, mu1, sig1, amp2, mu2, sig2, …, c0, c1]
    """
    c0 = params[-2]
    c1 = params[-1]
    n_gauss = (len(params) - 2) // 3
    y = c0 + c1 * (x - np.mean(x))
    for i in range(n_gauss):
        amp = params[3 * i]
        mu = params[3 * i + 1]
        sig = params[3 * i + 2]
        y = y + amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return y


# ---------------------------------------------------------------------------
# Fit quality diagnostics
# ---------------------------------------------------------------------------

def _compute_diagnostics(
    residuals: np.ndarray,
) -> tuple[float, float]:
    """Compute Shapiro-Wilk normality p-value and Durbin-Watson statistic.

    Shapiro-Wilk
    ~~~~~~~~~~~~~
    Tests H₀: residuals are drawn from a normal distribution.
    A high p-value (> 0.05) means residuals are consistent with Gaussian
    noise and the model is a good fit.

    Durbin-Watson
    ~~~~~~~~~~~~~
    DW = Σ(e_t − e_{t−1})² / Σ e_t²

    DW ≈ 2 → no autocorrelation (ideal).
    DW < 1.5 → positive autocorrelation → residual systematics (fringes,
    unmodeled blends).
    DW > 2.5 → negative autocorrelation (rare in spectroscopy).
    """
    r = residuals[np.isfinite(residuals)]
    if r.size < 8:
        return float("nan"), float("nan")

    # Shapiro-Wilk (limit to 5000 samples for performance)
    try:
        _, sw_p = shapiro(r[:5000])
    except Exception:
        sw_p = float("nan")

    # Durbin-Watson
    diffs = np.diff(r)
    ss_res = np.sum(r ** 2)
    dw = float(np.sum(diffs ** 2) / ss_res) if ss_res > 0 else float("nan")

    return float(sw_p), float(dw)


def _compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion: BIC = n·ln(RSS/n) + k·ln(n).

    Lower BIC = better model.  Used to compare Gaussian (k=5) vs Voigt (k=6)
    fits without over-penalizing the simpler model.
    """
    if n <= 0 or rss <= 0:
        return float("nan")
    return n * math.log(rss / n) + k * math.log(n)


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------

def fit_peak_gaussian(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    peak: Peak,
    window_half_width_micron: float = 0.08,
) -> GaussianFit | None:
    """Fit a single Gaussian + linear baseline around a detected peak.

    The fit window is ±``window_half_width_micron`` around the peak
    wavelength.  We require at least 7 data points for a 5-parameter fit
    (amplitude, center, sigma, baseline offset, baseline slope).
    """
    wl = np.asarray(wavelength_micron, dtype=float)
    y = np.asarray(flux, dtype=float)

    win = np.abs(wl - peak.wavelength_micron) <= window_half_width_micron
    if np.count_nonzero(win) < 7:
        return None

    xw = wl[win]
    yw = y[win]
    y_floor = float(np.nanmedian(yw))
    amp0 = max(peak.flux - y_floor, np.nanmax(yw) - y_floor, 1e-8)
    sigma0 = max(window_half_width_micron / 3.0, 5e-4)

    p0 = [amp0, peak.wavelength_micron, sigma0, y_floor, 0.0]
    bounds = (
        [0.0, peak.wavelength_micron - window_half_width_micron, 1e-5, -np.inf, -np.inf],
        [np.inf, peak.wavelength_micron + window_half_width_micron, window_half_width_micron, np.inf, np.inf],
    )

    try:
        popt, pcov = curve_fit(_gaussian_linear, xw, yw, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None

    model = _gaussian_linear(xw, *popt)
    resid = yw - model
    dof = max(1, xw.size - len(popt))
    red_chi2 = float(np.nansum(resid ** 2) / dof)
    sw_p, dw = _compute_diagnostics(resid)

    errs = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    return GaussianFit(
        peak_index=peak.index,
        amplitude=float(popt[0]),
        center_micron=float(popt[1]),
        sigma_micron=float(popt[2]),
        baseline_offset=float(popt[3]),
        baseline_slope=float(popt[4]),
        center_uncertainty_micron=float(errs[1]) if errs.size > 1 else float("nan"),
        sigma_uncertainty_micron=float(errs[2]) if errs.size > 2 else float("nan"),
        reduced_chi2=red_chi2,
        residual_normality_p=sw_p,
        durbin_watson=dw,
    )


# ---------------------------------------------------------------------------
# Voigt fitting
# ---------------------------------------------------------------------------

def fit_peak_voigt(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    peak: Peak,
    window_half_width_micron: float = 0.08,
) -> VoigtFit | None:
    """Fit a Voigt profile + linear baseline around a detected peak.

    The Voigt profile is the convolution of Gaussian and Lorentzian
    components, adding one parameter (gamma_L) compared to the pure
    Gaussian fit.  Use BIC to decide whether the extra parameter is
    justified.
    """
    wl = np.asarray(wavelength_micron, dtype=float)
    y = np.asarray(flux, dtype=float)

    win = np.abs(wl - peak.wavelength_micron) <= window_half_width_micron
    if np.count_nonzero(win) < 8:  # Need ≥8 points for 6 parameters
        return None

    xw = wl[win]
    yw = y[win]
    y_floor = float(np.nanmedian(yw))
    amp0 = max(peak.flux - y_floor, np.nanmax(yw) - y_floor, 1e-8)
    sigma0 = max(window_half_width_micron / 4.0, 5e-4)
    gamma0 = sigma0 * 0.1  # Start with small Lorentzian contribution

    p0 = [amp0, peak.wavelength_micron, sigma0, gamma0, y_floor, 0.0]
    bounds = (
        [0.0, peak.wavelength_micron - window_half_width_micron, 1e-5, 1e-6, -np.inf, -np.inf],
        [np.inf, peak.wavelength_micron + window_half_width_micron, window_half_width_micron, window_half_width_micron, np.inf, np.inf],
    )

    try:
        popt, pcov = curve_fit(_voigt_linear, xw, yw, p0=p0, bounds=bounds, maxfev=30000)
    except Exception:
        return None

    model = _voigt_linear(xw, *popt)
    resid = yw - model
    dof = max(1, xw.size - len(popt))
    red_chi2 = float(np.nansum(resid ** 2) / dof)
    rss = float(np.nansum(resid ** 2))
    bic = _compute_bic(xw.size, len(popt), rss)
    sw_p, dw = _compute_diagnostics(resid)

    errs = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    return VoigtFit(
        peak_index=peak.index,
        amplitude=float(popt[0]),
        center_micron=float(popt[1]),
        sigma_gauss_micron=float(popt[2]),
        gamma_lorentz_micron=float(popt[3]),
        baseline_offset=float(popt[4]),
        baseline_slope=float(popt[5]),
        center_uncertainty_micron=float(errs[1]) if errs.size > 1 else float("nan"),
        reduced_chi2=red_chi2,
        bic=bic,
        residual_normality_p=sw_p,
        durbin_watson=dw,
    )


# ---------------------------------------------------------------------------
# Multi-Gaussian deblending
# ---------------------------------------------------------------------------

def fit_multiplet(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    peaks_in_window: list[Peak],
    window_half_width_micron: float = 0.15,
) -> list[GaussianFit]:
    """Simultaneously fit 2–4 Gaussians + shared baseline for blended lines.

    When multiple peaks fall within a narrow wavelength range (e.g.,
    [Ne V] 14.32 + [Cl II] 14.37), fitting them independently with
    single Gaussians leads to biased amplitudes and centroids.  Joint
    fitting on a shared baseline yields deblended parameters.

    The fit window is centered on the mean wavelength of the group.
    """
    if not peaks_in_window:
        return []
    if len(peaks_in_window) > 4:
        # Too many → fall back to individual fits
        return []

    wl = np.asarray(wavelength_micron, dtype=float)
    y = np.asarray(flux, dtype=float)

    center_wl = np.mean([p.wavelength_micron for p in peaks_in_window])
    win = np.abs(wl - center_wl) <= window_half_width_micron
    if np.count_nonzero(win) < 3 * len(peaks_in_window) + 4:
        return []

    xw = wl[win]
    yw = y[win]
    y_floor = float(np.nanmedian(yw))
    n_gauss = len(peaks_in_window)

    # Build initial parameters: [amp1, mu1, sig1, amp2, mu2, sig2, …, c0, c1]
    p0: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    for pk in peaks_in_window:
        amp0 = max(pk.flux - y_floor, 1e-8)
        p0.extend([amp0, pk.wavelength_micron, max(0.005, window_half_width_micron / 5.0)])
        lb.extend([0.0, pk.wavelength_micron - 0.03, 1e-5])
        ub.extend([np.inf, pk.wavelength_micron + 0.03, window_half_width_micron])
    p0.extend([y_floor, 0.0])
    lb.extend([-np.inf, -np.inf])
    ub.extend([np.inf, np.inf])

    try:
        popt, pcov = curve_fit(
            _multi_gaussian_linear, xw, yw, p0=p0,
            bounds=(lb, ub), maxfev=50000,
        )
    except Exception:
        return []

    model = _multi_gaussian_linear(xw, *popt)
    resid = yw - model
    dof = max(1, xw.size - len(popt))
    red_chi2 = float(np.nansum(resid ** 2) / dof)
    sw_p, dw = _compute_diagnostics(resid)
    errs = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))

    fits: list[GaussianFit] = []
    for i, pk in enumerate(peaks_in_window):
        fits.append(GaussianFit(
            peak_index=pk.index,
            amplitude=float(popt[3 * i]),
            center_micron=float(popt[3 * i + 1]),
            sigma_micron=float(popt[3 * i + 2]),
            baseline_offset=float(popt[-2]),
            baseline_slope=float(popt[-1]),
            center_uncertainty_micron=float(errs[3 * i + 1]) if errs.size > 3 * i + 1 else float("nan"),
            sigma_uncertainty_micron=float(errs[3 * i + 2]) if errs.size > 3 * i + 2 else float("nan"),
            reduced_chi2=red_chi2,
            residual_normality_p=sw_p,
            durbin_watson=dw,
        ))
    return fits


# ---------------------------------------------------------------------------
# Batch fitting with model selection
# ---------------------------------------------------------------------------

def fit_all_peaks(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    peaks: list[Peak],
    window_half_width_micron: float = 0.08,
    try_voigt: bool = True,
) -> tuple[list[GaussianFit], list[VoigtFit]]:
    """Fit all detected peaks with Gaussian (and optionally Voigt) profiles.

    When ``try_voigt=True``, both models are fit and the Voigt result is
    kept only if it has lower BIC than the Gaussian (the extra Lorentzian
    parameter is justified by the data).

    Returns
    -------
    gaussian_fits : list of Gaussian fits for all peaks
    voigt_fits : list of Voigt fits where BIC favoured the Voigt model
    """
    gauss_fits: list[GaussianFit] = []
    voigt_fits: list[VoigtFit] = []

    for peak in peaks:
        gfit = fit_peak_gaussian(
            wavelength_micron=wavelength_micron,
            flux=flux,
            peak=peak,
            window_half_width_micron=window_half_width_micron,
        )
        if gfit is not None:
            gauss_fits.append(gfit)

        if try_voigt:
            vfit = fit_peak_voigt(
                wavelength_micron=wavelength_micron,
                flux=flux,
                peak=peak,
                window_half_width_micron=window_half_width_micron,
            )
            if vfit is not None:
                # Compare BIC — keep Voigt only if it is strictly better
                if gfit is not None:
                    # Compute Gaussian BIC for comparison
                    wl = np.asarray(wavelength_micron, dtype=float)
                    win = np.abs(wl - peak.wavelength_micron) <= window_half_width_micron
                    n_pts = int(np.count_nonzero(win))
                    gauss_rss = gfit.reduced_chi2 * max(1, n_pts - 5)
                    gauss_bic = _compute_bic(n_pts, 5, gauss_rss)
                    if vfit.bic < gauss_bic:
                        voigt_fits.append(vfit)
                else:
                    voigt_fits.append(vfit)

    return gauss_fits, voigt_fits


# ---------------------------------------------------------------------------
# Redshift estimation from matched lines
# ---------------------------------------------------------------------------

def estimate_redshift_from_matches(
    matches: list[LineMatch],
    fits: list[GaussianFit] | None = None,
) -> tuple[float, float]:
    """Weighted-mean redshift from matched line identifications.

    When Gaussian fits are available, uses fitted line centers and their
    uncertainties for inverse-variance weighting.  Otherwise falls back
    to peak positions weighted by match confidence.

    Returns (z_mean, z_scatter).
    """
    if not matches:
        return 0.0, float("nan")

    fit_center_by_peak = {f.peak_index: f.center_micron for f in (fits or [])}
    fit_error_by_peak = {
        f.peak_index: max(f.center_uncertainty_micron, 1e-8)
        for f in (fits or [])
    }

    z_vals: list[float] = []
    weights: list[float] = []
    for m in matches:
        center = fit_center_by_peak.get(m.peak_index, m.observed_wavelength_micron)
        z = center / m.rest_wavelength_micron - 1.0
        z_vals.append(z)

        if m.peak_index in fit_error_by_peak:
            z_sigma = fit_error_by_peak[m.peak_index] / m.rest_wavelength_micron
            weights.append(1.0 / max(z_sigma ** 2, 1e-12))
        else:
            weights.append(max(1e-6, m.confidence))

    w = np.asarray(weights, dtype=float)
    z_arr = np.asarray(z_vals, dtype=float)
    z_mean = float(np.average(z_arr, weights=w))
    variance = float(np.average((z_arr - z_mean) ** 2, weights=w))
    return z_mean, math.sqrt(max(0.0, variance))
