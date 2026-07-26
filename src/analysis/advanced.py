"""
Advanced Spectral Analysis: Redshift Estimation and Confidence Scoring.

This module extends the basic line-identification workflow with:
    1. Gaussian profile fitting with proper uncertainty propagation
    2. Redshift estimation via cross-correlation with the line catalog
    3. Multi-factor confidence scoring for each line identification

Redshift estimation algorithm
-----------------------------
For each trial redshift z on a grid, we shift the rest-frame catalog to
observed wavelengths (λ_obs = λ_rest × (1+z)) and count how many
detected peaks fall within tolerance of a shifted catalog line.  The best
z maximises a weighted match score:

    S(z) = Σ_matches  w_snr × w_proximity × w_prior

This is essentially a 1D cross-correlation between the detected line
pattern and the catalog template, which is the standard approach for
spectroscopic redshift determination (e.g., Baldry et al. 2014).

Confidence scoring
------------------
Each identified line receives a confidence score C ∈ [0, 1] combining
four orthogonal quality indicators:

    C = w₁·f(SNR) + w₂·f(Δλ) + w₃·f(prior) + w₄·f(χ²)

where:
    f(SNR)    = min(1, SNR/10)           — saturates at SNR=10
    f(Δλ)     = 1 - |Δλ|/tolerance       — linear proximity score
    f(prior)  = 0.8 for common, 0.5 else — astrophysical prior
    f(χ²)     = exp(-|χ²_red - 1|)       — Gaussian fit quality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from ..utils.constants import get_line_database

# Lines that are strong and commonly detected in the MIR
_COMMON_LINES = {
    "[Ne II]", "[Ne III]", "[S III]", "[S IV]", "[Ar II]", "[Ar III]",
    "H₂", "H I", "PAH", "[Fe II]", "[O IV]",
}

SPEED_OF_LIGHT_KM_S: float = 299792.458
FWHM_FACTOR: float = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.35482


def get_miri_mrs_lsf(wavelength_um: float) -> tuple[float, float]:
    """Return JWST MIRI MRS resolving power R and instrumental FWHM (µm).

    Based on Labiano et al. (2021) and JWST MIRI MRS calibration pipeline metrics:
    - Channel 1 (4.9–7.65 µm):   R ~ 3100 – 3700
    - Channel 2 (7.5–11.7 µm):   R ~ 2800 – 3300
    - Channel 3 (11.55–17.98 µm): R ~ 2400 – 2900
    - Channel 4 (17.7–27.9 µm):  R ~ 1300 – 1700
    """
    wl = float(wavelength_um)
    if wl <= 7.65:
        r = 3500.0 - 100.0 * (wl - 5.0)
    elif wl <= 11.7:
        r = 3200.0 - 120.0 * (wl - 7.5)
    elif wl <= 18.0:
        r = 2800.0 - 100.0 * (wl - 11.55)
    else:
        r = 1700.0 - 40.0 * (wl - 17.7)

    r = float(np.clip(r, 1200.0, 4000.0))
    fwhm_inst_um = wl / r
    return r, fwhm_inst_um


def compute_intrinsic_kinematics(
    fwhm_obs_um: float,
    fwhm_obs_err_um: float,
    wavelength_um: float,
) -> dict[str, float]:
    """Deconvolve instrumental LSF and compute intrinsic velocity dispersion (km/s).

    Returns
    -------
    dict
        Keys: 'resolving_power', 'fwhm_inst_um', 'fwhm_intrinsic_um',
              'sigma_v_kms', 'sigma_v_err_kms'
    """
    r_power, fwhm_inst_um = get_miri_mrs_lsf(wavelength_um)

    # Deconvolve in quadrature
    diff_sq = fwhm_obs_um ** 2 - fwhm_inst_um ** 2
    if diff_sq > 0:
        fwhm_intrinsic_um = float(np.sqrt(diff_sq))
        # sigma_v = (c * FWHM_intrinsic) / (2.35482 * lambda_0)
        sigma_v_kms = (SPEED_OF_LIGHT_KM_S * fwhm_intrinsic_um) / (FWHM_FACTOR * wavelength_um)
        # Error propagation: d(sigma_v)/d(FWHM_obs) = (c / (2.35482 * lambda)) * (FWHM_obs / FWHM_intrinsic)
        sigma_v_err_kms = (
            (SPEED_OF_LIGHT_KM_S / (FWHM_FACTOR * wavelength_um)) *
            (fwhm_obs_um / fwhm_intrinsic_um) *
            fwhm_obs_err_um
        )
    else:
        # Line is unresolved (dominated by instrumental LSF)
        fwhm_intrinsic_um = 0.0
        sigma_v_kms = 0.0
        sigma_v_err_kms = 0.0

    return {
        "resolving_power": r_power,
        "fwhm_inst_um": fwhm_inst_um,
        "fwhm_intrinsic_um": fwhm_intrinsic_um,
        "sigma_v_kms": float(sigma_v_kms),
        "sigma_v_err_kms": float(sigma_v_err_kms),
    }


@dataclass
class ConfidenceResult:
    """Confidence assessment for a single line identification.

    Attributes
    ----------
    peak_wavelength : float
        Observed peak wavelength (µm).
    matched_species : str
        Identified species name.
    rest_wavelength : float
        Rest-frame wavelength (µm).
    snr : float
        Signal-to-noise ratio of the peak.
    wavelength_error : float
        |λ_obs - λ_catalog| in µm.
    confidence : float
        Overall confidence score [0, 1].
    snr_score : float
        SNR component of confidence.
    proximity_score : float
        Wavelength proximity component.
    prior_score : float
        Astrophysical prior component.
    fit_score : float
        Gaussian fit quality component.
    """
    peak_wavelength: float
    matched_species: str
    rest_wavelength: float
    snr: float
    wavelength_error: float
    confidence: float
    snr_score: float
    proximity_score: float
    prior_score: float
    fit_score: float


@dataclass
class RedshiftResult:
    """Result of redshift estimation.

    Attributes
    ----------
    best_z : float
        Best-fit redshift.
    z_uncertainty : float
        Estimated uncertainty on redshift.
    match_score : float
        Peak matching score at best z.
    n_matched : int
        Number of lines matched at best z.
    z_grid : np.ndarray
        Trial redshift grid.
    score_grid : np.ndarray
        Matching score at each trial z.
    matched_lines : list of dict
        Details of lines matched at best z.
    """
    best_z: float
    z_uncertainty: float
    match_score: float
    n_matched: int
    z_grid: np.ndarray
    score_grid: np.ndarray
    matched_lines: list[dict]


def estimate_redshift(
    peak_wavelengths: np.ndarray,
    peak_snr: Optional[np.ndarray] = None,
    z_min: float = 0.0,
    z_max: float = 0.5,
    z_step: float = 0.0001,
    tolerance_um: float = 0.05,
    categories: Optional[list[str]] = None,
) -> RedshiftResult:
    """Estimate source redshift by cross-correlating peaks with catalog.

    For each trial redshift, shifts the catalog to observed frame and
    counts weighted matches with the detected peaks.

    Parameters
    ----------
    peak_wavelengths : np.ndarray
        Observed wavelengths of detected peaks (µm).
    peak_snr : np.ndarray, optional
        SNR of each peak (for weighting).  Uniform weights if None.
    z_min, z_max : float
        Redshift search range.
    z_step : float
        Redshift grid spacing.
    tolerance_um : float
        Maximum |Δλ| for a match (µm).
    categories : list of str, optional
        Restrict catalog to these categories.

    Returns
    -------
    RedshiftResult
        Best-fit redshift and supporting diagnostics.
    """
    if len(peak_wavelengths) == 0:
        return RedshiftResult(
            best_z=0.0, z_uncertainty=z_step,
            match_score=0.0, n_matched=0,
            z_grid=np.array([0.0]), score_grid=np.array([0.0]),
            matched_lines=[],
        )

    if peak_snr is None:
        peak_snr = np.ones(len(peak_wavelengths))

    # Get rest-frame catalog
    catalog = get_line_database(redshift=0.0, categories=categories)
    rest_wavelengths = np.array([l["rest_wavelength_um"] for l in catalog])

    z_grid = np.arange(z_min, z_max + z_step, z_step)
    score_grid = np.zeros_like(z_grid)

    for i, z in enumerate(z_grid):
        obs_catalog = rest_wavelengths * (1.0 + z)

        total_score = 0.0
        for j, peak_wl in enumerate(peak_wavelengths):
            errors = np.abs(obs_catalog - peak_wl)
            min_error = np.min(errors)
            if min_error <= tolerance_um:
                # Weight by SNR and proximity
                proximity = 1.0 - min_error / tolerance_um
                total_score += peak_snr[j] * proximity

        score_grid[i] = total_score

    # Find best redshift
    best_idx = np.argmax(score_grid)
    best_z = z_grid[best_idx]
    best_score = score_grid[best_idx]

    # Estimate uncertainty from the width of the peak in the score function
    # Use the half-maximum width of the score peak
    half_max = best_score / 2.0
    above_half = np.where(score_grid >= half_max)[0]
    if len(above_half) >= 2:
        z_uncertainty = (z_grid[above_half[-1]] - z_grid[above_half[0]]) / 2.0
    else:
        z_uncertainty = z_step

    # Identify lines matched at best z
    obs_catalog = rest_wavelengths * (1.0 + best_z)
    matched_lines = []
    n_matched = 0

    for peak_wl in peak_wavelengths:
        errors = np.abs(obs_catalog - peak_wl)
        min_idx = np.argmin(errors)
        if errors[min_idx] <= tolerance_um:
            n_matched += 1
            matched_lines.append({
                "peak_wavelength": float(peak_wl),
                "matched_species": catalog[min_idx]["species"],
                "rest_wavelength": catalog[min_idx]["rest_wavelength_um"],
                "observed_catalog": float(obs_catalog[min_idx]),
                "error_um": float(errors[min_idx]),
            })

    return RedshiftResult(
        best_z=float(best_z),
        z_uncertainty=float(z_uncertainty),
        match_score=float(best_score),
        n_matched=n_matched,
        z_grid=z_grid,
        score_grid=score_grid,
        matched_lines=matched_lines,
    )


def compute_confidence(
    peak_wavelength: float,
    matched_species: str,
    rest_wavelength: float,
    snr: float,
    tolerance_um: float,
    reduced_chi2: float = 1.0,
    weights: tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
) -> ConfidenceResult:
    """Compute a multi-factor confidence score for a line identification.

    Parameters
    ----------
    peak_wavelength : float
        Observed peak wavelength (µm).
    matched_species : str
        Name of the matched species.
    rest_wavelength : float
        Rest wavelength of the matched catalog line (µm).
    snr : float
        Signal-to-noise ratio of the peak.
    tolerance_um : float
        Tolerance used for matching (µm).
    reduced_chi2 : float
        Reduced χ² from Gaussian fit (1.0 = perfect).
    weights : tuple of 4 floats
        Weights for (SNR, proximity, prior, fit_quality) components.
        Must sum to 1.0.

    Returns
    -------
    ConfidenceResult
        Full confidence breakdown.
    """
    w_snr, w_prox, w_prior, w_fit = weights

    # 1. SNR score: saturates at SNR=10
    snr_score = min(1.0, max(0.0, snr / 10.0))

    # 2. Wavelength proximity: linear decay from 1 (exact) to 0 (at tolerance)
    error_um = abs(peak_wavelength - rest_wavelength)
    proximity_score = max(0.0, 1.0 - error_um / tolerance_um)

    # 3. Astrophysical prior: common MIR lines get higher weight
    prior_score = 0.8 if matched_species in _COMMON_LINES else 0.5

    # 4. Fit quality: Gaussian deviation from ideal χ²_red = 1
    #    exp(-|χ² - 1|) gives 1.0 for perfect fit, decays for poor fits
    fit_score = float(np.exp(-abs(reduced_chi2 - 1.0)))

    # Combined confidence
    confidence = (
        w_snr * snr_score
        + w_prox * proximity_score
        + w_prior * prior_score
        + w_fit * fit_score
    )

    return ConfidenceResult(
        peak_wavelength=peak_wavelength,
        matched_species=matched_species,
        rest_wavelength=rest_wavelength,
        snr=snr,
        wavelength_error=error_um,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        snr_score=snr_score,
        proximity_score=proximity_score,
        prior_score=prior_score,
        fit_score=fit_score,
    )


def compute_confidence_batch(
    peak_wavelengths: np.ndarray,
    line_matches: list,
    peak_snr: np.ndarray,
    gaussian_fits: list,
    tolerance_um: float = 0.05,
    redshift: float = 0.0,
) -> list[Optional[ConfidenceResult]]:
    """Compute confidence scores for a batch of line identifications.

    Parameters
    ----------
    peak_wavelengths : np.ndarray
        Observed wavelengths of detected peaks (µm).
    line_matches : list
        Line match results from ``identify_lines`` (may contain None).
    peak_snr : np.ndarray
        SNR for each peak.
    gaussian_fits : list
        GaussianFit results for each peak.
    tolerance_um : float
        Matching tolerance.
    redshift : float
        Applied redshift.

    Returns
    -------
    list of (ConfidenceResult or None)
        Confidence for each peak. None if peak was unmatched.
    """
    results = []

    for i, match in enumerate(line_matches):
        if match is None:
            results.append(None)
            continue

        # Get reduced chi² from Gaussian fit if available
        fit = gaussian_fits[i] if i < len(gaussian_fits) else None
        if fit is not None and fit.fit_success:
            # Approximate reduced chi²: (residual_rms / noise)²
            # Since we don't have the exact DOF, use 1.0 as a proxy
            reduced_chi2 = max(0.01, (fit.residual_rms ** 2))
        else:
            reduced_chi2 = 5.0  # Penalise failed fits

        snr_val = float(peak_snr[i]) if i < len(peak_snr) else 1.0

        # Compute rest wavelength accounting for redshift
        rest_wl = match.rest_wavelength_um

        conf = compute_confidence(
            peak_wavelength=match.peak_wavelength_um,
            matched_species=match.matched_species,
            rest_wavelength=rest_wl * (1.0 + redshift),
            snr=snr_val,
            tolerance_um=tolerance_um,
            reduced_chi2=reduced_chi2,
        )
        results.append(conf)

    return results
