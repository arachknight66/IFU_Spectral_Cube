"""
Emission-Line Peak Detection for IFU Spectra.

This module wraps ``scipy.signal.find_peaks`` with parameter choices
motivated by the physics of mid-infrared spectroscopy:

Parameter justification
-----------------------
* **prominence** (primary discriminator):
    Measures how far a peak rises above the *local* baseline formed by
    the higher of its two neighbouring troughs.  Unlike ``height``, which
    is measured from the global y=0 axis, prominence is insensitive to
    residual continuum offsets after imperfect subtraction.  We default to
    3× the noise RMS — the classical 3-sigma detection threshold.

* **width** (secondary filter):
    MIRI MRS spectral resolving power R = λ/Δλ ~ 1500–3700 means an
    unresolved line spans ~2–4 spectral pixels at Nyquist sampling.
    Features narrower than 2 channels are almost certainly cosmic-ray
    hits or hot-pixel artifacts; features wider than 30 channels are
    broad blends (PAH bands, instrumental artefacts).

* **distance** (duplicate guard):
    Two real emission lines cannot be closer than 1 resolution element
    (R ≈ 2000 → Δλ/λ = 5×10⁻⁴ → ~3 pixels at typical sampling).
    Setting distance ≥ 3 prevents splitting a single line into
    multiple detections due to noise substructure.

* **height** (optional):
    Absolute flux threshold.  Useful when the noise level varies across
    the band (e.g., higher thermal background at λ > 20 µm).  We leave
    it ``None`` by default and let prominence do the work.

References
----------
- Virtanen et al. (2020) SciPy 1.0 — scipy.signal.find_peaks
- Rieke et al. (2015) PASP 127, 584 — MIRI detector characteristics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, peak_widths


@dataclass
class PeakResult:
    """Container for detected emission-line peaks.

    Attributes
    ----------
    indices : np.ndarray
        Channel indices of detected peaks.
    wavelengths : np.ndarray
        Wavelengths of detected peaks (µm).
    prominences : np.ndarray
        Peak prominences (flux units).
    widths : np.ndarray
        Peak widths measured at half-prominence (in channels).
    widths_um : np.ndarray
        Peak widths converted to wavelength units (µm).
    heights : np.ndarray
        Peak heights (flux at peak position).
    snr : np.ndarray
        Signal-to-noise ratio of each peak (prominence / noise_rms).
    noise_rms : float
        Estimated noise RMS used for detection.
    n_peaks : int
        Number of peaks detected.
    """
    indices: np.ndarray
    wavelengths: np.ndarray
    prominences: np.ndarray
    widths: np.ndarray
    widths_um: np.ndarray
    heights: np.ndarray
    snr: np.ndarray
    noise_rms: float
    n_peaks: int


def estimate_noise_rms(spectrum: np.ndarray) -> float:
    """Estimate the noise RMS of a continuum-subtracted spectrum.

    Uses the Median Absolute Deviation (MAD) estimator, which is robust
    against the presence of emission lines (outliers).  Under the
    assumption that the noise is Gaussian, σ ≈ 1.4826 × MAD.

    This is preferable to np.std() because std is biased high when the
    spectrum contains real emission features — the very things we're
    trying to detect.

    Parameters
    ----------
    spectrum : np.ndarray
        1D continuum-subtracted flux array.

    Returns
    -------
    float
        Estimated noise RMS.
    """
    finite = spectrum[np.isfinite(spectrum)]
    if len(finite) == 0:
        return 1.0
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    # 1.4826 converts MAD to σ for a Gaussian distribution
    sigma = 1.4826 * mad
    return max(sigma, 1e-30)  # Prevent division by zero


def detect_peaks(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    noise_rms: Optional[float] = None,
    sigma_threshold: float = 3.0,
    min_width: int = 2,
    max_width: int = 30,
    min_distance: int = 3,
    height: Optional[float] = None,
) -> PeakResult:
    """Detect emission-line peaks in a 1D spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        1D continuum-subtracted flux array.
    wavelength : np.ndarray
        Corresponding wavelength axis (µm).
    noise_rms : float, optional
        Noise RMS for prominence threshold.  If None, estimated
        automatically using the MAD method.
    sigma_threshold : float
        Detection threshold in units of noise_rms.  Default 3.0
        gives a false-positive rate of ~0.3% per channel for
        Gaussian noise (1 - erf(3/√2) ≈ 0.0027).
    min_width : int
        Minimum peak width in channels.  Peaks narrower than this
        are likely artifacts (cosmic rays, hot pixels).
    max_width : int
        Maximum peak width in channels.  Peaks broader than this
        may be PAH features or blended complexes.
    min_distance : int
        Minimum separation between peaks (channels).  Should be ≥
        the instrumental resolution element.
    height : float, optional
        Minimum absolute flux at peak position.  None disables.

    Returns
    -------
    PeakResult
        Structured result with peak locations and properties.
    """
    if noise_rms is None:
        noise_rms = estimate_noise_rms(spectrum)

    prominence_threshold = sigma_threshold * noise_rms

    # Run scipy's peak finder with combined constraints
    indices, properties = find_peaks(
        spectrum,
        height=height,
        prominence=prominence_threshold,
        distance=min_distance,
        width=(min_width, max_width),
    )

    if len(indices) == 0:
        return PeakResult(
            indices=np.array([], dtype=int),
            wavelengths=np.array([]),
            prominences=np.array([]),
            widths=np.array([]),
            widths_um=np.array([]),
            heights=np.array([]),
            snr=np.array([]),
            noise_rms=noise_rms,
            n_peaks=0,
        )

    # Extract properties
    prominences = properties["prominences"]
    widths_channels = properties["widths"]
    heights_arr = spectrum[indices]

    # Convert widths from channels to wavelength units
    # Use the local spectral sampling (Δλ per channel)
    delta_lambda = np.gradient(wavelength)
    widths_um = widths_channels * delta_lambda[indices]

    # Compute per-peak SNR
    snr = prominences / noise_rms

    return PeakResult(
        indices=indices,
        wavelengths=wavelength[indices],
        prominences=prominences,
        widths=widths_channels,
        widths_um=np.abs(widths_um),
        heights=heights_arr,
        snr=snr,
        noise_rms=noise_rms,
        n_peaks=len(indices),
    )


def adaptive_detect(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    sigma_levels: tuple[float, ...] = (5.0, 4.0, 3.0),
    min_width: int = 2,
    max_width: int = 30,
    min_distance: int = 3,
) -> PeakResult:
    """Multi-pass adaptive peak detection.

    Runs detection at progressively lower sigma thresholds.  Peaks
    found at higher sigma (more confident) are kept unconditionally;
    peaks found at lower sigma are kept only if they don't overlap
    with previously detected peaks.

    This hierarchical strategy improves completeness for faint lines
    without increasing false positives from noise spikes that happen
    to sit near brighter features.

    Parameters
    ----------
    spectrum : np.ndarray
        1D continuum-subtracted spectrum.
    wavelength : np.ndarray
        Wavelength axis (µm).
    sigma_levels : tuple of float
        Detection thresholds to try, from most to least stringent.
    min_width, max_width, min_distance : int
        Standard peak-finding constraints (see ``detect_peaks``).

    Returns
    -------
    PeakResult
        Combined result from all passes.
    """
    noise_rms = estimate_noise_rms(spectrum)
    all_indices = set()

    for sigma in sorted(sigma_levels, reverse=True):
        result = detect_peaks(
            spectrum, wavelength,
            noise_rms=noise_rms,
            sigma_threshold=sigma,
            min_width=min_width,
            max_width=max_width,
            min_distance=min_distance,
        )
        for idx in result.indices:
            # Only add if not too close to an already-detected peak
            if all(abs(idx - existing) >= min_distance for existing in all_indices):
                all_indices.add(idx)

    # Build final result from merged indices
    if len(all_indices) == 0:
        return PeakResult(
            indices=np.array([], dtype=int),
            wavelengths=np.array([]),
            prominences=np.array([]),
            widths=np.array([]),
            widths_um=np.array([]),
            heights=np.array([]),
            snr=np.array([]),
            noise_rms=noise_rms,
            n_peaks=0,
        )

    final_indices = np.sort(np.array(list(all_indices), dtype=int))

    # Recompute properties for final peak set
    prominences_arr = np.array([
        spectrum[i] - min(
            np.min(spectrum[max(0, i - 15):i]) if i > 0 else spectrum[i],
            np.min(spectrum[i + 1:min(len(spectrum), i + 16)]) if i < len(spectrum) - 1 else spectrum[i],
        )
        for i in final_indices
    ])

    # Estimate widths via half-prominence interpolation
    try:
        w_results = peak_widths(spectrum, final_indices, rel_height=0.5)
        widths_channels = w_results[0]
    except Exception:
        widths_channels = np.full(len(final_indices), 3.0)

    delta_lambda = np.gradient(wavelength)
    widths_um = np.abs(widths_channels * delta_lambda[final_indices])

    return PeakResult(
        indices=final_indices,
        wavelengths=wavelength[final_indices],
        prominences=prominences_arr,
        widths=widths_channels,
        widths_um=widths_um,
        heights=spectrum[final_indices],
        snr=prominences_arr / noise_rms,
        noise_rms=noise_rms,
        n_peaks=len(final_indices),
    )
