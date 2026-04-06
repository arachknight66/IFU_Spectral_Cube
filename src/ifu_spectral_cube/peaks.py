"""Peak detection in 1-D extracted spectra.

Two complementary algorithms are provided:

1. **``detect_peaks_1d``** — Wrapper around ``scipy.signal.find_peaks`` with
   prominence- and width-based thresholds calibrated against robust channel
   noise (MAD estimator).  Best for isolated, well-resolved lines.

2. **``detect_peaks_cwt``** — Continuous wavelet transform (CWT) peak finder
   (``scipy.signal.find_peaks_cwt``).  CWT searches across a range of
   scales simultaneously, making it more robust for blended features or
   spectra with varying line widths across the bandpass.

Both methods can be combined via ``merge_peak_lists`` to reduce missed
detections while controlling false positives.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths

from .models import Peak
from .preprocessing import estimate_noise_sigma


# ---------------------------------------------------------------------------
# Instrument-aware parameter suggestion
# ---------------------------------------------------------------------------

def suggest_miri_peak_params(
    wavelength_micron: np.ndarray,
    resolving_power: float = 2500.0,
    min_snr: float = 4.5,
) -> dict[str, float | tuple[float, float]]:
    """Derive default peak-detection thresholds from MIRI instrument parameters.

    The MIRI MRS resolving power *R* sets the instrumental FWHM:

        FWHM_inst = λ / R

    which translates to a width in channel-units via the local dispersion
    Δλ.  Prominence and distance thresholds are then scaled to reject
    instrumental artifacts and noise spikes.
    """
    wl = np.asarray(wavelength_micron, dtype=float)
    dlam = float(np.nanmedian(np.diff(wl)))
    lam_ref = float(np.nanmedian(wl))
    instrumental_fwhm = lam_ref / resolving_power
    width_samples = max(1.0, instrumental_fwhm / max(dlam, 1e-9))
    return {
        "prominence_sigma": min_snr,
        "height_sigma": 0.0,
        "width": (0.5 * width_samples, 4.0 * width_samples),
        "distance": max(1, int(round(0.75 * width_samples))),
    }


# ---------------------------------------------------------------------------
# Standard peak detection (scipy.signal.find_peaks)
# ---------------------------------------------------------------------------

def detect_peaks_1d(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    prominence_sigma: float = 4.5,
    height_sigma: float = 0.0,
    width: tuple[float, float] | None = None,
    distance: int | None = None,
    noise_sigma: float | None = None,
) -> tuple[list[Peak], dict[str, np.ndarray], float]:
    """Detect significant spectral peaks using SNR-calibrated thresholds.

    Scientific design
    -----------------
    * **Prominence** is the height of a peak relative to its nearest
      enclosing contour line.  It is more robust than raw height in
      sloped or noisy baselines because it measures local contrast.
    * **Width bounds** (in channel units) reject single-channel hot-pixel
      artifacts (too narrow) and overly broad calibration residuals (too
      wide) that are not real spectral lines.
    * **Distance** ensures peaks are separated by at least ~0.75 FWHM_inst,
      preventing a single broad feature from spawning multiple detections.
    """
    wl = np.asarray(wavelength_micron, dtype=float)
    y = np.asarray(flux, dtype=float)
    if wl.shape != y.shape:
        raise ValueError("Wavelength and flux arrays must have same shape.")

    sigma = float(estimate_noise_sigma(y) if noise_sigma is None else noise_sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Unable to estimate a valid noise sigma from spectrum.")

    prominence = prominence_sigma * sigma
    height = height_sigma * sigma if height_sigma > 0 else None

    idx, props = find_peaks(
        y,
        prominence=prominence,
        height=height,
        width=width,
        distance=distance,
    )
    if idx.size == 0:
        return [], props, sigma

    if "widths" not in props:
        props["widths"] = peak_widths(y, idx, rel_height=0.5)[0]

    peaks: list[Peak] = []
    for i, pidx in enumerate(idx):
        prom = float(props["prominences"][i]) if "prominences" in props else float("nan")
        wsamp = float(props["widths"][i]) if "widths" in props else float("nan")
        peaks.append(
            Peak(
                index=int(pidx),
                wavelength_micron=float(wl[pidx]),
                flux=float(y[pidx]),
                prominence=prom,
                width_samples=wsamp,
                snr=float(prom / sigma),
            )
        )
    return peaks, props, sigma


# ---------------------------------------------------------------------------
# CWT peak detection
# ---------------------------------------------------------------------------

def detect_peaks_cwt(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    widths: np.ndarray | None = None,
    min_snr: float = 4.0,
    noise_perc: float = 10.0,
) -> tuple[list[Peak], float]:
    """Detect peaks using continuous wavelet transform.

    The CWT method convolves the spectrum with a family of Ricker (Mexican
    hat) wavelets at different scales.  A peak is reported if it appears
    consistently across multiple scales, making it naturally robust to:

    * **Blended features** — resolved at appropriate scale.
    * **Variable line widths** — unlike fixed-window methods, CWT adapts.
    * **Baseline curvature** — the wavelet's zero-mean property rejects
      slowly-varying continuum residuals.

    The ``widths`` parameter controls the range of scales (in channel
    units) to search.
    """
    wl = np.asarray(wavelength_micron, dtype=float)
    y = np.asarray(flux, dtype=float)

    sigma = estimate_noise_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Unable to estimate noise sigma for CWT detection.")

    if widths is None:
        # Default: search scales from 2 to 20 channels
        widths = np.arange(2, min(21, y.size // 4))

    idx = find_peaks_cwt(y, widths=widths, min_snr=min_snr, noise_perc=noise_perc)

    if len(idx) == 0:
        return [], sigma

    # Compute peak properties using scipy utilities
    w_half = peak_widths(y, idx, rel_height=0.5)[0]

    peaks: list[Peak] = []
    for i, pidx in enumerate(idx):
        if pidx < 0 or pidx >= y.size:
            continue
        flux_val = float(y[pidx])
        # Compute local prominence as peak height above baseline neighbours
        left = max(0, pidx - int(w_half[i]) - 2)
        right = min(y.size, pidx + int(w_half[i]) + 3)
        baseline = np.nanmin(y[left:right])
        prom = flux_val - baseline
        peaks.append(
            Peak(
                index=int(pidx),
                wavelength_micron=float(wl[pidx]),
                flux=flux_val,
                prominence=max(0.0, prom),
                width_samples=float(w_half[i]),
                snr=float(max(0.0, prom) / sigma),
            )
        )

    return peaks, sigma


# ---------------------------------------------------------------------------
# Peak list merging
# ---------------------------------------------------------------------------

def merge_peak_lists(
    peaks_a: list[Peak],
    peaks_b: list[Peak],
    merge_radius_channels: int = 3,
) -> list[Peak]:
    """Merge two peak lists, deduplicating nearby detections.

    If peaks from both lists fall within ``merge_radius_channels`` of each
    other, the one with higher SNR is retained.  This lets you combine
    results from ``detect_peaks_1d`` and ``detect_peaks_cwt`` to catch
    features that one method alone might miss.
    """
    if not peaks_a:
        return list(peaks_b)
    if not peaks_b:
        return list(peaks_a)

    # Start with a copy of peaks_a
    merged = list(peaks_a)
    indices_a = {p.index for p in merged}

    for pb in peaks_b:
        # Check if any existing peak is within merge radius
        matched = False
        for i, pa in enumerate(merged):
            if abs(pa.index - pb.index) <= merge_radius_channels:
                # Keep the higher-SNR detection
                if pb.snr > pa.snr:
                    merged[i] = pb
                matched = True
                break
        if not matched:
            merged.append(pb)

    return sorted(merged, key=lambda p: p.index)


# ---------------------------------------------------------------------------
# Per-peak local SNR refinement
# ---------------------------------------------------------------------------

def refine_peak_local_snr(
    wavelength_micron: np.ndarray,
    flux: np.ndarray,
    peaks: list[Peak],
    local_window_channels: int = 50,
) -> list[Peak]:
    """Recompute SNR for each peak using local noise in a surrounding window.

    The global noise estimate can be misleading when noise varies across
    the bandpass (e.g., MIRI sub-band boundaries, detector persistence).
    This function computes noise in a window around each peak, excluding
    the peak itself, to get a more accurate local SNR.

    The peak's ``local_snr`` field is updated in place and the peak list
    is returned.
    """
    y = np.asarray(flux, dtype=float)
    n = y.size
    half = max(10, local_window_channels // 2)

    for peak in peaks:
        lo = max(0, peak.index - half)
        hi = min(n, peak.index + half + 1)

        # Exclude the peak region itself (±2 FWHM-widths around center)
        excl_half = max(2, int(peak.width_samples * 1.5))
        excl_lo = max(lo, peak.index - excl_half)
        excl_hi = min(hi, peak.index + excl_half + 1)

        local_data = np.concatenate([y[lo:excl_lo], y[excl_hi:hi]])
        local_sigma = estimate_noise_sigma(local_data)

        if np.isfinite(local_sigma) and local_sigma > 0:
            peak.local_snr = float(peak.prominence / local_sigma)
        else:
            peak.local_snr = peak.snr  # fallback to global

    return peaks
