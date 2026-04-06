"""Spectral preprocessing: continuum subtraction, noise reduction, and defringing.

Scientific design notes
-----------------------
* **Continuum subtraction** uses a wide median filter as a non-parametric
  baseline that suppresses narrow features while following slow dust
  continuum or calibration curvature.  An iterative sigma-clipped variant
  rejects emission peaks before re-estimating to reduce positive bias.

* **Savitzky-Golay (SG) smoothing** performs local polynomial least-squares
  fitting.  For an SG kernel of length *2m+1* and polynomial order *p*, the
  smoothed value at sample *i* is:

      ŷ_i = Σ_{j=-m}^{m} c_j · y_{i+j}

  where *c_j* are the convolution coefficients obtained from a local
  polynomial regression.  Because these coefficients preserve polynomial
  moments up to order *p*, the line centroid and amplitude are unbiased for
  features well-approximated by polynomials of degree ≤ *p* within the
  window.  This is the key advantage over Gaussian convolution, which
  attenuates all frequencies indiscriminately.

* **Gaussian smoothing** is a symmetric low-pass filter whose transfer
  function is H(f) = exp(-2π²σ²f²).  It reduces noise more aggressively
  but broadens and attenuates narrow spectral features.

* **Defringing** targets periodic residuals left by Fabry-Perot etalon
  effects in the MIRI MRS detector.  We use a Lomb-Scargle periodogram to
  identify dominant fringe frequencies and subtract sinusoidal models.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _nearest_odd(value: int) -> int:
    """Round to nearest odd integer (required for symmetric filter kernels)."""
    return value if value % 2 == 1 else value + 1


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

def estimate_noise_sigma(spectrum: np.ndarray) -> float:
    """Robust channel-noise estimate from first differences.

    For uncorrelated channel noise with variance σ²:

        Var(x_i − x_{i−1}) = 2σ²

    Using the MAD (Median Absolute Deviation) instead of standard deviation
    makes the estimator robust against outliers (emission/absorption lines),
    which would positively bias a naive σ estimate.

    The factor 1.4826 converts MAD to Gaussian σ for a normal distribution.
    """
    spec = np.asarray(spectrum, dtype=float)
    finite = np.isfinite(spec)
    if finite.sum() < 4:
        return float("nan")
    diffs = np.diff(spec[finite])
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med))
    return 1.4826 * mad / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Data quality masking
# ---------------------------------------------------------------------------

def apply_dq_mask(
    data: np.ndarray,
    dq: np.ndarray | None,
    bad_dq_flags: int = 1,
) -> np.ndarray:
    """Replace data flagged by the JWST calibration DQ array with NaN.

    The JWST pipeline DQ extension is a bitfield.  By default we mask any
    pixel where bit 0 (DO_NOT_USE) is set.  Pass a bitwise-OR of flags to
    mask additional conditions (e.g., ``bad_dq_flags=1|4`` to also mask
    JUMP_DET).

    Masking *before* any filtering prevents hardware artifacts (hot pixels,
    cosmic-ray residuals) from propagating into continuum estimates,
    smoothing kernels, and peak detection.
    """
    out = np.array(data, dtype=float, copy=True)
    if dq is None:
        return out
    dq_arr = np.asarray(dq)
    if dq_arr.shape != out.shape:
        raise ValueError(
            f"DQ shape {dq_arr.shape} does not match data shape {out.shape}."
        )
    bad = (dq_arr.astype(int) & int(bad_dq_flags)) != 0
    out[bad] = np.nan
    return out


def apply_dq_mask_1d(
    spectrum: np.ndarray,
    dq_spectrum: np.ndarray | None,
    bad_dq_flags: int = 1,
) -> np.ndarray:
    """1-D convenience wrapper for ``apply_dq_mask``."""
    if dq_spectrum is None:
        return np.array(spectrum, dtype=float, copy=True)
    return apply_dq_mask(spectrum, dq_spectrum, bad_dq_flags)


# ---------------------------------------------------------------------------
# Continuum subtraction
# ---------------------------------------------------------------------------

def estimate_continuum_1d(
    spectrum: np.ndarray,
    window_length: int = 101,
) -> np.ndarray:
    """Continuum proxy from a spectral median filter.

    A wide odd kernel suppresses narrow features while following slow
    baseline changes from dust continuum or calibration residuals.
    """
    spec = np.asarray(spectrum, dtype=float).copy()
    window = max(5, _nearest_odd(window_length))
    if window >= spec.size:
        window = _nearest_odd(max(5, spec.size - 1))

    # Handle NaN values: interpolate, filter, restore
    nan_mask = ~np.isfinite(spec)
    if np.any(nan_mask):
        good = np.where(~nan_mask)[0]
        if good.size < window:
            return spec
        spec[nan_mask] = np.interp(np.where(nan_mask)[0], good, spec[good])
        result = median_filter(spec, size=window, mode="nearest")
        result[nan_mask] = np.nan
        return result

    return median_filter(spec, size=window, mode="nearest")


def estimate_continuum_iterative(
    spectrum: np.ndarray,
    window_length: int = 101,
    sigma_clip: float = 3.0,
    n_iter: int = 3,
) -> np.ndarray:
    """Iterative sigma-clipped continuum estimation.

    Standard median-filter continuum can be biased upward by strong
    emission features that fall partly within the kernel.  This routine
    iteratively:

    1. Estimate continuum via median filter.
    2. Compute residuals = spectrum − continuum.
    3. Mask channels where residuals > sigma_clip × MAD-noise.
    4. Interpolate masked channels and repeat.

    This follows the general strategy of IRAF's ``continuum`` task and is
    standard practice in optical/IR spectroscopy.
    """
    spec = np.asarray(spectrum, dtype=float).copy()
    n = spec.size
    mask = np.isfinite(spec)
    x = np.arange(n, dtype=float)

    for _ in range(n_iter):
        # Median-filter the currently-unmasked spectrum
        working = spec.copy()
        if not mask.all():
            # Fill masked channels with linear interpolation for filtering
            working[~mask] = np.interp(x[~mask], x[mask], spec[mask])
        cont = estimate_continuum_1d(working, window_length=window_length)

        residual = spec - cont
        sigma = estimate_noise_sigma(residual[mask]) if mask.sum() > 4 else 1e-30
        if not np.isfinite(sigma) or sigma <= 0:
            break
        # Clip emission above threshold
        emission = residual > sigma_clip * sigma
        mask = mask & ~emission

        if mask.sum() < 10:
            break

    # Final continuum from the cleaned spectrum
    working = spec.copy()
    if not mask.all() and mask.sum() > 2:
        working[~mask] = np.interp(x[~mask], x[mask], spec[mask])
    return estimate_continuum_1d(working, window_length=window_length)


def subtract_continuum_1d(
    spectrum: np.ndarray,
    window_length: int = 101,
    iterative: bool = False,
    sigma_clip: float = 3.0,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract continuum estimate and return (residual, continuum)."""
    if iterative:
        continuum = estimate_continuum_iterative(
            spectrum,
            window_length=window_length,
            sigma_clip=sigma_clip,
            n_iter=n_iter,
        )
    else:
        continuum = estimate_continuum_1d(spectrum, window_length=window_length)
    return spectrum - continuum, continuum


def subtract_continuum_cube(
    cube_data: np.ndarray,
    window_length: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract spectral continuum from each spaxel in a 3-D cube."""
    cube = np.asarray(cube_data, dtype=float)
    window = max(5, _nearest_odd(window_length))
    if window >= cube.shape[0]:
        window = _nearest_odd(max(5, cube.shape[0] - 1))
    continuum = median_filter(cube, size=(window, 1, 1), mode="nearest")
    return cube - continuum, continuum


# ---------------------------------------------------------------------------
# Noise reduction — Savitzky-Golay
# ---------------------------------------------------------------------------

def apply_savgol_smoothing(
    spectrum: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing by local polynomial least-squares.

    Mathematical justification
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    SG filtering fits a polynomial of degree *p* to each window of *2m+1*
    samples via ordinary least squares.  The convolution coefficients c_j
    satisfy:

        Σ_j c_j · j^k = δ_{k0}   for k = 0, 1, …, p

    This means polynomials up to degree *p* pass through the filter
    unchanged: the zeroth moment (total flux), first moment (centroid),
    and higher moments are preserved, which is critical for retaining
    line amplitude and position fidelity.

    Compared with Gaussian convolution, SG better preserves low-order
    moments for narrow spectral features at comparable noise reduction.
    """
    spec = np.asarray(spectrum, dtype=float).copy()
    window = max(polyorder + 2, _nearest_odd(window_length))
    if window >= spec.size:
        window = _nearest_odd(max(polyorder + 2, spec.size - 1))

    # Handle NaN values: interpolate over gaps, filter, restore NaN mask
    nan_mask = ~np.isfinite(spec)
    if np.any(nan_mask):
        good = np.where(~nan_mask)[0]
        if good.size < window:
            return spec  # Too few valid points to filter
        spec[nan_mask] = np.interp(np.where(nan_mask)[0], good, spec[good])
        result = savgol_filter(spec, window_length=window, polyorder=polyorder, mode="interp")
        result[nan_mask] = np.nan
        return result

    return savgol_filter(spec, window_length=window, polyorder=polyorder, mode="interp")


# ---------------------------------------------------------------------------
# Noise reduction — Gaussian
# ---------------------------------------------------------------------------

def apply_gaussian_smoothing(
    spectrum: np.ndarray,
    sigma_channels: float = 1.5,
) -> np.ndarray:
    """Gaussian kernel smoothing (low-pass filter).

    The transfer function is:

        H(f) = exp(−2π² σ² f²)

    This removes high-frequency noise but attenuates all features,
    including narrow emission lines.  Gaussian smoothing is more
    aggressive than SG for the same effective kernel width, making
    it useful for very low-SNR data where detecting faint lines is
    more important than preserving their exact profile.
    """
    spec = np.asarray(spectrum, dtype=float).copy()

    # Handle NaN values
    nan_mask = ~np.isfinite(spec)
    if np.any(nan_mask):
        good = np.where(~nan_mask)[0]
        if good.size < 3:
            return spec
        spec[nan_mask] = np.interp(np.where(nan_mask)[0], good, spec[good])
        result = gaussian_filter1d(spec, sigma=sigma_channels, mode="nearest")
        result[nan_mask] = np.nan
        return result

    return gaussian_filter1d(spec, sigma=sigma_channels, mode="nearest")


# ---------------------------------------------------------------------------
# Defringing
# ---------------------------------------------------------------------------

def defringe_1d(
    spectrum: np.ndarray,
    wavelength_micron: np.ndarray,
    period_range_micron: tuple[float, float] = (0.01, 0.1),
    max_components: int = 3,
    significance_power_ratio: float = 5.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Remove periodic fringe residuals from a 1-D spectrum.

    MIRI MRS spectra contain Fabry-Perot etalon fringes from internal
    reflections in the detector substrate.  The JWST calibration pipeline
    removes the dominant component, but residuals at the ~1-5 % level
    persist in Level-3 cubes and can mimic weak emission lines.

    Method
    ------
    1. Compute Lomb-Scargle periodogram over the specified period range.
    2. Identify up to ``max_components`` peaks whose power exceeds
       ``significance_power_ratio`` × median power.
    3. Subtract best-fit sinusoids at those periods.

    Returns the defringed spectrum and a diagnostics dict.
    """
    from scipy.signal import lombscargle

    wl = np.asarray(wavelength_micron, dtype=float)
    spec = np.asarray(spectrum, dtype=float).copy()
    finite = np.isfinite(spec) & np.isfinite(wl)

    info: dict[str, object] = {"n_components_removed": 0, "periods_micron": [], "amplitudes": []}

    if finite.sum() < 20:
        return spec, info

    x = wl[finite]
    y = spec[finite]
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # Convert period range to angular frequency range
    freq_min = 2.0 * np.pi / period_range_micron[1]
    freq_max = 2.0 * np.pi / period_range_micron[0]
    n_freqs = max(500, int(10 * (freq_max - freq_min) * (x.max() - x.min()) / (2 * np.pi)))
    angular_freqs = np.linspace(freq_min, freq_max, n_freqs)

    periods_removed: list[float] = []
    amplitudes_removed: list[float] = []
    residual = y_centered.copy()

    for _ in range(max_components):
        if residual.size < 20:
            break
        power = lombscargle(x, residual, angular_freqs, normalize=False)
        median_power = np.median(power)
        peak_idx = np.argmax(power)

        if power[peak_idx] < significance_power_ratio * median_power:
            break

        omega_best = angular_freqs[peak_idx]
        period = 2.0 * np.pi / omega_best

        # Fit amplitude and phase: y = A·cos(ωx) + B·sin(ωx)
        cos_term = np.cos(omega_best * x)
        sin_term = np.sin(omega_best * x)
        design = np.column_stack([cos_term, sin_term])
        coeffs, _, _, _ = np.linalg.lstsq(design, residual, rcond=None)
        model = design @ coeffs
        amplitude = float(np.sqrt(coeffs[0] ** 2 + coeffs[1] ** 2))

        residual = residual - model
        periods_removed.append(float(period))
        amplitudes_removed.append(amplitude)

    if periods_removed:
        # Reconstruct full-length defringed spectrum
        total_fringe = np.zeros_like(spec)
        y_full_centered = spec - y_mean
        for period, _ in zip(periods_removed, amplitudes_removed):
            omega = 2.0 * np.pi / period
            cos_t = np.cos(omega * wl)
            sin_t = np.sin(omega * wl)
            design_full = np.column_stack([cos_t, sin_t])
            # Re-fit on finite data only
            design_f = design_full[finite]
            c, _, _, _ = np.linalg.lstsq(design_f, y_full_centered, rcond=None)
            total_fringe += design_full @ c
            y_full_centered = y_full_centered - design_full @ c

        spec = spec - total_fringe

    info["n_components_removed"] = len(periods_removed)
    info["periods_micron"] = periods_removed
    info["amplitudes"] = amplitudes_removed
    return spec, info


# ---------------------------------------------------------------------------
# Smoother comparison
# ---------------------------------------------------------------------------

def compare_smoothing_metrics(
    original: np.ndarray,
    savgol_smoothed: np.ndarray,
    gaussian_smoothed: np.ndarray,
) -> dict[str, float]:
    """Quantitative comparison between smoothing methods.

    Metrics
    -------
    * **Noise reduction factor**: σ_original / σ_smoothed.  Higher means
      more noise suppression.
    * **Absolute area ratio**: ∫|smoothed| / ∫|original|.  Values close
      to 1.0 indicate good flux preservation; values < 1 mean the smoother
      is attenuating real signal.

    These proxies let you quantify the noise-vs-fidelity tradeoff on
    each target without manual inspection.
    """
    n0 = estimate_noise_sigma(original)
    n_sg = estimate_noise_sigma(savgol_smoothed)
    n_gauss = estimate_noise_sigma(gaussian_smoothed)

    area0 = float(np.nansum(np.abs(original)))
    area_sg = float(np.nansum(np.abs(savgol_smoothed)))
    area_gauss = float(np.nansum(np.abs(gaussian_smoothed)))

    return {
        "noise_original": n0,
        "noise_savgol": n_sg,
        "noise_gaussian": n_gauss,
        "noise_reduction_savgol": (n0 / n_sg) if np.isfinite(n0) and n_sg > 0 else float("nan"),
        "noise_reduction_gaussian": (n0 / n_gauss) if np.isfinite(n0) and n_gauss > 0 else float("nan"),
        "abs_area_ratio_savgol": (area_sg / area0) if area0 > 0 else float("nan"),
        "abs_area_ratio_gaussian": (area_gauss / area0) if area0 > 0 else float("nan"),
    }
