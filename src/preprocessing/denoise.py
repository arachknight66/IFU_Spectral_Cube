"""
Spectral Denoising: Savitzky-Golay and Gaussian Filtering.

Scientific Rationale
--------------------
The Savitzky-Golay (SG) filter is preferred over Gaussian smoothing for
spectral data because it performs a local polynomial least-squares fit
within a sliding window. This has two critical advantages for IFU spectra:

1. **Moment preservation**: An SG filter of polynomial order *p* preserves
   the 0th through *p*-th statistical moments of spectral features. A
   Gaussian kernel, by contrast, systematically broadens all features by
   adding the kernel σ in quadrature to the intrinsic line width:
       σ²_observed = σ²_intrinsic + σ²_kernel.

2. **Edge fidelity**: Gaussian smoothing attenuates sharp edges (e.g.,
   absorption band edges, ice features at 6.0 µm and 15.2 µm). The SG
   polynomial fit adapts to the local curvature, retaining edge structure.

Mathematical foundation
-----------------------
The SG filter computes smoothed values via convolution with pre-computed
coefficients c_j derived from the normal equations of least-squares:

    ŷ_i = Σ_{j=-k}^{k} c_j · y_{i+j}

where k = (window_length - 1) / 2.  The coefficients depend only on window
size and polynomial order, not on the data.  This makes the filter:
    - O(N) in computation (convolution, not per-point fitting)
    - Exactly equivalent to evaluating the fitted polynomial at the centre

For MIRI MRS data with spectral resolving power R = λ/Δλ ~ 1500–3700,
emission lines are only a few channels wide.  Gaussian convolution with
FWHM comparable to the line-spread function degrades detection
significance by ~30–40%, while SG filtering with polyorder=3 preserves
>95% of peak amplitude for lines wider than half the filter window.

References
----------
- Savitzky & Golay (1964), Analytical Chemistry 36(8), 1627–1639
- Press et al., Numerical Recipes, §14.8
- Steinier et al. (1972), Analytical Chemistry 44(11), 1906–1909
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# =========================================================================
# Primary filter: Savitzky-Golay
# =========================================================================

def savgol_denoise(
    spectrum: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a 1D spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        1D flux array to be smoothed.
    window_length : int
        Length of the filter window (must be odd and > polyorder).
        Controls the trade-off between noise suppression and feature
        preservation: larger windows suppress more noise but may
        attenuate narrow features.
    polyorder : int
        Order of the polynomial fit within each window. Higher orders
        preserve more spectral detail but suppress less noise.
        Recommended: 2 or 3 for emission line work.

    Returns
    -------
    np.ndarray
        Smoothed spectrum of the same length.

    Raises
    ------
    ValueError
        If window_length is even, or polyorder >= window_length.

    Notes
    -----
    For very short spectra (< window_length), the function falls back
    to returning the input unchanged with a warning.
    """
    if len(spectrum) < window_length:
        warnings.warn(
            f"Spectrum length ({len(spectrum)}) < window_length "
            f"({window_length}). Returning unsmoothed spectrum.",
            stacklevel=2,
        )
        return spectrum.copy()

    return savgol_filter(
        spectrum,
        window_length=window_length,
        polyorder=polyorder,
        mode="nearest",  # Handle boundaries by extending nearest value
    )


# =========================================================================
# Comparison filter: Gaussian
# =========================================================================

def gaussian_denoise(
    spectrum: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Apply Gaussian smoothing to a 1D spectrum.

    Convolves the spectrum with a Gaussian kernel of standard deviation
    σ (in channels).  Provided for comparison with Savitzky-Golay.

    **Trade-off vs. Savitzky-Golay:**
    - Better noise suppression at equal kernel size
    - But broadens ALL spectral features: σ²_obs = σ²_true + σ²_kernel
    - Reduces peak amplitude: A_obs = A_true × σ_true / σ_obs
    - Not suitable when line profile parameters (FWHM, flux) matter

    Parameters
    ----------
    spectrum : np.ndarray
        1D flux array.
    sigma : float
        Standard deviation of the Gaussian kernel in channels.
        Equivalent FWHM = 2.3548 × sigma channels.

    Returns
    -------
    np.ndarray
        Smoothed spectrum.
    """
    return gaussian_filter1d(spectrum, sigma=sigma, mode="nearest")


# =========================================================================
# Cube-level operations
# =========================================================================

def savgol_denoise_cube(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay denoising to every spaxel in a 3D cube.

    Iterates over all (y, x) positions and smooths each spectrum
    along the wavelength axis independently.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_wavelength, n_y, n_x).
    window_length : int
        SG filter window length.
    polyorder : int
        SG polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed cube of the same shape.
    """
    smoothed = np.empty_like(data)
    n_wl, ny, nx = data.shape

    for iy in range(ny):
        for ix in range(nx):
            smoothed[:, iy, ix] = savgol_denoise(
                data[:, iy, ix],
                window_length=window_length,
                polyorder=polyorder,
            )

    return smoothed


def gaussian_denoise_cube(
    data: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Apply Gaussian smoothing to every spaxel in a 3D cube.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_wavelength, n_y, n_x).
    sigma : float
        Gaussian kernel sigma (channels).

    Returns
    -------
    np.ndarray
        Smoothed cube.
    """
    smoothed = np.empty_like(data)
    n_wl, ny, nx = data.shape

    for iy in range(ny):
        for ix in range(nx):
            smoothed[:, iy, ix] = gaussian_denoise(
                data[:, iy, ix],
                sigma=sigma,
            )

    return smoothed


# =========================================================================
# Filter comparison utility
# =========================================================================

def compare_filters(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
    gaussian_sigma: float = 2.0,
) -> dict[str, np.ndarray | float]:
    """Compare Savitzky-Golay and Gaussian smoothing on a spectrum.

    Returns a dictionary of results including smoothed spectra,
    residuals, and diagnostic metrics for each filter:
    - RMS of residuals (noise removed)
    - Peak amplitude preservation ratio (max of smoothed / max of raw)

    Parameters
    ----------
    spectrum : np.ndarray
        Raw 1D spectrum.
    wavelength : np.ndarray
        Wavelength axis (µm).
    savgol_window : int
        SG window length.
    savgol_polyorder : int
        SG polynomial order.
    gaussian_sigma : float
        Gaussian kernel sigma.

    Returns
    -------
    dict
        Keys: 'wavelength', 'raw', 'savgol', 'gaussian',
              'residual_savgol', 'residual_gaussian',
              'rms_savgol', 'rms_gaussian',
              'peak_ratio_savgol', 'peak_ratio_gaussian'
    """
    sg = savgol_denoise(spectrum, savgol_window, savgol_polyorder)
    gs = gaussian_denoise(spectrum, gaussian_sigma)

    res_sg = spectrum - sg
    res_gs = spectrum - gs

    raw_max = np.max(np.abs(spectrum))
    if raw_max == 0:
        raw_max = 1.0

    return {
        "wavelength": wavelength,
        "raw": spectrum,
        "savgol": sg,
        "gaussian": gs,
        "residual_savgol": res_sg,
        "residual_gaussian": res_gs,
        "rms_savgol": float(np.std(res_sg)),
        "rms_gaussian": float(np.std(res_gs)),
        "peak_ratio_savgol": float(np.max(np.abs(sg)) / raw_max),
        "peak_ratio_gaussian": float(np.max(np.abs(gs)) / raw_max),
    }
