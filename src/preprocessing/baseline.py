"""
Continuum (Baseline) Subtraction for Spectral Analysis.

In mid-infrared IFU spectroscopy, the observed flux at each spaxel is a
superposition of:
    1. Continuum emission (thermal dust, stellar photospheres)
    2. Spectral features (emission lines, absorption bands, PAH features)

To isolate spectral features for peak detection and line identification,
we must first estimate and subtract the continuum. This module implements
an iterative sigma-clipping polynomial fit:

Algorithm
---------
    1. Fit a polynomial of order *n* to the full spectrum.
    2. Compute residuals = spectrum − fit.
    3. Identify points with residuals > k·σ (emission features) or
       < −k·σ (absorption features).
    4. Mask those points and re-fit the polynomial to the remaining data.
    5. Iterate until convergence (mask unchanged) or max_iterations reached.

This approach prevents emission lines from biasing the continuum estimate
upward, which would cause under-subtraction and missed detections.
"""

from __future__ import annotations

import numpy as np


def subtract_continuum(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    poly_order: int = 3,
    sigma_clip: float = 3.0,
    max_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and subtract the continuum from a 1D spectrum.

    Parameters
    ----------
    wavelength : np.ndarray
        1D wavelength axis (used as the independent variable for fitting).
    spectrum : np.ndarray
        1D flux array.
    poly_order : int
        Degree of the polynomial continuum model. Order 1–2 for smooth
        continua (e.g., evolved stars); order 3–5 for complex shapes
        (embedded protostars, AGN).
    sigma_clip : float
        Number of standard deviations for iterative clipping. Higher
        values retain more points but risk fitting emission features
        into the continuum.
    max_iterations : int
        Maximum number of sigma-clipping iterations.

    Returns
    -------
    tuple of (continuum_subtracted, continuum)
        - continuum_subtracted : spectrum with continuum removed
        - continuum : the estimated continuum model

    Notes
    -----
    The wavelength axis is normalized to [0, 1] before fitting to
    improve numerical conditioning of the polynomial fit, especially
    at high orders.
    """
    # Normalize wavelength for numerical stability
    wl_min, wl_max = wavelength.min(), wavelength.max()
    wl_range = wl_max - wl_min
    if wl_range == 0:
        wl_norm = np.zeros_like(wavelength)
    else:
        wl_norm = (wavelength - wl_min) / wl_range

    # Initialize mask: True = include in fit
    mask = np.isfinite(spectrum)

    for _ in range(max_iterations):
        if np.sum(mask) < poly_order + 1:
            # Not enough points for fit — return zeros
            break

        # Fit polynomial to unmasked points
        coeffs = np.polyfit(wl_norm[mask], spectrum[mask], poly_order)
        continuum = np.polyval(coeffs, wl_norm)

        # Compute residuals
        residuals = spectrum - continuum
        sigma = np.std(residuals[mask])

        if sigma == 0:
            break

        # Update mask: clip points with |residual| > sigma_clip * sigma
        new_mask = mask & (np.abs(residuals) < sigma_clip * sigma)

        # Check convergence
        if np.array_equal(new_mask, mask):
            break

        mask = new_mask

    # Final fit
    if np.sum(mask) >= poly_order + 1:
        coeffs = np.polyfit(wl_norm[mask], spectrum[mask], poly_order)
        continuum = np.polyval(coeffs, wl_norm)
    else:
        # Fallback: use median as flat continuum
        continuum = np.full_like(spectrum, np.nanmedian(spectrum))
    
    continuum_subtracted = spectrum - continuum

    return continuum_subtracted, continuum


def subtract_continuum_cube(
    wavelength: np.ndarray,
    data: np.ndarray,
    poly_order: int = 3,
    sigma_clip: float = 3.0,
    max_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract continuum from every spaxel in a 3D cube.

    Parameters
    ----------
    wavelength : np.ndarray
        1D wavelength axis.
    data : np.ndarray
        3D cube (n_wavelength, n_y, n_x).
    poly_order : int
        Polynomial order for continuum.
    sigma_clip : float
        Sigma clipping threshold.
    max_iterations : int
        Maximum iterations for convergence.

    Returns
    -------
    tuple of (subtracted_cube, continuum_cube)
    """
    n_wl, ny, nx = data.shape
    subtracted = np.empty_like(data)
    continuum = np.empty_like(data)

    for iy in range(ny):
        for ix in range(nx):
            sub, cont = subtract_continuum(
                wavelength,
                data[:, iy, ix],
                poly_order=poly_order,
                sigma_clip=sigma_clip,
                max_iterations=max_iterations,
            )
            subtracted[:, iy, ix] = sub
            continuum[:, iy, ix] = cont

    return subtracted, continuum
