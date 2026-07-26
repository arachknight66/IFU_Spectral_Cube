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
    err: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate and subtract the continuum from a 1D spectrum with error propagation.

    Parameters
    ----------
    wavelength : np.ndarray
        1D wavelength axis.
    spectrum : np.ndarray
        1D flux array.
    poly_order : int
        Degree of the polynomial continuum model.
    sigma_clip : float
        Number of standard deviations for iterative clipping.
    max_iterations : int
        Maximum number of sigma-clipping iterations.
    err : np.ndarray, optional
        Standard uncertainty array corresponding to spectrum.

    Returns
    -------
    tuple
        If err is None: (continuum_subtracted, continuum)
        If err is provided: (continuum_subtracted, continuum, processed_err)
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
    if err is not None:
        mask = mask & np.isfinite(err) & (err > 0)

    coeffs = None
    cov = None
    weights = None

    for _ in range(max_iterations):
        if np.sum(mask) < poly_order + 1:
            break

        if err is not None:
            weights = 1.0 / err[mask]
        else:
            weights = None

        # Fit polynomial to unmasked points
        coeffs, cov = np.polyfit(wl_norm[mask], spectrum[mask], poly_order, w=weights, cov=True)
        continuum = np.polyval(coeffs, wl_norm)

        # Compute residuals
        residuals = spectrum - continuum
        if err is not None:
            norm_residuals = residuals / np.maximum(1e-12, err)
            sigma = np.std(norm_residuals[mask])
            if sigma == 0:
                break
            new_mask = mask & (np.abs(norm_residuals) < sigma_clip * max(1.0, sigma))
        else:
            sigma = np.std(residuals[mask])
            if sigma == 0:
                break
            new_mask = mask & (np.abs(residuals) < sigma_clip * sigma)

        if np.array_equal(new_mask, mask):
            break

        mask = new_mask

    # Final fit
    if np.sum(mask) >= poly_order + 1:
        if err is not None:
            weights = 1.0 / err[mask]
        else:
            weights = None
        coeffs, cov = np.polyfit(wl_norm[mask], spectrum[mask], poly_order, w=weights, cov=True)
        continuum = np.polyval(coeffs, wl_norm)
    else:
        continuum = np.full_like(spectrum, np.nanmedian(spectrum))
    
    continuum_subtracted = spectrum - continuum

    if err is not None:
        # Variance of polynomial evaluation: Var(c_p x^p + ...)
        if cov is not None:
            V = np.vander(wl_norm, poly_order + 1)
            # Var(cont_i) = sum_j sum_k V_ij V_ik Cov_jk
            cont_var = np.einsum('ij,jk,ik->i', V, cov, V)
            cont_var = np.maximum(0.0, cont_var)
        else:
            cont_var = np.zeros_like(spectrum)
        
        processed_err = np.sqrt(err ** 2 + cont_var)
        return continuum_subtracted, continuum, processed_err

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
