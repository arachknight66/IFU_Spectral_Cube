"""
Spectrum Extraction Utilities.

Provides functions to extract spectra from the SpectralCube with
optional preprocessing (denoising + continuum subtraction) applied
as part of the extraction workflow.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.cube import SpectralCube
from ..preprocessing.denoise import savgol_denoise
from ..preprocessing.baseline import subtract_continuum
from ..utils.config import PipelineConfig


def extract_spectrum(
    cube: SpectralCube,
    x: int,
    y: int,
    config: Optional[PipelineConfig] = None,
    preprocess: bool = True,
) -> dict[str, np.ndarray]:
    """Extract a single-spaxel spectrum with optional preprocessing.

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    x, y : int
        Pixel coordinates.
    config : PipelineConfig, optional
        Configuration for preprocessing parameters. Uses defaults if None.
    preprocess : bool
        If True, apply denoising and continuum subtraction.

    Returns
    -------
    dict with keys:
        'wavelength' : np.ndarray — wavelength axis
        'raw' : np.ndarray — raw extracted spectrum
        'denoised' : np.ndarray — after SG filtering (if preprocessed)
        'continuum' : np.ndarray — estimated continuum (if preprocessed)
        'processed' : np.ndarray — final continuum-subtracted spectrum
    """
    if config is None:
        config = PipelineConfig()

    raw = cube.get_spectrum(x, y)
    raw_err = cube.get_spectrum_err(x, y)
    wavelength = cube.wavelength.copy()

    if raw_err is None:
        # Fallback noise estimate using MAD
        mad = np.nanmedian(np.abs(raw - np.nanmedian(raw)))
        raw_err = np.full_like(raw, max(1e-12, 1.4826 * mad))

    result = {
        "wavelength": wavelength,
        "raw": raw,
        "raw_err": raw_err,
    }

    if preprocess:
        # Step 1: Denoise with error propagation
        denoised, denoised_err = savgol_denoise(
            raw,
            window_length=config.savgol_window,
            polyorder=config.savgol_polyorder,
            err=raw_err,
        )
        result["denoised"] = denoised
        result["denoised_err"] = denoised_err

        # Step 2: Continuum subtraction with error propagation
        continuum_sub, continuum, processed_err = subtract_continuum(
            wavelength,
            denoised,
            poly_order=config.continuum_poly_order,
            sigma_clip=config.sigma_clip,
            err=denoised_err,
        )
        result["continuum"] = continuum
        result["processed"] = continuum_sub
        result["processed_err"] = processed_err
    else:
        result["denoised"] = raw.copy()
        result["denoised_err"] = raw_err.copy()
        result["continuum"] = np.zeros_like(raw)
        result["processed"] = raw.copy()
        result["processed_err"] = raw_err.copy()

    return result


def extract_region_average(
    cube: SpectralCube,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    config: Optional[PipelineConfig] = None,
    preprocess: bool = True,
) -> dict[str, np.ndarray]:
    """Extract a spatially averaged spectrum over a rectangular region.

    Useful for increasing S/N by averaging over multiple spaxels in a
    spatially coherent region (e.g., a knot of emission).

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    x_range : tuple of (x_start, x_end)
        Inclusive x-pixel range.
    y_range : tuple of (y_start, y_end)
        Inclusive y-pixel range.
    config : PipelineConfig, optional
        Configuration parameters.
    preprocess : bool
        Whether to apply preprocessing.

    Returns
    -------
    dict
        Same structure as extract_spectrum, but spatially averaged.
    """
    if config is None:
        config = PipelineConfig()

    x_start, x_end = x_range
    y_start, y_end = y_range

    # Validate ranges
    ny, nx = cube.spatial_shape
    x_start = max(0, x_start)
    x_end = min(nx - 1, x_end)
    y_start = max(0, y_start)
    y_end = min(ny - 1, y_end)

    # Extract sub-cube and average spatially
    sub_cube = cube.data[:, y_start:y_end + 1, x_start:x_end + 1]
    raw = np.nanmean(sub_cube, axis=(1, 2))
    wavelength = cube.wavelength.copy()

    result = {
        "wavelength": wavelength,
        "raw": raw,
    }

    if preprocess:
        denoised = savgol_denoise(
            raw,
            window_length=config.savgol_window,
            polyorder=config.savgol_polyorder,
        )
        result["denoised"] = denoised

        continuum_sub, continuum = subtract_continuum(
            wavelength,
            denoised,
            poly_order=config.continuum_poly_order,
            sigma_clip=config.sigma_clip,
        )
        result["continuum"] = continuum
        result["processed"] = continuum_sub
    else:
        result["denoised"] = raw.copy()
        result["continuum"] = np.zeros_like(raw)
        result["processed"] = raw.copy()

    return result
