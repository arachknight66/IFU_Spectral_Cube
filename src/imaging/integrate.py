"""
Band-Integrated Image Generation.

Creates a 2D image by summing flux using physical Δλ bin widths across a specified
wavelength range.
"""

from __future__ import annotations

import numpy as np

from ..core.cube import SpectralCube
from .component_map import extract_component_map, WavelengthCoverageError


def integrated_image(
    cube: SpectralCube,
    wl_start: float,
    wl_end: float,
    method: str = "sum",
) -> np.ndarray:
    """Generate a band-integrated 2D image using physical Δλ bin widths.

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    wl_start : float
        Start of wavelength range (µm, inclusive).
    wl_end : float
        End of wavelength range (µm, inclusive).
    method : str
        'sum' for total flux or 'mean' for average flux per channel.

    Returns
    -------
    np.ndarray
        2D image of shape (n_y, n_x).
    """
    wl_center = (wl_start + wl_end) / 2.0
    width = wl_end - wl_start
    if width <= 0:
        width = 0.001

    try:
        cmap = extract_component_map(
            cube,
            central_wavelength_um=wl_center,
            integration_width_um=width,
            feature_name="integrated_band",
            continuum_subtraction=False,
            allow_one_sided_continuum=True,
        )
        if method == "mean":
            # Divide by total band width to get mean intensity
            return cmap.data / width
        return cmap.data
    except WavelengthCoverageError:
        return cube.integrate_band(wl_start, wl_end, method=method)
