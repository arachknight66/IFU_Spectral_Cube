"""
Band-Integrated Image Generation.

Creates a 2D image by summing or averaging flux across a specified
wavelength range. This is analogous to photometric imaging through
a broadband filter, but with user-defined bandpass.
"""

from __future__ import annotations

import numpy as np

from ..core.cube import SpectralCube


def integrated_image(
    cube: SpectralCube,
    wl_start: float,
    wl_end: float,
    method: str = "sum",
) -> np.ndarray:
    """Generate a band-integrated 2D image.

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
    return cube.integrate_band(wl_start, wl_end, method=method)
