"""
Monochromatic Wavelength Slice Extraction.

Extracts 2D spatial images at specific wavelength channels from the
IFU data cube — essentially "narrowband imaging" at each spectral
channel of the cube.
"""

from __future__ import annotations

import numpy as np

from ..core.cube import SpectralCube


def wavelength_slice(cube: SpectralCube, index: int) -> np.ndarray:
    """Extract 2D spatial image at a given channel index.

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    index : int
        Channel index (0-based).

    Returns
    -------
    np.ndarray
        2D image of shape (n_y, n_x).
    """
    return cube.get_slice(index)


def wavelength_slice_at(
    cube: SpectralCube,
    target_wavelength: float,
) -> tuple[np.ndarray, int, float]:
    """Extract 2D image at the channel nearest to a target wavelength.

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    target_wavelength : float
        Target wavelength in µm.

    Returns
    -------
    tuple of (image, index, actual_wavelength)
        - image : 2D array (n_y, n_x)
        - index : channel index used
        - actual_wavelength : actual wavelength of that channel (µm)
    """
    index = cube.find_nearest_channel(target_wavelength)
    actual_wl = cube.get_wavelength_at(index)
    image = cube.get_slice(index)
    return image, index, actual_wl
