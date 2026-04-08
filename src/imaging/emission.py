"""
Emission Line Map Generation.

Creates continuum-subtracted narrowband emission maps by:
    1. Integrating flux in a narrow band centered on the line ("on-band")
    2. Estimating the continuum from adjacent wavelength windows ("off-band")
    3. Subtracting the scaled continuum from the on-band

This is the standard technique for isolating emission line flux in
IFU data, equivalent to narrowband imaging with continuum subtraction
in direct imaging.
"""

from __future__ import annotations

import numpy as np

from ..core.cube import SpectralCube


def emission_line_map(
    cube: SpectralCube,
    line_center: float,
    line_width: float = 0.1,
    continuum_width: float = 0.2,
    continuum_offset: float | None = None,
) -> np.ndarray:
    """Generate a continuum-subtracted emission line map.

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    line_center : float
        Central wavelength of the emission line (µm).
    line_width : float
        Full width of the on-band window (µm). The on-band spans
        [line_center - line_width/2, line_center + line_width/2].
    continuum_width : float
        Width of each continuum estimation window (µm).
    continuum_offset : float or None
        Offset from line center to continuum windows. If None,
        defaults to line_width (continuum windows abut the on-band).

    Returns
    -------
    np.ndarray
        2D continuum-subtracted emission line map (n_y, n_x).

    Notes
    -----
    Continuum windows are placed symmetrically:
        Blue:  [line_center - offset - cont_width, line_center - offset]
        Red:   [line_center + offset, line_center + offset + cont_width]

    If either continuum window falls outside the cube's wavelength
    range, only the available window is used.
    """
    if continuum_offset is None:
        continuum_offset = line_width

    half_line = line_width / 2.0

    # On-band: centered on the line
    on_start = line_center - half_line
    on_end = line_center + half_line

    # Off-band windows
    blue_start = line_center - continuum_offset - continuum_width
    blue_end = line_center - continuum_offset
    red_start = line_center + continuum_offset
    red_end = line_center + continuum_offset + continuum_width

    wl_min, wl_max = cube.wavelength_range

    # Compute on-band image
    try:
        on_band = cube.integrate_band(on_start, on_end, method="mean")
    except ValueError:
        # No channels in on-band — return zeros
        return np.zeros(cube.spatial_shape)

    # Compute continuum from available off-band windows
    continuum_images = []

    if blue_start >= wl_min:
        try:
            blue_cont = cube.integrate_band(blue_start, blue_end, method="mean")
            continuum_images.append(blue_cont)
        except ValueError:
            pass

    if red_end <= wl_max:
        try:
            red_cont = cube.integrate_band(red_start, red_end, method="mean")
            continuum_images.append(red_cont)
        except ValueError:
            pass

    if len(continuum_images) == 0:
        # No continuum estimate possible — return on-band as-is
        return on_band
    elif len(continuum_images) == 1:
        continuum = continuum_images[0]
    else:
        continuum = np.nanmean(continuum_images, axis=0)

    # Scale continuum: the on-band has a certain number of channels,
    # but since we're using 'mean', the continuum average directly
    # estimates the per-channel continuum level
    emission_map = on_band - continuum

    return emission_map
