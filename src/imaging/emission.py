"""
Emission Line Map Generation.

Creates continuum-subtracted narrowband emission maps by integrating flux
and estimating local continuum, leveraging the Phase 5 ComponentMap engine.
"""

from __future__ import annotations

import numpy as np

from ..core.cube import SpectralCube
from .component_map import extract_component_map, WavelengthCoverageError


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
        Full width of the on-band window (µm).
    continuum_width : float
        Width of each continuum estimation window (µm).
    continuum_offset : float or None
        Offset from line center to continuum windows.

    Returns
    -------
    np.ndarray
        2D continuum-subtracted emission line map (n_y, n_x).
    """
    if continuum_offset is None:
        continuum_offset = line_width

    gap_um = continuum_offset

    try:
        cmap = extract_component_map(
            cube,
            central_wavelength_um=line_center,
            integration_width_um=line_width,
            feature_name=f"emission_{line_center:.3f}um",
            continuum_subtraction={
                "enabled": True,
                "sideband_width_um": continuum_width,
                "gap_um": gap_um,
                "method": "adjacent_sidebands",
            },
            allow_one_sided_continuum=True,
        )
        return cmap.data
    except WavelengthCoverageError:
        return np.zeros(cube.spatial_shape, dtype=float)
