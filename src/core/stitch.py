"""
Spectral Stitching Module.

Combines multiple SpectralCube objects (e.g., from different MIRI channels)
into a single continuous cube.

Handles:
1. Spatial coordinate reprojection (WCS mapping) to a common reference grid.
2. Spectral axis concatenation and monotonic sorting.
"""

from __future__ import annotations

import warnings

import numpy as np
try:
    from astropy.wcs import WCS
except ImportError:
    WCS = None
from scipy.ndimage import map_coordinates

from src.core.cube import SpectralCube


def stitch_cubes(cubes: list[SpectralCube], reference_index: int = 0) -> SpectralCube:
    """Spatially reproject and spectrally stitch multiple cubes.

    Projects all cubes onto the spatial grid (WCS) of the reference cube
    using bilinear interpolation. The spectral channels are concatenated
    and sorted so the resulting wavelength axis is strictly monotonic.

    Parameters
    ----------
    cubes : list of SpectralCube
        The data cubes to stitch.
    reference_index : int
        Index of the cube whose spatial grid and WCS will be used
        as the target grid for all other cubes.

    Returns
    -------
    SpectralCube
        A new combined SpectralCube. Unobserved pixels in projected
        cubes are filled with NaNs.
    """
    if not cubes:
        raise ValueError("Cannot stitch an empty list of cubes.")
    if len(cubes) == 1:
        return cubes[0]

    ref_cube = cubes[reference_index]
    ny, nx = ref_cube.spatial_shape

    try:
        ref_wcs = WCS(ref_cube.header).celestial
        if ref_wcs.pixel_n_dim != 2:
            raise ValueError(f"WCS projection has {ref_wcs.pixel_n_dim} dimensions.")
        use_wcs = True
    except Exception as e:
        warnings.warn(f"Failed to parse celestial WCS from reference cube: {e}. "
                      f"Falling back to 1:1 pixel mapping.")
        use_wcs = False

    if use_wcs:
        # Build grid of coordinates in the reference cube
        x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
        sky_coords = ref_wcs.pixel_to_world(x_grid, y_grid)

    all_wavelengths = []
    all_data = []

    for idx, cube in enumerate(cubes):
        all_wavelengths.append(cube.wavelength)

        # If it's the exact reference cube, no need to interpolate
        if cube is ref_cube:
            all_data.append(cube.data)
            continue

        c_ny, c_nx = cube.spatial_shape
        
        # If geometries match perfectly and WCS mapping is disabled, fast-path it
        if not use_wcs and c_ny == ny and c_nx == nx:
            all_data.append(cube.data)
            continue

        # Spatial resampling
        resampled_data = np.zeros((cube.n_wavelengths, ny, nx), dtype=np.float64)

        if use_wcs:
            try:
                target_wcs = WCS(cube.header).celestial
                if target_wcs.pixel_n_dim != 2:
                    raise ValueError(f"WCS projection has {target_wcs.pixel_n_dim} dimensions.")
                # Convert reference sky coordinates to target pixel coordinates
                t_x, t_y = target_wcs.world_to_pixel(sky_coords)
            except Exception as e:
                warnings.warn(f"Failed to parse celestial WCS for cube '{cube.filepath}': {e}. "
                              f"Assuming 1:1 pixel mapping.")
                t_x, t_y = np.meshgrid(np.arange(nx), np.arange(ny))
        else:
            t_x, t_y = np.meshgrid(np.arange(nx), np.arange(ny))

        # Perform bilinear interpolation for each spectral channel
        for i in range(cube.n_wavelengths):
            resampled_data[i] = map_coordinates(
                cube.data[i],
                [t_y, t_x],
                order=1,           # Bilinear
                mode='constant',
                cval=np.nan,       # Pad unseen areas with NaN
            )

        all_data.append(resampled_data)

    # Concatenate spectrally
    combined_wl = np.concatenate(all_wavelengths)
    combined_data = np.concatenate(all_data, axis=0)

    # Sort strictly by wavelength
    sort_idx = np.argsort(combined_wl)
    combined_wl = combined_wl[sort_idx]
    combined_data = combined_data[sort_idx]

    # Combine filenames for the new metadata
    source_files = [str(c.filepath) for c in cubes]
    new_filepath = "stitched: " + ", ".join(m.split("/")[-1] for m in source_files)

    # Create new header (using reference, but updating spectral naxis)
    new_header = ref_cube.header.copy() if ref_cube.header else {}
    new_header["NAXIS3"] = len(combined_wl)
    # Spectral CDELT3/CRVAL3 are no longer strictly linear after stitching
    if "CDELT3" in new_header:
        del new_header["CDELT3"]
    if "CRVAL3" in new_header:
        del new_header["CRVAL3"]
    new_header["COMMENT"] = "Stitched from multiple cubes by JWST IFU Pipeline."

    return SpectralCube(
        data=combined_data,
        wavelength=combined_wl,
        header=new_header,
        filepath=new_filepath,
    )
