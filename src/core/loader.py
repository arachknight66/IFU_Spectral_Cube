"""
FITS Cube Loader for JWST MIRI IFU Data.

Handles the complexities of reading JWST pipeline-produced FITS files:
    - Multi-extension FITS (science data in 'SCI' extension or HDU 1)
    - WCS-based wavelength axis construction
    - Graceful fallback when WCS metadata is incomplete
    - Metadata extraction from headers

The JWST MIRI MRS pipeline produces Level-3 (s3d) cubes with:
    - Axis 1 (NAXIS1): Right Ascension
    - Axis 2 (NAXIS2): Declination
    - Axis 3 (NAXIS3): Wavelength
    yielding a data array of shape (n_wavelength, n_y, n_x).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .cube import SpectralCube


def load_fits_cube(filepath: str | Path) -> SpectralCube:
    """Load a FITS spectral cube and construct a SpectralCube object.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file containing the IFU data cube.

    Returns
    -------
    SpectralCube
        Fully initialized spectral cube with data, wavelength axis,
        and metadata.

    Raises
    ------
    FileNotFoundError
        If the FITS file does not exist.
    ValueError
        If no 3D data array can be found in the FITS file.

    Notes
    -----
    The loader attempts the following strategy for finding the data:
        1. Look for a 'SCI' extension (JWST standard).
        2. Fall back to the first extension with 3D data.
        3. Fall back to the primary HDU if it contains 3D data.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    with fits.open(filepath) as hdul:
        data, header = _extract_cube_data(hdul)
        wavelength = _build_wavelength_axis(header, data.shape[0])
        metadata = _extract_metadata(header)

    # Handle NaN/inf values in the data
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return SpectralCube(
        data=data,
        wavelength=wavelength,
        header=metadata,
        filepath=str(filepath),
    )


def _extract_cube_data(
    hdul: fits.HDUList,
) -> tuple[np.ndarray, fits.Header]:
    """Locate and extract 3D data from a FITS HDU list.

    Strategy:
        1. 'SCI' named extension (JWST pipeline convention)
        2. First extension with NAXIS=3
        3. Primary HDU if 3D

    Returns
    -------
    tuple of (data, header)
    """
    # Strategy 1: named SCI extension
    if "SCI" in hdul:
        ext = hdul["SCI"]
        if ext.data is not None:
            data = np.squeeze(ext.data)
            if data.ndim == 3:
                return data.astype(np.float64), ext.header

    # Strategy 2: first 3D extension
    for ext in hdul:
        if ext.data is not None:
            data = np.squeeze(ext.data)
            if data.ndim == 3:
                return data.astype(np.float64), ext.header

    # Strategy 3: primary HDU
    if hdul[0].data is not None:
        data = np.squeeze(hdul[0].data)
        if data.ndim == 3:
            return data.astype(np.float64), hdul[0].header

    shapes = []
    for ext in hdul:
        if ext.data is not None:
            shapes.append(f"'{ext.name}': {ext.data.shape}")
        else:
            shapes.append(f"'{ext.name}': Empty")

    raise ValueError(
        "No 3D data cube found in FITS file. "
        "Expected a data array with NAXIS=3. "
        f"Found HDU shapes: {', '.join(shapes)}"
    )


def _build_wavelength_axis(
    header: fits.Header,
    n_channels: int,
) -> np.ndarray:
    """Construct the wavelength axis from FITS header WCS information.

    Attempts WCS-based construction using CRPIX3, CRVAL3, CDELT3.
    Falls back to a linear index array if WCS keywords are missing.

    Parameters
    ----------
    header : fits.Header
        FITS header containing WCS keywords.
    n_channels : int
        Number of spectral channels (NAXIS3).

    Returns
    -------
    np.ndarray
        1D wavelength array in microns.
    """
    # Try full WCS approach first
    try:
        wcs = WCS(header)
        if wcs.naxis >= 3:
            # Build pixel coordinate array for spectral axis
            pixel_coords = np.zeros((n_channels, wcs.naxis))
            pixel_coords[:, 2] = np.arange(n_channels)  # spectral axis
            world_coords = wcs.pixel_to_world_values(pixel_coords)
            # world_coords is an array of shape (n_channels, naxis)
            # Spectral axis is the 3rd (index 2)
            if isinstance(world_coords, np.ndarray):
                wavelength = world_coords[:, 2]
            else:
                wavelength = np.array([w[2] for w in world_coords])

            # Convert to microns if necessary (JWST uses meters in WCS)
            wavelength = _ensure_microns(wavelength, header)

            if np.all(np.isfinite(wavelength)) and np.all(wavelength > 0):
                return wavelength
    except Exception:
        pass

    # Fallback: manual CRPIX3/CRVAL3/CDELT3
    try:
        crpix3 = header.get("CRPIX3", 1.0)
        crval3 = header.get("CRVAL3")
        cdelt3 = header.get("CDELT3") or header.get("CD3_3")

        if crval3 is not None and cdelt3 is not None:
            pixel_indices = np.arange(n_channels)
            wavelength = crval3 + (pixel_indices - (crpix3 - 1)) * cdelt3
            wavelength = _ensure_microns(wavelength, header)

            if np.all(np.isfinite(wavelength)) and np.all(wavelength > 0):
                return wavelength
    except Exception:
        pass

    # Last resort: channel indices
    warnings.warn(
        "Could not construct wavelength axis from WCS. "
        "Using channel indices as placeholder. "
        "Spectral analysis results will be in channel units.",
        stacklevel=2,
    )
    return np.arange(n_channels, dtype=np.float64)


def _ensure_microns(
    wavelength: np.ndarray,
    header: fits.Header,
) -> np.ndarray:
    """Convert wavelength array to microns if needed.

    JWST WCS stores wavelengths in meters. Detects this by checking
    if values are << 1 (i.e., in meters) and converts to µm.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in unknown units.
    header : fits.Header
        Header for unit information.

    Returns
    -------
    np.ndarray
        Wavelength in microns.
    """
    # Check CUNIT3 first
    cunit3 = header.get("CUNIT3", "").strip().lower()
    if cunit3 in ("m", "meter", "meters"):
        return wavelength * 1e6
    elif cunit3 in ("um", "micron", "microns"):
        return wavelength
    elif cunit3 in ("nm", "nanometer", "nanometers"):
        return wavelength * 1e-3
    elif cunit3 in ("angstrom", "a", "ang"):
        return wavelength * 1e-4

    # Heuristic: if median wavelength < 1e-3, assume meters
    median_wl = np.nanmedian(wavelength)
    if median_wl < 1e-3:
        return wavelength * 1e6
    elif median_wl > 1000:
        # Likely Angstroms
        return wavelength * 1e-4

    return wavelength


def _extract_metadata(header: fits.Header) -> dict[str, Any]:
    """Extract scientifically relevant metadata from the FITS header.

    Returns
    -------
    dict
        Dictionary of key metadata fields.
    """
    keys_of_interest = [
        "TELESCOP", "INSTRUME", "DETECTOR", "FILTER", "CHANNEL",
        "BAND", "SUBARRAY", "GRATNG14",
        "TARGNAME", "TARG_RA", "TARG_DEC",
        "DATE-OBS", "TIME-OBS", "EXPTIME", "EFFINTTM",
        "NAXIS1", "NAXIS2", "NAXIS3",
        "CRPIX3", "CRVAL3", "CDELT3", "CUNIT3",
        "PROGRAM", "TITLE", "PI_NAME",
        "BUNIT",
    ]

    metadata: dict[str, Any] = {}
    for key in keys_of_interest:
        val = header.get(key)
        if val is not None:
            metadata[key] = val

    return metadata
