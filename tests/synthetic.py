"""
Synthetic IFU Cube Generator for Pipeline Testing.

Creates a realistic 3D spectral cube with:
    - Known emission lines (Gaussian profiles) at MIRI wavelengths
    - Polynomial continuum emission
    - Gaussian noise with controllable SNR
    - Spatial structure (a brightness gradient + compact source)

This allows end-to-end pipeline validation: the known input lines
can be compared against detections to verify correctness.
"""

from __future__ import annotations

import numpy as np

from src.core.cube import SpectralCube


# Known lines to inject (subset of the MIRI database)
_SYNTHETIC_LINES = [
    {"species": "[Ne II]",  "wavelength": 12.814, "amplitude": 8.0,  "sigma": 0.03},
    {"species": "H₂ S(1)", "wavelength": 17.035, "amplitude": 5.0,  "sigma": 0.035},
    {"species": "[S III]",  "wavelength": 18.713, "amplitude": 6.5,  "sigma": 0.04},
    {"species": "H₂ S(3)", "wavelength": 9.665,  "amplitude": 3.5,  "sigma": 0.025},
    {"species": "[Ar III]", "wavelength": 8.991,  "amplitude": 4.0,  "sigma": 0.025},
    {"species": "[S IV]",   "wavelength": 10.511, "amplitude": 5.5,  "sigma": 0.03},
    {"species": "PAH",      "wavelength": 11.3,   "amplitude": 12.0, "sigma": 0.15},
    {"species": "H₂ S(2)", "wavelength": 12.279, "amplitude": 2.5,  "sigma": 0.025},
]


def generate_synthetic_cube(
    n_wavelength: int = 500,
    ny: int = 30,
    nx: int = 30,
    wl_start: float = 5.0,
    wl_end: float = 28.0,
    noise_level: float = 0.5,
    continuum_level: float = 10.0,
    seed: int = 42,
    lines: list[dict] | None = None,
) -> SpectralCube:
    """Generate a synthetic IFU data cube with known spectral features.

    Parameters
    ----------
    n_wavelength : int
        Number of spectral channels.
    ny, nx : int
        Spatial dimensions.
    wl_start, wl_end : float
        Wavelength range in µm.
    noise_level : float
        RMS of Gaussian noise (per spaxel).
    continuum_level : float
        Base amplitude of the continuum.
    seed : int
        Random seed for reproducibility.
    lines : list of dict, optional
        Override default emission lines. Each dict has keys:
        'species', 'wavelength', 'amplitude', 'sigma'.

    Returns
    -------
    SpectralCube
        Synthetic cube with known emission features.

    Examples
    --------
    >>> cube = generate_synthetic_cube()
    >>> print(cube)
    SpectralCube(shape=(500, 30, 30), λ=[5.000, 28.000] µm, file='synthetic')
    """
    rng = np.random.default_rng(seed)
    if lines is None:
        lines = _SYNTHETIC_LINES

    # Build wavelength axis
    wavelength = np.linspace(wl_start, wl_end, n_wavelength)

    # Build continuum: a gentle polynomial curve mimicking thermal dust
    wl_norm = (wavelength - wl_start) / (wl_end - wl_start)
    continuum_template = continuum_level * (
        1.0
        + 0.5 * wl_norm
        - 0.3 * wl_norm**2
        + 0.1 * wl_norm**3
    )

    # Build emission lines template
    line_template = np.zeros_like(wavelength)
    for line in lines:
        wl_center = line["wavelength"]
        amplitude = line["amplitude"]
        sigma = line["sigma"]
        line_template += amplitude * np.exp(
            -0.5 * ((wavelength - wl_center) / sigma) ** 2
        )

    # Build 3D cube
    data = np.zeros((n_wavelength, ny, nx), dtype=np.float64)

    # Spatial structure: smooth brightness gradient
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    brightness_map = 1.0 + 0.3 * np.exp(
        -((x_grid - nx / 2) ** 2 + (y_grid - ny / 2) ** 2) / (2 * (nx / 4) ** 2)
    )

    # Compact source: extra emission at center
    source_map = 2.0 * np.exp(
        -((x_grid - nx / 2) ** 2 + (y_grid - ny / 2) ** 2) / (2 * 3.0**2)
    )

    for iy in range(ny):
        for ix in range(nx):
            # Continuum scales with brightness
            spectrum = continuum_template * brightness_map[iy, ix]
            # Lines enhanced at source position
            spectrum += line_template * (1.0 + source_map[iy, ix])
            # Add noise
            spectrum += rng.normal(0, noise_level, n_wavelength)
            data[:, iy, ix] = spectrum

    return SpectralCube(
        data=data,
        wavelength=wavelength,
        header={
            "TELESCOP": "JWST",
            "INSTRUME": "MIRI",
            "DETECTOR": "SYNTHETIC",
            "NAXIS1": nx,
            "NAXIS2": ny,
            "NAXIS3": n_wavelength,
            "CRVAL3": wl_start,
            "CDELT3": (wl_end - wl_start) / n_wavelength,
            "CUNIT3": "um",
            "BUNIT": "MJy/sr",
        },
        filepath="synthetic",
    )


def get_injected_lines() -> list[dict]:
    """Return the list of emission lines injected into the synthetic cube.

    Useful for verifying pipeline detections against ground truth.
    """
    return [line.copy() for line in _SYNTHETIC_LINES]
