"""
SpectralCube — Core data container for IFU spectral cubes.

This class is the central data object passed through the entire pipeline.
It wraps the 3D flux array and wavelength axis together so that downstream
modules never handle raw arrays without their associated spectral
coordinates.

Design decisions:
    - Uses a standard class (not a dataclass) to allow controlled __init__
      validation and lazy-computed properties.
    - The cube is assumed to be in (n_wavelength, n_y, n_x) orientation
      following the JWST pipeline convention.
    - Methods return copies by default to prevent accidental mutation
      of the underlying data.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SpectralCube:
    """A 3D spectral cube with wavelength metadata.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_wavelength, n_y, n_x) containing flux values.
    wavelength : np.ndarray
        1D array of wavelengths in microns, length n_wavelength.
    header : dict
        Metadata from the FITS header.
    filepath : str, optional
        Original file path (for provenance tracking).

    Attributes
    ----------
    data : np.ndarray
        The 3D flux cube.
    wavelength : np.ndarray
        Wavelength axis in microns.
    header : dict
        FITS metadata.
    filepath : str
        Source file path.
    """

    def __init__(
        self,
        data: np.ndarray,
        wavelength: np.ndarray,
        header: dict[str, Any] | None = None,
        filepath: str = "",
        err: np.ndarray | None = None,
        dq: np.ndarray | None = None,
    ) -> None:
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D data array, got shape {data.shape}"
            )
        if wavelength.ndim != 1:
            raise ValueError(
                f"Wavelength must be 1D, got shape {wavelength.shape}"
            )
        if len(wavelength) != data.shape[0]:
            raise ValueError(
                f"Wavelength length ({len(wavelength)}) does not match "
                f"spectral axis ({data.shape[0]})"
            )
        if err is not None:
            if err.shape != data.shape:
                raise ValueError(
                    f"Error array shape {err.shape} does not match data shape {data.shape}"
                )
        if dq is not None:
            if dq.shape != data.shape:
                raise ValueError(
                    f"Data Quality array shape {dq.shape} does not match data shape {data.shape}"
                )

        self.data: np.ndarray = data
        self.wavelength: np.ndarray = wavelength
        self.header: dict[str, Any] = header or {}
        self.filepath: str = filepath
        self.err: np.ndarray | None = err
        self.dq: np.ndarray | None = dq

    @property
    def has_err(self) -> bool:
        """True if standard uncertainty array is present."""
        return self.err is not None

    @property
    def has_dq(self) -> bool:
        """True if Data Quality flag array is present."""
        return self.dq is not None

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the data cube (n_wavelength, n_y, n_x)."""
        return self.data.shape

    @property
    def n_wavelengths(self) -> int:
        """Number of spectral channels."""
        return self.data.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Spatial dimensions (n_y, n_x)."""
        return (self.data.shape[1], self.data.shape[2])

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """Min and max wavelength in microns."""
        return (float(self.wavelength.min()), float(self.wavelength.max()))

    # ------------------------------------------------------------------ #
    #  Core access methods
    # ------------------------------------------------------------------ #

    def get_spectrum(self, x: int, y: int) -> np.ndarray:
        """Extract the spectrum at a single spaxel position.

        Parameters
        ----------
        x : int
            Pixel coordinate along the x-axis (NAXIS1).
        y : int
            Pixel coordinate along the y-axis (NAXIS2).

        Returns
        -------
        np.ndarray
            1D flux array of length n_wavelength.

        Raises
        ------
        IndexError
            If (x, y) is outside the spatial extent of the cube.
        """
        ny, nx = self.spatial_shape
        if not (0 <= x < nx and 0 <= y < ny):
            raise IndexError(
                f"Pixel ({x}, {y}) out of bounds for cube with "
                f"spatial shape (nx={nx}, ny={ny})"
            )
        return self.data[:, y, x].copy()

    def get_spectrum_err(self, x: int, y: int) -> np.ndarray | None:
        """Extract the spectrum error array at a single spaxel position.

        Returns None if no error array is present in the cube.
        """
        ny, nx = self.spatial_shape
        if not (0 <= x < nx and 0 <= y < ny):
            raise IndexError(
                f"Pixel ({x}, {y}) out of bounds for cube with "
                f"spatial shape (nx={nx}, ny={ny})"
            )
        if self.err is None:
            return None
        return self.err[:, y, x].copy()

    def get_spectrum_dq(self, x: int, y: int) -> np.ndarray | None:
        """Extract the Data Quality flag array at a single spaxel position.

        Returns None if no DQ array is present in the cube.
        """
        ny, nx = self.spatial_shape
        if not (0 <= x < nx and 0 <= y < ny):
            raise IndexError(
                f"Pixel ({x}, {y}) out of bounds for cube with "
                f"spatial shape (nx={nx}, ny={ny})"
            )
        if self.dq is None:
            return None
        return self.dq[:, y, x].copy()

    def get_slice(self, index: int) -> np.ndarray:
        """Extract a 2D spatial image at a given wavelength channel.

        Parameters
        ----------
        index : int
            Spectral channel index (0-based).

        Returns
        -------
        np.ndarray
            2D array of shape (n_y, n_x).

        Raises
        ------
        IndexError
            If index is outside the spectral range.
        """
        if not (0 <= index < self.n_wavelengths):
            raise IndexError(
                f"Channel index {index} out of range "
                f"[0, {self.n_wavelengths})"
            )
        return self.data[index, :, :].copy()

    def integrate_band(
        self,
        wl_start: float,
        wl_end: float,
        method: str = "sum",
    ) -> np.ndarray:
        """Integrate flux over a wavelength band.

        Produces a 2D image by summing (or averaging) flux across
        all channels within [wl_start, wl_end].

        Parameters
        ----------
        wl_start : float
            Starting wavelength in microns (inclusive).
        wl_end : float
            Ending wavelength in microns (inclusive).
        method : str
            'sum' for total flux or 'mean' for average flux.

        Returns
        -------
        np.ndarray
            2D integrated image of shape (n_y, n_x).

        Raises
        ------
        ValueError
            If no channels fall within the specified range.
        """
        mask = (self.wavelength >= wl_start) & (self.wavelength <= wl_end)
        if not np.any(mask):
            raise ValueError(
                f"No channels in wavelength range "
                f"[{wl_start:.4f}, {wl_end:.4f}] µm. "
                f"Cube range: [{self.wavelength.min():.4f}, "
                f"{self.wavelength.max():.4f}] µm"
            )

        sub_cube = self.data[mask, :, :]

        if method == "mean":
            return np.nanmean(sub_cube, axis=0)
        elif method == "sum":
            return np.nansum(sub_cube, axis=0)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'sum' or 'mean'.")

    def get_wavelength_at(self, index: int) -> float:
        """Return the wavelength value at a given channel index.

        Parameters
        ----------
        index : int
            Spectral channel index.

        Returns
        -------
        float
            Wavelength in microns.
        """
        return float(self.wavelength[index])

    def find_nearest_channel(self, target_wavelength: float) -> int:
        """Find the channel index nearest to a target wavelength.

        Parameters
        ----------
        target_wavelength : float
            Target wavelength in microns.

        Returns
        -------
        int
            Index of the nearest channel.
        """
        return int(np.argmin(np.abs(self.wavelength - target_wavelength)))

    def __repr__(self) -> str:
        nw, ny, nx = self.shape
        wl_min, wl_max = self.wavelength_range
        return (
            f"SpectralCube(shape=({nw}, {ny}, {nx}), "
            f"λ=[{wl_min:.3f}, {wl_max:.3f}] µm, "
            f"file='{self.filepath}')"
        )
