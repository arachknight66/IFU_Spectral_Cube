"""Scientifically safe in-memory representation of a validated IFU cube."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS


@dataclass(frozen=True)
class CubeValidationReport:
    """Structured, serializable summary of FITS ingestion checks."""

    shape: tuple[int, int, int]
    orientation: str
    wavelength_range_um: tuple[float, float]
    wavelength_sampling_um: float | None
    flux_unit: str | None
    extensions_found: tuple[str, ...]
    wcs_status: str
    valid_pixel_count: int
    invalid_pixel_count: int
    dq_flagged_pixel_count: int
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    source_path: str = ""
    provenance_path: str | None = None
    source_identifiers: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "orientation": self.orientation,
                "wavelength_range_um": list(self.wavelength_range_um),
                "wavelength_sampling_um": self.wavelength_sampling_um, "flux_unit": self.flux_unit,
                "extensions_found": list(self.extensions_found), "wcs_status": self.wcs_status,
                "valid_pixel_count": self.valid_pixel_count, "invalid_pixel_count": self.invalid_pixel_count,
                "dq_flagged_pixel_count": self.dq_flagged_pixel_count, "warnings": list(self.warnings),
                "errors": list(self.errors), "source_path": self.source_path,
                "provenance_path": self.provenance_path, "source_identifiers": dict(self.source_identifiers)}


class SpectralCube:
    """A validated `(wavelength, y, x)` cube without destructive data edits.

    Input arrays are copied and stored read-only.  Array properties return
    writable copies, so downstream code cannot mutate scientific source data.
    Variance extensions retain their native variance units; ``error`` is a
    standard-deviation view from ``ERR`` or, if absent, quadrature-combined
    non-negative variance components.
    """

    def __init__(self, data: np.ndarray, wavelength: np.ndarray, header: Header | Mapping[str, Any] | None = None, *, science_header: Header | None = None,
                 primary_header: Header | None = None, wcs: WCS | None = None, flux_unit: str | None = None,
                 filepath: str | Path = "", err: np.ndarray | None = None, dq: np.ndarray | None = None,
                 variances: Mapping[str, np.ndarray] | None = None, provenance_path: str | Path | None = None,
                 provenance: Mapping[str, Any] | None = None, validation_report: CubeValidationReport | None = None) -> None:
        flux = self._freeze(data, "SCI", dtype=None)
        wave = self._freeze(wavelength, "wavelength", dtype=float)
        if flux.ndim != 3 or wave.ndim != 1 or wave.size != flux.shape[0]:
            raise ValueError("cube requires 3D (wavelength, y, x) data and matching 1D wavelength")
        if not (np.all(np.isfinite(wave)) and np.all(wave > 0) and np.all(np.diff(wave) > 0)):
            raise ValueError("wavelength must be finite, positive, and strictly increasing")
        self._data, self._wavelength = flux, wave
        self._err = self._checked_array(err, "ERR", flux.shape)
        self._dq = self._checked_array(dq, "DQ", flux.shape, preserve_dtype=True)
        self._variances = {name: self._checked_array(value, name, flux.shape) for name, value in (variances or {}).items()}
        if header is not None and science_header is not None:
            raise ValueError("use either header or science_header, not both")
        supplied_header = science_header if science_header is not None else header
        self._science_header = Header(supplied_header or {}).copy()
        self._primary_header = (primary_header or Header()).copy()
        self._wcs = deepcopy(wcs) if wcs is not None else None
        self._flux_unit = flux_unit
        self._filepath = str(filepath)
        self._provenance_path = str(provenance_path) if provenance_path else None
        self._provenance = deepcopy(dict(provenance or {}))
        self._validity = np.isfinite(self._data)
        self._validity.setflags(write=False)
        self.validation_report = validation_report

    @staticmethod
    def _freeze(value: np.ndarray, name: str, dtype: Any = float) -> np.ndarray:
        arr = np.array(value, dtype=dtype, copy=True)
        arr.setflags(write=False)
        return arr

    def _checked_array(self, value: np.ndarray | None, name: str, shape: tuple[int, ...], preserve_dtype: bool = False) -> np.ndarray | None:
        if value is None:
            return None
        arr = self._freeze(value, name, dtype=None if preserve_dtype else float)
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} does not match SCI shape {shape}")
        return arr

    @property
    def data(self) -> np.ndarray: return self._data.copy()
    @property
    def wavelength(self) -> np.ndarray: return self._wavelength.copy()
    @property
    def err(self) -> np.ndarray | None: return None if self._err is None else self._err.copy()
    @property
    def dq(self) -> np.ndarray | None: return None if self._dq is None else self._dq.copy()
    @property
    def validity_mask(self) -> np.ndarray: return self._validity.copy()
    @property
    def header(self) -> Header: return self._science_header.copy()
    @property
    def science_header(self) -> Header: return self._science_header.copy()
    @property
    def primary_header(self) -> Header: return self._primary_header.copy()
    @property
    def wcs(self) -> WCS | None: return deepcopy(self._wcs)
    @property
    def flux_unit(self) -> str | None: return self._flux_unit
    @property
    def filepath(self) -> str: return self._filepath
    @property
    def provenance_path(self) -> str | None: return self._provenance_path
    @property
    def provenance(self) -> dict[str, Any]: return deepcopy(self._provenance)
    @property
    def variances(self) -> dict[str, np.ndarray]: return {name: value.copy() for name, value in self._variances.items()}
    @property
    def has_err(self) -> bool: return self._err is not None or bool(self._variances)
    @property
    def has_dq(self) -> bool: return self._dq is not None
    @property
    def shape(self) -> tuple[int, int, int]: return self._data.shape
    @property
    def n_wavelengths(self) -> int: return self._data.shape[0]
    @property
    def spatial_shape(self) -> tuple[int, int]: return self._data.shape[1:]
    @property
    def wavelength_range(self) -> tuple[float, float]: return float(self._wavelength[0]), float(self._wavelength[-1])

    def standard_deviation(self) -> np.ndarray | None:
        if self._err is not None:
            return self._err.copy()
        if not self._variances:
            return None
        total = np.zeros(self.shape, dtype=float)
        valid = np.ones(self.shape, dtype=bool)
        for variance in self._variances.values():
            valid &= np.isfinite(variance) & (variance >= 0)
            total += np.where(valid, variance, 0.0)
        total[~valid] = np.nan
        return np.sqrt(total)

    def dq_mask(self, bitmask: int | None = None) -> np.ndarray:
        if self._dq is None:
            return np.zeros(self.shape, dtype=bool)
        return self._dq != 0 if bitmask is None else (self._dq & bitmask) != 0

    def valid_science_mask(self, *, include_dq: bool = False, bitmask: int | None = None) -> np.ndarray:
        valid = self._validity.copy()
        if include_dq:
            valid &= ~self.dq_mask(bitmask)
        return valid

    def masked_flux(self, *, include_dq: bool = False, bitmask: int | None = None) -> np.ma.MaskedArray:
        return np.ma.array(self._data.copy(), mask=~self.valid_science_mask(include_dq=include_dq, bitmask=bitmask), copy=False)

    def filled_flux(self, fill_value: float, *, include_dq: bool = False, bitmask: int | None = None) -> np.ndarray:
        """Return a derived filled array; source SCI values are never changed."""
        if not np.isfinite(fill_value):
            raise ValueError("fill_value must be finite and explicitly supplied")
        return self.masked_flux(include_dq=include_dq, bitmask=bitmask).filled(fill_value)

    def get_spectrum(self, x: int, y: int) -> np.ndarray:
        self._check_spaxel(x, y); return self._data[:, y, x].copy()
    def get_spectrum_err(self, x: int, y: int) -> np.ndarray | None:
        self._check_spaxel(x, y); error = self.standard_deviation(); return None if error is None else error[:, y, x]
    def get_spectrum_dq(self, x: int, y: int) -> np.ndarray | None:
        self._check_spaxel(x, y); return None if self._dq is None else self._dq[:, y, x].copy()
    def get_slice(self, index: int) -> np.ndarray:
        if not 0 <= index < self.n_wavelengths: raise IndexError(f"channel index {index} out of range")
        return self._data[index].copy()
    def _check_spaxel(self, x: int, y: int) -> None:
        ny, nx = self.spatial_shape
        if not (0 <= x < nx and 0 <= y < ny): raise IndexError(f"pixel ({x}, {y}) out of bounds for ({nx}, {ny})")
    def integrate_band(self, wl_start: float, wl_end: float, method: str = "sum") -> np.ndarray:
        selected = (self._wavelength >= wl_start) & (self._wavelength <= wl_end)
        if not selected.any(): raise ValueError("no channels in requested wavelength range")
        if method == "sum": return np.nansum(self._data[selected], axis=0)
        if method == "mean": return np.nanmean(self._data[selected], axis=0)
        raise ValueError("method must be 'sum' or 'mean'")
    def get_wavelength_at(self, index: int) -> float: return float(self._wavelength[index])
    def find_nearest_channel(self, target_wavelength: float) -> int: return int(np.argmin(abs(self._wavelength - target_wavelength)))
    def __repr__(self) -> str:
        return f"SpectralCube(shape={self.shape}, wavelength_um={self.wavelength_range}, file='{self.filepath}')"
