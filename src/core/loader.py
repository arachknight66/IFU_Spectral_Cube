"""Strict FITS ingestion for JWST-style MIRI Level-3 spectral cubes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from .cube import CubeValidationReport, SpectralCube


class CubeLoadError(ValueError):
    """Raised when a FITS product cannot be scientifically loaded safely."""


def load_fits_cube(filepath: str | Path, *, allow_placeholder_wavelength: bool = False) -> SpectralCube:
    """Load a `.fits`/`.fits.gz` cube without modifying source science values.

    A spectral WCS is mandatory by default. ``allow_placeholder_wavelength`` is
    intentionally opt-in and is only for explicitly non-scientific inspection.
    """
    source = Path(filepath).resolve()
    if source.suffix.lower() not in {".fits", ".gz"} or not source.is_file():
        raise FileNotFoundError(f"expected existing .fits or .fits.gz file, got {source}")
    warnings: list[str] = []
    with fits.open(source, memmap=False) as hdul:
        science = _required_extension(hdul, "SCI")
        raw_flux = np.asarray(science.data)
        if raw_flux.ndim != 3:
            raise CubeLoadError(f"SCI must be 3D; found shape {raw_flux.shape}")
        try:
            science_wcs = WCS(science.header, relax=False)
            flux, wavelength, spectral_numpy_axis, reversed_axis = _orient_with_spectral_wcs(raw_flux, science_wcs)
            wcs_status = "valid spectral WCS"
        except Exception as exc:
            if not allow_placeholder_wavelength:
                raise CubeLoadError(f"SCI spectral WCS is missing, ambiguous, or invalid: {exc}") from exc
            flux = np.array(raw_flux, copy=True)
            spectral_numpy_axis, reversed_axis = 0, False
            wavelength = np.arange(1, flux.shape[0] + 1, dtype=float)
            science_wcs = None
            wcs_status = "placeholder wavelength (explicit opt-in)"
            warnings.append("placeholder channel wavelengths used; cube is not suitable for science analysis")
        arrays, found = _coupled_extensions(hdul, raw_flux.shape, spectral_numpy_axis, reversed_axis)
        err = arrays.pop("ERR", None)
        dq = arrays.pop("DQ", None)
        variances = arrays
        provenance_path, provenance = _load_provenance(source, warnings)
        identifiers = _source_identifiers(science.header, hdul[0].header, provenance)
        validity = np.isfinite(flux)
        dq_count = int(np.count_nonzero(dq)) if dq is not None else 0
        sampling = float(np.median(np.diff(wavelength))) if wavelength.size > 1 else None
        report = CubeValidationReport(
            shape=tuple(int(i) for i in flux.shape), orientation="(wavelength, y, x)" + ("; reversed from descending WCS" if reversed_axis else ""),
            wavelength_range_um=(float(wavelength[0]), float(wavelength[-1])), wavelength_sampling_um=sampling,
            flux_unit=str(science.header.get("BUNIT")) if science.header.get("BUNIT") else None,
            extensions_found=tuple(found), wcs_status=wcs_status, valid_pixel_count=int(validity.sum()),
            invalid_pixel_count=int((~validity).sum()), dq_flagged_pixel_count=dq_count,
            warnings=tuple(warnings), source_path=str(source), provenance_path=str(provenance_path) if provenance_path else None,
            source_identifiers=identifiers,
        )
        return SpectralCube(flux, wavelength, science_header=science.header, primary_header=hdul[0].header,
                            wcs=science_wcs, flux_unit=report.flux_unit, filepath=source, err=err, dq=dq,
                            variances=variances, provenance_path=provenance_path, provenance=provenance,
                            validation_report=report)


def _required_extension(hdul: fits.HDUList, name: str) -> fits.ImageHDU:
    if name not in hdul or hdul[name].data is None:
        available = ", ".join(f"{hdu.name}:{None if hdu.data is None else hdu.data.shape}" for hdu in hdul)
        raise CubeLoadError(f"required {name} extension is absent or empty; available HDUs: {available}")
    return hdul[name]


def _orient_with_spectral_wcs(data: np.ndarray, wcs: WCS) -> tuple[np.ndarray, np.ndarray, int, bool]:
    if wcs.pixel_n_dim != 3 or wcs.world_n_dim < 3:
        raise CubeLoadError(f"expected 3D WCS, found pixel_n_dim={wcs.pixel_n_dim}, world_n_dim={wcs.world_n_dim}")
    physical = list(wcs.world_axis_physical_types)
    spectral_world = [i for i, value in enumerate(physical) if value and ("em." in value or "spect" in value)]
    if len(spectral_world) != 1:
        raise CubeLoadError(f"expected exactly one spectral world axis; found {physical}")
    world_axis = spectral_world[0]
    correlated = np.flatnonzero(wcs.axis_correlation_matrix[world_axis])
    if len(correlated) != 1:
        raise CubeLoadError("spectral WCS axis is coupled ambiguously to pixel axes")
    pixel_axis = int(correlated[0])
    spectral_numpy_axis = data.ndim - 1 - pixel_axis
    values = _wavelength_from_wcs(wcs, world_axis, pixel_axis, data.shape[spectral_numpy_axis])
    oriented = np.moveaxis(np.array(data, copy=True), spectral_numpy_axis, 0)
    coupled_reversed = bool(np.all(np.diff(values) < 0))
    if coupled_reversed:
        values = values[::-1].copy()
        oriented = oriented[::-1].copy()
    if not (np.all(np.isfinite(values)) and np.all(values > 0) and np.all(np.diff(values) > 0)):
        raise CubeLoadError("spectral WCS wavelengths must be finite, positive, and strictly monotonic")
    return oriented, values, spectral_numpy_axis, coupled_reversed


def _wavelength_from_wcs(wcs: WCS, world_axis: int, pixel_axis: int, length: int) -> np.ndarray:
    pixels = [np.full(length, crpix - 1.0) for crpix in wcs.wcs.crpix]
    pixels[pixel_axis] = np.arange(length, dtype=float)
    world = wcs.all_pix2world(*pixels, 0)
    unit_text = wcs.world_axis_units[world_axis]
    if not unit_text:
        raise CubeLoadError("spectral WCS has no declared unit")
    try:
        return (np.asarray(world[world_axis], dtype=float) * u.Unit(unit_text)).to_value(u.um)
    except Exception as exc:
        raise CubeLoadError(f"spectral WCS unit {unit_text!r} cannot convert to microns") from exc


def _coupled_extensions(hdul: fits.HDUList, source_shape: tuple[int, ...], spectral_numpy_axis: int, reversed_axis: bool) -> tuple[dict[str, np.ndarray], list[str]]:
    arrays: dict[str, np.ndarray] = {}
    found = ["SCI"]
    for name in ("ERR", "VAR_POISSON", "VAR_RNOISE", "DQ"):
        if name not in hdul or hdul[name].data is None:
            continue
        raw = np.asarray(hdul[name].data)
        if raw.shape != source_shape:
            raise CubeLoadError(f"{name} shape {raw.shape} does not match SCI source shape {source_shape}")
        value = np.moveaxis(np.array(raw, copy=True), spectral_numpy_axis, 0)
        if reversed_axis:
            value = value[::-1].copy()
        arrays[name] = value
        found.append(name)
    return arrays, found


def _load_provenance(source: Path, warnings: list[str]) -> tuple[Path | None, dict[str, Any]]:
    candidate = source.with_name(source.name + ".provenance.json")
    if not candidate.exists():
        return None, {}
    try:
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict): raise ValueError("not a JSON object")
        return candidate, parsed
    except Exception as exc:
        warnings.append(f"provenance sidecar unreadable: {exc}")
        return candidate, {}


def _source_identifiers(science: fits.Header, primary: fits.Header, provenance: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("OBS_ID", "PROGRAM", "PROGRAMID", "TARGNAME", "FILENAME"):
        value = science.get(key, primary.get(key))
        if value is not None: result[key] = str(value)
    mast = provenance.get("mast_product", {}) if isinstance(provenance, dict) else {}
    for key in ("mast_uri", "observation_id", "product_filename"):
        if mast.get(key) is not None: result[key] = str(mast[key])
    return result
