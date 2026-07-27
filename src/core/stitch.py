"""
Uncertainty-aware spatial alignment, reprojection, stitching, and mosaicking module.

Combines multiple SpectralCube objects (e.g., from different MIRI channels or mosaic tiles)
onto a common sky-coordinate grid (Astropy WCS) without losing flux units, uncertainty
information, DQ masks, WCS, or provenance.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS
import astropy.units as u

try:
    from reproject import reproject_interp
    HAS_REPROJECT = True
except ImportError:  # pragma: no cover
    reproject_interp = None
    HAS_REPROJECT = False

from src.core.cube import SpectralCube


class AlignmentError(ValueError):
    """Base exception for alignment and reprojection failures."""


class WCSAlignmentError(AlignmentError):
    """Raised when valid celestial WCS is unavailable or incompatible."""


class IncompatibleUnitsError(AlignmentError):
    """Raised when flux units across input cubes are incompatible."""


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for spatial alignment and spectral stitching."""

    reference: str | SpectralCube | WCS = "union"
    pixel_scale_deg: float | None = None
    interpolation: str = "bilinear"
    overlap_policy: str = "combine_overlaps"
    combine_method: str = "inverse_variance"
    wavelength_step_um: float | None = None
    dq_bitmask: int | None = None

    def __post_init__(self) -> None:
        if self.interpolation not in ("bilinear", "nearest"):
            raise ValueError(f"Unsupported interpolation: {self.interpolation}. Must be 'bilinear' or 'nearest'.")
        if self.overlap_policy not in ("combine_overlaps", "resample_common", "retain_separate"):
            raise ValueError(f"Unsupported overlap policy: {self.overlap_policy}.")
        if self.combine_method not in ("inverse_variance", "equal_weight"):
            raise ValueError(f"Unsupported combine method: {self.combine_method}.")


@dataclass
class AlignmentResult:
    """Structured result container for aligned and mosaicked spectral cubes."""

    aligned_cube: SpectralCube
    output_wcs: WCS
    coverage_map: np.ndarray
    propagated_uncertainty: np.ndarray | None
    propagated_dq: np.ndarray | None
    input_cube_ids: list[str]
    config: AlignmentConfig
    validation_report: dict[str, Any]


def _extract_celestial_wcs(cube: SpectralCube) -> WCS:
    """Extract and validate 2D celestial WCS from a SpectralCube."""
    wcs_obj = cube.wcs
    if wcs_obj is None:
        try:
            wcs_obj = WCS(cube.header)
        except Exception as e:
            raise WCSAlignmentError(f"Failed to parse WCS from header for cube '{cube.filepath}': {e}") from e

    try:
        celestial_wcs = wcs_obj.celestial
    except Exception as e:
        raise WCSAlignmentError(f"No celestial sub-WCS found in cube '{cube.filepath}': {e}") from e

    if celestial_wcs is None or not getattr(celestial_wcs, "has_celestial", False):
        raise WCSAlignmentError(
            f"Cube '{cube.filepath}' lacks valid 2D celestial coordinates (RA/Dec)."
        )

    if celestial_wcs.pixel_n_dim != 2 or celestial_wcs.world_n_dim != 2:
        raise WCSAlignmentError(
            f"Celestial WCS for '{cube.filepath}' must be 2D (got pixel_n_dim={celestial_wcs.pixel_n_dim})."
        )

    return celestial_wcs


def _validate_units(cubes: Sequence[SpectralCube]) -> str | None:
    """Verify that all input cubes have compatible flux units."""
    units = [c.flux_unit for c in cubes if c.flux_unit is not None]
    if not units:
        return None

    first_unit_str = units[0]
    try:
        first_u = u.Unit(first_unit_str)
    except Exception:
        for un in units[1:]:
            if un != first_unit_str:
                raise IncompatibleUnitsError(f"Incompatible flux units: '{first_unit_str}' vs '{un}'.")
        return first_unit_str

    for un in units[1:]:
        try:
            current_u = u.Unit(un)
            if not first_u.is_equivalent(current_u):
                raise IncompatibleUnitsError(f"Incompatible flux units: '{first_unit_str}' vs '{un}'.")
        except Exception as e:
            if isinstance(e, IncompatibleUnitsError):
                raise
            raise IncompatibleUnitsError(f"Incompatible flux units: '{first_unit_str}' vs '{un}'.") from e

    return first_unit_str


def _construct_master_wcs(
    cubes: Sequence[SpectralCube],
    celestial_wcs_list: list[WCS],
    config: AlignmentConfig,
) -> tuple[WCS, tuple[int, int]]:
    """Construct output celestial WCS and spatial shape (height, width)."""
    if isinstance(config.reference, SpectralCube):
        ref_wcs = _extract_celestial_wcs(config.reference)
        ny, nx = config.reference.spatial_shape
        return ref_wcs, (ny, nx)
    elif isinstance(config.reference, WCS):
        ref_wcs = config.reference.celestial
        if ref_wcs is None or not getattr(ref_wcs, "has_celestial", False):
            raise WCSAlignmentError("Custom reference WCS does not contain valid 2D celestial coordinates.")
        nx = int(getattr(ref_wcs, "array_shape", (100, 100))[1]) if getattr(ref_wcs, "array_shape", None) else 100
        ny = int(getattr(ref_wcs, "array_shape", (100, 100))[0]) if getattr(ref_wcs, "array_shape", None) else 100
        return ref_wcs, (ny, nx)
    elif config.reference == "ref_0":
        return celestial_wcs_list[0], cubes[0].spatial_shape
    elif config.reference != "union":
        raise ValueError(f"Unknown reference option: {config.reference}")

    # UNION FOOTPRINT CALCULATION
    ra_min, ra_max = 1e9, -1e9
    dec_min, dec_max = 1e9, -1e9
    scales = []

    for cube, wcs_cel in zip(cubes, celestial_wcs_list):
        ny, nx = cube.spatial_shape
        x_corners = np.array([0, nx - 1, nx - 1, 0], dtype=float)
        y_corners = np.array([0, 0, ny - 1, ny - 1], dtype=float)
        world_corners = wcs_cel.pixel_to_world_values(x_corners, y_corners)
        ras, decs = world_corners[0], world_corners[1]
        
        ra_min = min(ra_min, np.min(ras))
        ra_max = max(ra_max, np.max(ras))
        dec_min = min(dec_min, np.min(decs))
        dec_max = max(dec_max, np.max(decs))

        pix_scales = np.abs(wcs_cel.wcs.cdelt) if hasattr(wcs_cel.wcs, "cdelt") and np.any(wcs_cel.wcs.cdelt) else None
        if pix_scales is None or len(pix_scales) < 2 or pix_scales[0] == 0:
            scale = np.sqrt(((ras[1] - ras[0])**2 + (decs[1] - decs[0])**2) / (nx**2 + 1e-10))
            scales.append(scale)
        else:
            scales.append(np.mean(pix_scales[:2]))

    if ra_min >= ra_max or dec_min >= dec_max:
        raise WCSAlignmentError("Invalid bounding box calculated for union WCS footprint.")

    pixel_scale = config.pixel_scale_deg or float(np.median(scales))
    if pixel_scale <= 0:
        pixel_scale = 0.0001

    center_ra = (ra_min + ra_max) / 2.0
    center_dec = (dec_min + dec_max) / 2.0

    cos_dec = np.cos(np.radians(center_dec))
    delta_ra = (ra_max - ra_min) * (cos_dec if cos_dec > 0.1 else 1.0)
    delta_dec = dec_max - dec_min

    width = int(np.ceil(delta_ra / pixel_scale)) + 10
    height = int(np.ceil(delta_dec / pixel_scale)) + 10

    master_wcs = WCS(naxis=2)
    master_wcs.wcs.crval = [center_ra, center_dec]
    master_wcs.wcs.crpix = [width / 2.0 + 0.5, height / 2.0 + 0.5]
    master_wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
    master_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return master_wcs, (height, width)


def align_and_stitch_cubes(
    cubes: list[SpectralCube],
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Spatially align, reproject, and spectrally combine multiple SpectralCube objects.

    Parameters
    ----------
    cubes : list of SpectralCube
        Input cubes to align.
    config : AlignmentConfig, optional
        Alignment and combination parameters.

    Returns
    -------
    AlignmentResult
        Structured output containing aligned SpectralCube, WCS, coverage map,
        uncertainty, DQ, and provenance.
    """
    if not cubes:
        raise ValueError("Cannot align an empty list of cubes.")

    config = config or AlignmentConfig()
    cube_ids = [c.filepath if c.filepath else f"cube_{i}" for i, c in enumerate(cubes)]

    # Validate units across all input cubes
    flux_unit = _validate_units(cubes)

    # Validate celestial WCS for all input cubes
    celestial_wcs_list = [_extract_celestial_wcs(c) for c in cubes]

    # Handle single cube no-op fast path
    if len(cubes) == 1:
        c = cubes[0]
        cov_map = c.validity_mask.astype(float)
        std_err = c.standard_deviation()
        report = {
            "status": "no_op_single_cube",
            "warnings": [],
            "n_input_cubes": 1,
            "spatial_shape": c.spatial_shape,
            "n_wavelengths": c.n_wavelengths,
        }
        return AlignmentResult(
            aligned_cube=c,
            output_wcs=celestial_wcs_list[0],
            coverage_map=cov_map,
            propagated_uncertainty=std_err,
            propagated_dq=c.dq,
            input_cube_ids=cube_ids,
            config=config,
            validation_report=report,
        )

    # Construct Master Celestial WCS Grid
    master_wcs, (out_ny, out_nx) = _construct_master_wcs(cubes, celestial_wcs_list, config)

    # Determine Master Wavelength Grid
    wavelength_matching = all(
        len(c.wavelength) == len(cubes[0].wavelength)
        and np.allclose(c.wavelength, cubes[0].wavelength, rtol=1e-5, atol=1e-7)
        for c in cubes
    )

    if wavelength_matching and config.overlap_policy != "resample_common":
        master_wavelength = cubes[0].wavelength.copy()
    else:
        # Construct unified monotonic wavelength grid
        all_wl = np.concatenate([c.wavelength for c in cubes])
        wl_min, wl_max = np.min(all_wl), np.max(all_wl)

        if config.wavelength_step_um is not None and config.wavelength_step_um > 0:
            step = config.wavelength_step_um
        else:
            diffs = [np.diff(c.wavelength) for c in cubes]
            concat_diffs = np.concatenate(diffs)
            step = float(np.median(concat_diffs[concat_diffs > 0])) if len(concat_diffs) > 0 else 0.001

        master_wavelength = np.arange(wl_min, wl_max + step / 2.0, step)

    if not (np.all(np.isfinite(master_wavelength)) and np.all(master_wavelength > 0) and np.all(np.diff(master_wavelength) > 0)):
        master_wavelength = np.unique(master_wavelength)
        master_wavelength.sort()

    n_wave = len(master_wavelength)

    weighted_flux = np.zeros((n_wave, out_ny, out_nx), dtype=float)
    sum_weights = np.zeros((n_wave, out_ny, out_nx), dtype=float)
    coverage_count = np.zeros((n_wave, out_ny, out_nx), dtype=int)
    has_any_err = any(c.has_err for c in cubes)
    
    dq_dtype = np.uint32 if any(c.dq is not None and np.issubdtype(c.dq.dtype, np.unsignedinteger) for c in cubes) else np.int32
    combined_dq = np.zeros((n_wave, out_ny, out_nx), dtype=dq_dtype)
    has_any_dq = any(c.has_dq for c in cubes)

    order = 0 if config.interpolation == "nearest" else 1
    warnings_list = []

    # Process each cube and channel
    for idx, (c, in_wcs) in enumerate(zip(cubes, celestial_wcs_list)):
        c_std_err = c.standard_deviation()
        c_dq = c.dq

        for k_out, target_wl in enumerate(master_wavelength):
            if wavelength_matching and config.overlap_policy != "resample_common":
                k_in = k_out
                in_slice = c.data[k_in]
                in_err = c_std_err[k_in] if c_std_err is not None else None
                in_dq = c_dq[k_in] if c_dq is not None else None
            else:
                diffs = np.abs(c.wavelength - target_wl)
                k_closest = int(np.argmin(diffs))
                if target_wl < c.wavelength[0] or target_wl > c.wavelength[-1]:
                    continue

                in_slice = c.data[k_closest]
                in_err = c_std_err[k_closest] if c_std_err is not None else None
                in_dq = c_dq[k_closest] if c_dq is not None else None

            if HAS_REPROJECT:
                proj_flux, fp = reproject_interp(
                    (in_slice, in_wcs),
                    master_wcs,
                    shape_out=(out_ny, out_nx),
                    order=order,
                )
            else:
                proj_flux, fp = _fallback_reproject_2d(in_slice, in_wcs, master_wcs, (out_ny, out_nx), order=order)

            valid_fp = (fp > 0.99) & np.isfinite(proj_flux)

            if not np.any(valid_fp):
                continue

            if in_err is not None:
                in_var = in_err**2
                if HAS_REPROJECT:
                    proj_var, _ = reproject_interp(
                        (in_var, in_wcs),
                        master_wcs,
                        shape_out=(out_ny, out_nx),
                        order=order,
                    )
                else:
                    proj_var, _ = _fallback_reproject_2d(in_var, in_wcs, master_wcs, (out_ny, out_nx), order=order)

                proj_err = np.sqrt(np.maximum(0.0, proj_var))
                valid_fp &= np.isfinite(proj_err) & (proj_err > 0)
                weights = np.where(valid_fp, 1.0 / (proj_err**2 + 1e-30), 0.0)
            else:
                weights = np.where(valid_fp, 1.0, 0.0)

            if config.combine_method == "equal_weight":
                weights = np.where(valid_fp, 1.0, 0.0)

            weighted_flux[k_out, valid_fp] += weights[valid_fp] * proj_flux[valid_fp]
            sum_weights[k_out, valid_fp] += weights[valid_fp]
            coverage_count[k_out, valid_fp] += 1

            if in_dq is not None:
                if HAS_REPROJECT:
                    proj_dq, _ = reproject_interp(
                        (in_dq.astype(float), in_wcs),
                        master_wcs,
                        shape_out=(out_ny, out_nx),
                        order=0,
                    )
                else:
                    proj_dq, _ = _fallback_reproject_2d(in_dq.astype(float), in_wcs, master_wcs, (out_ny, out_nx), order=0)

                proj_dq_clean = np.nan_to_num(proj_dq, nan=0).astype(dq_dtype)
                combined_dq[k_out, valid_fp] |= proj_dq_clean[valid_fp]

    output_flux = np.full((n_wave, out_ny, out_nx), np.nan, dtype=float)
    output_err = np.full((n_wave, out_ny, out_nx), np.nan, dtype=float) if has_any_err else None

    valid_output = sum_weights > 0
    output_flux[valid_output] = weighted_flux[valid_output] / sum_weights[valid_output]

    if has_any_err and output_err is not None:
        if config.combine_method == "inverse_variance":
            output_err[valid_output] = 1.0 / np.sqrt(sum_weights[valid_output])
        else:
            output_err[valid_output] = 1.0 / np.sqrt(np.maximum(1, coverage_count[valid_output]))

    final_dq = combined_dq if has_any_dq else None

    new_header = cubes[0].header.copy() if cubes[0].header else Header()
    new_header["NAXIS1"] = out_nx
    new_header["NAXIS2"] = out_ny
    new_header["NAXIS3"] = n_wave

    for k, v in master_wcs.to_header().items():
        new_header[k] = v

    new_header["COMMENT"] = "Aligned and stitched by JWST IFU Pipeline Phase 4."
    if flux_unit:
        new_header["BUNIT"] = flux_unit

    new_filepath = "aligned: " + ", ".join(Path(cid).name for cid in cube_ids)

    provenance = {
        "action": "align_and_stitch_cubes",
        "input_cube_ids": cube_ids,
        "n_input_cubes": len(cubes),
        "combine_method": config.combine_method,
        "overlap_policy": config.overlap_policy,
        "interpolation": config.interpolation,
        "reference": str(config.reference),
        "flux_unit": flux_unit,
    }

    aligned_cube = SpectralCube(
        data=output_flux,
        wavelength=master_wavelength,
        header=new_header,
        err=output_err,
        dq=final_dq,
        wcs=master_wcs,
        flux_unit=flux_unit,
        filepath=new_filepath,
        provenance=provenance,
    )

    report = {
        "status": "aligned_and_stitched",
        "n_input_cubes": len(cubes),
        "output_shape": (n_wave, out_ny, out_nx),
        "wavelength_range_um": (float(master_wavelength[0]), float(master_wavelength[-1])),
        "flux_unit": flux_unit,
        "warnings": warnings_list,
        "has_uncertainty": has_any_err,
        "has_dq": has_any_dq,
    }

    return AlignmentResult(
        aligned_cube=aligned_cube,
        output_wcs=master_wcs,
        coverage_map=coverage_count,
        propagated_uncertainty=output_err,
        propagated_dq=final_dq,
        input_cube_ids=cube_ids,
        config=config,
        validation_report=report,
    )


def _fallback_reproject_2d(
    image: np.ndarray,
    in_wcs: WCS,
    out_wcs: WCS,
    out_shape: tuple[int, int],
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback 2D image reprojection when `reproject` package is missing."""
    from scipy.ndimage import map_coordinates

    out_ny, out_nx = out_shape
    grid_y, grid_x = np.mgrid[0:out_ny, 0:out_nx]
    
    world_coords = out_wcs.pixel_to_world_values(grid_x.ravel(), grid_y.ravel())
    in_x, in_y = in_wcs.world_to_pixel_values(world_coords[0], world_coords[1])
    
    in_y_2d = in_y.reshape(out_ny, out_nx)
    in_x_2d = in_x.reshape(out_ny, out_nx)

    in_ny, in_nx = image.shape
    valid = (in_x_2d >= 0) & (in_x_2d <= in_nx - 1) & (in_y_2d >= 0) & (in_y_2d <= in_ny - 1)

    resampled = map_coordinates(
        image,
        [in_y_2d, in_x_2d],
        order=order,
        mode="constant",
        cval=np.nan,
    )

    footprint = np.where(valid & np.isfinite(resampled), 1.0, 0.0)
    resampled[footprint == 0] = np.nan
    return resampled, footprint


def stitch_cubes(cubes: list[SpectralCube], reference_index: int = 0) -> SpectralCube:
    """Backward-compatible wrapper around `align_and_stitch_cubes`.

    Parameters
    ----------
    cubes : list of SpectralCube
        Cubes to stitch.
    reference_index : int
        Index of the reference cube.

    Returns
    -------
    SpectralCube
        Combined SpectralCube.
    """
    config = AlignmentConfig(reference="ref_0" if reference_index == 0 else cubes[reference_index])
    res = align_and_stitch_cubes(cubes, config=config)
    return res.aligned_cube
