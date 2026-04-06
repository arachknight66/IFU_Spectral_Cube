from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning
import warnings

from .models import SpectralCube


def _find_image_hdu(hdul: fits.HDUList) -> fits.ImageHDU | fits.PrimaryHDU:
    if "SCI" in hdul:
        return hdul["SCI"]
    for hdu in hdul:
        if isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)) and hdu.data is not None:
            if np.asarray(hdu.data).ndim == 3:
                return hdu
    raise ValueError("No 3D science image HDU found in FITS file.")


def _find_optional_hdu(hdul: fits.HDUList, name: str) -> np.ndarray | None:
    if name in hdul and hdul[name].data is not None:
        arr = np.asarray(hdul[name].data, dtype=float)
        if arr.ndim == 3:
            return arr
    return None


def _identify_spectral_world_axis(wcs: WCS) -> int | None:
    """Identify the spectral world axis index from WCS metadata.

    Supports wavelength (WAVE/AWAV), frequency (FREQ), and velocity
    (VRAD/VELO/VOPT) axis types — covering both IR IFU and radio cubes.
    """
    physical = list(wcs.world_axis_physical_types or [])
    ctype = [str(v).upper() for v in wcs.wcs.ctype]

    for idx, ptype in enumerate(physical):
        token = (ptype or "").lower()
        if "em.wl" in token or "spect" in token:
            return idx
    for idx, ct in enumerate(ctype):
        if any(k in ct for k in ("WAVE", "FREQ", "AWAV", "VRAD", "VELO", "VOPT")):
            return idx
    return None


def _to_micron(
    values: np.ndarray,
    unit_hint: str | None,
    is_frequency_axis: bool,
    is_velocity_axis: bool = False,
    restfreq_hz: float | None = None,
) -> np.ndarray:
    """Convert spectral axis values to micron.

    Handles wavelength, frequency, and velocity axes.  For velocity
    axes, requires a rest frequency (RESTFRQ header) to compute the
    observed frequency via the radio convention:

        f_obs = f_rest × (1 − v/c)

    then converts frequency to wavelength.
    """
    if is_velocity_axis:
        # Radio velocity convention: v = c × (1 − f_obs/f_rest)
        # → f_obs = f_rest × (1 − v/c)
        c_ms = 299792458.0  # speed of light in m/s
        vel_ms = values  # assume m/s
        if unit_hint:
            try:
                vel_ms = u.Quantity(values, u.Unit(unit_hint)).to(u.m / u.s).value
            except Exception:
                pass
        if restfreq_hz is None or restfreq_hz <= 0:
            restfreq_hz = 1.420405751e9  # HI 21cm default
        freq_hz = restfreq_hz * (1.0 - vel_ms / c_ms)
        q = u.Quantity(freq_hz, u.Hz)
        return q.to(u.micron, equivalencies=u.spectral()).value

    if unit_hint:
        try:
            q = u.Quantity(values, u.Unit(unit_hint))
            return q.to(u.micron, equivalencies=u.spectral()).value
        except Exception:
            pass
    if is_frequency_axis:
        q = u.Quantity(values, u.Hz)
        return q.to(u.micron, equivalencies=u.spectral()).value
    return np.asarray(values, dtype=float)


def _wavelength_axis_from_wcs(wcs: WCS, data_shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    spectral_world_axis = _identify_spectral_world_axis(wcs)
    if spectral_world_axis is None:
        raise ValueError("Unable to infer spectral world axis from WCS metadata.")

    axis_corr = np.asarray(wcs.axis_correlation_matrix)
    correlated_pix_axes = np.where(axis_corr[spectral_world_axis])[0]
    if correlated_pix_axes.size == 0:
        raise ValueError("WCS has no correlated pixel axis for spectral coordinate.")

    spectral_pix_axis = int(correlated_pix_axes[0])
    spectral_data_axis = len(data_shape) - 1 - spectral_pix_axis
    n_spec = data_shape[spectral_data_axis]

    pix_coords: list[np.ndarray] = []
    for pix_axis in range(wcs.pixel_n_dim):
        data_axis = len(data_shape) - 1 - pix_axis
        axis_size = data_shape[data_axis]
        if pix_axis == spectral_pix_axis:
            pix_coords.append(np.arange(n_spec, dtype=float))
        else:
            pix_coords.append(np.full(n_spec, 0.5 * (axis_size - 1), dtype=float))

    world_vals = wcs.all_pix2world(*pix_coords, 0)
    wavelength_vals = np.asarray(world_vals[spectral_world_axis], dtype=float)
    unit_hint = (wcs.world_axis_units or [None] * wcs.world_n_dim)[spectral_world_axis]
    ctype_str = (wcs.wcs.ctype[spectral_world_axis] or "").upper()
    is_freq = "FREQ" in ctype_str
    is_vel = any(k in ctype_str for k in ("VRAD", "VELO", "VOPT"))
    restfreq = float(wcs.wcs.restfrq) if wcs.wcs.restfrq else None
    wavelength_micron = _to_micron(
        wavelength_vals, unit_hint, is_freq,
        is_velocity_axis=is_vel, restfreq_hz=restfreq,
    )
    return wavelength_micron, spectral_data_axis


def _wavelength_axis_from_header(header: fits.Header, n_spec: int) -> np.ndarray:
    """Reconstruct spectral axis from CRVAL3/CDELT3 header keywords.

    Handles wavelength (um, Angstrom), frequency (Hz), and velocity
    (m/s, km/s) units via CTYPE3/CUNIT3 inspection.
    """
    if header.get("CRVAL3") is None or header.get("CDELT3") is None:
        raise ValueError("Header does not contain CRVAL3/CDELT3 needed for wavelength reconstruction.")
    crval = float(header.get("CRVAL3"))
    cdelt = float(header.get("CDELT3"))
    crpix = float(header.get("CRPIX3", 1.0))
    cunit = header.get("CUNIT3", "um")
    ctype = (header.get("CTYPE3") or "").upper()

    channels = np.arange(n_spec, dtype=float) + 1.0
    values = crval + (channels - crpix) * cdelt

    is_freq = "FREQ" in ctype
    is_vel = any(k in ctype for k in ("VRAD", "VELO", "VOPT"))
    restfreq = float(header.get("RESTFRQ", header.get("RESTFREQ", 0.0)))
    return _to_micron(
        values, cunit, is_frequency_axis=is_freq,
        is_velocity_axis=is_vel, restfreq_hz=restfreq if restfreq > 0 else None,
    )


def _move_spectral_axis_first(data: np.ndarray, spectral_axis: int) -> np.ndarray:
    if spectral_axis == 0:
        return data
    return np.moveaxis(data, spectral_axis, 0)


def load_miri_ifu_cube(path: str | Path) -> SpectralCube:
    """
    Load JWST MIRI IFU cube and normalize to shape (n_lambda, ny, nx).

    This function keeps science metadata and aligns the spectral axis so downstream
    signal-processing steps are deterministic and less error-prone.
    """
    path = Path(path)
    with fits.open(path) as hdul:
        sci_hdu = _find_image_hdu(hdul)
        data = np.asarray(sci_hdu.data, dtype=float)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D cube, got shape {data.shape}.")

        err = _find_optional_hdu(hdul, "ERR")
        dq = _find_optional_hdu(hdul, "DQ")

        wcs = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            try:
                wcs = WCS(sci_hdu.header)
            except Exception:
                wcs = None

        if wcs is not None and wcs.pixel_n_dim == 3 and wcs.world_n_dim == 3:
            try:
                wavelength_micron, spectral_axis = _wavelength_axis_from_wcs(wcs, data.shape)
            except Exception:
                spectral_axis = 0
                wavelength_micron = _wavelength_axis_from_header(sci_hdu.header, data.shape[spectral_axis])
        else:
            spectral_axis = 0
            wavelength_micron = _wavelength_axis_from_header(sci_hdu.header, data.shape[spectral_axis])

        data = _move_spectral_axis_first(data, spectral_axis)
        err = _move_spectral_axis_first(err, spectral_axis) if err is not None else None
        dq = _move_spectral_axis_first(dq, spectral_axis) if dq is not None else None

        if wavelength_micron.shape[0] != data.shape[0]:
            raise ValueError(
                f"Wavelength axis length ({wavelength_micron.shape[0]}) does not match "
                f"spectral size ({data.shape[0]})."
            )

        flux_unit = sci_hdu.header.get("BUNIT")
        metadata = {
            "instrument": sci_hdu.header.get("INSTRUME"),
            "detector": sci_hdu.header.get("DETECTOR"),
            "target": sci_hdu.header.get("TARGNAME"),
            "filename": path.name,
        }

        return SpectralCube(
            data=data,
            wavelength_micron=wavelength_micron,
            header=sci_hdu.header,
            wcs=wcs,
            flux_unit=flux_unit,
            err=err,
            dq=dq,
            metadata=metadata,
        )


def clip_cube_wavelength(cube: SpectralCube, wmin: float | None, wmax: float | None) -> SpectralCube:
    if wmin is None and wmax is None:
        return cube
    mask = np.ones_like(cube.wavelength_micron, dtype=bool)
    if wmin is not None:
        mask &= cube.wavelength_micron >= wmin
    if wmax is not None:
        mask &= cube.wavelength_micron <= wmax
    if not np.any(mask):
        raise ValueError("Wavelength clip removed all channels.")

    return SpectralCube(
        data=cube.data[mask],
        wavelength_micron=cube.wavelength_micron[mask],
        header=cube.header,
        wcs=cube.wcs,
        flux_unit=cube.flux_unit,
        err=cube.err[mask] if cube.err is not None else None,
        dq=cube.dq[mask] if cube.dq is not None else None,
        metadata=dict(cube.metadata),
    )
