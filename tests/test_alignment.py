"""
Comprehensive unit tests for Phase 4: uncertainty-aware cube alignment, reprojection, stitching, and mosaicking.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.io.fits import Header

from src.core.cube import SpectralCube
from src.core.stitch import (
    align_and_stitch_cubes,
    stitch_cubes,
    AlignmentConfig,
    AlignmentResult,
    WCSAlignmentError,
    IncompatibleUnitsError,
)


def _make_test_cube(
    data: np.ndarray,
    wavelength: np.ndarray,
    ra: float = 150.0,
    dec: float = 2.0,
    cdelt: float = 0.0001,
    err: np.ndarray | None = None,
    dq: np.ndarray | None = None,
    flux_unit: str | None = "MJy/sr",
    filepath: str = "test_cube.fits",
) -> SpectralCube:
    """Helper to create a SpectralCube with valid celestial WCS."""
    ny, nx = data.shape[1:]
    header = Header()
    header["CRVAL1"] = ra
    header["CRVAL2"] = dec
    header["CRPIX1"] = float(nx // 2 + 1)
    header["CRPIX2"] = float(ny // 2 + 1)
    header["CDELT1"] = -cdelt
    header["CDELT2"] = cdelt
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["BUNIT"] = flux_unit or ""

    wcs_obj = WCS(header)
    wcs_3d = WCS(naxis=3)
    wcs_3d.wcs.crval = [ra, dec, wavelength[0]]
    wcs_3d.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, 1]
    wcs_3d.wcs.cdelt = [-cdelt, cdelt, wavelength[1] - wavelength[0] if len(wavelength) > 1 else 0.001]
    wcs_3d.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]

    return SpectralCube(
        data=data,
        wavelength=wavelength,
        header=header,
        err=err,
        dq=dq,
        wcs=wcs_3d,
        flux_unit=flux_unit,
        filepath=filepath,
    )


def test_single_cube_noop():
    """Requirement 1: A single validated cube acts as a no-op returning an AlignmentResult."""
    data = np.ones((5, 10, 10), dtype=float)
    wl = np.linspace(5.0, 6.0, 5)
    cube = _make_test_cube(data, wl)

    result = align_and_stitch_cubes([cube])
    assert isinstance(result, AlignmentResult)
    assert result.aligned_cube.shape == (5, 10, 10)
    assert len(result.input_cube_ids) == 1
    assert result.validation_report["status"] == "no_op_single_cube"


def test_identical_wcs_grids():
    """Requirement 2: Identical WCS grids combine correctly."""
    data1 = np.full((5, 10, 10), 10.0, dtype=float)
    data2 = np.full((5, 10, 10), 20.0, dtype=float)
    err1 = np.full((5, 10, 10), 2.0, dtype=float)
    err2 = np.full((5, 10, 10), 2.0, dtype=float)
    wl = np.linspace(5.0, 6.0, 5)

    c1 = _make_test_cube(data1, wl, err=err1, filepath="c1.fits")
    c2 = _make_test_cube(data2, wl, err=err2, filepath="c2.fits")

    result = align_and_stitch_cubes([c1, c2], config=AlignmentConfig(reference="ref_0"))
    res_cube = result.aligned_cube

    # Equal errors (2.0 and 2.0) -> inverse variance weights equal -> mean flux = 15.0
    np.testing.assert_allclose(res_cube.data, 15.0, rtol=1e-3)
    # Output error = 1 / sqrt(1/4 + 1/4) = 1 / sqrt(0.5) = sqrt(2) ≈ 1.4142
    expected_err = 2.0 / np.sqrt(2.0)
    np.testing.assert_allclose(res_cube.err, expected_err, rtol=1e-3)


def test_offset_spatial_wcs_grids():
    """Requirement 2 & 3: Offset spatial WCS grids construct union footprint and reproject."""
    data1 = np.ones((3, 20, 20), dtype=float)
    data2 = np.ones((3, 20, 20), dtype=float)
    wl = np.array([5.0, 5.5, 6.0])

    # Offset CRVAL1 slightly
    c1 = _make_test_cube(data1, wl, ra=150.0, dec=2.0, filepath="c1.fits")
    c2 = _make_test_cube(data2, wl, ra=150.001, dec=2.001, filepath="c2.fits")

    result = align_and_stitch_cubes([c1, c2], config=AlignmentConfig(reference="union"))
    assert result.aligned_cube.shape[0] == 3
    # Union spatial shape should be larger than individual 20x20 grids
    assert result.aligned_cube.spatial_shape[0] >= 20
    assert result.aligned_cube.spatial_shape[1] >= 20
    assert result.coverage_map.shape == result.aligned_cube.shape


def test_inverse_variance_weighting_math():
    """Requirement 4: Verify exact inverse-variance weighting formula."""
    data1 = np.full((2, 10, 10), 10.0, dtype=float)
    err1 = np.full((2, 10, 10), 1.0, dtype=float)  # weight w1 = 1.0

    data2 = np.full((2, 10, 10), 20.0, dtype=float)
    err2 = np.full((2, 10, 10), 2.0, dtype=float)  # weight w2 = 0.25

    wl = np.array([5.0, 6.0])
    c1 = _make_test_cube(data1, wl, err=err1, filepath="c1.fits")
    c2 = _make_test_cube(data2, wl, err=err2, filepath="c2.fits")

    result = align_and_stitch_cubes([c1, c2], config=AlignmentConfig(reference="ref_0"))
    
    # Expected flux = (10*1.0 + 20*0.25) / (1.0 + 0.25) = 15.0 / 1.25 = 12.0
    np.testing.assert_allclose(result.aligned_cube.data, 12.0, rtol=1e-3)

    # Expected err = 1 / sqrt(1.25) = 0.894427
    expected_err = 1.0 / np.sqrt(1.25)
    np.testing.assert_allclose(result.aligned_cube.err, expected_err, rtol=1e-3)


def test_dq_and_nan_mask_propagation():
    """Requirement 4: Bitwise OR combination of DQ flags and NaN propagation."""
    data1 = np.ones((2, 10, 10), dtype=float)
    data1[:, 0, 0] = np.nan  # Masked pixel

    dq1 = np.zeros((2, 10, 10), dtype=np.int32)
    dq1[:, :, :] = 1  # Flag 1

    data2 = np.ones((2, 10, 10), dtype=float)
    dq2 = np.zeros((2, 10, 10), dtype=np.int32)
    dq2[:, :, :] = 4  # Flag 4

    wl = np.array([5.0, 6.0])
    c1 = _make_test_cube(data1, wl, dq=dq1, filepath="c1.fits")
    c2 = _make_test_cube(data2, wl, dq=dq2, filepath="c2.fits")

    result = align_and_stitch_cubes([c1, c2], config=AlignmentConfig(reference="ref_0"))
    
    # Combined DQ where both valid should be 1 | 4 = 5
    assert result.aligned_cube.dq is not None
    assert result.aligned_cube.dq[0, 5, 5] == 5

    # Pixel (0,0) had NaN in c1, so only c2 is valid -> flux = 1.0, DQ = 4
    assert result.aligned_cube.dq[0, 0, 0] == 4
    assert np.isfinite(result.aligned_cube.data[0, 0, 0])


def test_incompatible_units_failure():
    """Requirement 4 & 2: Fail clearly on incompatible units or missing WCS."""
    data = np.ones((2, 10, 10), dtype=float)
    wl = np.array([5.0, 6.0])

    c1 = _make_test_cube(data, wl, flux_unit="MJy/sr")
    c2 = _make_test_cube(data, wl, flux_unit="Jy/beam")

    with pytest.raises(IncompatibleUnitsError):
        align_and_stitch_cubes([c1, c2])


def test_missing_wcs_failure():
    """Requirement 2: Fail clearly when valid 2D celestial WCS is unavailable."""
    data = np.ones((2, 10, 10), dtype=float)
    wl = np.array([5.0, 6.0])

    c1 = SpectralCube(data=data, wavelength=wl, header=Header())  # No celestial WCS
    c2 = _make_test_cube(data, wl)

    with pytest.raises(WCSAlignmentError):
        align_and_stitch_cubes([c1, c2])


def test_different_wavelength_grids_and_stitching():
    """Requirement 5: Spectral stitching across non-matching wavelength bands."""
    data1 = np.ones((10, 10, 10), dtype=float)
    wl1 = np.linspace(5.0, 9.0, 10)

    data2 = np.ones((12, 10, 10), dtype=float)
    wl2 = np.linspace(9.5, 15.0, 12)

    c1 = _make_test_cube(data1, wl1, filepath="channel1.fits")
    c2 = _make_test_cube(data2, wl2, filepath="channel2.fits")

    result = align_and_stitch_cubes([c1, c2], config=AlignmentConfig(overlap_policy="resample_common"))
    res_cube = result.aligned_cube

    # Check strictly increasing monotonic wavelength array
    diffs = np.diff(res_cube.wavelength)
    assert np.all(diffs > 0), "Wavelength array is not strictly monotonic!"
    assert res_cube.wavelength_range[0] == pytest.approx(5.0, abs=0.1)
    assert res_cube.wavelength_range[1] == pytest.approx(15.0, abs=0.1)


def test_provenance_and_header_usability():
    """Requirement 6: Result object contains output WCS, coverage map, and provenance."""
    data = np.ones((3, 10, 10), dtype=float)
    wl = np.array([5.0, 5.5, 6.0])
    c1 = _make_test_cube(data, wl, filepath="c1.fits")
    c2 = _make_test_cube(data, wl, filepath="c2.fits")

    result = align_and_stitch_cubes([c1, c2])
    
    assert result.aligned_cube.header["NAXIS1"] == result.aligned_cube.spatial_shape[1]
    assert result.aligned_cube.header["NAXIS2"] == result.aligned_cube.spatial_shape[0]
    assert result.aligned_cube.header["NAXIS3"] == result.aligned_cube.n_wavelengths
    assert "provenance" in result.aligned_cube.provenance or "action" in result.aligned_cube.provenance
    assert "c1.fits" in str(result.aligned_cube.filepath)
